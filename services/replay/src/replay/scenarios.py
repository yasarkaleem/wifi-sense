"""Loading and representing CSI replay scenarios from YAML.

Two scenario shapes:

- A plain (static) scenario — the original shape, one fixed baseline plus
  zero or more stationary `PersonMotion`s (unchanged behavior).
- A `type: trajectory` scenario — one or more people *walking* through a
  sequence of the room's zones (referencing the zone-named static scenarios
  already in this file, e.g. A1..B3), dwelling in each for a configured
  duration and cross-fading smoothly into the next rather than teleporting.
  See `effective_scenario()`, which resolves either shape into a plain
  `ScenarioConfig` snapshot for a given instant — the only thing
  `replay.generator.generate_frame()` needs to know about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml

DEFAULT_SCENARIOS_PATH: Path = Path(__file__).resolve().parent / "scenarios.yaml"


@dataclass(frozen=True)
class PersonMotion:
    """A single moving person's contribution to the CSI disturbance.

    The disturbance is centered on `subcarrier_center` and falls off with
    distance from it (width controlled by `subcarrier_spread`), modeling
    that a person's movement perturbs some subcarriers more than others.
    """

    walk_frequency_hz: float
    amplitude_disturbance: float
    subcarrier_center: float
    subcarrier_spread: float
    phase_offset_rad: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters for synthesizing CSI frames for one named scenario."""

    name: str
    description: str
    subcarrier_count: int
    channel: int
    source_mac: str
    rssi_base: float
    rssi_noise_std: float
    amplitude_baseline: float
    amplitude_noise_std: float
    phase_noise_std: float
    people: tuple[PersonMotion, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PathSegment:
    """One "stand still in this zone for this long" leg of a trajectory."""

    zone: str
    dwell_s: float


@dataclass(frozen=True)
class TrajectoryPath:
    """One person's walking path: a sequence of zones with dwell times.

    `loop` restarts the path from its first segment after the last one
    ends; `phase_offset_s` shifts where in the path this person starts at
    elapsed_s=0 (relative to the path's own total dwell time) — used so
    multiple simultaneous paths (the 2-person case) don't all start at the
    same zone in lockstep.
    """

    segments: tuple[PathSegment, ...]
    loop: bool = True
    phase_offset_s: float = 0.0

    @property
    def total_dwell_s(self) -> float:
        return sum(segment.dwell_s for segment in self.segments)


@dataclass(frozen=True)
class TrajectoryScenarioConfig:
    """One or more people walking through a sequence of zones.

    `zones` maps every zone name referenced by any path to the already-
    loaded static `ScenarioConfig` it refers to (resolved once at load
    time — see `load_scenarios()`) so `effective_scenario()` doesn't need
    the full scenario table to interpolate.
    """

    name: str
    description: str
    subcarrier_count: int
    channel: int
    source_mac: str
    transition_s: float
    paths: tuple[TrajectoryPath, ...]
    zones: dict[str, ScenarioConfig]


ScenarioLike = Union[ScenarioConfig, TrajectoryScenarioConfig]


def _build_scenario(name: str, raw: dict) -> ScenarioConfig:
    people = tuple(
        PersonMotion(
            walk_frequency_hz=person["walk_frequency_hz"],
            amplitude_disturbance=person["amplitude_disturbance"],
            subcarrier_center=person["subcarrier_center"],
            subcarrier_spread=person["subcarrier_spread"],
            phase_offset_rad=person.get("phase_offset_rad", 0.0),
        )
        for person in raw.get("people", [])
    )
    return ScenarioConfig(
        name=name,
        description=raw.get("description", ""),
        subcarrier_count=raw["subcarrier_count"],
        channel=raw["channel"],
        source_mac=raw["source_mac"],
        rssi_base=raw["rssi"]["base"],
        rssi_noise_std=raw["rssi"]["noise_std"],
        amplitude_baseline=raw["amplitude"]["baseline"],
        amplitude_noise_std=raw["amplitude"]["noise_std"],
        phase_noise_std=raw["phase"]["noise_std"],
        people=people,
    )


def _build_trajectory_scenario(
    name: str, raw: dict, static_scenarios: dict[str, ScenarioConfig]
) -> TrajectoryScenarioConfig:
    paths = []
    referenced_zones: dict[str, ScenarioConfig] = {}

    for path_raw in raw["paths"]:
        segments = tuple(
            PathSegment(zone=seg["zone"], dwell_s=float(seg["dwell_s"])) for seg in path_raw["segments"]
        )
        if not segments:
            raise ValueError(f"trajectory scenario {name!r}: a path has no segments")

        for segment in segments:
            if segment.zone not in static_scenarios:
                raise ValueError(
                    f"trajectory scenario {name!r} references unknown zone {segment.zone!r}; "
                    f"known scenarios: {sorted(static_scenarios)}"
                )
            referenced_zones[segment.zone] = static_scenarios[segment.zone]

        paths.append(
            TrajectoryPath(
                segments=segments,
                loop=path_raw.get("loop", True),
                phase_offset_s=float(path_raw.get("phase_offset_s", 0.0)),
            )
        )

    if not paths:
        raise ValueError(f"trajectory scenario {name!r}: needs at least one path")

    # A trajectory scenario emits one consistent wire format across every
    # zone it can visit — if the referenced zones disagree, that's a config
    # mistake worth failing loudly on rather than silently switching
    # subcarrier_count/channel/source_mac mid-stream.
    first_zone_name, first_zone = next(iter(referenced_zones.items()))
    for zone_name, zone in referenced_zones.items():
        for field_name in ("subcarrier_count", "channel", "source_mac"):
            if getattr(zone, field_name) != getattr(first_zone, field_name):
                raise ValueError(
                    f"trajectory scenario {name!r}: zone {zone_name!r}.{field_name}="
                    f"{getattr(zone, field_name)!r} does not match zone {first_zone_name!r}."
                    f"{field_name}={getattr(first_zone, field_name)!r} — all referenced zones "
                    "must share the same wire-format fields"
                )

    return TrajectoryScenarioConfig(
        name=name,
        description=raw.get("description", ""),
        subcarrier_count=first_zone.subcarrier_count,
        channel=first_zone.channel,
        source_mac=first_zone.source_mac,
        transition_s=float(raw.get("transition_s", 1.0)),
        paths=tuple(paths),
        zones=referenced_zones,
    )


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS_PATH) -> dict[str, ScenarioLike]:
    """Load every scenario defined in the YAML file at `path`, keyed by name.

    Two passes: every `type: static` (or untyped, the default) scenario is
    built first, since `type: trajectory` scenarios reference them by name.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "scenarios" not in raw:
        raise ValueError(f"{path}: expected a top-level 'scenarios' key")

    raw_scenarios: dict[str, dict] = raw["scenarios"]

    static_scenarios = {
        name: _build_scenario(name, cfg) for name, cfg in raw_scenarios.items() if cfg.get("type", "static") == "static"
    }

    scenarios: dict[str, ScenarioLike] = dict(static_scenarios)
    for name, cfg in raw_scenarios.items():
        if cfg.get("type") == "trajectory":
            scenarios[name] = _build_trajectory_scenario(name, cfg, static_scenarios)

    return scenarios


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_person(a: PersonMotion, b: PersonMotion, t: float) -> PersonMotion:
    return PersonMotion(
        walk_frequency_hz=_lerp(a.walk_frequency_hz, b.walk_frequency_hz, t),
        amplitude_disturbance=_lerp(a.amplitude_disturbance, b.amplitude_disturbance, t),
        subcarrier_center=_lerp(a.subcarrier_center, b.subcarrier_center, t),
        subcarrier_spread=_lerp(a.subcarrier_spread, b.subcarrier_spread, t),
        phase_offset_rad=_lerp(a.phase_offset_rad, b.phase_offset_rad, t),
    )


def effective_scenario(scenario: ScenarioLike, elapsed_s: float) -> ScenarioConfig:
    """Resolve `scenario` into a plain `ScenarioConfig` snapshot at
    `elapsed_s` — for a static `ScenarioConfig` this is a no-op (returned
    unchanged); for a `TrajectoryScenarioConfig` this computes each path's
    cross-faded position and blends them into one synthetic scenario whose
    `people` tuple `generate_frame()` can consume exactly like a static
    scenario's."""
    if isinstance(scenario, ScenarioConfig):
        return scenario

    people = []
    rssi_bases = []
    rssi_noise_stds = []
    amplitude_baselines = []
    amplitude_noise_stds = []
    phase_noise_stds = []

    for path in scenario.paths:
        total = path.total_dwell_s
        local_t = (elapsed_s + path.phase_offset_s) % total if path.loop else min(
            elapsed_s + path.phase_offset_s, total - 1e-9
        )

        n = len(path.segments)
        accumulated = 0.0
        segment_index = n - 1
        t_in_segment = path.segments[-1].dwell_s
        for i, segment in enumerate(path.segments):
            if local_t < accumulated + segment.dwell_s:
                segment_index = i
                t_in_segment = local_t - accumulated
                break
            accumulated += segment.dwell_s

        current_zone_name = path.segments[segment_index].zone
        if segment_index == 0 and not path.loop:
            blend = 1.0
            prev_zone_name = current_zone_name
        else:
            prev_zone_name = path.segments[segment_index - 1].zone
            blend = min(1.0, t_in_segment / scenario.transition_s)

        current_zone = scenario.zones[current_zone_name]
        prev_zone = scenario.zones[prev_zone_name]

        people.append(_lerp_person(prev_zone.people[0], current_zone.people[0], blend))
        rssi_bases.append(_lerp(prev_zone.rssi_base, current_zone.rssi_base, blend))
        rssi_noise_stds.append(_lerp(prev_zone.rssi_noise_std, current_zone.rssi_noise_std, blend))
        amplitude_baselines.append(_lerp(prev_zone.amplitude_baseline, current_zone.amplitude_baseline, blend))
        amplitude_noise_stds.append(_lerp(prev_zone.amplitude_noise_std, current_zone.amplitude_noise_std, blend))
        phase_noise_stds.append(_lerp(prev_zone.phase_noise_std, current_zone.phase_noise_std, blend))

    n_paths = len(scenario.paths)
    return ScenarioConfig(
        name=scenario.name,
        description=scenario.description,
        subcarrier_count=scenario.subcarrier_count,
        channel=scenario.channel,
        source_mac=scenario.source_mac,
        rssi_base=sum(rssi_bases) / n_paths,
        rssi_noise_std=sum(rssi_noise_stds) / n_paths,
        amplitude_baseline=sum(amplitude_baselines) / n_paths,
        amplitude_noise_std=sum(amplitude_noise_stds) / n_paths,
        phase_noise_std=sum(phase_noise_stds) / n_paths,
        people=tuple(people),
    )
