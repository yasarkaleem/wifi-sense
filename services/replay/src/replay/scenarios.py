"""Loading and representing CSI replay scenarios from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


def load_scenarios(path: str | Path = DEFAULT_SCENARIOS_PATH) -> dict[str, ScenarioConfig]:
    """Load every scenario defined in the YAML file at `path`, keyed by name."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "scenarios" not in raw:
        raise ValueError(f"{path}: expected a top-level 'scenarios' key")

    return {name: _build_scenario(name, cfg) for name, cfg in raw["scenarios"].items()}
