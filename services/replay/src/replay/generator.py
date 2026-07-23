"""Synthetic CSI frame generation.

Amplitude for each subcarrier is a baseline value plus sensor noise; each
person in the scenario adds a sinusoidal disturbance (modeling periodic
gait-induced multipath fading) localized around their assigned subcarrier
range via a Gaussian weighting. Phase gets a linear baseline ramp, a small
disturbance correlated with the amplitude disturbance, and its own noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from replay.scenarios import ScenarioConfig, ScenarioLike, effective_scenario

SCHEMA_VERSION = 1
SEQUENCE_NUMBER_WRAP = 2**32

# Phase disturbance is modeled as a fraction of the amplitude disturbance:
# movement perturbs phase too, but less dramatically than amplitude.
_PHASE_DISTURBANCE_RATIO = 0.1


@dataclass(frozen=True)
class CSIFrame:
    """A single CSI frame, matching docs/csi-frame-schema.md exactly."""

    schema_version: int
    timestamp_us: int
    source_mac: str
    rssi: int
    channel: int
    subcarrier_count: int
    amplitude: list[float]
    phase: list[float]
    sequence_number: int

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "timestamp_us": self.timestamp_us,
            "source_mac": self.source_mac,
            "rssi": self.rssi,
            "channel": self.channel,
            "subcarrier_count": self.subcarrier_count,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "sequence_number": self.sequence_number,
        }


def _disturbance_weight(subcarrier_index: int, center: float, spread: float) -> float:
    """Gaussian weighting localizing a person's effect around `center`."""
    return math.exp(-0.5 * ((subcarrier_index - center) / spread) ** 2)


def _wrap_phase(value: float) -> float:
    """Wrap a phase value into (-pi, pi]."""
    return (value + math.pi) % (2 * math.pi) - math.pi


def generate_frame(
    scenario: ScenarioConfig,
    *,
    elapsed_s: float,
    sequence_number: int,
    timestamp_us: int,
    rng: random.Random,
) -> CSIFrame:
    """Synthesize one CSI frame for `scenario` at time `elapsed_s`.

    `rng` is injected so generation is deterministic for a given seed.
    """
    n = scenario.subcarrier_count
    amplitude = [0.0] * n
    phase = [0.0] * n

    for i in range(n):
        amplitude_disturbance = 0.0
        phase_disturbance = 0.0
        for person in scenario.people:
            weight = _disturbance_weight(i, person.subcarrier_center, person.subcarrier_spread)
            wave = math.sin(
                2 * math.pi * person.walk_frequency_hz * elapsed_s + person.phase_offset_rad
            )
            amplitude_disturbance += weight * person.amplitude_disturbance * wave
            phase_disturbance += (
                weight * person.amplitude_disturbance * _PHASE_DISTURBANCE_RATIO * wave
            )

        baseline_phase = -math.pi + i * (2 * math.pi / n)

        amplitude[i] = max(
            0.0,
            scenario.amplitude_baseline
            + amplitude_disturbance
            + rng.gauss(0.0, scenario.amplitude_noise_std),
        )
        phase[i] = _wrap_phase(
            baseline_phase + phase_disturbance + rng.gauss(0.0, scenario.phase_noise_std)
        )

    rssi = round(scenario.rssi_base + rng.gauss(0.0, scenario.rssi_noise_std))

    return CSIFrame(
        schema_version=SCHEMA_VERSION,
        timestamp_us=timestamp_us,
        source_mac=scenario.source_mac,
        rssi=rssi,
        channel=scenario.channel,
        subcarrier_count=n,
        amplitude=amplitude,
        phase=phase,
        sequence_number=sequence_number % SEQUENCE_NUMBER_WRAP,
    )


def synthesize_recording(
    scenario: ScenarioLike, *, rate_hz: float, duration_s: float, seed: int | None = None
) -> np.ndarray:
    """Offline synthesis of a full recording's amplitude matrix — no socket,
    no real-time sleep, just `round(duration_s * rate_hz)` frames generated
    back to back with `elapsed_s = i / rate_hz`.

    Used by scripts/generate_zone_dataset.py (services/pipeline) to produce
    training data far faster than real-time subprocess capture would allow
    — a `TrajectoryScenarioConfig`'s cross-fades are resolved via
    `effective_scenario()` exactly as `replay.stream.stream_scenario()`
    resolves them per-tick when actually streaming live.

    Returns shape (n_frames, scenario.subcarrier_count).
    """
    rng = random.Random(seed)
    n_frames = max(1, round(duration_s * rate_hz))
    base_timestamp_us = 0

    rows: list[list[float]] = []
    for i in range(n_frames):
        elapsed_s = i / rate_hz
        resolved = effective_scenario(scenario, elapsed_s)
        frame = generate_frame(
            resolved,
            elapsed_s=elapsed_s,
            sequence_number=i,
            timestamp_us=base_timestamp_us + round(i * 1_000_000 / rate_hz),
            rng=rng,
        )
        rows.append(frame.amplitude)

    return np.array(rows, dtype=np.float64)
