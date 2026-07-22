"""Tests for synthetic CSI frame generation."""

from __future__ import annotations

import math
import random

import pytest

from replay.generator import SCHEMA_VERSION, SEQUENCE_NUMBER_WRAP, CSIFrame, generate_frame
from replay.scenarios import PersonMotion, ScenarioConfig


def make_scenario(**overrides) -> ScenarioConfig:
    defaults = dict(
        name="test",
        description="",
        subcarrier_count=64,
        channel=6,
        source_mac="24:6F:28:AB:CD:EF",
        rssi_base=-50.0,
        rssi_noise_std=0.0,
        amplitude_baseline=20.0,
        amplitude_noise_std=0.0,
        phase_noise_std=0.0,
        people=(),
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def test_frame_shape_matches_subcarrier_count():
    scenario = make_scenario(subcarrier_count=64)
    frame = generate_frame(
        scenario, elapsed_s=0.0, sequence_number=0, timestamp_us=0, rng=random.Random(1)
    )
    assert isinstance(frame, CSIFrame)
    assert frame.subcarrier_count == 64
    assert len(frame.amplitude) == 64
    assert len(frame.phase) == 64


def test_frame_fields_pass_through():
    scenario = make_scenario(channel=11, source_mac="AA:BB:CC:DD:EE:FF")
    frame = generate_frame(
        scenario, elapsed_s=1.23, sequence_number=42, timestamp_us=999, rng=random.Random(1)
    )
    assert frame.schema_version == SCHEMA_VERSION
    assert frame.timestamp_us == 999
    assert frame.sequence_number == 42
    assert frame.channel == 11
    assert frame.source_mac == "AA:BB:CC:DD:EE:FF"


def test_empty_room_amplitude_equals_baseline():
    """With zero people and zero noise, amplitude should equal the exact baseline."""
    scenario = make_scenario(people=())
    frame = generate_frame(
        scenario, elapsed_s=5.0, sequence_number=0, timestamp_us=0, rng=random.Random(1)
    )
    assert all(a == pytest.approx(scenario.amplitude_baseline) for a in frame.amplitude)


def test_person_present_disturbs_amplitude_near_center_subcarrier():
    """A person's disturbance should peak near their subcarrier center and
    vanish far from it."""
    person = PersonMotion(
        walk_frequency_hz=1.0,
        amplitude_disturbance=5.0,
        subcarrier_center=32,
        subcarrier_spread=5,
        phase_offset_rad=math.pi / 2,  # sin(pi/2) == 1 at t=0 -> peak disturbance
    )
    scenario = make_scenario(people=(person,))
    frame = generate_frame(
        scenario, elapsed_s=0.0, sequence_number=0, timestamp_us=0, rng=random.Random(1)
    )

    assert frame.amplitude[32] > scenario.amplitude_baseline + 4.0
    assert frame.amplitude[0] == pytest.approx(scenario.amplitude_baseline, abs=1e-6)


def test_amplitude_is_never_negative():
    person = PersonMotion(
        walk_frequency_hz=1.0,
        amplitude_disturbance=1000.0,  # deliberately huge to try to force negative
        subcarrier_center=0,
        subcarrier_spread=64,
        phase_offset_rad=-math.pi / 2,  # sin(-pi/2) == -1, biggest negative swing
    )
    scenario = make_scenario(amplitude_baseline=1.0, people=(person,))
    frame = generate_frame(
        scenario, elapsed_s=0.0, sequence_number=0, timestamp_us=0, rng=random.Random(1)
    )
    assert all(a >= 0.0 for a in frame.amplitude)


def test_two_people_disturbances_are_localized_independently():
    p1 = PersonMotion(1.0, 5.0, subcarrier_center=10, subcarrier_spread=3, phase_offset_rad=math.pi / 2)
    p2 = PersonMotion(1.0, 5.0, subcarrier_center=50, subcarrier_spread=3, phase_offset_rad=math.pi / 2)
    scenario = make_scenario(people=(p1, p2))
    frame = generate_frame(
        scenario, elapsed_s=0.0, sequence_number=0, timestamp_us=0, rng=random.Random(1)
    )

    assert frame.amplitude[10] > scenario.amplitude_baseline + 4.0
    assert frame.amplitude[50] > scenario.amplitude_baseline + 4.0
    assert frame.amplitude[30] == pytest.approx(scenario.amplitude_baseline, abs=0.5)


def test_sequence_number_wraps():
    scenario = make_scenario()
    frame = generate_frame(
        scenario,
        elapsed_s=0.0,
        sequence_number=SEQUENCE_NUMBER_WRAP + 5,
        timestamp_us=0,
        rng=random.Random(1),
    )
    assert frame.sequence_number == 5


def test_generation_is_deterministic_for_a_given_seed():
    scenario = make_scenario(amplitude_noise_std=1.0, phase_noise_std=0.1, rssi_noise_std=1.0)
    frame_a = generate_frame(
        scenario, elapsed_s=2.0, sequence_number=0, timestamp_us=0, rng=random.Random(7)
    )
    frame_b = generate_frame(
        scenario, elapsed_s=2.0, sequence_number=0, timestamp_us=0, rng=random.Random(7)
    )
    assert frame_a.amplitude == frame_b.amplitude
    assert frame_a.phase == frame_b.phase
    assert frame_a.rssi == frame_b.rssi
