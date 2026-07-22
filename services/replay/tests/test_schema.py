"""Tests validating generated CSI frames against docs/csi-frame-schema.md."""

from __future__ import annotations

import random

import jsonschema
import pytest

from replay.generator import generate_frame
from replay.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios
from replay.schema import validate_csi_frame


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(DEFAULT_SCENARIOS_PATH)


@pytest.mark.parametrize("scenario_name", ["empty_room", "one_person_walking", "two_people"])
def test_default_scenarios_produce_schema_valid_frames(scenarios, scenario_name):
    scenario = scenarios[scenario_name]
    rng = random.Random(0)
    for step in range(5):
        frame = generate_frame(
            scenario,
            elapsed_s=step * 0.01,
            sequence_number=step,
            timestamp_us=1_700_000_000_000_000 + step,
            rng=rng,
        )
        validate_csi_frame(frame.to_dict())  # raises on invalid


def test_validate_rejects_amplitude_length_mismatch():
    frame = {
        "schema_version": 1,
        "timestamp_us": 0,
        "source_mac": "AA:BB:CC:DD:EE:FF",
        "rssi": -50,
        "channel": 6,
        "subcarrier_count": 64,
        "amplitude": [0.0] * 63,  # wrong length
        "phase": [0.0] * 64,
        "sequence_number": 0,
    }
    with pytest.raises(ValueError):
        validate_csi_frame(frame)


def test_validate_rejects_bad_mac():
    frame = {
        "schema_version": 1,
        "timestamp_us": 0,
        "source_mac": "not-a-mac",
        "rssi": -50,
        "channel": 6,
        "subcarrier_count": 1,
        "amplitude": [0.0],
        "phase": [0.0],
        "sequence_number": 0,
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(frame)


def test_validate_rejects_missing_field():
    frame = {
        "schema_version": 1,
        "timestamp_us": 0,
        "source_mac": "AA:BB:CC:DD:EE:FF",
        "rssi": -50,
        "channel": 6,
        "subcarrier_count": 1,
        "amplitude": [0.0],
        "phase": [0.0],
        # missing sequence_number
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(frame)


def test_validate_rejects_wrong_type():
    frame = {
        "schema_version": 1,
        "timestamp_us": 0,
        "source_mac": "AA:BB:CC:DD:EE:FF",
        "rssi": "-50",  # should be an integer, not a string
        "channel": 6,
        "subcarrier_count": 1,
        "amplitude": [0.0],
        "phase": [0.0],
        "sequence_number": 0,
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(frame)
