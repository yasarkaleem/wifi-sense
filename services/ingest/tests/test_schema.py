"""Tests validating frames against docs/csi-frame-schema.md."""

from __future__ import annotations

import jsonschema
import pytest

from ingest.schema import validate_csi_frame


def make_frame(**overrides) -> dict:
    frame = {
        "schema_version": 1,
        "timestamp_us": 1_700_000_000_000_000,
        "source_mac": "24:6F:28:AB:CD:EF",
        "rssi": -52,
        "channel": 6,
        "subcarrier_count": 2,
        "amplitude": [12.3, 11.9],
        "phase": [0.12, -0.45],
        "sequence_number": 0,
    }
    frame.update(overrides)
    return frame


def test_valid_frame_passes():
    validate_csi_frame(make_frame())  # must not raise


def test_rejects_amplitude_length_mismatch():
    with pytest.raises(ValueError):
        validate_csi_frame(make_frame(amplitude=[1.0]))


def test_rejects_phase_length_mismatch():
    with pytest.raises(ValueError):
        validate_csi_frame(make_frame(phase=[1.0]))


def test_rejects_bad_mac():
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(make_frame(source_mac="not-a-mac"))


def test_rejects_missing_field():
    frame = make_frame()
    del frame["sequence_number"]
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(frame)


def test_rejects_wrong_type():
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(make_frame(rssi="-52"))


def test_rejects_unknown_field():
    with pytest.raises(jsonschema.ValidationError):
        validate_csi_frame(make_frame(extra_field="not allowed"))
