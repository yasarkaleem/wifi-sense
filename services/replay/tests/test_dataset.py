"""Tests for replay.dataset's loader for converted CSI recordings."""

from __future__ import annotations

import numpy as np

from replay.dataset import DatasetRecording, load_recording


def _write_recording(path, n_frames: int = 20, subcarrier_count: int = 90) -> dict:
    rng = np.random.default_rng(0)
    fields = dict(
        session_id="bed_1",
        timestamp_us=(1_700_000_000_000_000 + np.arange(n_frames) * 1000).astype(np.int64),
        amplitude=rng.uniform(0, 50, size=(n_frames, subcarrier_count)),
        phase=rng.uniform(-np.pi, np.pi, size=(n_frames, subcarrier_count)),
        rssi=np.zeros(n_frames, dtype=np.int64),
        channel=np.zeros(n_frames, dtype=np.int64),
        source_mac="00:00:00:00:00:00",
        subcarrier_count=subcarrier_count,
        label=np.full(n_frames, "bed", dtype="<U16"),
    )
    np.savez_compressed(path, **fields)
    return fields


def test_load_recording_roundtrips_all_fields(tmp_path):
    path = tmp_path / "bed_1.npz"
    fields = _write_recording(path)

    recording = load_recording(path)

    assert isinstance(recording, DatasetRecording)
    assert recording.session_id == "bed_1"
    assert recording.source_mac == "00:00:00:00:00:00"
    assert recording.subcarrier_count == 90
    assert np.array_equal(recording.timestamp_us, fields["timestamp_us"])
    assert np.allclose(recording.amplitude, fields["amplitude"])
    assert np.allclose(recording.phase, fields["phase"])
    assert np.array_equal(recording.rssi, fields["rssi"])
    assert np.array_equal(recording.channel, fields["channel"])
    assert np.array_equal(recording.label, fields["label"])


def test_n_frames_property(tmp_path):
    path = tmp_path / "bed_1.npz"
    _write_recording(path, n_frames=37)

    recording = load_recording(path)

    assert recording.n_frames == 37
