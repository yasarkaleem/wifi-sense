"""End-to-end smoke tests for UDP streaming (synthetic and dataset modes)."""

from __future__ import annotations

import json
import socket
import time

import numpy as np

from replay.dataset import DatasetRecording
from replay.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios
from replay.stream import stream_dataset, stream_scenario


def test_stream_scenario_sends_valid_json_frames_over_udp():
    scenario = load_scenarios(DEFAULT_SCENARIOS_PATH)["one_person_walking"]

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    stream_scenario(
        scenario=scenario,
        rate_hz=50,
        host=host,
        port=port,
        seed=1,
        duration_s=0.1,
    )

    payload, _ = receiver.recvfrom(65536)
    receiver.close()

    frame = json.loads(payload.decode("utf-8"))
    assert frame["subcarrier_count"] == 64
    assert len(frame["amplitude"]) == 64
    assert len(frame["phase"]) == 64
    assert frame["sequence_number"] == 0


def _make_recording(n_frames: int = 5, subcarrier_count: int = 90) -> DatasetRecording:
    rng = np.random.default_rng(0)
    return DatasetRecording(
        session_id="bed_1",
        timestamp_us=(1_700_000_000_000_000 + np.arange(n_frames) * 10_000).astype(np.int64),
        amplitude=rng.uniform(0, 50, size=(n_frames, subcarrier_count)),
        phase=rng.uniform(-np.pi, np.pi, size=(n_frames, subcarrier_count)),
        rssi=np.zeros(n_frames, dtype=np.int64),
        channel=np.full(n_frames, 6, dtype=np.int64),
        source_mac="00:00:00:00:00:00",
        subcarrier_count=subcarrier_count,
        label=np.full(n_frames, "bed", dtype="<U16"),
    )


def _receive_all(receiver: socket.socket, expected: int) -> list[dict]:
    frames = []
    for _ in range(expected):
        payload, _ = receiver.recvfrom(65536)
        frames.append(json.loads(payload.decode("utf-8")))
    return frames


def test_stream_dataset_sends_every_frame_once_by_default():
    recording = _make_recording(n_frames=5)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    # Tiny timestamp deltas (10ms) so the (unpaced) original-timing default
    # doesn't make the test slow.
    stream_dataset(recording=recording, host=host, port=port)

    frames = _receive_all(receiver, 5)
    receiver.close()

    assert [f["sequence_number"] for f in frames] == [0, 1, 2, 3, 4]
    assert frames[0]["subcarrier_count"] == 90
    assert len(frames[0]["amplitude"]) == 90
    assert len(frames[0]["phase"]) == 90
    assert np.allclose(frames[2]["amplitude"], recording.amplitude[2], atol=1e-6)
    assert np.allclose(frames[2]["phase"], recording.phase[2], atol=1e-6)


def test_stream_dataset_omits_label_field():
    recording = _make_recording(n_frames=1)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    stream_dataset(recording=recording, host=host, port=port)

    frame = _receive_all(receiver, 1)[0]
    receiver.close()

    assert "label" not in frame
    assert set(frame.keys()) == {
        "schema_version",
        "timestamp_us",
        "source_mac",
        "rssi",
        "channel",
        "subcarrier_count",
        "amplitude",
        "phase",
        "sequence_number",
    }


def test_stream_dataset_timestamp_is_real_send_time_not_historical():
    """The recording's own timestamps are ancient (2023-ish); the emitted
    frame's timestamp_us should reflect actual send time instead."""
    recording = _make_recording(n_frames=1)
    before_us = time.time_ns() // 1_000

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    stream_dataset(recording=recording, host=host, port=port)

    frame = _receive_all(receiver, 1)[0]
    receiver.close()
    after_us = time.time_ns() // 1_000

    assert before_us <= frame["timestamp_us"] <= after_us
    assert frame["timestamp_us"] != int(recording.timestamp_us[0])


def test_stream_dataset_rate_override_forces_fixed_pacing():
    recording = _make_recording(n_frames=5)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    start = time.monotonic()
    stream_dataset(recording=recording, host=host, port=port, rate_hz=50)
    elapsed = time.monotonic() - start

    frames = _receive_all(receiver, 5)
    receiver.close()

    assert len(frames) == 5
    # 5 frames at 50Hz -> 4 inter-frame gaps of 20ms = ~80ms minimum.
    assert elapsed >= 0.07


def test_stream_dataset_loop_sends_more_than_one_pass():
    recording = _make_recording(n_frames=3)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()

    stream_dataset(recording=recording, host=host, port=port, rate_hz=200, loop=True, duration_s=0.2)

    frames = _receive_all(receiver, 7)  # well beyond one 3-frame pass
    receiver.close()

    # sequence_number keeps incrementing across loop restarts, not resetting
    assert [f["sequence_number"] for f in frames] == list(range(7))
