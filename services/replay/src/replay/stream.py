"""Streaming loops that send CSI frames over UDP — synthetic (stream_scenario)
or from a converted real dataset recording (stream_dataset)."""

from __future__ import annotations

import json
import random
import socket
import time

from replay.dataset import DatasetRecording
from replay.generator import SCHEMA_VERSION, SEQUENCE_NUMBER_WRAP, CSIFrame, generate_frame
from replay.scenarios import ScenarioLike, effective_scenario


class UDPFrameSender:
    """Thin wrapper around a UDP socket for sending JSON CSI frames."""

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame_dict: dict) -> None:
        self._sock.sendto(json.dumps(frame_dict).encode("utf-8"), self._addr)

    def close(self) -> None:
        self._sock.close()


def stream_scenario(
    *,
    scenario: ScenarioLike,
    rate_hz: float,
    host: str,
    port: int,
    seed: int | None = None,
    duration_s: float | None = None,
) -> None:
    """Stream synthetic CSI frames for `scenario` over UDP to (host, port).

    `scenario` may be a static `ScenarioConfig` (unchanged behavior) or a
    `TrajectoryScenarioConfig` — resolved to a plain scenario snapshot each
    tick via `effective_scenario()`, so a walking person's cross-fade
    between zones is recomputed every frame from the real wall-clock
    elapsed time.

    Runs until interrupted (Ctrl+C) unless `duration_s` is given, in which
    case it stops after roughly that many seconds — used by tests.
    """
    rng = random.Random(seed)
    sender = UDPFrameSender(host, port)
    interval_s = 1.0 / rate_hz

    sequence_number = 0
    start = time.monotonic()
    next_send_at = start

    try:
        while duration_s is None or (time.monotonic() - start) < duration_s:
            elapsed_s = time.monotonic() - start
            frame = generate_frame(
                effective_scenario(scenario, elapsed_s),
                elapsed_s=elapsed_s,
                sequence_number=sequence_number,
                timestamp_us=time.time_ns() // 1_000,
                rng=rng,
            )
            sender.send(frame.to_dict())

            sequence_number += 1
            next_send_at += interval_s
            sleep_s = next_send_at - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


def stream_dataset(
    *,
    recording: DatasetRecording,
    host: str,
    port: int,
    rate_hz: float | None = None,
    duration_s: float | None = None,
    loop: bool = False,
) -> None:
    """Stream a converted dataset recording (see replay.dataset.load_recording)
    over UDP to (host, port).

    Pacing: if `rate_hz` is given, frames are sent at that fixed rate (like
    stream_scenario). Otherwise, frames are paced using the recording's own
    inter-frame timestamp deltas, reproducing the original capture's real
    timing (including any jitter or gaps between activities).

    Each emitted frame's `timestamp_us` is always the actual send time
    (matching stream_scenario and "capture time on the sender" in
    docs/csi-frame-schema.md) — not the recording's original historical
    timestamp, which pacing uses but never puts on the wire. The `label`
    field in `recording` (a training-only convenience, not part of the
    schema) is never sent either.

    Runs until interrupted (Ctrl+C), `duration_s` elapses, or the
    recording is exhausted once (unless `loop` is set, which restarts it).
    """
    sender = UDPFrameSender(host, port)
    start = time.monotonic()
    sequence_number = 0

    try:
        while True:
            for i in range(recording.n_frames):
                if duration_s is not None and (time.monotonic() - start) >= duration_s:
                    return

                frame = CSIFrame(
                    schema_version=SCHEMA_VERSION,
                    timestamp_us=time.time_ns() // 1_000,
                    source_mac=recording.source_mac,
                    rssi=int(recording.rssi[i]),
                    channel=int(recording.channel[i]),
                    subcarrier_count=recording.subcarrier_count,
                    amplitude=recording.amplitude[i].tolist(),
                    phase=recording.phase[i].tolist(),
                    sequence_number=sequence_number % SEQUENCE_NUMBER_WRAP,
                )
                sender.send(frame.to_dict())
                sequence_number += 1

                if i + 1 < recording.n_frames:
                    if rate_hz is not None:
                        sleep_s = 1.0 / rate_hz
                    else:
                        delta_us = int(recording.timestamp_us[i + 1]) - int(recording.timestamp_us[i])
                        sleep_s = max(0.0, delta_us / 1_000_000)
                    time.sleep(sleep_s)

            if not loop:
                return
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
