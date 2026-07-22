"""Cross-service integration test: spawns real replay + ingest + pipeline
subprocesses and asserts presence flips to True within 3s of switching the
replay scenario from empty_room to one_person_walking.

Unlike the rest of this test suite, this spawns services/replay and
services/ingest as subprocesses under the *same* interpreter pytest is
running under (see the "PyYAML"/"jsonschema" comment in pyproject.toml's
dev extra) — it does not import them as Python packages, honoring the
service-isolation rule in ../../../CLAUDE.md (network only, no in-process
imports across service boundaries).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import zmq

WIFI_SENSE_ROOT = Path(__file__).resolve().parents[3]
INGEST_SRC = WIFI_SENSE_ROOT / "services" / "ingest" / "src"
REPLAY_SRC = WIFI_SENSE_ROOT / "services" / "replay" / "src"
PIPELINE_SRC = WIFI_SENSE_ROOT / "services" / "pipeline" / "src"

# Tuned window/stride/calibration so calibration finishes in a few seconds
# and detection latency after a real disturbance is well under a second
# (see the exploratory simulation this was based on) — see
# services/pipeline/README.md for the reasoning against production defaults.
WINDOW_S = 1.0
STRIDE_S = 0.25
CALIBRATION_S = 3.0
N_SIGMAS = 6.0


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start(cmd: list[str], *, pythonpath: Path, env: dict) -> subprocess.Popen:
    proc_env = env.copy()
    proc_env["PYTHONPATH"] = str(pythonpath)
    return subprocess.Popen(cmd, env=proc_env, stderr=subprocess.PIPE, text=True)


def _stop(*procs: subprocess.Popen) -> None:
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.integration
def test_presence_flips_true_within_3s_of_scenario_switch():
    if not INGEST_SRC.exists() or not REPLAY_SRC.exists():
        pytest.skip("sibling services (ingest/replay) not found next to pipeline in this checkout")

    udp_port = _free_port()
    csi_pub_port = _free_port()
    presence_pub_port = _free_port()
    env = os.environ.copy()

    ingest_proc = _start(
        [
            sys.executable,
            "-m",
            "ingest",
            "--udp-host",
            "127.0.0.1",
            "--udp-port",
            str(udp_port),
            "--pub-host",
            "127.0.0.1",
            "--pub-port",
            str(csi_pub_port),
            "--plot",
            "none",
        ],
        pythonpath=INGEST_SRC,
        env=env,
    )
    pipeline_proc = _start(
        [
            sys.executable,
            "-m",
            "pipeline",
            "--sub-host",
            "127.0.0.1",
            "--sub-port",
            str(csi_pub_port),
            "--pub-host",
            "127.0.0.1",
            "--pub-port",
            str(presence_pub_port),
            "--window-s",
            str(WINDOW_S),
            "--stride-s",
            str(STRIDE_S),
            "--calibration-s",
            str(CALIBRATION_S),
            "--n-sigmas",
            str(N_SIGMAS),
        ],
        pythonpath=PIPELINE_SRC,
        env=env,
    )
    time.sleep(1.5)  # let ingest + pipeline bind their sockets before replay starts sending

    events: list[tuple[float, dict]] = []
    stop_collecting = threading.Event()

    def collect_events() -> None:
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://127.0.0.1:{presence_pub_port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"presence")
        sub.setsockopt(zmq.RCVTIMEO, 200)
        while not stop_collecting.is_set():
            try:
                _topic, payload = sub.recv_multipart()
            except zmq.Again:
                continue
            events.append((time.monotonic(), json.loads(payload.decode("utf-8"))))
        sub.close()
        ctx.term()

    collector = threading.Thread(target=collect_events, daemon=True)
    collector.start()

    def start_replay(scenario: str) -> subprocess.Popen:
        return _start(
            [
                sys.executable,
                "-m",
                "replay",
                "--scenario",
                scenario,
                "--rate",
                "100",
                "--target",
                f"localhost:{udp_port}",
            ],
            pythonpath=REPLAY_SRC,
            env=env,
        )

    replay_proc = start_replay("empty_room")

    try:
        # Let calibration complete and the detector settle on empty_room:
        # window fill (WINDOW_S) + calibration (CALIBRATION_S) + margin.
        time.sleep(WINDOW_S + CALIBRATION_S + 3.0)

        assert events, "pipeline never emitted a presence event during empty_room warm-up"
        recent_presence = [e["presence"] for _, e in events[-6:]]
        assert sum(recent_presence) <= 1, (
            f"expected presence to have settled False after empty_room calibration, "
            f"got recent events: {events[-6:]}"
        )

        events.clear()
        replay_proc.terminate()
        replay_proc.wait(timeout=3)
        switch_time = time.monotonic()
        replay_proc = start_replay("one_person_walking")

        deadline = switch_time + 3.0
        first_true_at = None
        while time.monotonic() < deadline:
            for received_at, event in events:
                if event["presence"]:
                    first_true_at = received_at
                    break
            if first_true_at is not None:
                break
            time.sleep(0.05)

        assert first_true_at is not None, (
            f"presence never flipped True within 3s of switching to one_person_walking; "
            f"events received: {events}"
        )
        latency_s = first_true_at - switch_time
        print(f"presence flipped True {latency_s:.2f}s after switching scenarios")
        assert latency_s < 3.0
    finally:
        stop_collecting.set()
        collector.join(timeout=2)
        _stop(replay_proc, pipeline_proc, ingest_proc)
