"""Cross-service integration test: spawns real replay + ingest +
pipeline.calibrate + pipeline subprocesses and proves that calibrating two
zones from synthetic replay data, then running the fitted localizer live,
recovers the correct zone.

Unlike the rest of this test suite, this spawns services/replay and
services/ingest as subprocesses under the *same* interpreter pytest is
running under (see pyproject.toml's dev/localize extras) — it does not
import them as Python packages, honoring the service-isolation rule in
../../../CLAUDE.md (network only, no in-process imports across service
boundaries).
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

# The subprocesses this test spawns (pipeline.calibrate, pipeline with
# --localizer-checkpoint) run under this same interpreter/PYTHONPATH, so if
# scikit-learn/joblib aren't importable here, they won't be there either.
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

WIFI_SENSE_ROOT = Path(__file__).resolve().parents[3]
INGEST_SRC = WIFI_SENSE_ROOT / "services" / "ingest" / "src"
REPLAY_SRC = WIFI_SENSE_ROOT / "services" / "replay" / "src"
PIPELINE_SRC = WIFI_SENSE_ROOT / "services" / "pipeline" / "src"

# Small window/stride so a short calibration run yields plenty of windows
# quickly; nperseg shrunk to match the smaller window. These MUST be
# identical between calibration and the live pipeline service below — the
# localizer's flattened features have to match shape exactly (see
# pipeline/models/localizer.py's module docstring).
WINDOW_S = 0.5
STRIDE_S = 0.1
N_COMPONENTS = 5
NPERSEG = 16
CALIBRATION_SECONDS = 3.0
ZONE_A = "A1"
ZONE_B = "B2"


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
        if proc is not None:
            proc.terminate()
    for proc in procs:
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.mark.integration
def test_calibration_and_inference_recover_the_correct_zone(tmp_path):
    if not INGEST_SRC.exists() or not REPLAY_SRC.exists():
        pytest.skip("sibling services (ingest/replay) not found next to pipeline in this checkout")

    udp_port = _free_port()
    csi_pub_port = _free_port()
    zones_pub_port = _free_port()
    env = os.environ.copy()

    calibration_dir = tmp_path / "calibration_data"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_dir / "localizer.joblib"

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

    def run_calibrate(zone: str) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pipeline.calibrate",
                "--zone",
                zone,
                "--seconds",
                str(CALIBRATION_SECONDS),
                "--sub-host",
                "127.0.0.1",
                "--sub-port",
                str(csi_pub_port),
                "--calibration-dir",
                str(calibration_dir),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--window-s",
                str(WINDOW_S),
                "--stride-s",
                str(STRIDE_S),
                "--n-components",
                str(N_COMPONENTS),
                "--nperseg",
                str(NPERSEG),
            ],
            env={**env, "PYTHONPATH": str(PIPELINE_SRC)},
            capture_output=True,
            text=True,
            timeout=CALIBRATION_SECONDS + 15,
        )
        assert result.returncode == 0, (
            f"calibrate --zone {zone} failed (exit {result.returncode}):\n{result.stderr}"
        )

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
    time.sleep(1.0)

    replay_proc = None
    pipeline_proc = None
    try:
        # --- Calibrate two zones ---------------------------------------
        replay_proc = start_replay(ZONE_A)
        run_calibrate(ZONE_A)
        _stop(replay_proc)
        replay_proc = None
        assert not checkpoint_path.exists(), "localizer shouldn't fit with only 1 zone calibrated"

        replay_proc = start_replay(ZONE_B)
        run_calibrate(ZONE_B)
        _stop(replay_proc)
        replay_proc = None
        assert checkpoint_path.exists(), "localizer should have been fit after calibrating a 2nd zone"

        # --- Run the real pipeline service with the fitted localizer ---
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
                str(zones_pub_port),
                "--window-s",
                str(WINDOW_S),
                "--stride-s",
                str(STRIDE_S),
                "--localizer-checkpoint",
                str(checkpoint_path),
                "--localizer-n-components",
                str(N_COMPONENTS),
                "--localizer-nperseg",
                str(NPERSEG),
            ],
            pythonpath=PIPELINE_SRC,
            env=env,
        )
        time.sleep(1.5)

        events: list[dict] = []
        stop_collecting = threading.Event()

        def collect_events() -> None:
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.connect(f"tcp://127.0.0.1:{zones_pub_port}")
            sub.setsockopt(zmq.SUBSCRIBE, b"zones")
            sub.setsockopt(zmq.RCVTIMEO, 200)
            while not stop_collecting.is_set():
                try:
                    _topic, payload = sub.recv_multipart()
                except zmq.Again:
                    continue
                events.append(json.loads(payload.decode("utf-8")))
            sub.close()
            ctx.term()

        collector = threading.Thread(target=collect_events, daemon=True)
        collector.start()

        def best_zone(event: dict) -> str:
            return max(event["zones"], key=lambda z: z["occupancy_probability"])["zone_id"]

        # --- Stream a FRESH zone-A recording (different seed) and check
        #     the pipeline's live predictions converge on zone A --------
        replay_proc = start_replay(ZONE_A)
        deadline = time.monotonic() + 10.0
        recent: list[str] = []
        while time.monotonic() < deadline:
            if len(events) >= 5:
                recent = [best_zone(e) for e in events[-5:]]
                if recent.count(ZONE_A) >= 4:
                    break
            time.sleep(0.1)
        assert recent.count(ZONE_A) >= 4, f"expected zone {ZONE_A} to dominate recent predictions, got: {recent}"

        # --- Switch to a FRESH zone-B recording; predictions should
        #     follow -----------------------------------------------------
        _stop(replay_proc)
        events.clear()
        replay_proc = start_replay(ZONE_B)
        deadline = time.monotonic() + 10.0
        recent = []
        while time.monotonic() < deadline:
            if len(events) >= 5:
                recent = [best_zone(e) for e in events[-5:]]
                if recent.count(ZONE_B) >= 4:
                    break
            time.sleep(0.1)
        assert recent.count(ZONE_B) >= 4, f"expected zone {ZONE_B} to dominate recent predictions, got: {recent}"

        stop_collecting.set()
        collector.join(timeout=2)
    finally:
        _stop(replay_proc, pipeline_proc, ingest_proc)
