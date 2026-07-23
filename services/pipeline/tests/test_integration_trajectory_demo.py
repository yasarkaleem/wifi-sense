"""Cross-service integration test for the walking-person demo (task steps
1-4): generates a small offline training dataset and trains tiny demo
checkpoints in-process (scripts/generate_zone_dataset.py,
scripts/train_demo_models.py — build-time tools, not services, so this is
consistent with their own documented in-process-import exception), then
spawns real replay/ingest/pipeline subprocesses streaming the
`one_person_walking_path` trajectory scenario and proves:

  - step 1: the trajectory scenario streams real CSI that visibly moves
    through multiple zones over time (not a static signal).
  - steps 2-3: the offline dataset + training scripts produce checkpoints
    pipeline.service can actually load and run inference with.
  - step 4: smoothed zone probabilities keep arriving on the `zones`
    topic from the live pipeline (exact EMA math is unit-tested in
    test_smoothing.py, not re-derived here).

The dashboard visual (step 5) and the full docker-compose walkthrough
(step 6) are out of scope for this test — see CLAUDE.md for the manual
verification steps.
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

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import generate_zone_dataset  # noqa: E402
import train_demo_models  # noqa: E402
import zmq  # noqa: E402

WIFI_SENSE_ROOT = Path(__file__).resolve().parents[3]
INGEST_SRC = WIFI_SENSE_ROOT / "services" / "ingest" / "src"
REPLAY_SRC = WIFI_SENSE_ROOT / "services" / "replay" / "src"
PIPELINE_SRC = WIFI_SENSE_ROOT / "services" / "pipeline" / "src"

# Small window/stride/nperseg so both the offline training data and the
# live pipeline run generate/consume windows quickly. These MUST match
# between generate_zone_dataset.py/train_demo_models.py and the live
# `pipeline` invocation below — same constraint test_integration_localization.py
# documents for calibration vs. inference.
WINDOW_S = 0.5
STRIDE_S = 0.2
N_COMPONENTS = 5
NPERSEG = 16
RATE_HZ = 100.0
SCENARIO = "one_person_walking_path"


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


@pytest.fixture(scope="module")
def demo_checkpoints(tmp_path_factory) -> dict[str, Path]:
    """Generates a small dataset and trains tiny demo checkpoints once for
    every test in this module — proves steps 2-3 actually work, not just
    that shipped checkpoints happen to exist."""
    work_dir = tmp_path_factory.mktemp("trajectory_demo")
    dataset_path = work_dir / "zone_demo.npz"
    models_dir = work_dir / "models"

    generate_zone_dataset.main(
        [
            "--seconds-per-recording",
            "3.0",
            "--recordings-per-class",
            "3",
            "--rate-hz",
            str(RATE_HZ),
            "--window-s",
            str(WINDOW_S),
            "--stride-s",
            str(STRIDE_S),
            "--hampel-window",
            "3",
            "--savgol-window",
            "5",
            "--savgol-polyorder",
            "2",
            "--seed",
            "42",
            "--output",
            str(dataset_path),
        ]
    )
    assert dataset_path.exists()

    train_demo_models.main(
        [
            "--dataset",
            str(dataset_path),
            "--n-components",
            str(N_COMPONENTS),
            "--nperseg",
            str(NPERSEG),
            "--epochs",
            "10",
            "--patience",
            "5",
            "--output-dir",
            str(models_dir),
        ]
    )

    counter_path = models_dir / "counter_demo.pt"
    localizer_path = models_dir / "localizer_demo.joblib"
    assert counter_path.exists()
    assert localizer_path.exists()
    return {"counter": counter_path, "localizer": localizer_path}


@pytest.mark.integration
def test_trajectory_scenario_produces_moving_smoothed_zone_and_count_1(demo_checkpoints):
    if not INGEST_SRC.exists() or not REPLAY_SRC.exists():
        pytest.skip("sibling services (ingest/replay) not found next to pipeline in this checkout")

    udp_port = _free_port()
    csi_pub_port = _free_port()
    events_pub_port = _free_port()
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
    time.sleep(1.0)

    replay_proc = None
    pipeline_proc = None
    try:
        replay_proc = _start(
            [
                sys.executable,
                "-m",
                "replay",
                "--scenario",
                SCENARIO,
                "--rate",
                str(RATE_HZ),
                "--target",
                f"localhost:{udp_port}",
            ],
            pythonpath=REPLAY_SRC,
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
                str(events_pub_port),
                "--window-s",
                str(WINDOW_S),
                "--stride-s",
                str(STRIDE_S),
                "--counter-checkpoint",
                str(demo_checkpoints["counter"]),
                "--counter-n-components",
                str(N_COMPONENTS),
                "--counter-nperseg",
                str(NPERSEG),
                "--localizer-checkpoint",
                str(demo_checkpoints["localizer"]),
                "--localizer-n-components",
                str(N_COMPONENTS),
                "--localizer-nperseg",
                str(NPERSEG),
            ],
            pythonpath=PIPELINE_SRC,
            env=env,
        )
        time.sleep(1.5)

        count_events: list[dict] = []
        zone_events: list[dict] = []
        stop_collecting = threading.Event()

        def collect_events() -> None:
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.connect(f"tcp://127.0.0.1:{events_pub_port}")
            sub.setsockopt(zmq.SUBSCRIBE, b"count")
            sub.setsockopt(zmq.SUBSCRIBE, b"zones")
            sub.setsockopt(zmq.RCVTIMEO, 200)
            while not stop_collecting.is_set():
                try:
                    topic, payload = sub.recv_multipart()
                except zmq.Again:
                    continue
                event = json.loads(payload.decode("utf-8"))
                if topic == b"count":
                    count_events.append(event)
                else:
                    zone_events.append(event)
            sub.close()
            ctx.term()

        collector = threading.Thread(target=collect_events, daemon=True)
        collector.start()

        # one_person_walking_path's loop is 20s (5+4+6+5); collect for long
        # enough to span at least two zone transitions.
        deadline = time.monotonic() + 18.0
        while time.monotonic() < deadline:
            if len(zone_events) >= 20 and len(count_events) >= 5:
                break
            time.sleep(0.2)

        stop_collecting.set()
        collector.join(timeout=2)
    finally:
        _stop(replay_proc, pipeline_proc, ingest_proc)

    assert len(count_events) > 0, "no count events received from the live pipeline"
    assert len(zone_events) > 0, "no zone events received from the live pipeline"

    # step 3: the demo counter should dominantly say "1 person".
    counts = [e["count"] for e in count_events]
    most_common_count = max(set(counts), key=counts.count)
    assert most_common_count == 1, f"expected count=1 to dominate, got counts: {counts}"

    # step 1: the trajectory genuinely moves through multiple zones over
    # time, rather than being a static disturbance the localizer pins to
    # one zone forever.
    def best_zone(event: dict) -> str:
        return max(event["zones"], key=lambda z: z["occupancy_probability"])["zone_id"]

    best_zones_seen = {best_zone(e) for e in zone_events}
    assert len(best_zones_seen) >= 2, f"expected the walking path to visit >= 2 zones, saw: {best_zones_seen}"

    # step 4: smoothed zone probabilities still arrive with the documented
    # shape (EMA math itself is unit-tested in test_smoothing.py).
    for event in zone_events:
        total = sum(z["occupancy_probability"] for z in event["zones"])
        assert total == pytest.approx(1.0, abs=0.05)
