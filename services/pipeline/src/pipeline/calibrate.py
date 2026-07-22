"""On-site calibration CLI for zone-level localization.

Usage:
    python -m pipeline.calibrate --zone A1 --seconds 60

Stand in a zone (see ../../../room.yaml) while services/replay or a live
ESP32 streams CSI through services/ingest. This subscribes to ingest's CSI
stream, windows + preprocesses it the same way pipeline/service.py does,
computes spectrogram features, and records them as that zone's labeled
calibration samples (services/pipeline/calibration_data/<zone_id>.npz,
overwriting any previous run for that zone).

After saving, it refits pipeline.models.localizer.ZoneLocalizer on every
zone calibrated so far — at least 2 zones' worth of data are needed before
a fit is possible — and saves the updated checkpoint
(services/pipeline/checkpoints/localizer.joblib). So recalibrating a zone
(e.g. after moving furniture) "fine-tunes" the deployed model on the next
run: run this once per zone, then point pipeline/service.py at the result
with --localizer-checkpoint.

Requires the "localize" extra (`pip install -e ".[localize]"`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import zmq

from pipeline.features.spectrogram import compute_spectrogram_features
from pipeline.models.localizer import ZoneLocalizer, load_all_calibration_samples, save_calibration_samples
from pipeline.preprocess import hampel_filter, savitzky_golay_smooth
from pipeline.room import DEFAULT_ROOM_CONFIG_PATH, load_room_config
from pipeline.windowing import RollingWindower

logger = logging.getLogger("pipeline.calibrate")

CSI_TOPIC = b"csi"

_PIPELINE_ROOT = Path(__file__).resolve().parents[2]  # services/pipeline
DEFAULT_CALIBRATION_DIR = _PIPELINE_ROOT / "calibration_data"
DEFAULT_CHECKPOINT_DIR = _PIPELINE_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_NAME = "localizer.joblib"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.calibrate",
        description="Record labeled CSI calibration samples for one room zone and refit the localizer.",
    )
    parser.add_argument("--zone", required=True, help="zone_id to calibrate, e.g. A1 (see room.yaml)")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to collect for (default: 60)")
    parser.add_argument("--sub-host", default="localhost", help="ingest ZeroMQ PUB host (default: localhost)")
    parser.add_argument("--sub-port", type=int, default=5567, help="ingest ZeroMQ PUB port (default: 5567)")
    parser.add_argument(
        "--room-config", type=Path, default=DEFAULT_ROOM_CONFIG_PATH, help="path to room.yaml"
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help=f"where per-zone calibration samples are stored (default: {DEFAULT_CALIBRATION_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"where the fitted localizer checkpoint is saved (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="incoming CSI frame rate (default: 100); must match pipeline/service.py's for consistent windows",
    )
    parser.add_argument("--window-s", type=float, default=2.0, help="window length in seconds (default: 2.0)")
    parser.add_argument("--stride-s", type=float, default=0.5, help="hop between windows in seconds (default: 0.5)")
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=11)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=5, help="PCA/spectrogram channel count")
    parser.add_argument("--nperseg", type=int, default=32)
    parser.add_argument("--noverlap", type=int, default=None)
    return parser


def collect_zone_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Subscribes to ingest's CSI stream for args.seconds seconds; returns
    (features, timestamps_us) for every window collected, where features
    has shape (n_windows, n_components, n_freq, n_time)."""
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://{args.sub_host}:{args.sub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, CSI_TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, 500)

    window_size = max(1, round(args.window_s * args.sample_rate_hz))
    stride_frames = max(1, round(args.stride_s * args.sample_rate_hz))
    windower = RollingWindower(window_size=window_size, stride_frames=stride_frames)

    features_list: list[np.ndarray] = []
    timestamps_list: list[int] = []
    start = time.monotonic()

    logger.info("collecting %.0fs of calibration data for zone %r...", args.seconds, args.zone)
    try:
        while (time.monotonic() - start) < args.seconds:
            try:
                _topic, payload = sub.recv_multipart()
            except zmq.Again:
                continue

            frame = json.loads(payload.decode("utf-8"))
            ready = windower.add(np.array(frame["amplitude"], dtype=np.float64), frame["timestamp_us"])
            if ready is None:
                continue

            raw_window, timestamp_us = ready
            cleaned = savitzky_golay_smooth(
                hampel_filter(raw_window, window_size=args.hampel_window, n_sigmas=args.hampel_sigmas),
                window_length=args.savgol_window,
                polyorder=args.savgol_polyorder,
            )
            features = compute_spectrogram_features(
                cleaned,
                n_components=args.n_components,
                sample_rate_hz=args.sample_rate_hz,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
            )
            features_list.append(features)
            timestamps_list.append(timestamp_us)
            if len(features_list) % 10 == 0:
                logger.info("  collected %d windows so far...", len(features_list))
    finally:
        sub.close()
        context.term()

    if not features_list:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.stack(features_list), np.array(timestamps_list, dtype=np.int64)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.window_s <= 0 or args.stride_s <= 0:
        parser.error("--window-s and --stride-s must be positive")

    room = load_room_config(args.room_config)
    if args.zone not in room.zone_ids:
        parser.error(f"unknown zone {args.zone!r}; configured zones: {', '.join(room.zone_ids)}")

    features, timestamps_us = collect_zone_samples(args)
    if len(features) == 0:
        raise SystemExit(
            "no windows collected — is services/ingest running and streaming CSI frames "
            f"to tcp://{args.sub_host}:{args.sub_port}?"
        )
    logger.info("collected %d windows for zone %r", len(features), args.zone)

    calibration_path = args.calibration_dir / f"{args.zone}.npz"
    save_calibration_samples(calibration_path, zone_id=args.zone, features=features, timestamps_us=timestamps_us)
    logger.info("saved calibration samples to %s", calibration_path)

    X, y, calibrated_zones = load_all_calibration_samples(args.calibration_dir, room.zone_ids)
    if len(calibrated_zones) < 2:
        logger.warning(
            "only %d zone(s) calibrated so far (%s); need at least 2 before the localizer can be "
            "fit — calibrate another zone next (python -m pipeline.calibrate --zone <other zone>)",
            len(calibrated_zones),
            calibrated_zones,
        )
        return

    localizer = ZoneLocalizer(room.zone_ids)
    localizer.fit(X, y)

    checkpoint_path = args.checkpoint_dir / DEFAULT_CHECKPOINT_NAME
    localizer.save(checkpoint_path)
    logger.info(
        "fit localizer on %d samples across %d zone(s) (%s); saved checkpoint to %s",
        len(y),
        len(calibrated_zones),
        calibrated_zones,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
