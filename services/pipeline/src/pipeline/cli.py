"""Command-line interface for the pipeline service.

Usage:
    python -m pipeline --sub-port 5567 --pub-port 5568

Defaults for every flag can also be set via environment variables
(PIPELINE_SUB_HOST, PIPELINE_SUB_PORT, PIPELINE_PUB_HOST, PIPELINE_PUB_PORT,
PIPELINE_SAMPLE_RATE_HZ, PIPELINE_WINDOW_S, PIPELINE_STRIDE_S,
PIPELINE_CALIBRATION_S, PIPELINE_N_SIGMAS, PIPELINE_PCA_COMPONENTS,
PIPELINE_HAMPEL_WINDOW, PIPELINE_HAMPEL_SIGMAS, PIPELINE_SAVGOL_WINDOW,
PIPELINE_SAVGOL_POLYORDER, PIPELINE_COUNTER_CHECKPOINT,
PIPELINE_COUNTER_N_COMPONENTS, PIPELINE_COUNTER_NPERSEG,
PIPELINE_COUNTER_NOVERLAP, PIPELINE_LOCALIZER_CHECKPOINT,
PIPELINE_LOCALIZER_N_COMPONENTS, PIPELINE_LOCALIZER_NPERSEG,
PIPELINE_LOCALIZER_NOVERLAP), which is how docker-compose.yml configures
this service.

ML-based people counting (publishing {timestamp, count, confidence} on the
`count` topic) is off by default and only activates when
--counter-checkpoint / PIPELINE_COUNTER_CHECKPOINT points at a checkpoint
produced by scripts/train_counter.py — see ../../../CLAUDE.md. It requires
the "ml" extra (`pip install -e ".[ml]"`); presence detection alone does
not need PyTorch.

Zone-level localization (publishing {timestamp, zones: [{zone_id,
occupancy_probability}, ...]} on the `zones` topic) is similarly off by
default, activating with --localizer-checkpoint / PIPELINE_LOCALIZER_CHECKPOINT
pointing at a checkpoint produced by `python -m pipeline.calibrate` — see
../../../CLAUDE.md's "Zone-level localization" section. It requires the
"localize" extra (`pip install -e ".[localize]"`). Its
--localizer-n-components/--localizer-nperseg/--localizer-noverlap MUST
match what the checkpoint was calibrated with exactly (unlike the
counter's CNN, the localizer's gradient-boosted classifier isn't
spatial-size-agnostic) — --window-s/--stride-s are shared with presence
detection and the counter, so those must match too.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_SUB_HOST = os.environ.get("PIPELINE_SUB_HOST", "localhost")
DEFAULT_SUB_PORT = int(os.environ.get("PIPELINE_SUB_PORT", "5567"))
DEFAULT_PUB_HOST = os.environ.get("PIPELINE_PUB_HOST", "0.0.0.0")
DEFAULT_PUB_PORT = int(os.environ.get("PIPELINE_PUB_PORT", "5568"))
DEFAULT_SAMPLE_RATE_HZ = float(os.environ.get("PIPELINE_SAMPLE_RATE_HZ", "100"))
DEFAULT_WINDOW_S = float(os.environ.get("PIPELINE_WINDOW_S", "2.0"))
DEFAULT_STRIDE_S = float(os.environ.get("PIPELINE_STRIDE_S", "0.5"))
DEFAULT_CALIBRATION_S = float(os.environ.get("PIPELINE_CALIBRATION_S", "5.0"))
DEFAULT_N_SIGMAS = float(os.environ.get("PIPELINE_N_SIGMAS", "6.0"))
DEFAULT_PCA_COMPONENTS = int(os.environ.get("PIPELINE_PCA_COMPONENTS", "5"))
DEFAULT_HAMPEL_WINDOW = int(os.environ.get("PIPELINE_HAMPEL_WINDOW", "7"))
DEFAULT_HAMPEL_SIGMAS = float(os.environ.get("PIPELINE_HAMPEL_SIGMAS", "3.0"))
DEFAULT_SAVGOL_WINDOW = int(os.environ.get("PIPELINE_SAVGOL_WINDOW", "11"))
DEFAULT_SAVGOL_POLYORDER = int(os.environ.get("PIPELINE_SAVGOL_POLYORDER", "3"))
DEFAULT_COUNTER_CHECKPOINT = os.environ.get("PIPELINE_COUNTER_CHECKPOINT") or None
DEFAULT_COUNTER_N_COMPONENTS = int(os.environ.get("PIPELINE_COUNTER_N_COMPONENTS", "5"))
DEFAULT_COUNTER_NPERSEG = int(os.environ.get("PIPELINE_COUNTER_NPERSEG", "32"))
_counter_noverlap_env = os.environ.get("PIPELINE_COUNTER_NOVERLAP")
DEFAULT_COUNTER_NOVERLAP = int(_counter_noverlap_env) if _counter_noverlap_env else None
DEFAULT_LOCALIZER_CHECKPOINT = os.environ.get("PIPELINE_LOCALIZER_CHECKPOINT") or None
DEFAULT_LOCALIZER_N_COMPONENTS = int(os.environ.get("PIPELINE_LOCALIZER_N_COMPONENTS", "5"))
DEFAULT_LOCALIZER_NPERSEG = int(os.environ.get("PIPELINE_LOCALIZER_NPERSEG", "32"))
_localizer_noverlap_env = os.environ.get("PIPELINE_LOCALIZER_NOVERLAP")
DEFAULT_LOCALIZER_NOVERLAP = int(_localizer_noverlap_env) if _localizer_noverlap_env else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Subscribe to ingest's CSI stream, preprocess it, run rule-based "
        "presence detection, and publish detections over ZeroMQ.",
    )
    parser.add_argument(
        "--sub-host",
        default=DEFAULT_SUB_HOST,
        help=f"ingest ZeroMQ PUB host to subscribe to (default: {DEFAULT_SUB_HOST})",
    )
    parser.add_argument(
        "--sub-port",
        type=int,
        default=DEFAULT_SUB_PORT,
        help=f"ingest ZeroMQ PUB port (default: {DEFAULT_SUB_PORT})",
    )
    parser.add_argument(
        "--pub-host",
        default=DEFAULT_PUB_HOST,
        help=f"presence-events ZeroMQ PUB bind host (default: {DEFAULT_PUB_HOST})",
    )
    parser.add_argument(
        "--pub-port",
        type=int,
        default=DEFAULT_PUB_PORT,
        help=f"presence-events ZeroMQ PUB bind port (default: {DEFAULT_PUB_PORT})",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="incoming CSI frame rate, used to convert window/stride seconds to "
        f"frame counts (default: {DEFAULT_SAMPLE_RATE_HZ})",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=DEFAULT_WINDOW_S,
        help=f"presence-detection window length in seconds (default: {DEFAULT_WINDOW_S})",
    )
    parser.add_argument(
        "--stride-s",
        type=float,
        default=DEFAULT_STRIDE_S,
        help=f"hop between windows in seconds (default: {DEFAULT_STRIDE_S})",
    )
    parser.add_argument(
        "--calibration-s",
        type=float,
        default=DEFAULT_CALIBRATION_S,
        help=f"empty-room baseline calibration duration in seconds (default: {DEFAULT_CALIBRATION_S})",
    )
    parser.add_argument(
        "--n-sigmas",
        type=float,
        default=DEFAULT_N_SIGMAS,
        help="presence threshold, in standard deviations above the calibrated "
        f"baseline (default: {DEFAULT_N_SIGMAS})",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help=f"top-k PCA components the motion score is computed from (default: {DEFAULT_PCA_COMPONENTS})",
    )
    parser.add_argument(
        "--hampel-window",
        type=int,
        default=DEFAULT_HAMPEL_WINDOW,
        help=f"Hampel filter window size (default: {DEFAULT_HAMPEL_WINDOW})",
    )
    parser.add_argument(
        "--hampel-sigmas",
        type=float,
        default=DEFAULT_HAMPEL_SIGMAS,
        help=f"Hampel outlier threshold (default: {DEFAULT_HAMPEL_SIGMAS})",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=DEFAULT_SAVGOL_WINDOW,
        help=f"Savitzky-Golay window length (default: {DEFAULT_SAVGOL_WINDOW})",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=DEFAULT_SAVGOL_POLYORDER,
        help=f"Savitzky-Golay polynomial order (default: {DEFAULT_SAVGOL_POLYORDER})",
    )
    parser.add_argument(
        "--counter-checkpoint",
        default=DEFAULT_COUNTER_CHECKPOINT,
        help="path to a scripts/train_counter.py checkpoint; enables ML people counting "
        "(publishes on the 'count' topic) when set. Requires the 'ml' extra. "
        f"(default: {DEFAULT_COUNTER_CHECKPOINT})",
    )
    parser.add_argument(
        "--counter-n-components",
        type=int,
        default=DEFAULT_COUNTER_N_COMPONENTS,
        help="PCA/spectrogram channel count the counter checkpoint was trained with "
        f"(default: {DEFAULT_COUNTER_N_COMPONENTS})",
    )
    parser.add_argument(
        "--counter-nperseg",
        type=int,
        default=DEFAULT_COUNTER_NPERSEG,
        help=f"STFT segment length in samples, must match training (default: {DEFAULT_COUNTER_NPERSEG})",
    )
    parser.add_argument(
        "--counter-noverlap",
        type=int,
        default=DEFAULT_COUNTER_NOVERLAP,
        help="STFT segment overlap in samples, must match training "
        f"(default: {DEFAULT_COUNTER_NOVERLAP}, i.e. nperseg // 2)",
    )
    parser.add_argument(
        "--localizer-checkpoint",
        default=DEFAULT_LOCALIZER_CHECKPOINT,
        help="path to a pipeline.calibrate checkpoint (localizer.joblib); enables zone-level "
        "localization (publishes on the 'zones' topic) when set. Requires the 'localize' extra. "
        f"(default: {DEFAULT_LOCALIZER_CHECKPOINT})",
    )
    parser.add_argument(
        "--localizer-n-components",
        type=int,
        default=DEFAULT_LOCALIZER_N_COMPONENTS,
        help="PCA/spectrogram channel count the localizer was calibrated with "
        f"(default: {DEFAULT_LOCALIZER_N_COMPONENTS})",
    )
    parser.add_argument(
        "--localizer-nperseg",
        type=int,
        default=DEFAULT_LOCALIZER_NPERSEG,
        help=f"STFT segment length in samples, must match calibration exactly (default: {DEFAULT_LOCALIZER_NPERSEG})",
    )
    parser.add_argument(
        "--localizer-noverlap",
        type=int,
        default=DEFAULT_LOCALIZER_NOVERLAP,
        help="STFT segment overlap in samples, must match calibration exactly "
        f"(default: {DEFAULT_LOCALIZER_NOVERLAP}, i.e. nperseg // 2)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.window_s <= 0 or args.stride_s <= 0:
        parser.error("--window-s and --stride-s must be positive")
    if args.sample_rate_hz <= 0:
        parser.error("--sample-rate-hz must be positive")

    from pipeline.service import run_service  # local import keeps --help fast

    run_service(args)


if __name__ == "__main__":
    main()
