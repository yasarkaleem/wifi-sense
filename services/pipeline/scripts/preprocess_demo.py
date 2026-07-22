#!/usr/bin/env python3
"""Notebook-style demo: pull live CSI frames from services/ingest's ZeroMQ
PUB socket, run them through pipeline.preprocess, and plot raw vs. cleaned
amplitude/phase side by side.

Start services/ingest (with --pub-port set) and services/replay first, e.g.:

    # terminal 1
    cd services/ingest && python -m ingest --pub-port 5567

    # terminal 2
    cd services/replay && python -m replay --scenario one_person_walking --target localhost:5566

    # terminal 3
    cd services/pipeline && pip install -e ".[demo]"
    python scripts/preprocess_demo.py --host localhost --port 5567

pipeline talks to ingest only over the network (ZeroMQ), never via a shared
Python import, matching the service-isolation rule in ../../../CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pipeline.preprocess import (
    hampel_filter,
    pca_reduce,
    sanitize_phase,
    savitzky_golay_smooth,
    segment_sliding_window,
)

TOPIC = b"csi"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="ingest ZeroMQ PUB host (default: localhost)")
    parser.add_argument("--port", type=int, default=5567, help="ingest ZeroMQ PUB port (default: 5567)")
    parser.add_argument(
        "--num-frames", type=int, default=400, help="frames to collect before plotting (default: 400)"
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="give up if this many seconds pass without a new frame (default: 30)",
    )
    parser.add_argument("--hampel-window", type=int, default=7, help="Hampel filter window size (default: 7)")
    parser.add_argument("--hampel-sigmas", type=float, default=3.0, help="Hampel outlier threshold (default: 3.0)")
    parser.add_argument("--savgol-window", type=int, default=11, help="Savitzky-Golay window length (default: 11)")
    parser.add_argument("--savgol-polyorder", type=int, default=3, help="Savitzky-Golay polynomial order (default: 3)")
    parser.add_argument("--pca-components", type=int, default=5, help="top-k PCA components to report (default: 5)")
    parser.add_argument(
        "--output", default=None, help="save the figure to this path instead of showing it interactively"
    )
    return parser.parse_args(argv)


def collect_frames(host: str, port: int, num_frames: int, timeout_s: float) -> list[dict]:
    import zmq

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC)
    sub.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))

    print(f"connecting to tcp://{host}:{port}, collecting {num_frames} frames...", file=sys.stderr)
    frames: list[dict] = []
    try:
        while len(frames) < num_frames:
            _topic, payload = sub.recv_multipart()
            frames.append(json.loads(payload.decode("utf-8")))
            if len(frames) % 50 == 0:
                print(f"  collected {len(frames)}/{num_frames}", file=sys.stderr)
    except zmq.Again:
        print(f"timed out after {len(frames)} frames (wanted {num_frames})", file=sys.stderr)
    finally:
        sub.close()
        context.term()

    if not frames:
        raise SystemExit(
            "no frames received - is services/ingest running with --pub-port set, "
            "and is services/replay streaming to it?"
        )
    return frames


def frames_to_matrices(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort collected frames by sequence_number and stack them into
    (n_frames, n_subcarriers) amplitude/phase matrices, estimating the
    sample rate from the frames' own timestamps."""
    frames = sorted(frames, key=lambda f: f["sequence_number"])
    amplitude = np.array([f["amplitude"] for f in frames], dtype=np.float64)
    phase = np.array([f["phase"] for f in frames], dtype=np.float64)

    timestamps_us = np.array([f["timestamp_us"] for f in frames], dtype=np.float64)
    duration_s = (timestamps_us[-1] - timestamps_us[0]) / 1e6
    sample_rate_hz = (len(frames) - 1) / duration_s if duration_s > 0 else 100.0

    return amplitude, phase, sample_rate_hz


def plot_raw_vs_cleaned(
    raw_amplitude: np.ndarray,
    cleaned_amplitude: np.ndarray,
    raw_phase: np.ndarray,
    sanitized_phase: np.ndarray,
    *,
    output: str | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], raw_amplitude, "raw amplitude", "viridis"),
        (axes[0, 1], cleaned_amplitude, "cleaned amplitude (Hampel + Savitzky-Golay)", "viridis"),
        (axes[1, 0], raw_phase, "raw phase", "twilight"),
        (axes[1, 1], sanitized_phase, "sanitized phase (unwrap + detrend)", "twilight"),
    ]
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("subcarrier index")
        ax.set_ylabel("frame")
        fig.colorbar(im, ax=ax)

    fig.suptitle("CSI preprocessing: raw vs. cleaned")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"saved figure to {output}", file=sys.stderr)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    frames = collect_frames(args.host, args.port, args.num_frames, args.timeout_s)
    amplitude, phase, sample_rate_hz = frames_to_matrices(frames)
    print(
        f"collected {amplitude.shape[0]} frames x {amplitude.shape[1]} subcarriers "
        f"(~{sample_rate_hz:.1f} Hz)",
        file=sys.stderr,
    )

    cleaned_amplitude = savitzky_golay_smooth(
        hampel_filter(amplitude, window_size=args.hampel_window, n_sigmas=args.hampel_sigmas),
        window_length=args.savgol_window,
        polyorder=args.savgol_polyorder,
    )
    sanitized_phase = sanitize_phase(phase)

    windows = segment_sliding_window(cleaned_amplitude, sample_rate_hz=sample_rate_hz)
    print(f"segmented into {windows.shape[0]} sliding windows of shape {windows.shape[1:]}", file=sys.stderr)

    if windows.shape[0] > 0:
        pooled = windows.reshape(-1, windows.shape[-1])
        n_components = min(args.pca_components, pooled.shape[0], pooled.shape[1])
        pca_result = pca_reduce(pooled, n_components=n_components)
        ratios = ", ".join(f"{r:.3f}" for r in pca_result.explained_variance_ratio)
        print(f"PCA top-{n_components} explained variance ratio: {ratios}", file=sys.stderr)

    plot_raw_vs_cleaned(amplitude, cleaned_amplitude, phase, sanitized_phase, output=args.output)


if __name__ == "__main__":
    main()
