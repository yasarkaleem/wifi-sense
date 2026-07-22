#!/usr/bin/env python3
"""Converts public WiFi CSI datasets into wifi-sense's canonical CSI frame
schema (../docs/csi-frame-schema.md), saved as one .npz file per recording
session under datasets/<name>/.

Supported:
    ut-har   UT-HAR (Yousefi et al., 2017). Fully implemented — see
             convert_ut_har() / ConvertedSession.
    widar3   Widar 3.0 (Zheng et al., 2019). Stub only — see
             convert_widar3() for why.

This only does *conversion*; it does not auto-download the (large,
interactively-gated) source archives. See README.md for manual download
steps, dataset licenses, and citation requirements before running this.

Usage:
    python download.py ut-har --source /path/to/Dataset/Data --out ut-har
    python download.py widar3 --source ... --out widar3   # raises NotImplementedError
    python download.py --list
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent

# UT-HAR was captured with the Linux 802.11n CSI Tool on an Intel 5300 NIC.
# Its raw per-packet export (see _parse_ut_har_input_csv) has no RSSI or
# WiFi channel column, so these are placeholders that let converted frames
# satisfy the canonical schema (which requires both, and requires channel
# >= 1 — see docs/csi-frame-schema.md) — not measured values.
UT_HAR_PLACEHOLDER_RSSI = 0
UT_HAR_PLACEHOLDER_CHANNEL = 1
UT_HAR_SOURCE_MAC = "00:00:00:00:00:00"  # not reported by the source dataset
UT_HAR_SUBCARRIER_COUNT = 90  # 30 OFDM subcarriers x 3 antennas, flattened (see docs/csi-frame-schema.md's notes)
UT_HAR_ACTIVITIES = ("bed", "fall", "walk", "pickup", "run", "sitdown", "standup")
UT_HAR_INPUT_COLUMNS = 1 + 90 + 90  # timestamp + amplitude(30 subcarriers x 3 antennas) + phase(same)


@dataclass(frozen=True)
class ConvertedSession:
    """One raw recording session, converted to the canonical CSI frame
    fields, plus a `label` array that is a training-only convenience field
    — NOT part of docs/csi-frame-schema.md. services/replay strips it when
    streaming these frames (see replay.dataset / replay.stream.stream_dataset);
    it exists here purely for scripts/train_har.py's supervised labels.
    """

    session_id: str
    timestamp_us: np.ndarray  # (n_frames,) int64
    amplitude: np.ndarray  # (n_frames, subcarrier_count) float64
    phase: np.ndarray  # (n_frames, subcarrier_count) float64
    rssi: np.ndarray  # (n_frames,) int64
    channel: np.ndarray  # (n_frames,) int64
    source_mac: str
    subcarrier_count: int
    label: np.ndarray  # (n_frames,) <U16, activity string, "" if none

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            session_id=self.session_id,
            timestamp_us=self.timestamp_us,
            amplitude=self.amplitude,
            phase=self.phase,
            rssi=self.rssi,
            channel=self.channel,
            source_mac=self.source_mac,
            subcarrier_count=self.subcarrier_count,
            label=self.label,
        )


# ---------------------------------------------------------------------------
# UT-HAR
# ---------------------------------------------------------------------------


def _parse_ut_har_input_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one UT-HAR `input_*.csv`.

    Column 0 is a timestamp (seconds, possibly fractional); columns 1:91
    are amplitude (30 subcarriers x 3 antennas, flattened); columns 91:181
    are phase (same layout). One row per captured CSI packet.
    """
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.shape[1] != UT_HAR_INPUT_COLUMNS:
        raise ValueError(
            f"{path}: expected {UT_HAR_INPUT_COLUMNS} columns "
            f"(1 timestamp + 90 amplitude + 90 phase), got {raw.shape[1]}"
        )
    timestamp_s = raw[:, 0]
    amplitude = raw[:, 1:91]
    phase = raw[:, 91:181]
    return timestamp_s, amplitude, phase


def _parse_ut_har_annotation_csv(path: Path) -> np.ndarray:
    """Parse one UT-HAR `annotation_*.csv`: one activity-label string per
    row (one of UT_HAR_ACTIVITIES, or blank/other for no activity), row-
    aligned with the matching input_*.csv."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row[0] if row else "" for row in csv.reader(f)]
    return np.array(rows, dtype="<U16")


def _find_ut_har_session_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """Match each input_*.csv to its corresponding annotation_*.csv by
    shared suffix (the part after the "input_"/"annotation_" prefix)."""
    inputs = {p.name[len("input_") :]: p for p in sorted(source_dir.glob("input_*.csv"))}
    annotations = {p.name[len("annotation_") :]: p for p in sorted(source_dir.glob("annotation_*.csv"))}
    common = sorted(set(inputs) & set(annotations))
    missing_annotation = sorted(set(inputs) - set(annotations))
    if missing_annotation:
        preview = missing_annotation[:5]
        suffix = "..." if len(missing_annotation) > 5 else ""
        print(
            f"WARNING: {len(missing_annotation)} input_*.csv file(s) have no matching "
            f"annotation_*.csv, skipping: {preview}{suffix}",
            file=sys.stderr,
        )
    return [(inputs[k], annotations[k]) for k in common]


def convert_ut_har_session(input_path: Path, annotation_path: Path) -> ConvertedSession:
    timestamp_s, amplitude, phase = _parse_ut_har_input_csv(input_path)
    label = _parse_ut_har_annotation_csv(annotation_path)

    n = min(len(timestamp_s), len(label))
    if len(timestamp_s) != len(label):
        print(
            f"WARNING: {input_path.name} has {len(timestamp_s)} frames but "
            f"{annotation_path.name} has {len(label)} labels; truncating both to {n}",
            file=sys.stderr,
        )
    timestamp_s, amplitude, phase, label = timestamp_s[:n], amplitude[:n], phase[:n], label[:n]

    session_id = input_path.name[len("input_") : -len(".csv")]
    # Source timestamps are seconds (float, Linux CSI Tool capture clock);
    # canonical frames use microseconds, matching a CSI frame's timestamp_us.
    timestamp_us = np.round(timestamp_s * 1_000_000).astype(np.int64)

    return ConvertedSession(
        session_id=session_id,
        timestamp_us=timestamp_us,
        amplitude=amplitude,
        phase=phase,
        rssi=np.full(n, UT_HAR_PLACEHOLDER_RSSI, dtype=np.int64),
        channel=np.full(n, UT_HAR_PLACEHOLDER_CHANNEL, dtype=np.int64),
        source_mac=UT_HAR_SOURCE_MAC,
        subcarrier_count=UT_HAR_SUBCARRIER_COUNT,
        label=label,
    )


def convert_ut_har(source_dir: Path, output_dir: Path) -> list[Path]:
    """Convert every input_*.csv/annotation_*.csv session pair under
    `source_dir` (UT-HAR's `Dataset/Data/` directory) into one .npz per
    session under `output_dir`."""
    pairs = _find_ut_har_session_pairs(source_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no input_*.csv / annotation_*.csv pairs found under {source_dir}; "
            "see datasets/README.md for the expected UT-HAR directory layout"
        )

    written = []
    for input_path, annotation_path in pairs:
        session = convert_ut_har_session(input_path, annotation_path)
        out_path = output_dir / f"{session.session_id}.npz"
        session.save(out_path)
        activity_counts = {
            activity: int((session.label == activity).sum())
            for activity in UT_HAR_ACTIVITIES
            if (session.label == activity).any()
        }
        print(
            f"  {session.session_id}: {len(session.label)} frames -> {out_path} ({activity_counts})",
            file=sys.stderr,
        )
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Widar 3.0 (stub)
# ---------------------------------------------------------------------------


def convert_widar3(source_dir: Path, output_dir: Path) -> list[Path]:
    """Stub — not implemented.

    Widar 3.0 (Zheng et al., MobiSys 2019) stores CSI as complex-valued
    `.mat` files (`scipy.io.loadmat`, not a flat per-packet CSV), keyed by
    a room/user/gesture/torso-location/orientation/receiver "domain" tuple
    rather than UT-HAR's simple per-session file pair, and ships derived
    Doppler Frequency Shift (DFS) and Body-coordinate Velocity Profile
    (BVP) features alongside the raw complex CSI. Converting it needs a
    materially different parser (complex amplitude/phase extraction,
    domain-aware session grouping, optional DFS/BVP passthrough) — out of
    scope here; raise clearly rather than pretend to support it.
    """
    raise NotImplementedError(
        "Widar 3.0 conversion is not implemented yet (different format: complex .mat CSI "
        "with DFS/BVP features and domain-tuple-keyed sessions, not a flat per-packet CSV — "
        "see convert_widar3()'s docstring). Request dataset access at "
        "http://tns.thss.tsinghua.edu.cn/widar3.0/ if you want to build this next."
    )


DATASETS = {
    "ut-har": convert_ut_har,
    "widar3": convert_widar3,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS), nargs="?", help="dataset to convert")
    parser.add_argument(
        "--source", type=Path, help="path to the extracted raw dataset directory (see README.md)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory for .npz files (default: datasets/<dataset>/)"
    )
    parser.add_argument("--list", action="store_true", help="list supported datasets and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list or args.dataset is None:
        print("Supported datasets:", file=sys.stderr)
        print(
            "  ut-har   UT-HAR, human activity recognition (Yousefi et al., 2017) — implemented",
            file=sys.stderr,
        )
        print(
            "  widar3   Widar 3.0, gesture recognition (Zheng et al., 2019) — stub, not yet implemented",
            file=sys.stderr,
        )
        print("\nSee datasets/README.md for licenses, citations, and manual download steps.", file=sys.stderr)
        parser.exit(0 if args.list else 1)

    if args.source is None:
        parser.error("--source is required (point it at the extracted raw dataset directory; see datasets/README.md)")
    if not args.source.exists():
        parser.error(f"--source {args.source} does not exist")

    output_dir = args.out or (DEFAULT_OUTPUT_ROOT / args.dataset)
    print(f"converting {args.dataset!r} from {args.source} -> {output_dir}", file=sys.stderr)

    written = DATASETS[args.dataset](args.source, output_dir)

    print(f"\nwrote {len(written)} session .npz file(s) to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
