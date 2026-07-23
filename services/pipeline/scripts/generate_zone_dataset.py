#!/usr/bin/env python3
"""Generates labeled CSI training windows for the "walking person" demo,
entirely offline (no subprocess, no network, no real-time sleep).

Unlike scripts/train_counter.py (which spawns real services/replay +
services/ingest subprocesses and records in real time, per ../../../CLAUDE.md's
service-isolation rule), this script imports services/replay's generator
directly in-process. That's a deliberate, narrow exception: this is a
build-time data-generation tool, not a running service, and offline
synthesis is what makes generating enough data fast — a real-time
recording of every zone plus both trajectory scenarios would need tens of
minutes of wall-clock time; direct in-process generation takes seconds.

Produces one dataset for two different downstream consumers
(scripts/train_demo_models.py):
  - Zone-labeled windows (one static per-zone scenario, room.yaml's
    A1..B3) for training the zone localizer.
  - Count-labeled windows (empty_room=0, one_person_walking_path=1,
    two_people_walking_paths=2) for training the people counter.
`empty_room` windows carry both a zone_label ("", i.e. unlabeled) and a
count_label (0), serving both consumers from the same recordings.

Generates `--recordings-per-class` INDEPENDENT recordings per class (each
its own random seed), not one long continuous one — matching
scripts/train_counter.py's own `--recordings-per-scenario` for the same
reason: a single continuous recording's windows overlap heavily (75% at
the default 2s window / 0.5s stride) and share one noise realization, so a
classifier trained on it can overfit to that one realization rather than
the zone's general signature. Verified empirically during development:
switching from one recording to several independent ones (plus giving
each zone scenario its own distinct walk_frequency_hz — see
scenarios.yaml) took the localizer from near-random on a fresh recording
to consistently correct.

Usage:
    pip install -e ".[dev,ml,localize]"
    python scripts/generate_zone_dataset.py

Output (default services/pipeline/datasets/zone_demo.npz, gitignored —
regenerate rather than commit, same convention as services/pipeline/checkpoints/):
    amplitude:      (N, window_size, subcarrier_count) float64, hampel+savgol
                     cleaned and windowed, matching pipeline/service.py's
                     live preprocessing exactly
    zone_label:      (N,) string, one of room.yaml's zone_ids or "" if this
                     window doesn't represent a single static zone
    count_label:     (N,) int64, 0/1/2
    source:          (N,) string, which scenario this window came from
    window_s, stride_s, sample_rate_hz: scalars, so downstream feature
                     extraction can verify it's matching what these windows
                     were actually windowed at
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WIFI_SENSE_ROOT = Path(__file__).resolve().parents[3]
REPLAY_SRC = WIFI_SENSE_ROOT / "services" / "replay" / "src"
if str(REPLAY_SRC) not in sys.path:
    sys.path.insert(0, str(REPLAY_SRC))

from replay.generator import synthesize_recording  # noqa: E402
from replay.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios  # noqa: E402

from pipeline.preprocess import hampel_filter, savitzky_golay_smooth, segment_sliding_window
from pipeline.room import DEFAULT_ROOM_CONFIG_PATH, load_room_config

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "datasets" / "zone_demo.npz"

EMPTY_ROOM_SCENARIO = "empty_room"
ONE_PERSON_TRAJECTORY_SCENARIO = "one_person_walking_path"
TWO_PEOPLE_TRAJECTORY_SCENARIO = "two_people_walking_paths"


def _windows_for_recording(
    amplitude: np.ndarray,
    *,
    sample_rate_hz: float,
    window_s: float,
    stride_s: float,
    hampel_window: int,
    hampel_sigmas: float,
    savgol_window: int,
    savgol_polyorder: int,
) -> np.ndarray:
    """Preprocess (hampel + savgol, matching pipeline/service.py exactly)
    and window one recording's raw amplitude matrix."""
    cleaned = savitzky_golay_smooth(
        hampel_filter(amplitude, window_size=hampel_window, n_sigmas=hampel_sigmas),
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )
    return segment_sliding_window(cleaned, sample_rate_hz=sample_rate_hz, window_s=window_s, stride_s=stride_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seconds-per-recording", type=float, default=6.0, help="length of each independent recording"
    )
    parser.add_argument(
        "--recordings-per-class", type=int, default=5, help="independent (differently-seeded) recordings per class"
    )
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=11)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios-file", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--room-config", type=Path, default=DEFAULT_ROOM_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    scenarios = load_scenarios(args.scenarios_file)
    room = load_room_config(args.room_config)

    all_amplitude: list[np.ndarray] = []
    all_zone_label: list[str] = []
    all_count_label: list[int] = []
    all_source: list[str] = []

    def add_class(scenario_name: str, *, zone_label: str, count_label: int, seed_base: int) -> None:
        total_windows = 0
        for recording_idx in range(args.recordings_per_class):
            seed = seed_base * 1000 + recording_idx
            scenario = scenarios[scenario_name]
            amplitude = synthesize_recording(
                scenario, rate_hz=args.rate_hz, duration_s=args.seconds_per_recording, seed=seed
            )
            windows = _windows_for_recording(
                amplitude,
                sample_rate_hz=args.rate_hz,
                window_s=args.window_s,
                stride_s=args.stride_s,
                hampel_window=args.hampel_window,
                hampel_sigmas=args.hampel_sigmas,
                savgol_window=args.savgol_window,
                savgol_polyorder=args.savgol_polyorder,
            )
            all_amplitude.append(windows)
            all_zone_label.extend([zone_label] * len(windows))
            all_count_label.extend([count_label] * len(windows))
            all_source.extend([scenario_name] * len(windows))
            total_windows += len(windows)
        print(
            f"generated {args.recordings_per_class} recording(s) of {scenario_name!r} "
            f"(zone_label={zone_label!r}, count_label={count_label}) -> {total_windows} windows",
            file=sys.stderr,
        )

    add_class(EMPTY_ROOM_SCENARIO, zone_label="", count_label=0, seed_base=args.seed)

    for i, zone_id in enumerate(room.zone_ids):
        if zone_id not in scenarios:
            raise SystemExit(
                f"room.yaml defines zone {zone_id!r} but {args.scenarios_file} has no matching scenario "
                "— every zone_id needs a same-named static scenario (see scenarios.yaml's A1..B3 block)"
            )
        add_class(zone_id, zone_label=zone_id, count_label=1, seed_base=args.seed + 1 + i)

    add_class(
        ONE_PERSON_TRAJECTORY_SCENARIO,
        zone_label="",
        count_label=1,
        seed_base=args.seed + 1 + len(room.zone_ids),
    )
    add_class(
        TWO_PEOPLE_TRAJECTORY_SCENARIO,
        zone_label="",
        count_label=2,
        seed_base=args.seed + 2 + len(room.zone_ids),
    )

    amplitude = np.concatenate(all_amplitude, axis=0)
    zone_label = np.array(all_zone_label, dtype="<U8")
    count_label = np.array(all_count_label, dtype=np.int64)
    source = np.array(all_source, dtype="<U40")

    print(f"\ntotal dataset: {len(count_label)} windows, amplitude shape {amplitude.shape}", file=sys.stderr)
    print(f"  zone-labeled windows: {int((zone_label != '').sum())}", file=sys.stderr)
    for count in sorted(set(all_count_label)):
        print(f"  count={count} windows: {int((count_label == count).sum())}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        amplitude=amplitude,
        zone_label=zone_label,
        count_label=count_label,
        source=source,
        window_s=args.window_s,
        stride_s=args.stride_s,
        sample_rate_hz=args.rate_hz,
    )
    print(f"\nsaved dataset to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
