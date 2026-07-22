"""Command-line interface for the replay service.

Usage:
    python -m replay --scenario one_person_walking --rate 100 --target localhost:5566
    python -m replay --dataset ut-har --file datasets/ut-har/bed_1.npz --target localhost:5566

Defaults for every flag can also be set via environment variables
(REPLAY_SCENARIO, REPLAY_RATE, REPLAY_TARGET, REPLAY_SCENARIOS_FILE,
REPLAY_DATASET, REPLAY_FILE), which is how docker-compose.yml configures
this service.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from replay.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios
from replay.stream import stream_dataset, stream_scenario

DEFAULT_SCENARIO = os.environ.get("REPLAY_SCENARIO", "empty_room")
DEFAULT_TARGET = os.environ.get("REPLAY_TARGET", "localhost:5566")
DEFAULT_SCENARIOS_FILE = os.environ.get("REPLAY_SCENARIOS_FILE", str(DEFAULT_SCENARIOS_PATH))
DEFAULT_DATASET = os.environ.get("REPLAY_DATASET") or None
DEFAULT_FILE = os.environ.get("REPLAY_FILE") or None

# --rate defaults to "unset" (None): synthetic mode falls back to
# SYNTHETIC_DEFAULT_RATE_HZ, dataset mode falls back to the recording's own
# original inter-frame timing — see main()'s branching below.
_rate_env = os.environ.get("REPLAY_RATE")
DEFAULT_RATE_HZ = float(_rate_env) if _rate_env else None
SYNTHETIC_DEFAULT_RATE_HZ = 100.0

KNOWN_DATASETS = ("ut-har", "widar3")


def parse_target(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"invalid target {value!r}, expected HOST:PORT")
    return host, int(port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay",
        description="Stream CSI frames over UDP to emulate a live ESP32 — either "
        "synthetic (--scenario) or a converted real dataset recording (--dataset/--file).",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"scenario name from --scenarios-file (default: {DEFAULT_SCENARIO}); "
        "ignored if --dataset is given",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_HZ,
        help="frames per second; synthetic mode defaults to "
        f"{SYNTHETIC_DEFAULT_RATE_HZ} if unset, dataset mode defaults to the "
        f"recording's own original timing if unset (default: {DEFAULT_RATE_HZ})",
    )
    parser.add_argument(
        "--target",
        type=parse_target,
        default=parse_target(DEFAULT_TARGET),
        metavar="HOST:PORT",
        help=f"UDP destination (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--scenarios-file",
        default=DEFAULT_SCENARIOS_FILE,
        help="path to the scenarios YAML file (default: packaged scenarios.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible synthetic CSI (default: unseeded); ignored if --dataset is given",
    )
    parser.add_argument(
        "--dataset",
        choices=KNOWN_DATASETS,
        default=DEFAULT_DATASET,
        help="stream a converted real dataset recording instead of a synthetic scenario "
        f"(requires --file; produced by datasets/download.py) (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(DEFAULT_FILE) if DEFAULT_FILE else None,
        help="path to a .npz recording produced by datasets/download.py (required with --dataset)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="restart the recording from the beginning when it ends (dataset mode only)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.rate is not None and args.rate <= 0:
        parser.error("--rate must be positive")

    host, port = args.target

    if args.dataset:
        if not args.file:
            parser.error("--file is required when --dataset is given")
        if not args.file.exists():
            parser.error(f"--file {args.file} does not exist")

        from replay.dataset import load_recording

        recording = load_recording(args.file)
        pacing = f"{args.rate}Hz" if args.rate is not None else "original timing"
        print(
            f"replay: streaming dataset={args.dataset!r} file={args.file} "
            f"session={recording.session_id!r} ({recording.n_frames} frames, {pacing}) "
            f"target={host}:{port}",
            file=sys.stderr,
        )
        stream_dataset(recording=recording, host=host, port=port, rate_hz=args.rate, loop=args.loop)
        return

    scenarios = load_scenarios(args.scenarios_file)
    if args.scenario not in scenarios:
        available = ", ".join(sorted(scenarios))
        parser.error(f"unknown scenario {args.scenario!r}; available: {available}")

    rate_hz = args.rate if args.rate is not None else SYNTHETIC_DEFAULT_RATE_HZ
    print(
        f"replay: streaming scenario={args.scenario!r} rate={rate_hz}Hz target={host}:{port}",
        file=sys.stderr,
    )
    stream_scenario(
        scenario=scenarios[args.scenario],
        rate_hz=rate_hz,
        host=host,
        port=port,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
