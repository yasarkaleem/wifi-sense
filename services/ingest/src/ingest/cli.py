"""Command-line interface for the ingest service.

Usage:
    python -m ingest --udp-port 5566 --pub-port 5567 --plot web

Defaults for every flag can also be set via environment variables
(INGEST_UDP_HOST, INGEST_UDP_PORT, INGEST_PUB_HOST, INGEST_PUB_PORT,
INGEST_BUFFER_SIZE, INGEST_PLOT, INGEST_PLOT_HOST, INGEST_PLOT_PORT,
INGEST_PLOT_REFRESH_HZ), which is how docker-compose.yml configures this
service.
"""

from __future__ import annotations

import argparse
import os

DEFAULT_UDP_HOST = os.environ.get("INGEST_UDP_HOST", "0.0.0.0")
DEFAULT_UDP_PORT = int(os.environ.get("INGEST_UDP_PORT", "5566"))
DEFAULT_PUB_HOST = os.environ.get("INGEST_PUB_HOST", "0.0.0.0")
DEFAULT_PUB_PORT = int(os.environ.get("INGEST_PUB_PORT", "5567"))
DEFAULT_BUFFER_SIZE = int(os.environ.get("INGEST_BUFFER_SIZE", "200"))
DEFAULT_PLOT = os.environ.get("INGEST_PLOT", "none")
DEFAULT_PLOT_HOST = os.environ.get("INGEST_PLOT_HOST", "0.0.0.0")
DEFAULT_PLOT_PORT = int(os.environ.get("INGEST_PLOT_PORT", "8090"))
DEFAULT_PLOT_REFRESH_HZ = float(os.environ.get("INGEST_PLOT_REFRESH_HZ", "10"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="Receive CSI frames over UDP, validate/buffer them, and "
        "publish them over ZeroMQ (with an optional debug live waterfall plot).",
    )
    parser.add_argument(
        "--udp-host",
        default=DEFAULT_UDP_HOST,
        help=f"UDP bind host (default: {DEFAULT_UDP_HOST})",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=DEFAULT_UDP_PORT,
        help=f"UDP bind port (default: {DEFAULT_UDP_PORT})",
    )
    parser.add_argument(
        "--pub-host",
        default=DEFAULT_PUB_HOST,
        help=f"ZeroMQ PUB bind host (default: {DEFAULT_PUB_HOST})",
    )
    parser.add_argument(
        "--pub-port",
        type=int,
        default=DEFAULT_PUB_PORT,
        help=f"ZeroMQ PUB bind port, 0 to disable (default: {DEFAULT_PUB_PORT})",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help=f"ring buffer capacity in frames (default: {DEFAULT_BUFFER_SIZE})",
    )
    parser.add_argument(
        "--plot",
        choices=["none", "web", "matplotlib"],
        default=DEFAULT_PLOT,
        help=f"debug live amplitude waterfall (default: {DEFAULT_PLOT})",
    )
    parser.add_argument(
        "--plot-host",
        default=DEFAULT_PLOT_HOST,
        help=f"web plot bind host (default: {DEFAULT_PLOT_HOST})",
    )
    parser.add_argument(
        "--plot-port",
        type=int,
        default=DEFAULT_PLOT_PORT,
        help=f"web plot bind port (default: {DEFAULT_PLOT_PORT})",
    )
    parser.add_argument(
        "--plot-refresh-hz",
        type=float,
        default=DEFAULT_PLOT_REFRESH_HZ,
        help=f"plot refresh rate (default: {DEFAULT_PLOT_REFRESH_HZ})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.buffer_size <= 0:
        parser.error("--buffer-size must be positive")

    from ingest.service import run_service  # local import keeps --help fast

    run_service(args)


if __name__ == "__main__":
    main()
