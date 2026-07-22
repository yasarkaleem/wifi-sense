"""Tests for CLI argument parsing."""

from __future__ import annotations

import pytest

from ingest.cli import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.udp_port == 5566
    assert args.pub_port == 5567
    assert args.plot == "none"


def test_overrides():
    args = build_parser().parse_args(
        ["--udp-port", "1234", "--pub-port", "1235", "--plot", "web", "--plot-port", "9090"]
    )
    assert args.udp_port == 1234
    assert args.pub_port == 1235
    assert args.plot == "web"
    assert args.plot_port == 9090


def test_invalid_plot_choice_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--plot", "bogus"])
