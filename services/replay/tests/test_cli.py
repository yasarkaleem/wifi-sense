"""Tests for CLI argument parsing and validation, including dataset mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from replay.cli import build_parser, main


def test_defaults():
    args = build_parser().parse_args([])
    assert args.scenario == "empty_room"
    assert args.rate is None
    assert args.dataset is None
    assert args.file is None


def test_scenario_overrides():
    args = build_parser().parse_args(["--scenario", "two_people", "--rate", "50"])
    assert args.scenario == "two_people"
    assert args.rate == 50.0


def test_dataset_flags_parse():
    args = build_parser().parse_args(["--dataset", "ut-har", "--file", "some/path.npz"])
    assert args.dataset == "ut-har"
    assert args.file == Path("some/path.npz")


def test_invalid_dataset_choice_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dataset", "bogus-dataset"])


def test_dataset_without_file_errors():
    with pytest.raises(SystemExit):
        main(["--dataset", "ut-har"])


def test_dataset_with_nonexistent_file_errors(tmp_path):
    with pytest.raises(SystemExit):
        main(["--dataset", "ut-har", "--file", str(tmp_path / "missing.npz")])


def test_negative_rate_errors():
    with pytest.raises(SystemExit):
        main(["--rate", "-1"])


def test_loop_flag_defaults_false():
    args = build_parser().parse_args([])
    assert args.loop is False
