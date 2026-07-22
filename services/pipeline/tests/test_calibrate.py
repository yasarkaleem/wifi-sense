"""Unit tests for pipeline.calibrate's CLI parsing and validation.

Only covers paths that fail before any network I/O (argument validation,
zone-name checking against room.yaml) — the full collect-and-fit workflow
is covered by tests/test_integration_localization.py."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from pipeline.calibrate import build_parser, main  # noqa: E402


def test_defaults():
    args = build_parser().parse_args(["--zone", "A1"])
    assert args.zone == "A1"
    assert args.seconds == 60.0
    assert args.sub_host == "localhost"
    assert args.sub_port == 5567
    assert args.window_s == 2.0
    assert args.stride_s == 0.5


def test_zone_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_zone_errors_before_any_network_io():
    with pytest.raises(SystemExit):
        main(["--zone", "Z9", "--seconds", "1"])


def test_non_positive_seconds_errors():
    with pytest.raises(SystemExit):
        main(["--zone", "A1", "--seconds", "0"])


def test_non_positive_window_s_errors():
    with pytest.raises(SystemExit):
        main(["--zone", "A1", "--seconds", "1", "--window-s", "0"])


def test_non_positive_stride_s_errors():
    with pytest.raises(SystemExit):
        main(["--zone", "A1", "--seconds", "1", "--stride-s", "-1"])


def test_custom_room_config_is_respected(tmp_path):
    room_path = tmp_path / "room.yaml"
    room_path.write_text("grid:\n  rows: 1\n  columns: 2\n")

    # "A1" is valid for this custom 1x2 room, but should still fail fast
    # (no ingest running) rather than hang — SystemExit from no-data path.
    with pytest.raises(SystemExit):
        main(["--zone", "A1", "--seconds", "0.2", "--room-config", str(room_path), "--sub-port", "1"])


def test_zone_invalid_for_custom_room_config(tmp_path):
    room_path = tmp_path / "room.yaml"
    room_path.write_text("grid:\n  rows: 1\n  columns: 2\n")

    with pytest.raises(SystemExit):
        main(["--zone", "B1", "--seconds", "1", "--room-config", str(room_path)])
