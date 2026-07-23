"""Unit tests for scripts/generate_zone_dataset.py — fast, in-process (no
subprocess), matching test_train_har.py's convention of importing a
scripts/ module directly (pythonpath includes "scripts", see pyproject.toml)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("yaml")

import generate_zone_dataset  # noqa: E402

COMMON_ARGS = [
    "--seconds-per-recording",
    "1.0",
    "--recordings-per-class",
    "2",
    "--rate-hz",
    "100",
    "--window-s",
    "0.5",
    "--stride-s",
    "0.5",
    "--hampel-window",
    "3",
    "--savgol-window",
    "5",
    "--savgol-polyorder",
    "2",
]


def test_generates_expected_classes(tmp_path):
    output = tmp_path / "zone_demo.npz"
    generate_zone_dataset.main([*COMMON_ARGS, "--output", str(output)])
    assert output.exists()

    with np.load(output) as npz:
        zone_label = npz["zone_label"]
        count_label = npz["count_label"]
        source = npz["source"]
        amplitude = npz["amplitude"]

    assert len(zone_label) == len(count_label) == len(source) == len(amplitude)
    assert len(amplitude) > 0

    # Every zone_id (A1..B3) shows up as its own zone_label with count=1.
    for zone_id in ("A1", "A2", "A3", "B1", "B2", "B3"):
        mask = zone_label == zone_id
        assert mask.any(), f"no windows labeled {zone_id!r}"
        assert (count_label[mask] == 1).all()

    # empty_room: zone_label="" and count=0.
    empty_mask = (source == "empty_room") & (zone_label == "")
    assert empty_mask.any()
    assert (count_label[empty_mask] == 0).all()

    # Trajectory scenarios: unlabeled zone, count 1 and 2 respectively.
    one_person_mask = source == "one_person_walking_path"
    assert one_person_mask.any()
    assert (zone_label[one_person_mask] == "").all()
    assert (count_label[one_person_mask] == 1).all()

    two_people_mask = source == "two_people_walking_paths"
    assert two_people_mask.any()
    assert (zone_label[two_people_mask] == "").all()
    assert (count_label[two_people_mask] == 2).all()


def test_window_shape_matches_window_s_and_subcarrier_count(tmp_path):
    output = tmp_path / "zone_demo.npz"
    generate_zone_dataset.main([*COMMON_ARGS, "--output", str(output)])
    with np.load(output) as npz:
        amplitude = npz["amplitude"]
        assert amplitude.shape[1] == 50  # 0.5s * 100Hz
        assert amplitude.shape[2] == 64  # scenarios.yaml's subcarrier_count


def test_recordings_per_class_multiplies_window_count(tmp_path):
    output_1 = tmp_path / "one.npz"
    output_2 = tmp_path / "two.npz"
    generate_zone_dataset.main(
        [
            "--seconds-per-recording",
            "1.0",
            "--recordings-per-class",
            "1",
            "--rate-hz",
            "100",
            "--window-s",
            "0.5",
            "--stride-s",
            "0.5",
            "--hampel-window",
            "3",
            "--savgol-window",
            "5",
            "--savgol-polyorder",
            "2",
            "--output",
            str(output_1),
        ]
    )
    generate_zone_dataset.main(
        [
            "--seconds-per-recording",
            "1.0",
            "--recordings-per-class",
            "3",
            "--rate-hz",
            "100",
            "--window-s",
            "0.5",
            "--stride-s",
            "0.5",
            "--hampel-window",
            "3",
            "--savgol-window",
            "5",
            "--savgol-polyorder",
            "2",
            "--output",
            str(output_2),
        ]
    )
    with np.load(output_1) as npz1, np.load(output_2) as npz2:
        n1 = len(npz1["count_label"])
        n2 = len(npz2["count_label"])
    assert n2 == 3 * n1
