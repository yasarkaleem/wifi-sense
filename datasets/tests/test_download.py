"""Tests for download.py's UT-HAR converter, using synthetic raw CSV files
that match the real dataset's verified column layout exactly:
input_*.csv: [timestamp, 90 amplitude columns, 90 phase columns]
annotation_*.csv: one activity-label string per row, row-aligned.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

import download


def _write_session(tmp_path, name: str, n_frames: int, activity: str, *, start_timestamp_s: float = 1000.0):
    """Writes a synthetic input_<name>.csv / annotation_<name>.csv pair
    matching UT-HAR's real raw format, with deterministic content so tests
    can assert on exact values."""
    rng = np.random.default_rng(hash(name) % (2**32))
    amplitude = rng.uniform(0, 50, size=(n_frames, 90))
    phase = rng.uniform(-np.pi, np.pi, size=(n_frames, 90))
    timestamps = start_timestamp_s + np.arange(n_frames) * 0.001  # 1kHz-ish capture

    input_path = tmp_path / f"input_{name}.csv"
    rows = np.column_stack([timestamps, amplitude, phase])
    np.savetxt(input_path, rows, delimiter=",")

    annotation_path = tmp_path / f"annotation_{name}.csv"
    with open(annotation_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _ in range(n_frames):
            writer.writerow([activity])

    return input_path, annotation_path, amplitude, phase, timestamps


def test_convert_ut_har_session_matches_source_values(tmp_path):
    input_path, annotation_path, amplitude, phase, timestamps = _write_session(tmp_path, "bed_1", 50, "bed")

    session = download.convert_ut_har_session(input_path, annotation_path)

    assert session.session_id == "bed_1"
    assert session.subcarrier_count == 90
    assert session.amplitude.shape == (50, 90)
    assert session.phase.shape == (50, 90)
    assert np.allclose(session.amplitude, amplitude, atol=1e-4)
    assert np.allclose(session.phase, phase, atol=1e-4)
    assert np.array_equal(session.timestamp_us, np.round(timestamps * 1_000_000).astype(np.int64))
    assert set(session.label) == {"bed"}
    assert session.source_mac == download.UT_HAR_SOURCE_MAC


def test_convert_ut_har_writes_one_npz_per_session(tmp_path):
    _write_session(tmp_path, "bed_1", 20, "bed")
    _write_session(tmp_path, "walk_1", 20, "walk")
    out_dir = tmp_path / "out"

    written = download.convert_ut_har(tmp_path, out_dir)

    assert len(written) == 2
    assert {p.name for p in written} == {"bed_1.npz", "walk_1.npz"}
    for p in written:
        assert p.exists()


def test_converted_npz_roundtrips_correctly(tmp_path):
    input_path, annotation_path, amplitude, phase, _ = _write_session(tmp_path, "fall_2", 30, "fall")
    out_dir = tmp_path / "out"
    download.convert_ut_har(tmp_path, out_dir)

    with np.load(out_dir / "fall_2.npz", allow_pickle=False) as npz:
        assert str(npz["session_id"]) == "fall_2"
        assert int(npz["subcarrier_count"]) == 90
        assert np.allclose(npz["amplitude"], amplitude, atol=1e-4)
        assert np.allclose(npz["phase"], phase, atol=1e-4)
        assert npz["rssi"].shape == (30,)
        assert npz["channel"].shape == (30,)
        assert set(npz["label"]) == {"fall"}


def test_frame_fields_satisfy_canonical_schema_types(tmp_path):
    """Spot-check one converted frame against docs/csi-frame-schema.md's
    field types, without importing another service's schema module (see
    ../../CLAUDE.md's service-isolation rule)."""
    _write_session(tmp_path, "walk_3", 10, "walk")
    out_dir = tmp_path / "out"
    download.convert_ut_har(tmp_path, out_dir)

    with np.load(out_dir / "walk_3.npz", allow_pickle=False) as npz:
        i = 0
        frame = {
            "schema_version": download.SCHEMA_VERSION,
            "timestamp_us": int(npz["timestamp_us"][i]),
            "source_mac": str(npz["source_mac"]),
            "rssi": int(npz["rssi"][i]),
            "channel": int(npz["channel"][i]),
            "subcarrier_count": int(npz["subcarrier_count"]),
            "amplitude": npz["amplitude"][i].tolist(),
            "phase": npz["phase"][i].tolist(),
            "sequence_number": i,
        }

    assert isinstance(frame["schema_version"], int)
    assert isinstance(frame["timestamp_us"], int)
    assert isinstance(frame["source_mac"], str)
    assert isinstance(frame["rssi"], int)
    assert isinstance(frame["channel"], int)
    assert isinstance(frame["subcarrier_count"], int) and frame["subcarrier_count"] > 0
    assert len(frame["amplitude"]) == frame["subcarrier_count"]
    assert len(frame["phase"]) == frame["subcarrier_count"]
    assert all(isinstance(v, float) for v in frame["amplitude"])
    assert all(isinstance(v, float) for v in frame["phase"])


def test_mismatched_lengths_are_truncated_not_crashed(tmp_path):
    input_path, annotation_path, _, _, _ = _write_session(tmp_path, "run_1", 40, "run")
    # Truncate the annotation file to simulate a length mismatch.
    lines = annotation_path.read_text().splitlines()[:25]
    annotation_path.write_text("\n".join(lines) + "\n")

    session = download.convert_ut_har_session(input_path, annotation_path)

    assert len(session.label) == 25
    assert session.amplitude.shape[0] == 25


def test_missing_annotation_file_is_skipped_not_crashed(tmp_path):
    _write_session(tmp_path, "bed_1", 10, "bed")
    # Input with no matching annotation file.
    orphan = tmp_path / "input_orphan_1.csv"
    np.savetxt(orphan, np.zeros((5, 181)), delimiter=",")

    written = download.convert_ut_har(tmp_path, tmp_path / "out")

    assert len(written) == 1
    assert written[0].name == "bed_1.npz"


def test_wrong_column_count_raises_value_error(tmp_path):
    bad_input = tmp_path / "input_bad_1.csv"
    np.savetxt(bad_input, np.zeros((10, 100)), delimiter=",")  # not 181 columns

    with pytest.raises(ValueError, match="181 columns"):
        download._parse_ut_har_input_csv(bad_input)


def test_no_session_pairs_found_raises_clear_error(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no input_.*annotation_"):
        download.convert_ut_har(empty_dir, tmp_path / "out")


def test_widar3_stub_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError, match="Widar 3.0"):
        download.convert_widar3(tmp_path, tmp_path / "out")


def test_cli_list_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        download.main(["--list"])
    assert exc_info.value.code == 0


def test_cli_requires_source(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        download.main(["ut-har"])
    assert exc_info.value.code != 0


def test_cli_rejects_nonexistent_source(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        download.main(["ut-har", "--source", str(tmp_path / "does-not-exist")])
    assert exc_info.value.code != 0


def test_cli_end_to_end(tmp_path):
    _write_session(tmp_path, "standup_1", 15, "standup")
    out_dir = tmp_path / "out"

    download.main(["ut-har", "--source", str(tmp_path), "--out", str(out_dir)])

    assert (out_dir / "standup_1.npz").exists()
