"""Unit tests for pipeline.room's room.yaml loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.room import DEFAULT_ROOM_CONFIG_PATH, _find_default_room_config, load_room_config


def _write_room(tmp_path, content: str):
    path = tmp_path / "room.yaml"
    path.write_text(content)
    return path


def test_default_room_config_path_points_at_the_real_repo_root_room_yaml():
    assert DEFAULT_ROOM_CONFIG_PATH.name == "room.yaml"
    assert DEFAULT_ROOM_CONFIG_PATH.exists()


def test_find_default_room_config_does_not_crash_with_shallow_ancestry(tmp_path, monkeypatch):
    """Regression test: a fixed parents[N] index raised IndexError when
    this module is imported from a shallower directory tree than the full
    monorepo checkout — e.g. the Docker image, which only copies src/ (see
    Dockerfile), giving this file 3 ancestors instead of 4. Simulate that
    by pointing __file__'s resolution at a shallow tmp_path tree with no
    room.yaml anywhere above it."""
    shallow = tmp_path / "app" / "src" / "pipeline" / "room.py"
    shallow.parent.mkdir(parents=True)
    shallow.write_text("# stub")

    monkeypatch.setattr("pipeline.room.__file__", str(shallow))
    result = _find_default_room_config()  # must not raise

    assert isinstance(result, Path)
    assert result.name == "room.yaml"


def test_default_room_config_is_a_2x3_grid():
    room = load_room_config(DEFAULT_ROOM_CONFIG_PATH)
    assert room.rows == 2
    assert room.columns == 3
    assert room.zone_ids == ("A1", "A2", "A3", "B1", "B2", "B3")


def test_default_room_config_has_zone_labels():
    room = load_room_config(DEFAULT_ROOM_CONFIG_PATH)
    assert room.zone_labels["A1"] != ""
    assert set(room.zone_labels) == set(room.zone_ids)


def test_custom_grid_shape(tmp_path):
    path = _write_room(
        tmp_path,
        """
        grid:
          rows: 3
          columns: 2
        """,
    )
    room = load_room_config(path)
    assert room.rows == 3
    assert room.columns == 2
    assert room.zone_ids == ("A1", "A2", "B1", "B2", "C1", "C2")


def test_single_zone_grid(tmp_path):
    path = _write_room(tmp_path, "grid:\n  rows: 1\n  columns: 1\n")
    room = load_room_config(path)
    assert room.zone_ids == ("A1",)


def test_missing_zone_label_defaults_to_empty_string(tmp_path):
    path = _write_room(tmp_path, "grid:\n  rows: 1\n  columns: 2\n")
    room = load_room_config(path)
    assert room.zone_labels == {"A1": "", "A2": ""}


def test_missing_grid_key_raises(tmp_path):
    path = _write_room(tmp_path, "zones: {}\n")
    with pytest.raises(ValueError, match="grid"):
        load_room_config(path)


def test_zero_rows_raises(tmp_path):
    path = _write_room(tmp_path, "grid:\n  rows: 0\n  columns: 3\n")
    with pytest.raises(ValueError):
        load_room_config(path)


def test_zero_columns_raises(tmp_path):
    path = _write_room(tmp_path, "grid:\n  rows: 2\n  columns: 0\n")
    with pytest.raises(ValueError):
        load_room_config(path)


def test_too_many_rows_raises(tmp_path):
    path = _write_room(tmp_path, "grid:\n  rows: 27\n  columns: 1\n")
    with pytest.raises(ValueError):
        load_room_config(path)
