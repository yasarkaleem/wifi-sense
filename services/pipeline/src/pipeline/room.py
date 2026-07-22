"""Loads room.yaml — the physical room's zone grid definition.

Shared by pipeline/calibrate.py (validates --zone, knows every zone_id to
fit a ZoneLocalizer over) — pipeline/models/localizer.py itself doesn't
need this module, since a fitted/saved ZoneLocalizer carries its own
zone_ids. Only needed for the "localize" extra
(`pip install -e ".[dev,localize]"`), so PyYAML lives there, not in
pipeline's core dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

def _find_default_room_config() -> Path:
    """Walk up from this file looking for room.yaml (the monorepo root's
    canonical location). A fixed parent-index (e.g. parents[4]) would
    raise IndexError in contexts that don't mirror the full checkout depth
    above services/pipeline/src/ — e.g. the Docker image, which only
    copies src/, giving this file just 3 ancestors instead of the 4 a full
    checkout has.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "room.yaml"
        if candidate.exists():
            return candidate
    # No room.yaml found anywhere above — fall back to where a full
    # monorepo checkout would put it, so error messages point somewhere
    # sensible instead of crashing here.
    fallback_index = min(4, len(here.parents) - 1)
    return here.parents[fallback_index] / "room.yaml"


DEFAULT_ROOM_CONFIG_PATH: Path = _find_default_room_config()


@dataclass(frozen=True)
class RoomConfig:
    rows: int
    columns: int
    zone_ids: tuple[str, ...]  # e.g. ("A1", "A2", "A3", "B1", "B2", "B3"), row-major
    zone_labels: dict[str, str]  # zone_id -> human-readable description (may be "")


def _zone_id(row_index: int, column_index: int) -> str:
    """0-indexed (row, column) -> "A1"-style zone id (row letter, 1-indexed column)."""
    return f"{chr(ord('A') + row_index)}{column_index + 1}"


def load_room_config(path: str | Path = DEFAULT_ROOM_CONFIG_PATH) -> RoomConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "grid" not in raw:
        raise ValueError(f"{path}: expected a top-level 'grid' key with 'rows'/'columns'")

    grid = raw["grid"]
    rows = int(grid["rows"])
    columns = int(grid["columns"])
    if rows < 1 or columns < 1:
        raise ValueError(f"{path}: grid.rows and grid.columns must both be >= 1, got {rows}x{columns}")
    if rows > 26:
        raise ValueError(f"{path}: grid.rows must be <= 26 (single-letter row IDs), got {rows}")

    zone_ids = tuple(_zone_id(r, c) for r in range(rows) for c in range(columns))

    raw_zones = raw.get("zones") or {}
    zone_labels = {zone_id: (raw_zones.get(zone_id) or {}).get("label", "") for zone_id in zone_ids}

    return RoomConfig(rows=rows, columns=columns, zone_ids=zone_ids, zone_labels=zone_labels)
