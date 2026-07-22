"""JSON Schema for CSI frames, mirroring docs/csi-frame-schema.md exactly.

Unlike services/replay (which only validates its own synthetic output in
tests), ingest validates real, untrusted UDP input at runtime, so
`jsonschema` is a hard runtime dependency here rather than a dev-only one.
"""

from __future__ import annotations

import jsonschema

CSI_FRAME_JSON_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CSI Frame",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "timestamp_us",
        "source_mac",
        "rssi",
        "channel",
        "subcarrier_count",
        "amplitude",
        "phase",
        "sequence_number",
    ],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "timestamp_us": {"type": "integer", "minimum": 0},
        "source_mac": {
            "type": "string",
            "pattern": "^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$",
        },
        "rssi": {"type": "integer"},
        "channel": {"type": "integer", "minimum": 1},
        "subcarrier_count": {"type": "integer", "minimum": 1},
        "amplitude": {"type": "array", "items": {"type": "number"}},
        "phase": {"type": "array", "items": {"type": "number"}},
        "sequence_number": {"type": "integer", "minimum": 0},
    },
}


def validate_csi_frame(frame: dict) -> None:
    """Validate `frame` against the CSI frame schema.

    Raises `jsonschema.ValidationError` for structural/type violations, or
    `ValueError` for the amplitude/phase-length-must-equal-subcarrier_count
    rule, which plain JSON Schema can't express.
    """
    jsonschema.validate(frame, CSI_FRAME_JSON_SCHEMA)

    n = frame["subcarrier_count"]
    if len(frame["amplitude"]) != n or len(frame["phase"]) != n:
        raise ValueError(
            f"amplitude/phase length must equal subcarrier_count ({n}): "
            f"got {len(frame['amplitude'])}/{len(frame['phase'])}"
        )
