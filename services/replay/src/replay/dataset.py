"""Loads a converted CSI recording (produced by datasets/download.py) for
replay's dataset streaming mode (`python -m replay --dataset ... --file ...`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetRecording:
    """One recording session, as saved by datasets/download.py's
    ConvertedSession.save().

    `label` is a training-only convenience field (per-frame activity
    annotation, if the source dataset has one) — it is NOT part of
    ../../../docs/csi-frame-schema.md and is never sent over the wire; see
    replay.stream.stream_dataset.
    """

    session_id: str
    timestamp_us: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    rssi: np.ndarray
    channel: np.ndarray
    source_mac: str
    subcarrier_count: int
    label: np.ndarray

    @property
    def n_frames(self) -> int:
        return len(self.timestamp_us)


def load_recording(path: str | Path) -> DatasetRecording:
    with np.load(path, allow_pickle=False) as npz:
        return DatasetRecording(
            session_id=str(npz["session_id"]),
            timestamp_us=npz["timestamp_us"],
            amplitude=npz["amplitude"],
            phase=npz["phase"],
            rssi=npz["rssi"],
            channel=npz["channel"],
            source_mac=str(npz["source_mac"]),
            subcarrier_count=int(npz["subcarrier_count"]),
            label=npz["label"],
        )
