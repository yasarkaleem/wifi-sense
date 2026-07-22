"""Turns a live stream of per-frame amplitude rows into fixed-size,
fixed-stride windows, for online (as opposed to batch) presence detection."""

from __future__ import annotations

from collections import deque

import numpy as np


class RollingWindower:
    """Buffers incoming (n_subcarriers,) amplitude rows and yields a
    (window_size, n_subcarriers) window every `stride_frames` new rows,
    once at least `window_size` rows have accumulated."""

    def __init__(self, *, window_size: int, stride_frames: int) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride_frames < 1:
            raise ValueError(f"stride_frames must be >= 1, got {stride_frames}")
        self.window_size = window_size
        self.stride_frames = stride_frames
        self._rows: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_last_window = 0

    def add(self, amplitude: np.ndarray, timestamp_us: int) -> tuple[np.ndarray, int] | None:
        """Add one frame's amplitude row.

        Returns (window, timestamp_us) — the window and the timestamp of
        its most recent frame — once enough new frames have arrived to
        emit the next window; otherwise None.
        """
        self._rows.append(np.asarray(amplitude, dtype=np.float64))
        self._frames_since_last_window += 1

        if len(self._rows) < self.window_size:
            return None
        if self._frames_since_last_window < self.stride_frames:
            return None

        self._frames_since_last_window = 0
        return np.stack(self._rows), timestamp_us
