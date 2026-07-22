"""Thread-safe fixed-capacity ring buffer of recent CSI frames.

Holds the recent history the debug waterfall plot renders. The UDP
receiver thread appends to it while a plot server/thread reads snapshots
concurrently, so all access goes through a lock.
"""

from __future__ import annotations

import threading
from collections import deque


class RingBuffer:
    """A ring buffer of CSI frame dicts with a fixed maximum size."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._frames: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, frame: dict) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> list[dict]:
        """Return a point-in-time copy of the buffered frames, oldest first."""
        with self._lock:
            return list(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)
