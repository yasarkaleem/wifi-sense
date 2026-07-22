"""Rule-based presence detector.

Motion in a room perturbs CSI far more than sensor noise does, and that
perturbation concentrates in the dominant PCA components of a window of
(preprocessed) CSI amplitude. This scores each window by how much its top
components vary over time, calibrates an adaptive threshold from an
initial "empty room" baseline period, and emits a presence event per
window once calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.preprocess import pca_reduce


@dataclass(frozen=True)
class PresenceEvent:
    """A single presence-detection result for one preprocessed window.

    `timestamp` is microseconds since epoch — same unit/epoch as a CSI
    frame's `timestamp_us` — taken from the most recent frame in the
    window this event was computed from.
    """

    timestamp: int
    presence: bool
    motion_intensity: float  # 0.0-1.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "presence": self.presence,
            "motion_intensity": self.motion_intensity,
        }


def window_motion_score(window: np.ndarray, *, n_components: int = 5) -> float:
    """Score how much a window of CSI amplitude moved over time.

    Projects `window` onto its top-k PCA components and sums each
    component's variance across the window's frames. A static scene's
    dominant structure barely changes frame to frame, so its components
    have low variance; a moving person perturbs the dominant components
    substantially, raising this score.

    Args:
        window: shape (n_frames, n_subcarriers), preprocessed CSI amplitude.
        n_components: number of top PCA components to include (clamped to
            min(n_frames, n_subcarriers) if the window is smaller than that).

    Returns:
        A non-negative scalar motion score (sum of per-component variance).
    """
    n_frames, n_subcarriers = window.shape
    k = min(n_components, n_frames, n_subcarriers)
    if k < 1:
        return 0.0
    result = pca_reduce(window, n_components=k)
    return float(np.var(result.transformed, axis=0).sum())


class PresenceDetector:
    """Adaptive-threshold presence detector.

    Feed successive preprocessed CSI windows via `update()`. For the first
    `calibration_s` seconds (measured from each window's own timestamp,
    not wall clock), the room is assumed empty: the motion scores seen
    during that period calibrate a threshold (`n_sigmas` standard
    deviations above their mean). `update()` returns None while
    calibrating; once calibrated, every call returns a `PresenceEvent`.
    """

    def __init__(
        self,
        *,
        calibration_s: float = 5.0,
        n_sigmas: float = 6.0,
        n_components: int = 5,
        min_std: float = 1e-6,
    ) -> None:
        if calibration_s <= 0:
            raise ValueError(f"calibration_s must be positive, got {calibration_s}")
        if n_sigmas <= 0:
            raise ValueError(f"n_sigmas must be positive, got {n_sigmas}")
        self.calibration_s = calibration_s
        self.n_sigmas = n_sigmas
        self.n_components = n_components
        self.min_std = min_std

        self._calibration_scores: list[float] = []
        self._calibration_start_us: int | None = None
        self._baseline_mean: float = 0.0
        self._threshold: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @property
    def baseline_mean(self) -> float:
        return self._baseline_mean

    def update(self, window: np.ndarray, timestamp_us: int) -> PresenceEvent | None:
        """Score `window` and, once calibrated, return a `PresenceEvent`.

        Returns None while still in the calibration period.
        """
        score = window_motion_score(window, n_components=self.n_components)

        if self._calibration_start_us is None:
            self._calibration_start_us = timestamp_us

        if not self.is_calibrated:
            self._calibration_scores.append(score)
            elapsed_s = (timestamp_us - self._calibration_start_us) / 1_000_000
            if elapsed_s < self.calibration_s:
                return None
            self._finalize_calibration()

        return self._emit(score, timestamp_us)

    def _finalize_calibration(self) -> None:
        scores = np.asarray(self._calibration_scores, dtype=np.float64)
        self._baseline_mean = float(scores.mean())
        std = float(scores.std())
        self._threshold = self._baseline_mean + self.n_sigmas * max(std, self.min_std)

    def _emit(self, score: float, timestamp_us: int) -> PresenceEvent:
        assert self._threshold is not None  # calibration always runs first
        presence = score > self._threshold
        gap = max(self._threshold - self._baseline_mean, self.min_std)
        # Scaled so intensity == 0.5 exactly at the threshold crossing and
        # saturates at 1.0 twice as far above baseline as the threshold is.
        intensity = float(np.clip((score - self._baseline_mean) / (2.0 * gap), 0.0, 1.0))
        return PresenceEvent(timestamp=timestamp_us, presence=bool(presence), motion_intensity=intensity)
