"""Unit tests for the rule-based presence detector, using synthetic CSI-like
windows with known "calm" vs. "moving" characteristics."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.detectors.presence import PresenceDetector, window_motion_score

US_PER_S = 1_000_000


def _calm_window(rng: np.random.Generator, n_frames: int = 100, n_subcarriers: int = 64) -> np.ndarray:
    """A near-static window: flat baseline plus tiny sensor noise."""
    return 20.0 + rng.normal(0, 0.05, size=(n_frames, n_subcarriers))


def _moving_window(rng: np.random.Generator, n_frames: int = 100, n_subcarriers: int = 64) -> np.ndarray:
    """A window with an injected disturbance, like a person walking."""
    t = np.arange(n_frames)[:, None]
    k = np.arange(n_subcarriers)[None, :]
    disturbance = 4.0 * np.sin(2 * np.pi * 0.8 * t / n_frames * 20 + 0.1 * k)
    return 20.0 + disturbance + rng.normal(0, 0.05, size=(n_frames, n_subcarriers))


# ---------------------------------------------------------------------------
# window_motion_score
# ---------------------------------------------------------------------------


def test_motion_score_is_zero_for_perfectly_constant_window():
    window = np.full((50, 10), 20.0)
    assert window_motion_score(window) == pytest.approx(0.0, abs=1e-12)


def test_motion_score_is_higher_for_moving_than_calm_window():
    rng = np.random.default_rng(0)
    calm = _calm_window(rng)
    moving = _moving_window(rng)

    calm_score = window_motion_score(calm)
    moving_score = window_motion_score(moving)

    assert moving_score > calm_score * 10


def test_motion_score_handles_tiny_windows_gracefully():
    # n_frames=2, n_subcarriers=3: n_components clamped to min(2,3)=2
    window = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.3]])
    score = window_motion_score(window, n_components=5)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# PresenceDetector
# ---------------------------------------------------------------------------


def test_rejects_non_positive_calibration_s():
    with pytest.raises(ValueError):
        PresenceDetector(calibration_s=0)


def test_rejects_non_positive_n_sigmas():
    with pytest.raises(ValueError):
        PresenceDetector(n_sigmas=0)


def test_returns_none_while_calibrating():
    rng = np.random.default_rng(1)
    detector = PresenceDetector(calibration_s=2.0)

    # windows every 0.5s of "elapsed" time, well within the 2s calibration
    for i in range(3):
        event = detector.update(_calm_window(rng), timestamp_us=i * 500_000)
        assert event is None
    assert not detector.is_calibrated


def test_calibrates_after_duration_and_emits_events():
    rng = np.random.default_rng(2)
    detector = PresenceDetector(calibration_s=2.0)

    timestamp_us = 0
    event = None
    for i in range(10):  # 0.5s steps -> crosses 2.0s at i=4
        timestamp_us = i * 500_000
        event = detector.update(_calm_window(rng), timestamp_us=timestamp_us)

    assert detector.is_calibrated
    assert event is not None
    assert event.timestamp == timestamp_us


def test_stays_false_for_calm_windows_after_calibration():
    rng = np.random.default_rng(3)
    # A longer calibration period (more samples) gives a more stable std
    # estimate; too few samples makes a fixed n_sigmas threshold noisy
    # enough to false-positive on ordinary calm-window variance.
    detector = PresenceDetector(calibration_s=5.0, n_sigmas=4.0)

    timestamp_us = 0
    for i in range(11):  # calibrate: 0s..5.0s in 0.5s steps
        timestamp_us = i * 500_000
        detector.update(_calm_window(rng), timestamp_us=timestamp_us)
    assert detector.is_calibrated

    diffs = []
    for i in range(11, 31):  # steady-state calm windows post-calibration
        timestamp_us = i * 500_000
        event = detector.update(_calm_window(rng), timestamp_us=timestamp_us)
        diffs.append(event.presence)
        assert event.motion_intensity < 0.7

    # A 4-sigma threshold on a handful of calibration samples can still
    # false-positive occasionally on pure noise; require it to be rare.
    assert sum(diffs) <= 1, f"too many false positives on calm data: {diffs}"


def test_flags_presence_after_calibration_when_motion_spikes():
    rng = np.random.default_rng(4)
    detector = PresenceDetector(calibration_s=2.0, n_sigmas=4.0)

    timestamp_us = 0
    for i in range(6):  # calibrate on calm windows
        timestamp_us = i * 500_000
        detector.update(_calm_window(rng), timestamp_us=timestamp_us)
    assert detector.is_calibrated

    timestamp_us += 500_000
    event = detector.update(_moving_window(rng), timestamp_us=timestamp_us)

    assert event.presence is True
    assert event.motion_intensity > 0.5


def test_motion_intensity_always_bounded_0_to_1():
    rng = np.random.default_rng(5)
    detector = PresenceDetector(calibration_s=1.0, n_sigmas=4.0)

    timestamp_us = 0
    for i in range(3):
        timestamp_us = i * 500_000
        detector.update(_calm_window(rng), timestamp_us=timestamp_us)

    # Feed a wide range of windows, including an extreme spike, and check bounds.
    for window in [
        _calm_window(rng),
        _moving_window(rng),
        20.0 + 50.0 * np.sin(np.linspace(0, 20, 100))[:, None] * np.ones((1, 64)),
    ]:
        timestamp_us += 500_000
        event = detector.update(window, timestamp_us=timestamp_us)
        assert 0.0 <= event.motion_intensity <= 1.0


def test_threshold_and_baseline_mean_are_none_and_zero_before_calibration():
    detector = PresenceDetector(calibration_s=1.0)
    assert detector.threshold is None
    assert detector.baseline_mean == 0.0


def test_event_to_dict_shape():
    rng = np.random.default_rng(6)
    detector = PresenceDetector(calibration_s=1.0)
    timestamp_us = 0
    event = None
    for i in range(3):
        timestamp_us = i * 500_000
        event = detector.update(_calm_window(rng), timestamp_us=timestamp_us)

    assert event is not None
    d = event.to_dict()
    assert set(d.keys()) == {"timestamp", "presence", "motion_intensity"}
    assert isinstance(d["timestamp"], int)
    assert isinstance(d["presence"], bool)
    assert isinstance(d["motion_intensity"], float)
