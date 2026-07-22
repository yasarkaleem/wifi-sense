"""Unit tests for pipeline.preprocess, using synthetic signals with known
noise/outliers so the filters' output can be checked against ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.preprocess import (
    hampel_filter,
    pca_reduce,
    sanitize_phase,
    savitzky_golay_smooth,
    segment_sliding_window,
)

# ---------------------------------------------------------------------------
# hampel_filter
# ---------------------------------------------------------------------------


def _clean_amplitude_signal(n_frames: int, n_subcarriers: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_frames)[:, None]
    k = np.arange(n_subcarriers)[None, :]
    return 20.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t + 0.1 * k) + rng.normal(0, 0.05, size=(n_frames, n_subcarriers))


def test_hampel_filter_recovers_clean_signal_at_outlier_positions():
    rng = np.random.default_rng(0)
    n_frames, n_subcarriers = 200, 4
    clean = _clean_amplitude_signal(n_frames, n_subcarriers, rng)

    contaminated = clean.copy()
    outlier_positions = [(20, 0), (75, 1), (140, 2), (199, 3)]
    for row, col in outlier_positions:
        contaminated[row, col] += 50.0  # gross spike, far outside normal range

    cleaned = hampel_filter(contaminated, window_size=7, n_sigmas=3.0)

    for row, col in outlier_positions:
        # A 50-unit spike should be corrected back down to within a couple
        # units of the true clean value (boundary windows have less context
        # to estimate the local median from, so allow a bit more slack there
        # than a mid-signal outlier would need).
        assert abs(cleaned[row, col] - clean[row, col]) < 2.5, (
            f"outlier at ({row},{col}) not corrected: cleaned={cleaned[row, col]}, "
            f"clean={clean[row, col]}"
        )


def test_hampel_filter_leaves_most_clean_samples_unchanged():
    rng = np.random.default_rng(1)
    clean = _clean_amplitude_signal(150, 3, rng)
    contaminated = clean.copy()
    contaminated[60, 1] += 50.0

    cleaned = hampel_filter(contaminated, window_size=7, n_sigmas=3.0)

    mask = np.ones_like(clean, dtype=bool)
    mask[60, 1] = False
    diffs = np.abs(cleaned[mask] - clean[mask])
    # A well-behaved Hampel filter only touches genuine outliers; allow a
    # handful of borderline false positives near naturally flat stretches
    # (small local MAD makes ordinary noise look "outlier-ish"), but they
    # must stay rare and small.
    n_changed = int(np.count_nonzero(diffs > 1e-6))
    assert n_changed < 5, f"too many non-outlier samples were altered: {n_changed}"
    assert np.all(diffs < 2.5)


def test_hampel_filter_corrects_outlier_at_the_very_last_frame():
    """Regression test: with naive edge-padding, an outlier sitting on the
    boundary sample gets replicated into its own window and can dominate
    the median, hiding itself from detection."""
    rng = np.random.default_rng(7)
    clean = _clean_amplitude_signal(50, 2, rng)
    contaminated = clean.copy()
    contaminated[-1, 0] += 50.0
    contaminated[0, 1] += 50.0

    cleaned = hampel_filter(contaminated, window_size=7, n_sigmas=3.0)

    assert abs(cleaned[-1, 0] - clean[-1, 0]) < 2.5
    assert abs(cleaned[0, 1] - clean[0, 1]) < 2.5


def test_hampel_filter_rejects_even_window():
    with pytest.raises(ValueError):
        hampel_filter(np.zeros((10, 2)), window_size=6)


def test_hampel_filter_rejects_too_few_frames():
    with pytest.raises(ValueError):
        hampel_filter(np.zeros((5, 2)), window_size=7)


def test_hampel_filter_empty_input():
    out = hampel_filter(np.empty((0, 5)))
    assert out.shape == (0, 5)


# ---------------------------------------------------------------------------
# savitzky_golay_smooth
# ---------------------------------------------------------------------------


def test_savgol_smoothing_reduces_noise_toward_clean_signal():
    rng = np.random.default_rng(2)
    n_frames, n_subcarriers = 300, 4
    t = np.arange(n_frames)[:, None]
    k = np.arange(n_subcarriers)[None, :]
    clean = 20.0 + 2.0 * np.sin(2 * np.pi * 0.02 * t + 0.1 * k)
    noisy = clean + rng.normal(0, 0.8, size=clean.shape)

    smoothed = savitzky_golay_smooth(noisy, window_length=15, polyorder=3)

    rmse_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_smoothed = np.sqrt(np.mean((smoothed - clean) ** 2))
    assert rmse_smoothed < rmse_noisy * 0.5
    assert smoothed.shape == clean.shape


def test_savgol_rejects_even_window_length():
    with pytest.raises(ValueError):
        savitzky_golay_smooth(np.zeros((20, 2)), window_length=10, polyorder=3)


def test_savgol_rejects_polyorder_too_high():
    with pytest.raises(ValueError):
        savitzky_golay_smooth(np.zeros((20, 2)), window_length=5, polyorder=5)


def test_savgol_rejects_too_few_frames():
    with pytest.raises(ValueError):
        savitzky_golay_smooth(np.zeros((5, 2)), window_length=11, polyorder=3)


# ---------------------------------------------------------------------------
# sanitize_phase
# ---------------------------------------------------------------------------


def _wrap(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi


def test_sanitize_phase_is_invariant_to_injected_linear_trend():
    """Two different CFO/STO-style linear trends applied to the same
    underlying (symmetric, near-zero-linear-component) phase pattern should
    sanitize to (almost) the same result."""
    n_subcarriers = 64
    k = np.arange(n_subcarriers, dtype=np.float64)
    center = (n_subcarriers - 1) / 2
    base_pattern = 0.3 * np.cos(2 * np.pi * (k - center) / n_subcarriers)  # even -> ~zero linear component

    rng = np.random.default_rng(3)
    slope1, intercept1 = rng.uniform(-0.05, 0.05), rng.uniform(-1.0, 1.0)
    slope2, intercept2 = rng.uniform(-0.05, 0.05), rng.uniform(-1.0, 1.0)

    raw1 = _wrap(base_pattern + slope1 * k + intercept1)[None, :]
    raw2 = _wrap(base_pattern + slope2 * k + intercept2)[None, :]

    sanitized1 = sanitize_phase(raw1)[0]
    sanitized2 = sanitize_phase(raw2)[0]

    assert np.allclose(sanitized1, sanitized2, atol=0.05)


def test_sanitize_phase_removes_pure_linear_ramp():
    """A raw phase that's *only* a linear ramp (no real signal) should
    sanitize to ~zero everywhere."""
    n_subcarriers = 32
    k = np.arange(n_subcarriers, dtype=np.float64)
    raw = _wrap(0.04 * k + 0.7)[None, :]

    sanitized = sanitize_phase(raw)[0]

    assert np.allclose(sanitized, 0.0, atol=1e-6)


def test_sanitize_phase_batch_matches_per_frame():
    n_frames, n_subcarriers = 10, 32
    k = np.arange(n_subcarriers, dtype=np.float64)
    rng = np.random.default_rng(4)
    base_pattern = 0.2 * np.sin(2 * np.pi * k / n_subcarriers)

    raw = np.stack(
        [
            _wrap(base_pattern + rng.uniform(-0.05, 0.05) * k + rng.uniform(-1, 1))
            for _ in range(n_frames)
        ]
    )

    batch_result = sanitize_phase(raw)
    per_frame_result = np.stack([sanitize_phase(raw[i][None, :])[0] for i in range(n_frames)])

    assert np.allclose(batch_result, per_frame_result)


def test_sanitize_phase_no_discontinuities_across_wrap_boundary():
    """A phase that crosses the +/-pi boundary across subcarriers should
    unwrap cleanly, leaving no large jumps in the sanitized output."""
    n_subcarriers = 64
    k = np.arange(n_subcarriers, dtype=np.float64)
    # Steep-ish ramp that wraps multiple times across the subcarrier axis.
    raw = _wrap(0.3 * k)[None, :]

    sanitized = sanitize_phase(raw)[0]
    max_jump = np.max(np.abs(np.diff(sanitized)))
    assert max_jump < 1.0  # far below the ~pi jump a broken unwrap would leave


def test_sanitize_phase_empty_input():
    out = sanitize_phase(np.empty((0, 10)))
    assert out.shape == (0, 10)


# ---------------------------------------------------------------------------
# segment_sliding_window
# ---------------------------------------------------------------------------


def test_segment_sliding_window_shape_and_content():
    n_frames, n_subcarriers = 100, 3
    x = np.arange(n_frames * n_subcarriers, dtype=np.float64).reshape(n_frames, n_subcarriers)

    windows = segment_sliding_window(x, sample_rate_hz=10.0, window_s=2.0, stride_s=0.5)
    # window_size = 20, stride = 5 -> n_windows = floor((100-20)/5) + 1 = 17
    assert windows.shape == (17, 20, n_subcarriers)
    assert np.array_equal(windows[0], x[0:20])
    assert np.array_equal(windows[1], x[5:25])
    assert np.array_equal(windows[-1], x[80:100])


def test_segment_sliding_window_defaults():
    x = np.zeros((300, 2))
    windows = segment_sliding_window(x, sample_rate_hz=100.0)
    # defaults: window_s=2.0 -> 200 samples, stride_s=0.5 -> 50 samples
    assert windows.shape[1:] == (200, 2)
    assert windows.shape[0] == (300 - 200) // 50 + 1


def test_segment_sliding_window_too_few_frames_returns_empty():
    x = np.zeros((10, 4))
    windows = segment_sliding_window(x, sample_rate_hz=10.0, window_s=2.0, stride_s=0.5)
    assert windows.shape == (0, 20, 4)


def test_segment_sliding_window_rejects_non_positive_window():
    with pytest.raises(ValueError):
        segment_sliding_window(np.zeros((50, 2)), sample_rate_hz=10.0, window_s=0.0)


def test_segment_sliding_window_rejects_non_positive_stride():
    with pytest.raises(ValueError):
        segment_sliding_window(np.zeros((50, 2)), sample_rate_hz=10.0, stride_s=0.0)


# ---------------------------------------------------------------------------
# pca_reduce
# ---------------------------------------------------------------------------


def test_pca_reduce_recovers_known_low_rank_structure():
    rng = np.random.default_rng(5)
    n_samples, n_features = 300, 20

    t = np.linspace(0, 4 * np.pi, n_samples)
    direction1 = rng.normal(size=n_features)
    direction1 /= np.linalg.norm(direction1)
    direction2 = rng.normal(size=n_features)
    direction2 -= direction2 @ direction1 * direction1  # orthogonalize
    direction2 /= np.linalg.norm(direction2)

    latent1 = np.sin(t)
    latent2 = 0.5 * np.cos(2 * t)
    x = np.outer(latent1, direction1) + np.outer(latent2, direction2)
    x += rng.normal(0, 1e-4, size=x.shape)  # tiny noise, keeps rank effectively 2

    result = pca_reduce(x, n_components=2)

    assert result.transformed.shape == (n_samples, 2)
    assert result.components.shape == (2, n_features)
    assert result.mean.shape == (n_features,)
    assert result.explained_variance_ratio.sum() > 0.999

    # Same benign macOS-Accelerate-BLAS warning as in pca_reduce itself
    # (verified no actual NaN/Inf); suppress it here too.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        reconstructed = result.transformed @ result.components + result.mean
    rmse = np.sqrt(np.mean((reconstructed - x) ** 2))
    assert rmse < 1e-2


def test_pca_reduce_rejects_too_many_components():
    x = np.zeros((10, 5))
    with pytest.raises(ValueError):
        pca_reduce(x, n_components=6)


def test_pca_reduce_rejects_non_positive_components():
    x = np.zeros((10, 5))
    with pytest.raises(ValueError):
        pca_reduce(x, n_components=0)


def test_pca_reduce_explained_variance_ratio_is_sorted_descending():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(100, 10)) * np.array([5, 4, 3, 2, 1, 1, 1, 1, 1, 1])
    result = pca_reduce(x, n_components=5)
    ratios = result.explained_variance_ratio
    assert np.all(np.diff(ratios) <= 1e-9)  # non-increasing
