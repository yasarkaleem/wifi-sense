"""Unit tests for pipeline.features.spectrogram."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.features.spectrogram import compute_spectrogram_features


def _window(n_frames: int = 200, n_subcarriers: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 20.0 + rng.normal(0, 0.3, size=(n_frames, n_subcarriers))


def test_default_shape_matches_documented_example():
    features = compute_spectrogram_features(_window(200, 64), n_components=5, sample_rate_hz=100.0, nperseg=32)
    assert features.shape == (5, 17, 14)
    assert features.dtype == np.float32


def test_freq_bins_formula():
    for nperseg in (16, 32, 64):
        features = compute_spectrogram_features(_window(200, 64), n_components=3, nperseg=nperseg)
        assert features.shape[1] == nperseg // 2 + 1


def test_channel_count_always_equals_n_components():
    for n_components in (1, 3, 8):
        features = compute_spectrogram_features(_window(200, 64), n_components=n_components, nperseg=32)
        assert features.shape[0] == n_components


def test_pads_with_zero_channels_when_window_too_small_for_requested_components():
    # n_subcarriers=2 clamps PCA to k=2, but n_components=5 is requested
    window = _window(n_frames=64, n_subcarriers=2)
    features = compute_spectrogram_features(window, n_components=5, nperseg=32)
    assert features.shape[0] == 5
    assert np.array_equal(features[2:], np.zeros_like(features[2:]))


def test_raises_when_window_shorter_than_nperseg():
    with pytest.raises(ValueError):
        compute_spectrogram_features(_window(n_frames=10, n_subcarriers=64), nperseg=32)


def test_per_channel_normalization_zero_mean_unit_std():
    features = compute_spectrogram_features(_window(200, 64), n_components=5, nperseg=32)
    per_channel_mean = features.mean(axis=(1, 2))
    per_channel_std = features.std(axis=(1, 2))
    assert np.allclose(per_channel_mean, 0.0, atol=1e-5)
    assert np.allclose(per_channel_std, 1.0, atol=1e-4)


def test_output_has_no_nan_or_inf():
    features = compute_spectrogram_features(_window(200, 64), n_components=5, nperseg=32)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_time_bins_shrink_with_shorter_window():
    features_short = compute_spectrogram_features(_window(100, 64), n_components=3, nperseg=32)
    features_long = compute_spectrogram_features(_window(200, 64), n_components=3, nperseg=32)
    assert features_short.shape[2] < features_long.shape[2]


def test_different_scenarios_produce_different_features():
    """Sanity check that the feature pipeline is sensitive to a real
    disturbance, not just noise-invariant: a window with an injected
    sinusoidal disturbance should look clearly different from a calm one."""
    calm = _window(200, 64, seed=1)

    t = np.arange(200)[:, None]
    k = np.arange(64)[None, :]
    weight = np.exp(-0.5 * ((k - 32) / 18) ** 2)
    disturbance = 4.0 * weight * np.sin(2 * np.pi * 0.8 * (t / 100.0))
    moving = calm + disturbance

    calm_features = compute_spectrogram_features(calm, n_components=5, nperseg=32)
    moving_features = compute_spectrogram_features(moving, n_components=5, nperseg=32)

    assert not np.allclose(calm_features, moving_features, atol=0.5)
