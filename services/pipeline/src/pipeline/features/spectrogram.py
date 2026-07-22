"""STFT-based time-frequency features for people counting.

Turns a window of preprocessed CSI amplitude into a small multi-channel
"image": one time-frequency spectrogram per top PCA component, stacked
along a channel axis. That (channels, height, width) layout is exactly
what PyTorch's Conv2d expects, and is the input shape
pipeline.models.counter.PeopleCounterCNN is built around.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from pipeline.preprocess import pca_reduce

_NORMALIZE_EPS = 1e-8


def compute_spectrogram_features(
    window: np.ndarray,
    *,
    n_components: int = 5,
    sample_rate_hz: float = 100.0,
    nperseg: int = 32,
    noverlap: int | None = None,
) -> np.ndarray:
    """Compute a stacked per-PCA-component spectrogram for one CSI window.

    Steps: PCA-reduce the window to its top `n_components` components (each
    a (n_frames,) time series), STFT each one independently, take the
    log-magnitude, then per-component normalize to zero mean / unit std —
    so the classifier sees relative time-frequency structure rather than
    absolute CSI signal scale (which varies with transmit power, distance,
    etc.). Normalizing per-channel (rather than globally) matters here
    because PCA components have very different natural magnitudes — the
    first component captures most of the variance, so a global
    normalization would make later components look like near-constant
    noise.

    Args:
        window: shape (n_frames, n_subcarriers), preprocessed CSI amplitude
            (e.g. output of hampel_filter + savitzky_golay_smooth).
        n_components: number of top PCA components — this becomes the
            output's channel count (`PeopleCounterCNN.in_channels` must
            match it). If the window is smaller than this in either
            dimension, the PCA count is clamped and the result is
            zero-padded back up to `n_components` channels, so the output
            shape's channel count is always exactly `n_components`.
        sample_rate_hz: CSI frame rate (Hz) the window was captured at;
            passed through to `scipy.signal.stft`'s `fs`.
        nperseg: STFT segment length in samples.
        noverlap: STFT segment overlap in samples (default: scipy's own
            default of `nperseg // 2`, i.e. 50% overlap).

    Returns:
        shape (n_components, n_freq_bins, n_time_bins), float32, where
        n_freq_bins = nperseg // 2 + 1 and n_time_bins depends on
        (n_frames, nperseg, noverlap). Concretely, with all defaults above
        and a 200-frame (2s @ 100Hz) window: (5, 17, 14).
    """
    n_frames, n_subcarriers = window.shape
    if n_frames < nperseg:
        raise ValueError(f"need at least nperseg={nperseg} frames, got {n_frames}")

    k = min(n_components, n_frames, n_subcarriers)
    pca_result = pca_reduce(window, n_components=k)  # transformed: (n_frames, k)

    channels = []
    for i in range(k):
        _freqs, _times, stft_matrix = stft(
            pca_result.transformed[:, i],
            fs=sample_rate_hz,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        channels.append(np.abs(stft_matrix))

    magnitude = np.stack(channels, axis=0)  # (k, n_freq_bins, n_time_bins)
    log_magnitude = np.log1p(magnitude)

    mean = log_magnitude.mean(axis=(1, 2), keepdims=True)
    std = log_magnitude.std(axis=(1, 2), keepdims=True)
    normalized = (log_magnitude - mean) / (std + _NORMALIZE_EPS)

    if k < n_components:
        # Guarantees the channel count always equals n_components, which
        # PeopleCounterCNN's fixed in_channels relies on.
        pad = np.zeros((n_components - k, *normalized.shape[1:]), dtype=normalized.dtype)
        normalized = np.concatenate([normalized, pad], axis=0)

    return normalized.astype(np.float32)
