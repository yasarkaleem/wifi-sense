"""CSI preprocessing: pure, testable NumPy functions.

Unless noted otherwise, per-subcarrier time series are laid out as
`(n_frames, n_subcarriers)` — rows are time (one row per CSI frame, in
capture order), columns are subcarriers. This matches how CSI frames stack:
each frame's `amplitude`/`phase` array (see ../../../docs/csi-frame-schema.md)
becomes one row.

None of these functions perform I/O or hold state — they take arrays in,
return arrays out, so they can be unit tested with synthetic signals and
composed freely by the pipeline service.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter


def _require_2d(x: np.ndarray, name: str = "x") -> None:
    if x.ndim != 2:
        raise ValueError(f"expected a 2D array (n_frames, n_subcarriers), got {name}.shape={x.shape}")


def hampel_filter(x: np.ndarray, *, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Remove per-subcarrier outliers with a Hampel filter.

    For each time step and each subcarrier independently, compares the
    sample to the median of a centered sliding window (size `window_size`,
    must be odd) of that subcarrier's own time series. Samples further than
    `n_sigmas` scaled median-absolute-deviations (MAD) from that median are
    replaced by the median; everything else passes through unchanged.

    Args:
        x: shape (n_frames, n_subcarriers).
        window_size: odd number of samples in the sliding window.
        n_sigmas: outlier threshold in MAD-scaled standard deviations.

    Returns:
        shape (n_frames, n_subcarriers), float64, outliers replaced.
    """
    _require_2d(x)
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")

    x = np.asarray(x, dtype=np.float64)
    n_frames, _n_subcarriers = x.shape
    if n_frames == 0:
        return x.copy()
    if n_frames < window_size:
        raise ValueError(f"need at least window_size={window_size} frames, got {n_frames}")

    half = window_size // 2
    # mode="reflect" mirrors interior samples without repeating the edge
    # sample itself; mode="edge" would repeat it `half` times, which lets
    # a boundary outlier dominate its own window's median.
    padded = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    # windows: (n_frames, n_subcarriers, window_size)
    windows = sliding_window_view(padded, window_size, axis=0)

    medians = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - medians[..., None]), axis=-1) * 1.4826

    deviation = np.abs(x - medians)
    # If MAD is exactly zero (a perfectly flat local window), any nonzero
    # deviation is anomalous by definition; fall back to that instead of
    # dividing by zero.
    is_outlier = np.where(mad > 0, deviation > n_sigmas * mad, deviation > 0)

    return np.where(is_outlier, medians, x)


def savitzky_golay_smooth(x: np.ndarray, *, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Smooth each subcarrier's time series with a Savitzky-Golay filter.

    Fits a local polynomial (degree `polyorder`) by least squares over a
    sliding window (`window_length` samples, must be odd) and evaluates it
    at the window center — smooths high-frequency noise while preserving
    the shape of slower disturbances better than a moving average.

    Args:
        x: shape (n_frames, n_subcarriers).
        window_length: odd number of samples in the fitting window.
        polyorder: polynomial degree fit within each window (< window_length).

    Returns:
        shape (n_frames, n_subcarriers), float64.
    """
    _require_2d(x)
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length <= polyorder:
        raise ValueError(f"window_length ({window_length}) must be > polyorder ({polyorder})")

    n_frames = x.shape[0]
    if n_frames < window_length:
        raise ValueError(f"need at least window_length={window_length} frames to smooth, got {n_frames}")

    return savgol_filter(np.asarray(x, dtype=np.float64), window_length, polyorder, axis=0)


def sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """Unwrap phase and remove the per-frame linear trend across subcarriers.

    Raw CSI phase is wrapped to (-pi, pi] and carries a roughly linear
    component across the subcarrier axis caused by carrier frequency offset
    (CFO) and sampling time offset (STO) between transmitter and receiver.
    This unwraps each frame's phase across subcarriers, then fits and
    subtracts a least-squares line (in subcarrier index) per frame, leaving
    the CFO/STO-corrected phase.

    Args:
        phase: shape (n_frames, n_subcarriers), radians, wrapped.

    Returns:
        shape (n_frames, n_subcarriers), radians, unwrapped and detrended.
    """
    _require_2d(phase, "phase")
    phase = np.asarray(phase, dtype=np.float64)
    n_frames, n_subcarriers = phase.shape
    if n_frames == 0:
        return phase.copy()

    unwrapped = np.unwrap(phase, axis=1)

    k = np.arange(n_subcarriers, dtype=np.float64)
    k_centered = k - k.mean()
    denom = np.sum(k_centered**2)
    if denom == 0:  # single-subcarrier degenerate case: no trend to fit
        return unwrapped - unwrapped.mean(axis=1, keepdims=True)

    y_mean = unwrapped.mean(axis=1, keepdims=True)
    # See the matching np.errstate note in pca_reduce: some BLAS backends
    # (e.g. macOS Accelerate) emit spurious divide/overflow warnings from
    # this matmul with no actual NaN/Inf in the result.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        slope = (unwrapped - y_mean) @ k_centered / denom  # shape (n_frames,)
    intercept = y_mean[:, 0] - slope * k.mean()
    trend = slope[:, None] * k[None, :] + intercept[:, None]

    return unwrapped - trend


def segment_sliding_window(
    x: np.ndarray, *, sample_rate_hz: float, window_s: float = 2.0, stride_s: float = 0.5
) -> np.ndarray:
    """Slice a time series into overlapping fixed-length windows.

    Args:
        x: shape (n_frames, n_subcarriers).
        sample_rate_hz: frames per second, used to convert window_s/stride_s
            into sample counts.
        window_s: window length in seconds (default 2.0).
        stride_s: hop between consecutive window starts, in seconds
            (default 0.5; smaller than window_s means windows overlap).

    Returns:
        shape (n_windows, window_size, n_subcarriers), where
        window_size = round(window_s * sample_rate_hz). Empty
        (shape (0, window_size, n_subcarriers)) if `x` has fewer frames
        than window_size.
    """
    _require_2d(x)
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

    window_size = int(round(window_s * sample_rate_hz))
    stride = int(round(stride_s * sample_rate_hz))
    if window_size < 1:
        raise ValueError(f"window_s * sample_rate_hz must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride_s * sample_rate_hz must be >= 1, got {stride}")

    n_frames, n_subcarriers = x.shape
    if n_frames < window_size:
        return np.empty((0, window_size, n_subcarriers), dtype=x.dtype)

    # sliding_window_view(x, window_size, axis=0) -> (n_windows_all, n_subcarriers, window_size)
    all_windows = sliding_window_view(x, window_size, axis=0)
    all_windows = np.moveaxis(all_windows, -1, 1)  # -> (n_windows_all, window_size, n_subcarriers)
    return all_windows[::stride].copy()


@dataclass(frozen=True)
class PCAResult:
    """Result of `pca_reduce`."""

    transformed: np.ndarray  # (n_samples, n_components)
    components: np.ndarray  # (n_components, n_features), principal axes (unit vectors)
    mean: np.ndarray  # (n_features,), subtracted from x before projecting
    explained_variance_ratio: np.ndarray  # (n_components,), fraction of total variance each component captures


def pca_reduce(x: np.ndarray, *, n_components: int = 5) -> PCAResult:
    """Reduce subcarriers to their top-k principal components.

    Mean-centers `x` across samples (frames) and projects it onto the
    directions of largest variance via SVD.

    Args:
        x: shape (n_samples, n_features) — typically (n_frames, n_subcarriers).
        n_components: number of components to keep (top-k by variance).

    Returns:
        PCAResult with `transformed` of shape (n_samples, n_components).
    """
    _require_2d(x)
    n_samples, n_features = x.shape
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed min(n_samples, n_features) "
            f"= min({n_samples}, {n_features})"
        )

    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean

    # Some BLAS backends (e.g. macOS Accelerate) emit spurious divide/overflow
    # RuntimeWarnings from internal SIMD codepaths on tiny singular values
    # even though the result contains no actual NaN/Inf; suppress just here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:n_components]
        transformed = centered @ components.T

    explained_variance = (s**2) / max(n_samples - 1, 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = (
        explained_variance[:n_components] / total_variance
        if total_variance > 0
        else np.zeros(n_components)
    )

    return PCAResult(
        transformed=transformed,
        components=components,
        mean=mean,
        explained_variance_ratio=explained_variance_ratio,
    )
