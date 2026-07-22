"""Unit tests for the online rolling-window buffer."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.windowing import RollingWindower


def test_rejects_non_positive_window_size():
    with pytest.raises(ValueError):
        RollingWindower(window_size=0, stride_frames=1)


def test_rejects_non_positive_stride():
    with pytest.raises(ValueError):
        RollingWindower(window_size=5, stride_frames=0)


def test_no_window_until_buffer_is_full():
    windower = RollingWindower(window_size=5, stride_frames=2)
    for i in range(4):
        assert windower.add(np.array([float(i)]), timestamp_us=i) is None


def test_first_window_emitted_at_exact_window_size():
    windower = RollingWindower(window_size=5, stride_frames=2)
    for i in range(4):
        windower.add(np.array([float(i)]), timestamp_us=i)
    result = windower.add(np.array([4.0]), timestamp_us=4)

    assert result is not None
    window, timestamp_us = result
    assert window.shape == (5, 1)
    assert np.array_equal(window[:, 0], [0, 1, 2, 3, 4])
    assert timestamp_us == 4


def test_subsequent_windows_emitted_at_stride_and_slide():
    windower = RollingWindower(window_size=5, stride_frames=2)
    results = [windower.add(np.array([float(i)]), timestamp_us=i) for i in range(9)]

    ready_indices = [i for i, r in enumerate(results) if r is not None]
    # first window ready at i=4 (index 4, 5th frame), then every 2 frames after
    assert ready_indices == [4, 6, 8]

    window_at_6, ts_at_6 = results[6]
    assert np.array_equal(window_at_6[:, 0], [2, 3, 4, 5, 6])
    assert ts_at_6 == 6

    window_at_8, ts_at_8 = results[8]
    assert np.array_equal(window_at_8[:, 0], [4, 5, 6, 7, 8])
    assert ts_at_8 == 8


def test_preserves_subcarrier_dimension():
    windower = RollingWindower(window_size=3, stride_frames=1)
    for i in range(3):
        result = windower.add(np.array([i, i * 10, i * 100], dtype=np.float64), timestamp_us=i)

    assert result is not None
    window, _ts = result
    assert window.shape == (3, 3)
    assert np.array_equal(window[-1], [2, 20, 200])
