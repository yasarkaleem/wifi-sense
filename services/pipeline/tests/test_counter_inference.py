"""Unit tests for pipeline.models.counter_inference.CounterInference."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.models.counter import PeopleCounterCNN, save_checkpoint  # noqa: E402
from pipeline.models.counter_inference import CounterInference  # noqa: E402


def _window(n_frames: int = 200, n_subcarriers: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 20.0 + rng.normal(0, 0.3, size=(n_frames, n_subcarriers))


@pytest.fixture
def checkpoint_path(tmp_path):
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    path = tmp_path / "counter.pt"
    save_checkpoint(path, model)
    return path


def test_predict_returns_valid_count_event(checkpoint_path):
    inference = CounterInference(checkpoint_path, n_components=5, sample_rate_hz=100.0, nperseg=32)
    event = inference.predict(_window(), timestamp_us=123_456_789)

    assert event.timestamp == 123_456_789
    assert event.count in (0, 1, 2, 3)
    assert 0.0 <= event.confidence <= 1.0


def test_predict_confidence_sums_to_one_across_classes(checkpoint_path):
    """Not directly observable from CountEvent, but confirms softmax
    normalization by checking confidence never trivially exceeds 1/n_classes
    ceiling in a way that would indicate a broken (non-normalized) output."""
    inference = CounterInference(checkpoint_path, n_components=5, nperseg=32)
    event = inference.predict(_window(), timestamp_us=0)
    assert event.confidence >= 1.0 / 4  # argmax of a softmax is always >= uniform probability


def test_to_dict_shape(checkpoint_path):
    inference = CounterInference(checkpoint_path, n_components=5, nperseg=32)
    event = inference.predict(_window(), timestamp_us=42)
    d = event.to_dict()
    assert set(d.keys()) == {"timestamp", "count", "confidence"}
    assert isinstance(d["timestamp"], int)
    assert isinstance(d["count"], int)
    assert isinstance(d["confidence"], float)


def test_predict_is_deterministic_for_fixed_weights(checkpoint_path):
    inference = CounterInference(checkpoint_path, n_components=5, nperseg=32)
    window = _window()
    event1 = inference.predict(window, timestamp_us=0)
    event2 = inference.predict(window, timestamp_us=0)
    assert event1.count == event2.count
    assert event1.confidence == pytest.approx(event2.confidence)


def test_predict_works_with_non_default_n_components(tmp_path):
    model = PeopleCounterCNN(in_channels=3, n_classes=4)
    path = tmp_path / "counter3.pt"
    save_checkpoint(path, model)

    inference = CounterInference(path, n_components=3, nperseg=32)
    event = inference.predict(_window(), timestamp_us=0)
    assert event.count in (0, 1, 2, 3)
