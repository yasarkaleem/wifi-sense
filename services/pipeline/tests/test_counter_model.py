"""Unit tests for pipeline.models.counter: architecture shape/size sanity
and the <20ms CPU real-time inference budget."""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pipeline.models.counter import (  # noqa: E402
    CLASS_COUNTS,
    CLASS_LABELS,
    N_CLASSES,
    PeopleCounterCNN,
    load_checkpoint,
    save_checkpoint,
)


def test_class_constants_consistent():
    assert len(CLASS_COUNTS) == N_CLASSES
    assert len(CLASS_LABELS) == N_CLASSES
    assert CLASS_COUNTS == (0, 1, 2, 3)
    assert CLASS_LABELS[-1] == "3+"


def test_forward_pass_output_shape():
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    x = torch.randn(8, 5, 17, 14)  # batch of 8, matching compute_spectrogram_features' default shape
    logits = model(x)
    assert logits.shape == (8, 4)


def test_forward_pass_handles_batch_size_one():
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    x = torch.randn(1, 5, 17, 14)
    logits = model(x)
    assert logits.shape == (1, 4)


def test_global_average_pool_makes_model_agnostic_to_spatial_size():
    """Different nperseg/window_s settings change n_freq_bins/n_time_bins;
    the model must still run without needing retraining for that."""
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    for shape in [(5, 9, 8), (5, 17, 14), (5, 33, 30)]:
        x = torch.randn(2, *shape)
        logits = model(x)
        assert logits.shape == (2, 4)


def test_model_is_small():
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 50_000, f"model has {n_params} params, expected a small CPU-real-time-friendly model"


def test_cpu_inference_latency_under_20ms():
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    model.eval()
    x = torch.randn(1, 5, 17, 14)

    with torch.no_grad():
        for _ in range(5):  # warm up (lazy kernel init, etc.)
            model(x)

        n_runs = 50
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        elapsed_s = time.perf_counter() - start

    mean_latency_ms = (elapsed_s / n_runs) * 1000
    assert mean_latency_ms < 20.0, f"mean inference latency {mean_latency_ms:.2f}ms exceeds the 20ms budget"


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    checkpoint_path = tmp_path / "counter.pt"
    save_checkpoint(checkpoint_path, model)

    loaded = load_checkpoint(checkpoint_path)
    assert loaded.in_channels == 5
    assert loaded.n_classes == 4
    assert loaded.training is False  # eval mode

    x = torch.randn(1, 5, 17, 14)
    with torch.no_grad():
        original_logits = model.eval()(x)
        loaded_logits = loaded(x)
    assert torch.allclose(original_logits, loaded_logits)


def test_save_checkpoint_accepts_custom_class_labels(tmp_path):
    """A checkpoint for a differently-trained model (e.g. scripts/train_har.py's
    7-class activity model) must store ITS OWN class labels, not people-counting's."""
    model = PeopleCounterCNN(in_channels=5, n_classes=7)
    activity_labels = ("bed", "fall", "walk", "pickup", "run", "sitdown", "standup")
    checkpoint_path = tmp_path / "har.pt"

    save_checkpoint(checkpoint_path, model, class_labels=activity_labels)

    raw = torch.load(checkpoint_path, weights_only=False)
    assert raw["class_labels"] == list(activity_labels)


def test_save_checkpoint_rejects_mismatched_class_labels(tmp_path):
    model = PeopleCounterCNN(in_channels=5, n_classes=4)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.pt", model, class_labels=("only", "two"))


def test_load_checkpoint_reconstructs_correct_in_channels(tmp_path):
    model = PeopleCounterCNN(in_channels=3, n_classes=4)
    checkpoint_path = tmp_path / "counter.pt"
    save_checkpoint(checkpoint_path, model)

    loaded = load_checkpoint(checkpoint_path)
    x = torch.randn(1, 3, 10, 10)
    with torch.no_grad():
        loaded(x)  # should not raise a channel-mismatch error
