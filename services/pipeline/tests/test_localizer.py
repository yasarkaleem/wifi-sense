"""Unit tests for pipeline.models.localizer: ZoneLocalizer fit/predict,
calibration sample storage, and the LocalizationEvent/ZoneProbability
output shape."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from pipeline.models.localizer import (  # noqa: E402
    LocalizationEvent,
    ZoneLocalizer,
    ZoneProbability,
    load_all_calibration_samples,
    load_calibration_samples,
    predict_from_window,
    save_calibration_samples,
)

ZONE_IDS = ("A1", "A2", "A3", "B1", "B2", "B3")


def _synthetic_features(zone_index: int, rng: np.random.Generator, n_components=5, n_freq=17, n_time=14) -> np.ndarray:
    """A per-zone-distinguishable feature tensor: a fixed offset per zone
    plus noise, so a real classifier has something learnable."""
    bias = zone_index * 2.0
    return bias + rng.normal(0, 0.3, size=(n_components, n_freq, n_time))


def _make_dataset(n_per_zone: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for zone_index in range(len(ZONE_IDS)):
        for _ in range(n_per_zone):
            X.append(_synthetic_features(zone_index, rng))
            y.append(zone_index)
    return np.stack(X), np.array(y, dtype=np.int64)


# ---------------------------------------------------------------------------
# ZoneLocalizer construction
# ---------------------------------------------------------------------------


def test_rejects_fewer_than_two_zones():
    with pytest.raises(ValueError):
        ZoneLocalizer(["A1"])


def test_accepts_two_zones():
    localizer = ZoneLocalizer(["A1", "A2"])
    assert localizer.zone_ids == ("A1", "A2")
    assert not localizer.is_fitted


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises():
    localizer = ZoneLocalizer(ZONE_IDS)
    with pytest.raises(RuntimeError):
        localizer.predict(np.zeros((5, 17, 14)), timestamp_us=0)


def test_fit_requires_at_least_two_distinct_zones():
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=10)
    only_a1 = y == 0
    with pytest.raises(ValueError, match="at least 2 zones"):
        localizer.fit(X[only_a1], y[only_a1])


def test_fit_rejects_out_of_range_labels():
    localizer = ZoneLocalizer(["A1", "A2"])
    X = np.zeros((4, 5, 17, 14))
    y = np.array([0, 1, 5, 0])  # 5 is out of range for 2 zones
    with pytest.raises(ValueError, match="outside"):
        localizer.fit(X, y)


def test_fit_predict_recovers_correct_zone():
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=30, seed=1)
    localizer.fit(X, y)
    assert localizer.is_fitted

    rng = np.random.default_rng(99)
    for zone_index, zone_id in enumerate(ZONE_IDS):
        test_features = _synthetic_features(zone_index, rng)
        event = localizer.predict(test_features, timestamp_us=12345)
        assert event.best_zone.zone_id == zone_id, f"expected {zone_id}, got {event.best_zone.zone_id}"


def test_predict_covers_every_configured_zone_even_with_partial_calibration():
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=15, seed=2)
    only_two = np.isin(y, [0, 1])  # only A1, A2 calibrated
    localizer.fit(X[only_two], y[only_two])

    event = localizer.predict(_synthetic_features(0, np.random.default_rng(3)), timestamp_us=0)

    assert len(event.zones) == len(ZONE_IDS)
    assert {z.zone_id for z in event.zones} == set(ZONE_IDS)
    uncalibrated = [z for z in event.zones if z.zone_id not in ("A1", "A2")]
    assert all(z.occupancy_probability == 0.0 for z in uncalibrated)


def test_predict_probabilities_sum_to_approximately_one():
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=20, seed=4)
    localizer.fit(X, y)

    event = localizer.predict(_synthetic_features(2, np.random.default_rng(5)), timestamp_us=0)
    total = sum(z.occupancy_probability for z in event.zones)
    assert total == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=20, seed=6)
    localizer.fit(X, y)

    path = tmp_path / "localizer.joblib"
    localizer.save(path)
    loaded = ZoneLocalizer.load(path)

    assert loaded.zone_ids == localizer.zone_ids
    assert loaded.is_fitted

    features = _synthetic_features(3, np.random.default_rng(7))
    original_event = localizer.predict(features, timestamp_us=42)
    loaded_event = loaded.predict(features, timestamp_us=42)
    assert original_event.to_dict() == loaded_event.to_dict()


# ---------------------------------------------------------------------------
# LocalizationEvent / ZoneProbability
# ---------------------------------------------------------------------------


def test_localization_event_to_dict_shape():
    zones = (ZoneProbability("A1", 0.7), ZoneProbability("A2", 0.3))
    event = LocalizationEvent(timestamp=123, zones=zones)
    d = event.to_dict()
    assert d == {
        "timestamp": 123,
        "zones": [
            {"zone_id": "A1", "occupancy_probability": 0.7},
            {"zone_id": "A2", "occupancy_probability": 0.3},
        ],
    }


def test_best_zone_returns_argmax():
    zones = (ZoneProbability("A1", 0.2), ZoneProbability("A2", 0.5), ZoneProbability("A3", 0.3))
    event = LocalizationEvent(timestamp=0, zones=zones)
    assert event.best_zone.zone_id == "A2"


# ---------------------------------------------------------------------------
# predict_from_window (raw window -> spectrogram -> predict)
# ---------------------------------------------------------------------------


def test_predict_from_window_end_to_end():
    localizer = ZoneLocalizer(ZONE_IDS)
    X, y = _make_dataset(n_per_zone=20, seed=8)
    localizer.fit(X, y)

    rng = np.random.default_rng(9)
    window = 20.0 + rng.normal(0, 0.3, size=(200, 64))  # raw preprocessed CSI amplitude window

    event = predict_from_window(localizer, window, timestamp_us=999, n_components=5, sample_rate_hz=100.0, nperseg=32)

    assert isinstance(event, LocalizationEvent)
    assert event.timestamp == 999
    assert len(event.zones) == len(ZONE_IDS)


# ---------------------------------------------------------------------------
# Calibration sample storage
# ---------------------------------------------------------------------------


def test_save_and_load_calibration_samples_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    features = rng.normal(size=(12, 5, 17, 14))
    timestamps_us = np.arange(12, dtype=np.int64) * 1000

    path = tmp_path / "A1.npz"
    save_calibration_samples(path, zone_id="A1", features=features, timestamps_us=timestamps_us)
    samples = load_calibration_samples(path)

    assert samples.zone_id == "A1"
    assert np.allclose(samples.features, features)
    assert np.array_equal(samples.timestamps_us, timestamps_us)


def test_load_all_calibration_samples_only_finds_calibrated_zones(tmp_path):
    rng = np.random.default_rng(11)
    save_calibration_samples(
        tmp_path / "A1.npz", zone_id="A1", features=rng.normal(size=(5, 5, 17, 14)), timestamps_us=np.arange(5)
    )
    save_calibration_samples(
        tmp_path / "B2.npz", zone_id="B2", features=rng.normal(size=(7, 5, 17, 14)), timestamps_us=np.arange(7)
    )

    X, y, calibrated = load_all_calibration_samples(tmp_path, ZONE_IDS)

    assert calibrated == ["A1", "B2"]
    assert len(y) == 12
    assert set(y.tolist()) == {ZONE_IDS.index("A1"), ZONE_IDS.index("B2")}
    assert X.shape == (12, 5, 17, 14)


def test_load_all_calibration_samples_empty_dir_returns_empty(tmp_path):
    X, y, calibrated = load_all_calibration_samples(tmp_path, ZONE_IDS)
    assert calibrated == []
    assert len(y) == 0
