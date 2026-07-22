"""Zone-level localization via fingerprinting: a gradient-boosted
classifier mapping a window's spectrogram features to a probability
distribution over room zones (see ../../../../room.yaml).

Gradient boosting (scikit-learn's HistGradientBoostingClassifier) instead
of a CNN specifically for the on-site calibration workflow
(pipeline/calibrate.py, `python -m pipeline.calibrate --zone A1 --seconds 60`):
it trains in well under a second on the modest sample counts a short
per-zone calibration run yields, with no epochs/learning-rate/early-stopping
to tune each time a technician recalibrates a zone. The trade-off: unlike
PeopleCounterCNN's global-average-pooling (spatial-size-agnostic),
flattening the spectrogram tensor for a tabular classifier means
calibration and inference MUST use identical window_s/stride_s/n_components/
nperseg/noverlap — a shape mismatch raises a clear scikit-learn error
rather than silently misbehaving, but it won't auto-adapt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.features.spectrogram import compute_spectrogram_features


@dataclass(frozen=True)
class ZoneProbability:
    zone_id: str
    occupancy_probability: float

    def to_dict(self) -> dict:
        return {"zone_id": self.zone_id, "occupancy_probability": self.occupancy_probability}


@dataclass(frozen=True)
class LocalizationEvent:
    """A single zone-localization result for one preprocessed window.

    `timestamp` is microseconds since epoch — same unit/epoch as a CSI
    frame's `timestamp_us`. `zones` covers every zone the localizer was
    constructed with (see ZoneLocalizer.zone_ids), in that order;
    probabilities sum to ~1.0 for zones with calibration data, or are all
    0.0 if none of the configured zones have been calibrated yet.
    """

    timestamp: int
    zones: tuple[ZoneProbability, ...]

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "zones": [z.to_dict() for z in self.zones]}

    @property
    def best_zone(self) -> ZoneProbability:
        return max(self.zones, key=lambda z: z.occupancy_probability)


def _flatten(features: np.ndarray) -> np.ndarray:
    """(n_components, n_freq, n_time) -> (1, n_features), or a batch
    (n_samples, n_components, n_freq, n_time) -> (n_samples, n_features)."""
    if features.ndim == 3:
        features = features[None, ...]
    return features.reshape(features.shape[0], -1)


class ZoneLocalizer:
    """Wraps a HistGradientBoostingClassifier trained on flattened
    spectrogram features, one class per zone_id."""

    def __init__(self, zone_ids: Sequence[str], *, random_state: int = 0) -> None:
        if len(zone_ids) < 2:
            raise ValueError(f"need at least 2 zones to localize between, got {len(zone_ids)}")
        self.zone_ids: tuple[str, ...] = tuple(zone_ids)
        self._model = HistGradientBoostingClassifier(random_state=random_state)
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X: (n_samples, n_components, n_freq, n_time) spectrogram
        features. y: (n_samples,) zone indices into self.zone_ids."""
        if len(y) == 0:
            raise ValueError("y is empty; no calibration samples to fit")
        unique_classes = np.unique(y)
        if unique_classes.min() < 0 or unique_classes.max() >= len(self.zone_ids):
            raise ValueError(f"y contains zone indices outside 0..{len(self.zone_ids) - 1}: {unique_classes}")
        if len(unique_classes) < 2:
            calibrated = [self.zone_ids[c] for c in unique_classes]
            raise ValueError(
                f"need calibration data for at least 2 zones to fit, got data for only "
                f"{len(unique_classes)} zone(s) ({calibrated}); calibrate at least one more zone first"
            )
        self._model.fit(_flatten(X), y)
        self._is_fitted = True

    def predict(self, features: np.ndarray, timestamp_us: int) -> LocalizationEvent:
        """features: (n_components, n_freq, n_time) — one window's
        spectrogram features (see compute_spectrogram_features / predict_from_window)."""
        if not self._is_fitted:
            raise RuntimeError("ZoneLocalizer.fit() must be called before predict()")

        probs = self._model.predict_proba(_flatten(features))[0]
        # predict_proba only covers classes seen during fit (sorted label
        # order in self._model.classes_); expand to every configured zone,
        # with 0 probability for any zone with no calibration data yet.
        full_probs = np.zeros(len(self.zone_ids))
        for cls, p in zip(self._model.classes_, probs):
            full_probs[cls] = p

        zones = tuple(
            ZoneProbability(zone_id=zone_id, occupancy_probability=float(p))
            for zone_id, p in zip(self.zone_ids, full_probs)
        )
        return LocalizationEvent(timestamp=timestamp_us, zones=zones)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "zone_ids": self.zone_ids, "is_fitted": self._is_fitted}, path)

    @classmethod
    def load(cls, path: str | Path) -> ZoneLocalizer:
        payload = joblib.load(path)
        localizer = cls(payload["zone_ids"])
        localizer._model = payload["model"]
        localizer._is_fitted = payload["is_fitted"]
        return localizer


def predict_from_window(
    localizer: ZoneLocalizer,
    window: np.ndarray,
    timestamp_us: int,
    *,
    n_components: int = 5,
    sample_rate_hz: float = 100.0,
    nperseg: int = 32,
    noverlap: int | None = None,
) -> LocalizationEvent:
    """Convenience: compute spectrogram features from a raw preprocessed
    window and run ZoneLocalizer.predict() in one call — what
    pipeline/service.py's live loop uses."""
    features = compute_spectrogram_features(
        window,
        n_components=n_components,
        sample_rate_hz=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return localizer.predict(features, timestamp_us)


# ---------------------------------------------------------------------------
# Calibration sample storage: pipeline/calibrate.py records one .npz per
# zone (features + window timestamps), overwriting on recalibration; these
# helpers load them back for ZoneLocalizer.fit().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationSamples:
    zone_id: str
    features: np.ndarray  # (n_samples, n_components, n_freq, n_time)
    timestamps_us: np.ndarray  # (n_samples,)


def save_calibration_samples(path: str | Path, *, zone_id: str, features: np.ndarray, timestamps_us: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, zone_id=zone_id, features=features, timestamps_us=timestamps_us)


def load_calibration_samples(path: str | Path) -> CalibrationSamples:
    with np.load(path, allow_pickle=False) as npz:
        return CalibrationSamples(
            zone_id=str(npz["zone_id"]),
            features=npz["features"],
            timestamps_us=npz["timestamps_us"],
        )


def load_all_calibration_samples(
    calibration_dir: str | Path, zone_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Loads every zone's calibration .npz found in `calibration_dir`
    (only for zones in `zone_ids`, in that order), returning concatenated
    (X, y) ready for ZoneLocalizer.fit(), plus which zone_ids had data."""
    calibration_dir = Path(calibration_dir)
    all_features = []
    all_labels = []
    calibrated_zones = []
    for zone_index, zone_id in enumerate(zone_ids):
        path = calibration_dir / f"{zone_id}.npz"
        if not path.exists():
            continue
        samples = load_calibration_samples(path)
        all_features.append(samples.features)
        all_labels.append(np.full(len(samples.features), zone_index, dtype=np.int64))
        calibrated_zones.append(zone_id)

    if not all_features:
        return np.empty((0,)), np.empty((0,), dtype=np.int64), []
    return np.concatenate(all_features), np.concatenate(all_labels), calibrated_zones
