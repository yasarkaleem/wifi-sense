"""Loads a trained PeopleCounterCNN checkpoint and wraps it for live
inference: a preprocessed CSI window in, a {timestamp, count, confidence}
event out. This is what pipeline/service.py plugs into the running
pipeline; scripts/train_counter.py produces the checkpoints it loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pipeline.features.spectrogram import compute_spectrogram_features
from pipeline.models.counter import CLASS_COUNTS, load_checkpoint


@dataclass(frozen=True)
class CountEvent:
    """A single people-count prediction for one preprocessed window.

    `timestamp` is microseconds since epoch — same unit/epoch as a CSI
    frame's `timestamp_us`. `count` of 3 means "3 or more" (see
    pipeline.models.counter.CLASS_COUNTS). `confidence` is the predicted
    class's softmax probability, 0.0-1.0.
    """

    timestamp: int
    count: int
    confidence: float

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "count": self.count, "confidence": self.confidence}


class CounterInference:
    """Loads a trained checkpoint once; call `predict()` per preprocessed
    window (same window shape/preprocessing the presence detector uses)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        n_components: int = 5,
        sample_rate_hz: float = 100.0,
        nperseg: int = 32,
        noverlap: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_components = n_components
        self.sample_rate_hz = sample_rate_hz
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.device = torch.device(device)
        self.model = load_checkpoint(checkpoint_path, device=self.device)

    @torch.no_grad()
    def predict(self, window: np.ndarray, timestamp_us: int) -> CountEvent:
        """window: shape (n_frames, n_subcarriers), preprocessed CSI amplitude."""
        features = compute_spectrogram_features(
            window,
            n_components=self.n_components,
            sample_rate_hz=self.sample_rate_hz,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)  # (1, C, F, T)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probs).item())
        return CountEvent(
            timestamp=timestamp_us,
            count=CLASS_COUNTS[predicted_idx],
            confidence=float(probs[predicted_idx].item()),
        )
