"""A small CNN classifying a CSI spectrogram window into a fixed set of
classes. Originally built for people counting (0/1/2/3+, scripts/train_counter.py)
but architecturally generic — scripts/train_har.py reuses the same class
for UT-HAR activity recognition (7 classes) with a different `n_classes`.

Sized for CPU real-time inference (~5K parameters at the default
in_channels=5). Global average pooling makes it agnostic to the exact
spectrogram spatial size, so it doesn't need retraining if
`nperseg`/`noverlap`/`window_s` change — only the channel count
(`n_components`, fixed at both training and inference time) has to match
`in_channels`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

# Class index -> people count. Index 3 ("3+") means "3 or more".
CLASS_COUNTS: tuple[int, ...] = (0, 1, 2, 3)
CLASS_LABELS: tuple[str, ...] = ("0", "1", "2", "3+")
N_CLASSES = len(CLASS_COUNTS)


class PeopleCounterCNN(nn.Module):
    """Input: (batch, in_channels, n_freq_bins, n_time_bins) — the output of
    pipeline.features.spectrogram.compute_spectrogram_features (in_channels
    must equal that function's n_components), batched. Output: (batch,
    n_classes) logits."""

    def __init__(self, *, in_channels: int = 5, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def save_checkpoint(
    path: str | Path,
    model: PeopleCounterCNN,
    *,
    class_labels: tuple[str, ...] = CLASS_LABELS,
    extra: dict | None = None,
) -> None:
    """Save a checkpoint that `load_checkpoint` can reconstruct the model
    from, without the caller needing to know its hyperparameters.

    `class_labels` defaults to the people-counting labels ("0".."3+");
    pass your own (e.g. UT-HAR's 7 activity names) for a model trained on
    a different task — it must have `model.n_classes` entries.
    """
    if len(class_labels) != model.n_classes:
        raise ValueError(f"class_labels has {len(class_labels)} entries but model.n_classes={model.n_classes}")
    payload = {
        "model_state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "n_classes": model.n_classes,
        "class_labels": list(class_labels),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> PeopleCounterCNN:
    """Load a checkpoint saved by `save_checkpoint`, returning a model in
    eval mode on `device`."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PeopleCounterCNN(in_channels=checkpoint["in_channels"], n_classes=checkpoint["n_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
