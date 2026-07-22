"""Shared training/evaluation utilities for spectrogram-classifier
training scripts (scripts/train_counter.py, scripts/train_har.py).

Dataset-agnostic: operates on already-extracted `(N, C, F, T)` feature
arrays and integer labels, however they were collected. Both scripts
differ only in how they build that (X, y) pair — one from live synthetic
replay scenarios, one from a converted real dataset on disk.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def stratified_split(y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits indices per-class so every class is represented in both
    train and val, even with a small total dataset."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


class SpectrogramDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def evaluate_loss_and_accuracy(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    patience: int,
    lr: float,
    device: torch.device,
) -> list[dict]:
    """Trains with early stopping on validation loss; restores the model to
    its best-validation-loss weights before returning."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += X_batch.size(0)
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            file=sys.stderr,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)", file=sys.stderr)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(cm: np.ndarray, class_labels: tuple[str, ...]) -> dict:
    report: dict = {"per_class": {}}
    for i, label in enumerate(class_labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report["per_class"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }
    report["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    report["confusion_matrix"] = cm.tolist()
    report["class_labels"] = list(class_labels)
    return report


def print_report(report: dict) -> None:
    print(f"\nvalidation accuracy: {report['accuracy']:.3f}\n", file=sys.stderr)
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}", file=sys.stderr)
    for label, m in report["per_class"].items():
        print(
            f"{label:>10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10d}",
            file=sys.stderr,
        )
    print("\nconfusion matrix (rows=true, cols=predicted):", file=sys.stderr)
    header = "          " + "".join(f"{label:>10}" for label in report["class_labels"])
    print(header, file=sys.stderr)
    for label, row in zip(report["class_labels"], report["confusion_matrix"]):
        print(f"{label:>10}" + "".join(f"{v:>10d}" for v in row), file=sys.stderr)
