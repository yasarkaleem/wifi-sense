#!/usr/bin/env python3
"""Trains the "walking person" demo's two models — the people counter
(0/1/2) and the zone localizer — on the dataset produced by
scripts/generate_zone_dataset.py, and saves both to services/pipeline/models/,
which pipeline/cli.py auto-loads on startup if present (see ../../../CLAUDE.md).

Unlike scripts/train_counter.py's checkpoints/ (gitignored — regenerate,
don't commit; see ../../../.gitignore), the output of THIS script is meant
to be committed: it's what makes `docker compose up` show a working
count/zone-heatmap demo with no manual training step. Keep the dataset
small (see generate_zone_dataset.py's defaults) so the committed
checkpoints stay small too — this is a proof-of-concept demo model, not a
production one, exactly like train_counter.py's own checkpoints.

Usage:
    pip install -e ".[dev,ml,localize]"
    python scripts/generate_zone_dataset.py
    python scripts/train_demo_models.py

Outputs (under --output-dir, default services/pipeline/models/):
    counter_demo.pt       - PeopleCounterCNN checkpoint, n_classes=3 (0/1/2)
    localizer_demo.joblib - ZoneLocalizer checkpoint
    demo_metrics.json     - both models' validation reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.features.spectrogram import compute_spectrogram_features
from pipeline.models.counter import PeopleCounterCNN, save_checkpoint
from pipeline.models.localizer import ZoneLocalizer
from pipeline.room import DEFAULT_ROOM_CONFIG_PATH, load_room_config
from pipeline.training import (
    SpectrogramDataset,
    classification_report,
    confusion_matrix,
    print_report,
    stratified_split,
    train_model,
)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "zone_demo.npz"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models"

COUNT_CLASS_LABELS = ("0", "1", "2")
N_COUNT_CLASSES = len(COUNT_CLASS_LABELS)


def _compute_features(amplitude_windows: np.ndarray, *, n_components: int, sample_rate_hz: float, nperseg: int, noverlap: int | None) -> np.ndarray:
    return np.stack(
        [
            compute_spectrogram_features(
                w, n_components=n_components, sample_rate_hz=sample_rate_hz, nperseg=nperseg, noverlap=noverlap
            )
            for w in amplitude_windows
        ]
    )


def train_counter(
    X: np.ndarray, y: np.ndarray, *, args: argparse.Namespace, device: torch.device
) -> tuple[PeopleCounterCNN, dict]:
    train_idx, val_idx = stratified_split(y, args.val_fraction, args.seed)
    train_loader = DataLoader(
        SpectrogramDataset(X[train_idx], y[train_idx]), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(SpectrogramDataset(X[val_idx], y[val_idx]), batch_size=args.batch_size, shuffle=False)

    model = PeopleCounterCNN(in_channels=args.n_components, n_classes=N_COUNT_CLASSES).to(device)
    print("\ntraining counter (0/1/2)...", file=sys.stderr)
    history = train_model(
        model, train_loader, val_loader, epochs=args.epochs, patience=args.patience, lr=args.lr, device=device
    )

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits = model(X_batch.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, N_COUNT_CLASSES)
    report = classification_report(cm, COUNT_CLASS_LABELS)
    report["history"] = history
    print_report(report)
    return model, report


def train_localizer(X: np.ndarray, y: np.ndarray, zone_ids: tuple[str, ...]) -> ZoneLocalizer:
    print(f"\ntraining localizer ({len(zone_ids)} zones)...", file=sys.stderr)
    localizer = ZoneLocalizer(zone_ids)
    localizer.fit(X, y)
    return localizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--room-config", type=Path, default=DEFAULT_ROOM_CONFIG_PATH)
    parser.add_argument("--n-components", type=int, default=5, help="PCA/spectrogram channel count")
    parser.add_argument("--nperseg", type=int, default=32)
    parser.add_argument("--noverlap", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.dataset.exists():
        raise SystemExit(f"{args.dataset} does not exist — run scripts/generate_zone_dataset.py first")

    torch.manual_seed(args.seed)
    room = load_room_config(args.room_config)

    with np.load(args.dataset) as npz:
        amplitude = npz["amplitude"]
        zone_label = npz["zone_label"]
        count_label = npz["count_label"]
        sample_rate_hz = float(npz["sample_rate_hz"])

    print(f"loaded {len(amplitude)} windows from {args.dataset}", file=sys.stderr)

    print("computing spectrogram features for all windows...", file=sys.stderr)
    features = _compute_features(
        amplitude,
        n_components=args.n_components,
        sample_rate_hz=sample_rate_hz,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
    )

    device = torch.device(args.device)
    counter_model, counter_report = train_counter(features, count_label, args=args, device=device)

    zone_mask = zone_label != ""
    zone_index = {zone_id: i for i, zone_id in enumerate(room.zone_ids)}
    localizer_y = np.array([zone_index[z] for z in zone_label[zone_mask]], dtype=np.int64)
    localizer = train_localizer(features[zone_mask], localizer_y, room.zone_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    counter_path = args.output_dir / "counter_demo.pt"
    save_checkpoint(
        counter_path,
        counter_model,
        class_labels=COUNT_CLASS_LABELS,
        extra={
            "n_components": args.n_components,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "sample_rate_hz": sample_rate_hz,
        },
    )
    print(f"\nsaved counter checkpoint to {counter_path}", file=sys.stderr)

    localizer_path = args.output_dir / "localizer_demo.joblib"
    localizer.save(localizer_path)
    print(f"saved localizer checkpoint to {localizer_path}", file=sys.stderr)

    metrics_path = args.output_dir / "demo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counter": counter_report,
                "localizer": {
                    "zone_ids": list(room.zone_ids),
                    "n_samples": int(zone_mask.sum()),
                    "n_components": args.n_components,
                    "nperseg": args.nperseg,
                    "noverlap": args.noverlap,
                    "sample_rate_hz": sample_rate_hz,
                },
            },
            f,
            indent=2,
        )
    print(f"saved metrics report to {metrics_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
