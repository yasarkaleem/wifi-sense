#!/usr/bin/env python3
"""Trains pipeline.models.counter.PeopleCounterCNN — repurposed for
activity recognition (7 classes) rather than people counting — on the
UT-HAR dataset, with a train/test split BY SESSION (not by window), so no
window from a given recording appears on both sides of the split (adjacent
windows within one continuous recording are highly correlated; splitting
at the window level would leak information and inflate reported accuracy).

Requires converted UT-HAR data — run datasets/download.py first:

    cd datasets
    python download.py ut-har --source /path/to/Dataset/Data --out ut-har

Then:

    cd services/pipeline
    pip install -e ".[dev,ml]"
    python scripts/train_har.py --data-dir ../../datasets/ut-har

Outputs (under --output-dir, default services/pipeline/checkpoints/):
    har_best.pt       - best checkpoint (by validation loss)
    har_metrics.json  - accuracy, per-class precision/recall/f1, and
                         confusion matrix, all on the held-out TEST
                         sessions (never seen during training), plus which
                         session IDs ended up in train vs. test

See datasets/README.md for UT-HAR's license and citation requirements
before using it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.features.spectrogram import compute_spectrogram_features
from pipeline.models.counter import PeopleCounterCNN, save_checkpoint
from pipeline.preprocess import hampel_filter, savitzky_golay_smooth, segment_sliding_window
from pipeline.training import (
    SpectrogramDataset,
    classification_report,
    confusion_matrix,
    print_report,
    stratified_split,
    train_model,
)

WIFI_SENSE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = WIFI_SENSE_ROOT / "datasets" / "ut-har"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

# UT-HAR's 7 activity classes (Yousefi et al., 2017) — order fixes class
# index. Matches datasets/download.py's UT_HAR_ACTIVITIES; duplicated
# rather than imported since datasets/ isn't an installed dependency of
# pipeline (see ../../../CLAUDE.md's service-isolation rule).
ACTIVITY_CLASSES: tuple[str, ...] = ("bed", "fall", "walk", "pickup", "run", "sitdown", "standup")


@dataclass(frozen=True)
class Session:
    session_id: str
    amplitude: np.ndarray  # (n_frames, subcarrier_count), raw
    label: np.ndarray  # (n_frames,), per-frame activity string
    sample_rate_hz: float  # estimated from the session's own timestamps


# ---------------------------------------------------------------------------
# Loading + session-level split
# ---------------------------------------------------------------------------


def load_sessions(data_dir: Path) -> list[Session]:
    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no .npz session files found under {data_dir}; run datasets/download.py first "
            "(see this script's module docstring)"
        )

    sessions = []
    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            timestamp_us = npz["timestamp_us"]
            amplitude = npz["amplitude"]
            label = npz["label"]
            session_id = str(npz["session_id"])

        if len(timestamp_us) < 2:
            print(f"  skipping {path.name}: too few frames ({len(timestamp_us)})", file=sys.stderr)
            continue

        # Real captures have per-frame timing jitter; estimate the rate
        # from the median inter-frame gap rather than assuming a fixed Hz.
        dt_us = float(np.median(np.diff(timestamp_us)))
        sample_rate_hz = 1_000_000 / dt_us if dt_us > 0 else 0.0
        if sample_rate_hz <= 0:
            print(f"  skipping {path.name}: could not estimate a sample rate", file=sys.stderr)
            continue

        sessions.append(Session(session_id=session_id, amplitude=amplitude, label=label, sample_rate_hz=sample_rate_hz))
    return sessions


def dominant_activity(label: np.ndarray) -> str | None:
    """The most common activity label in a session (ignoring frames whose
    label isn't one of ACTIVITY_CLASSES, e.g. blank/no-activity)."""
    values, counts = np.unique(label, return_counts=True)
    candidates = [(v, c) for v, c in zip(values, counts) if v in ACTIVITY_CLASSES]
    if not candidates:
        return None
    return max(candidates, key=lambda vc: vc[1])[0]


def session_split(sessions: list[Session], test_fraction: float, seed: int) -> tuple[list[Session], list[Session]]:
    """Splits whole SESSIONS (not windows) into train/test, stratified by
    each session's dominant activity, so every activity is represented on
    both sides and no window from one recording leaks across the split."""
    rng = np.random.default_rng(seed)
    by_activity: dict[str, list[Session]] = {}
    for session in sessions:
        activity = dominant_activity(session.label) or "unknown"
        by_activity.setdefault(activity, []).append(session)

    train: list[Session] = []
    test: list[Session] = []
    for group in by_activity.values():
        indices = list(range(len(group)))
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_fraction)) if len(indices) > 1 else 0
        test_indices = set(indices[:n_test])
        for i in indices:
            (test if i in test_indices else train).append(group[i])
    return train, test


# ---------------------------------------------------------------------------
# Windowing + majority-vote labeling + feature extraction
# ---------------------------------------------------------------------------


def build_examples(
    session: Session,
    *,
    window_s: float,
    stride_s: float,
    label_threshold: float,
    hampel_window: int,
    hampel_sigmas: float,
    savgol_window: int,
    savgol_polyorder: int,
    n_components: int,
    nperseg: int,
    noverlap: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses one session, segments it into windows, majority-vote
    labels each window from the session's per-frame labels (dropping
    windows where no single activity reaches `label_threshold` of the
    window's frames — e.g. transitions, or background/no-activity), and
    computes spectrogram features for the windows that are kept."""
    n_frames = session.amplitude.shape[0]
    if n_frames < max(hampel_window, savgol_window):
        return np.empty((0,)), np.empty((0,), dtype=np.int64)

    cleaned = savitzky_golay_smooth(
        hampel_filter(session.amplitude, window_size=hampel_window, n_sigmas=hampel_sigmas),
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )
    amplitude_windows = segment_sliding_window(
        cleaned, sample_rate_hz=session.sample_rate_hz, window_s=window_s, stride_s=stride_s
    )
    # segment_sliding_window is dtype-agnostic (pure indexing, no
    # arithmetic), so it windows the string label array in perfect lockstep
    # with the amplitude windows above.
    label_windows = segment_sliding_window(
        session.label[:, None], sample_rate_hz=session.sample_rate_hz, window_s=window_s, stride_s=stride_s
    )[:, :, 0]

    features, labels = [], []
    for amp_window, lbl_window in zip(amplitude_windows, label_windows):
        values, counts = np.unique(lbl_window, return_counts=True)
        best_idx = int(np.argmax(counts))
        best_label, best_count = values[best_idx], counts[best_idx]
        if best_label not in ACTIVITY_CLASSES or best_count / len(lbl_window) < label_threshold:
            continue
        features.append(
            compute_spectrogram_features(
                amp_window,
                n_components=n_components,
                sample_rate_hz=session.sample_rate_hz,
                nperseg=nperseg,
                noverlap=noverlap,
            )
        )
        labels.append(ACTIVITY_CLASSES.index(str(best_label)))

    if not features:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.stack(features), np.array(labels, dtype=np.int64)


def build_dataset(sessions: list[Session], **kwargs) -> tuple[np.ndarray, np.ndarray]:
    all_features, all_labels = [], []
    for session in sessions:
        features, labels = build_examples(session, **kwargs)
        if len(labels):
            all_features.append(features)
            all_labels.append(labels)
    if not all_features:
        return np.empty((0,)), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_features), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="directory of converted UT-HAR .npz session files"
    )
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.6,
        help="fraction of a window's frames that must share the majority activity label to "
        "keep the window, else it's dropped (matches the original UT-HAR paper's convention)",
    )
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--savgol-window", type=int, default=11)
    parser.add_argument("--savgol-polyorder", type=int, default=3)
    parser.add_argument("--n-components", type=int, default=5, help="PCA/spectrogram channel count")
    parser.add_argument("--nperseg", type=int, default=32)
    parser.add_argument("--noverlap", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="fraction of SESSIONS (not windows) held out for test, stratified by dominant activity",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="fraction of the remaining TRAIN windows further held out for early-stopping validation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"loading sessions from {args.data_dir}...", file=sys.stderr)
    sessions = load_sessions(args.data_dir)
    print(f"loaded {len(sessions)} sessions", file=sys.stderr)

    train_sessions, test_sessions = session_split(sessions, args.test_fraction, args.seed)
    print(
        f"session split: {len(train_sessions)} train / {len(test_sessions)} test "
        "(split by whole session, not by window)",
        file=sys.stderr,
    )
    print(f"  train sessions: {[s.session_id for s in train_sessions]}", file=sys.stderr)
    print(f"  test sessions:  {[s.session_id for s in test_sessions]}", file=sys.stderr)

    example_kwargs = dict(
        window_s=args.window_s,
        stride_s=args.stride_s,
        label_threshold=args.label_threshold,
        hampel_window=args.hampel_window,
        hampel_sigmas=args.hampel_sigmas,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        n_components=args.n_components,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
    )

    print("\nextracting windows + spectrogram features...", file=sys.stderr)
    X_train_full, y_train_full = build_dataset(train_sessions, **example_kwargs)
    X_test, y_test = build_dataset(test_sessions, **example_kwargs)
    print(f"train: {len(y_train_full)} windows, test: {len(y_test)} windows", file=sys.stderr)
    if len(y_train_full) == 0 or len(y_test) == 0:
        raise SystemExit(
            "not enough windows survived majority-vote labeling — try more/longer recordings, "
            "a shorter --window-s, or a lower --label-threshold"
        )

    for i, activity in enumerate(ACTIVITY_CLASSES):
        n_train = int((y_train_full == i).sum())
        n_test = int((y_test == i).sum())
        print(f"  {activity:>8}: {n_train:4d} train windows, {n_test:4d} test windows", file=sys.stderr)

    # Held-out TEST sessions are never touched until final evaluation below.
    # Split TRAIN further into train/val purely for early stopping.
    train_idx, val_idx = stratified_split(y_train_full, args.val_fraction, args.seed)
    train_dataset = SpectrogramDataset(X_train_full[train_idx], y_train_full[train_idx])
    val_dataset = SpectrogramDataset(X_train_full[val_idx], y_train_full[val_idx])
    test_dataset = SpectrogramDataset(X_test, y_test)
    print(
        f"train/val/test windows: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}", file=sys.stderr
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = PeopleCounterCNN(in_channels=args.n_components, n_classes=len(ACTIVITY_CLASSES)).to(device)

    print("\ntraining...", file=sys.stderr)
    history = train_model(
        model, train_loader, val_loader, epochs=args.epochs, patience=args.patience, lr=args.lr, device=device
    )

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch.to(device))
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y_batch.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    cm = confusion_matrix(y_true, y_pred, len(ACTIVITY_CLASSES))
    report = classification_report(cm, ACTIVITY_CLASSES)
    report["history"] = history
    report["train_sessions"] = [s.session_id for s in train_sessions]
    report["test_sessions"] = [s.session_id for s in test_sessions]

    print("\n=== held-out TEST session results (sessions never seen during training) ===", file=sys.stderr)
    print_report(report)

    checkpoint_path = args.output_dir / "har_best.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        class_labels=ACTIVITY_CLASSES,
        extra={
            "task": "ut_har_activity_recognition",
            "n_components": args.n_components,
            "nperseg": args.nperseg,
            "noverlap": args.noverlap,
            "window_s": args.window_s,
            "stride_s": args.stride_s,
        },
    )
    print(f"\nsaved checkpoint to {checkpoint_path}", file=sys.stderr)

    metrics_path = args.output_dir / "har_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved metrics report to {metrics_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
