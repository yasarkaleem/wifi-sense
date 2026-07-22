"""Unit tests for scripts/train_har.py's pure logic: session loading,
session-level (not window-level) train/test splitting, and majority-vote
window labeling. Uses small synthetic .npz files written directly (not via
datasets/download.py) to stay self-contained within pipeline's own test
suite."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import train_har  # noqa: E402


def _make_session(session_id: str, activity: str, n_frames: int = 100, rate_hz: float = 100.0) -> train_har.Session:
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    amplitude = 20.0 + rng.normal(0, 0.5, size=(n_frames, 90))
    label = np.full(n_frames, activity, dtype="<U16")
    return train_har.Session(session_id=session_id, amplitude=amplitude, label=label, sample_rate_hz=rate_hz)


def _write_npz(tmp_path, session_id: str, activity: str, n_frames: int = 100, rate_hz: float = 100.0):
    rng = np.random.default_rng(abs(hash(session_id)) % (2**32))
    amplitude = 20.0 + rng.normal(0, 0.5, size=(n_frames, 90))
    phase = rng.uniform(-np.pi, np.pi, size=(n_frames, 90))
    dt_us = round(1_000_000 / rate_hz)
    timestamp_us = (1_700_000_000_000_000 + np.arange(n_frames) * dt_us).astype(np.int64)
    label = np.full(n_frames, activity, dtype="<U16")

    np.savez_compressed(
        tmp_path / f"{session_id}.npz",
        session_id=session_id,
        timestamp_us=timestamp_us,
        amplitude=amplitude,
        phase=phase,
        rssi=np.zeros(n_frames, dtype=np.int64),
        channel=np.ones(n_frames, dtype=np.int64),
        source_mac="00:00:00:00:00:00",
        subcarrier_count=90,
        label=label,
    )


# ---------------------------------------------------------------------------
# load_sessions
# ---------------------------------------------------------------------------


def test_load_sessions_estimates_sample_rate_from_timestamps(tmp_path):
    _write_npz(tmp_path, "walk_1", "walk", n_frames=50, rate_hz=200.0)

    sessions = train_har.load_sessions(tmp_path)

    assert len(sessions) == 1
    assert sessions[0].session_id == "walk_1"
    assert sessions[0].sample_rate_hz == pytest.approx(200.0, rel=0.01)


def test_load_sessions_skips_too_short_sessions(tmp_path):
    _write_npz(tmp_path, "walk_1", "walk", n_frames=1)
    _write_npz(tmp_path, "walk_2", "walk", n_frames=50)

    sessions = train_har.load_sessions(tmp_path)

    assert len(sessions) == 1
    assert sessions[0].session_id == "walk_2"


def test_load_sessions_raises_when_directory_has_no_npz(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_har.load_sessions(tmp_path)


# ---------------------------------------------------------------------------
# dominant_activity
# ---------------------------------------------------------------------------


def test_dominant_activity_picks_majority_label():
    label = np.array(["walk"] * 8 + ["fall"] * 2, dtype="<U16")
    assert train_har.dominant_activity(label) == "walk"


def test_dominant_activity_ignores_non_activity_labels():
    label = np.array([""] * 5 + ["bed"] * 3, dtype="<U16")
    assert train_har.dominant_activity(label) == "bed"


def test_dominant_activity_returns_none_when_no_known_activity_present():
    label = np.array(["", "", "noactivity"], dtype="<U16")
    assert train_har.dominant_activity(label) is None


# ---------------------------------------------------------------------------
# session_split
# ---------------------------------------------------------------------------


def test_session_split_never_puts_a_session_in_both_train_and_test():
    sessions = [_make_session(f"walk_{i}", "walk") for i in range(5)] + [
        _make_session(f"bed_{i}", "bed") for i in range(5)
    ]

    train, test = train_har.session_split(sessions, test_fraction=0.4, seed=0)

    train_ids = {s.session_id for s in train}
    test_ids = {s.session_id for s in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {s.session_id for s in sessions}


def test_session_split_stratifies_by_dominant_activity():
    sessions = [_make_session(f"walk_{i}", "walk") for i in range(10)] + [
        _make_session(f"bed_{i}", "bed") for i in range(10)
    ]

    train, test = train_har.session_split(sessions, test_fraction=0.2, seed=0)

    test_activities = [train_har.dominant_activity(s.label) for s in test]
    assert test_activities.count("walk") == 2
    assert test_activities.count("bed") == 2


def test_session_split_single_session_activity_goes_to_test_not_lost():
    sessions = [_make_session("bed_1", "bed")]
    train, test = train_har.session_split(sessions, test_fraction=0.2, seed=0)
    assert len(train) + len(test) == 1


# ---------------------------------------------------------------------------
# build_examples (windowing + majority-vote labeling + feature extraction)
# ---------------------------------------------------------------------------


_EXAMPLE_KWARGS = dict(
    window_s=0.3,
    stride_s=0.15,
    label_threshold=0.6,
    hampel_window=7,
    hampel_sigmas=3.0,
    savgol_window=11,
    savgol_polyorder=3,
    n_components=5,
    nperseg=16,
    noverlap=None,
)


def test_build_examples_labels_pure_activity_windows_correctly():
    session = _make_session("walk_1", "walk", n_frames=100, rate_hz=100.0)  # 1s, uniform "walk" label

    features, labels = train_har.build_examples(session, **_EXAMPLE_KWARGS)

    assert len(labels) > 0
    assert set(labels.tolist()) == {train_har.ACTIVITY_CLASSES.index("walk")}
    assert features.shape[0] == len(labels)
    assert features.shape[1] == _EXAMPLE_KWARGS["n_components"]


def test_build_examples_drops_windows_below_label_threshold():
    n_frames = 100
    rng = np.random.default_rng(1)
    amplitude = 20.0 + rng.normal(0, 0.5, size=(n_frames, 90))
    # Alternate frame-by-frame between two activities so every window
    # (30 frames, much wider than one frame) sees roughly a 50/50 split ->
    # no majority reaches the 0.6 threshold -> every window is dropped.
    label = np.array(["walk" if i % 2 == 0 else "fall" for i in range(n_frames)], dtype="<U16")
    session = train_har.Session(session_id="mixed", amplitude=amplitude, label=label, sample_rate_hz=100.0)

    features, labels = train_har.build_examples(session, **_EXAMPLE_KWARGS)

    assert len(labels) == 0


def test_build_examples_drops_no_activity_windows():
    n_frames = 100
    rng = np.random.default_rng(2)
    amplitude = 20.0 + rng.normal(0, 0.5, size=(n_frames, 90))
    label = np.full(n_frames, "", dtype="<U16")  # no activity at all
    session = train_har.Session(session_id="empty", amplitude=amplitude, label=label, sample_rate_hz=100.0)

    features, labels = train_har.build_examples(session, **_EXAMPLE_KWARGS)

    assert len(labels) == 0


def test_build_examples_too_few_frames_returns_empty():
    session = _make_session("short", "walk", n_frames=5, rate_hz=100.0)
    features, labels = train_har.build_examples(session, **_EXAMPLE_KWARGS)
    assert len(labels) == 0


def test_build_dataset_concatenates_across_sessions():
    sessions = [_make_session("walk_1", "walk"), _make_session("bed_1", "bed")]
    X, y = train_har.build_dataset(sessions, **_EXAMPLE_KWARGS)

    assert len(y) > 0
    assert set(y.tolist()) == {
        train_har.ACTIVITY_CLASSES.index("walk"),
        train_har.ACTIVITY_CLASSES.index("bed"),
    }
