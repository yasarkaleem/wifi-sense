# pipeline

Subscribes to CSI frames published by `services/ingest`, preprocesses each
window, runs a rule-based presence detector (always), an ML people counter
(if a checkpoint is configured), and a zone-level localizer (if a
checkpoint is configured), and publishes detection events over its own
ZeroMQ PUB socket.

## Preprocessing (`pipeline/preprocess.py`)

Pure, dependency-free-of-I/O functions operating on `(n_frames,
n_subcarriers)` NumPy arrays (rows = time, one row per CSI frame; columns =
subcarriers) — see each function's docstring for exact shapes:

- `hampel_filter` — per-subcarrier outlier removal (sliding-window
  median + MAD).
- `savitzky_golay_smooth` — per-subcarrier polynomial smoothing.
- `sanitize_phase` — unwraps phase across subcarriers and removes the
  per-frame linear trend (CFO/STO correction).
- `segment_sliding_window` — slices a time series into overlapping
  fixed-length windows (default 2s window / 0.5s stride).
- `pca_reduce` — SVD-based PCA across subcarriers, keeping the top-k
  components.

## Presence detection (`pipeline/detectors/presence.py`)

Rule-based, not ML:

1. Each window's *cleaned* amplitude (Hampel + Savitzky-Golay) is projected
   onto its top-k PCA components (`window_motion_score`); the sum of each
   component's variance across the window is a single "how much did this
   move" scalar — a static scene stays near zero, a moving person's
   dominant components swing substantially.
2. `PresenceDetector` treats the first `calibration_s` seconds of *that
   score* as an empty-room baseline and sets an adaptive threshold at
   `baseline_mean + n_sigmas * baseline_std`.
3. Once calibrated, every window produces a `PresenceEvent`:
   `{timestamp, presence: bool, motion_intensity: float 0-1}` — `timestamp`
   is microseconds (same unit/epoch as a CSI frame's `timestamp_us`);
   `motion_intensity` is 0 at the baseline mean, 0.5 exactly at the
   threshold, saturating at 1.0 twice as far above baseline as the
   threshold is.

`pipeline/windowing.py`'s `RollingWindower` turns the live per-frame stream
into these fixed-size, fixed-stride windows online (as opposed to
`preprocess.segment_sliding_window`, which segments an already-collected
batch).

**Tuning note:** with `services/replay`'s synthetic noise levels (amplitude
noise_std 0.3), an `n_sigmas` of 4 occasionally false-positives on pure
empty-room noise because overlapping calibration windows are highly
correlated (little new data per window), understating the true score
variance. The real disturbance signal is enormous by comparison (~450x the
baseline in that same synthetic data), so the default is a more
conservative `n_sigmas=6.0` — verified empirically to false-positive-free
across 1300+ simulated events while barely affecting detection latency.

## ML people counting (`pipeline/features/spectrogram.py`, `pipeline/models/`)

Classifies each window into 0/1/2/3+ people, alongside the presence
detector. Off by default (no ML dependency for presence-only use) — enable
with `--counter-checkpoint`. See
[`CLAUDE.md`](../../CLAUDE.md#ml-based-people-counting) for the full model
architecture writeup and retraining walkthrough; short version:

```bash
pip install -e ".[dev,ml]"

# train (spawns real replay+ingest, ~4-5 min for the small default dataset)
python scripts/train_counter.py

# serve, alongside presence detection, publishing on the `count` topic
python -m pipeline --sub-port 5567 --pub-port 5568 \
  --counter-checkpoint checkpoints/counter_best.pt
```

`compute_spectrogram_features` turns a window into a `(n_components,
n_freq_bins, n_time_bins)` stacked-STFT tensor (e.g. `(5, 17, 14)` for the
defaults); `PeopleCounterCNN` is a ~5.6K-parameter CNN sized for CPU
real-time inference (measured ~1.1ms/window, well under the 20ms budget —
see `tests/test_counter_model.py`).

## Activity recognition on UT-HAR (`scripts/train_har.py`)

Reuses the same `PeopleCounterCNN` architecture and spectrogram pipeline
as the people counter above, retargeted at UT-HAR's 7-class activity
labels instead of a people count (only `n_classes` differs — the model is
architecture-generic). See
[`CLAUDE.md`](../../CLAUDE.md#real-datasets) for the full writeup; short
version:

```bash
# 1. convert UT-HAR first (see ../../datasets/README.md for the manual
#    download step and license/citation requirements)
cd ../../datasets && python download.py ut-har --source /path/to/Dataset/Data --out ut-har

# 2. train
cd ../services/pipeline
pip install -e ".[dev,ml]"
python scripts/train_har.py --data-dir ../../datasets/ut-har
```

The key difference from `train_counter.py`: the train/test split happens
**by whole recording session**, not by window — see
`scripts/train_har.py`'s `session_split()`. Splitting at the window level
would leak information (adjacent windows in one continuous recording are
highly correlated) and inflate reported accuracy. Each window's label is a
majority vote over its frames' per-frame annotations, dropping windows
where no single activity reaches `--label-threshold` (default 0.6).
Outputs `har_best.pt` / `har_metrics.json` (accuracy, confusion matrix,
and the exact train/test session ID split) under `--output-dir`.

## Zone-level localization (`pipeline/models/localizer.py`, `pipeline/calibrate.py`)

Fingerprints each window against a room zone (see
[`../../room.yaml`](../../room.yaml), default 2x3 grid: `A1`..`B3`),
alongside the presence detector and counter. Off by default — enable with
`--localizer-checkpoint`. See
[`CLAUDE.md`](../../CLAUDE.md#zone-level-localization) for the full
writeup; short version:

```bash
pip install -e ".[dev,localize]"

# calibrate each zone on-site (stand in the zone while replay/ESP32 streams)
python -m pipeline.calibrate --zone A1 --seconds 60
python -m pipeline.calibrate --zone A2 --seconds 60
# ...repeat for every zone; needs >= 2 calibrated zones before it fits...

# serve, publishing on the `zones` topic
python -m pipeline --sub-port 5567 --pub-port 5568 \
  --localizer-checkpoint checkpoints/localizer.joblib
```

`ZoneLocalizer` is a `scikit-learn` `HistGradientBoostingClassifier` over
*flattened* spectrogram features — chosen over a CNN so on-site
calibration fits in well under a second with no training-loop tuning.
Unlike the counter's CNN (spatial-size-agnostic via global average
pooling), this means `--window-s`/`--stride-s`/`--localizer-n-components`/
`--localizer-nperseg`/`--localizer-noverlap` must match *exactly* between
calibration and serving — a mismatch raises a clear scikit-learn error
rather than silently misbehaving. Publishes `{timestamp, zones:
[{zone_id, occupancy_probability}, ...]}` — one entry per configured zone.

`tests/test_integration_localization.py` proves the whole loop for real:
calibrates two zones from live replay subprocesses, then runs the actual
`pipeline` service against a *fresh* recording of each zone and checks its
predictions converge on the right one.

## Running the service

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # add ".[ml]" too for --counter-checkpoint

python -m pipeline --sub-port 5567 --pub-port 5568
```

CLI flags (all optional; every one is also settable via a `PIPELINE_*` env
var — see `pipeline/cli.py`'s module docstring for the full list):

| Flag              | Default     |
|--------------------|-------------|
| `--sub-host` / `--sub-port` | `localhost` / `5567` (ingest's PUB) |
| `--pub-host` / `--pub-port` | `0.0.0.0` / `5568` (this service's PUB) |
| `--sample-rate-hz` | `100`       |
| `--window-s` / `--stride-s` | `2.0` / `0.5` |
| `--calibration-s` | `5.0`        |
| `--n-sigmas`       | `6.0`        |
| `--pca-components` | `5`         |
| `--counter-checkpoint` | unset (ML counting off) |
| `--counter-n-components` / `--counter-nperseg` / `--counter-noverlap` | `5` / `32` / unset (must match training) |
| `--localizer-checkpoint` | unset (localization off) |
| `--localizer-n-components` / `--localizer-nperseg` / `--localizer-noverlap` | `5` / `32` / unset (must match calibration exactly) |

Try it end to end (three terminals):

```bash
# terminal 1
cd ../ingest && python -m ingest --pub-port 5567

# terminal 2
python -m pipeline --sub-port 5567 --pub-port 5568

# terminal 3
cd ../replay && python -m replay --scenario one_person_walking --target localhost:5566
```

Pipeline logs every presence transition to stderr, e.g.
`presence -> True (motion_intensity=0.87, threshold=1.234)`, every count
change if `--counter-checkpoint` is set (`count -> 2 (confidence=0.81)`),
and every predicted-zone change if `--localizer-checkpoint` is set
(`zone -> A1 (occupancy_probability=0.73)`).

## Demo (`scripts/preprocess_demo.py`)

Pulls live frames from `services/ingest`'s ZeroMQ PUB socket, runs them
through the preprocessing functions above, and plots raw vs. cleaned
amplitude/phase as side-by-side heatmaps:

```bash
pip install -e ".[demo]"
python scripts/preprocess_demo.py --host localhost --port 5567
```

Add `--output path.png` to save the figure instead of opening a window
(useful over SSH/headless).

## Via docker compose

```bash
docker compose up --build ingest pipeline
```

The image bakes in the `ml` and `localize` extras so ML counting and zone
localization are always available; each stays inactive unless you mount a
checkpoint and set `PIPELINE_COUNTER_CHECKPOINT` / `PIPELINE_LOCALIZER_CHECKPOINT`
(or pass `--counter-checkpoint` / `--localizer-checkpoint`) in
`docker-compose.yml`.

## Tests

```bash
pip install -e ".[dev]"        # add ".[ml]" and/or ".[localize]" for those tests
pytest                          # unit tests only run instantly; counter/localizer
                                 # tests auto-skip (not fail) if torch/scikit-learn
                                 # aren't installed
pytest -m integration           # spawns real replay+ingest+pipeline subprocesses;
                                 # requires sibling services/replay and
                                 # services/ingest checked out (they always are,
                                 # it's the same monorepo) — see
                                 # tests/test_integration_presence.py and
                                 # tests/test_integration_localization.py
```
