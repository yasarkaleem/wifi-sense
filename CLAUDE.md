# wifi-sense

## Project goal

An open-source WiFi CSI (Channel State Information) sensing platform that
counts the number of people in a room and estimates their zone-level
location — without cameras. It uses fluctuations in WiFi signal propagation
(amplitude/phase across subcarriers), captured by an ESP32, and machine
learning to infer occupancy and coarse position.

## Architecture

```
ESP32 (CSI capture)
   │  UDP, JSON CSI frames (see docs/csi-frame-schema.md)
   ▼
services/ingest    (Python)  — validates frames, buffers a ring buffer,
   │                           republishes over a ZeroMQ PUB socket
   │  ZeroMQ PUB/SUB, topic "csi", same JSON frame shape
   ▼
services/pipeline  (Python)  — preprocesses each window (Hampel filter,
   │                           Savitzky-Golay smoothing, phase sanitization),
   │                           runs a rule-based presence detector (PCA
   │                           variance of top components vs. an
   │                           empty-room-calibrated adaptive threshold) and,
   │                           if checkpoints are configured, a small CNN
   │                           people counter and/or a gradient-boosted
   │                           zone localizer, republishing all over a
   │                           ZeroMQ PUB socket
   │  ZeroMQ PUB/SUB, topic "presence": {timestamp, presence, motion_intensity}
   │  ZeroMQ PUB/SUB, topic "count":    {timestamp, count, confidence}
   │  ZeroMQ PUB/SUB, topic "zones":    {timestamp, zones: [{zone_id, occupancy_probability}, ...]}
   ▼
services/api       (Node.js) — subscribes to all three topics, merges them
   │                           into one combined live state, persists a
   │                           history of it to SQLite
   │  REST: GET /status, GET /history?minutes=n
   │  WebSocket /ws: pushes the merged state on every pipeline event
   ▼
dashboard          (React)   — live person count, motion-intensity gauge,
                               zone heatmap, 10-minute history chart
```

`services/ingest` also exposes an optional debug live waterfall/heatmap
(web page or matplotlib) of its ring buffer, for visually confirming CSI
disturbances without waiting on the rest of the pipeline.

`services/replay` is a standalone tool that streams CSI frames over UDP in
the same wire format as the ESP32, so the rest of the pipeline can be
developed and tested without physical hardware — either synthetic
scenarios (`--scenario`) or a real, converted dataset recording
(`--dataset`/`--file`; see "Real datasets" below).

`firmware/esp32-csi` is the ESP-IDF project that runs on the ESP32. One
firmware image, two roles chosen via `idf.py menuconfig` (or NVS at
runtime, no reflash needed — see `firmware/esp32-csi/README.md`):
**receiver** enables CSI capture (`esp_wifi_set_csi_config`/
`_set_csi_rx_cb`/`_set_csi`), decodes each captured frame off the WiFi
driver callback via a rate-limited FreeRTOS queue (dropping frames rather
than blocking if a consumer falls behind), and sends it as JSON matching
`docs/csi-frame-schema.md` to `services/ingest`; **sender** is a second
ESP32 on the same network that broadcasts small UDP packets at a fixed,
configurable rate (default 100Hz) purely to give the receiver
CSI-capturable traffic at a controlled rate. Both roles reconnect
automatically on WiFi drop with capped exponential backoff. Supported
chips: ESP32/S2/S3/C3 (C5/C6/C61 have CSI hardware but a different
config/struct layout the firmware doesn't decode — `idf.py build` fails
loudly if targeted). Written and reviewed against Espressif's real
esp-csi/ESP-IDF sources but **not compiled/flashed on hardware** in this
repo's own development so far — see the "Verification status" note at
the top of `firmware/esp32-csi/README.md` before trusting it blind.

## CSI frame schema

The canonical wire format for a single CSI reading is defined in
[`docs/csi-frame-schema.md`](docs/csi-frame-schema.md). At a glance, a frame
carries: `timestamp_us`, `source_mac`, `rssi`, `channel`,
`subcarrier_count`, `amplitude` (array), `phase` (array), and
`sequence_number`. Every producer and consumer of CSI frames must conform to
that schema — update the doc first if the shape needs to change, then update
producers/consumers to match.

## ML-based people counting

`services/pipeline` classifies each window into 0/1/2/3+ people with a
small CNN, alongside (not instead of) the rule-based presence detector.
It's off by default — presence detection alone has no ML dependency — and
activates when `services/pipeline`'s `--counter-checkpoint` /
`PIPELINE_COUNTER_CHECKPOINT` points at a trained checkpoint.

**Feature extraction** (`pipeline/features/spectrogram.py`): each window's
cleaned CSI amplitude is PCA-reduced to its top-k components (default
k=5), each component's time series gets its own STFT, and the resulting
log-magnitude spectrograms are stacked into one `(n_components,
n_freq_bins, n_time_bins)` tensor — e.g. `(5, 17, 14)` for the default 2s
window at 100Hz with `nperseg=32`. Each channel is normalized to zero
mean / unit std independently, since different PCA components have very
different natural magnitudes.

**Model** (`pipeline/models/counter.py`, `PeopleCounterCNN`): two
`Conv2d → BatchNorm2d → ReLU` blocks (in_channels→16→32) followed by
global average pooling and a linear classifier to 4 classes. ~5,600
parameters. Global average pooling makes it agnostic to the exact
spectrogram spatial size, so changing `nperseg`/`window_s` doesn't require
retraining — only the channel count (`n_components`) has to match between
training and inference. Measured CPU inference latency: ~0.2ms for the
model alone, ~1.1ms including feature extraction — both far under the
20ms/window real-time budget (verified in
`tests/test_counter_model.py::test_cpu_inference_latency_under_20ms`).

**Inference wrapper** (`pipeline/models/counter_inference.py`,
`CounterInference`): loads a checkpoint once, and `.predict(window,
timestamp_us)` runs feature extraction + the model + softmax, returning a
`CountEvent` — `count` of 3 means "3 or more"; `confidence` is the
predicted class's softmax probability.

**Retraining** (`scripts/train_counter.py`): trains entirely on synthetic
data, spawning real `services/replay` + `services/ingest` subprocesses per
recording (network only, no in-process imports — same isolation rule as
everywhere else) and mapping each replay scenario to a class label:
`empty_room`→0, `one_person_walking`→1, `two_people`→2, `three_people`→3.
The `three_people` scenario exists in
`services/replay/src/replay/scenarios.yaml` specifically to supply the 3+
class.

```bash
cd services/pipeline
pip install -e ".[dev,ml]"
python scripts/train_counter.py                                    # ~4-5 min, small dataset
python scripts/train_counter.py --seconds-per-recording 30 --recordings-per-scenario 8  # more data, slower
```

Since data collection runs each replay scenario in real time, more data
costs real wall-clock time (`n_scenarios × recordings_per_scenario ×
seconds_per_recording`). Outputs land in `services/pipeline/checkpoints/`
(gitignored — regenerate rather than commit): `counter_best.pt` (the
checkpoint, saved at the best validation-loss epoch under early stopping)
and `counter_metrics.json` (accuracy, per-class precision/recall/f1, and a
confusion matrix on the held-out validation split). Point the service at
the result:

```bash
python -m pipeline --counter-checkpoint services/pipeline/checkpoints/counter_best.pt
```

This produces a small proof-of-concept model from purely synthetic data —
useful for validating the training/inference mechanics end to end, not a
production-grade classifier. Real deployment needs real captured CSI with
far more diversity (multiple rooms, walking patterns, furniture layouts).

## Real datasets

`datasets/download.py` converts public WiFi CSI datasets into the
canonical CSI frame schema, saved as one `.npz` file per recording session
under `datasets/<name>/`. See
[`datasets/README.md`](datasets/README.md) for licenses, citations, and
manual download steps (the source archives are too large/interactively
gated to auto-download reliably, so conversion — the part fully under our
control — is what's automated).

- **UT-HAR** (Yousefi et al., 2017) — implemented. Human activity
  recognition: 7 activities, 1 person, Intel 5300 NIC. `download.py`
  parses its raw `input_*.csv`/`annotation_*.csv` session pairs (1
  timestamp + 90 amplitude + 90 phase columns per row) directly — no
  intermediate pre-windowed format.
- **Widar 3.0** (Zheng et al., 2019) — stub only (`NotImplementedError`).
  Complex-valued `.mat` CSI with derived DFS/BVP features and
  domain-tuple-keyed sessions — a materially different parser than
  UT-HAR's flat CSV, out of scope for now.

**Streaming a converted recording** (`services/replay`):

```bash
python -m replay --dataset ut-har --file datasets/ut-har/bed_1.npz --target localhost:5566
```

Pacing defaults to the recording's own inter-frame timestamps (faithfully
reproducing the original capture's timing, gaps included); `--rate`
overrides to a fixed rate instead. Either way, each emitted frame's
`timestamp_us` is the actual send time (matching synthetic mode and
"capture time on the sender" in docs/csi-frame-schema.md), not the
recording's original historical timestamp. The per-frame activity label
`datasets/download.py` stores in the `.npz` is a training-only convenience
field, not part of the wire schema — replay never sends it.

**Training on UT-HAR** (`scripts/train_har.py`): reuses
`pipeline.models.counter.PeopleCounterCNN` and the same spectrogram
feature pipeline as `train_counter.py`, repurposed for 7-class activity
classification instead of people counting (the model is architecture-generic
— only `n_classes` differs). The critical difference from
`train_counter.py`: the train/test split happens **by whole session**, not
by window — adjacent windows within one continuous recording are highly
correlated, so splitting at the window level would leak information across
train/test and inflate reported accuracy. Each window's label is a
majority vote over its frames' annotations (dropping windows where no
single activity reaches `--label-threshold`, default 0.6, matching the
original paper's convention). Outputs `har_best.pt` and `har_metrics.json`
(accuracy, confusion matrix, and which session IDs landed in train vs.
test) under `services/pipeline/checkpoints/`.

```bash
cd services/pipeline
pip install -e ".[dev,ml]"
python scripts/train_har.py --data-dir ../../datasets/ut-har
```

## Zone-level localization

[`room.yaml`](room.yaml) (repo root) defines the room as a grid of zones —
default 2x3 (rows A, B; columns 1-3), zone IDs like `A1`, row-major. This
isn't just a room-*description* file: `services/replay/src/replay/scenarios.yaml`
has one scenario per zone (`A1`..`B3`) with a distinguishable synthetic
disturbance, and `services/pipeline` fingerprints a window against a
classifier trained per zone_id — both keyed to the same IDs by convention
(not by reading room.yaml — see "Coding conventions" below).

**Fingerprinting model** (`pipeline/models/localizer.py`, `ZoneLocalizer`):
a `scikit-learn` `HistGradientBoostingClassifier` over *flattened*
spectrogram features (the same `compute_spectrogram_features` used by the
counter/HAR models), one class per zone. Chosen over a CNN specifically
for on-site calibration: it fits in well under a second on the modest
sample counts a short per-zone run yields, with no
epochs/learning-rate/early-stopping to tune each time a technician
recalibrates a zone. The trade-off — flattening loses the CNN's
global-average-pooling spatial-size invariance — is that
`--window-s`/`--stride-s`/`--localizer-n-components`/`--localizer-nperseg`/
`--localizer-noverlap` must match *exactly* between calibration and
inference, or scikit-learn raises a clear feature-count mismatch error.

**Calibration CLI** (`pipeline/calibrate.py`, `python -m pipeline.calibrate
--zone A1 --seconds 60`): stand in a zone while `services/replay` (or a
live ESP32) streams through `services/ingest`; it subscribes to that CSI
stream, windows + preprocesses it exactly like `pipeline/service.py`,
records spectrogram features labeled with `--zone` to
`services/pipeline/calibration_data/<zone_id>.npz` (overwriting any
previous run for that zone — recalibrating "fine-tunes" it), then reloads
every zone calibrated so far and refits `ZoneLocalizer` from scratch
(gradient boosting isn't incrementally trainable, but refitting is cheap
enough not to matter) to `services/pipeline/checkpoints/localizer.joblib`.
Needs at least 2 calibrated zones before a fit is possible — with just one,
it saves the samples and tells you to calibrate another.

```bash
pip install -e ".[dev,localize]"
python -m pipeline.calibrate --zone A1 --seconds 60
python -m pipeline.calibrate --zone A2 --seconds 60
# ...repeat per zone...

python -m pipeline --localizer-checkpoint checkpoints/localizer.joblib
```

**Live output**: `{timestamp, zones: [{zone_id, occupancy_probability},
...]}` — one entry per configured zone (0 probability for any zone not
yet calibrated), published on the `zones` topic whenever
`--localizer-checkpoint` is set. `services/pipeline/tests/test_integration_localization.py`
proves the full loop end to end: calibrate two zones from real replay
subprocesses, run the live `pipeline` service against a *fresh* recording
of each zone, and confirm its predictions converge on the right one.

## API and dashboard

`services/api` subscribes to all three of `services/pipeline`'s ZeroMQ
topics and merges them into one combined live state: `{timestamp,
presence, motion_intensity, count, confidence, zones}`. Since
`services/pipeline` processes presence/count/zones for the same CSI
window in the same loop iteration (when count/zones are enabled), events
sharing a `timestamp` **upsert into one SQLite row** rather than three —
see `services/api/src/store.js`. Served two ways: `GET /status` /
`GET /history?minutes=n` (REST), and a WebSocket at `/ws` that pushes the
merged state on every pipeline event (plus the current state immediately
on connect). `count`/`confidence`/`zones` stay `null` until their
checkpoints are configured (see the two sections above);
`presence`/`motion_intensity` are always live.

`dashboard` (React) renders that state: a people-count card, a
motion-intensity gauge, a room heatmap colored by per-zone occupancy
probability, and a 10-minute history chart — all fed by `services/api`'s
REST (initial values) and WebSocket (live updates), with automatic
reconnection if the socket drops. It's a **browser** app, so it reaches
`services/api` via the api container's *host* port mapping
(`http://localhost:3001`), never the Docker-internal service name — see
`dashboard/src/api.js`. Cards for fields that aren't populated yet (count,
zones) show an explicit "not loaded" state with the exact command to
enable them, rather than blank or fabricated data.

`docker compose up` (no profile flags, no manual steps) starts every
service, including `replay` streaming a synthetic scenario — so the
dashboard at **http://localhost:3000** shows real, live-changing presence/
motion-intensity data immediately. Count and zone data need a trained
counter checkpoint / calibrated localizer first (see the two sections
above) — checkpoints are gitignored (regenerate, don't commit), so a
fresh clone won't have them until you train/calibrate.

## Coding conventions

- **Python** (`services/ingest`, `services/pipeline`, `services/replay`):
  Python 3.11+, type hints on all function signatures and public
  attributes. Prefer `from __future__ import annotations` plus standard
  library `typing` constructs over comments describing types.
- **Node.js** (`services/api`): Node 20+, ES modules only (`"type": "module"`
  in `package.json`, `import`/`export` syntax — no `require`).
- **Dashboard**: React, ES modules, same JS conventions as `services/api`.
- Keep services decoupled: they communicate only over the network (UDP,
  HTTP, WebSocket) using the schemas documented in `docs/`, never via shared
  in-process imports across service boundaries.

## Standalone + docker compose rule

**Every service must be runnable two ways:**

1. **Standalone**, directly on the host (`python -m ...` / `npm start` /
   `npm run dev`), for fast local iteration.
2. **Via `docker-compose.yml`** at the repo root, for running the full
   pipeline end-to-end.

When adding or changing a service, keep both paths working — don't add a
dependency or piece of config that only works inside Docker (or only works
outside it) without updating the other path too.
