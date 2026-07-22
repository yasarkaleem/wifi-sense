# replay

Streams CSI frames over UDP in the same wire format the ESP32 firmware
uses ([`docs/csi-frame-schema.md`](../../docs/csi-frame-schema.md)), so
the rest of the pipeline can be developed and tested without physical
hardware — either synthetic scenarios or a real dataset recording
converted by [`../../datasets/download.py`](../../datasets/download.py).

## Scenarios (synthetic)

Scenarios are defined in [`src/replay/scenarios.yaml`](src/replay/scenarios.yaml):

- `empty_room` — no people, sensor noise only.
- `one_person_walking` — one person, periodic gait-induced amplitude
  disturbance localized around a range of subcarriers.
- `two_people` — two people moving independently at different gait
  frequencies, each localized around a different subcarrier range.
- `three_people` — three people, same idea, used as the "3+" class when
  training `services/pipeline`'s people counter.

Each person's disturbance is a sinusoid (`walk_frequency_hz`) weighted by a
Gaussian centered on `subcarrier_center` with width `subcarrier_spread`, so
the effect is strongest near that subcarrier and fades with distance from
it. Point `--scenarios-file` at your own YAML to define custom scenarios
without touching code.

## Dataset mode (real data)

Streams a `.npz` recording produced by
[`../../datasets/download.py`](../../datasets/download.py) (e.g. converted
UT-HAR sessions) instead of synthetic generation:

```bash
python -m replay --dataset ut-har --file ../../datasets/ut-har/bed_1.npz --target localhost:5566
```

Pacing defaults to the recording's own inter-frame timestamps, faithfully
reproducing the original capture's timing (gaps included); pass `--rate`
to force a fixed rate instead. `--loop` restarts the recording when it
ends. Every emitted frame's `timestamp_us` is the actual send time (not
the recording's original historical timestamp) — matching synthetic mode
and "capture time on the sender" in the schema doc. The recording's
per-frame activity label (if any) is a training-only field, never sent
over the wire.

`--scenario`/`--scenarios-file`/`--seed` are ignored when `--dataset` is
given.

## Standalone

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m replay --scenario one_person_walking --rate 100 --target localhost:5566
```

CLI flags:

| Flag               | Default                  | Env var override        |
|---------------------|---------------------------|--------------------------|
| `--scenario`        | `empty_room`               | `REPLAY_SCENARIO`        |
| `--rate`             | synthetic: `100`Hz; dataset: original timing | `REPLAY_RATE`             |
| `--target`           | `localhost:5566`           | `REPLAY_TARGET`           |
| `--scenarios-file`   | packaged `scenarios.yaml`  | `REPLAY_SCENARIOS_FILE`   |
| `--seed`             | unseeded                   | —                         |
| `--dataset`          | unset (synthetic mode)     | `REPLAY_DATASET`          |
| `--file`             | unset                      | `REPLAY_FILE`             |
| `--loop`             | off                        | —                         |

## Via docker compose

`replay` is opt-in (profile `replay`) since it's a hardware emulator, not
part of the always-on pipeline:

```bash
docker compose --profile replay up --build replay
```

By default it targets the `ingest` service on the docker network
(`ingest:5566`); see the `replay` service's `environment` block in
`docker-compose.yml` to change the scenario/rate. `../../datasets/` is
mounted read-only at `/datasets`, so a converted recording is reachable
in-container too — set `REPLAY_DATASET=ut-har` and
`REPLAY_FILE=/datasets/ut-har/bed_1.npz` to use it instead of a synthetic
scenario.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
