# Roadmap

Where wifi-sense is headed. Nothing here is scheduled or promised — it's a
statement of direction so contributors can see where new work fits (or
doesn't) before opening a PR. If you want to work on one of these, open an
issue first to coordinate; see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Multi-receiver tracking

Today, `services/pipeline` consumes CSI from a single receiver ESP32 and
localizes to whichever zone its `ZoneLocalizer` fingerprint matches best —
one vantage point, one set of subcarrier disturbances. Multiple receivers
placed around a room see the same disturbance from different angles, which
should sharpen both presence detection (less sensitive to a single blind
spot) and zone localization (triangulation instead of single-point
fingerprinting), and is a prerequisite for tracking more than one person
distinctly rather than just reporting an aggregate count.

Rough shape: each receiver keeps its own `source_mac` in
[`docs/csi-frame-schema.md`](csi-frame-schema.md), so `services/ingest`
already disambiguates streams at the wire level. The work is upstream of
that — `services/pipeline` windowing/preprocessing is currently written
assuming one CSI stream; extending it to fuse `N` time-aligned streams
(same window, multiple `source_mac`s) into one feature vector, and
retraining `ZoneLocalizer`/`PeopleCounterCNN` on multi-receiver features,
without breaking the single-receiver path that most home deployments will
still use.

## TF-Lite edge inference

`services/pipeline`'s ML models (`PeopleCounterCNN`, `ZoneLocalizer`) run
today as a Python service on whatever host runs `docker compose` — a
Raspberry Pi or always-on home server. Running inference directly on the
ESP32 receiver would cut a network hop and let the firmware ship a
presence/count signal even if `services/pipeline` is offline, at the cost
of the ESP32's far tighter memory/compute budget than a host machine.

Rough shape: export `PeopleCounterCNN` (already tiny — ~5,600 parameters,
see the root `CLAUDE.md`'s "ML-based people counting" section) to
TF-Lite Micro, port the spectrogram feature extraction
(`pipeline/features/spectrogram.py`) to fixed-point C suitable for
`firmware/esp32-csi`, and add an on-device inference path there
alongside (not replacing) today's raw-CSI-over-UDP mode — some
deployments will still want raw CSI for `services/pipeline`'s
richer/upgradeable models. `ZoneLocalizer` (gradient boosting) is a worse
fit for this than the CNN and likely stays host-side.

## MQTT / Home Assistant integration

`services/api` currently exposes REST + WebSocket for `dashboard/`. Home
automation setups overwhelmingly integrate over MQTT rather than polling a
custom REST API, and a proper Home Assistant integration (ideally via
MQTT discovery, so entities show up automatically) is the most-requested
way to get wifi-sense's occupancy/zone signal into an existing smart-home
setup — turning on lights, adjusting HVAC by zone, etc.

Rough shape: a new consumer alongside `services/api` (or a mode of it)
that subscribes to the same `services/pipeline` ZeroMQ topics
(`presence`/`count`/`zones` — see the root `CLAUDE.md`'s "Architecture"
section) and republishes to an MQTT broker using Home Assistant's MQTT
discovery payload format, so a person-count sensor and per-zone occupancy
binary sensors appear in Home Assistant with no manual YAML.

## Contributing to the roadmap

These are directions, not a queue — pick one, open an issue describing
your approach before investing significant time, and see
[`CONTRIBUTING.md`](../CONTRIBUTING.md) for the general workflow
(standalone + docker compose rule, test expectations, etc.). Ideas not
listed here are welcome too; this file just tracks the ones already under
discussion.
