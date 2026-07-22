# Contributing to wifi-sense

Thanks for considering a contribution. This is a hobby-scale, open-source
project — please keep that in mind: PRs are reviewed as time allows, and
there's no SLA on issues.

## Before you start

- **Bug fixes / small improvements**: just open a PR.
- **New features, or anything touching the CSI frame schema or a service's
  public interface**: open an issue first to discuss the approach. This
  avoids wasted work on something that doesn't fit the project's
  direction — see [`docs/roadmap.md`](docs/roadmap.md) for known
  directions (multi-receiver tracking, TF-Lite edge inference, MQTT/Home
  Assistant integration) that already have rough shapes sketched out.
- **Firmware changes** (`firmware/esp32-csi`): especially welcome from
  anyone with real hardware to test on — the current firmware was written
  against Espressif's documented API surface but has not been
  compiled/flashed as part of this project's own development (see the
  "Verification status" note in `firmware/esp32-csi/README.md`). Reports
  of what did or didn't work on real boards are as valuable as code.

## Project structure

See the root [`README.md`](README.md) for the architecture diagram and
[`CLAUDE.md`](CLAUDE.md) for the full design rationale behind each piece
(why presence detection is rule-based but counting/localization are ML,
how the CSI frame schema is versioned, etc.) — read the relevant section
of `CLAUDE.md` before changing a service's internals, since a lot of
non-obvious decisions are documented there.

## Development setup

Every service is runnable two ways, and **both must keep working** for
any change you make:

1. **Standalone**, directly on the host — for fast local iteration. See
   each service's own `README.md` for its exact commands (they differ:
   `python -m ingest` vs. `npm start` vs. `idf.py build`).
2. **Via `docker compose up --build`** at the repo root — for testing the
   full pipeline end-to-end.

Don't add a dependency or config value that only works one of those two
ways without updating the other path too.

## Coding conventions

- **Python** (`services/ingest`, `services/pipeline`, `services/replay`,
  `datasets`): Python 3.11+, type hints on all function signatures and
  public attributes, `from __future__ import annotations` preferred over
  comment-based typing.
- **Node.js** (`services/api`): Node 20+, ES modules only (no `require`).
- **Dashboard**: React, ES modules, same JS conventions as `services/api`.
- **Firmware** (`firmware/esp32-csi`): standard ESP-IDF project
  conventions — see that directory's `README.md`.
- Services communicate **only over the network** (UDP, HTTP, WebSocket,
  ZeroMQ) using the schemas in `docs/`, never via shared in-process
  imports across service boundaries. This is what lets each service have
  its own language/runtime and be developed independently.
- Match the style already in the file you're editing over introducing a
  new convention — consistency within a service matters more than any
  single stylistic preference.

## The CSI frame schema

[`docs/csi-frame-schema.md`](docs/csi-frame-schema.md) is the contract
every producer (firmware, `services/replay`) and consumer
(`services/ingest`, `services/pipeline`) agrees on. If a change requires
altering this shape:

1. Update the schema doc first, and bump `schema_version`.
2. Update every producer and consumer to match, in the same PR.
3. Call out the breaking change prominently in the PR description.

## Tests

Each service has its own test suite — see its `README.md` for the exact
command, but in general:

```bash
# Python services (ingest, pipeline, replay) and datasets/
cd services/<name>   # or datasets/
pip install -e ".[dev]"   # pipeline also has [ml] and [localize] extras —
                           # see services/pipeline/README.md for when you need them
pytest

# Node services (api) and dashboard/
cd services/api      # or dashboard/
npm install
npm test             # dashboard has no test suite yet; `npm run build` is its CI check
```

CI (`.github/workflows/ci.yml`) runs all of these, plus a build of every
Docker image, on every PR — it's the same thing described above, so if it
passes locally it should pass in CI. Please add or update tests for any
behavior change; a PR that changes behavior with no test coverage is
harder to review and easier to regress later.

## Commit / PR process

- Keep PRs focused — one logical change per PR is easier to review than a
  bundle of unrelated fixes.
- Write commit messages and PR descriptions that explain *why*, not just
  *what* — the diff already shows what changed.
- Make sure CI is green before requesting review.
- By submitting a contribution, you agree it's licensed under this
  project's [Apache License 2.0](LICENSE), per that license's own terms
  on Contributions (section 5).

## Reporting issues

Bug reports: what you expected, what happened instead, and how to
reproduce it (which service, what input/config). For firmware issues,
include the chip variant (ESP32/S2/S3/C3 — see the compatibility table in
`firmware/esp32-csi/README.md`) and, if possible, serial console output.
