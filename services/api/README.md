# api

Subscribes to `services/pipeline`'s ZeroMQ PUB socket (`presence`/`count`/
`zones` topics), merges events into one combined live state, persists a
history of it to SQLite, and serves both over REST and WebSocket for
`dashboard/`.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /status` | The current merged state: `{timestamp, presence, motion_intensity, count, confidence, zones}` |
| `GET /history?minutes=n` | `{minutes, count, snapshots: [...]}` — snapshots from the last `n` minutes (default 10), oldest first |
| `GET /healthz` | `{ok: true}` |
| `WS /ws` | Pushes the merged state on every pipeline event; sends the current state immediately on connect |

Any field pipeline hasn't published yet is `null` — `count`/`confidence`
stay `null` unless `services/pipeline` has a `--counter-checkpoint`
configured, and `zones` stays `null` unless it has a
`--localizer-checkpoint` configured (see `../pipeline/README.md`).
`presence`/`motion_intensity` are always live, since presence detection
has no ML dependency.

## Storage (`src/store.js`)

SQLite, one row per CSI window `timestamp_us`. `services/pipeline`
processes presence/count/zones for the same window in the same loop
iteration (when all three are enabled), so events sharing a `timestamp_us`
**upsert into the same row** instead of three separate ones — a history
query returns one clean combined reading per point in time, not three
staggered partial ones. Rows older than `API_RETENTION_MINUTES` (default
24h) are pruned periodically.

## Standalone

```bash
npm install
API_SUB_HOST=localhost API_SUB_PORT=5568 npm start
```

Env vars (all optional): `API_SUB_HOST`/`API_SUB_PORT` (pipeline's PUB,
default `localhost`/`5568`), `API_HTTP_HOST`/`API_HTTP_PORT` (default
`0.0.0.0`/`3001`), `API_DB_PATH` (default `./data/wifi-sense.db`),
`API_CORS_ORIGIN` (default `*`), `API_RETENTION_MINUTES` (default 1440),
`API_PRUNE_INTERVAL_MS` (default 300000).

## Via docker compose

```bash
docker compose up --build ingest pipeline api
```

## Tests

```bash
npm install
npm test
```

Covers `state.js` (event merging) and `store.js` (SQLite upsert/history/
prune semantics) with Node's built-in test runner — no extra test
dependency. The ZeroMQ subscriber and HTTP/WS wiring are exercised by
running the real service (see the root `docker-compose.yml` and this
project's other services' `tests/test_integration_*.py` for the pattern).
