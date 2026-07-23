# dashboard

React app: live people count, a motion-intensity gauge, a room heatmap
colored by per-zone occupancy probability (with an animated dot tracking
the estimated person position and a fading trail behind it), and a
10-minute motion-intensity history chart — all fed by `services/api` over
REST (initial values) and WebSocket (live updates). Dark, minimal theme
(see `src/colors.js` / `src/styles.css`, following the `dataviz` design
skill's dark-surface palette).

## How it stays live (`src/useLiveData.js`)

On mount: `GET /status` (current state) and `GET /history?minutes=10`
(backfill) run once. Then a WebSocket to `/ws` extends both live — every
push updates the current-state cards and appends to the rolling 10-minute
history buffer (deduped/merged by `timestamp`, oldest entries older than
10 minutes dropped). If the socket drops (e.g. `api` restarting), the
badge in the header flips to "Reconnecting…" and it retries automatically
— no page refresh needed.

## Graceful degradation

`count`/`confidence`/`zones` are `null` until `services/pipeline` has a
`--counter-checkpoint` / `--localizer-checkpoint` configured (see
`../CLAUDE.md`) — the People Count card and Room Heatmap show an explicit
"not loaded yet" state with the exact command to enable them, rather than
blank or fake data. `presence`/`motion_intensity` have no ML dependency
and are always live. Via `docker compose up`, both are configured out of
the box (see "Via docker compose" below), so this state is mainly seen
running `services/pipeline` standalone without those flags.

## Room heatmap: the animated dot (`src/components/ZoneHeatmap.jsx`, `src/zones.js`)

On top of the existing per-zone grid coloring, a dot renders at the
probability-weighted centroid of `zones` (`weightedCentroid()` in
`zones.js` — each zone's grid position weighted by its
`occupancy_probability`, normalized), moved via a CSS `transition` on
`left`/`top` (~500ms, matching the tween timing already used elsewhere in
this dashboard, e.g. `.motion-gauge__fill`) so it glides between positions
rather than jumping. The last ~10 centroid positions are kept in
component state and rendered behind it as smaller, progressively more
transparent dots — a fading trail of recent movement, plain CSS opacity,
no animation library. A small legend line under the grid shows the
current single most-likely zone and its probability (`mostLikelyZone()` in
`zones.js`). The dot hides itself when total probability across all zones
is ~0 (nobody localized anywhere).

## Where `api`'s URL comes from (`src/api.js`)

This runs in the **browser**, not in a Docker container, so it must reach
`api` via a host-reachable address (`http://localhost:3001`, matching
`../docker-compose.yml`'s port mapping) — not the Docker-internal service
name `api`. `VITE_API_URL` overrides this at dev-server-start time if
needed; the default already matches both `docker compose up` and plain
`npm run dev` (api's own standalone default port is the same, 3001).

## Standalone

```bash
npm install
npm run dev
```

Requires `services/api` running (see `../services/api/README.md`), which
in turn requires `services/ingest`/`services/pipeline` (and, for live
synthetic data, `services/replay`) — or just run the whole stack via
docker compose below.

## Via docker compose

```bash
docker compose up --build
```

Then open **http://localhost:3000**. `replay` streams the
`one_person_walking_path` trajectory scenario by default (see
`../services/replay/README.md`) and `pipeline` loads the committed
"walking person" demo checkpoints automatically (see
`../services/pipeline/README.md`) — so every card, including the Room
Heatmap's animated dot, shows real changing data immediately with no
manual steps. Set `REPLAY_SCENARIO=two_people_walking_paths` to see the
People Count card switch to 2.

## Design notes

Hand-rolled SVG (no charting library) for the gauge and history chart,
following the `dataviz` skill: thin 2px lines, a single hue for the
gauge/heatmap's sequential (magnitude) encoding — darkest = near-zero
(recedes into the dark surface), brightest = high magnitude, per the
skill's "flips anchor in dark" rule — recessive gridlines, and a
crosshair + tooltip on hover for the history chart.
