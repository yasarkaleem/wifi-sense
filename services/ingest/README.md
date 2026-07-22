# ingest

Listens on UDP for CSI frames (from the ESP32 firmware or `services/replay`),
validates them against
[`docs/csi-frame-schema.md`](../../docs/csi-frame-schema.md), buffers the
recent history in a ring buffer, and exposes accepted frames two ways:

1. A **ZeroMQ PUB socket** (`tcp://<pub-host>:<pub-port>`, topic `csi`) that
   downstream services (`services/pipeline`) subscribe to.
2. An optional **debug live waterfall/heatmap** of amplitude across
   subcarriers over time, as either a web page (`--plot web`, no extra
   dependencies) or a matplotlib window (`--plot matplotlib`).

Invalid datagrams (malformed JSON, schema violations, wrong array lengths)
are logged and dropped rather than crashing the service.

## Standalone

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ".[plot]" too for --plot matplotlib

python -m ingest --plot web
```

Then, in another terminal, point `services/replay` at it and open
`http://localhost:8090/` in a browser to watch the waterfall:

```bash
cd ../replay
python -m replay --scenario empty_room --target localhost:5566
# switch to --scenario one_person_walking and watch the heatmap change
```

### CLI flags

| Flag                | Default                 | Env var override           |
|-----------------------|---------------------------|-------------------------------|
| `--udp-host`           | `0.0.0.0`                  | `INGEST_UDP_HOST`              |
| `--udp-port`           | `5566`                     | `INGEST_UDP_PORT`              |
| `--pub-host`           | `0.0.0.0`                  | `INGEST_PUB_HOST`              |
| `--pub-port`           | `5567` (`0` disables)      | `INGEST_PUB_PORT`              |
| `--buffer-size`        | `200` frames                | `INGEST_BUFFER_SIZE`           |
| `--plot`               | `none` (`web`/`matplotlib`)| `INGEST_PLOT`                  |
| `--plot-host`          | `0.0.0.0`                  | `INGEST_PLOT_HOST`             |
| `--plot-port`          | `8090`                     | `INGEST_PLOT_PORT`             |
| `--plot-refresh-hz`    | `10`                       | `INGEST_PLOT_REFRESH_HZ`       |

## Via docker compose

```bash
docker compose up --build ingest
```

The compose service sets `INGEST_PLOT=web` and publishes port `8090`, so
`http://localhost:8090/` shows the live waterfall automatically. Add the
`replay` service (profile `replay`) to see it react to scenario changes:

```bash
docker compose --profile replay up --build ingest replay
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
