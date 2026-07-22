// Entry point: subscribes to services/pipeline's ZeroMQ PUB socket,
// merges events into in-memory state + SQLite history, and serves both
// over REST (GET /status, GET /history) and WebSocket (/ws) — see ../README.md.

import { loadConfig } from './config.js';
import { createApiServer } from './server.js';
import { createState } from './state.js';
import { openStore } from './store.js';
import { subscribeToPipeline } from './subscriber.js';

async function main() {
  const config = loadConfig();
  const state = createState();
  const store = openStore(config.dbPath);

  const { httpServer, broadcast } = createApiServer({
    state,
    store,
    corsOrigin: config.corsOrigin,
  });

  httpServer.listen(config.httpPort, config.httpHost, () => {
    console.log(`api: listening on http://${config.httpHost}:${config.httpPort} (WebSocket at /ws)`);
  });

  const pruneInterval = setInterval(() => {
    const removed = store.prune(config.retentionMinutes);
    if (removed > 0) {
      console.log(`api: pruned ${removed} snapshot(s) older than ${config.retentionMinutes} minutes`);
    }
  }, config.pruneIntervalMs);
  pruneInterval.unref();

  const controller = new AbortController();
  const shutdown = () => {
    console.log('api: shutting down...');
    controller.abort();
    clearInterval(pruneInterval);
    httpServer.close();
    store.close();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  await subscribeToPipeline({
    host: config.subHost,
    port: config.subPort,
    signal: controller.signal,
    onEvent(topic, event) {
      let merged;
      switch (topic) {
        case 'presence':
          merged = state.applyPresence(event);
          store.recordPresence(event);
          break;
        case 'count':
          merged = state.applyCount(event);
          store.recordCount(event);
          break;
        case 'zones':
          merged = state.applyZones(event);
          store.recordZones(event);
          break;
        default:
          return;
      }
      broadcast(merged);
    },
  });
}

main().catch((err) => {
  console.error('api: fatal error:', err);
  process.exit(1);
});
