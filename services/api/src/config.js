// Environment-variable configuration, following the same PIPELINE_*/
// REPLAY_*-style convention the Python services use (see ../../../CLAUDE.md).

export function loadConfig(env = process.env) {
  return {
    subHost: env.API_SUB_HOST || 'localhost',
    subPort: Number(env.API_SUB_PORT || 5568),
    httpHost: env.API_HTTP_HOST || '0.0.0.0',
    httpPort: Number(env.API_HTTP_PORT || 3001),
    dbPath: env.API_DB_PATH || './data/wifi-sense.db',
    corsOrigin: env.API_CORS_ORIGIN || '*',
    // How long history rows are kept before being pruned (see store.js).
    retentionMinutes: Number(env.API_RETENTION_MINUTES || 24 * 60),
    pruneIntervalMs: Number(env.API_PRUNE_INTERVAL_MS || 5 * 60 * 1000),
  };
}
