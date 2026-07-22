// Talks to services/api. Runs in the browser (not in a Docker container),
// so it must use a host-reachable address — the api container's HOST port
// mapping (see ../../docker-compose.yml), not its Docker-internal service
// name. VITE_API_URL lets this be overridden at build/dev-server start
// time; the default matches both `docker compose up` and plain
// `npm run dev` (api's own standalone default port is the same, 3001).

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:3001';

export const WS_URL = `${API_BASE.replace(/^http/, 'ws')}/ws`;

async function getJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return res.json();
}

export function fetchStatus() {
  return getJson('/status');
}

export function fetchHistory(minutes = 10) {
  return getJson(`/history?minutes=${encodeURIComponent(minutes)}`);
}
