// HTTP (REST) + WebSocket server. WebSocket is attached to the same
// http.Server instance as Express, at /ws.

import { createServer } from 'node:http';

import cors from 'cors';
import express from 'express';
import { WebSocketServer } from 'ws';

const MIN_HISTORY_MINUTES = 1;
const MAX_HISTORY_MINUTES = 7 * 24 * 60; // 1 week

export function createApiServer({ state, store, corsOrigin }) {
  const app = express();
  app.use(cors({ origin: corsOrigin }));

  app.get('/status', (_req, res) => {
    res.json(state.get());
  });

  app.get('/history', (req, res) => {
    const raw = req.query.minutes;
    const minutes = raw === undefined ? 10 : Number(raw);
    if (!Number.isFinite(minutes) || minutes < MIN_HISTORY_MINUTES || minutes > MAX_HISTORY_MINUTES) {
      res.status(400).json({
        error: `minutes must be a number between ${MIN_HISTORY_MINUTES} and ${MAX_HISTORY_MINUTES}`,
      });
      return;
    }
    const snapshots = store.history(minutes);
    res.json({ minutes, count: snapshots.length, snapshots });
  });

  app.get('/healthz', (_req, res) => res.json({ ok: true }));

  const httpServer = createServer(app);
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });

  wss.on('connection', (ws) => {
    // Immediately bring a newly-connected client up to date, rather than
    // leaving it blank until the next pipeline event arrives.
    ws.send(JSON.stringify(state.get()));
  });

  function broadcast(payload) {
    const data = JSON.stringify(payload);
    for (const client of wss.clients) {
      if (client.readyState === client.OPEN) {
        client.send(data);
      }
    }
  }

  return { app, httpServer, wss, broadcast };
}
