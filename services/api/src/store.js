// SQLite-backed history. One row per window timestamp_us — presence,
// count, and zones events for the SAME window (as pipeline/service.py
// processes them together, sharing one timestamp_us) upsert into the
// SAME row instead of three separate rows, so a history query returns
// one clean combined reading per point in time.

import fs from 'node:fs';
import path from 'node:path';

import Database from 'better-sqlite3';

const SCHEMA = `
  CREATE TABLE IF NOT EXISTS snapshots (
    timestamp_us INTEGER PRIMARY KEY,
    presence INTEGER,
    motion_intensity REAL,
    count INTEGER,
    confidence REAL,
    zones TEXT,
    received_at_us INTEGER NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp_us ON snapshots(timestamp_us);
`;

function rowToSnapshot(row) {
  return {
    timestamp: row.timestamp_us,
    presence: row.presence === null ? null : Boolean(row.presence),
    motion_intensity: row.motion_intensity,
    count: row.count,
    confidence: row.confidence,
    zones: row.zones ? JSON.parse(row.zones) : null,
  };
}

export function openStore(dbPath) {
  if (dbPath !== ':memory:') {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  }
  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.exec(SCHEMA);

  const upsertPresence = db.prepare(`
    INSERT INTO snapshots (timestamp_us, presence, motion_intensity, received_at_us)
    VALUES (@timestamp_us, @presence, @motion_intensity, @received_at_us)
    ON CONFLICT(timestamp_us) DO UPDATE SET
      presence = excluded.presence,
      motion_intensity = excluded.motion_intensity,
      received_at_us = excluded.received_at_us
  `);

  const upsertCount = db.prepare(`
    INSERT INTO snapshots (timestamp_us, count, confidence, received_at_us)
    VALUES (@timestamp_us, @count, @confidence, @received_at_us)
    ON CONFLICT(timestamp_us) DO UPDATE SET
      count = excluded.count,
      confidence = excluded.confidence,
      received_at_us = excluded.received_at_us
  `);

  const upsertZones = db.prepare(`
    INSERT INTO snapshots (timestamp_us, zones, received_at_us)
    VALUES (@timestamp_us, @zones, @received_at_us)
    ON CONFLICT(timestamp_us) DO UPDATE SET
      zones = excluded.zones,
      received_at_us = excluded.received_at_us
  `);

  const selectSince = db.prepare(`
    SELECT * FROM snapshots WHERE timestamp_us >= ? ORDER BY timestamp_us ASC
  `);

  const deleteBefore = db.prepare(`DELETE FROM snapshots WHERE timestamp_us < ?`);

  return {
    recordPresence(event) {
      upsertPresence.run({
        timestamp_us: event.timestamp,
        presence: event.presence === null || event.presence === undefined ? null : event.presence ? 1 : 0,
        motion_intensity: event.motion_intensity ?? null,
        received_at_us: Date.now() * 1000,
      });
    },

    recordCount(event) {
      upsertCount.run({
        timestamp_us: event.timestamp,
        count: event.count ?? null,
        confidence: event.confidence ?? null,
        received_at_us: Date.now() * 1000,
      });
    },

    recordZones(event) {
      upsertZones.run({
        timestamp_us: event.timestamp,
        zones: event.zones ? JSON.stringify(event.zones) : null,
        received_at_us: Date.now() * 1000,
      });
    },

    /** Snapshots from the last `minutes` minutes, oldest first. */
    history(minutes) {
      const sinceUs = Date.now() * 1000 - minutes * 60 * 1_000_000;
      return selectSince.all(sinceUs).map(rowToSnapshot);
    },

    /** Deletes rows older than `retentionMinutes` — called periodically, not on every write. */
    prune(retentionMinutes) {
      const cutoffUs = Date.now() * 1000 - retentionMinutes * 60 * 1_000_000;
      const result = deleteBefore.run(cutoffUs);
      return result.changes;
    },

    close() {
      db.close();
    },
  };
}
