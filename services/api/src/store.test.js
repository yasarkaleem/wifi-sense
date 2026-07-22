import assert from 'node:assert/strict';
import { test } from 'node:test';

import { openStore } from './store.js';

test('recordPresence then history returns the snapshot', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;
  store.recordPresence({ timestamp: nowUs, presence: true, motion_intensity: 0.6 });

  const rows = store.history(10);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].timestamp, nowUs);
  assert.equal(rows[0].presence, true);
  assert.equal(rows[0].motion_intensity, 0.6);
  assert.equal(rows[0].count, null);
  assert.equal(rows[0].zones, null);

  store.close();
});

test('events sharing a timestamp_us merge into one row instead of three', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;

  store.recordPresence({ timestamp: nowUs, presence: false, motion_intensity: 0.1 });
  store.recordCount({ timestamp: nowUs, count: 2, confidence: 0.75 });
  store.recordZones({ timestamp: nowUs, zones: [{ zone_id: 'A1', occupancy_probability: 0.9 }] });

  const rows = store.history(10);
  assert.equal(rows.length, 1, 'expected one merged row, not three');
  assert.equal(rows[0].presence, false);
  assert.equal(rows[0].motion_intensity, 0.1);
  assert.equal(rows[0].count, 2);
  assert.equal(rows[0].confidence, 0.75);
  assert.deepEqual(rows[0].zones, [{ zone_id: 'A1', occupancy_probability: 0.9 }]);

  store.close();
});

test('recording count for a timestamp does not clobber that timestamp\'s existing presence fields', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;

  store.recordPresence({ timestamp: nowUs, presence: true, motion_intensity: 0.8 });
  store.recordCount({ timestamp: nowUs, count: 3, confidence: 0.5 });

  const rows = store.history(10);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].presence, true, 'presence should survive a later count-only upsert for the same timestamp');
  assert.equal(rows[0].motion_intensity, 0.8);
  assert.equal(rows[0].count, 3);

  store.close();
});

test('history excludes snapshots older than the requested window', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;
  const twentyMinAgoUs = nowUs - 20 * 60 * 1_000_000;

  store.recordPresence({ timestamp: twentyMinAgoUs, presence: true, motion_intensity: 0.3 });
  store.recordPresence({ timestamp: nowUs, presence: true, motion_intensity: 0.4 });

  const last10 = store.history(10);
  assert.equal(last10.length, 1);
  assert.equal(last10[0].timestamp, nowUs);

  const last30 = store.history(30);
  assert.equal(last30.length, 2);

  store.close();
});

test('history returns results ordered oldest first', () => {
  const store = openStore(':memory:');
  const base = Date.now() * 1000;

  store.recordPresence({ timestamp: base + 2000, presence: true, motion_intensity: 0.2 });
  store.recordPresence({ timestamp: base, presence: true, motion_intensity: 0.1 });
  store.recordPresence({ timestamp: base + 1000, presence: true, motion_intensity: 0.15 });

  const rows = store.history(10);
  assert.deepEqual(
    rows.map((r) => r.timestamp),
    [base, base + 1000, base + 2000],
  );

  store.close();
});

test('null presence/count fields round-trip as null, not 0/false', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;
  store.recordCount({ timestamp: nowUs, count: null, confidence: null });

  const rows = store.history(10);
  assert.equal(rows[0].count, null);
  assert.equal(rows[0].confidence, null);
  assert.equal(rows[0].presence, null);

  store.close();
});

test('prune removes rows older than the retention window and reports how many', () => {
  const store = openStore(':memory:');
  const nowUs = Date.now() * 1000;
  const oldUs = nowUs - 2 * 24 * 60 * 60 * 1_000_000; // 2 days ago

  store.recordPresence({ timestamp: oldUs, presence: true, motion_intensity: 0.1 });
  store.recordPresence({ timestamp: nowUs, presence: true, motion_intensity: 0.2 });

  const removed = store.prune(24 * 60); // keep last 24h
  assert.equal(removed, 1);

  const rows = store.history(24 * 60);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].timestamp, nowUs);

  store.close();
});
