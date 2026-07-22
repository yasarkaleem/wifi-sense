import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createState } from './state.js';

test('starts with every field null', () => {
  const state = createState();
  assert.deepEqual(state.get(), {
    timestamp: null,
    presence: null,
    motion_intensity: null,
    count: null,
    confidence: null,
    zones: null,
  });
});

test('applyPresence sets presence/motion_intensity/timestamp only', () => {
  const state = createState();
  const merged = state.applyPresence({ timestamp: 100, presence: true, motion_intensity: 0.42 });

  assert.equal(merged.timestamp, 100);
  assert.equal(merged.presence, true);
  assert.equal(merged.motion_intensity, 0.42);
  assert.equal(merged.count, null);
  assert.equal(merged.zones, null);
});

test('applyCount sets count/confidence/timestamp only', () => {
  const state = createState();
  const merged = state.applyCount({ timestamp: 200, count: 2, confidence: 0.81 });

  assert.equal(merged.timestamp, 200);
  assert.equal(merged.count, 2);
  assert.equal(merged.confidence, 0.81);
  assert.equal(merged.presence, null);
});

test('applyZones sets zones/timestamp only', () => {
  const state = createState();
  const zones = [{ zone_id: 'A1', occupancy_probability: 0.9 }];
  const merged = state.applyZones({ timestamp: 300, zones });

  assert.equal(merged.timestamp, 300);
  assert.deepEqual(merged.zones, zones);
  assert.equal(merged.count, null);
});

test('fields from different topics accumulate rather than overwrite each other', () => {
  const state = createState();
  state.applyPresence({ timestamp: 1, presence: true, motion_intensity: 0.5 });
  state.applyCount({ timestamp: 2, count: 1, confidence: 0.7 });
  const merged = state.applyZones({ timestamp: 3, zones: [{ zone_id: 'A1', occupancy_probability: 1 }] });

  assert.equal(merged.presence, true);
  assert.equal(merged.motion_intensity, 0.5);
  assert.equal(merged.count, 1);
  assert.equal(merged.confidence, 0.7);
  assert.equal(merged.timestamp, 3); // most recent event's timestamp wins
});

test('get() returns a fresh copy each time, not a live reference', () => {
  const state = createState();
  const snapshot1 = state.get();
  state.applyPresence({ timestamp: 1, presence: true, motion_intensity: 1 });
  const snapshot2 = state.get();

  assert.equal(snapshot1.presence, null);
  assert.equal(snapshot2.presence, true);
});
