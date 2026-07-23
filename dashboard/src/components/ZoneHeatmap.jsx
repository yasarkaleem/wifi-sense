import { useEffect, useRef, useState } from 'react';

import { sequentialColor } from '../colors.js';
import { mostLikelyZone, parseZoneId, weightedCentroid } from '../zones.js';

// How many past centroid positions the fading trail keeps.
const TRAIL_LENGTH = 10;

// Percentage position (within the grid) of a cell's center, given its
// 0-indexed {row, col} and the grid's row/column counts. Ignores the
// grid's `gap` (a few px against 60px+ cells) — close enough for this
// visualization.
function cellCenterPercent({ row, col }, rows, cols) {
  return { left: `${((col + 0.5) / cols) * 100}%`, top: `${((row + 0.5) / rows) * 100}%` };
}

export default function ZoneHeatmap({ zones, timestamp }) {
  const [trail, setTrail] = useState([]); // oldest first, [{row, col}, ...]
  const lastTimestampRef = useRef(null);

  const centroid = weightedCentroid(zones);

  useEffect(() => {
    if (timestamp == null || timestamp === lastTimestampRef.current) return;
    lastTimestampRef.current = timestamp;
    if (centroid) {
      setTrail((prev) => [...prev, centroid].slice(-TRAIL_LENGTH));
    }
    // Only re-run when a genuinely new window's timestamp arrives —
    // `centroid` is recomputed fresh every render and shouldn't itself
    // retrigger this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timestamp]);

  if (!zones || zones.length === 0) {
    return (
      <section className="card zone-heatmap" aria-label="Room occupancy heatmap">
        <h2 className="card__title">Room heatmap</h2>
        <p className="card__caption card__caption--muted">
          No zone data yet — calibrate the localizer with{' '}
          <code>python -m pipeline.calibrate</code> and set{' '}
          <code>--localizer-checkpoint</code> on services/pipeline (see CLAUDE.md).
        </p>
      </section>
    );
  }

  const parsed = zones.map((z) => ({ ...z, pos: parseZoneId(z.zone_id) })).filter((z) => z.pos !== null);
  const rows = Math.max(...parsed.map((z) => z.pos.row)) + 1;
  const cols = Math.max(...parsed.map((z) => z.pos.col)) + 1;

  const grid = Array.from({ length: rows }, () => Array(cols).fill(null));
  for (const zone of parsed) {
    grid[zone.pos.row][zone.pos.col] = zone;
  }

  const best = mostLikelyZone(zones);

  return (
    <section className="card zone-heatmap" aria-label="Room occupancy heatmap">
      <h2 className="card__title">Room heatmap</h2>
      <div className="zone-heatmap__stage">
        <div
          className="zone-heatmap__grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)` }}
        >
          {grid.flatMap((row, r) =>
            row.map((cell, c) => (
              <div
                key={`${r}-${c}`}
                className="zone-heatmap__cell"
                style={{ background: cell ? sequentialColor(cell.occupancy_probability) : 'transparent' }}
                title={
                  cell ? `${cell.zone_id}: ${Math.round(cell.occupancy_probability * 100)}% occupied` : 'unconfigured zone'
                }
              >
                {cell && (
                  <>
                    <span className="zone-heatmap__label">{cell.zone_id}</span>
                    <span className="zone-heatmap__pct">{Math.round(cell.occupancy_probability * 100)}%</span>
                  </>
                )}
              </div>
            )),
          )}
        </div>

        {/* Fading trail, oldest (most transparent) first so newer dots
            paint on top. Index-keyed on purpose: as the array shifts
            (oldest point dropped, newest appended) each slot's position
            updates in place, and the CSS transition on left/top makes
            that read as the trail "flowing" rather than resetting. */}
        {trail.map((pos, i) => (
          <div
            key={i}
            className="zone-heatmap__trail-dot"
            style={{ ...cellCenterPercent(pos, rows, cols), opacity: (0.5 * (i + 1)) / trail.length }}
          />
        ))}

        {centroid && <div className="zone-heatmap__dot" style={cellCenterPercent(centroid, rows, cols)} />}
      </div>

      {best && (
        <p className="zone-heatmap__legend">
          Most likely: <strong>{best.zone_id}</strong> ({Math.round(best.occupancy_probability * 100)}%)
        </p>
      )}
    </section>
  );
}
