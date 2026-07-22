import { sequentialColor } from '../colors.js';
import { parseZoneId } from '../zones.js';

export default function ZoneHeatmap({ zones }) {
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

  return (
    <section className="card zone-heatmap" aria-label="Room occupancy heatmap">
      <h2 className="card__title">Room heatmap</h2>
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
              title={cell ? `${cell.zone_id}: ${Math.round(cell.occupancy_probability * 100)}% occupied` : 'unconfigured zone'}
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
    </section>
  );
}
