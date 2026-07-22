import { useMemo, useState } from 'react';

import { BASELINE, GRIDLINE, INK_MUTED, SURFACE } from '../colors.js';

const WIDTH = 600;
const HEIGHT = 220;
const PADDING = { top: 16, right: 16, bottom: 28, left: 32 };
const LINE_COLOR = '#3987e5'; // categorical slot 1, dark mode — single series, so no legend needed
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1];

function formatAgo(msAgo) {
  const totalSeconds = Math.max(0, Math.round(msAgo / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s ago`;
  return `${Math.round(totalSeconds / 60)}m ago`;
}

export default function HistoryChart({ history }) {
  const [hoverIndex, setHoverIndex] = useState(null);

  const points = useMemo(
    () => history.filter((h) => h.motion_intensity !== null && h.motion_intensity !== undefined),
    [history],
  );

  const innerWidth = WIDTH - PADDING.left - PADDING.right;
  const innerHeight = HEIGHT - PADDING.top - PADDING.bottom;

  if (points.length < 2) {
    return (
      <section className="card history-chart" aria-label="Motion intensity history">
        <h2 className="card__title">Motion intensity — last 10 minutes</h2>
        <p className="card__caption card__caption--muted">collecting data…</p>
      </section>
    );
  }

  const minTs = points[0].timestamp;
  const maxTs = points[points.length - 1].timestamp;
  const spanUs = Math.max(maxTs - minTs, 1);

  const xFor = (ts) => PADDING.left + ((ts - minTs) / spanUs) * innerWidth;
  const yFor = (v) => PADDING.top + (1 - v) * innerHeight;

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xFor(p.timestamp).toFixed(1)} ${yFor(p.motion_intensity).toFixed(1)}`)
    .join(' ');

  function handleMouseMove(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const relativeX = ((event.clientX - rect.left) / rect.width) * WIDTH;
    let nearest = 0;
    let nearestDist = Infinity;
    points.forEach((p, i) => {
      const dist = Math.abs(xFor(p.timestamp) - relativeX);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = i;
      }
    });
    setHoverIndex(nearest);
  }

  const hovered = hoverIndex !== null ? points[hoverIndex] : null;
  const nowMs = Date.now();

  return (
    <section className="card history-chart" aria-label="Motion intensity history">
      <h2 className="card__title">Motion intensity — last 10 minutes</h2>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="history-chart__svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIndex(null)}
        role="img"
        aria-label="Line chart of motion intensity over the last 10 minutes"
      >
        {Y_TICKS.map((tick) => (
          <line
            key={tick}
            x1={PADDING.left}
            x2={WIDTH - PADDING.right}
            y1={yFor(tick)}
            y2={yFor(tick)}
            stroke={GRIDLINE}
            strokeWidth={1}
          />
        ))}
        {Y_TICKS.map((tick) => (
          <text key={tick} x={PADDING.left - 6} y={yFor(tick) + 4} textAnchor="end" className="history-chart__axis-label">
            {tick}
          </text>
        ))}
        <line
          x1={PADDING.left}
          x2={WIDTH - PADDING.right}
          y1={HEIGHT - PADDING.bottom}
          y2={HEIGHT - PADDING.bottom}
          stroke={BASELINE}
          strokeWidth={1}
        />
        <text x={PADDING.left} y={HEIGHT - 8} textAnchor="start" className="history-chart__axis-label">
          10m ago
        </text>
        <text x={WIDTH - PADDING.right} y={HEIGHT - 8} textAnchor="end" className="history-chart__axis-label">
          now
        </text>

        <path d={linePath} fill="none" stroke={LINE_COLOR} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

        {hovered && (
          <>
            <line
              x1={xFor(hovered.timestamp)}
              x2={xFor(hovered.timestamp)}
              y1={PADDING.top}
              y2={HEIGHT - PADDING.bottom}
              stroke={INK_MUTED}
              strokeWidth={1}
              strokeDasharray="3,3"
            />
            <circle
              cx={xFor(hovered.timestamp)}
              cy={yFor(hovered.motion_intensity)}
              r={4}
              fill={LINE_COLOR}
              stroke={SURFACE}
              strokeWidth={2}
            />
          </>
        )}
      </svg>
      <div className="history-chart__tooltip" aria-live="polite">
        {hovered ? (
          <>
            <strong>{Math.round(hovered.motion_intensity * 100)}%</strong>
            <span>{formatAgo(nowMs - hovered.timestamp / 1000)}</span>
          </>
        ) : (
          <span className="card__caption--muted">hover the chart for details</span>
        )}
      </div>
    </section>
  );
}
