import { GRIDLINE, sequentialColor } from '../colors.js';

const SIZE = 200;
const STROKE = 16;
const R = SIZE / 2 - STROKE;
const CX = SIZE / 2;
const CY = SIZE / 2;
const ARC_PATH = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;

export default function MotionGauge({ value, presence }) {
  const hasValue = value !== null && value !== undefined;
  const clamped = hasValue ? Math.max(0, Math.min(1, value)) : 0;

  return (
    <section className="card motion-gauge" aria-label="Motion intensity">
      <h2 className="card__title">Motion intensity</h2>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE / 2 + STROKE}`}
        className="motion-gauge__svg"
        role="img"
        aria-label={hasValue ? `Motion intensity ${Math.round(clamped * 100)} percent` : 'Motion intensity unknown'}
      >
        <path d={ARC_PATH} fill="none" stroke={GRIDLINE} strokeWidth={STROKE} strokeLinecap="round" pathLength={100} />
        {hasValue && (
          <path
            d={ARC_PATH}
            fill="none"
            stroke={sequentialColor(clamped)}
            strokeWidth={STROKE}
            strokeLinecap="round"
            pathLength={100}
            strokeDasharray={100}
            strokeDashoffset={100 * (1 - clamped)}
            className="motion-gauge__fill"
          />
        )}
      </svg>
      <div className="motion-gauge__value">{hasValue ? `${Math.round(clamped * 100)}%` : '—'}</div>
      <p className="card__caption card__caption--muted">
        {presence === null ? 'waiting for data…' : presence ? 'presence detected' : 'no motion detected'}
      </p>
    </section>
  );
}
