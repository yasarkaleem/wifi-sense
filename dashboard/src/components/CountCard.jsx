function formatCount(count) {
  if (count === null || count === undefined) return '—';
  return count >= 3 ? '3+' : String(count);
}

export default function CountCard({ count, confidence, presence }) {
  const hasCount = count !== null && count !== undefined;

  return (
    <section className="card count-card" aria-label="Live people count">
      <h2 className="card__title">People count</h2>
      <div className="count-card__value">{formatCount(count)}</div>
      {hasCount ? (
        <p className="card__caption">
          {confidence !== null && confidence !== undefined
            ? `${Math.round(confidence * 100)}% confidence`
            : 'confidence unknown'}
        </p>
      ) : (
        <p className="card__caption card__caption--muted">
          {presence === null
            ? 'waiting for data…'
            : presence
              ? 'presence detected (counter model not loaded)'
              : 'room empty (counter model not loaded)'}
        </p>
      )}
    </section>
  );
}
