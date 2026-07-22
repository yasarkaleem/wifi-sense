const LABELS = {
  connecting: 'Connecting…',
  open: 'Live',
  closed: 'Reconnecting…',
};

export default function ConnectionBadge({ status }) {
  const label = LABELS[status] ?? status;
  return (
    <div className={`connection-badge connection-badge--${status}`}>
      <span className="connection-badge__dot" aria-hidden="true" />
      {label}
    </div>
  );
}
