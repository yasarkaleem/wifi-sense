// Parses a "A1"-style zone_id (see ../../room.yaml / pipeline/room.py's
// _zone_id encoding: row letter, 1-indexed column number) into a 0-indexed
// {row, col} grid position, so ZoneHeatmap can lay zones out without the
// dashboard needing to fetch room.yaml itself.

export function parseZoneId(zoneId) {
  const match = /^([A-Za-z])(\d+)$/.exec(zoneId ?? '');
  if (!match) return null;
  return {
    row: match[1].toUpperCase().charCodeAt(0) - 65, // 'A' -> 0
    col: parseInt(match[2], 10) - 1, // 1-indexed -> 0-indexed
  };
}
