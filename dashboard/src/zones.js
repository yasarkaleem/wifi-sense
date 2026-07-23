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

// Probability-weighted mean grid position across `zones` — where the
// "person dot" ZoneHeatmap renders should sit. Returns null when nobody's
// localized anywhere (total probability ~0), so the dot can hide itself
// rather than snapping to (0, 0).
export function weightedCentroid(zones) {
  if (!zones || zones.length === 0) return null;

  let totalWeight = 0;
  let rowSum = 0;
  let colSum = 0;
  for (const zone of zones) {
    const pos = parseZoneId(zone.zone_id);
    if (!pos) continue;
    const weight = zone.occupancy_probability ?? 0;
    totalWeight += weight;
    rowSum += pos.row * weight;
    colSum += pos.col * weight;
  }

  if (totalWeight < 1e-6) return null;
  return { row: rowSum / totalWeight, col: colSum / totalWeight };
}

// The single most-likely zone across `zones` (by occupancy_probability),
// or null if `zones` is empty — used for ZoneHeatmap's legend line.
export function mostLikelyZone(zones) {
  if (!zones || zones.length === 0) return null;
  return zones.reduce((best, z) => (z.occupancy_probability > best.occupancy_probability ? z : best));
}
