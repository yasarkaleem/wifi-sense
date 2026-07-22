// Color tokens, following the dataviz skill's dark-surface palette
// (references/palette.md): one hue (blue) for sequential/magnitude
// encoding. In dark mode the ramp's anchor flips vs. light mode — the
// DARKEST step means "near zero" and recedes toward the dark surface,
// the BRIGHTEST step means "high magnitude" and stands out — the reverse
// of how the same ramp reads on a light surface.

export const SURFACE = '#1a1a19';
export const PAGE_PLANE = '#0d0d0d';
export const INK_PRIMARY = '#ffffff';
export const INK_SECONDARY = '#c3c2b7';
export const INK_MUTED = '#898781';
export const GRIDLINE = '#2c2c2a';
export const BASELINE = '#383835';
export const BORDER = 'rgba(255,255,255,0.10)';

export const STATUS_GOOD = '#0ca30c';
export const STATUS_CRITICAL = '#e66767';

// Blue sequential ramp (see palette.md), ordered darkest -> brightest so
// index 0 sits nearest the dark surface ("near zero") and the last index
// is the brightest ("high magnitude").
const SEQUENTIAL_STOPS = [
  SURFACE, // recedes into the surface at 0
  '#0d366b', // step 700
  '#104281', // step 650
  '#184f95', // step 600
  '#1c5cab', // step 550
  '#256abf', // step 500
  '#2a78d6', // step 450
  '#3987e5', // step 400
  '#5598e7', // step 350
  '#86b6ef', // step 250 — stop short of the near-white steps to avoid glare on dark
];

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

/** value in [0, 1] -> a CSS rgb() string along the dark-mode sequential ramp. */
export function sequentialColor(value) {
  const t = Math.max(0, Math.min(1, value ?? 0));
  const scaled = t * (SEQUENTIAL_STOPS.length - 1);
  const i = Math.floor(scaled);
  const frac = scaled - i;
  const [r1, g1, b1] = hexToRgb(SEQUENTIAL_STOPS[i]);
  const [r2, g2, b2] = hexToRgb(SEQUENTIAL_STOPS[Math.min(i + 1, SEQUENTIAL_STOPS.length - 1)]);
  const r = Math.round(lerp(r1, r2, frac));
  const g = Math.round(lerp(g1, g2, frac));
  const b = Math.round(lerp(b1, b2, frac));
  return `rgb(${r}, ${g}, ${b})`;
}
