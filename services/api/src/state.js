// In-memory "current combined state", merged from whichever of
// services/pipeline's presence/count/zones topics have published so far.
// Backs GET /status and the snapshot a newly-connected WebSocket client
// gets immediately. Any field pipeline hasn't published yet (e.g. count/
// zones when no counter/localizer checkpoint is configured) stays null —
// see ../README.md.

const EMPTY_STATE = Object.freeze({
  timestamp: null,
  presence: null,
  motion_intensity: null,
  count: null,
  confidence: null,
  zones: null,
});

export function createState() {
  let current = { ...EMPTY_STATE };

  return {
    get() {
      return { ...current };
    },
    applyPresence(event) {
      current = {
        ...current,
        timestamp: event.timestamp,
        presence: event.presence,
        motion_intensity: event.motion_intensity,
      };
      return { ...current };
    },
    applyCount(event) {
      current = {
        ...current,
        timestamp: event.timestamp,
        count: event.count,
        confidence: event.confidence,
      };
      return { ...current };
    },
    applyZones(event) {
      current = {
        ...current,
        timestamp: event.timestamp,
        zones: event.zones,
      };
      return { ...current };
    },
  };
}
