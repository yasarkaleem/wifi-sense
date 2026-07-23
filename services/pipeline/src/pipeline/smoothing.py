"""Temporal smoothing for zone-localization output.

A raw per-window zone prediction can flicker between adjacent zones frame
to frame — the localizer has no memory of its own. `ZoneEMASmoother`
applies a simple exponential moving average per zone_id across
consecutive `LocalizationEvent`s, so the published probabilities move
smoothly rather than jumping.
"""

from __future__ import annotations

from pipeline.models.localizer import LocalizationEvent, ZoneProbability


class ZoneEMASmoother:
    """Exponential moving average over the last `span` windows' zone
    probabilities, one independent EMA per zone_id.

    `alpha = 2 / (span + 1)` is the standard span-N EMA weighting (span=3
    -> alpha=0.5). The first event seen passes through unchanged (there's
    no prior state to blend with); every event after that is blended with
    the running per-zone average. Since EMA is linear and each event's
    probabilities already sum to ~1, the smoothed output's probabilities
    still sum to ~1.
    """

    def __init__(self, span: int = 3) -> None:
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        self.alpha = 2.0 / (span + 1)
        self._state: dict[str, float] | None = None

    def update(self, event: LocalizationEvent) -> LocalizationEvent:
        if self._state is None:
            self._state = {z.zone_id: z.occupancy_probability for z in event.zones}
        else:
            for z in event.zones:
                prev = self._state.get(z.zone_id, z.occupancy_probability)
                self._state[z.zone_id] = self.alpha * z.occupancy_probability + (1 - self.alpha) * prev

        smoothed_zones = tuple(
            ZoneProbability(zone_id=z.zone_id, occupancy_probability=self._state[z.zone_id]) for z in event.zones
        )
        return LocalizationEvent(timestamp=event.timestamp, zones=smoothed_zones)

    def reset(self) -> None:
        self._state = None
