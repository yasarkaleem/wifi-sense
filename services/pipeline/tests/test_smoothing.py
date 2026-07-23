"""Unit tests for pipeline.smoothing.ZoneEMASmoother — pure math, hand-
computed expected values."""

from __future__ import annotations

import pytest

from pipeline.models.localizer import LocalizationEvent, ZoneProbability
from pipeline.smoothing import ZoneEMASmoother


def _event(timestamp: int, probs: dict[str, float]) -> LocalizationEvent:
    return LocalizationEvent(
        timestamp=timestamp, zones=tuple(ZoneProbability(zone_id=z, occupancy_probability=p) for z, p in probs.items())
    )


def test_span_3_gives_alpha_0_5():
    smoother = ZoneEMASmoother(span=3)
    assert smoother.alpha == pytest.approx(0.5)


def test_first_event_passes_through_unchanged():
    smoother = ZoneEMASmoother(span=3)
    event = _event(0, {"A1": 0.8, "A2": 0.2})
    smoothed = smoother.update(event)
    assert smoothed.zones[0].occupancy_probability == pytest.approx(0.8)
    assert smoothed.zones[1].occupancy_probability == pytest.approx(0.2)


def test_second_event_blends_with_alpha():
    smoother = ZoneEMASmoother(span=3)  # alpha=0.5
    smoother.update(_event(0, {"A1": 0.8, "A2": 0.2}))
    smoothed = smoother.update(_event(1, {"A1": 0.0, "A2": 1.0}))
    # A1: 0.5*0.0 + 0.5*0.8 = 0.4 ; A2: 0.5*1.0 + 0.5*0.2 = 0.6
    values = {z.zone_id: z.occupancy_probability for z in smoothed.zones}
    assert values["A1"] == pytest.approx(0.4)
    assert values["A2"] == pytest.approx(0.6)


def test_repeated_updates_converge_toward_new_value():
    smoother = ZoneEMASmoother(span=3)
    smoother.update(_event(0, {"A1": 0.0}))
    for i in range(1, 20):
        smoothed = smoother.update(_event(i, {"A1": 1.0}))
    assert smoothed.zones[0].occupancy_probability > 0.999


def test_probabilities_still_sum_to_one_after_smoothing():
    smoother = ZoneEMASmoother(span=3)
    smoother.update(_event(0, {"A1": 0.5, "A2": 0.3, "A3": 0.2}))
    smoothed = smoother.update(_event(1, {"A1": 0.1, "A2": 0.1, "A3": 0.8}))
    total = sum(z.occupancy_probability for z in smoothed.zones)
    assert total == pytest.approx(1.0)


def test_new_zone_id_not_seen_before_starts_from_its_own_value():
    smoother = ZoneEMASmoother(span=3)
    smoother.update(_event(0, {"A1": 1.0}))
    smoothed = smoother.update(_event(1, {"A1": 0.5, "A2": 0.5}))
    values = {z.zone_id: z.occupancy_probability for z in smoothed.zones}
    # A2 has no prior state -> prev defaults to its own new value -> unchanged
    assert values["A2"] == pytest.approx(0.5)
    assert values["A1"] == pytest.approx(0.75)  # 0.5*0.5 + 0.5*1.0


def test_reset_clears_state():
    smoother = ZoneEMASmoother(span=3)
    smoother.update(_event(0, {"A1": 1.0}))
    smoother.reset()
    smoothed = smoother.update(_event(1, {"A1": 0.3}))
    assert smoothed.zones[0].occupancy_probability == pytest.approx(0.3)


def test_invalid_span_raises():
    with pytest.raises(ValueError):
        ZoneEMASmoother(span=0)
