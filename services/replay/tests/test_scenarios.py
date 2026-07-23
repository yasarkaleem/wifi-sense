"""Tests for loading scenario YAML configs."""

from __future__ import annotations

import pytest

from replay.scenarios import (
    DEFAULT_SCENARIOS_PATH,
    PathSegment,
    ScenarioConfig,
    TrajectoryPath,
    TrajectoryScenarioConfig,
    _build_trajectory_scenario,
    effective_scenario,
    load_scenarios,
)


def test_default_scenarios_file_defines_required_scenarios():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert set(scenarios) >= {"empty_room", "one_person_walking", "two_people"}
    for scenario in scenarios.values():
        assert isinstance(scenario, (ScenarioConfig, TrajectoryScenarioConfig))
        assert scenario.subcarrier_count == 64


def test_default_scenarios_file_defines_trajectory_scenarios():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert "one_person_walking_path" in scenarios
    assert "two_people_walking_paths" in scenarios
    assert isinstance(scenarios["one_person_walking_path"], TrajectoryScenarioConfig)
    assert len(scenarios["one_person_walking_path"].paths) == 1
    assert len(scenarios["two_people_walking_paths"].paths) == 2


def test_empty_room_has_no_people():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert scenarios["empty_room"].people == ()


def test_two_people_scenario_has_two_people():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert len(scenarios["two_people"].people) == 2


def test_trajectory_scenario_references_unknown_zone_raises():
    static = {}
    raw = {
        "paths": [{"segments": [{"zone": "Z9", "dwell_s": 5}]}],
    }
    with pytest.raises(ValueError, match="unknown zone"):
        _build_trajectory_scenario("bad", raw, static)


def test_trajectory_scenario_mismatched_zone_config_raises():
    zone_a = ScenarioConfig(
        name="A1",
        description="",
        subcarrier_count=64,
        channel=6,
        source_mac="AA:AA:AA:AA:AA:AA",
        rssi_base=-45,
        rssi_noise_std=1.0,
        amplitude_baseline=20.0,
        amplitude_noise_std=0.3,
        phase_noise_std=0.05,
        people=(),
    )
    zone_b = ScenarioConfig(
        name="B1",
        description="",
        subcarrier_count=128,  # mismatched on purpose
        channel=6,
        source_mac="AA:AA:AA:AA:AA:AA",
        rssi_base=-60,
        rssi_noise_std=1.5,
        amplitude_baseline=20.0,
        amplitude_noise_std=0.3,
        phase_noise_std=0.05,
        people=(),
    )
    static = {"A1": zone_a, "B1": zone_b}
    raw = {
        "paths": [
            {"segments": [{"zone": "A1", "dwell_s": 5}, {"zone": "B1", "dwell_s": 5}]},
        ],
    }
    with pytest.raises(ValueError, match="subcarrier_count"):
        _build_trajectory_scenario("bad", raw, static)


def _make_zone(name: str, *, subcarrier_center: float, rssi_base: float = -50.0) -> ScenarioConfig:
    from replay.scenarios import PersonMotion

    return ScenarioConfig(
        name=name,
        description="",
        subcarrier_count=64,
        channel=6,
        source_mac="AA:AA:AA:AA:AA:AA",
        rssi_base=rssi_base,
        rssi_noise_std=0.0,
        amplitude_baseline=20.0,
        amplitude_noise_std=0.0,
        phase_noise_std=0.0,
        people=(
            PersonMotion(
                walk_frequency_hz=1.0,
                amplitude_disturbance=4.0,
                subcarrier_center=subcarrier_center,
                subcarrier_spread=10.0,
                phase_offset_rad=0.0,
            ),
        ),
    )


def _make_trajectory(transition_s: float = 1.0) -> TrajectoryScenarioConfig:
    zone_a = _make_zone("A1", subcarrier_center=10.0, rssi_base=-40.0)
    zone_b = _make_zone("A2", subcarrier_center=50.0, rssi_base=-60.0)
    return TrajectoryScenarioConfig(
        name="test_traj",
        description="",
        subcarrier_count=64,
        channel=6,
        source_mac="AA:AA:AA:AA:AA:AA",
        transition_s=transition_s,
        paths=(
            TrajectoryPath(
                segments=(PathSegment(zone="A1", dwell_s=5.0), PathSegment(zone="A2", dwell_s=5.0)),
                loop=True,
            ),
        ),
        zones={"A1": zone_a, "A2": zone_b},
    )


def test_effective_scenario_returns_static_scenario_unchanged():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    static = scenarios["empty_room"]
    assert effective_scenario(static, 12.3) is static


def test_effective_scenario_settles_on_zone_well_after_transition():
    traj = _make_trajectory(transition_s=1.0)
    # 3s into the A1 segment (which starts at local_t=0), well past the 1s
    # transition -> fully settled at A1's signature.
    resolved = effective_scenario(traj, 3.0)
    assert resolved.people[0].subcarrier_center == pytest.approx(10.0)
    assert resolved.rssi_base == pytest.approx(-40.0)


def test_effective_scenario_crossfades_at_segment_start():
    traj = _make_trajectory(transition_s=1.0)
    # Segment boundary: A1 dwell_s=5 ends at local_t=5, A2 begins. Halfway
    # through the 1s transition (local_t=5.5) should be ~halfway blended
    # from A1's signature (prev, since it loops) to A2's.
    resolved = effective_scenario(traj, 5.5)
    assert resolved.people[0].subcarrier_center == pytest.approx((10.0 + 50.0) / 2, abs=1.0)
    assert resolved.rssi_base == pytest.approx((-40.0 + -60.0) / 2, abs=1.0)


def test_effective_scenario_no_discontinuity_across_the_transition_window():
    traj = _make_trajectory(transition_s=1.0)
    # Sample densely across the A1->A2 boundary (local_t=5) and check no
    # single-step jump exceeds what a continuous linear interpolation
    # would produce over that step size.
    times = [5.0 + i * 0.05 for i in range(21)]  # 5.0 .. 6.0
    centers = [effective_scenario(traj, t).people[0].subcarrier_center for t in times]
    max_step = max(abs(b - a) for a, b in zip(centers, centers[1:]))
    # Over transition_s=1.0s with a 40-unit total swing (10->50), a 0.05s
    # step should move at most ~2 units; allow generous slack.
    assert max_step < 5.0


def test_effective_scenario_multiple_paths_are_independent():
    zone_a = _make_zone("A1", subcarrier_center=10.0)
    zone_b = _make_zone("B1", subcarrier_center=60.0)
    traj = TrajectoryScenarioConfig(
        name="two_path_test",
        description="",
        subcarrier_count=64,
        channel=6,
        source_mac="AA:AA:AA:AA:AA:AA",
        transition_s=0.5,
        paths=(
            TrajectoryPath(segments=(PathSegment(zone="A1", dwell_s=10.0),), loop=True),
            TrajectoryPath(segments=(PathSegment(zone="B1", dwell_s=10.0),), loop=True),
        ),
        zones={"A1": zone_a, "B1": zone_b},
    )
    resolved = effective_scenario(traj, 5.0)
    assert len(resolved.people) == 2
    centers = sorted(p.subcarrier_center for p in resolved.people)
    assert centers[0] == pytest.approx(10.0)
    assert centers[1] == pytest.approx(60.0)
