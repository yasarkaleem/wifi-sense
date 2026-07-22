"""Tests for loading scenario YAML configs."""

from __future__ import annotations

from replay.scenarios import DEFAULT_SCENARIOS_PATH, ScenarioConfig, load_scenarios


def test_default_scenarios_file_defines_required_scenarios():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert set(scenarios) >= {"empty_room", "one_person_walking", "two_people"}
    for scenario in scenarios.values():
        assert isinstance(scenario, ScenarioConfig)
        assert scenario.subcarrier_count == 64


def test_empty_room_has_no_people():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert scenarios["empty_room"].people == ()


def test_two_people_scenario_has_two_people():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert len(scenarios["two_people"].people) == 2
