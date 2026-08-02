"""Tests for src/ingestion/attitude_adapter.py. No network calls at all -
this hazard type is synthetic-only by design (see module docstring: no
real public data source exists for spacecraft attitude/pointing status).
"""
from src.ingestion.attitude_adapter import SyntheticAttitudeAdapter
from src.pipeline import classify_attitude_severity
from src.schemas import Severity


def test_fetch_batch_spans_all_four_severity_bands():
    adapter = SyntheticAttitudeAdapter(run_id="test1")

    events = adapter.fetch_batch(limit=4)

    assert len(events) == 4
    severities = {classify_attitude_severity(e.raw_data["pointing_error_deg"]) for e in events}
    assert severities == {Severity.NOMINAL, Severity.WATCH, Severity.WARNING, Severity.CRITICAL}


def test_fetch_batch_events_are_clearly_labeled_synthetic():
    adapter = SyntheticAttitudeAdapter(run_id="test1")

    events = adapter.fetch_batch(limit=4)

    for event in events:
        assert event.source == "synthetic-attitude-fixture"
        assert "synthetic" in event.event_id
        raw = event.raw_data
        assert "object_a_id" not in raw  # single-object shape, not a pair
        assert "perigee_altitude_km" not in raw  # not the decay hazard's shape
        assert raw["pointing_error_deg"] >= 0
        assert raw["angular_rate_deg_s"] >= 0
        assert 0 <= raw["solar_panel_power_pct"] <= 100


def test_fetch_batch_respects_limit():
    adapter = SyntheticAttitudeAdapter(run_id="test1")

    events = adapter.fetch_batch(limit=2)

    assert len(events) == 2


def test_fetch_batch_event_ids_are_unique_across_separate_scans():
    """Same collision-avoidance reasoning as every other adapter in this
    project: DecisionLogger matches an event_id's FIRST logged occurrence,
    so repeat scans need distinct ids."""
    first_scan = SyntheticAttitudeAdapter(run_id="run-a").fetch_batch(limit=4)
    second_scan = SyntheticAttitudeAdapter(run_id="run-b").fetch_batch(limit=4)

    first_ids = {e.event_id for e in first_scan}
    second_ids = {e.event_id for e in second_scan}
    assert first_ids.isdisjoint(second_ids)
