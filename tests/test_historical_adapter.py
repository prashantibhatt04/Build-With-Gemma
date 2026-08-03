"""Tests for src/ingestion/historical_adapter.py. No network calls - this
adapter never makes any on its own; it replays fixed, documented
historical numbers. Its optional real-repropagation cross-check is
exercised here against a faked SpaceTrackClient - real live verification
against an actual account is documented separately in
ROADMAP_TO_PRODUCT.md.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingestion.historical_adapter import (
    HistoricalEvent, HistoricalReplayAdapter, IRIDIUM_COSMOS_COLLISION, _parse_bare_tle, real_repropagate_event,
)
from src.pipeline import make_analyze_node, make_decide_node
from src.schemas import ManeuverPlan, Severity, VerifiedClearance

# Two real, fixed, overlapping-altitude TLEs (small RAAN/mean-anomaly
# perturbations of the real ISS TLE - same fixtures test_celestrak_adapter.py
# uses, chosen so a real closest-approach search has something genuine to
# find) - not the real Iridium 33/Cosmos 2251 elements, since this test
# only needs to prove the MECHANISM (fetch -> parse -> propagate) works,
# not re-validate historical accuracy again (that's what live
# verification against a real account is for).
BARE_TLE_A = (
    "1 30001U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998\r\n"
    "2 30001  51.6402 175.0000 0004018  88.8954 100.0000 15.54059185113452\r\n"
)
BARE_TLE_B = (
    "1 30002U 98067A   18135.61844383  .00002728  00000-0  48567-4 0  9998\r\n"
    "2 30002  51.6402 190.0000 0004018  88.8954 200.0000 15.54059185113452\r\n"
)


class FakeGemmaClient:
    def __init__(self):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend="ollama")

    def generate(self, prompt: str, system=None, timeout: int = 60, format=None) -> str:
        return "GO - stubbed response."


def test_fetch_batch_returns_the_real_documented_numbers():
    adapter = HistoricalReplayAdapter(run_id="test1")
    events = adapter.fetch_batch(limit=5)

    assert len(events) == 1
    event = events[0]
    raw = event.raw_data

    assert event.source == "historical-replay"
    # NORAD catalog numbers, distance, and relative velocity are all real,
    # documented values (see IRIDIUM_COSMOS_COLLISION's sourcing comment) -
    # not invented for this test.
    assert raw["object_a_id"] == "24946"  # Iridium 33
    assert raw["object_a_name"] == "IRIDIUM 33"
    assert raw["object_b_id"] == "22675"  # Cosmos 2251
    assert raw["object_b_name"] == "COSMOS 2251"
    assert raw["min_distance_km"] == 0.584
    assert raw["relative_velocity_km_s"] == 11.7
    assert raw["time_of_closest_approach"] == "2009-02-10T16:56:00+00:00"
    assert "historical_event" in raw
    assert "historical_source" in raw
    assert "COLLISION" in raw["historical_actual_outcome"]


def test_fetch_batch_event_id_includes_run_id_for_uniqueness():
    """Same collision-avoidance reasoning as SyntheticCriticalAdapter:
    DecisionLogger matches an event_id's FIRST logged occurrence, so
    repeat replays need distinct ids or a second run would silently
    update the first run's stale entry instead of creating its own."""
    first = HistoricalReplayAdapter(run_id="run-a").fetch_batch(limit=1)[0]
    second = HistoricalReplayAdapter(run_id="run-b").fetch_batch(limit=1)[0]

    assert first.event_id != second.event_id
    assert "run-a" in first.event_id
    assert "run-b" in second.event_id


def test_fetch_batch_respects_limit():
    adapter = HistoricalReplayAdapter(run_id="test1")
    assert adapter.fetch_batch(limit=0) == []
    assert len(adapter.fetch_batch(limit=1)) == 1


def test_real_historical_prediction_classifies_as_critical_through_the_real_pipeline():
    """The core claim of this phase: this system's ordinary, unmodified
    severity threshold - fed the REAL 584m SOCRATES prediction, not a
    synthetic stand-in - classifies it as CRITICAL and computes a
    maneuver, exactly as it would for any live conjunction. Nothing in
    analyze_node/decide_node is special-cased for historical replay."""
    event = HistoricalReplayAdapter(run_id="test1").fetch_batch(limit=1)[0]
    client = FakeGemmaClient()
    analyze_node = make_analyze_node(client)
    decide_node = make_decide_node(client)

    analyzed = analyze_node({
        "telemetry": event, "finding": None, "decision": None, "log_path": None,
    })
    assert analyzed["finding"].severity == Severity.CRITICAL

    result = decide_node(analyzed)
    decision = result["decision"]

    assert isinstance(decision.maneuver_plan, ManeuverPlan)
    assert isinstance(decision.verified_clearance, VerifiedClearance)
    assert decision.verified_clearance.cleared is True


def test_default_historical_events_matches_iridium_cosmos_constant():
    from src.ingestion.historical_adapter import DEFAULT_HISTORICAL_EVENTS
    assert DEFAULT_HISTORICAL_EVENTS == (IRIDIUM_COSMOS_COLLISION,)


def test_fetch_batch_omits_cross_check_fields_when_no_spacetrack_client():
    event = HistoricalReplayAdapter(run_id="test1").fetch_batch(limit=1)[0]
    assert "real_repropagated_min_distance_km" not in event.raw_data
    assert "real_repropagation_error" not in event.raw_data


def test_parse_bare_tle_splits_a_real_shaped_two_line_response():
    l1, l2 = _parse_bare_tle(BARE_TLE_A, norad_id="30001")
    assert l1.startswith("1 30001U")
    assert l2.startswith("2 30001")


def test_parse_bare_tle_raises_for_malformed_input():
    with pytest.raises(ValueError, match="30001"):
        _parse_bare_tle("not a tle", norad_id="30001")


def _fake_spacetrack_client(tle_by_norad_id: dict[str, str]):
    client = MagicMock()
    client.fetch_historical_tle_text.side_effect = lambda norad_id, at_or_before: tle_by_norad_id[norad_id]
    return client


def test_real_repropagate_event_uses_the_real_orbital_physics_stack():
    event = HistoricalEvent(
        id_slug="test-event", object_a_id="30001", object_a_name="TEST SAT A",
        object_b_id="30002", object_b_name="TEST SAT B", min_distance_km=0.584,
        time_of_closest_approach="2018-05-15T00:00:00+00:00", relative_velocity_km_s=11.7,
        historical_event="test", historical_source="test", historical_actual_outcome="test",
    )
    client = _fake_spacetrack_client({"30001": BARE_TLE_A, "30002": BARE_TLE_B})

    result = real_repropagate_event(event, client, lookahead_hours=48)

    assert result["real_repropagated_min_distance_km"] > 0
    assert result["real_repropagated_relative_velocity_km_s"] > 0
    assert isinstance(result["real_repropagated_time_of_closest_approach"], str)
    assert isinstance(result["real_repropagated_object_a_tle_epoch"], str)
    assert isinstance(result["real_repropagated_object_b_tle_epoch"], str)
    client.fetch_historical_tle_text.assert_any_call("30001", event.time_of_closest_approach)
    client.fetch_historical_tle_text.assert_any_call("30002", event.time_of_closest_approach)


def test_fetch_batch_attaches_real_cross_check_when_client_provided():
    client = _fake_spacetrack_client({"24946": BARE_TLE_A, "22675": BARE_TLE_B})
    adapter = HistoricalReplayAdapter(run_id="test1", spacetrack_client=client)

    event = adapter.fetch_batch(limit=1)[0]

    assert "real_repropagated_min_distance_km" in event.raw_data
    # The documented number this event is actually classified on is
    # untouched by the cross-check having run.
    assert event.raw_data["min_distance_km"] == 0.584


def test_fetch_batch_records_a_cross_check_failure_without_raising():
    client = MagicMock()
    client.fetch_historical_tle_text.side_effect = RuntimeError("real network failure")
    adapter = HistoricalReplayAdapter(run_id="test1", spacetrack_client=client)

    event = adapter.fetch_batch(limit=1)[0]

    assert "real network failure" in event.raw_data["real_repropagation_error"]
    assert "real_repropagated_min_distance_km" not in event.raw_data
    assert event.raw_data["min_distance_km"] == 0.584  # the replay itself was never blocked
