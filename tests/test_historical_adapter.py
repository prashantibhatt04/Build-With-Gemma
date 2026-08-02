"""Tests for src/ingestion/historical_adapter.py. No network calls - this
adapter never makes any; it replays fixed, documented historical numbers.
"""
from types import SimpleNamespace

from src.ingestion.historical_adapter import HistoricalReplayAdapter, IRIDIUM_COSMOS_COLLISION
from src.pipeline import make_analyze_node, make_decide_node
from src.schemas import ManeuverPlan, Severity, VerifiedClearance


class FakeGemmaClient:
    def __init__(self):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend="ollama")

    def generate(self, prompt: str, system=None, timeout: int = 60) -> str:
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
