"""Tests for src/ingestion/synthetic_adapter.py."""
from datetime import datetime, timezone

from src.ingestion.synthetic_adapter import SyntheticCriticalAdapter


def test_fetch_batch_derives_tca_relative_to_now_not_a_fixed_calendar_date():
    """Real bug this guards against: a fixed calendar date (e.g.
    "2026-08-01T20:00:00") is CRITICAL-and-urgent only until that date
    passes - after which every "Run synthetic CRITICAL scenario" demo
    click shows the real tca_urgency_label() feature
    (src/dashboard_data.py) rendering "TCA already passed..." on the
    project's own flagship demo fixture, found live during a VC-demo
    readiness review."""
    before = datetime.now(timezone.utc)
    events = SyntheticCriticalAdapter(run_id="test").fetch_batch(limit=1)
    after = datetime.now(timezone.utc)

    tca = datetime.fromisoformat(events[0].raw_data["time_of_closest_approach"])

    assert before < tca < after.replace(year=after.year + 1)  # sanity: real future date, not stale
    assert tca > after  # the actual guard: TCA is genuinely still in the future


def test_fetch_batch_uses_the_same_tca_across_the_whole_batch():
    """All events in one batch represent one screening run at one
    moment - they should share the same TCA, not each get a
    microseconds-apart value from being computed in a loop."""
    events = SyntheticCriticalAdapter(run_id="test").fetch_batch(limit=4)

    tcas = {e.raw_data["time_of_closest_approach"] for e in events}
    assert len(tcas) == 1
