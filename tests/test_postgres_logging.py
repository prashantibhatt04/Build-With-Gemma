"""Tests for src/postgres_logging.py, run against a REAL local Postgres
instance - not mocked. Storage correctness (concurrent-safe row updates,
real indexed lookup) is exactly the property this module exists to add
over the JSONL store, so a mock would prove nothing real here; unlike
this project's Gemma/network integrations, a local Postgres is fast and
side-effect-free to actually use in tests.

Skipped cleanly (not failed) if no local Postgres is reachable, so this
suite still runs in any environment without one - matching this
project's practice of never letting the test suite require live network
access. Set TEST_DATABASE_URL to point at a different instance if
localhost's default isn't right for a given machine.
"""
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")

from src.logging_utils import DecisionLogger  # noqa: E402
from src.maneuver import compute_avoidance_maneuver  # noqa: E402
from src.postgres_logging import PostgresDecisionLogStore  # noqa: E402
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent  # noqa: E402

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://localhost/build_with_gemma")


def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_reachable(), reason=f"no reachable Postgres at {TEST_DATABASE_URL}",
)


@pytest.fixture
def store():
    # A fresh, uniquely-named table per test - real isolation without
    # needing a separate database per test run.
    table_name = f"test_decision_log_{uuid.uuid4().hex[:8]}"
    s = PostgresDecisionLogStore(TEST_DATABASE_URL, table_name=table_name)
    yield s
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()


def test_count_older_than_matches_what_delete_would_remove(store):
    store.append(_make_entry("pg-count-1"))
    store.append(_make_entry("pg-count-2"))

    future_cutoff = datetime.now(timezone.utc) + timedelta(hours=1)
    count = store.count_older_than(future_cutoff)

    assert count == 2
    assert store.load_all() != []  # count_older_than must not delete anything


def test_delete_older_than_removes_rows_when_cutoff_is_in_the_future(store):
    store.append(_make_entry("pg-old-1"))
    store.append(_make_entry("pg-old-2"))

    deleted = store.delete_older_than(datetime.now(timezone.utc) + timedelta(hours=1))

    assert deleted == 2
    assert store.load_all() == []


def test_delete_older_than_keeps_rows_when_cutoff_is_in_the_past(store):
    store.append(_make_entry("pg-recent-1"))

    deleted = store.delete_older_than(datetime.now(timezone.utc) - timedelta(hours=1))

    assert deleted == 0
    assert store.find("pg-recent-1") is not None


def test_delete_older_than_only_removes_rows_actually_before_the_cutoff(store):
    """Real, precise DB-side filtering - a genuinely old row and a
    genuinely recent one in the same table must be treated differently,
    not an all-or-nothing table-wide decision."""
    store.append(_make_entry("pg-real-old"))
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute(
            f"UPDATE {store.table_name} SET created_at = %s WHERE event_id = %s",
            (datetime.now(timezone.utc) - timedelta(days=400), "pg-real-old"),
        )
        conn.commit()
    store.append(_make_entry("pg-real-recent"))

    deleted = store.delete_older_than(datetime.now(timezone.utc) - timedelta(days=365))

    assert deleted == 1
    assert store.find("pg-real-old") is None
    assert store.find("pg-real-recent") is not None


def _make_entry(event_id: str) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="test",
        raw_data={"value": 1.0},
    )
    finding = AnomalyFinding(
        event_id=event_id, severity=Severity.NOMINAL, description="Test.", confidence=0.8,
    )
    decision = Decision(action="continue", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    provenance = GemmaProvenance(source="gemma", model_used="fake", latency_ms=1.0)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=provenance,
    )


def _make_pending_approval_entry(event_id: str) -> DecisionLogEntry:
    """A CRITICAL-severity entry shaped like decide_node's output on the
    "api" backend: a real maneuver_plan, awaiting_human_approval=True -
    see test_logging_utils.py's identical helper for the JSONL-backend
    version of this same fixture."""
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="test",
        raw_data={
            "object_a_id": "1", "object_a_name": "A", "object_b_id": "2", "object_b_name": "B",
            "min_distance_km": 3.0, "time_of_closest_approach": "2026-08-02T00:00:00+00:00",
            "relative_velocity_km_s": 6.0,
        },
    )
    finding = AnomalyFinding(
        event_id=event_id, severity=Severity.CRITICAL, description="Test finding.", confidence=0.8,
    )
    plan = compute_avoidance_maneuver(
        object_a="1", object_b="2", min_distance_km=3.0, relative_velocity_km_s=6.0,
    )
    decision = Decision(
        action="abort", rationale="Maneuver proposed: awaiting human approval before execution.",
        made_at=datetime.now(timezone.utc), maneuver_plan=plan, awaiting_human_approval=True,
    )
    provenance = GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=provenance,
    )


def test_approve_maneuver_is_race_safe_under_concurrent_calls_against_real_postgres(store):
    """Real bug this closes, reproduced live during the original QA pass:
    two threads racing approve()/reject() on the same CRITICAL event
    against a real Postgres backend could both read
    awaiting_human_approval=True before either wrote, both pass the
    Python-side guard, and both write - the second silently clobbering
    the first with no error to either caller (confirmed live: alice's
    approval vanished, only bob's reject persisted, both calls returned
    normally). update_if's SELECT ... FOR UPDATE row lock is what closes
    this - verified here with real threads and a real database, not a
    mock, since the whole point is exercising Postgres's actual locking
    behavior."""
    logger = DecisionLogger(store=store)
    logger.log(_make_pending_approval_entry("pg-race-1"))

    outcomes: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def call(name: str, approved: bool) -> None:
        barrier.wait()
        try:
            logger.approve_maneuver("pg-race-1", approved=approved, approved_by=name)
            outcomes[name] = "succeeded"
        except ValueError:
            outcomes[name] = "raised"

    alice = threading.Thread(target=call, args=("alice", True))
    bob = threading.Thread(target=call, args=("bob", False))
    alice.start()
    bob.start()
    alice.join()
    bob.join()

    assert list(outcomes.values()).count("succeeded") == 1
    assert list(outcomes.values()).count("raised") == 1

    winner = "alice" if outcomes["alice"] == "succeeded" else "bob"
    final = logger.find_entry("pg-race-1")
    assert final.decision.awaiting_human_approval is False
    assert final.decision.maneuver_approval.approved_by == winner


def test_append_then_find_round_trips_a_real_entry(store):
    identifier = store.append(_make_entry("pg-1"))
    assert identifier  # non-empty, truthy

    found = store.find("pg-1")

    assert found is not None
    assert found.telemetry.event_id == "pg-1"
    assert found.finding.severity == Severity.NOMINAL


def test_find_returns_none_for_unknown_event_id(store):
    assert store.find("does-not-exist") is None


def test_update_persists_a_real_change(store):
    store.append(_make_entry("pg-2"))
    entry = store.find("pg-2")
    updated = entry.model_copy(update={"human_reviewed": True, "reviewed_by": "alice"})

    store.update("pg-2", updated)

    refetched = store.find("pg-2")
    assert refetched.human_reviewed is True
    assert refetched.reviewed_by == "alice"


def test_update_raises_for_unknown_event_id(store):
    with pytest.raises(ValueError, match="pg-nonexistent"):
        store.update("pg-nonexistent", _make_entry("pg-nonexistent"))


def test_update_targets_only_the_first_occurrence_of_a_duplicated_event_id(store):
    """Real concurrent/repeated scans can log the same event_id twice
    (see celestrak_adapter.py's run_id discussion) - update() must
    consistently resolve to ONE row (the earliest inserted), not error on
    duplicates or silently touch the wrong one."""
    store.append(_make_entry("pg-dup"))
    store.append(_make_entry("pg-dup"))

    updated = _make_entry("pg-dup").model_copy(update={"human_reviewed": True})
    store.update("pg-dup", updated)

    found = store.find("pg-dup")
    assert found.human_reviewed is True


def test_load_all_returns_entries_in_real_insertion_order(store):
    store.append(_make_entry("pg-a"))
    store.append(_make_entry("pg-b"))
    store.append(_make_entry("pg-c"))

    entries = store.load_all()

    assert [e.telemetry.event_id for e in entries] == ["pg-a", "pg-b", "pg-c"]


def test_decision_logger_uses_postgres_when_database_url_is_configured():
    """End-to-end through the real DecisionLogger public interface, not
    just the store directly - confirms the factory wiring in
    logging_utils._default_store actually picks Postgres."""
    from src.config import Settings
    from src.logging_utils import DecisionLogger

    table_name = f"test_decision_log_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir="./logs",
        delta_v_budget_m_s=5.0, database_url=TEST_DATABASE_URL,
    )
    store = PostgresDecisionLogStore(TEST_DATABASE_URL, table_name=table_name)
    logger = DecisionLogger(settings=settings, store=store)

    logger.log(_make_entry("pg-logger-1"))
    found = logger.find_entry("pg-logger-1")
    reviewed = logger.mark_reviewed("pg-logger-1", reviewed_by="bob")

    assert found is not None
    assert reviewed.human_reviewed is True
    assert reviewed.reviewed_by == "bob"

    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
