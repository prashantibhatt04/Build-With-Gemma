"""Tests for DecisionLogger, including the human-review stub
(find_entry/mark_reviewed). No network calls."""
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.logging_utils import DecisionLogger
from src.maneuver import compute_avoidance_maneuver
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _settings(tmp_path) -> Settings:
    return Settings(
        gemma_backend="ollama",
        gemma_model="gemma4:e4b",
        ollama_host="http://localhost:11434",
        gemma_api_key="",
        gemma_model_api="gemma-4-26b-a4b-it",
        log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )


def _make_entry(event_id: str) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id,
        timestamp=datetime.now(timezone.utc),
        source="test",
        raw_data={"value": 1.0},
    )
    finding = AnomalyFinding(
        event_id=event_id, severity=Severity.NOMINAL,
        description="Test finding.", confidence=0.8,
    )
    decision = Decision(
        action="continue", rationale="Test rationale.", made_at=datetime.now(timezone.utc),
    )
    provenance = GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=provenance,
    )


def _make_pending_approval_entry(event_id: str, min_distance_km: float = 3.0) -> DecisionLogEntry:
    """A CRITICAL-severity entry shaped like decide_node's output when the
    configured backend is "api": maneuver_plan is set, but nothing has been
    executed/verified yet - awaiting_human_approval is True."""
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="test",
        raw_data={
            "object_a_id": "1", "object_a_name": "A", "object_b_id": "2", "object_b_name": "B",
            "min_distance_km": min_distance_km, "time_of_closest_approach": "2026-08-02T00:00:00+00:00",
            "relative_velocity_km_s": 6.0,
        },
    )
    finding = AnomalyFinding(
        event_id=event_id, severity=Severity.CRITICAL, description="Test finding.", confidence=0.8,
    )
    plan = compute_avoidance_maneuver(
        object_a="1", object_b="2", min_distance_km=min_distance_km, relative_velocity_km_s=6.0,
    )
    decision = Decision(
        action="abort", rationale="Maneuver proposed: awaiting human approval before execution.",
        made_at=datetime.now(timezone.utc), maneuver_plan=plan, awaiting_human_approval=True,
    )
    provenance = GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=provenance,
    )


def test_approve_maneuver_verifies_and_records_human_approval(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_pending_approval_entry("critical-1"))

    updated = logger.approve_maneuver("critical-1", approved=True, approved_by="alice")

    assert updated.decision.awaiting_human_approval is False
    assert updated.decision.verified_clearance is not None
    assert updated.decision.verified_clearance.cleared is True
    assert updated.decision.maneuver_approval is not None
    assert updated.decision.maneuver_approval.mode == "human"
    assert updated.decision.maneuver_approval.approved is True
    assert updated.decision.maneuver_approval.approved_by == "alice"
    assert "APPROVED by alice" in updated.decision.rationale

    # Persisted, not just returned in memory.
    refetched = logger.find_entry("critical-1")
    assert refetched is not None
    assert refetched[2].decision.verified_clearance is not None


def test_approve_maneuver_rejection_leaves_nothing_executed(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_pending_approval_entry("critical-2"))

    updated = logger.approve_maneuver("critical-2", approved=False, approved_by="bob")

    assert updated.decision.awaiting_human_approval is False
    assert updated.decision.verified_clearance is None  # nothing executed
    assert updated.decision.maneuver_approval.approved is False
    assert updated.decision.maneuver_approval.approved_by == "bob"
    assert "REJECTED by bob" in updated.decision.rationale


def test_approve_maneuver_raises_for_unknown_event_id(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_pending_approval_entry("critical-3"))

    with pytest.raises(ValueError, match="no-such-event"):
        logger.approve_maneuver("no-such-event", approved=True, approved_by="alice")


def test_approve_maneuver_raises_if_not_awaiting_approval(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_entry("nominal-1"))  # NOMINAL, never awaiting approval

    with pytest.raises(ValueError, match="not awaiting human approval"):
        logger.approve_maneuver("nominal-1", approved=True, approved_by="alice")


def test_mark_reviewed_updates_matching_entry_and_leaves_others_untouched(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_entry("event-1"))
    logger.log(_make_entry("event-2"))
    logger.log(_make_entry("event-3"))

    updated = logger.mark_reviewed("event-2", reviewed_by="alice")

    assert updated.human_reviewed is True
    assert updated.reviewed_by == "alice"
    assert updated.human_reviewed_at is not None
    assert (datetime.now(timezone.utc) - updated.human_reviewed_at).total_seconds() < 5

    # Re-read from disk to confirm the rewrite actually persisted, and that
    # the other two lines were left alone.
    found_2 = logger.find_entry("event-2")
    assert found_2 is not None
    assert found_2[2].human_reviewed is True
    assert found_2[2].reviewed_by == "alice"

    for event_id in ("event-1", "event-3"):
        found = logger.find_entry(event_id)
        assert found is not None
        assert found[2].human_reviewed is False
        assert found[2].reviewed_by is None


def test_mark_reviewed_raises_for_unknown_event_id(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_entry("event-1"))

    with pytest.raises(ValueError, match="no-such-event"):
        logger.mark_reviewed("no-such-event", reviewed_by="alice")


def test_find_entry_returns_none_when_no_logs_exist(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    assert logger.find_entry("anything") is None


def test_load_all_entries_returns_empty_list_when_no_logs_exist(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    assert logger.load_all_entries() == []


def test_load_all_entries_returns_every_logged_entry_in_order(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_entry("event-1"))
    logger.log(_make_entry("event-2"))
    logger.log(_make_pending_approval_entry("event-3"))

    entries = logger.load_all_entries()

    assert [e.telemetry.event_id for e in entries] == ["event-1", "event-2", "event-3"]
    assert entries[2].decision.awaiting_human_approval is True


def test_load_all_entries_reflects_in_place_rewrites(tmp_path):
    """mark_reviewed/approve_maneuver rewrite a line in place - confirms
    load_all_entries picks up the UPDATED content, not a stale copy."""
    logger = DecisionLogger(settings=_settings(tmp_path))
    logger.log(_make_entry("event-1"))
    logger.mark_reviewed("event-1", reviewed_by="alice")

    entries = logger.load_all_entries()

    assert len(entries) == 1
    assert entries[0].human_reviewed is True
    assert entries[0].reviewed_by == "alice"
