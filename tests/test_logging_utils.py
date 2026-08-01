"""Tests for DecisionLogger, including the human-review stub
(find_entry/mark_reviewed). No network calls."""
from datetime import datetime, timezone

import pytest

from src.config import Settings
from src.logging_utils import DecisionLogger
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
