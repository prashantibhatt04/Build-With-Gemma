"""Tests for scripts/run_demo.py - just the step logic that has real,
independently-testable behavior (not the interactive Confirm() pauses or
step ordering, which are exercised by actually running the demo)."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from scripts.run_demo import DemoContext, _step_audit_trail
from src.config import Settings
from src.logging_utils import DecisionLogger
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _entry(event_id: str, source: str) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source=source,
        raw_data={"object_a_name": "SAT-A", "object_b_name": "SAT-B", "min_distance_km": 50.0},
    )
    finding = AnomalyFinding(event_id=event_id, severity=Severity.WATCH, description="Test.", confidence=0.8)
    decision = Decision(action="continue", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_step_audit_trail_shows_the_reviewed_entry_not_just_the_last_line(tmp_path, monkeypatch):
    """Real bug this guards against: mark_reviewed rewrites its entry's
    line IN PLACE (see logging_utils.py's update_if) - it does NOT move
    that line to the end of the file. A real demo run logs decay/
    attitude events AFTER the conjunction event that gets reviewed, so
    the last line in the file is a DIFFERENT entry with
    human_reviewed=False - directly contradicting this step's own
    narration ("the one just marked reviewed above")."""
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )
    monkeypatch.setattr("scripts.run_demo.settings", settings)

    logger = DecisionLogger(settings=settings)
    logger.log(_entry("conj-reviewed", "celestrak"))
    logger.log(_entry("attitude-logged-after", "synthetic-attitude-fixture"))  # appended AFTER, never reviewed
    logger.mark_reviewed("conj-reviewed", reviewed_by="demo-reviewer")

    console = Console(record=True, width=120)
    ctx = DemoContext(console=console, reviewable_event_id="conj-reviewed")

    _step_audit_trail(ctx)

    output = console.export_text()
    assert "conj-reviewed" in output
    assert "attitude-logged-after" not in output
    assert '"human_reviewed": true' in output


def test_step_audit_trail_falls_back_to_last_line_without_a_reviewable_event(tmp_path, monkeypatch):
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )
    monkeypatch.setattr("scripts.run_demo.settings", settings)

    logger = DecisionLogger(settings=settings)
    logger.log(_entry("only-entry", "celestrak"))

    console = Console(record=True, width=120)
    ctx = DemoContext(console=console, reviewable_event_id=None)

    _step_audit_trail(ctx)

    assert "only-entry" in console.export_text()
