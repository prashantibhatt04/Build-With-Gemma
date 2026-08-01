"""Tests for src/display.py. Renders to an in-memory Console(record=True)
so output can be asserted on without touching a real terminal."""
from datetime import datetime, timezone

from rich.console import Console

from src.display import render_entries, render_entry
from src.maneuver import compute_avoidance_maneuver, verify_maneuver
from src.schemas import (
    AnomalyFinding,
    Decision,
    DecisionLogEntry,
    GemmaProvenance,
    ManeuverApproval,
    Severity,
    TelemetryEvent,
)


def _provenance() -> GemmaProvenance:
    return GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0)


def _conjunction_entry(severity: Severity, min_distance_km: float = 50.0) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id="conj-display-test",
        timestamp=datetime.now(timezone.utc),
        source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "SAT-A",
            "object_b_id": "2", "object_b_name": "SAT-B",
            "min_distance_km": min_distance_km,
            "time_of_closest_approach": "2026-08-02T00:00:00+00:00",
            "relative_velocity_km_s": 5.0,
        },
    )
    finding = AnomalyFinding(
        event_id=telemetry.event_id, severity=severity,
        description="Test finding.", confidence=0.8,
    )

    maneuver_plan = None
    verified_clearance = None
    budget_insufficient = False
    if severity == Severity.CRITICAL:
        maneuver_plan = compute_avoidance_maneuver(
            object_a="1", object_b="2",
            min_distance_km=min_distance_km, relative_velocity_km_s=5.0,
        )
        verified_clearance = verify_maneuver(min_distance_km, maneuver_plan)

    decision = Decision(
        action="continue", rationale="Recommendation: continue. Test rationale.",
        made_at=datetime.now(timezone.utc),
        maneuver_plan=maneuver_plan, verified_clearance=verified_clearance,
        budget_insufficient=budget_insufficient,
    )
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=_provenance(),
    )


def test_render_entry_shows_severity_badge_and_rationale():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.WATCH)

    render_entry(console, entry)
    output = console.export_text()

    assert "WATCH" in output
    assert "SAT-A vs SAT-B" in output
    assert "50.00km" in output
    assert "Test rationale" in output
    # No maneuver for WATCH - no panel should appear.
    assert "MANEUVER" not in output


def test_render_entry_shows_maneuver_panel_when_executed():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)

    render_entry(console, entry)
    output = console.export_text()

    assert "CRITICAL" in output
    assert "AUTONOMOUS MANEUVER EXECUTED" in output
    assert "radial-outward" in output
    assert "cleared=True" in output


def test_render_entry_shows_budget_insufficient_panel():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)
    # Simulate what decide_node does when the budget tracker refuses: plan
    # stays populated, verified_clearance is cleared, flag is set.
    entry.decision.verified_clearance = None
    entry.decision.budget_insufficient = True

    render_entry(console, entry)
    output = console.export_text()

    assert "BUDGET INSUFFICIENT" in output
    assert "NOT executed" in output


def test_render_entry_shows_awaiting_approval_panel_without_crashing():
    """Regression test: this is the exact state (maneuver_plan set,
    verified_clearance=None, budget_insufficient=False,
    awaiting_human_approval=True) that crashed render_entry with an
    AttributeError before this state existed in the renderer's logic."""
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)
    entry.decision.verified_clearance = None
    entry.decision.awaiting_human_approval = True

    render_entry(console, entry)
    output = console.export_text()

    assert "AWAITING HUMAN APPROVAL" in output
    assert "NOT executed yet" in output


def test_render_entry_shows_human_approved_execution_panel():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)
    entry.decision.maneuver_approval = ManeuverApproval(
        mode="human", approved=True, approved_by="alice",
        approved_at=datetime.now(timezone.utc), reason="Approved via CLI.",
    )

    render_entry(console, entry)
    output = console.export_text()

    assert "approved by alice" in output
    assert "AUTONOMOUS" not in output  # this one was human-approved, not autonomous


def test_render_entry_shows_rejected_panel():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)
    entry.decision.verified_clearance = None
    entry.decision.maneuver_approval = ManeuverApproval(
        mode="human", approved=False, approved_by="bob",
        approved_at=datetime.now(timezone.utc), reason="Rejected via CLI.",
    )

    render_entry(console, entry)
    output = console.export_text()

    assert "MANEUVER REJECTED" in output
    assert "REJECTED by bob" in output


def test_render_entry_shows_vetoed_panel_distinct_from_human_rejection():
    """Phase 9: a Gemma veto (mode="autonomous", approved=False) is a
    different situation from a human rejecting a cloud-pending proposal -
    it should render with its own title, not "MANEUVER REJECTED"."""
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.CRITICAL, min_distance_km=2.0)
    entry.decision.verified_clearance = None
    entry.decision.maneuver_approval = ManeuverApproval(
        mode="autonomous", approved=False, approved_by="Gemma (autonomous safety review)",
        approved_at=datetime.now(timezone.utc),
        reason="Maneuver was independently verified safe by deterministic physics, but vetoed.",
    )

    render_entry(console, entry)
    output = console.export_text()

    assert "MANEUVER VETOED" in output
    assert "AUTONOMOUS SAFETY REVIEW" in output
    assert "MANEUVER REJECTED" not in output


def test_render_entry_handles_non_conjunction_event_without_crashing():
    console = Console(record=True, width=120)
    telemetry = TelemetryEvent(
        event_id="dummy-event-1", timestamp=datetime.now(timezone.utc),
        source="dummy-sensor", raw_data={"value": 42.0},
    )
    finding = AnomalyFinding(
        event_id=telemetry.event_id, severity=Severity.NOMINAL,
        description="Placeholder.", confidence=0.5,
    )
    decision = Decision(
        action="continue", rationale="Recommendation: continue.",
        made_at=datetime.now(timezone.utc),
    )
    entry = DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=_provenance(),
    )

    render_entry(console, entry)
    output = console.export_text()

    assert "NOMINAL" in output
    assert "dummy-event-1" in output


def test_render_entries_handles_multiple_entries():
    console = Console(record=True, width=120)
    entries = [_conjunction_entry(Severity.WATCH), _conjunction_entry(Severity.CRITICAL, 1.0)]

    render_entries(entries, console=console)
    output = console.export_text()

    assert output.count("SAT-A vs SAT-B") == 2
    assert "AUTONOMOUS MANEUVER EXECUTED" in output
