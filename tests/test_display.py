"""Tests for src/display.py. Renders to an in-memory Console(record=True)
so output can be asserted on without touching a real terminal."""
from datetime import datetime, timezone

from rich.console import Console

from src.display import classify_decision_status, render_entries, render_entry
from src.maneuver import compute_avoidance_maneuver, verify_maneuver
from src.schemas import (
    AnomalyFinding,
    Decision,
    DecisionLogEntry,
    GemmaProvenance,
    ManeuverApproval,
    ManeuverPlan,
    Severity,
    TelemetryEvent,
    VerifiedClearance,
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


def test_render_entry_shows_pc_when_severity_source_is_probability_of_collision():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.WARNING)
    entry.telemetry.raw_data["collision_probability"] = 2.5e-05
    entry.finding.severity_source = "probability-of-collision"

    render_entry(console, entry)
    output = console.export_text()

    assert "Pc=2.50e-05" in output


def test_render_entry_omits_pc_when_severity_source_is_distance_threshold():
    console = Console(record=True, width=120)
    entry = _conjunction_entry(Severity.WATCH)
    entry.finding.severity_source = "distance-threshold"

    render_entry(console, entry)
    output = console.export_text()

    assert "Pc=" not in output


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


def test_render_entry_shows_object_name_and_perigee_for_decay_hazard():
    """Decay hazard (Phase 14) events are single-object, not a pair - the
    subject line should show the real object name + perigee, not fall
    back to the raw event_id the way truly-generic telemetry does."""
    console = Console(record=True, width=120)
    telemetry = TelemetryEvent(
        event_id="decay-33821", timestamp=datetime.now(timezone.utc),
        source="celestrak-decay",
        raw_data={
            "object_id": "33821", "object_name": "COSMOS 2251 DEB",
            "perigee_altitude_km": 415.9, "apogee_altitude_km": 462.3,
            "bstar": 0.0008, "tle_epoch_age_hours": 12.0,
        },
    )
    finding = AnomalyFinding(
        event_id=telemetry.event_id, severity=Severity.WATCH,
        description="Test decay finding.", confidence=0.9,
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

    assert "COSMOS 2251 DEB" in output
    assert "perigee 416km" in output
    assert "decay-33821" not in output  # real subject shown, not the raw event_id


def test_render_entry_shows_object_name_and_pointing_error_for_attitude_hazard():
    """Attitude hazard (Phase 18) events also carry "object_name" like
    decay events do - must be distinguished by a field decay's shape
    doesn't have (pointing_error_deg), not misread as decay and crash
    trying to read a nonexistent perigee_altitude_km."""
    console = Console(record=True, width=120)
    telemetry = TelemetryEvent(
        event_id="attitude-99010", timestamp=datetime.now(timezone.utc),
        source="synthetic-attitude-fixture",
        raw_data={
            "object_id": "99010", "object_name": "SYNTH-SAT-CRITICAL",
            "pointing_error_deg": 70.0, "angular_rate_deg_s": 4.5,
            "solar_panel_power_pct": 22.0,
        },
    )
    finding = AnomalyFinding(
        event_id=telemetry.event_id, severity=Severity.CRITICAL,
        description="Test attitude finding.", confidence=0.8,
    )
    decision = Decision(
        action="abort", rationale="Recommendation: abort.",
        made_at=datetime.now(timezone.utc),
    )
    entry = DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=_provenance(),
    )

    render_entry(console, entry)
    output = console.export_text()

    assert "SYNTH-SAT-CRITICAL" in output
    assert "pointing error 70" in output
    assert "attitude-99010" not in output  # real subject shown, not the raw event_id


def _plan() -> ManeuverPlan:
    return compute_avoidance_maneuver(
        object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0,
    )


def _clearance() -> VerifiedClearance:
    return verify_maneuver(2.0, _plan())


def test_classify_decision_status_returns_none_for_non_critical():
    decision = Decision(action="continue", rationale="r", made_at=datetime.now(timezone.utc))
    assert classify_decision_status(decision) is None


def test_classify_decision_status_budget_insufficient():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(), budget_insufficient=True,
    )
    assert classify_decision_status(decision) == "budget_insufficient"


def test_classify_decision_status_awaiting_human_approval():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(), awaiting_human_approval=True,
    )
    assert classify_decision_status(decision) == "awaiting_human_approval"


def test_classify_decision_status_executed_autonomous():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(), verified_clearance=_clearance(),
        maneuver_approval=ManeuverApproval(
            mode="autonomous", approved=True, approved_at=datetime.now(timezone.utc), reason="r",
        ),
    )
    assert classify_decision_status(decision) == "executed_autonomous"


def test_classify_decision_status_executed_human_approved():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(), verified_clearance=_clearance(),
        maneuver_approval=ManeuverApproval(
            mode="human", approved=True, approved_by="alice",
            approved_at=datetime.now(timezone.utc), reason="r",
        ),
    )
    assert classify_decision_status(decision) == "executed_human_approved"


def test_classify_decision_status_vetoed_by_gemma():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(),
        maneuver_approval=ManeuverApproval(
            mode="autonomous", approved=False, approved_by="Gemma",
            approved_at=datetime.now(timezone.utc), reason="r",
        ),
    )
    assert classify_decision_status(decision) == "vetoed_by_gemma"


def test_classify_decision_status_rejected_by_human():
    decision = Decision(
        action="abort", rationale="r", made_at=datetime.now(timezone.utc),
        maneuver_plan=_plan(),
        maneuver_approval=ManeuverApproval(
            mode="human", approved=False, approved_by="bob",
            approved_at=datetime.now(timezone.utc), reason="r",
        ),
    )
    assert classify_decision_status(decision) == "rejected_by_human"


def test_render_entries_handles_multiple_entries():
    console = Console(record=True, width=120)
    entries = [_conjunction_entry(Severity.WATCH), _conjunction_entry(Severity.CRITICAL, 1.0)]

    render_entries(entries, console=console)
    output = console.export_text()

    assert output.count("SAT-A vs SAT-B") == 2
    assert "AUTONOMOUS MANEUVER EXECUTED" in output
