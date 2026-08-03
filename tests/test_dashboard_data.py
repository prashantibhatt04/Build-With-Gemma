"""Tests for src/dashboard_data.py - pure transforms, no Streamlit involved."""
from datetime import datetime, timezone

from src.dashboard_data import compute_metrics, entries_to_rows, pending_approvals
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


def _provenance(source: str = "gemma") -> GemmaProvenance:
    return GemmaProvenance(source=source, model_used="fake-model", latency_ms=1.0)


def _conjunction_entry(
    event_id: str,
    severity: Severity,
    min_distance_km: float = 50.0,
    decision_overrides: dict | None = None,
    human_reviewed: bool = False,
    rationale_source: str = "gemma",
) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "SAT-A",
            "object_b_id": "2", "object_b_name": "SAT-B",
            "min_distance_km": min_distance_km,
            "time_of_closest_approach": "2026-08-02T00:00:00+00:00",
            "relative_velocity_km_s": 5.0,
        },
    )
    finding = AnomalyFinding(
        event_id=event_id, severity=severity, description="Test finding.", confidence=0.8,
    )

    maneuver_plan = None
    if severity == Severity.CRITICAL:
        maneuver_plan = compute_avoidance_maneuver(
            object_a="1", object_b="2", min_distance_km=min_distance_km, relative_velocity_km_s=5.0,
        )

    decision_kwargs = dict(
        action="continue" if severity != Severity.CRITICAL else "abort",
        rationale="Test rationale.", made_at=datetime.now(timezone.utc),
        maneuver_plan=maneuver_plan,
    )
    if decision_overrides:
        decision_kwargs.update(decision_overrides)
    decision = Decision(**decision_kwargs)

    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=_provenance(rationale_source),
        human_reviewed=human_reviewed,
    )


def test_entries_to_rows_maps_severity_status_and_subject():
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)
    entry = _conjunction_entry(
        "event-1", Severity.CRITICAL, min_distance_km=2.0,
        decision_overrides={
            "maneuver_plan": plan,
            "verified_clearance": verify_maneuver(2.0, plan),
            "maneuver_approval": ManeuverApproval(
                mode="autonomous", approved=True, approved_at=datetime.now(timezone.utc), reason="r",
            ),
        },
    )

    rows = entries_to_rows([entry])

    assert len(rows) == 1
    row = rows[0]
    assert row["event_id"] == "event-1"
    assert row["severity"] == "critical"
    assert row["subject"] == "SAT-A vs SAT-B"
    assert row["min_distance_km"] == 2.0
    assert row["status"] == "executed (autonomous)"
    assert row["rationale_source"] == "gemma"
    assert row["human_reviewed"] is False
    # Real, decision-relevant fact - how close is only half the picture,
    # when matters just as much - parsed to a real datetime (not left as
    # the raw ISO string) so the dashboard table sorts/displays it like
    # the existing timestamp column.
    assert row["time_of_closest_approach"] == datetime(2026, 8, 2, tzinfo=timezone.utc)


def test_entries_to_rows_falls_back_to_event_id_for_non_conjunction():
    telemetry = TelemetryEvent(
        event_id="dummy-1", timestamp=datetime.now(timezone.utc),
        source="dummy-sensor", raw_data={"value": 1.0},
    )
    finding = AnomalyFinding(
        event_id="dummy-1", severity=Severity.NOMINAL, description="d", confidence=0.5,
    )
    decision = Decision(action="continue", rationale="r", made_at=datetime.now(timezone.utc))
    entry = DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
    )

    rows = entries_to_rows([entry])

    assert rows[0]["subject"] == "dummy-1"
    assert rows[0]["status"] == "no maneuver"
    assert rows[0]["min_distance_km"] is None


def test_entries_to_rows_shows_object_name_for_decay_hazard():
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
        event_id="decay-33821", severity=Severity.WATCH, description="d", confidence=0.9,
    )
    decision = Decision(action="continue", rationale="r", made_at=datetime.now(timezone.utc))
    entry = DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
    )

    rows = entries_to_rows([entry])

    assert rows[0]["subject"] == "COSMOS 2251 DEB"
    assert rows[0]["min_distance_km"] is None
    assert rows[0]["perigee_altitude_km"] == 415.9
    # TCA is a conjunction-only concept - a decay row has no closest
    # approach to a specific other object, so this must stay None rather
    # than crash or default to something misleading.
    assert rows[0]["time_of_closest_approach"] is None


def test_entries_to_rows_shows_object_name_for_attitude_hazard():
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
        event_id="attitude-99010", severity=Severity.CRITICAL, description="d", confidence=0.8,
    )
    decision = Decision(action="abort", rationale="r", made_at=datetime.now(timezone.utc))
    entry = DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
    )

    rows = entries_to_rows([entry])

    assert rows[0]["subject"] == "SYNTH-SAT-CRITICAL"
    assert rows[0]["min_distance_km"] is None
    assert rows[0]["perigee_altitude_km"] is None
    assert rows[0]["pointing_error_deg"] == 70.0


def test_compute_metrics_counts_every_status_bucket():
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)
    clearance = verify_maneuver(2.0, plan)
    now = datetime.now(timezone.utc)

    entries = [
        _conjunction_entry("watch-1", Severity.WATCH, rationale_source="fallback"),
        _conjunction_entry(
            "budget-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={"maneuver_plan": plan, "budget_insufficient": True},
        ),
        _conjunction_entry(
            "awaiting-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={"maneuver_plan": plan, "awaiting_human_approval": True},
        ),
        _conjunction_entry(
            "auto-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={
                "maneuver_plan": plan, "verified_clearance": clearance,
                "maneuver_approval": ManeuverApproval(mode="autonomous", approved=True, approved_at=now, reason="r"),
            },
        ),
        _conjunction_entry(
            "human-approved-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={
                "maneuver_plan": plan, "verified_clearance": clearance,
                "maneuver_approval": ManeuverApproval(
                    mode="human", approved=True, approved_by="alice", approved_at=now, reason="r",
                ),
            },
        ),
        _conjunction_entry(
            "vetoed-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={
                "maneuver_plan": plan,
                "maneuver_approval": ManeuverApproval(
                    mode="autonomous", approved=False, approved_by="Gemma", approved_at=now, reason="r",
                ),
            },
        ),
        _conjunction_entry(
            "rejected-1", Severity.CRITICAL, min_distance_km=2.0,
            decision_overrides={
                "maneuver_plan": plan,
                "maneuver_approval": ManeuverApproval(
                    mode="human", approved=False, approved_by="bob", approved_at=now, reason="r",
                ),
            },
        ),
    ]

    metrics = compute_metrics(entries)

    assert metrics["total"] == 7
    assert metrics["critical"] == 6
    assert metrics["budget_insufficient"] == 1
    assert metrics["awaiting_human_approval"] == 1
    assert metrics["executed_autonomous"] == 1
    assert metrics["executed_human_approved"] == 1
    assert metrics["vetoed_by_gemma"] == 1
    assert metrics["rejected_by_human"] == 1
    # 6 of 7 entries used the default "gemma" rationale source (watch-1 used "fallback").
    assert metrics["gemma_rationale_pct"] == (6 / 7 * 100)


def test_compute_metrics_handles_empty_list():
    metrics = compute_metrics([])
    assert metrics["total"] == 0
    assert metrics["gemma_rationale_pct"] == 0.0


def test_pending_approvals_returns_only_awaiting_entries():
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)
    awaiting = _conjunction_entry(
        "awaiting-1", Severity.CRITICAL, min_distance_km=2.0,
        decision_overrides={"maneuver_plan": plan, "awaiting_human_approval": True},
    )
    not_awaiting = _conjunction_entry("watch-1", Severity.WATCH)

    result = pending_approvals([awaiting, not_awaiting])

    assert result == [awaiting]
