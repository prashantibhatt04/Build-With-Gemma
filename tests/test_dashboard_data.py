"""Tests for src/dashboard_data.py - pure transforms, no Streamlit involved."""
from datetime import datetime, timezone

from src.dashboard_data import (
    compute_metrics,
    entries_to_rows,
    filter_entries,
    needs_attention,
    pending_approvals,
    tca_urgency_label,
)
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


def test_pending_approvals_sorted_by_soonest_time_of_closest_approach():
    """Real behavior this guards: the most time-urgent maneuver must
    surface at the TOP of a real operator's approval queue, not wherever
    it happened to land chronologically in the log - an operator working
    top-to-bottom should always see the soonest TCA first."""
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)

    soon = _conjunction_entry(
        "soon", Severity.CRITICAL, min_distance_km=2.0,
        decision_overrides={"maneuver_plan": plan, "awaiting_human_approval": True},
    )
    soon.telemetry.raw_data["time_of_closest_approach"] = "2026-08-02T01:00:00+00:00"

    later = _conjunction_entry(
        "later", Severity.CRITICAL, min_distance_km=2.0,
        decision_overrides={"maneuver_plan": plan, "awaiting_human_approval": True},
    )
    later.telemetry.raw_data["time_of_closest_approach"] = "2026-08-03T00:00:00+00:00"

    result = pending_approvals([later, soon])  # deliberately logged out of TCA order

    assert result == [soon, later]


def test_tca_urgency_label_reports_time_remaining_for_a_future_tca():
    now = datetime(2026, 8, 2, 0, 0, 0, tzinfo=timezone.utc)

    label = tca_urgency_label("2026-08-02T03:12:00+00:00", now=now)

    assert label == "TCA in 3h 12m"


def test_tca_urgency_label_flags_a_tca_that_has_already_passed():
    """Real gap this closes: an operator approving/rejecting a maneuver
    previously had no way to tell from the pending-approval card whether
    the conjunction it's FOR has already happened - approving a maneuver
    after its own TCA has no effect."""
    now = datetime(2026, 8, 2, 5, 0, 0, tzinfo=timezone.utc)

    label = tca_urgency_label("2026-08-02T03:12:00+00:00", now=now)

    assert label == "TCA already passed 1h 48m ago"


def _dummy_entry(event_id: str, severity: Severity, source: str) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source=source, raw_data={"value": 1.0},
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="d", confidence=0.5)
    decision = Decision(action="continue", rationale="r", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
    )


def test_filter_entries_with_no_filters_returns_everything_unchanged():
    """Real behavior this guards: the default dashboard state (nothing
    selected in either multiselect) must be the original unfiltered
    "All decisions" table, not an empty one."""
    entries = [
        _dummy_entry("a", Severity.CRITICAL, "celestrak"),
        _dummy_entry("b", Severity.WATCH, "dummy-sensor"),
    ]

    assert filter_entries(entries) == entries
    assert filter_entries(entries, severities=[], sources=[]) == entries


def test_filter_entries_narrows_by_severity():
    critical = _dummy_entry("a", Severity.CRITICAL, "celestrak")
    watch = _dummy_entry("b", Severity.WATCH, "celestrak")

    result = filter_entries([critical, watch], severities=["critical"])

    assert result == [critical]


def test_filter_entries_narrows_by_source():
    celestrak_entry = _dummy_entry("a", Severity.WATCH, "celestrak")
    dummy_entry = _dummy_entry("b", Severity.WATCH, "dummy-sensor")

    result = filter_entries([celestrak_entry, dummy_entry], sources=["dummy-sensor"])

    assert result == [dummy_entry]


def test_filter_entries_combines_severity_and_source_as_an_and():
    matches_both = _dummy_entry("a", Severity.CRITICAL, "celestrak")
    wrong_severity = _dummy_entry("b", Severity.WATCH, "celestrak")
    wrong_source = _dummy_entry("c", Severity.CRITICAL, "dummy-sensor")

    result = filter_entries(
        [matches_both, wrong_severity, wrong_source], severities=["critical"], sources=["celestrak"],
    )

    assert result == [matches_both]


def test_filter_entries_multiple_selected_values_are_an_or_within_each_dimension():
    critical = _dummy_entry("a", Severity.CRITICAL, "celestrak")
    watch = _dummy_entry("b", Severity.WATCH, "celestrak")
    nominal = _dummy_entry("c", Severity.NOMINAL, "celestrak")

    result = filter_entries([critical, watch, nominal], severities=["critical", "watch"])

    assert result == [critical, watch]


def _decay_entry(event_id: str, severity: Severity, human_reviewed: bool = False) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak-decay",
        raw_data={
            "object_id": "33821", "object_name": "COSMOS 2251 DEB",
            "perigee_altitude_km": 150.0, "apogee_altitude_km": 200.0,
            "bstar": 0.001, "tle_epoch_age_hours": 12.0,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="d", confidence=0.9)
    decision = Decision(action="abort", rationale="r", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
        human_reviewed=human_reviewed,
    )


def _attitude_entry(event_id: str, severity: Severity) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="synthetic-attitude-fixture",
        raw_data={
            "object_id": "99010", "object_name": "SYNTH-SAT",
            "pointing_error_deg": 70.0, "angular_rate_deg_s": 4.5, "solar_panel_power_pct": 22.0,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="d", confidence=0.8)
    decision = Decision(action="abort", rationale="r", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision, rationale_provenance=_provenance(),
    )


def test_needs_attention_includes_critical_decay_finding():
    """Real gap this closes: a CRITICAL decay finding gets Action.ABORT
    and real Gemma narration but no maneuver/approval machinery at all
    (there's no avoidance burn for "your perigee is too low") - without
    this, it was indistinguishable from NOMINAL/WATCH noise in the "All
    decisions" table."""
    entry = _decay_entry("decay-1", Severity.CRITICAL)

    assert needs_attention([entry]) == [entry]


def test_needs_attention_includes_critical_attitude_finding():
    entry = _attitude_entry("attitude-1", Severity.CRITICAL)

    assert needs_attention([entry]) == [entry]


def test_needs_attention_excludes_conjunction_critical_with_its_own_maneuver_workflow():
    """Conjunction CRITICALs already have a real workflow of their own
    (pending_approvals, or a full autonomous-execution/veto record) -
    listing them here too would be a duplicate, not a genuinely different
    unmet need."""
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=2.0, relative_velocity_km_s=5.0)
    entry = _conjunction_entry(
        "conj-1", Severity.CRITICAL, min_distance_km=2.0,
        decision_overrides={"maneuver_plan": plan, "verified_clearance": verify_maneuver(2.0, plan)},
    )

    assert needs_attention([entry]) == []


def test_needs_attention_excludes_already_reviewed_entries():
    """Once a human has acknowledged (mark_reviewed) a decay/attitude
    CRITICAL finding, it should stop needing attention - the same
    acknowledgment mechanism the Inspect panel already provides."""
    entry = _decay_entry("decay-1", Severity.CRITICAL, human_reviewed=True)

    assert needs_attention([entry]) == []


def test_needs_attention_excludes_non_critical_severities():
    entry = _decay_entry("decay-1", Severity.WARNING)

    assert needs_attention([entry]) == []
