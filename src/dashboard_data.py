"""Pure data transforms for scripts/dashboard.py - no Streamlit import here,
so these are directly unit-testable without simulating a UI (same reason
src/preflight.py's checks are separated from scripts/run_demo.py's printing).
"""
from __future__ import annotations

from .display import classify_decision_status
from .schemas import DecisionLogEntry

STATUS_LABELS = {
    None: "no maneuver",
    "budget_insufficient": "budget insufficient",
    "awaiting_human_approval": "awaiting human approval",
    "executed_autonomous": "executed (autonomous)",
    "executed_human_approved": "executed (human-approved)",
    "vetoed_by_gemma": "vetoed by Gemma",
    "rejected_by_human": "rejected by human",
    "unknown": "unknown",
}


def entries_to_rows(entries: list[DecisionLogEntry]) -> list[dict]:
    """DecisionLogEntry list -> flat dicts for a table. Uses
    classify_decision_status - the same classifier the terminal renderer
    (src/display.py) uses - as the single source of truth for maneuver
    state, so the dashboard can never disagree with the CLI about what
    state a decision is in."""
    rows = []
    for entry in entries:
        raw = entry.telemetry.raw_data
        status = classify_decision_status(entry.decision)
        if "object_a_name" in raw and "object_b_name" in raw:
            subject = f"{raw['object_a_name']} vs {raw['object_b_name']}"
        elif "object_name" in raw:
            # Decay (Phase 14) or attitude/pointing-loss (Phase 18) hazard -
            # single object, not a pair. Both shapes carry "object_name",
            # so this branch is safe for either without needing to
            # distinguish them further - unlike display.py's per-hazard
            # detail suffix, this is just the bare name.
            subject = raw["object_name"]
        else:
            subject = entry.telemetry.event_id
        rows.append({
            "timestamp": entry.telemetry.timestamp,
            "event_id": entry.telemetry.event_id,
            "source": entry.telemetry.source,
            "severity": entry.finding.severity.value,
            "action": entry.decision.action.value,
            "subject": subject,
            "min_distance_km": raw.get("min_distance_km"),
            "perigee_altitude_km": raw.get("perigee_altitude_km"),
            "pointing_error_deg": raw.get("pointing_error_deg"),
            "collision_probability": raw.get("collision_probability"),
            "severity_source": entry.finding.severity_source,
            "real_repropagated_min_distance_km": raw.get("real_repropagated_min_distance_km"),
            "status": STATUS_LABELS.get(status, status),
            "rationale_source": entry.rationale_provenance.source,
            "human_reviewed": entry.human_reviewed,
        })
    return rows


def compute_metrics(entries: list[DecisionLogEntry]) -> dict:
    """Aggregate counts across the full entry list for the dashboard's
    top metrics row."""
    total = len(entries)
    statuses = [classify_decision_status(e.decision) for e in entries]
    gemma_count = sum(1 for e in entries if e.rationale_provenance.source == "gemma")
    return {
        "total": total,
        "critical": sum(1 for e in entries if e.finding.severity.value == "critical"),
        "executed_autonomous": statuses.count("executed_autonomous"),
        "executed_human_approved": statuses.count("executed_human_approved"),
        "vetoed_by_gemma": statuses.count("vetoed_by_gemma"),
        "rejected_by_human": statuses.count("rejected_by_human"),
        "awaiting_human_approval": statuses.count("awaiting_human_approval"),
        "budget_insufficient": statuses.count("budget_insufficient"),
        "gemma_rationale_pct": (gemma_count / total * 100) if total else 0.0,
    }


def pending_approvals(entries: list[DecisionLogEntry]) -> list[DecisionLogEntry]:
    return [e for e in entries if e.decision.awaiting_human_approval]
