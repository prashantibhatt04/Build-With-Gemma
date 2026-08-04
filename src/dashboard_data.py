"""Pure data transforms for scripts/dashboard.py - no Streamlit import here,
so these are directly unit-testable without simulating a UI (same reason
src/preflight.py's checks are separated from scripts/run_demo.py's printing).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from .display import classify_decision_status
from .schemas import DecisionLogEntry, Severity

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
        # Real, decision-relevant fact that used to reach Gemma's own
        # prompts (see pipeline.py) but never a human looking at this
        # table - a 25km miss distance means something very different 2
        # hours out than 2 days out, and there was previously no way to
        # tell which from the dashboard. Parsed to a real datetime (like
        # the timestamp column above) rather than left as the raw ISO
        # string, so the table sorts and displays it consistently -
        # conjunction-only, so None for decay/attitude rows.
        raw_tca = raw.get("time_of_closest_approach")
        rows.append({
            "timestamp": entry.telemetry.timestamp,
            "event_id": entry.telemetry.event_id,
            "source": entry.telemetry.source,
            "severity": entry.finding.severity.value,
            "action": entry.decision.action.value,
            "subject": subject,
            "time_of_closest_approach": datetime.fromisoformat(raw_tca) if raw_tca else None,
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
    """awaiting_human_approval entries, sorted by soonest real
    time_of_closest_approach first - the most time-urgent maneuver
    surfaces at the top of a real operator's queue instead of wherever it
    happened to land chronologically in the log. Every entry here is
    conjunction-shaped (awaiting_human_approval is only ever set when a
    maneuver_plan exists - see decide_node, src/pipeline.py), so
    time_of_closest_approach is always real and present; .get with a
    fallback just avoids a crash if that invariant is ever violated,
    sorting anything missing it to the front (a missing TCA is itself
    worth an operator's attention first)."""
    pending = [e for e in entries if e.decision.awaiting_human_approval]
    return sorted(pending, key=lambda e: e.telemetry.raw_data.get("time_of_closest_approach", ""))


def tca_urgency_label(time_of_closest_approach: str, now: Optional[datetime] = None) -> str:
    """A real, decision-relevant urgency label from a raw ISO
    time_of_closest_approach string - "TCA in 3h 12m" or "TCA already
    passed 1h 5m ago" - so an operator approving/rejecting a maneuver can
    see at a glance whether the event this maneuver is actually FOR has
    already happened, not just how close it eventually got. now defaults
    to the real current time; overridable for tests."""
    now = now or datetime.now(timezone.utc)
    tca = datetime.fromisoformat(time_of_closest_approach)
    delta = tca - now
    passed = delta.total_seconds() < 0
    hours, remainder = divmod(int(abs(delta).total_seconds()), 3600)
    minutes = remainder // 60
    if passed:
        return f"TCA already passed {hours}h {minutes}m ago"
    return f"TCA in {hours}h {minutes}m"


def filter_entries(
    entries: list[DecisionLogEntry],
    severities: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
) -> list[DecisionLogEntry]:
    """Narrows the "All decisions" table to a chosen severity/source
    subset - parity with scripts/api.py's GET /decisions, which already
    supports these same two filters but had no dashboard equivalent.
    Under continuous scheduled operation (scripts/scheduler.py) this
    table grows unbounded forever, and without any way to narrow it, the
    human UI degrades exactly when the product's headline feature
    (continuous background operation) is working as intended - real
    operator value, not just API parity for its own sake.

    An empty/None severities or sources means "no filter on that
    dimension" (matching the API's Optional query params), not "match
    nothing" - so the default (nothing selected in either dashboard
    multiselect) is the original unfiltered behavior, unchanged."""
    filtered = entries
    if severities:
        filtered = [e for e in filtered if e.finding.severity.value in severities]
    if sources:
        filtered = [e for e in filtered if e.telemetry.source in sources]
    return filtered


def needs_attention(entries: list[DecisionLogEntry]) -> list[DecisionLogEntry]:
    """Real CRITICAL findings with no maneuver-approval workflow of their
    own - decay/attitude hazards, where decide_node (src/pipeline.py)
    only ever computes a maneuver_plan for conjunction-shaped raw_data
    ("your perigee is too low" or "you're tumbling" has no avoidance burn
    to propose). Without this, a CRITICAL decay/attitude finding is
    logged with Action.ABORT and real Gemma narration but genuinely
    nothing else - no pending_approvals entry, no distinguishing flag -
    and sits as one row among NOMINAL/WATCH noise in the "All decisions"
    table unless an operator happens to filter for it.

    Deliberately excludes conjunction CRITICALs (maneuver_plan is not
    None) - those already have their own real workflow (pending_approvals
    for the human-approval path, or a full autonomous-execution/veto
    record when self-approved), so listing them here too would be a
    duplicate, not a genuinely different unmet need. Also excludes
    already-reviewed entries (human_reviewed), the same acknowledgment
    mechanism mark_reviewed already provides - once a human has looked at
    it, it should stop needing attention."""
    return [
        e for e in entries
        if e.finding.severity == Severity.CRITICAL
        and e.decision.maneuver_plan is None
        and not e.human_reviewed
    ]
