"""Rich-based terminal rendering for pipeline output. Display only - never
touches what gets written to the JSONL audit log (see logging_utils.py),
which stays byte-for-byte the same regardless of how this renders to the
screen.
"""
from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .schemas import Decision, DecisionLogEntry, Severity

SEVERITY_STYLE = {
    Severity.NOMINAL: "green",
    Severity.WATCH: "yellow",
    Severity.WARNING: "dark_orange",
    Severity.CRITICAL: "bold red",
}


def _severity_badge(severity: Severity) -> Text:
    return Text(f"[{severity.value.upper()}]", style=SEVERITY_STYLE[severity])


def classify_decision_status(decision: Decision) -> Optional[str]:
    """Classifies which of the mutually-exclusive CRITICAL-maneuver states
    a Decision is in: "budget_insufficient", "awaiting_human_approval",
    "executed_autonomous", "executed_human_approved", "vetoed_by_gemma", or
    "rejected_by_human". Returns None when there's no maneuver at all
    (non-CRITICAL severity).

    Single source of truth for this branching, shared by render_entry
    below and scripts/dashboard.py's risk board - both need to categorize
    the same states and should never be able to drift out of sync with
    each other.
    """
    if decision.maneuver_plan is None:
        return None
    if decision.budget_insufficient:
        return "budget_insufficient"
    if decision.awaiting_human_approval:
        return "awaiting_human_approval"
    if decision.verified_clearance is not None:
        approval = decision.maneuver_approval
        if approval is not None and approval.mode == "human":
            return "executed_human_approved"
        return "executed_autonomous"
    approval = decision.maneuver_approval
    if approval is not None and not approval.approved and approval.mode == "autonomous":
        return "vetoed_by_gemma"
    if approval is not None and not approval.approved:
        return "rejected_by_human"
    # decide_node always resolves a CRITICAL, budget-affordable maneuver
    # into exactly one of the states above - reachable here only if that
    # invariant is ever violated.
    return "unknown"


def render_entry(console: Console, entry: DecisionLogEntry) -> None:
    """Prints one pipeline result as a single color-coded summary line,
    plus a visually distinct panel for whichever maneuver state applies:
    blocked by budget, awaiting human approval, executed (autonomous or
    human-approved), vetoed by Gemma's autonomous safety review, or
    rejected by a human. See schemas.Decision / schemas.ManeuverApproval
    for what drives each state."""
    raw = entry.telemetry.raw_data
    badge = _severity_badge(entry.finding.severity)

    if "object_a_name" in raw and "object_b_name" in raw:
        subject = f"{raw['object_a_name']} vs {raw['object_b_name']} ({raw['min_distance_km']:.2f}km)"
        if "time_of_closest_approach" in raw:
            # A 25km miss reads very differently 2 hours out vs. 2 days
            # out - this was already sent to Gemma's own prompts (see
            # pipeline.py) but never shown to the human reading this line.
            subject += f" [TCA {raw['time_of_closest_approach']}]"
        if entry.finding.severity_source == "probability-of-collision":
            # A real Space-Track CDM was matched to this conjunction (see
            # src/ingestion/cdm_enrichment.py) - severity came from a real
            # Pc, not the distance threshold, and that's worth surfacing
            # right in the subject line rather than only in the raw log.
            subject += f" [Pc={raw['collision_probability']:.2e}]"
    elif "perigee_altitude_km" in raw:
        # Decay hazard (Phase 14) - single object, not a pair. Checked by
        # its own distinguishing field, not just "object_name", since the
        # attitude hazard (Phase 18) also has a single "object_name" but a
        # different shape entirely.
        subject = f"{raw['object_name']} (perigee {raw['perigee_altitude_km']:.0f}km)"
    elif "pointing_error_deg" in raw:
        # Attitude/pointing-loss hazard (Phase 18) - synthetic-only, see
        # src/ingestion/attitude_adapter.py's module docstring for why.
        subject = f"{raw['object_name']} (pointing error {raw['pointing_error_deg']:.0f}°)"
    else:
        # Non-conjunction, non-decay, non-attitude telemetry (e.g.
        # DummyAdapter) has no object identity at all.
        subject = entry.telemetry.event_id

    console.print(badge, subject, "—", entry.decision.rationale)

    if "real_repropagated_min_distance_km" in raw:
        # A real, independent physics cross-check (src/ingestion/
        # historical_adapter.py's real_repropagate_event) - genuinely
        # re-derived from real historical TLEs, not the documented number
        # this event was actually classified on above.
        console.print(
            f"    [dim]Real re-propagation cross-check: "
            f"{raw['real_repropagated_min_distance_km']:.3f}km "
            f"(documented: {raw['min_distance_km']:.3f}km)[/dim]"
        )
    elif "real_repropagation_error" in raw:
        console.print(f"    [dim]Real re-propagation cross-check failed: {raw['real_repropagation_error']}[/dim]")

    plan = entry.decision.maneuver_plan
    if plan is None:
        return

    decision = entry.decision
    approval = decision.maneuver_approval
    status = classify_decision_status(decision)

    if status == "budget_insufficient":
        console.print(Panel(
            f"Required: ~{plan.magnitude_delta_v:.2f} m/s ({plan.direction}) - "
            "NOT executed, insufficient remaining delta-v budget.",
            title="BUDGET INSUFFICIENT — ESCALATE FOR REVIEW",
            title_align="left",
            border_style="yellow",
        ))
    elif status == "awaiting_human_approval":
        console.print(Panel(
            f"Proposed: {plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s delta-v "
            f"(target clearance {plan.target_clearance_km:.1f}km) - "
            "NOT executed yet, awaiting human approval.",
            title="AWAITING HUMAN APPROVAL",
            title_align="left",
            border_style="magenta",
        ))
    elif status in ("executed_autonomous", "executed_human_approved"):
        clearance = decision.verified_clearance
        if status == "executed_human_approved":
            title = f"MANEUVER EXECUTED (approved by {approval.approved_by})"
            border = "green"
        else:
            title = "AUTONOMOUS MANEUVER EXECUTED"
            border = "red"
        console.print(Panel(
            f"Direction: {plan.direction}\n"
            f"Delta-v: ~{plan.magnitude_delta_v:.2f} m/s\n"
            f"Verified new separation: {clearance.new_min_distance_km:.2f}km "
            f"(cleared={clearance.cleared})",
            title=title,
            title_align="left",
            border_style=border,
        ))
    elif status == "vetoed_by_gemma":
        # Gemma vetoed a maneuver that deterministic physics already
        # verified as safe (see pipeline._maneuver_veto_check) - distinct
        # from a human rejecting a cloud-pending proposal below, so it gets
        # its own title/detail rather than reusing "MANEUVER REJECTED".
        console.print(Panel(
            f"Proposed maneuver ({plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s), "
            "already independently verified safe by deterministic physics, was "
            f"VETOED by {approval.approved_by} before execution.\n"
            f"Reason: {approval.reason}",
            title="MANEUVER VETOED — AUTONOMOUS SAFETY REVIEW",
            title_align="left",
            border_style="red",
        ))
    elif status == "rejected_by_human":
        console.print(Panel(
            f"Proposed maneuver ({plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s) "
            f"was REJECTED by {approval.approved_by} - not executed.",
            title="MANEUVER REJECTED",
            title_align="left",
            border_style="red",
        ))


def render_entries(entries: list[DecisionLogEntry], console: Console | None = None) -> None:
    console = console or Console()
    for entry in entries:
        render_entry(console, entry)
