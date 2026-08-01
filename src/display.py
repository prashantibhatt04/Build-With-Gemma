"""Rich-based terminal rendering for pipeline output. Display only - never
touches what gets written to the JSONL audit log (see logging_utils.py),
which stays byte-for-byte the same regardless of how this renders to the
screen.
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .schemas import DecisionLogEntry, Severity

SEVERITY_STYLE = {
    Severity.NOMINAL: "green",
    Severity.WATCH: "yellow",
    Severity.WARNING: "dark_orange",
    Severity.CRITICAL: "bold red",
}


def _severity_badge(severity: Severity) -> Text:
    return Text(f"[{severity.value.upper()}]", style=SEVERITY_STYLE[severity])


def render_entry(console: Console, entry: DecisionLogEntry) -> None:
    """Prints one pipeline result as a single color-coded summary line,
    plus a visually distinct panel for whichever maneuver state applies:
    blocked by budget, awaiting human approval, executed (autonomous or
    human-approved), or rejected by a human. See schemas.Decision /
    schemas.ManeuverApproval for what drives each state."""
    raw = entry.telemetry.raw_data
    badge = _severity_badge(entry.finding.severity)

    if "object_a_name" in raw and "object_b_name" in raw:
        subject = f"{raw['object_a_name']} vs {raw['object_b_name']} ({raw['min_distance_km']:.2f}km)"
    else:
        # Non-conjunction telemetry (e.g. DummyAdapter) has no object pair.
        subject = entry.telemetry.event_id

    console.print(badge, subject, "—", entry.decision.rationale)

    plan = entry.decision.maneuver_plan
    if plan is None:
        return

    decision = entry.decision
    approval = decision.maneuver_approval

    if decision.budget_insufficient:
        console.print(Panel(
            f"Required: ~{plan.magnitude_delta_v:.2f} m/s ({plan.direction}) - "
            "NOT executed, insufficient remaining delta-v budget.",
            title="BUDGET INSUFFICIENT — ESCALATE FOR REVIEW",
            title_align="left",
            border_style="yellow",
        ))
    elif decision.awaiting_human_approval:
        console.print(Panel(
            f"Proposed: {plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s delta-v "
            f"(target clearance {plan.target_clearance_km:.1f}km) - "
            "NOT executed yet, awaiting human approval.",
            title="AWAITING HUMAN APPROVAL",
            title_align="left",
            border_style="magenta",
        ))
    elif decision.verified_clearance is not None:
        clearance = decision.verified_clearance
        if approval is not None and approval.mode == "human":
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
    elif approval is not None and not approval.approved:
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
