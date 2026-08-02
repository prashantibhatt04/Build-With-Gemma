#!/usr/bin/env python3
"""Interactive, step-by-step guided demo of everything built in this
project so far. Each step explains what it demonstrates and why, in plain
language - meant to be understandable by someone reading this on GitHub
with no other context, not just someone who was in the room while it was
built.

Run interactively (pauses between steps, press Enter to continue):
    python scripts/run_demo.py

Run straight through with no pauses (for CI / quick smoke-testing):
    python scripts/run_demo.py --auto

Adding a new phase: append a new Step(...) to STEPS below, in order, with
its own action function. Keep the explanation self-contained - assume the
reader has no other context than what's printed by the steps before it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from src.config import Settings, settings
from src.display import render_entries
from src.gemma_client import GemmaClient
from src.ingestion.celestrak_adapter import CelesTrakAdapter
from src.ingestion.decay_adapter import DecayRiskAdapter
from src.ingestion.historical_adapter import HistoricalReplayAdapter
from src.ingestion.synthetic_adapter import SyntheticCriticalAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import run_once
from src.preflight import run_all_checks
from src.schemas import DecisionLogEntry


@dataclass
class DemoContext:
    """Shared state later steps can read from earlier ones."""

    console: Console
    auto: bool = False
    all_entries: list[DecisionLogEntry] = field(default_factory=list)
    reviewable_event_id: Optional[str] = None
    # Unique per script invocation - see SyntheticCriticalAdapter's
    # docstring for why this matters (repeat demo runs otherwise collide
    # on the same hardcoded event_id, and mark_reviewed/find_entry return
    # the FIRST match by event_id, not the most recent one).
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class Step:
    phase: str
    title: str
    explanation: str
    action: Callable[[DemoContext], None]


# ---------------------------------------------------------------------------
# Step actions
# ---------------------------------------------------------------------------

def _step_preflight(ctx: DemoContext) -> None:
    results = run_all_checks(settings)
    table = Table(show_header=True)
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for r in results:
        status = "[green]PASS[/green]" if r.ok else "[red]FAIL[/red]"
        table.add_row(r.name, status, r.detail)
    ctx.console.print(table)


def _step_live_orbital_data(ctx: DemoContext) -> None:
    # Real multi-group screening (Phase 10) - default groups are real
    # crewed stations vs. a real debris field, cross-screened for actual
    # current conjunctions, not a small staged pair. See
    # CelesTrakAdapter's class docstring for how this stays fast.
    adapter = CelesTrakAdapter()
    entries = run_once(adapter=adapter, limit=2)
    stats = adapter.last_scan_stats
    if stats:
        ctx.console.print(
            f"Screened {stats['total_pairs_screened']} real pairwise conjunctions across "
            f"{stats['total_objects']} real objects from {', '.join(stats['groups'])} "
            f"({stats['pairs_refined']} closest-by-coarse-distance refined to full precision)."
        )
    render_entries(entries, console=ctx.console)
    ctx.all_entries.extend(entries)


def _step_critical_maneuver_and_budget(ctx: DemoContext) -> None:
    tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)
    adapter = SyntheticCriticalAdapter(run_id=ctx.run_id, id_prefix="conj-run-demo")
    entries = run_once(adapter=adapter, budget_tracker=tracker, limit=4)
    render_entries(entries, console=ctx.console)
    ctx.all_entries.extend(entries)

    # Whether any of the above end up autonomous vs. awaiting human
    # approval is driven entirely by this machine's configured Gemma
    # backend (see decide_node) - nothing hardcoded here. If any are
    # pending, resolve them live: prompt for approve/reject when
    # interactive, auto-approve in --auto mode so CI doesn't hang.
    logger = DecisionLogger()
    for i, entry in enumerate(entries):
        if not entry.decision.awaiting_human_approval:
            continue
        ctx.console.print()
        plan = entry.decision.maneuver_plan
        ctx.console.print(Panel(
            f"Event: {entry.telemetry.event_id}\n"
            f"Proposed: {plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s delta-v\n"
            f"Target clearance: {plan.target_clearance_km:.1f} km\n"
            "Ground control (cloud backend) is reachable, so this maneuver "
            "will not execute without your approval.",
            title="HUMAN APPROVAL REQUIRED",
            title_align="left",
            border_style="magenta",
        ))
        approved = True if ctx.auto else Confirm.ask("Approve this maneuver?", default=True)
        updated = logger.approve_maneuver(
            entry.telemetry.event_id, approved=approved, approved_by="demo-operator",
        )
        entries[i] = updated
        ctx.all_entries[ctx.all_entries.index(entry)] = updated

        verdict = "[green]APPROVED[/green]" if approved else "[red]REJECTED[/red]"
        ctx.console.print(f"{verdict} by demo-operator.")
        if updated.decision.verified_clearance is not None:
            vc = updated.decision.verified_clearance
            ctx.console.print(f"Verified new separation: {vc.new_min_distance_km:.2f}km (cleared={vc.cleared})")

    # The last synthetic event is always the budget-insufficient case given
    # the fixed budget/delta-v math above - a good candidate for the
    # separate post-hoc human-review step below.
    ctx.reviewable_event_id = entries[-1].telemetry.event_id


def _step_historical_replay(ctx: DemoContext) -> None:
    adapter = HistoricalReplayAdapter(run_id=ctx.run_id)
    entries = run_once(adapter=adapter, limit=1)
    entry = entries[0]
    raw = entry.telemetry.raw_data

    ctx.console.print(Panel(
        f"{raw['historical_event']}\n\n"
        f"Source: {raw['historical_source']}\n\n"
        f"What actually happened: {raw['historical_actual_outcome']}",
        title="HISTORICAL RECORD — NOT A LIVE RISK",
        title_align="left",
        border_style="cyan",
    ))
    render_entries(entries, console=ctx.console)
    ctx.console.print(
        f"\nThis system's deterministic threshold (<5km = CRITICAL) classified the "
        f"real, documented {raw['min_distance_km'] * 1000:.0f}m prediction as "
        f"[bold red]{entry.finding.severity.value.upper()}[/bold red] - correctly "
        "and unambiguously, using the exact same math as every other event in "
        "this walkthrough. Nothing was special-cased for this replay."
    )
    ctx.all_entries.extend(entries)


def _step_decay_risk(ctx: DemoContext) -> None:
    adapter = DecayRiskAdapter(sample_size=200)
    entries = run_once(adapter=adapter, limit=3)
    render_entries(entries, console=ctx.console)
    worst = entries[0]
    raw = worst.telemetry.raw_data
    ctx.console.print(
        f"\nLowest real perigee in this scan: {raw['object_name']} at "
        f"{raw['perigee_altitude_km']:.0f} km, classified "
        f"[bold]{worst.finding.severity.value.upper()}[/bold] - the same "
        "deterministic threshold used for every decay assessment, real data "
        "or not. Real debris fields tend to sit in a stable-ish altitude "
        "band by now (the lowest-perigee fragments already decayed away "
        "years ago), so WATCH is typically the most severe outcome this "
        "specific real scan finds on any given run - same 'real data rarely "
        "produces the most severe case on demand' pattern as CRITICAL "
        "conjunctions."
    )
    ctx.all_entries.extend(entries)


def _step_failover(ctx: DemoContext) -> None:
    if settings.gemma_backend == "ollama":
        # This machine is configured for a LOCAL-only demo - deliberately
        # breaking Ollama here would make a real call out to the cloud
        # backend, which defeats the point of a local-only run. Skip
        # rather than silently touch the network anyway.
        ctx.console.print(
            "[dim]Skipped - this run is configured for local-only (Ollama). "
            "Demonstrating this step here would require a real call to the "
            "cloud backend, which a local-only demo should never do. Run "
            "with GEMMA_BACKEND=api configured to see this step.[/dim]"
        )
        return

    broken = Settings(
        gemma_backend="ollama", gemma_model=settings.gemma_model,
        ollama_host="http://localhost:1",  # intentionally unreachable
        gemma_api_key=settings.gemma_api_key, gemma_model_api=settings.gemma_model_api,
        log_dir=settings.log_dir, delta_v_budget_m_s=settings.delta_v_budget_m_s,
    )
    client = GemmaClient(settings=broken)
    try:
        # The broken local backend fails near-instantly (connection
        # refused), so this timeout only bounds the real fallback (cloud)
        # attempt - and the hosted API's latency varies a lot in practice
        # (observed 2-35s+ for the same trivial prompt), so a short
        # timeout here risks reporting a working fallback as unreachable.
        text = client.generate(prompt="Reply with the single word: ok", timeout=45)
        ctx.console.print(
            f"Local Ollama was intentionally made unreachable for this step. "
            f"Backend that actually answered: [bold]{client.last_backend_used}[/bold]"
        )
        ctx.console.print(f"Response: {text!r}")
    except Exception as exc:  # noqa: BLE001 - demo step, report and move on
        ctx.console.print(f"[yellow]Both backends unreachable in this environment: {exc}[/yellow]")


def _step_human_review(ctx: DemoContext) -> None:
    if not ctx.reviewable_event_id:
        ctx.console.print("[yellow]No event available to mark reviewed - skipped.[/yellow]")
        return
    logger = DecisionLogger()
    updated = logger.mark_reviewed(ctx.reviewable_event_id, reviewed_by="demo-reviewer")
    ctx.console.print(f"Marked [bold]{ctx.reviewable_event_id}[/bold] as reviewed.")
    ctx.console.print(
        f"human_reviewed={updated.human_reviewed}  "
        f"reviewed_by={updated.reviewed_by!r}  at={updated.human_reviewed_at}"
    )


def _step_audit_trail(ctx: DemoContext) -> None:
    log_files = sorted(Path(settings.log_dir).glob("decisions-*.jsonl"))
    if not log_files:
        ctx.console.print("[yellow]No log file found yet.[/yellow]")
        return
    lines = log_files[-1].read_text().strip().splitlines()
    ctx.console.print(f"Log file: {log_files[-1]} ({len(lines)} lines total)")
    if lines:
        ctx.console.print("Most recent entry (raw, exactly as persisted):")
        ctx.console.print_json(data=json.loads(lines[-1]))


def _step_test_suite(ctx: DemoContext) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True,
    )
    tail = result.stdout.strip().splitlines()[-3:] if result.stdout else ["(no output)"]
    for line in tail:
        ctx.console.print(line)


def _step_summary(ctx: DemoContext) -> None:
    entries = ctx.all_entries
    if not entries:
        ctx.console.print("[yellow]No events recorded this run.[/yellow]")
        return

    table = Table(show_header=True)
    table.add_column("Severity")
    table.add_column("Count")
    by_severity = Counter(e.finding.severity.value for e in entries)
    for severity, count in sorted(by_severity.items()):
        table.add_row(severity, str(count))
    ctx.console.print(table)

    gemma_count = sum(1 for e in entries if e.rationale_provenance.source == "gemma")
    fallback_count = sum(1 for e in entries if e.rationale_provenance.source == "fallback")
    maneuvers_executed = sum(1 for e in entries if e.decision.verified_clearance is not None)
    maneuvers_blocked = sum(1 for e in entries if e.decision.budget_insufficient)
    # Rejected maneuvers split by WHO rejected them - a Gemma veto (mode=
    # "autonomous") is a different situation from a human rejecting a
    # cloud-pending proposal (mode="human"), so they're reported separately
    # rather than lumped into one "rejected" count (see display.py).
    gemma_vetoed = sum(
        1 for e in entries
        if e.decision.maneuver_approval is not None
        and not e.decision.maneuver_approval.approved
        and e.decision.maneuver_approval.mode == "autonomous"
    )
    human_rejected = sum(
        1 for e in entries
        if e.decision.maneuver_approval is not None
        and not e.decision.maneuver_approval.approved
        and e.decision.maneuver_approval.mode == "human"
    )
    still_pending = sum(1 for e in entries if e.decision.awaiting_human_approval)
    autonomous = sum(
        1 for e in entries
        if e.decision.maneuver_approval is not None
        and e.decision.maneuver_approval.mode == "autonomous"
        and e.decision.maneuver_approval.approved
    )
    human_approved = sum(
        1 for e in entries
        if e.decision.maneuver_approval is not None
        and e.decision.maneuver_approval.mode == "human"
        and e.decision.maneuver_approval.approved
    )
    ctx.console.print(
        f"Total events: {len(entries)}  |  Gemma rationale: {gemma_count}  |  "
        f"Fallback rationale: {fallback_count}"
    )
    ctx.console.print(
        f"Maneuvers executed: {maneuvers_executed} (autonomous: {autonomous}, human-approved: {human_approved})  "
        f"|  Blocked by budget: {maneuvers_blocked}  |  Vetoed by Gemma: {gemma_vetoed}  "
        f"|  Rejected by human: {human_rejected}"
        + (f"  |  Still awaiting approval: {still_pending}" if still_pending else "")
    )


# ---------------------------------------------------------------------------
# Steps, in order. Add new phases here - one Step per phase (or sub-part of
# a phase worth explaining on its own), appended to the end.
# ---------------------------------------------------------------------------

STEPS: list[Step] = [
    Step(
        phase="Setup",
        title="Preflight check",
        explanation=(
            "Before demoing anything, confirm the environment is actually "
            "configured correctly: which Gemma backend (local Ollama, or a "
            "hosted cloud API) is active, whether it's reachable right now, "
            "and whether the audit log directory can be written to. If this "
            "fails, the rest of the demo still runs, but Gemma calls will "
            "fall back to plain deterministic text instead of AI-generated "
            "explanations."
        ),
        action=_step_preflight,
    ),
    Step(
        phase="Phase 2, 10",
        title="Real orbital data: CelesTrak + Skyfield/SGP4, cross-group screening",
        explanation=(
            "This project tracks satellite/debris collision risk. This step "
            "fetches REAL satellite tracking data (TLEs - Two-Line Elements) "
            "live from CelesTrak - by default, real crewed space stations "
            "(ISS, Tiangong, ...) and a real debris field (fragments from "
            "the 2009 Cosmos 2251/Iridium 33 collision) - then screens "
            "EVERY pairwise conjunction across both groups combined, not "
            "just within one, using REAL orbital mechanics (a two-pass "
            "coarse/fine search with Skyfield's SGP4 propagator) to find "
            "the closest predicted approach between each pair over the "
            "next 48 hours. Nothing here is simulated - these are real "
            "objects currently in orbit. Screening every pair this "
            "precisely doesn't scale on its own, so each object's coarse "
            "trajectory is computed once and reused across all its pairs, "
            "and only the closest candidates by that coarse estimate get "
            "the expensive precise refinement - watch the line below for "
            "exactly how many objects/pairs were actually screened."
        ),
        action=_step_live_orbital_data,
    ),
    Step(
        phase="Phases 0-3, 5, 8-9",
        title="CRITICAL conjunctions: maneuver, verification, budget, and approval (human or Gemma)",
        explanation=(
            "Severity (NOMINAL/WATCH/WARNING/CRITICAL) and the resulting "
            "action are decided by a plain distance threshold, NOT by the "
            "AI model - this needs to be reliable, so severity/action are "
            "never Gemma's call. Real orbital data rarely produces a "
            "CRITICAL case (<5km predicted separation) on demand, so this "
            "step uses 4 synthetic CRITICAL-range conjunctions (clearly "
            "labeled as synthetic in their source field) to demonstrate "
            "that path on purpose. For each one, a simplified avoidance "
            "maneuver is calculated deterministically. If the shared "
            "delta-v budget (a stand-in for real spacecraft fuel limits) "
            "can't afford it, the system says so plainly and escalates for "
            "review rather than pretending to act - watch the last of the "
            "4 events hit this. Otherwise, what happens next depends on "
            "THIS MACHINE'S configured Gemma backend, which this system "
            "treats as a proxy for whether ground control is currently "
            "reachable: with a LOCAL backend (Ollama), the maneuver is "
            "independently re-verified by deterministic physics, and THEN "
            "Gemma itself gets those same numbers and issues a real "
            "GO/NO-GO verdict - standing in for the unavailable human, "
            "since a real probe often can't wait for one (light-delay, "
            "communication blackout). Gemma can only make this MORE "
            "conservative than the physics check, never less: it can veto "
            "an already-verified-safe maneuver, but it never gets to "
            "approve one the physics hasn't already cleared. With the "
            "CLOUD backend (a hosted API), the maneuver is calculated but "
            "held pending instead - ground control is reachable, so a real "
            "human must explicitly approve it below before it executes and "
            "gets verified."
        ),
        action=_step_critical_maneuver_and_budget,
    ),
    Step(
        phase="Phase 12",
        title="Historical replay: would this system have caught a real collision?",
        explanation=(
            "Everything above uses live or synthetic data. This step "
            "instead replays a REAL, documented historical conjunction "
            "through the exact same pipeline, unmodified - the 2009 "
            "Iridium 33/Cosmos 2251 collision, the first confirmed "
            "accidental collision between two intact satellites. The "
            "numbers aren't invented: CelesTrak's own account of this "
            "event (celestrak.org/events/collision/) documents that its "
            "SOCRATES conjunction-screening system predicted a 584m "
            "closest approach in its final report before the collision - "
            "and had predicted this same conjunction in all 14 reports "
            "issued that week. It just never made the priority list (rank "
            "#152 that day) and nobody acted on it - a triage failure, not "
            "a detection failure. This step feeds that real 584m number "
            "into this system's ordinary, unmodified severity threshold "
            "and watches what happens - no special-casing for this being "
            "a historical replay."
        ),
        action=_step_historical_replay,
    ),
    Step(
        phase="Phase 14",
        title="A second real hazard: orbital decay / re-entry risk",
        explanation=(
            "Everything above is about collision risk between two objects. "
            "This step proves the pipeline isn't conjunction-specific by "
            "construction (see schemas.py's own docstring: telemetry/"
            "finding/decision are deliberately 'idea-agnostic') - it screens "
            "REAL objects individually (not pairs) for orbital decay risk, "
            "using real perigee altitude and BSTAR drag term Skyfield's SGP4 "
            "model already parses from the SAME TLE data this project "
            "already fetches. No new data source, no new credentials. An "
            "object's perigee altitude alone is a real, well-established "
            "decay signal - below ~200km, real objects reliably reenter "
            "within days to weeks, regardless of other factors. Unlike "
            "conjunctions, a CRITICAL decay finding does NOT trigger "
            "maneuver/budget/veto/approval machinery in this phase - that's "
            "conjunction-specific scope (an avoidance burn makes no sense "
            "for 'your perigee is too low'); it gets a real deterministic "
            "classification and real Gemma narration, same as everything "
            "else, just no maneuver plan."
        ),
        action=_step_decay_risk,
    ),
    Step(
        phase="Phase 4",
        title="Local/cloud Gemma failover",
        explanation=(
            "Every plain-language explanation above came from Gemma - but "
            "WHICH Gemma (a model running locally via Ollama, or a hosted "
            "model in the cloud) is a runtime choice, and this step proves "
            "the system recovers automatically if its primary choice "
            "becomes unreachable. It deliberately breaks the local "
            "connection, makes one real call, and shows which backend "
            "actually ended up answering. In the real pipeline this is "
            "recorded in the audit log for every decision (see two steps "
            "from now) - so it's always visible which backend produced a "
            "given explanation, never hidden. SKIPPED when this machine is "
            "configured local-only (GEMMA_BACKEND=ollama) - a local-only "
            "demo should never make a real call to the cloud, even to "
            "prove a fallback path exists."
        ),
        action=_step_failover,
    ),
    Step(
        phase="Phase 5",
        title="Human review",
        explanation=(
            "Full autonomy isn't the goal here - especially for the "
            "budget-blocked case above, a person needs to be able to "
            "review and sign off on what the system did (or couldn't do). "
            "This step marks that exact logged decision as reviewed and "
            "confirms it's now permanently recorded that way in the audit "
            "trail, not just held in memory for this process."
        ),
        action=_step_human_review,
    ),
    Step(
        phase="Throughout",
        title="The audit trail itself",
        explanation=(
            "Every single decision this system makes - the raw sensor "
            "data, the deterministic classification, Gemma's explanation "
            "with an explicit note on whether it's trustworthy (real model "
            "output vs. a deterministic fallback), and now the human "
            "review status - is written to an append-only JSON-lines file. "
            "This step shows the actual persisted record for the most "
            "recent decision (the one just marked reviewed above), not a "
            "summary of it."
        ),
        action=_step_audit_trail,
    ),
    Step(
        phase="Throughout",
        title="Automated test suite",
        explanation=(
            "None of the above happened without a safety net. Every piece "
            "of logic demonstrated in this walkthrough - orbital math, "
            "severity thresholds, maneuver math, budget tracking, Gemma "
            "retry/fallback, human review - has automated tests that run "
            "without any network access, so this can be verified on any "
            "machine (including CI) without Ollama or a real API key."
        ),
        action=_step_test_suite,
    ),
    Step(
        phase="Wrap-up",
        title="Summary",
        explanation="Totals across everything demonstrated in this run.",
        action=_step_summary,
    ),
]


def run_steps(auto: bool) -> None:
    console = Console()
    ctx = DemoContext(console=console, auto=auto)

    console.print(Panel(
        "This is a guided walkthrough of a deep-space collision-avoidance "
        "pipeline: it pulls real satellite tracking data, computes real "
        "orbital mechanics, classifies risk with deterministic thresholds "
        "(not AI-decided - reliability matters here), and uses Gemma to "
        "explain findings and decisions in plain language, with maneuver "
        "execution, budget limits, and human review for the most severe "
        "cases. Each step below explains itself before running.",
        title="Deep Space Navigation - Track 2",
        border_style="cyan",
    ))

    for i, step in enumerate(STEPS, start=1):
        console.rule(f"[bold]Step {i}/{len(STEPS)} — {step.phase}: {step.title}[/bold]")
        console.print(step.explanation)
        console.print()
        if not auto and not Confirm.ask("Run this step?", default=True):
            console.print("[dim]Skipped.[/dim]")
            continue
        step.action(ctx)
        console.print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--auto", action="store_true",
        help="Run straight through with no pauses (for CI / quick smoke-testing).",
    )
    args = parser.parse_args()
    run_steps(auto=args.auto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
