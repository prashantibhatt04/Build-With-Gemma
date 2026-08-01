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
from datetime import datetime, timezone
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
from src.ingestion.base_adapter import DataSourceAdapter
from src.ingestion.celestrak_adapter import CelesTrakAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import run_once
from src.preflight import run_all_checks
from src.schemas import DecisionLogEntry, TelemetryEvent


@dataclass
class DemoContext:
    """Shared state later steps can read from earlier ones."""

    console: Console
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


class SyntheticCriticalAdapter(DataSourceAdapter):
    """4 synthetic CRITICAL-range conjunctions sharing one budget tracker.
    Real data rarely produces CRITICAL (<5km) on demand, so this demos that
    path - and budget depletion - deterministically. Clearly labeled as
    synthetic via source/event_id, matching DEMO.md's Stage 5.

    event_id includes run_id so repeat demo runs don't collide on the same
    id - mark_reviewed/find_entry (DecisionLogger) match on the FIRST
    logged entry with a given event_id, so a reused id from an earlier run
    would silently update that older entry instead of this run's.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        events = []
        for i in range(limit):
            raw = {
                "object_a_id": f"9900{i}", "object_a_name": f"SYNTH-A-{i}",
                "object_b_id": f"9901{i}", "object_b_name": f"SYNTH-B-{i}",
                "min_distance_km": 3.0,
                "time_of_closest_approach": "2026-08-01T20:00:00+00:00",
                "relative_velocity_km_s": 6.0,
            }
            events.append(TelemetryEvent(
                event_id=f"conj-run-demo-{self.run_id}-{i}",
                timestamp=datetime.now(timezone.utc),
                source="synthetic-critical-fixture",
                raw_data=raw,
            ))
        return events


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
    entries = run_once(adapter=CelesTrakAdapter(sample_size=15), limit=2)
    render_entries(entries, console=ctx.console)
    ctx.all_entries.extend(entries)


def _step_critical_maneuver_and_budget(ctx: DemoContext) -> None:
    tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)
    entries = run_once(adapter=SyntheticCriticalAdapter(run_id=ctx.run_id), budget_tracker=tracker, limit=4)
    render_entries(entries, console=ctx.console)
    ctx.all_entries.extend(entries)
    # The last one is always the budget-insufficient case given the fixed
    # budget/delta-v math above - a good candidate for the human-review step.
    ctx.reviewable_event_id = entries[-1].telemetry.event_id


def _step_failover(ctx: DemoContext) -> None:
    broken = Settings(
        gemma_backend="ollama", gemma_model=settings.gemma_model,
        ollama_host="http://localhost:1",  # intentionally unreachable
        gemma_api_key=settings.gemma_api_key, gemma_model_api=settings.gemma_model_api,
        log_dir=settings.log_dir, delta_v_budget_m_s=settings.delta_v_budget_m_s,
    )
    client = GemmaClient(settings=broken)
    try:
        text = client.generate(prompt="Reply with the single word: ok", timeout=15)
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
    maneuvers_executed = sum(1 for e in entries if e.decision.maneuver_plan and not e.decision.budget_insufficient)
    maneuvers_blocked = sum(1 for e in entries if e.decision.budget_insufficient)
    ctx.console.print(
        f"Total events: {len(entries)}  |  Gemma rationale: {gemma_count}  |  "
        f"Fallback rationale: {fallback_count}  |  Maneuvers executed: {maneuvers_executed}  |  "
        f"Maneuvers blocked by budget: {maneuvers_blocked}"
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
        phase="Phase 2",
        title="Real orbital data: CelesTrak + Skyfield/SGP4",
        explanation=(
            "This project tracks satellite/debris collision risk. This step "
            "fetches REAL satellite tracking data (TLEs - Two-Line Elements) "
            "live from CelesTrak, then runs REAL orbital mechanics (a "
            "two-pass coarse/fine search using Skyfield's SGP4 propagator) "
            "to find the closest predicted approach between pairs of "
            "objects over the next 48 hours. Nothing here is simulated - "
            "these are real objects currently in orbit."
        ),
        action=_step_live_orbital_data,
    ),
    Step(
        phase="Phases 0-3, 5",
        title="CRITICAL conjunctions: autonomous maneuver, verification, and delta-v budget",
        explanation=(
            "Severity (NOMINAL/WATCH/WARNING/CRITICAL) and the resulting "
            "action are decided by a plain distance threshold, NOT by the "
            "AI model - this needs to be reliable, so Gemma is only used "
            "afterward to explain a decision that's already been made "
            "deterministically. Real orbital data rarely produces a "
            "CRITICAL case (<5km predicted separation) on demand, so this "
            "step uses 4 synthetic CRITICAL-range conjunctions (clearly "
            "labeled as synthetic in their source field) to demonstrate "
            "that path on purpose. For each one: a simplified avoidance "
            "maneuver is calculated, independently re-verified to confirm "
            "it would actually clear the danger threshold, and only then "
            "reported as a completed action - narrated by Gemma in past "
            "tense, not as a suggestion. All 4 events share ONE limited "
            "delta-v budget (a stand-in for real spacecraft fuel limits), "
            "so watch it run out: the last event can't afford its "
            "maneuver, and the system explicitly says so and escalates for "
            "human review instead of silently pretending to have acted."
        ),
        action=_step_critical_maneuver_and_budget,
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
            "given explanation, never hidden."
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
    ctx = DemoContext(console=console)

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
