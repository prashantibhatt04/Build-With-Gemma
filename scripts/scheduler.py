#!/usr/bin/env python3
"""Continuous background screening loop - ROADMAP_TO_PRODUCT.md Phase 4.

Every other way to run this project's pipeline is on-demand: open the
dashboard and click a button, or run scripts/run_demo.py once. A real
operator needs continuous screening instead - this runs the same real
hazard screens the dashboard/CLI already expose (real CelesTrak
conjunctions, real decay risk) on a fixed interval, indefinitely, using
whichever DecisionLogger backend is configured (JSONL files by default,
or Postgres - src/postgres_logging.py - when DATABASE_URL is set).
Postgres matters specifically here: JSONL's find()/update() rewrite a
whole file in place, which is fine for a demo's occasional clicks but not
safe under the concurrent writes a real continuous process running
alongside a dashboard/CLI operator would produce.

Usage:
    python scripts/scheduler.py                          # real cadence: ~every 8h (matches Space-Track's CDM refresh), runs forever
    python scripts/scheduler.py --interval-seconds 60     # faster cadence, for demo/testing
    python scripts/scheduler.py --max-iterations 3        # stop after N ticks, for testing - omit to run forever

One shared GemmaClient and DeltaVBudgetTracker persist across every tick,
not reconstructed per tick - the same lesson this project's own QA pass
already learned the hard way for scripts/dashboard.py and
scripts/run_demo.py (see PHASE_PROGRESS.md's QA-pass fix #1): a fresh
client per call silently defeats GemmaClient's circuit breaker, and a
fresh budget tracker per tick would silently reset the delta-v budget
every cycle regardless of what already executed.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerting import send_health_alert
from src.config import settings
from src.gemma_client import GemmaClient
from src.ingestion.base_adapter import DataSourceAdapter
from src.ingestion.celestrak_adapter import CelesTrakAdapter
from src.ingestion.decay_adapter import DecayRiskAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import run_once
from src.schemas import DecisionLogEntry

# Fires a real "system health" alert (src/alerting.py) once the scheduler
# has failed this many ticks in a row - matching GemmaClient's own
# circuit-breaker threshold (src/gemma_client.py) for consistency, not a
# coincidence: both exist to notice "this has been broken for a while
# now", not overreact to one transient blip. Fires ONCE per outage (not
# on every failure past the threshold) - a success resets the count, so a
# genuinely new outage later fires again.
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 3


def default_adapters() -> list[DataSourceAdapter]:
    """The real hazard screens run every tick: real cross-group
    conjunctions and real decay risk, the same two data sources
    scripts/dashboard.py's sidebar buttons already expose. A fresh
    instance per tick is correct here (unlike GemmaClient/the budget
    tracker) - each adapter's own run_id-per-instance scheme (see
    celestrak_adapter.py) is what keeps event_ids unique across separate
    ticks in the first place."""
    return [CelesTrakAdapter(), DecayRiskAdapter()]

# Space-Track's real CDMs refresh roughly every 8 hours (see
# ROADMAP_TO_PRODUCT.md Phase 4) - a sensible real-world default cadence,
# overridable via --interval-seconds for demo/testing.
DEFAULT_INTERVAL_SECONDS = 8 * 60 * 60


def run_tick(
    client: GemmaClient,
    logger: DecisionLogger,
    budget_tracker: DeltaVBudgetTracker,
    tick_number: int,
    adapters: list[DataSourceAdapter],
    limit_per_adapter: int = 10,
) -> list[DecisionLogEntry]:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] tick {tick_number}: screening {len(adapters)} real data source(s)...")

    entries: list[DecisionLogEntry] = []
    for adapter in adapters:
        entries += run_once(
            adapter=adapter, client=client, logger=logger,
            budget_tracker=budget_tracker, limit=limit_per_adapter,
        )

    if entries:
        counts = Counter(e.finding.severity.value for e in entries)
        print(f"  logged {len(entries)} entries: {dict(counts)}")
    else:
        print("  logged 0 entries")
    return entries


def update_consecutive_failures(succeeded: bool, consecutive_failures: int) -> tuple[int, bool]:
    """Returns (new_consecutive_failures, should_alert_now). A success
    resets the count to 0; a failure increments it, and should_alert_now
    is True only on the EXACT tick that crosses
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD - not on every failure after,
    which would spam the same outage repeatedly."""
    if succeeded:
        return 0, False
    consecutive_failures += 1
    return consecutive_failures, consecutive_failures == CONSECUTIVE_FAILURE_ALERT_THRESHOLD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Stop after N ticks instead of running forever (useful for testing).",
    )
    args = parser.parse_args()

    client = GemmaClient()
    logger = DecisionLogger()
    budget_tracker = DeltaVBudgetTracker(settings.delta_v_budget_m_s)

    backend_label = "Postgres" if settings.database_url else "JSONL files"
    print(
        f"Scheduler starting - storage: {backend_label}, "
        f"interval: {args.interval_seconds:.0f}s"
        + (f", max_iterations={args.max_iterations}" if args.max_iterations is not None else ", running until Ctrl-C")
    )

    tick_number = 0
    consecutive_failures = 0
    try:
        while args.max_iterations is None or tick_number < args.max_iterations:
            tick_number += 1
            try:
                run_tick(client, logger, budget_tracker, tick_number, adapters=default_adapters())
                consecutive_failures, should_alert = update_consecutive_failures(True, consecutive_failures)
            except Exception as exc:
                # A single bad tick (e.g. CelesTrak transiently
                # unreachable) must not kill a process meant to run
                # indefinitely - logged and skipped, the next tick tries
                # again on schedule. Several IN A ROW is different - an
                # operator relying on this to run unattended needs to
                # find out the scheduler itself might be down, not just
                # silently get zero new findings.
                print(f"  tick {tick_number} failed: {exc}")
                consecutive_failures, should_alert = update_consecutive_failures(False, consecutive_failures)
                if should_alert:
                    send_health_alert(
                        f"scheduler has failed {consecutive_failures} consecutive ticks - "
                        f"latest error: {exc}",
                        settings,
                    )

            if args.max_iterations is not None and tick_number >= args.max_iterations:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nScheduler stopped (Ctrl-C).")


if __name__ == "__main__":
    main()
