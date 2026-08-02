#!/usr/bin/env python3
"""Live mission-ops dashboard: a visual risk board + human-approval inbox
over the exact same append-only audit log the CLI/demo write to - this is
a read/act layer, not a second copy of the pipeline's decision logic. All
non-UI logic (loading entries, computing table rows/metrics) lives in
src/dashboard_data.py, which has no Streamlit dependency and is directly
unit-tested; this file is UI wiring only.

Run:
    streamlit run scripts/dashboard.py

Sidebar buttons can generate NEW real activity for the log to show:
fetching live CelesTrak conjunctions (Phase 10's cross-group screening),
running the synthetic CRITICAL fixture (real orbital data rarely produces
a CRITICAL case on demand - see src/ingestion/synthetic_adapter.py), and
replaying a real documented historical conjunction (Phase 12 - see
src/ingestion/historical_adapter.py). All three go through the real
pipeline (src/pipeline.run_once) - nothing here bypasses or duplicates it.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import settings
from src.dashboard_data import compute_metrics, entries_to_rows, pending_approvals
from src.ingestion.celestrak_adapter import CelesTrakAdapter
from src.ingestion.historical_adapter import HistoricalReplayAdapter
from src.ingestion.synthetic_adapter import SyntheticCriticalAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.pipeline import run_once
from src.schemas import DecisionLogEntry


def _render_metrics(metrics: dict) -> None:
    row1 = st.columns(4)
    row1[0].metric("Total events", metrics["total"])
    row1[1].metric("CRITICAL", metrics["critical"])
    row1[2].metric("Autonomous executed", metrics["executed_autonomous"])
    row1[3].metric("Human-approved executed", metrics["executed_human_approved"])

    row2 = st.columns(4)
    row2[0].metric("Vetoed by Gemma", metrics["vetoed_by_gemma"])
    row2[1].metric("Rejected by human", metrics["rejected_by_human"])
    row2[2].metric("Awaiting approval", metrics["awaiting_human_approval"])
    row2[3].metric("Blocked by budget", metrics["budget_insufficient"])

    st.caption(f"Gemma-authored rationale: {metrics['gemma_rationale_pct']:.0f}% (rest is deterministic fallback)")


def _render_pending_approvals(logger: DecisionLogger, pending: list[DecisionLogEntry], operator: str) -> None:
    st.subheader(f"Pending human approval ({len(pending)})")
    if not pending:
        st.caption("Nothing awaiting approval right now.")
        return

    for entry in pending:
        plan = entry.decision.maneuver_plan
        raw = entry.telemetry.raw_data
        with st.container(border=True):
            st.markdown(f"**{entry.telemetry.event_id}** — {raw.get('object_a_name')} vs {raw.get('object_b_name')}")
            st.write(
                f"Min distance: {raw.get('min_distance_km'):.2f} km  |  "
                f"Proposed: {plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s delta-v  |  "
                f"Target clearance: {plan.target_clearance_km:.1f} km"
            )
            st.caption(entry.decision.rationale)
            col_approve, col_reject = st.columns(2)
            if col_approve.button("Approve", key=f"approve_{entry.telemetry.event_id}", type="primary"):
                logger.approve_maneuver(entry.telemetry.event_id, approved=True, approved_by=operator)
                st.rerun()
            if col_reject.button("Reject", key=f"reject_{entry.telemetry.event_id}"):
                logger.approve_maneuver(entry.telemetry.event_id, approved=False, approved_by=operator)
                st.rerun()


def _render_review_panel(logger: DecisionLogger, entries: list[DecisionLogEntry], operator: str) -> None:
    st.subheader("Inspect / mark reviewed")
    if not entries:
        st.caption("No logged decisions yet.")
        return

    event_ids = [e.telemetry.event_id for e in reversed(entries)]  # most recent first
    selected_id = st.selectbox("Event", event_ids)
    selected = next(e for e in entries if e.telemetry.event_id == selected_id)

    st.json(selected.model_dump(mode="json"))
    if selected.human_reviewed:
        st.caption(f"Already reviewed by {selected.reviewed_by} at {selected.human_reviewed_at}")
    elif st.button("Mark reviewed", key=f"review_{selected_id}"):
        logger.mark_reviewed(selected_id, reviewed_by=operator)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Deep Space Navigation - Mission Ops", layout="wide")
    st.title("Deep Space Navigation — Mission Ops Dashboard")
    st.caption(
        "Live risk board and human-approval inbox over the real append-only audit log "
        "(logs/decisions-*.jsonl) - not a mock, not a second copy of the pipeline."
    )

    logger = DecisionLogger(settings=settings)

    with st.sidebar:
        st.header("Controls")
        operator = st.text_input("Operator name", value="dashboard-operator")
        st.write(f"Configured Gemma backend: **{settings.gemma_backend}**")
        st.caption(
            "local (ollama) -> autonomous execution with Gemma's own GO/NO-GO veto review; "
            "cloud (api) -> maneuvers held pending human approval below."
        )

        st.divider()
        st.subheader("Generate live activity")
        fetch_limit = st.number_input("CelesTrak results to fetch", min_value=1, max_value=50, value=5)
        if st.button("Fetch live CelesTrak conjunctions"):
            with st.spinner("Screening real CelesTrak data..."):
                adapter = CelesTrakAdapter()
                run_once(adapter=adapter, logger=logger, limit=fetch_limit)
                stats = adapter.last_scan_stats
            if stats:
                st.success(
                    f"Screened {stats['total_pairs_screened']} pairs across "
                    f"{stats['total_objects']} objects ({stats['groups']})."
                )
            st.rerun()

        if st.button("Run synthetic CRITICAL scenario"):
            with st.spinner("Running synthetic CRITICAL conjunctions..."):
                run_id = uuid.uuid4().hex[:8]
                adapter = SyntheticCriticalAdapter(run_id=run_id, id_prefix="conj-dashboard")
                tracker = DeltaVBudgetTracker(starting_budget_m_s=settings.delta_v_budget_m_s)
                run_once(adapter=adapter, logger=logger, budget_tracker=tracker, limit=4)
            st.rerun()

        if st.button("Replay historical event (Iridium 33 / Cosmos 2251, 2009)"):
            with st.spinner("Replaying the real 2009 collision record..."):
                run_id = uuid.uuid4().hex[:8]
                adapter = HistoricalReplayAdapter(run_id=run_id)
                run_once(adapter=adapter, logger=logger, limit=1)
            st.rerun()

        st.divider()
        if st.button("Refresh"):
            st.rerun()

    entries = logger.load_all_entries()

    _render_metrics(compute_metrics(entries))
    st.divider()
    _render_pending_approvals(logger, pending_approvals(entries), operator)
    st.divider()

    st.subheader("All decisions")
    if entries:
        st.dataframe(entries_to_rows(entries), width="stretch", hide_index=True)
    else:
        st.caption("No logged decisions yet - use the sidebar to generate some.")
    st.divider()

    _render_review_panel(logger, entries, operator)


main()
