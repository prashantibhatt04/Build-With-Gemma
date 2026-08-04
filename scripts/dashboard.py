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
a CRITICAL case on demand - see src/ingestion/synthetic_adapter.py),
replaying a real documented historical conjunction (Phase 12 - see
src/ingestion/historical_adapter.py), screening real objects for
orbital decay/re-entry risk (Phase 14 - a second real hazard type,
see src/ingestion/decay_adapter.py), and running a synthetic attitude/
pointing-loss scenario (Phase 18 - a third hazard type, necessarily
synthetic-only since no real public data source for spacecraft attitude
exists, unlike TLE-derived position - see
src/ingestion/attitude_adapter.py). All five go through the real
pipeline (src/pipeline.run_once) - nothing here bypasses or duplicates it.

The inspect panel can also render a real orbit plot (3D trajectories +
distance-over-time) for any celestrak-sourced event, by re-fetching both
objects' current TLEs and re-propagating with the same physics
src/orbital.py already uses - see src/orbit_plot_data.py.

"Ask about the mission log" is real retrieval-augmented Q&A over this
exact log - real local-Ollama embeddings, real cosine-similarity
ranking, Gemma answering from ONLY the retrieved entries, never outside
knowledge - see src/rag.py.

"Trends" aggregates the real accumulated log itself (severity mix per
day, recurring real objects, Gemma-vs-fallback rationale mix over time)
- no other view in this dashboard shows more than one event or one
instant at a time - see src/trends.py.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.alerting import send_test_alert
from src.auth import authenticate
from src.config import settings
from src.dashboard_data import (
    compute_metrics,
    entries_to_rows,
    filter_entries,
    needs_attention,
    pending_approvals,
    tca_urgency_label,
)
from src.ingestion.attitude_adapter import SyntheticAttitudeAdapter
from src.ingestion.celestrak_adapter import CelesTrakAdapter
from src.ingestion.decay_adapter import DecayRiskAdapter
from src.ingestion.historical_adapter import HistoricalReplayAdapter
from src.ingestion.spacetrack_adapter import EnrichedSpaceTrackAdapter
from src.ingestion.spacetrack_client import SpaceTrackAuthError, SpaceTrackClient
from src.ingestion.synthetic_adapter import SyntheticCriticalAdapter
from src.gemma_client import GemmaClient
from src.live_positions import build_live_globe_figure, fetch_live_positions
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from src.orbit_plot_data import build_3d_trajectory_figure, build_distance_chart, fetch_trajectory_data
from src.pipeline import run_once
from src.rag import answer_question
from src.schemas import DecisionLogEntry
from src.trends import build_rationale_source_trend_figure, build_severity_trend_figure, is_real_live_source, recurring_objects


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


def _render_all_decisions_table(entries: list[DecisionLogEntry]) -> None:
    """The real risk board itself - under continuous scheduled operation
    (scripts/scheduler.py) this grows unbounded forever, and until now had
    no way to narrow it, unlike scripts/api.py's GET /decisions (which
    already supports severity/source filters). Filters are real narrowing
    via filter_entries (src/dashboard_data.py), not just a display-side
    highlight - the caption below always reports the real filtered vs.
    total count so it's never ambiguous whether a filter is silently
    hiding rows."""
    st.subheader("All decisions")
    if not entries:
        st.caption("No logged decisions yet - use the sidebar to generate some.")
        return

    all_severities = sorted({e.finding.severity.value for e in entries})
    all_sources = sorted({e.telemetry.source for e in entries})
    filter_cols = st.columns(2)
    # Real bug this fixes: without an explicit, STABLE key, Streamlit
    # partly derives a widget's identity from its `options` argument -
    # which changes every time the underlying log grows (a new severity/
    # source value shows up, or entries_to_rows' set ordering shifts).
    # An operator's active filter selection was silently reverting to
    # empty the moment background activity (e.g. a scheduler tick, or
    # clicking a sidebar fetch button) changed what's in the log - no
    # crash, just quietly wrong UI state mid-review. A fixed key isn't
    # tied to `options` at all, so the real selected value survives a
    # rerun even as the available options list changes underneath it.
    selected_severities = filter_cols[0].multiselect(
        "Filter by severity", all_severities, key="all_decisions_severity_filter",
    )
    selected_sources = filter_cols[1].multiselect(
        "Filter by source", all_sources, key="all_decisions_source_filter",
    )

    filtered = filter_entries(entries, selected_severities, selected_sources)
    st.caption(f"Showing {len(filtered)} of {len(entries)} logged decisions.")
    if filtered:
        st.dataframe(entries_to_rows(filtered), width="stretch", hide_index=True)
    else:
        st.caption("No decisions match the selected filters.")


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
            # Real, decision-relevant urgency - a human approving/
            # rejecting needs to know not just how close this gets but
            # whether the event it's FOR has already happened, since
            # approving a maneuver after its own TCA has no effect.
            tca_raw = raw.get("time_of_closest_approach")
            if tca_raw:
                urgency = tca_urgency_label(tca_raw)
                if "already passed" in urgency:
                    st.warning(f"⚠️ {urgency} - approving this maneuver no longer has any effect.")
                else:
                    st.caption(urgency)
            st.caption(entry.decision.rationale)
            col_approve, col_reject = st.columns(2)
            if col_approve.button("Approve", key=f"approve_{entry.telemetry.event_id}", type="primary"):
                try:
                    logger.approve_maneuver(entry.telemetry.event_id, approved=True, approved_by=operator)
                except ValueError as exc:
                    st.error(f"Couldn't approve this maneuver: {exc}")
                else:
                    st.rerun()
            if col_reject.button("Reject", key=f"reject_{entry.telemetry.event_id}"):
                try:
                    logger.approve_maneuver(entry.telemetry.event_id, approved=False, approved_by=operator)
                except ValueError as exc:
                    st.error(f"Couldn't reject this maneuver: {exc}")
                else:
                    st.rerun()


def _render_needs_attention(logger: DecisionLogger, attention: list[DecisionLogEntry], operator: str) -> None:
    """A real CRITICAL decay/attitude finding has no maneuver/approval
    workflow of its own (see needs_attention's docstring,
    src/dashboard_data.py) - there's no avoidance burn for "your perigee
    is too low" or "you're tumbling" - so without a dedicated section
    like this one, it was previously indistinguishable from NOMINAL/WATCH
    noise in the "All decisions" table. "Acknowledge" reuses the same
    mark_reviewed this dashboard's Inspect panel already uses - there's
    no separate approve/reject decision to make here, just a real human
    confirming they've seen it."""
    st.subheader(f"Needs attention ({len(attention)})")
    if not attention:
        st.caption("No CRITICAL decay/attitude findings awaiting acknowledgment.")
        return

    for entry in attention:
        raw = entry.telemetry.raw_data
        subject = raw.get("object_name", entry.telemetry.event_id)
        with st.container(border=True):
            st.markdown(f"**{entry.telemetry.event_id}** — {subject}")
            if "perigee_altitude_km" in raw:
                st.write(f"Perigee altitude: {raw['perigee_altitude_km']:.1f} km")
            elif "pointing_error_deg" in raw:
                st.write(f"Pointing error: {raw['pointing_error_deg']:.1f}°")
            st.caption(entry.decision.rationale)
            if st.button("Acknowledge", key=f"acknowledge_{entry.telemetry.event_id}"):
                try:
                    logger.mark_reviewed(entry.telemetry.event_id, reviewed_by=operator)
                except ValueError as exc:
                    st.error(f"Couldn't acknowledge this finding: {exc}")
                else:
                    st.rerun()


def _render_orbit_plot(entry: DecisionLogEntry) -> None:
    raw = entry.telemetry.raw_data
    # Reuses trends.is_real_live_source rather than a hardcoded
    # source == "celestrak" check - a QA pass found the original check
    # wrongly excluded real Space-Track events too, which carry the
    # exact same real NORAD object_a_id/object_b_id (SpaceTrackAdapter
    # is a thin CelesTrakAdapter subclass - see spacetrack_adapter.py)
    # and are just as propagatable, often with a higher-confidence real
    # Pc attached besides.
    if not is_real_live_source(entry) or "object_a_id" not in raw:
        st.caption(
            f"Orbit plot unavailable - this event's source is {entry.telemetry.source!r}, "
            "not a real NORAD-catalogued object pair. Only real live conjunction scans "
            "(CelesTrak or Space-Track) have real TLEs to propagate (synthetic fixtures "
            "aren't real orbits; historical replays document a real past event but "
            "CelesTrak's/Space-Track's public APIs only ever serve CURRENT TLEs, not "
            "archival ones - see src/ingestion/historical_adapter.py)."
        )
        return

    if st.button("Generate orbit plot", key=f"orbit_{entry.telemetry.event_id}"):
        with st.spinner("Fetching current TLEs and propagating real trajectories..."):
            try:
                data = fetch_trajectory_data(
                    raw["object_a_id"], raw.get("object_a_name", ""),
                    raw["object_b_id"], raw.get("object_b_name", ""),
                )
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't fetch/propagate a live orbit plot: {exc}")
                return
        st.caption(
            "Recomputed live from each object's CURRENT TLE, starting now - orbital "
            "elements update over time, so this may differ from the min_distance_km "
            "logged when this decision was originally made."
        )
        st.plotly_chart(build_3d_trajectory_figure(data), width="stretch")
        st.plotly_chart(build_distance_chart(data), width="stretch")


def _render_live_tracking() -> None:
    st.subheader("Live tracking: real crewed stations, right now")
    st.caption(
        "Real current positions (not a triage result) for CelesTrak's "
        "\"stations\" group - ISS, Tiangong, and their currently-docked "
        "visiting vehicles - the same real, named assets the conjunction "
        "screening above treats as the payload actually worth protecting."
    )
    if st.button("Show live positions"):
        with st.spinner("Fetching real current TLEs and propagating..."):
            try:
                positions = fetch_live_positions()
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't fetch live positions: {exc}")
                return
        st.plotly_chart(build_live_globe_figure(positions), width="stretch")
        st.caption(f"{len(positions)} real objects, positions computed for right now.")


def _render_trends(entries: list[DecisionLogEntry]) -> None:
    st.subheader("Trends")
    st.caption(
        "Every other view above shows one event or the current instant. This "
        "aggregates the real accumulated log itself: severity mix per real day, "
        "which real objects keep showing up across separate scans, and how much "
        "narration is genuinely coming from Gemma vs. the deterministic fallback."
    )
    if not entries:
        st.caption("No logged decisions yet.")
        return

    real_entries = [e for e in entries if is_real_live_source(e)]
    st.caption(
        f"{len(real_entries)} of {len(entries)} total logged entries are real, live "
        "CelesTrak scans - the two charts below exclude synthetic fixtures and the "
        "historical replay, which are repeatable demo data that would otherwise "
        "dominate a view about real patterns over time just because a demo was run "
        "more than once."
    )
    if not real_entries:
        st.caption(
            "No real live-scan entries yet - use \"Fetch live CelesTrak conjunctions\" "
            "or \"Screen for orbital decay risk\" in the sidebar to generate some."
        )
    else:
        st.plotly_chart(build_severity_trend_figure(real_entries), width="stretch")
        st.plotly_chart(build_rationale_source_trend_figure(real_entries), width="stretch")

    st.markdown("**Recurring objects** (most logged appearances first, real and synthetic alike)")
    recurring = recurring_objects(entries, top_n=10)
    if recurring:
        st.dataframe(recurring, width="stretch", hide_index=True)
    else:
        st.caption("No entries carry a real object identity yet.")


def _render_mission_log_search(entries: list[DecisionLogEntry], client: GemmaClient) -> None:
    st.subheader("Ask about the mission log")
    st.caption(
        "Real retrieval, not fine-tuning: embeds every logged entry via a local "
        "Ollama embedding model, ranks by real cosine similarity against your "
        "question, and asks Gemma to answer using ONLY the retrieved entries "
        "below - never outside knowledge. Requires a reachable local Ollama for "
        "embeddings regardless of which backend narrates elsewhere."
    )
    query = st.text_input("Question", placeholder="e.g. which CRITICAL events were vetoed and why?")
    if st.button("Search the mission log") and query:
        with st.spinner("Embedding, ranking, and asking Gemma..."):
            try:
                result = answer_question(query, entries, client)
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(
                    f"Couldn't search the mission log: {exc}\n\n"
                    "Mission-log search needs a reachable local Ollama for embeddings "
                    "(GEMMA_EMBED_MODEL) even if your primary backend is the hosted API."
                )
                return
        st.write(result["answer"])
        if result["retrieved_event_ids"]:
            st.caption(f"Grounded in real logged entries: {', '.join(result['retrieved_event_ids'])}")


def _render_review_panel(logger: DecisionLogger, entries: list[DecisionLogEntry], operator: str) -> None:
    st.subheader("Inspect / mark reviewed")
    if not entries:
        st.caption("No logged decisions yet.")
        return

    event_ids = [e.telemetry.event_id for e in reversed(entries)]  # most recent first
    # Labeled with severity+subject (the same fields the "All decisions"
    # table shows), not just a bare event_id - reusing entries_to_rows so
    # this can never disagree with the table about what a row means.
    rows_by_id = {row["event_id"]: row for row in entries_to_rows(entries)}

    def _event_label(event_id: str) -> str:
        row = rows_by_id[event_id]
        return f"[{row['severity'].upper()}] {row['subject']} — {event_id}"

    # Real bug this fixes: without an explicit, STABLE key, Streamlit
    # partly derives this widget's identity from `event_ids` - which
    # grows every time a new entry is logged (e.g. a concurrent
    # scheduler tick, or clicking a sidebar fetch button while reviewing
    # an older event). An operator mid-review of a specific, deliberately
    # -selected older event was silently reverted back to the newest
    # entry the moment new data arrived - no crash, just quietly wrong
    # UI state. A fixed key isn't tied to the options list at all, so
    # Streamlit finds the real previously-selected value in the new,
    # longer list and keeps it selected instead of resetting.
    selected_id = st.selectbox(
        "Event", event_ids, format_func=_event_label, key="review_panel_event_select",
    )
    selected = next(e for e in entries if e.telemetry.event_id == selected_id)

    st.json(selected.model_dump(mode="json"))
    if selected.human_reviewed:
        st.caption(f"Already reviewed by {selected.reviewed_by} at {selected.human_reviewed_at}")
    elif st.button("Mark reviewed", key=f"review_{selected_id}"):
        try:
            logger.mark_reviewed(selected_id, reviewed_by=operator)
        except ValueError as exc:
            st.error(f"Couldn't mark this decision reviewed: {exc}")
        else:
            st.rerun()

    st.divider()
    st.subheader("Orbit plot")
    _render_orbit_plot(selected)


def _get_shared_client() -> GemmaClient:
    """One GemmaClient per browser session, reused across every button
    click and rerun - a QA pass found that the old behavior (each click
    implicitly getting its own fresh client via run_once's client=None
    default) meant GemmaClient's circuit breaker could never accumulate
    consecutive failures across separate clicks, only within a single
    click's own run_once() event batch - defeating the point of a
    breaker meant to protect against an extended outage spanning
    multiple real interactions, not just multiple events in one batch."""
    if "gemma_client" not in st.session_state:
        st.session_state["gemma_client"] = GemmaClient(settings=settings)
    return st.session_state["gemma_client"]


def _watching_own_assets() -> bool:
    return bool(settings.watched_norad_ids)


def _celestrak_screening_kwargs() -> dict:
    """When a real operator has configured their own asset(s) via
    WATCHED_NORAD_IDS, every conjunction-screening button below screens
    those specific real objects instead of CelesTrak's own "stations"
    placeholder - a customer's own satellite is the actual "asset" side
    of the question this whole product answers, not a demo stand-in for
    it. Falls back to CelesTrakAdapter's own defaults (stations vs.
    cosmos-2251-debris) completely unchanged when nothing's configured,
    so the existing zero-setup demo experience is untouched."""
    if not _watching_own_assets():
        return {}
    return {"groups": ("cosmos-2251-debris",), "watched_norad_ids": settings.watched_norad_ids}


def _spacetrack_screening_kwargs() -> dict:
    """Same real substitution as _celestrak_screening_kwargs above, for
    the Space-Track path - a real bulk NORAD_CAT_ID query (see
    SpaceTrackAdapter/fetch_spacetrack_by_catalog_ids) instead of one
    request per object."""
    if not _watching_own_assets():
        return {}
    return {
        "group_name_patterns": {"cosmos-2251-debris": "COSMOS 2251 DEB"},
        "watched_norad_ids": settings.watched_norad_ids,
    }


def _render_monitoring_status() -> None:
    if _watching_own_assets():
        st.sidebar.success(
            f"🛰️ Monitoring your own asset(s): NORAD ID(s) {', '.join(settings.watched_norad_ids)}"
        )
    else:
        st.sidebar.info(
            "🛰️ Monitoring CelesTrak's demo 'stations' group (ISS, Tiangong, ...), "
            "not a specific asset. Set WATCHED_NORAD_IDS in .env to your own "
            "satellite's NORAD catalog ID to monitor it instead."
        )


_DEFAULT_SEVERITY_THRESHOLDS = {
    "conjunction_critical_km": 5.0,
    "conjunction_warning_km": 25.0,
    "conjunction_watch_km": 100.0,
    "decay_critical_perigee_km": 200.0,
    "decay_warning_perigee_km": 300.0,
    "decay_watch_perigee_km": 500.0,
    "attitude_critical_deg": 45.0,
    "attitude_warning_deg": 15.0,
    "attitude_watch_deg": 5.0,
}


def _using_custom_severity_thresholds() -> bool:
    return any(getattr(settings, field) != default for field, default in _DEFAULT_SEVERITY_THRESHOLDS.items())


def _render_severity_threshold_status() -> None:
    """Surfaces the real, currently-active CRITICAL/WARNING/WATCH cutoffs
    (see Settings.conjunction_critical_km and friends, src/config.py) so
    an operator who just configured e.g. CONJUNCTION_CRITICAL_KM in .env
    can actually confirm it took effect, without reading source code -
    the same discoverability gap already closed for WATCHED_NORAD_IDS
    (see _render_monitoring_status above and ROADMAP_TO_PRODUCT.md)."""
    label = "⚙️ Hazard severity thresholds"
    label += " (customized)" if _using_custom_severity_thresholds() else " (defaults)"
    with st.sidebar.expander(label):
        st.caption("Real, deterministic cutoffs - not Gemma-derived. Set via .env, see .env.example.")
        st.write(
            f"**Conjunction distance** - CRITICAL < {settings.conjunction_critical_km}km, "
            f"WARNING < {settings.conjunction_warning_km}km, WATCH < {settings.conjunction_watch_km}km"
        )
        st.write(
            f"**Decay perigee altitude** - CRITICAL < {settings.decay_critical_perigee_km}km, "
            f"WARNING < {settings.decay_warning_perigee_km}km, WATCH < {settings.decay_watch_perigee_km}km"
        )
        st.write(
            f"**Attitude pointing error** - CRITICAL ≥ {settings.attitude_critical_deg}°, "
            f"WARNING ≥ {settings.attitude_warning_deg}°, WATCH ≥ {settings.attitude_watch_deg}°"
        )


def _render_alert_status() -> None:
    """Real gap this closes: an operator configuring ALERT_WEBHOOK_URL
    for the first time had no way to confirm it's wired correctly (right
    URL, right channel, no firewall issue) short of waiting for a real
    CRITICAL hazard - the worst possible moment to discover a broken
    pipe. "Send test alert" reuses the exact same real webhook send path
    (src/alerting.py) a real CRITICAL page would use, just with a
    distinctly-labeled test message that can never be mistaken for a
    real page."""
    if not settings.alert_webhook_url:
        st.sidebar.info(
            "🔕 CRITICAL-event webhook alerting is not configured. Set ALERT_WEBHOOK_URL "
            "in .env to get real-time pages for CRITICAL findings."
        )
        return

    st.sidebar.success("🔔 CRITICAL-event webhook alerting is configured.")
    if st.sidebar.button("Send test alert"):
        with st.spinner("Sending a real test webhook POST..."):
            sent = send_test_alert(settings)
        if sent:
            st.sidebar.success("Test alert sent - check your configured channel.")
        else:
            st.sidebar.error("Test alert failed to send - check ALERT_WEBHOOK_URL and your network.")


def _get_shared_budget_tracker() -> DeltaVBudgetTracker:
    """One DeltaVBudgetTracker per browser session, reused across every
    button click and rerun - the same real bug class this project
    already fixed once for GemmaClient's circuit breaker above
    (_get_shared_client). run_once's own budget_tracker=None default
    constructs a brand-new tracker with the FULL starting budget on
    every call - without this, a real multi-scan operator session could
    burn most of the budget on one CRITICAL event, then silently get it
    back to full on the very next click, defeating the "BUDGET
    INSUFFICIENT - ESCALATE FOR REVIEW" safety path this tracker exists
    to trigger."""
    if "budget_tracker" not in st.session_state:
        st.session_state["budget_tracker"] = DeltaVBudgetTracker(starting_budget_m_s=settings.delta_v_budget_m_s)
    return st.session_state["budget_tracker"]


def _require_operator_identity() -> str:
    """Real authentication gate (ROADMAP_TO_PRODUCT.md Phase 5) - see
    src/auth.py's module docstring for why this exists. When
    OPERATOR_TOKENS is configured, nothing else in this app renders until
    a valid token is entered (st.stop() below), and the returned identity
    is the one the token actually maps to - not free text a visitor
    could type. When unconfigured, behavior is exactly what it was before
    this phase (a free-text field), but with a visible warning instead of
    a silent gap - an operator glancing at the page should be able to
    tell whether "approved_by" actually means anything here."""
    if not settings.operator_tokens:
        st.warning(
            "⚠️ Unauthenticated dashboard - OPERATOR_TOKENS is not configured, so "
            "the operator name below is free text anyone with this page open can "
            "set to anything. Set OPERATOR_TOKENS in .env before using this for "
            "real approve/reject/review actions.",
            icon="⚠️",
        )
        return st.sidebar.text_input("Operator name", value="dashboard-operator")

    if "authenticated_operator" in st.session_state:
        operator = st.session_state["authenticated_operator"]
        st.sidebar.success(f"Signed in as **{operator}**")
        if st.sidebar.button("Sign out"):
            del st.session_state["authenticated_operator"]
            st.rerun()
        return operator

    st.info("Enter your operator token to continue.")
    token = st.text_input("Operator token", type="password")
    if st.button("Sign in"):
        operator = authenticate(token, settings.operator_tokens)
        if operator is None:
            st.error("Invalid token.")
        else:
            st.session_state["authenticated_operator"] = operator
            st.rerun()
    st.stop()


def main() -> None:
    st.set_page_config(page_title="Deep Space Navigation - Mission Ops", layout="wide")
    st.title("Deep Space Navigation — Mission Ops Dashboard")
    st.caption(
        "Live risk board and human-approval inbox over the real append-only audit log "
        "(logs/decisions-*.jsonl) - not a mock, not a second copy of the pipeline."
    )

    operator = _require_operator_identity()

    client = _get_shared_client()
    budget_tracker = _get_shared_budget_tracker()
    logger = DecisionLogger(settings=settings)

    _render_live_tracking()
    st.divider()

    with st.sidebar:
        st.header("Controls")
        st.write(f"Configured Gemma backend: **{settings.gemma_backend}**")
        st.caption(
            "local (ollama) -> autonomous execution with Gemma's own GO/NO-GO veto review; "
            "cloud (api) -> maneuvers held pending human approval below."
        )

        st.divider()
        _render_monitoring_status()
        _render_severity_threshold_status()
        _render_alert_status()

        st.divider()
        st.subheader("Generate live activity")
        fetch_limit = st.number_input("CelesTrak results to fetch", min_value=1, max_value=50, value=5)
        if st.button("Fetch live CelesTrak conjunctions"):
            try:
                with st.spinner("Screening real CelesTrak data..."):
                    adapter = CelesTrakAdapter(**_celestrak_screening_kwargs())
                    run_once(adapter=adapter, client=client, logger=logger, budget_tracker=budget_tracker, limit=fetch_limit)
                    stats = adapter.last_scan_stats
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't fetch live CelesTrak data: {exc}")
            else:
                if stats:
                    st.success(
                        f"Screened {stats['total_pairs_screened']} pairs across "
                        f"{stats['total_objects']} objects ({stats['groups']})."
                    )
                st.rerun()

        if settings.spacetrack_username and settings.spacetrack_password:
            if st.button("Fetch Space-Track conjunctions (real Pc when available)"):
                try:
                    with st.spinner("Screening real Space-Track data + checking for a real CDM match..."):
                        st_client = SpaceTrackClient(settings.spacetrack_username, settings.spacetrack_password)
                        adapter = EnrichedSpaceTrackAdapter(client=st_client, **_spacetrack_screening_kwargs())
                        entries = run_once(
                            adapter=adapter, client=client, logger=logger,
                            budget_tracker=budget_tracker, limit=fetch_limit,
                        )
                        stats = adapter.last_scan_stats
                except SpaceTrackAuthError as exc:
                    st.error(f"Space-Track authentication failed: {exc}")
                except Exception as exc:  # noqa: BLE001 - report and let the user retry
                    st.error(f"Couldn't fetch live Space-Track data: {exc}")
                else:
                    pc_count = sum(1 for e in entries if e.finding.severity_source == "probability-of-collision")
                    if stats:
                        st.success(
                            f"Screened {stats['total_pairs_screened']} pairs across "
                            f"{stats['total_objects']} objects ({stats['groups']}) - "
                            f"{pc_count} classified by a real Pc, rest by distance threshold."
                        )
                    st.rerun()
        else:
            st.caption(
                "Space-Track screening (real Pc severity when a CDM matches) is available once "
                "SPACETRACK_USERNAME/SPACETRACK_PASSWORD are set in .env."
            )

        if st.button("Run synthetic CRITICAL scenario"):
            try:
                with st.spinner("Running synthetic CRITICAL conjunctions..."):
                    run_id = uuid.uuid4().hex[:8]
                    adapter = SyntheticCriticalAdapter(run_id=run_id, id_prefix="conj-dashboard")
                    run_once(adapter=adapter, client=client, logger=logger, budget_tracker=budget_tracker, limit=4)
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't run the synthetic CRITICAL scenario: {exc}")
            else:
                st.rerun()

        if st.button("Replay historical event (Iridium 33 / Cosmos 2251, 2009)"):
            try:
                with st.spinner("Replaying the real 2009 collision record..."):
                    run_id = uuid.uuid4().hex[:8]
                    # A real independent cross-check (re-propagating real
                    # historical TLEs, not just replaying the documented
                    # number) when Space-Track credentials are configured
                    # - optional, and never blocks the replay itself if
                    # it fails (see HistoricalReplayAdapter's docstring).
                    st_client = None
                    if settings.spacetrack_username and settings.spacetrack_password:
                        st_client = SpaceTrackClient(settings.spacetrack_username, settings.spacetrack_password)
                    adapter = HistoricalReplayAdapter(run_id=run_id, spacetrack_client=st_client)
                    run_once(adapter=adapter, client=client, logger=logger, budget_tracker=budget_tracker, limit=1)
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't replay the historical event: {exc}")
            else:
                st.rerun()

        if st.button("Screen for orbital decay risk"):
            try:
                with st.spinner("Screening real objects for decay risk..."):
                    run_id = uuid.uuid4().hex[:8]
                    adapter = DecayRiskAdapter(run_id=run_id, catalog_ids=settings.watched_norad_ids)
                    run_once(adapter=adapter, client=client, logger=logger, limit=5)
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't screen for decay risk: {exc}")
            else:
                st.rerun()

        if st.button("Run synthetic attitude/pointing-loss scenario"):
            try:
                with st.spinner("Running synthetic attitude/pointing readings..."):
                    run_id = uuid.uuid4().hex[:8]
                    adapter = SyntheticAttitudeAdapter(run_id=run_id)
                    run_once(adapter=adapter, client=client, logger=logger, limit=4)
            except Exception as exc:  # noqa: BLE001 - report and let the user retry
                st.error(f"Couldn't run the attitude scenario: {exc}")
            else:
                st.rerun()

        st.divider()
        if st.button("Refresh"):
            st.rerun()

    entries = logger.load_all_entries()

    _render_metrics(compute_metrics(entries))
    st.divider()
    _render_pending_approvals(logger, pending_approvals(entries), operator)
    st.divider()
    _render_needs_attention(logger, needs_attention(entries), operator)
    st.divider()

    _render_all_decisions_table(entries)
    st.divider()

    _render_trends(entries)
    st.divider()

    _render_mission_log_search(entries, client)
    st.divider()

    _render_review_panel(logger, entries, operator)


main()
