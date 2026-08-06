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

import pandas as pd
import streamlit as st

from src.alerting import send_test_alert
from src.auth import authenticate
from src.config import settings
from src.dashboard_data import (
    ENTRY_ROW_COLUMNS,
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
from src.schemas import DecisionLogEntry, Severity
from src.trends import build_rationale_source_trend_figure, build_severity_trend_figure, is_real_live_source, recurring_objects
from src.ui_style import (
    SEVERITY_BADGE_COLORS,
    SEVERITY_BADGE_TEXT_COLORS,
    muted_severity_rgba,
    severity_badge_html,
    system_note_html,
)

# Scoped to the Approve/Reject buttons specifically via their own real,
# unique `key=` prefixes (Streamlit stamps a `st-key-<key>` class on an
# element's wrapper when it has an explicit key) - never a blanket
# `button[kind=...]` rule, which would also repaint every other button in
# this app (Refresh, Sign out, Acknowledge, ...) that never asked for it.
# Approve is `type="primary"`, which without this would render in
# Streamlit's default theme primary color (a reddish coral) - confusingly
# close to a destructive action for what should read as the safe/affirm
# choice, so this forces a real solid green. Reject stays a neutral
# outline, tinted red only via border/text color, not a filled button -
# visually secondary to Approve, not a second competing call to action.
# A system monospace stack (no network font load, no .streamlit/config.toml
# override needed) rather than a single named face - "JetBrains Mono"/"IBM
# Plex Mono" are named first so a machine that actually has one of them
# installed uses it, but every other name here is a real cross-platform
# monospace already on that OS, so this never silently falls back to the
# proportional sans body font. Ops-dashboard convention (Grafana/Datadog):
# numeric values in monospace/tabular figures so digits align vertically
# across tiles, the same reason a boarding-pass gate number or a stock
# ticker price is never set in a proportional face.
_MONO_FONT_STACK = (
    '"JetBrains Mono", "IBM Plex Mono", ui-monospace, "SF Mono", "Cascadia Code", '
    '"Roboto Mono", Consolas, "Liberation Mono", monospace'
)

_APPROVE_REJECT_CSS = """
<style>
div[class*="st-key-approve_"] button {
    background-color: #16A34A !important;
    border-color: #16A34A !important;
    color: white !important;
}
div[class*="st-key-reject_"] button {
    background-color: transparent !important;
    border: 1px solid #DC2626 !important;
    color: #DC2626 !important;
}
/* Real bug this fixes: st.button(disabled=True) is functionally
   disabled (unclickable) but the solid-green rule above has equal CSS
   specificity and would otherwise keep painting it the same green as a
   normal, clickable Approve - a :disabled selector is strictly more
   specific, so this wins regardless of source order, muting a
   past-TCA Approve to a real greyed-out, "this does nothing" look. */
div[class*="st-key-approve_"] button:disabled {
    background-color: #D1D5DB !important;
    border-color: #D1D5DB !important;
    color: #6B7280 !important;
    cursor: not-allowed;
}
[data-testid="stMetric"] {
    display: flex;
    flex-direction: column-reverse;
    gap: 2px;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}
</style>
"""

# Split out from _APPROVE_REJECT_CSS above only so _MONO_FONT_STACK can be
# interpolated in with an f-string without having to escape every other
# brace in that larger, plain-string CSS block. font-variant-numeric
# rides along even where a fallback face isn't a true monospace - modern
# sans fonts widely support OpenType tabular-nums, which alone already
# gets digits lining up vertically.
_METRIC_VALUE_TYPOGRAPHY_CSS = f"""
<style>
[data-testid="stMetricValue"] {{
    font-weight: 700;
    font-family: {_MONO_FONT_STACK};
    font-variant-numeric: tabular-nums;
}}
</style>
"""

# A colored left-border accent per severity on the "Pending human
# approval"/"Needs attention" cards below - reuses ui_style's own
# SEVERITY_BADGE_COLORS (the same palette the badges/table/charts already
# use) rather than a new, separately-chosen set of colors, so a red
# border always means the same red CRITICAL means everywhere else.
# Scoped via each card's own `sevcard_<severity>_...` container key (the
# same `st-key-*` substring-match technique _APPROVE_REJECT_CSS already
# uses for Approve/Reject), generated from the dict so this can't drift
# out of sync with the badge colors by hand-editing one and not the other.
_SEVERITY_CARD_BORDER_CSS = "<style>\n" + "\n".join(
    f'div[class*="st-key-sevcard_{severity}_"] {{ border-left: 4px solid {color} !important; }}'
    for severity, color in SEVERITY_BADGE_COLORS.items()
) + "\n</style>"

# Which of the 8 top KPI cards represent a risk/problem count worth a
# subtle number tint, and which severity's color that tint borrows from -
# CRITICAL genuinely is a critical-severity count; Rejected/Blocked are
# real operational friction (a human said no, or the delta-v budget ran
# out) rather than an unresolved threat, so they borrow the WARNING hue
# instead of the more alarming CRITICAL red. Total/Autonomous/Human-
# approved/Vetoed/Awaiting stay in the default text color - they're
# either neutral counts or (autonomous/human-approved/vetoed) evidence
# the system is working as intended, not a problem count.
_KPI_RISK_TINTS = {"critical": "critical", "rejected": "warning", "blocked": "warning"}

# Only the number, never the label or the card background - subtle by
# design (per the UI review), via muted_severity_rgba's alpha blending
# rather than a second hand-picked "risk color" palette.
_KPI_TINT_CSS = "<style>\n" + "\n".join(
    f'div[class*="st-key-kpi_{slug}"] [data-testid="stMetricValue"] '
    f"{{ color: {muted_severity_rgba(severity)} !important; }}"
    for slug, severity in _KPI_RISK_TINTS.items()
) + "\n</style>"

# Real gap this closes: st.divider() defaults to a real 32px top+bottom
# margin (64px of blank vertical space per divider, measured live) - with
# a divider between nearly every section on this page, that added up to
# a genuinely uneven, "spacious" feel wherever a section's own content
# was short (a title + caption + button, say) versus a section that's
# inherently dense (the decisions table). Halving it tightens every
# section boundary on the page uniformly, rather than hand-tuning
# individual sections' own padding one at a time.
_DENSITY_CSS = """
<style>
hr { margin: 1rem 0 !important; }
</style>
"""

# Replaces st.tabs() as the page's top-level section navigation. Real bug
# this fixes: st.tabs() renders every panel's content into the DOM on
# every run and only toggles visibility via CSS (display:none) - it never
# unmounts the inactive ones. The "Full Log" tab's decisions table is
# glide-data-grid, a canvas-based grid (see _METRIC_VALUE_TYPOGRAPHY_CSS's
# docstring for the same rendering engine flagged in Phase 1) that
# measures its own container's pixel width once, when it first mounts. A
# viewer who clicks straight into "Full Log" as their very first action
# causes that mount to happen while the panel is still hidden behind
# another tab's display:none, so the grid measures a zero-width container
# and never draws anything - reproduced live via DOM inspection (the
# table's wrapper measured 0x0px, zero <canvas> elements present, even
# seconds after the click).
#
# The standard-looking fix (inject a small script that dispatches a
# window resize event once the tab becomes visible) was tried and
# empirically does NOT work here: a resize/scroll event dispatched from
# injected JS is never a trusted browser event (Event.isTrusted is
# read-only and can't be set by a script), and this specific bug only
# clears on a genuinely trusted browser gesture (a real click physically
# toggling the panel's CSS away and back) - confirmed live by testing a
# real CDP-level window resize (still trusted, still didn't fix it) next
# to a real trusted tab-switch click (reliably did). Since a page script
# fundamentally cannot synthesize a trusted event, no JS-only fix is
# possible without either forking glide-data-grid or never mounting the
# grid hidden in the first place.
#
# So instead of st.tabs(), the active section is tracked explicitly in
# st.session_state and only that section's content is ever executed -
# the "Full Log" table is never mounted while hidden, because it's never
# mounted at all until the button below actually makes it the active
# section (a real Python rerun, at which point its container starts
# visible from the very first paint). This also means the active section
# now survives st.rerun() calls from other actions (Approve/Reject/
# Refresh/...) - a strict improvement over st.tabs(), whose selection was
# purely client-side and never visible to the Python script at all.
_SECTION_TABS: list[tuple[str, str]] = [
    ("Overview", "overview"),
    ("Approvals & Attention", "approvals"),
    ("Trends", "trends"),
    ("Full Log", "full_log"),
]
_ACTIVE_SECTION_KEY = "active_section"

# Strips the default button chrome (border, background, rounded corners)
# down to a plain text label with a bottom-border underline, so the
# button row reads as a tab bar rather than a row of buttons - the active
# tab gets the underline + full-opacity bold text, inactive tabs are
# muted. #FF4B4B matches Streamlit's own default accent color (the same
# red st.tabs() itself uses for its active-tab underline), so swapping
# the implementation doesn't change the visible accent color.
_SECTION_NAV_CSS = """
<style>
div[class*="st-key-navtab_"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    box-shadow: none !important;
    opacity: 0.6;
    font-weight: 400;
    padding: 0.25rem 0.1rem 0.6rem 0.1rem !important;
}
div[class*="st-key-navtab_"] button:hover {
    opacity: 1;
    color: #FF4B4B !important;
}
div[class*="st-key-navtab_"][class*="-active"] button {
    border-bottom: 2px solid #FF4B4B !important;
    opacity: 1;
    font-weight: 600;
}
</style>
"""


def _render_section_nav() -> str:
    """Renders the tab-styled button row and returns the currently active
    section's label (one of _SECTION_TABS' first elements) - reading
    st.session_state fresh after any click, not a stale closed-over
    value, so the caller always sees this run's real selection."""
    if _ACTIVE_SECTION_KEY not in st.session_state:
        st.session_state[_ACTIVE_SECTION_KEY] = _SECTION_TABS[0][0]

    cols = st.columns(len(_SECTION_TABS))
    for col, (label, slug) in zip(cols, _SECTION_TABS):
        is_active = st.session_state[_ACTIVE_SECTION_KEY] == label
        key = f"navtab_{slug}-active" if is_active else f"navtab_{slug}"
        with col:
            if st.button(label, key=key, use_container_width=True):
                st.session_state[_ACTIVE_SECTION_KEY] = label
                st.rerun()
    return st.session_state[_ACTIVE_SECTION_KEY]


def _render_hero_status(pending: list[DecisionLogEntry], attention: list[DecisionLogEntry]) -> None:
    """A single, prominent "state of the world" strip - the first thing
    meant to land, above everything else on the page. Counts genuinely
    OPEN critical situations - NOT metrics["critical"] (a lifetime total
    that includes everything already resolved: autonomous-executed,
    human-approved, vetoed, rejected). Deliberately no new counting logic:
    `pending` and `attention` are the exact same real lists
    (pending_approvals(entries), needs_attention(entries) - see
    src/dashboard_data.py) that already feed the "Pending human approval
    (N)" and "Needs attention (N)" section headers further down this
    page, passed in rather than recomputed here. Both are filtered to
    CRITICAL severity defensively - pending_approvals() is CRITICAL-only
    by construction (awaiting_human_approval is only ever set on a
    CRITICAL conjunction) and needs_attention() already filters to
    CRITICAL in its own definition, but filtering again here means this
    can never silently count a WATCH/WARNING entry even if either of
    those invariants ever changes."""
    open_critical = (
        sum(1 for e in pending if e.finding.severity == Severity.CRITICAL)
        + sum(1 for e in attention if e.finding.severity == Severity.CRITICAL)
    )
    severity = "critical" if open_critical > 0 else "nominal"
    color = SEVERITY_BADGE_COLORS[severity]

    if open_critical > 0:
        situation_word = "situation" if open_critical == 1 else "situations"
        headline = (
            f'<span style="font-family:{_MONO_FONT_STACK};font-weight:800;">{open_critical}</span> '
            f"critical {situation_word} open"
        )
    else:
        headline = "ALL CLEAR &mdash; no critical situations open"

    st.markdown(
        f'<div style="background-color:{muted_severity_rgba(severity, 0.12)};'
        f'border:1px solid {muted_severity_rgba(severity, 0.4)};border-left:6px solid {color};'
        f'border-radius:0.5rem;padding:0.9rem 1.25rem;margin-bottom:1rem;">'
        f'<span style="font-size:1.35rem;letter-spacing:0.02em;color:{color};">{headline}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_metrics(metrics: dict) -> None:
    """3-per-row grid of bordered metric cards (3 rows: 3+3+2) - a real
    number reads faster as a scannable grid than as two anonymous rows of
    st.metric calls. Real bug this fixes: at 4 cards per row, the
    longer labels ("Human-approved executed", "Rejected by human",
    "Awaiting approval", "Blocked by budget", "Vetoed by Gemma") clipped
    mid-word behind an ellipsis at the dashboard's normal content width
    (~1100px) - 3 per row gives each card ~33% instead of ~25% of the row,
    which is enough for every real label here to render in full. Still
    plain st.metric widgets underneath (not custom HTML) - the CSS
    injected in main() (_APPROVE_REJECT_CSS) flips each card's internal
    layout so the big bold value sits above its small uppercase label,
    without giving up Streamlit's native metric styling/accessibility.
    Each card also gets a stable `kpi_<slug>` container key - unused for
    layout, but what _KPI_TINT_CSS above hooks into to tint just the
    number (not the label/card) for the risk-indicating KPIs."""
    cards = [
        ("total", "Total events", metrics["total"]),
        ("critical", "CRITICAL", metrics["critical"]),
        ("autonomous", "Autonomous executed", metrics["executed_autonomous"]),
        ("human_approved", "Human-approved executed", metrics["executed_human_approved"]),
        ("vetoed", "Vetoed by Gemma", metrics["vetoed_by_gemma"]),
        ("rejected", "Rejected by human", metrics["rejected_by_human"]),
        ("awaiting", "Awaiting approval", metrics["awaiting_human_approval"]),
        ("blocked", "Blocked by budget", metrics["budget_insufficient"]),
    ]
    for row_start in range(0, len(cards), 3):
        row_cards = cards[row_start:row_start + 3]
        row = st.columns(3)
        for col, (slug, label, value) in zip(row, row_cards):
            with col.container(border=True, key=f"kpi_{slug}"):
                st.metric(label, value)

    st.markdown(
        '<div style="margin-top:0.5rem;">'
        '<span style="background-color:rgba(127,127,127,0.15);padding:3px 10px;'
        'border-radius:999px;font-size:0.8rem;font-weight:600;">'
        f"Gemma-authored rationale: {metrics['gemma_rationale_pct']:.0f}% "
        "(rest is deterministic fallback)</span></div>",
        unsafe_allow_html=True,
    )


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
        df = pd.DataFrame(entries_to_rows(filtered), columns=ENTRY_ROW_COLUMNS)
        styled = df.style.map(_severity_cell_style, subset=["severity"])
        st.dataframe(
            styled, width="stretch", hide_index=True,
            column_config={
                # Real bug this fixes: the default auto-sized column was
                # clipping the full "YYYY-MM-DD HH:MM:SS+00:00" timestamp
                # text for these two columns specifically - both wide
                # enough to matter, unlike the other columns.
                "time_of_closest_approach": st.column_config.DatetimeColumn(
                    "time_of_closest_approach", width="medium", format="YYYY-MM-DD HH:mm:ss",
                ),
                "timestamp": st.column_config.DatetimeColumn(
                    "timestamp", width="medium", format="YYYY-MM-DD HH:mm:ss",
                ),
                # Real bug this fixes: "action" (and its short
                # categorical neighbors) had no explicit width, leaving
                # them to whatever the grid's auto-fit happened to
                # squeeze them to - reported as silently clipped at the
                # table's edge. An explicit "small" width reserves enough
                # room for the real values here ("continue"/"hold"/
                # "abort", a source slug, etc.) regardless of what's
                # scrolled into view - the table already scrolls
                # horizontally on its own for the remaining columns.
                "action": st.column_config.TextColumn("action", width="small"),
                "severity": st.column_config.TextColumn("severity", width="small"),
                "source": st.column_config.TextColumn("source", width="small"),
            },
        )
    else:
        st.caption("No decisions match the selected filters.")


def _severity_cell_style(value: object) -> str:
    """Pandas Styler callback (see _render_all_decisions_table) - the same
    severity badge color palette as severity_badge_html, applied as a
    real cell background rather than an HTML span, since st.dataframe
    doesn't render arbitrary HTML inside cells. Text color comes from
    SEVERITY_BADGE_TEXT_COLORS (not a flat white) for the same real WCAG
    contrast reason severity_badge_html uses it - white-on-watch was only
    a 2.94:1 contrast ratio."""
    severity = str(value).lower()
    color = SEVERITY_BADGE_COLORS.get(severity, "#6B7280")
    text_color = SEVERITY_BADGE_TEXT_COLORS.get(severity, "white")
    return (
        f"background-color:{color};color:{text_color};font-weight:600;"
        "text-align:center;text-transform:uppercase;border-radius:4px;"
    )


def _render_pending_approvals(logger: DecisionLogger, pending: list[DecisionLogEntry], operator: str) -> None:
    st.subheader(f"Pending human approval ({len(pending)})")
    if not pending:
        st.caption("Nothing awaiting approval right now.")
        return

    for entry in pending:
        plan = entry.decision.maneuver_plan
        raw = entry.telemetry.raw_data
        severity = entry.finding.severity.value
        card_key = f"sevcard_{severity}_pending_{entry.telemetry.event_id}"
        with st.container(border=True, key=card_key):
            st.markdown(
                f"**{entry.telemetry.event_id}** — {raw.get('object_a_name')} vs {raw.get('object_b_name')} "
                f"{severity_badge_html(severity)}",
                unsafe_allow_html=True,
            )
            st.write(
                f"Min distance: {raw.get('min_distance_km'):.2f} km  |  "
                f"Proposed: {plan.direction}, ~{plan.magnitude_delta_v:.2f} m/s delta-v  |  "
                f"Target clearance: {plan.target_clearance_km:.1f} km"
            )
            # Real, decision-relevant urgency - a human approving/
            # rejecting needs to know not just how close this gets but
            # whether the event it's FOR has already happened, since
            # approving a maneuver after its own TCA has no effect.
            # tca_expired also drives disabling the Approve button below
            # - the warning text alone wasn't enough to stop someone from
            # clicking a fully-enabled Approve on a maneuver that can no
            # longer do anything.
            tca_expired = False
            tca_raw = raw.get("time_of_closest_approach")
            if tca_raw:
                urgency = tca_urgency_label(tca_raw)
                if "already passed" in urgency:
                    tca_expired = True
                    st.warning(
                        f"{urgency} - approving this maneuver no longer has any effect.",
                        icon=":material/warning:",
                    )
                else:
                    st.caption(urgency)
            st.caption(entry.decision.rationale)
            col_approve, col_reject = st.columns(2)
            approve_help = (
                "TCA has already passed - approving this maneuver no longer has any effect."
                if tca_expired else None
            )
            if col_approve.button(
                "✓ Approve", key=f"approve_{entry.telemetry.event_id}", type="primary",
                disabled=tca_expired, help=approve_help,
            ):
                try:
                    logger.approve_maneuver(entry.telemetry.event_id, approved=True, approved_by=operator)
                except ValueError as exc:
                    st.error(f"Couldn't approve this maneuver: {exc}")
                else:
                    st.rerun()
            # Reject stays enabled even past TCA - closing out a stale
            # item (so it stops sitting in this queue) is still a real,
            # useful action, unlike Approve which would do nothing.
            if col_reject.button("✕ Reject", key=f"reject_{entry.telemetry.event_id}", type="secondary"):
                try:
                    logger.approve_maneuver(entry.telemetry.event_id, approved=False, approved_by=operator)
                except ValueError as exc:
                    st.error(f"Couldn't reject this maneuver: {exc}")
                else:
                    st.rerun()


def _needs_attention_group_key(entry: DecisionLogEntry) -> tuple[str, str | None]:
    """Groups a needs_attention entry by real object identity + the one
    metric that actually distinguishes it (perigee altitude or pointing
    error) - NOT by its Gemma-authored rationale text, which genuinely
    varies in phrasing between two otherwise-identical findings (same
    object, same reading) since it's a fresh LLM call each time. Two
    entries with the same object name and the same rounded metric are the
    same real finding logged more than once, even if their rationale
    sentences read differently word-for-word."""
    raw = entry.telemetry.raw_data
    object_name = raw.get("object_name", entry.telemetry.event_id)
    if "perigee_altitude_km" in raw:
        metric_label = f"perigee {raw['perigee_altitude_km']:.1f} km"
    elif "pointing_error_deg" in raw:
        metric_label = f"pointing error {raw['pointing_error_deg']:.0f}°"
    else:
        metric_label = None
    return (object_name, metric_label)


def _render_single_needs_attention_card(logger: DecisionLogger, entry: DecisionLogEntry, operator: str) -> None:
    """The body of one needs-attention card for a single entry - factored
    out so a group of exactly one (the common case once real findings
    stop repeating) renders identically to before this grouping existed."""
    raw = entry.telemetry.raw_data
    subject = raw.get("object_name", entry.telemetry.event_id)
    st.markdown(
        f"**{entry.telemetry.event_id}** — {subject} {severity_badge_html(entry.finding.severity.value)}",
        unsafe_allow_html=True,
    )
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


def _render_needs_attention(logger: DecisionLogger, attention: list[DecisionLogEntry], operator: str) -> None:
    """A real CRITICAL decay/attitude finding has no maneuver/approval
    workflow of its own (see needs_attention's docstring,
    src/dashboard_data.py) - there's no avoidance burn for "your perigee
    is too low" or "you're tumbling" - so without a dedicated section
    like this one, it was previously indistinguishable from NOMINAL/WATCH
    noise in the "All decisions" table. "Acknowledge" reuses the same
    mark_reviewed this dashboard's Inspect panel already uses - there's
    no separate approve/reject decision to make here, just a real human
    confirming they've seen it.

    Display-layer grouping only, over the same real entries needs_attention()
    already returns - repeated findings sharing the same real object and
    metric (e.g. the same synthetic fixture logged several times in one
    demo run) collapse into one summary card with a count and an expander,
    instead of N nearly-identical cards differing only by event_id. A
    group of size 1 renders exactly as it did before this existed."""
    st.subheader(f"Needs attention ({len(attention)})")
    if not attention:
        st.caption("No CRITICAL decay/attitude findings awaiting acknowledgment.")
        return

    groups: dict[tuple[str, str | None], list[DecisionLogEntry]] = {}
    for entry in attention:
        groups.setdefault(_needs_attention_group_key(entry), []).append(entry)

    for group_index, (group_key, group_entries) in enumerate(groups.items()):
        object_name, metric_label = group_key
        severity = group_entries[0].finding.severity.value
        card_key = f"sevcard_{severity}_attn_{group_index}"
        with st.container(border=True, key=card_key):
            if len(group_entries) == 1:
                _render_single_needs_attention_card(logger, group_entries[0], operator)
                continue

            header = f"{len(group_entries)} {object_name} events"
            if metric_label:
                header += f" — {metric_label}"
            st.markdown(f"**{header}** {severity_badge_html(severity)}", unsafe_allow_html=True)
            st.caption("Same finding logged multiple times - expand to review or acknowledge individually.")
            with st.expander(f"Show {len(group_entries)} individual events"):
                if st.button(f"Acknowledge all {len(group_entries)}", key=f"acknowledge_group_{card_key}"):
                    errors = []
                    for entry in group_entries:
                        try:
                            logger.mark_reviewed(entry.telemetry.event_id, reviewed_by=operator)
                        except ValueError as exc:
                            errors.append(f"{entry.telemetry.event_id}: {exc}")
                    if errors:
                        st.error("Some entries couldn't be acknowledged: " + "; ".join(errors))
                    else:
                        st.rerun()
                for entry in group_entries:
                    st.markdown(f"**{entry.telemetry.event_id}**")
                    st.caption(entry.decision.rationale)
                    if st.button("Acknowledge", key=f"acknowledge_{entry.telemetry.event_id}"):
                        try:
                            logger.mark_reviewed(entry.telemetry.event_id, reviewed_by=operator)
                        except ValueError as exc:
                            st.error(f"Couldn't acknowledge this finding: {exc}")
                        else:
                            st.rerun()
                    st.divider()


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
                    object_a_group=raw.get("object_a_group"), object_b_group=raw.get("object_b_group"),
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


# (field key in telemetry.raw_data, display label, format string) - only
# whichever of these an entry's own hazard shape actually carries gets
# shown (a conjunction has min_distance_km, a decay reading has
# perigee_altitude_km, an attitude reading has pointing_error_deg/
# solar_panel_power_pct - never all of them at once). See
# _render_telemetry_summary below.
_TELEMETRY_SUMMARY_FIELDS = [
    ("min_distance_km", "Min distance (km)", "{:.2f}"),
    ("perigee_altitude_km", "Perigee altitude (km)", "{:.1f}"),
    ("pointing_error_deg", "Pointing error (deg)", "{:.1f}"),
    ("solar_panel_power_pct", "Solar panel power (%)", "{:.0f}"),
    ("collision_probability", "Collision probability (Pc)", "{:.2e}"),
]


def _render_telemetry_summary(entry: DecisionLogEntry) -> None:
    """A side-by-side key-value row of this entry's real operational
    metrics, pulled out ahead of the full nested JSON below so an
    operator can read the numbers that actually matter (whichever this
    entry's hazard type carries) without first expanding/scanning the raw
    record. Gemma call latency is always present (every entry has a real
    rationale_provenance), so it's always included."""
    raw = entry.telemetry.raw_data
    present = [
        (label, fmt.format(raw[key]))
        for key, label, fmt in _TELEMETRY_SUMMARY_FIELDS
        if raw.get(key) is not None
    ]
    present.append(("Gemma latency (ms)", f"{entry.rationale_provenance.latency_ms:.0f}"))
    for col, (label, value) in zip(st.columns(len(present)), present):
        col.metric(label, value)


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

    st.markdown(
        f"**{selected.telemetry.event_id}** {severity_badge_html(selected.finding.severity.value)}",
        unsafe_allow_html=True,
    )
    _render_telemetry_summary(selected)
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
            f":material/satellite_alt: Monitoring your own asset(s): NORAD ID(s) {', '.join(settings.watched_norad_ids)}"
        )
    else:
        st.sidebar.markdown(
            system_note_html(
                "Monitoring CelesTrak's demo 'stations' group (ISS, Tiangong, ...), "
                "not a specific asset. Set WATCHED_NORAD_IDS in .env to your own "
                "satellite's NORAD catalog ID to monitor it instead.",
                icon="satellite_alt",
            ),
            unsafe_allow_html=True,
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
    label = ":material/settings: Hazard severity thresholds"
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
        st.sidebar.markdown(
            system_note_html(
                "CRITICAL-event webhook alerting is not configured. "
                "Set ALERT_WEBHOOK_URL in .env to get real-time pages for CRITICAL findings.",
                icon="notifications_off",
            ),
            unsafe_allow_html=True,
        )
        return

    st.sidebar.success(":material/notifications_active: CRITICAL-event webhook alerting is configured.")
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
        # A neutral system-note box, not st.warning - a design-review
        # pass found this reading as a risk finding (it shared the same
        # yellow/amber family as a real WATCH-severity badge) when it's
        # really just an app-config reminder, not anything the pipeline
        # is tracking as dangerous. See ui_style.system_note_html.
        st.markdown(
            system_note_html(
                "Unauthenticated dashboard - OPERATOR_TOKENS is not configured, so "
                "the operator name below is free text anyone with this page open can "
                "set to anything. Set OPERATOR_TOKENS in .env before using this for "
                "real approve/reject/review actions.",
                icon="warning",
            ),
            unsafe_allow_html=True,
        )
        return st.sidebar.text_input("Operator name", value="dashboard-operator")

    if "authenticated_operator" in st.session_state:
        operator = st.session_state["authenticated_operator"]
        st.sidebar.success(f"Signed in as **{operator}**")
        if st.sidebar.button("Sign out"):
            del st.session_state["authenticated_operator"]
            st.rerun()
        return operator

    st.markdown(
        system_note_html("Enter your operator token to continue.", icon="login"),
        unsafe_allow_html=True,
    )
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
    st.set_page_config(page_title="ConjunctionWatch: Mission Ops", layout="wide")
    st.markdown(_APPROVE_REJECT_CSS, unsafe_allow_html=True)
    st.markdown(_METRIC_VALUE_TYPOGRAPHY_CSS, unsafe_allow_html=True)
    st.markdown(_SEVERITY_CARD_BORDER_CSS, unsafe_allow_html=True)
    st.markdown(_KPI_TINT_CSS, unsafe_allow_html=True)
    st.markdown(_DENSITY_CSS, unsafe_allow_html=True)
    st.title("ConjunctionWatch: Mission Ops Dashboard")
    st.caption(
        "Live risk board and human-approval inbox over the real append-only audit log "
        "(logs/decisions-*.jsonl) - not a mock, not a second copy of the pipeline."
    )

    operator = _require_operator_identity()

    client = _get_shared_client()
    budget_tracker = _get_shared_budget_tracker()
    logger = DecisionLogger(settings=settings)
    # Loaded here (not further down, where this call used to live) so the
    # hero status strip below - meant to be the first thing a viewer's
    # eye lands on - has the real counts available before anything else
    # renders. entries/metrics/pending/attention are each computed
    # exactly once and reused by every section below (_render_metrics,
    # the hero strip, pending approvals, needs attention, the table,
    # Trends) - never reloaded or recomputed.
    entries = logger.load_all_entries()
    metrics = compute_metrics(entries)
    pending = pending_approvals(entries)
    attention = needs_attention(entries)

    with st.sidebar:
        # Three visual sub-sections, in the order an operator actually
        # needs them: who am I / what's configured, what's being watched,
        # then the buttons that generate new activity.
        st.subheader(":material/settings: System Config & Auth")
        st.write(f"Configured Gemma backend: **{settings.gemma_backend}**")
        st.caption(
            "local (ollama) -> autonomous execution with Gemma's own GO/NO-GO veto review; "
            "cloud (api) -> maneuvers held pending human approval below."
        )

        st.divider()
        st.subheader(":material/satellite_alt: Active Monitors")
        _render_monitoring_status()
        _render_severity_threshold_status()
        _render_alert_status()

        st.divider()
        st.subheader(":material/science: Simulation Triggers")
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

    # Structural reorganization (a design pass, not a rewrite of any
    # section's contents) - the page used to be one long top-to-bottom
    # scroll; grouped into named sections so an operator can jump
    # straight to "Approvals & Attention" or "Full Log" instead of
    # scrolling past everything else every time. See _SECTION_NAV_CSS's
    # docstring for why this is a session-state-tracked button row
    # instead of st.tabs() - every _render_* call below is otherwise
    # unchanged, just gated behind `if active_section == ...:` instead of
    # `with tab:` - same functions, same arguments, same entries/
    # metrics/pending/attention computed once above and passed through.
    st.markdown(_SECTION_NAV_CSS, unsafe_allow_html=True)
    active_section = _render_section_nav()
    st.divider()

    if active_section == "Overview":
        # Hero status first - Overview is the default section on a
        # fresh page load, so this is still the first thing seen.
        _render_hero_status(pending, attention)
        _render_live_tracking()
        st.divider()
        _render_metrics(metrics)

    elif active_section == "Approvals & Attention":
        _render_pending_approvals(logger, pending, operator)
        st.divider()
        _render_needs_attention(logger, attention, operator)

    elif active_section == "Trends":
        _render_trends(entries)

    elif active_section == "Full Log":
        # Mission-log search and the review/JSON-inspector panel aren't
        # named explicitly in the "Full Log" spec, but both are real
        # ways of looking INTO the log (a natural-language search over
        # it, and a raw per-entry inspector) rather than an aggregate
        # view like Trends - grouped here rather than invented a 5th
        # section.
        _render_all_decisions_table(entries)
        st.divider()
        _render_mission_log_search(entries, client)
        st.divider()
        _render_review_panel(logger, entries, operator)


main()
