"""Trend/analytics view over the real audit log - pure data transforms,
no Streamlit here (same separation dashboard_data.py already uses so
these stay directly unit-testable), and no new network/AI calls, just
aggregating what's already logged.

Every other dashboard view answers "what does this ONE event look like"
(the decision table, the inspect panel) or "what's happening right now"
(metrics, live tracking). This answers "what's the pattern over time" -
severity mix per day, which real objects keep showing up across
separate scans, and how much of the narration is genuinely coming from
Gemma vs. the deterministic fallback - none of which is visible anywhere
else in this dashboard, which only ever shows one event or one instant
at a time.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date

import plotly.graph_objects as go

from .schemas import DecisionLogEntry

SEVERITY_ORDER = ["nominal", "watch", "warning", "critical"]
SEVERITY_COLORS = {"nominal": "#2ca02c", "watch": "#bcbd22", "warning": "#ff7f0e", "critical": "#d62728"}

# Real, LIVE scans only - not the synthetic fixtures
# (SyntheticCriticalAdapter, SyntheticAttitudeAdapter) or the historical
# replay. A QA pass found that the trend charts below were originally
# aggregating every logged entry indiscriminately: after a demo has been
# run a few times, the severity-by-day chart is mostly synthetic-fixture
# noise, not real screened data - a viewer could easily read "lots of
# real CRITICAL conjunctions" off a chart that's actually mostly repeated
# demo runs, directly contradicting this project's own repeatedly-verified
# finding that real data rarely lands in CRITICAL. The historical replay
# is excluded too: it's real-in-origin, but replaying the same fixed 2009
# record on every demo run makes it behave like synthetic data for "trend
# over time" purposes - it doesn't represent anything new happening.
#
# "spacetrack" (SpaceTrackAdapter/EnrichedSpaceTrackAdapter) is real and
# live too, and was originally missing here - a second real QA pass
# found real Space-Track scans were silently excluded from these charts
# and mislabeled "real: false" in the Recurring Objects table below,
# even though they can carry a genuinely higher-confidence real Pc than
# a plain CelesTrak distance screen.
REAL_LIVE_SOURCES = {"celestrak", "celestrak-decay", "spacetrack"}


def is_real_live_source(entry: DecisionLogEntry) -> bool:
    """True only for entries from a real, live CelesTrak scan (real
    current conjunctions or real current decay-risk screening) - False
    for synthetic fixtures and the historical replay. See
    REAL_LIVE_SOURCES for why both are excluded from "real" here."""
    return entry.telemetry.source in REAL_LIVE_SOURCES


def _day(entry: DecisionLogEntry) -> date:
    """The real calendar day this entry belongs to - keyed off
    decision.made_at (set right after the Gemma call completes, moments
    before log_node writes it to disk) rather than telemetry.timestamp
    (set at ingestion, before analyze/decide even run). A QA pass found
    that using telemetry.timestamp could disagree with which
    logs/decisions-YYYY-MM-DD.jsonl file an entry actually landed in, if
    a slow Gemma call (this project has observed 2-35s+ for the hosted
    API) happened to straddle a UTC-midnight boundary; made_at narrows
    that gap to log_node's own near-instant write, not the full
    analyze-decide latency."""
    return entry.decision.made_at.date()


def severity_counts_by_day(entries: list[DecisionLogEntry]) -> dict[date, dict[str, int]]:
    """{day: {severity_value: count}} - every day that has at least one
    logged entry gets a key, and every severity actually seen that day is
    present in its inner dict (missing severities are simply absent, not
    zero-filled - callers building a chart handle that with .get(x, 0))."""
    counts: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        counts[_day(entry)][entry.finding.severity.value] += 1
    return {day: dict(day_counts) for day, day_counts in counts.items()}


def build_severity_trend_figure(entries: list[DecisionLogEntry]) -> go.Figure:
    """Stacked bar, one bar per real day this log has activity for, one
    color per severity - real day-level granularity, matching how this
    project's own log files are already split (logs/decisions-YYYY-MM-DD.jsonl),
    not an arbitrary bucket size. Callers should pass only real-live
    entries (see is_real_live_source) - this function itself stays
    source-agnostic and directly testable with any entries, but the
    dashboard filters before calling it."""
    by_day = severity_counts_by_day(entries)
    days = sorted(by_day.keys())

    fig = go.Figure()
    for severity in SEVERITY_ORDER:
        y = [by_day[day].get(severity, 0) for day in days]
        fig.add_trace(go.Bar(x=days, y=y, name=severity.upper(), marker_color=SEVERITY_COLORS[severity]))
    fig.update_layout(
        barmode="stack", title="Findings by severity, per day (real live scans only)",
        xaxis_title="Day", yaxis_title="Count",
    )
    return fig


def _object_ids_and_names(entry: DecisionLogEntry) -> list[tuple[str, str]]:
    """Every real object identity a logged entry touches - two for a
    conjunction pair, one for a single-object hazard (decay/attitude),
    none for telemetry with no real object identity at all (e.g.
    DummyAdapter's generic payload)."""
    raw = entry.telemetry.raw_data
    if "object_a_id" in raw:
        return [(raw["object_a_id"], raw["object_a_name"]), (raw["object_b_id"], raw["object_b_name"])]
    if "object_id" in raw:
        return [(raw["object_id"], raw["object_name"])]
    return []


def recurring_objects(entries: list[DecisionLogEntry], top_n: int = 10) -> list[dict]:
    """Real objects ranked by how many separate logged events they
    appeared in, most-frequent first - "which real objects keep showing
    up across scans" isn't visible anywhere else in this dashboard.

    Grouped by object_id ALONE, not by (object_id, object_name) - a QA
    pass found that grouping by the pair would silently undercount a
    real object's recurrence (splitting it into two separate rows) if its
    name string ever varied by so much as whitespace between two
    fetches. The most-recently-seen name is what's displayed. Includes
    entries from every source (synthetic fixtures included, unlike the
    two trend charts above) since "did I click this demo button 30
    times" is still real information worth seeing - but each row is
    labeled "real" so a synthetic object's count is never mistaken for
    a real one."""
    counts: Counter[str] = Counter()
    latest_name: dict[str, str] = {}
    is_real: dict[str, bool] = {}
    for entry in entries:
        real = is_real_live_source(entry)
        for object_id, object_name in _object_ids_and_names(entry):
            counts[object_id] += 1
            latest_name[object_id] = object_name
            is_real[object_id] = real
    return [
        {"object_id": object_id, "object_name": latest_name[object_id], "count": count, "real": is_real[object_id]}
        for object_id, count in counts.most_common(top_n)
    ]


def rationale_source_counts_by_day(entries: list[DecisionLogEntry]) -> dict[date, dict[str, int]]:
    """{day: {"gemma": count, "fallback": count}} - same day-bucketing as
    severity_counts_by_day, but tracking GemmaProvenance.source instead."""
    counts: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        counts[_day(entry)][entry.rationale_provenance.source] += 1
    return {day: dict(day_counts) for day, day_counts in counts.items()}


def build_rationale_source_trend_figure(entries: list[DecisionLogEntry]) -> go.Figure:
    """Stacked bar showing how much of each day's narration genuinely
    came from Gemma vs. the deterministic fallback - a day where fallback
    spikes is a real signal (Gemma was unreachable a lot that day), not
    just a cosmetic detail. Same real-live-only convention as
    build_severity_trend_figure - the dashboard filters before calling."""
    by_day = rationale_source_counts_by_day(entries)
    days = sorted(by_day.keys())

    fig = go.Figure()
    for source, color in [("gemma", "#1f77b4"), ("fallback", "#7f7f7f")]:
        y = [by_day[day].get(source, 0) for day in days]
        fig.add_trace(go.Bar(x=days, y=y, name=source, marker_color=color))
    fig.update_layout(
        barmode="stack", title="Rationale source per day, real live scans only (Gemma vs. deterministic fallback)",
        xaxis_title="Day", yaxis_title="Count",
    )
    return fig
