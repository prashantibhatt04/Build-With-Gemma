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


def _day(entry: DecisionLogEntry) -> date:
    return entry.telemetry.timestamp.date()


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
    not an arbitrary bucket size."""
    by_day = severity_counts_by_day(entries)
    days = sorted(by_day.keys())

    fig = go.Figure()
    for severity in SEVERITY_ORDER:
        y = [by_day[day].get(severity, 0) for day in days]
        fig.add_trace(go.Bar(x=days, y=y, name=severity.upper(), marker_color=SEVERITY_COLORS[severity]))
    fig.update_layout(
        barmode="stack", title="Findings by severity, per day",
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
    up across scans" isn't visible anywhere else in this dashboard."""
    counter: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        for object_id, object_name in _object_ids_and_names(entry):
            counter[(object_id, object_name)] += 1
    return [
        {"object_id": object_id, "object_name": object_name, "count": count}
        for (object_id, object_name), count in counter.most_common(top_n)
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
    just a cosmetic detail."""
    by_day = rationale_source_counts_by_day(entries)
    days = sorted(by_day.keys())

    fig = go.Figure()
    for source, color in [("gemma", "#1f77b4"), ("fallback", "#7f7f7f")]:
        y = [by_day[day].get(source, 0) for day in days]
        fig.add_trace(go.Bar(x=days, y=y, name=source, marker_color=color))
    fig.update_layout(
        barmode="stack", title="Rationale source per day (Gemma vs. deterministic fallback)",
        xaxis_title="Day", yaxis_title="Count",
    )
    return fig
