"""Real Prometheus-format metrics for scripts/api.py - a post-Phase-6
improvement (ROADMAP_TO_PRODUCT.md).

Two different kinds of real metric, tracked two different ways:

1. Audit-log-derived gauges (how many CRITICAL findings exist right now,
   what fraction of narration came from real Gemma vs. fallback) are
   computed FRESH from the real DecisionLogger every time Prometheus
   scrapes - a real Collector (the standard prometheus_client extension
   point for "value derived from external state"), not a cached counter
   that could silently drift from what's actually logged.
2. Process-level counters (real HTTP requests handled, real rate-limit
   rejections) are genuine monotonically-increasing Prometheus Counters,
   incremented once per real request as it happens - these describe the
   API process itself, not the audit log's content, so a snapshot
   Collector would be the wrong primitive for them.
"""
from __future__ import annotations

from collections import Counter as PyCounter
from typing import Callable

from prometheus_client import CollectorRegistry, Counter, generate_latest
from prometheus_client.core import GaugeMetricFamily

from .dashboard_data import STATUS_LABELS
from .display import classify_decision_status
from .schemas import DecisionLogEntry

# A dedicated registry, not prometheus_client's global default - avoids
# real "duplicate metric" registration errors if this module is ever
# reloaded (e.g. under a dev-mode `uvicorn --reload`) within one process.
registry = CollectorRegistry()

http_requests_total = Counter(
    "mission_ops_http_requests_total",
    "Total HTTP requests handled, by method/route template/status",
    ["method", "route", "status"],
    registry=registry,
)
rate_limited_requests_total = Counter(
    "mission_ops_rate_limited_requests_total",
    "Total requests rejected by the rate limiter",
    registry=registry,
)


class DecisionLogCollector:
    """load_entries is injected (not a DecisionLogger constructed
    directly in this module) so tests can supply a fixed entry list
    without a real log directory - the same dependency-injection
    reasoning as scripts/api.py's own get_logger() FastAPI dependency."""

    def __init__(self, load_entries: Callable[[], list[DecisionLogEntry]]):
        self._load_entries = load_entries

    def collect(self):
        entries = self._load_entries()

        total = GaugeMetricFamily(
            "mission_ops_decisions_total", "Total decisions currently in the real audit log",
        )
        total.add_metric([], len(entries))
        yield total

        severity_counts = PyCounter(e.finding.severity.value for e in entries)
        by_severity = GaugeMetricFamily(
            "mission_ops_decisions_by_severity", "Logged decisions by severity", labels=["severity"],
        )
        for severity in ("nominal", "watch", "warning", "critical"):
            by_severity.add_metric([severity], severity_counts.get(severity, 0))
        yield by_severity

        status_counts = PyCounter(classify_decision_status(e.decision) for e in entries)
        by_status = GaugeMetricFamily(
            "mission_ops_maneuver_status",
            "CRITICAL-maneuver decisions by real resolution status",
            labels=["status"],
        )
        for status_key in STATUS_LABELS:
            if status_key is None:  # "no maneuver at all" - not a real maneuver-status bucket
                continue
            by_status.add_metric([status_key], status_counts.get(status_key, 0))
        yield by_status

        gemma_count = sum(1 for e in entries if e.rationale_provenance.source == "gemma")
        rationale_ratio = GaugeMetricFamily(
            "mission_ops_gemma_rationale_ratio",
            "Fraction (0-1) of logged rationale that came from real Gemma output, not the deterministic fallback",
        )
        rationale_ratio.add_metric([], (gemma_count / len(entries)) if entries else 0.0)
        yield rationale_ratio


def render_metrics() -> bytes:
    return generate_latest(registry)
