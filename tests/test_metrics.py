"""Tests for src/metrics.py - real prometheus_client objects throughout
(not mocked), parsing the real exposition-format text render_metrics()
produces to confirm the actual values, not just that it runs."""
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, generate_latest

from src.metrics import DecisionLogCollector, http_requests_total, rate_limited_requests_total, registry
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _entry(event_id: str, severity: Severity, rationale_source: str = "gemma") -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={"min_distance_km": 50.0},
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="Test.", confidence=0.8)
    decision = Decision(action="continue", rationale="Test.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source=rationale_source, model_used="fake", latency_ms=1.0),
    )


def _render(collector) -> str:
    registry = CollectorRegistry()
    registry.register(collector)
    return generate_latest(registry).decode("utf-8")


def test_reports_total_decisions():
    entries = [_entry("e1", Severity.WATCH), _entry("e2", Severity.CRITICAL)]
    output = _render(DecisionLogCollector(load_entries=lambda: entries))

    assert 'mission_ops_decisions_total 2.0' in output


def test_reports_decisions_by_severity_including_zero_counts():
    entries = [_entry("e1", Severity.CRITICAL), _entry("e2", Severity.CRITICAL)]
    output = _render(DecisionLogCollector(load_entries=lambda: entries))

    assert 'mission_ops_decisions_by_severity{severity="critical"} 2.0' in output
    assert 'mission_ops_decisions_by_severity{severity="nominal"} 0.0' in output
    assert 'mission_ops_decisions_by_severity{severity="watch"} 0.0' in output
    assert 'mission_ops_decisions_by_severity{severity="warning"} 0.0' in output


def test_reports_gemma_rationale_ratio():
    entries = [
        _entry("e1", Severity.WATCH, rationale_source="gemma"),
        _entry("e2", Severity.WATCH, rationale_source="fallback"),
        _entry("e3", Severity.WATCH, rationale_source="fallback"),
        _entry("e4", Severity.WATCH, rationale_source="fallback"),
    ]
    output = _render(DecisionLogCollector(load_entries=lambda: entries))

    assert 'mission_ops_gemma_rationale_ratio 0.25' in output


def test_handles_an_empty_log_without_dividing_by_zero():
    output = _render(DecisionLogCollector(load_entries=lambda: []))

    assert 'mission_ops_decisions_total 0.0' in output
    assert 'mission_ops_gemma_rationale_ratio 0.0' in output


def test_maneuver_status_never_includes_a_none_label():
    entries = [_entry("e1", Severity.NOMINAL)]  # no maneuver at all
    output = _render(DecisionLogCollector(load_entries=lambda: entries))

    assert 'status="None"' not in output
    assert 'mission_ops_maneuver_status{status="budget_insufficient"} 0.0' in output


def test_http_request_counter_is_a_real_process_counter_not_log_derived():
    """Unlike the DecisionLogCollector gauges above (recomputed fresh
    every scrape from the real audit log), this counter increments once
    per real request and keeps its value across scrapes - a genuinely
    different metric shape for a genuinely different kind of fact."""
    before = generate_latest(registry).decode("utf-8")
    before_count = _extract_counter_value(before, 'mission_ops_http_requests_total{method="GET",route="/health",status="200"}')

    http_requests_total.labels(method="GET", route="/health", status="200").inc()

    after = generate_latest(registry).decode("utf-8")
    after_count = _extract_counter_value(after, 'mission_ops_http_requests_total{method="GET",route="/health",status="200"}')

    assert after_count == before_count + 1


def test_rate_limited_counter_increments_independently():
    before = generate_latest(registry).decode("utf-8")
    before_count = _extract_counter_value(before, "mission_ops_rate_limited_requests_total")

    rate_limited_requests_total.inc()

    after = generate_latest(registry).decode("utf-8")
    after_count = _extract_counter_value(after, "mission_ops_rate_limited_requests_total")

    assert after_count == before_count + 1


def _extract_counter_value(exposition_text: str, metric_line_prefix: str) -> float:
    for line in exposition_text.splitlines():
        if line.startswith(metric_line_prefix + " "):
            return float(line.split(" ")[-1])
    return 0.0
