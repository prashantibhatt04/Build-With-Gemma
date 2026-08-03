"""Tests for src/alerting.py. No real network calls - requests.post is
mocked throughout."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.alerting import build_alert_text, send_critical_alert, send_health_alert
from src.config import Settings
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _settings(**overrides) -> Settings:
    defaults = dict(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir="./logs",
        delta_v_budget_m_s=5.0, alert_webhook_url="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _conjunction_entry(severity: Severity) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id="conj-test-1", timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "SAT-A", "object_b_id": "2", "object_b_name": "SAT-B",
            "min_distance_km": 2.5,
        },
    )
    finding = AnomalyFinding(event_id=telemetry.event_id, severity=severity, description="Test.", confidence=0.9)
    decision = Decision(action="abort", rationale="Recommendation: abort.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def _decay_entry() -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id="decay-test-1", timestamp=datetime.now(timezone.utc), source="celestrak-decay",
        raw_data={
            "object_id": "33821", "object_name": "COSMOS 2251 DEB",
            "perigee_altitude_km": 150.0, "apogee_altitude_km": 200.0, "bstar": 0.001,
        },
    )
    finding = AnomalyFinding(event_id=telemetry.event_id, severity=Severity.CRITICAL, description="Test.", confidence=0.9)
    decision = Decision(action="abort", rationale="Recommendation: abort.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def _attitude_entry() -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id="attitude-test-1", timestamp=datetime.now(timezone.utc), source="synthetic-attitude-fixture",
        raw_data={
            "object_id": "99010", "object_name": "SYNTH-SAT-CRITICAL",
            "pointing_error_deg": 70.0, "angular_rate_deg_s": 4.5, "solar_panel_power_pct": 22.0,
        },
    )
    finding = AnomalyFinding(event_id=telemetry.event_id, severity=Severity.CRITICAL, description="Test.", confidence=0.8)
    decision = Decision(action="abort", rationale="Recommendation: abort.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_build_alert_text_covers_conjunction_shape():
    text = build_alert_text(_conjunction_entry(Severity.CRITICAL))
    assert "SAT-A vs SAT-B" in text
    assert "2.50km" in text
    assert "conj-test-1" in text
    assert "abort" in text
    assert "Recommendation: abort." in text


def test_build_alert_text_covers_decay_shape():
    text = build_alert_text(_decay_entry())
    assert "COSMOS 2251 DEB" in text
    assert "perigee 150km" in text


def test_build_alert_text_covers_attitude_shape():
    text = build_alert_text(_attitude_entry())
    assert "SYNTH-SAT-CRITICAL" in text
    assert "pointing error 70" in text


@patch("src.alerting.requests.post")
def test_send_critical_alert_skips_non_critical_severity(mock_post):
    entry = _conjunction_entry(Severity.WATCH)
    result = send_critical_alert(entry, _settings(alert_webhook_url="https://example.com/webhook"))

    assert result is False
    mock_post.assert_not_called()


@patch("src.alerting.requests.post")
def test_send_critical_alert_skips_when_no_webhook_configured(mock_post):
    entry = _conjunction_entry(Severity.CRITICAL)
    result = send_critical_alert(entry, _settings(alert_webhook_url=""))

    assert result is False
    mock_post.assert_not_called()


@patch("src.alerting.requests.post")
def test_send_critical_alert_posts_real_payload_for_critical(mock_post):
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    entry = _conjunction_entry(Severity.CRITICAL)

    result = send_critical_alert(entry, _settings(alert_webhook_url="https://example.com/webhook"))

    assert result is True
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "https://example.com/webhook"
    payload = mock_post.call_args.kwargs["json"]
    assert "text" in payload
    assert "SAT-A vs SAT-B" in payload["text"]


@patch("src.alerting.requests.post")
def test_send_critical_alert_returns_false_and_does_not_raise_on_network_failure(mock_post):
    mock_post.side_effect = requests.RequestException("connection refused")
    entry = _conjunction_entry(Severity.CRITICAL)

    result = send_critical_alert(entry, _settings(alert_webhook_url="https://example.com/webhook"))

    assert result is False  # did not raise


@patch("src.alerting.requests.post")
def test_send_health_alert_skips_when_no_webhook_configured(mock_post):
    result = send_health_alert("scheduler down", _settings(alert_webhook_url=""))

    assert result is False
    mock_post.assert_not_called()


@patch("src.alerting.requests.post")
def test_send_health_alert_posts_a_distinct_prefix_from_critical_alerts(mock_post):
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())

    result = send_health_alert(
        "scheduler has failed 3 consecutive ticks", _settings(alert_webhook_url="https://example.com/webhook"),
    )

    assert result is True
    payload = mock_post.call_args.kwargs["json"]
    assert "SYSTEM HEALTH" in payload["text"]
    assert "scheduler has failed 3 consecutive ticks" in payload["text"]
    assert "CRITICAL" not in payload["text"]  # never confusable with a real conjunction alert


@patch("src.alerting.requests.post")
def test_send_health_alert_returns_false_and_does_not_raise_on_network_failure(mock_post):
    mock_post.side_effect = requests.RequestException("connection refused")

    result = send_health_alert("scheduler down", _settings(alert_webhook_url="https://example.com/webhook"))

    assert result is False  # did not raise
