"""Tests for src/alerting.py. No real network calls - requests.post is
mocked throughout."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.alerting import ALERT_COOLDOWN_HOURS, build_alert_text, hazard_key, send_critical_alert, send_health_alert
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


def _conjunction_entry_at(
    made_at: datetime, event_id: str = "conj-test-1", object_a_id: str = "1", object_b_id: str = "2",
    severity: Severity = Severity.CRITICAL,
) -> DecisionLogEntry:
    """Like _conjunction_entry above, but with a caller-controlled made_at
    and event_id/object ids - needed to build realistic "same hazard,
    different scheduler tick" fixtures for the cooldown tests below (a
    real re-detected hazard keeps its object ids but gets a fresh
    event_id/made_at every tick)."""
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=made_at, source="celestrak",
        raw_data={
            "object_a_id": object_a_id, "object_a_name": "SAT-A", "object_b_id": object_b_id, "object_b_name": "SAT-B",
            "min_distance_km": 2.5,
        },
    )
    finding = AnomalyFinding(event_id=telemetry.event_id, severity=severity, description="Test.", confidence=0.9)
    decision = Decision(action="abort", rationale="Recommendation: abort.", made_at=made_at)
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def test_hazard_key_is_order_independent_for_conjunctions():
    """Real behavior this guards: the same conjunction pair can be scanned
    as "A vs B" or "B vs A" depending on catalog order - both must
    collapse to the same hazard identity, or the cooldown below would
    never actually suppress a real re-detected conjunction."""
    a_vs_b = _conjunction_entry_at(datetime.now(timezone.utc), object_a_id="33765", object_b_id="33818")
    b_vs_a = _conjunction_entry_at(datetime.now(timezone.utc), object_a_id="33818", object_b_id="33765")

    assert hazard_key(a_vs_b) == hazard_key(b_vs_a)


def test_hazard_key_differs_for_a_different_pair():
    assert hazard_key(_conjunction_entry_at(datetime.now(timezone.utc), object_a_id="1", object_b_id="2")) != hazard_key(
        _conjunction_entry_at(datetime.now(timezone.utc), object_a_id="1", object_b_id="3")
    )


def test_hazard_key_uses_object_id_for_decay_and_attitude():
    assert hazard_key(_decay_entry()) == "decay:33821"
    assert hazard_key(_attitude_entry()) == "attitude:99010"


@patch("src.alerting.requests.post")
def test_send_critical_alert_suppressed_when_same_hazard_alerted_recently(mock_post):
    """The real gap this closes: without this suppression, the same
    still-unresolved conjunction re-detected on the next scheduler tick
    would re-fire a webhook alert every tick, indefinitely - a real
    operator would eventually mute or distrust the channel."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    now = datetime.now(timezone.utc)
    earlier_alert = _conjunction_entry_at(now - timedelta(hours=1), event_id="conj-tick-1")
    new_detection = _conjunction_entry_at(now, event_id="conj-tick-2")

    result = send_critical_alert(
        new_detection, _settings(alert_webhook_url="https://example.com/webhook"),
        recent_critical_entries=[earlier_alert],
    )

    assert result is False
    mock_post.assert_not_called()


@patch("src.alerting.requests.post")
def test_send_critical_alert_fires_again_after_cooldown_expires(mock_post):
    """The suppression above must not become a permanent silence for a
    hazard that's still CRITICAL days later - a real operator still wants
    to be reminded periodically, just not every single tick."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    now = datetime.now(timezone.utc)
    stale_alert = _conjunction_entry_at(
        now - timedelta(hours=ALERT_COOLDOWN_HOURS + 1), event_id="conj-tick-1",
    )
    new_detection = _conjunction_entry_at(now, event_id="conj-tick-2")

    result = send_critical_alert(
        new_detection, _settings(alert_webhook_url="https://example.com/webhook"),
        recent_critical_entries=[stale_alert],
    )

    assert result is True
    mock_post.assert_called_once()


@patch("src.alerting.requests.post")
def test_send_critical_alert_not_suppressed_by_a_different_hazard(mock_post):
    """A recent CRITICAL alert for an unrelated pair must never suppress
    a genuinely new, different hazard."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    now = datetime.now(timezone.utc)
    unrelated_alert = _conjunction_entry_at(
        now - timedelta(minutes=5), event_id="conj-other", object_a_id="9", object_b_id="10",
    )
    new_detection = _conjunction_entry_at(now, event_id="conj-tick-2", object_a_id="1", object_b_id="2")

    result = send_critical_alert(
        new_detection, _settings(alert_webhook_url="https://example.com/webhook"),
        recent_critical_entries=[unrelated_alert],
    )

    assert result is True


@patch("src.alerting.requests.post")
def test_send_critical_alert_ignores_non_critical_entries_for_the_same_hazard(mock_post):
    """A WARNING/WATCH entry for the same object pair (e.g. before it
    escalated to CRITICAL) must not count as "already alerted" - only a
    genuine prior CRITICAL alert should suppress."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    now = datetime.now(timezone.utc)
    earlier_non_critical = _conjunction_entry_at(
        now - timedelta(hours=1), event_id="conj-tick-1", severity=Severity.WARNING,
    )
    new_detection = _conjunction_entry_at(now, event_id="conj-tick-2")

    result = send_critical_alert(
        new_detection, _settings(alert_webhook_url="https://example.com/webhook"),
        recent_critical_entries=[earlier_non_critical],
    )

    assert result is True


@patch("src.alerting.requests.post")
def test_send_critical_alert_is_not_self_suppressed(mock_post):
    """A caller that (harmlessly) includes the entry itself in
    recent_critical_entries (e.g. because it was already logged before
    the alert call) must not have that count as "already alerted" -
    hazard_key equality alone isn't enough to exclude it; the entry's own
    distinct event_id is what does."""
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    entry = _conjunction_entry_at(datetime.now(timezone.utc), event_id="conj-tick-1")

    result = send_critical_alert(
        entry, _settings(alert_webhook_url="https://example.com/webhook"),
        recent_critical_entries=[entry],
    )

    assert result is True


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
def test_send_critical_alert_does_not_leak_the_webhook_secret_on_failure(mock_post, capsys):
    """Real bug this closes: Slack/Discord/Teams webhook URLs embed their
    auth token directly in the path, and requests' own exception messages
    routinely include the request URL - sometimes the full URL, sometimes
    (as reproduced here) just the path, e.g. "Max retries exceeded with
    url: /services/.../SECRETTOKEN". Printing that unredacted on a failed
    send - which this module deliberately never treats as fatal, so it
    WILL print - would leak a live secret straight into process/container
    logs."""
    webhook_url = "https://hooks.slack.com/services/T000/B111/SECRETTOKEN123"
    mock_post.side_effect = requests.exceptions.ConnectionError(
        "HTTPSConnectionPool(host='hooks.slack.com', port=443): Max retries exceeded with "
        "url: /services/T000/B111/SECRETTOKEN123 (Caused by NewConnectionError(...))"
    )
    entry = _conjunction_entry(Severity.CRITICAL)

    result = send_critical_alert(entry, _settings(alert_webhook_url=webhook_url))

    assert result is False
    printed = capsys.readouterr().out
    assert "SECRETTOKEN123" not in printed
    assert "redacted" in printed


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


@patch("src.alerting.requests.post")
def test_send_health_alert_does_not_leak_the_webhook_secret_on_failure(mock_post, capsys):
    """Same real bug as the CRITICAL-alert version above, for the
    separate health-alert send path - both call the same
    requests.post/print pattern, so both needed the fix independently."""
    webhook_url = "https://hooks.slack.com/services/T000/B111/SECRETTOKEN123"
    mock_post.side_effect = requests.exceptions.ConnectionError(
        "Max retries exceeded with url: /services/T000/B111/SECRETTOKEN123"
    )

    result = send_health_alert("scheduler down", _settings(alert_webhook_url=webhook_url))

    assert result is False
    printed = capsys.readouterr().out
    assert "SECRETTOKEN123" not in printed
    assert "redacted" in printed
