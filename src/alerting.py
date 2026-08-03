"""Real-time webhook alerting for CRITICAL findings.

Right now, a CRITICAL event just sits in the dashboard/audit log until
someone happens to look - this closes that gap with a real HTTP POST to
a configurable webhook the moment a CRITICAL decision is logged. Slack
Incoming Webhook compatible (`{"text": ...}`) - also works with Discord,
Microsoft Teams (via a compatible payload), or any custom HTTP receiver
that accepts a JSON body with a "text" field.

Deterministic firing condition (finding.severity == Severity.CRITICAL) -
not Gemma's call, matching this project's "Gemma narrates, never decides"
design everywhere else. The alert body reuses the already-generated real
Gemma rationale text rather than making a new Gemma call - no extra
latency, no extra API cost, and it's the same explanation a human would
already see in the dashboard.

Disabled by default: alert_webhook_url defaults to "" (see
Settings.alert_webhook_url), which is a no-op, not an error - this
project's existing test suite and demo runs are unaffected unless
someone explicitly configures a webhook. A failed POST (bad URL,
receiver down, network error) is caught and reported, never raised - an
alerting outage must never block or crash the actual triage pipeline,
same fail-safe philosophy as every Gemma call in this project.
"""
from __future__ import annotations

from urllib.parse import urlparse

import requests

from .config import Settings, settings as default_settings
from .schemas import DecisionLogEntry, Severity

WEBHOOK_TIMEOUT_S = 10


def _redact_webhook_url(text: str, webhook_url: str) -> str:
    """Scrubs the real webhook URL out of an error message before it's
    ever printed. Slack/Discord/Teams incoming-webhook URLs embed their
    auth token directly in the URL - and `requests`' own exception
    messages (ConnectionError, Timeout, ...) routinely include the
    request URL verbatim, sometimes as the full URL and sometimes as just
    its path (e.g. "Max retries exceeded with url: /services/T000/TOKEN"),
    so both forms are redacted here. Without this, a webhook outage -
    deliberately never fatal, by this module's own fail-safe design -
    would otherwise print a live secret straight into process/container
    logs, undermining the "never baked into the image" secret-handling
    intent docker-compose.yml documents for this exact setting."""
    redacted = text.replace(webhook_url, "<redacted-webhook-url>")
    path = urlparse(webhook_url).path
    if path and path != "/":
        redacted = redacted.replace(path, "<redacted-webhook-path>")
    return redacted


def _subject_for_entry(entry: DecisionLogEntry) -> str:
    """A one-line real subject for the alert, covering all three hazard
    shapes - deliberately its own minimal formatter rather than a fourth
    reuse of display.py's/dashboard_data.py's/rag.py's own versions of
    this, since each already serves a different output (terminal width,
    table column, embedding text) and none is a clean fit for a short
    alert line."""
    raw = entry.telemetry.raw_data
    if "object_a_name" in raw:
        return f"{raw['object_a_name']} vs {raw['object_b_name']} ({raw['min_distance_km']:.2f}km)"
    if "perigee_altitude_km" in raw:
        return f"{raw['object_name']} (perigee {raw['perigee_altitude_km']:.0f}km)"
    if "pointing_error_deg" in raw:
        return f"{raw['object_name']} (pointing error {raw['pointing_error_deg']:.0f}°)"
    return entry.telemetry.event_id


def build_alert_text(entry: DecisionLogEntry) -> str:
    """The real alert body - subject, deterministic action, and the
    already-generated real Gemma rationale (or deterministic fallback
    text if Gemma was unreachable) - not a re-summarization."""
    return (
        f"\U0001f6a8 CRITICAL: {_subject_for_entry(entry)}\n"
        f"event_id: {entry.telemetry.event_id}\n"
        f"Action: {entry.decision.action.value}\n"
        f"{entry.decision.rationale}"
    )


def send_critical_alert(entry: DecisionLogEntry, settings: Settings = default_settings) -> bool:
    """Sends a real webhook POST if entry.finding.severity is CRITICAL and
    a webhook URL is configured. Returns True only if an alert was
    actually sent successfully - False for "nothing to send" (not
    CRITICAL, or no URL configured) and for a real send failure alike, so
    callers that don't need to distinguish the two can just check
    truthiness; see the return value's docstring-adjacent log line for
    which case actually happened."""
    if entry.finding.severity != Severity.CRITICAL:
        return False
    if not settings.alert_webhook_url:
        return False

    try:
        response = requests.post(
            settings.alert_webhook_url,
            json={"text": build_alert_text(entry)},
            timeout=WEBHOOK_TIMEOUT_S,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        detail = _redact_webhook_url(str(exc), settings.alert_webhook_url)
        print(f"WARNING: CRITICAL alert webhook failed for {entry.telemetry.event_id}: {detail}")
        return False


def send_health_alert(message: str, settings: Settings = default_settings) -> bool:
    """A real webhook POST about the SYSTEM's own health - e.g.
    scripts/scheduler.py firing this after several consecutive tick
    failures (ROADMAP_TO_PRODUCT.md Phase 5). Distinct from
    send_critical_alert on purpose: a CRITICAL conjunction alert says "a
    real risk was found"; this says "the thing that's supposed to be
    finding risks might not be working" - an operator who only ever
    wired up the first kind would have no way to learn the second thing
    ever happened. Same webhook URL/mechanism (a small real deployment
    plausibly wants both kinds of alert in the same channel), same
    fail-safe behavior (never raises - a health-alerting failure must not
    also take down the process it's trying to report on), but a visibly
    different message prefix so the two are never confused for each
    other in a shared channel."""
    if not settings.alert_webhook_url:
        return False

    try:
        response = requests.post(
            settings.alert_webhook_url,
            json={"text": f"⚠️ SYSTEM HEALTH: {message}"},
            timeout=WEBHOOK_TIMEOUT_S,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        detail = _redact_webhook_url(str(exc), settings.alert_webhook_url)
        print(f"WARNING: system health alert webhook failed: {detail}")
        return False
