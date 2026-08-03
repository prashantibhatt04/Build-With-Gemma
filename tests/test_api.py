"""Tests for scripts/api.py using FastAPI's real TestClient (no live
network - requests go straight into the ASGI app in-process). The
DecisionLogger dependency is overridden to point at a real, isolated
tmp_path-backed JSONL store per test (not the real .env-configured one) -
everything else (routing, auth, response models, error mapping) is
exercised for real, unmocked.
"""
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import scripts.api as api_module
from scripts.api import app, get_logger, settings as api_settings
from src.config import Settings
from src.logging_utils import DecisionLogger
from src.maneuver import compute_avoidance_maneuver
from src.rate_limit import RateLimiter
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@contextmanager
def _operator_tokens(tokens: dict[str, str]):
    """`api_settings` is the same frozen-dataclass singleton every module
    that did `from src.config import settings` shares - object.__setattr__
    bypasses the frozen check to mutate it in place for the duration of a
    test, restored afterward (same pattern as test_dashboard_app.py's
    identical need)."""
    original = api_settings.operator_tokens
    object.__setattr__(api_settings, "operator_tokens", tokens)
    try:
        yield
    finally:
        object.__setattr__(api_settings, "operator_tokens", original)


def _make_entry(event_id: str, severity: Severity = Severity.NOMINAL, source: str = "celestrak") -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source=source,
        raw_data={"object_a_name": "SAT-A", "object_b_name": "SAT-B", "min_distance_km": 50.0},
    )
    finding = AnomalyFinding(event_id=event_id, severity=severity, description="Test.", confidence=0.8)
    decision = Decision(action="continue", rationale="Test rationale.", made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


def _make_pending_entry(event_id: str) -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "A", "object_b_id": "2", "object_b_name": "B",
            "min_distance_km": 3.0, "time_of_closest_approach": "2026-08-03T00:00:00+00:00",
            "relative_velocity_km_s": 6.0,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=Severity.CRITICAL, description="Test.", confidence=0.8)
    plan = compute_avoidance_maneuver(object_a="1", object_b="2", min_distance_km=3.0, relative_velocity_km_s=6.0)
    decision = Decision(
        action="abort", rationale="Awaiting approval.", made_at=datetime.now(timezone.utc),
        maneuver_plan=plan, awaiting_human_approval=True,
    )
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


@pytest.fixture
def client(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    app.dependency_overrides[get_logger] = lambda: logger

    with _operator_tokens({}):
        yield TestClient(app), logger

    app.dependency_overrides.clear()


def test_health_reports_ok_and_touches_real_storage(client):
    test_client, _logger = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["storage_backend"] == "jsonl"


def test_unauthenticated_reads_get_a_visible_warning_header(client):
    test_client, _logger = client
    response = test_client.get("/decisions")
    assert response.status_code == 200
    assert "unauthenticated-api" in response.headers["X-Warning"]


def test_list_decisions_returns_real_logged_entries(client):
    test_client, logger = client
    logger.log(_make_entry("e1", Severity.WATCH))
    logger.log(_make_entry("e2", Severity.CRITICAL))

    response = test_client.get("/decisions")

    assert response.status_code == 200
    ids = {e["telemetry"]["event_id"] for e in response.json()}
    assert ids == {"e1", "e2"}


def test_list_decisions_filters_by_severity_and_source(client):
    test_client, logger = client
    logger.log(_make_entry("e1", Severity.WATCH, source="celestrak"))
    logger.log(_make_entry("e2", Severity.CRITICAL, source="celestrak-decay"))

    response = test_client.get("/decisions", params={"severity": "critical"})
    assert [e["telemetry"]["event_id"] for e in response.json()] == ["e2"]

    response = test_client.get("/decisions", params={"source": "celestrak-decay"})
    assert [e["telemetry"]["event_id"] for e in response.json()] == ["e2"]


def test_list_decisions_respects_limit_and_offset(client):
    test_client, logger = client
    for i in range(5):
        logger.log(_make_entry(f"e{i}"))

    response = test_client.get("/decisions", params={"limit": 2, "offset": 1})
    assert [e["telemetry"]["event_id"] for e in response.json()] == ["e1", "e2"]


def test_get_decision_returns_404_for_unknown_event_id(client):
    test_client, _logger = client
    response = test_client.get("/decisions/does-not-exist")
    assert response.status_code == 404


def test_get_decision_returns_the_real_entry(client):
    test_client, logger = client
    logger.log(_make_entry("e1"))

    response = test_client.get("/decisions/e1")

    assert response.status_code == 200
    assert response.json()["telemetry"]["event_id"] == "e1"


def test_pending_approval_endpoint_only_returns_awaiting_entries(client):
    test_client, logger = client
    logger.log(_make_entry("e1"))
    logger.log(_make_pending_entry("e2"))

    response = test_client.get("/decisions/pending-approval")

    assert [e["telemetry"]["event_id"] for e in response.json()] == ["e2"]


def test_stats_summary_endpoint_reflects_real_logged_entries(client):
    test_client, logger = client
    logger.log(_make_entry("e1", Severity.CRITICAL))
    logger.log(_make_entry("e2", Severity.NOMINAL))

    response = test_client.get("/stats/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2


def test_metrics_endpoint_serves_real_prometheus_exposition_text(client):
    test_client, logger = client
    logger.log(_make_entry("e1", Severity.CRITICAL))

    response = test_client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert 'mission_ops_decisions_total 1.0' in body
    assert 'mission_ops_decisions_by_severity{severity="critical"} 1.0' in body


def test_metrics_endpoint_requires_no_authentication(client):
    """A Prometheus scraper is infrastructure, not a human operator -
    same "no credential for a health/monitoring probe" reasoning /health
    already uses."""
    test_client, _logger = client
    with _operator_tokens({"alice": "secret123"}):
        response = test_client.get("/metrics")  # deliberately no Authorization header

    assert response.status_code == 200


def test_metrics_endpoint_labels_by_route_template_not_raw_resolved_path(client):
    """Real cardinality-safety check: hitting /decisions/{event_id} for
    two different real event_ids must produce ONE route label
    ("/decisions/{event_id}"), not two, or a real Prometheus deployment
    would accumulate one label series per distinct id ever queried."""
    test_client, logger = client
    logger.log(_make_entry("e1"))
    logger.log(_make_entry("e2"))

    test_client.get("/decisions/e1")
    test_client.get("/decisions/e2")

    body = test_client.get("/metrics").text
    assert 'route="/decisions/{event_id}"' in body
    assert 'route="/decisions/e1"' not in body
    assert 'route="/decisions/e2"' not in body


def test_write_endpoints_return_503_when_operator_tokens_unconfigured(client):
    test_client, logger = client
    logger.log(_make_pending_entry("e1"))

    response = test_client.post("/decisions/e1/approve")

    assert response.status_code == 503


def test_approve_requires_a_valid_token_when_configured(client):
    test_client, logger = client
    logger.log(_make_pending_entry("e1"))

    with _operator_tokens({"alice": "secret123"}):
        no_auth = test_client.post("/decisions/e1/approve")
        assert no_auth.status_code == 401

        wrong_token = test_client.post("/decisions/e1/approve", headers={"Authorization": "Bearer wrong"})
        assert wrong_token.status_code == 401

        good = test_client.post("/decisions/e1/approve", headers={"Authorization": "Bearer secret123"})

    assert good.status_code == 200
    body = good.json()
    assert body["decision"]["maneuver_approval"]["approved"] is True
    assert body["decision"]["maneuver_approval"]["approved_by"] == "alice"


def test_reject_records_the_authenticated_operator_and_leaves_nothing_executed(client):
    test_client, logger = client
    logger.log(_make_pending_entry("e1"))

    with _operator_tokens({"bob": "tok2"}):
        response = test_client.post("/decisions/e1/reject", headers={"Authorization": "Bearer tok2"})

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["maneuver_approval"]["approved"] is False
    assert body["decision"]["maneuver_approval"]["approved_by"] == "bob"
    assert body["decision"]["verified_clearance"] is None


def test_approve_returns_409_for_an_entry_not_awaiting_approval(client):
    test_client, logger = client
    logger.log(_make_entry("e1", Severity.NOMINAL))  # never awaiting approval

    with _operator_tokens({"alice": "secret123"}):
        response = test_client.post("/decisions/e1/approve", headers={"Authorization": "Bearer secret123"})

    assert response.status_code == 409


def test_approve_returns_404_for_an_unknown_event_id(client):
    test_client, _logger = client

    with _operator_tokens({"alice": "secret123"}):
        response = test_client.post("/decisions/does-not-exist/approve", headers={"Authorization": "Bearer secret123"})

    assert response.status_code == 404


def test_review_defaults_reviewed_by_to_the_authenticated_operator(client):
    test_client, logger = client
    logger.log(_make_entry("e1"))

    with _operator_tokens({"alice": "secret123"}):
        response = test_client.post(
            "/decisions/e1/review", json={}, headers={"Authorization": "Bearer secret123"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["human_reviewed"] is True
    assert body["reviewed_by"] == "alice"


def test_review_accepts_an_explicit_reviewed_by_override(client):
    test_client, logger = client
    logger.log(_make_entry("e1"))

    with _operator_tokens({"alice": "secret123"}):
        response = test_client.post(
            "/decisions/e1/review", json={"reviewed_by": "carol"}, headers={"Authorization": "Bearer secret123"},
        )

    assert response.json()["reviewed_by"] == "carol"


@contextmanager
def _rate_limiter(limiter):
    """Swaps the module-level rate limiter singleton for the duration of
    a test - it's constructed once at import time (so its real token
    buckets accumulate state across real requests, not reset per-request),
    which means tests need to override the same shared instance rather
    than pass one in per-call."""
    original = api_module._rate_limiter
    api_module._rate_limiter = limiter
    try:
        yield
    finally:
        api_module._rate_limiter = original


def test_rate_limit_returns_429_with_retry_after_once_exhausted(client):
    test_client, _logger = client
    with _rate_limiter(RateLimiter(capacity=2, refill_per_second=0.001)):
        first = test_client.get("/health")
        second = test_client.get("/health")
        third = test_client.get("/health")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "Retry-After" in third.headers
    assert int(third.headers["Retry-After"]) >= 1


def test_rate_limit_disabled_when_limiter_is_none(client):
    test_client, _logger = client
    with _rate_limiter(None):
        responses = [test_client.get("/health") for _ in range(10)]

    assert all(r.status_code == 200 for r in responses)


def test_rate_limit_buckets_by_authenticated_operator_not_shared_source(client):
    """Two different real operators hitting the API from what TestClient
    reports as the same source (its fixed test host) must not share one
    budget - each valid token gets its own bucket."""
    test_client, _logger = client
    with _operator_tokens({"alice": "tok-a", "bob": "tok-b"}):
        with _rate_limiter(RateLimiter(capacity=1, refill_per_second=0.001)):
            alice_first = test_client.get("/decisions", headers={"Authorization": "Bearer tok-a"})
            alice_second = test_client.get("/decisions", headers={"Authorization": "Bearer tok-a"})
            bob_first = test_client.get("/decisions", headers={"Authorization": "Bearer tok-b"})

    assert alice_first.status_code == 200
    assert alice_second.status_code == 429
    assert bob_first.status_code == 200  # bob's own, independent bucket
