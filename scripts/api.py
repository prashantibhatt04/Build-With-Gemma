#!/usr/bin/env python3
"""Real REST API - ROADMAP_TO_PRODUCT.md Phase 6.

Every other way to get data out of this project is human-facing: the
dashboard UI, or reading the audit log/database directly. A real
operator's own mission-control software needs to query findings and
resolve pending approvals programmatically - this is that surface.

Deliberately thin: every endpoint here delegates to the exact same
DecisionLogger (JSONL or Postgres, whichever DATABASE_URL selects - see
src/logging_utils.py) and the exact same src/dashboard_data.py transforms
scripts/dashboard.py already uses - nothing API-specific about storage or
aggregation logic, matching this project's "one real implementation,
multiple surfaces" discipline everywhere else (display.py's
classify_decision_status is the other clear example). Response models
are this project's own real Pydantic v2 schemas (schemas.DecisionLogEntry
and friends) served directly - no separate translation layer, since
they're already the right shape.

Auth reuses src/auth.py's OPERATOR_TOKENS scheme (Phase 5) - not a
second, weaker scheme. Read endpoints stay open if OPERATOR_TOKENS is
unconfigured (same zero-setup default as the dashboard, with a visible
`X-Warning` response header instead of a silent gap); write endpoints
(approve/reject/mark-reviewed) are stricter than the dashboard on
purpose - they refuse to run at all (503) rather than fall back to
anything resembling free-text identity, since a programmatic caller has
no "just don't worry about it for now" equivalent to a human glancing at
a warning banner.

Real rate limiting (src/rate_limit.py) is on by default - a token-bucket
limiter per authenticated operator (or source IP, unauthenticated),
configurable via API_RATE_LIMIT_PER_MINUTE, 0 disables it. Real
Prometheus metrics are served at the conventional /metrics path
(src/metrics.py) - the JSON severity/status summary that used to live
there moved to /stats/summary to make room for it.

Run:
    uvicorn scripts.api:app --reload           # dev
    uvicorn scripts.api:app --host 0.0.0.0     # production (see docker-compose.yml)

Interactive docs at /docs (FastAPI's automatic OpenAPI UI) once running.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.routing import Match

from src.auth import authenticate
from src.config import settings
from src.dashboard_data import compute_metrics, entries_to_rows, pending_approvals
from src.logging_utils import DecisionLogger
from src.metrics import DecisionLogCollector, http_requests_total, rate_limited_requests_total, registry, render_metrics
from src.rate_limit import RateLimiter
from src.schemas import DecisionLogEntry

app = FastAPI(
    title="Deep Space Navigation - Mission Ops API",
    description="Programmatic access to the real conjunction/decay/attitude audit log.",
    version="1.0.0",
)

# None when API_RATE_LIMIT_PER_MINUTE=0 - disabled entirely, e.g. for
# local dev. A real, on-by-default value otherwise (see src/config.py) -
# constructed once at import time, not per-request, so its token buckets
# actually accumulate state across real requests.
_rate_limiter = (
    RateLimiter(capacity=settings.api_rate_limit_per_minute, refill_per_second=settings.api_rate_limit_per_minute / 60.0)
    if settings.api_rate_limit_per_minute > 0
    else None
)


def _rate_limit_client_id(request: Request) -> str:
    """Prefers the authenticated operator identity over the raw source
    IP when a valid token is presented - so a shared-IP deployment (e.g.
    several operators behind one NAT/proxy) doesn't unfairly bucket
    unrelated real operators together under one limit. Falls back to
    source IP for unauthenticated requests, which is the only real
    identity available for those."""
    authorization = request.headers.get("authorization", "")
    if authorization.startswith("Bearer ") and settings.operator_tokens:
        operator = authenticate(authorization[len("Bearer ") :].strip(), settings.operator_tokens)
        if operator is not None:
            return f"operator:{operator}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if _rate_limiter is not None:
        client_id = _rate_limit_client_id(request)
        allowed, retry_after = _rate_limiter.allow(client_id)
        if not allowed:
            rate_limited_requests_total.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
            )
    return await call_next(request)


def _route_template_for(request: Request) -> str:
    """Resolves the matched route TEMPLATE (e.g. "/decisions/{event_id}"),
    never the raw resolved path - baking a real event_id into a metric's
    label value would produce unbounded real label cardinality in a real
    Prometheus deployment (a new label value, forever retained, for every
    distinct id anyone ever queried).

    Matches directly against request.app.routes rather than reading
    request.scope["route"] - that key is only set once Starlette's own
    router actually runs, which a middleware wrapping the router (like
    _rate_limit below) can short-circuit before it ever does, e.g.
    returning a 429 for a real, distinct path like
    "/decisions/some-real-id". Matching here works the same whether or
    not routing already happened, so every response gets a real bounded
    template label - falling back to the raw path only for a genuinely
    unmapped path (Starlette's own 404), which is itself a bounded,
    finite set of real routes this app defines."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
    return request.url.path


@app.middleware("http")
async def _count_requests(request: Request, call_next):
    """A real, monotonically increasing process counter - see
    src/metrics.py's module docstring for why this is a Counter and not
    part of the audit-log-derived Collector below."""
    response = await call_next(request)
    route_template = _route_template_for(request)
    http_requests_total.labels(method=request.method, route=route_template, status=str(response.status_code)).inc()
    return response


@app.middleware("http")
async def _warn_when_unauthenticated(request, call_next):
    """The visible equivalent of the dashboard's "⚠️ Unauthenticated"
    banner (see scripts/dashboard.py) - a machine client has no UI to
    show a warning in, so it goes in a response header instead. Applies
    to every response, not just reads, so a caller hitting a 503 from a
    write endpoint still sees *why* in the same place."""
    response = await call_next(request)
    if not settings.operator_tokens:
        response.headers["X-Warning"] = (
            "unauthenticated-api: OPERATOR_TOKENS is not configured - read access is open to anyone reaching this API"
        )
    return response


def get_logger() -> DecisionLogger:
    # A real FastAPI dependency (not just an inline helper) specifically
    # so tests can override it (app.dependency_overrides[get_logger] = ...)
    # to point at a controlled log_dir/store instead of mutating the
    # global `settings` singleton. Constructed per-request rather than
    # shared at module scope - unlike GemmaClient/DeltaVBudgetTracker
    # (see scripts/scheduler.py's own docstring on why THOSE must be
    # shared across calls to accumulate real state), DecisionLogger holds
    # no in-process state of its own to lose between requests - every
    # call already opens its own real connection/file handle underneath
    # (see JSONLDecisionLogStore/PostgresDecisionLogStore).
    return DecisionLogger(settings=settings)


def _load_entries_for_metrics() -> list[DecisionLogEntry]:
    """Looks up `app.dependency_overrides` explicitly, at CALL time (once
    per real scrape) - not just `get_logger()` directly. A prometheus_client
    Collector is invoked by generate_latest(), completely outside FastAPI's
    own request/Depends() machinery, so it would otherwise silently ignore
    `app.dependency_overrides[get_logger] = ...` (exactly how tests point
    every other endpoint at an isolated logger) and always read whichever
    real, ambient .env-configured log get_logger() itself constructs -
    a real bug caught by this project's own "test against the real
    override mechanism, not an assumption about it" discipline."""
    logger_factory = app.dependency_overrides.get(get_logger, get_logger)
    return logger_factory().load_all_entries()


# Registered once at import time, not per-request - a real Prometheus
# Collector recomputes its gauges fresh every time /metrics is actually
# scraped (see DecisionLogCollector.collect), so there's no request-time
# cost to registering it eagerly here.
registry.register(DecisionLogCollector(load_entries=_load_entries_for_metrics))


def _parse_bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return ""
    return authorization[len("Bearer ") :].strip()


def optional_operator(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """For read endpoints: if OPERATOR_TOKENS is configured, a valid
    token is required (401 otherwise); if unconfigured, read access
    stays open - same zero-setup default the dashboard uses."""
    if not settings.operator_tokens:
        return None
    operator = authenticate(_parse_bearer_token(authorization), settings.operator_tokens)
    if operator is None:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization: Bearer <token>")
    return operator


def require_operator(authorization: Optional[str] = Header(None)) -> str:
    """For write endpoints: OPERATOR_TOKENS must be configured at all
    (503 otherwise - no anonymous fallback for a programmatic mutation,
    stricter than the dashboard's free-text default) and the token must
    be valid (401 otherwise)."""
    if not settings.operator_tokens:
        raise HTTPException(
            status_code=503,
            detail="OPERATOR_TOKENS is not configured - write endpoints are disabled until it is.",
        )
    operator = authenticate(_parse_bearer_token(authorization), settings.operator_tokens)
    if operator is None:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization: Bearer <token>")
    return operator


def _call_storage_or_503(fn, *args, **kwargs):
    """Every real storage call in this module goes through this, so a
    transient backend blip (a Postgres restart - realistic under
    docker-compose's own `restart: unless-stopped` topology) surfaces as
    the same clean, typed 503 /health already gives, on every endpoint -
    not a bare unhandled 500 with no signal a programmatic caller can act
    on for four out of five real storage-reading routes."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"storage unreachable: {exc}") from exc


def _raise_for_logger_value_error(exc: ValueError) -> None:
    """DecisionLogger.approve_maneuver/mark_reviewed raise a plain
    ValueError for two real, distinct situations - a programmatic caller
    benefits from telling them apart, unlike the dashboard's single
    st.error() text. "No logged decision found" -> 404 (doesn't exist);
    anything else (already resolved, budget-blocked, not CRITICAL) -> 409
    (exists, but not in a state this action applies to)."""
    if "No logged decision found" in str(exc):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=409, detail=str(exc)) from exc


class MetricsResponse(BaseModel):
    total: int
    critical: int
    executed_autonomous: int
    executed_human_approved: int
    vetoed_by_gemma: int
    rejected_by_human: int
    awaiting_human_approval: int
    budget_insufficient: int
    gemma_rationale_pct: float


class ReviewRequest(BaseModel):
    reviewed_by: Optional[str] = None  # falls back to the authenticated operator if omitted


@app.get("/health")
def health(logger: DecisionLogger = Depends(get_logger)):
    """No auth required - a load balancer/orchestrator health probe
    shouldn't need a credential. Actually touches storage (not just
    "the process is up") - a real, if cheap, liveness check. Takes
    logger via Depends like every other route (rather than calling
    get_logger() directly in the body) specifically so
    app.dependency_overrides[get_logger] - the exact mechanism every
    test uses to point at an isolated store - actually applies here too;
    calling it directly silently bypassed that."""
    try:
        logger.load_all_entries()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"storage unreachable: {exc}") from exc
    backend = "postgres" if settings.database_url else "jsonl"
    return {"status": "ok", "storage_backend": backend}


@app.get("/decisions", response_model=list[DecisionLogEntry])
def list_decisions(
    severity: Optional[str] = Query(None, description="Filter by severity: nominal/watch/warning/critical"),
    source: Optional[str] = Query(None, description="Filter by telemetry source, e.g. celestrak"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    operator: Optional[str] = Depends(optional_operator),
    logger: DecisionLogger = Depends(get_logger),
):
    entries = _call_storage_or_503(logger.load_all_entries)
    if severity is not None:
        entries = [e for e in entries if e.finding.severity.value == severity]
    if source is not None:
        entries = [e for e in entries if e.telemetry.source == source]
    return entries[offset : offset + limit]


@app.get("/decisions/pending-approval", response_model=list[DecisionLogEntry])
def list_pending_approvals(
    operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    return pending_approvals(_call_storage_or_503(logger.load_all_entries))


@app.get("/decisions/export")
def export_decisions_csv(
    severity: Optional[str] = Query(None, description="Filter by severity: nominal/watch/warning/critical"),
    source: Optional[str] = Query(None, description="Filter by telemetry source, e.g. celestrak"),
    operator: Optional[str] = Depends(optional_operator),
    logger: DecisionLogger = Depends(get_logger),
) -> Response:
    """Real CSV export of the audit log - a genuine gap found by a real
    PM/customer review: the only way to get this audit trail out of the
    product was a live dashboard URL or raw JSON from the endpoints
    above - a real operator handing this to an insurer, regulator, or
    their own ops review needs a portable file, not a link. Reuses the
    exact same entries_to_rows transform the dashboard's own table uses,
    not a second format that could drift out of sync. Registered BEFORE
    /decisions/{event_id} below - Starlette matches path routes in
    registration order, so "export" would otherwise be captured as a
    literal event_id and 404. Same severity/source filters as the plain
    /decisions list above, but no limit/offset - export means "give me
    everything matching the filter," not a paginated page of it."""
    entries = _call_storage_or_503(logger.load_all_entries)
    if severity is not None:
        entries = [e for e in entries if e.finding.severity.value == severity]
    if source is not None:
        entries = [e for e in entries if e.telemetry.source == source]

    csv_bytes = pd.DataFrame(entries_to_rows(entries)).to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=decisions.csv"},
    )


@app.get("/decisions/{event_id}", response_model=DecisionLogEntry)
def get_decision(
    event_id: str, operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    entry = _call_storage_or_503(logger.find_entry, event_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No logged decision found with event_id={event_id!r}")
    return entry


@app.get("/stats/summary", response_model=MetricsResponse)
def get_stats_summary(
    operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    """The JSON severity/status summary - same real numbers the dashboard's
    metrics row shows. Was originally served at /metrics; moved here once
    /metrics became real Prometheus exposition text (the conventional path
    scrapers expect), so the two don't collide."""
    return compute_metrics(_call_storage_or_503(logger.load_all_entries))


@app.get("/metrics")
def get_prometheus_metrics():
    """Real Prometheus exposition-format text - see src/metrics.py.
    Deliberately NOT gated by optional_operator: a Prometheus scraper is
    real infrastructure, not a human operator, and gating it the same way
    as /decisions would mean either baking a token into scrape config
    (itself a real secret-handling concern) or leaving metrics
    unreachable by default - same "a health/monitoring probe shouldn't
    need a credential" reasoning /health already uses."""
    return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.post("/decisions/{event_id}/approve", response_model=DecisionLogEntry)
def approve_decision(
    event_id: str, operator: str = Depends(require_operator), logger: DecisionLogger = Depends(get_logger),
):
    try:
        return logger.approve_maneuver(event_id, approved=True, approved_by=operator)
    except ValueError as exc:
        _raise_for_logger_value_error(exc)


@app.post("/decisions/{event_id}/reject", response_model=DecisionLogEntry)
def reject_decision(
    event_id: str, operator: str = Depends(require_operator), logger: DecisionLogger = Depends(get_logger),
):
    try:
        return logger.approve_maneuver(event_id, approved=False, approved_by=operator)
    except ValueError as exc:
        _raise_for_logger_value_error(exc)


@app.post("/decisions/{event_id}/review", response_model=DecisionLogEntry)
def review_decision(
    event_id: str,
    # A default, not a bare required `body: ReviewRequest` - every field
    # on ReviewRequest is itself optional (reviewed_by falls back to the
    # authenticated operator), but without a default here FastAPI still
    # required a JSON body to be present on the wire at all, so a bare
    # `POST .../review` with just an Authorization header 422'd instead
    # of working as documented.
    body: ReviewRequest = ReviewRequest(),
    operator: str = Depends(require_operator),
    logger: DecisionLogger = Depends(get_logger),
):
    try:
        return logger.mark_reviewed(event_id, reviewed_by=body.reviewed_by or operator)
    except ValueError as exc:
        _raise_for_logger_value_error(exc)
