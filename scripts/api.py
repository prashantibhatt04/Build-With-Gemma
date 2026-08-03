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

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from src.auth import authenticate
from src.config import settings
from src.dashboard_data import compute_metrics, entries_to_rows, pending_approvals
from src.logging_utils import DecisionLogger
from src.schemas import DecisionLogEntry

app = FastAPI(
    title="Deep Space Navigation - Mission Ops API",
    description="Programmatic access to the real conjunction/decay/attitude audit log.",
    version="1.0.0",
)


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
def health():
    """No auth required - a load balancer/orchestrator health probe
    shouldn't need a credential. Actually touches storage (not just
    "the process is up") - a real, if cheap, liveness check."""
    logger = get_logger()
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
    entries = logger.load_all_entries()
    if severity is not None:
        entries = [e for e in entries if e.finding.severity.value == severity]
    if source is not None:
        entries = [e for e in entries if e.telemetry.source == source]
    return entries[offset : offset + limit]


@app.get("/decisions/pending-approval", response_model=list[DecisionLogEntry])
def list_pending_approvals(
    operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    return pending_approvals(logger.load_all_entries())


@app.get("/decisions/{event_id}", response_model=DecisionLogEntry)
def get_decision(
    event_id: str, operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    entry = logger.find_entry(event_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No logged decision found with event_id={event_id!r}")
    return entry


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    operator: Optional[str] = Depends(optional_operator), logger: DecisionLogger = Depends(get_logger),
):
    return compute_metrics(logger.load_all_entries())


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
    body: ReviewRequest,
    operator: str = Depends(require_operator),
    logger: DecisionLogger = Depends(get_logger),
):
    try:
        return logger.mark_reviewed(event_id, reviewed_by=body.reviewed_by or operator)
    except ValueError as exc:
        _raise_for_logger_value_error(exc)
