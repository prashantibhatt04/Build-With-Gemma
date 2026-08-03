"""Pre-demo environment health checks: config sanity, Gemma connectivity,
filesystem writability. Pure, structured results so both scripts/run_demo.py
and tests can use the same logic without duplicating it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings, settings as default_settings
from .gemma_client import GemmaClient, GemmaClientError


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_config(settings: Settings) -> CheckResult:
    if settings.gemma_backend not in ("ollama", "api"):
        return CheckResult(
            "Config: GEMMA_BACKEND valid", False,
            f"unexpected value {settings.gemma_backend!r}, expected 'ollama' or 'api'",
        )
    if settings.gemma_backend == "api" and not settings.gemma_api_key:
        return CheckResult(
            "Config: GEMMA_BACKEND valid", False,
            "GEMMA_BACKEND=api but GEMMA_API_KEY is empty",
        )
    model = settings.gemma_model if settings.gemma_backend == "ollama" else settings.gemma_model_api
    return CheckResult(
        "Config: GEMMA_BACKEND valid", True,
        f"backend={settings.gemma_backend}, model={model}",
    )


def check_log_dir_writable(settings: Settings) -> CheckResult:
    try:
        path = Path(settings.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink()
        return CheckResult("Log directory writable", True, str(path))
    except OSError as exc:
        return CheckResult("Log directory writable", False, str(exc))


def check_postgres_reachable(settings: Settings) -> CheckResult:
    """The real check that actually matters once DATABASE_URL is
    configured - a cheap real connection attempt (no query), same
    connect-and-close pattern test_postgres_logging.py's own
    reachability probe uses. psycopg is imported lazily here, same as
    logging_utils._default_store, so the common JSONL-only path never
    requires it installed at all."""
    try:
        import psycopg
    except ImportError as exc:
        return CheckResult(
            "Postgres reachable", False,
            f"DATABASE_URL is configured but psycopg isn't installed: {exc}",
        )
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5):
            pass
    except psycopg.OperationalError as exc:
        return CheckResult("Postgres reachable", False, str(exc))
    return CheckResult("Postgres reachable", True, "connected successfully")


def check_storage_writable(settings: Settings) -> CheckResult:
    """Checks whichever backend DecisionLogger will actually use for
    this configuration (see logging_utils._default_store) - real
    Postgres reachability when DATABASE_URL is set, the JSONL log_dir
    otherwise. A real bug this closes: run_all_checks used to always run
    check_log_dir_writable, even for a Postgres-only deployment where
    nothing ever writes to log_dir - a false preflight failure for a
    directory that doesn't matter, while never checking the one thing
    that DOES matter for that deployment shape (whether the configured
    database is even reachable)."""
    if settings.database_url:
        return check_postgres_reachable(settings)
    return check_log_dir_writable(settings)


def check_gemma_reachable(settings: Settings) -> CheckResult:
    client = GemmaClient(settings=settings)
    try:
        # The hosted API's latency varies a lot in practice (observed
        # 2-35s+ for the same trivial prompt) - too short a timeout here
        # reports a perfectly reachable backend as "unreachable" just
        # because it happened to be slow on this one call.
        client.generate(prompt="Reply with the single word: ok", timeout=45)
    except GemmaClientError as exc:
        return CheckResult("Gemma reachable", False, str(exc))

    backend = client.last_backend_used
    note = "" if backend == settings.gemma_backend else f" (fell back from {settings.gemma_backend})"
    return CheckResult("Gemma reachable", True, f"responded via {backend}{note}")


def run_all_checks(settings: Settings = default_settings) -> list[CheckResult]:
    """Config and storage checks first (fast, and checks whichever
    backend is actually configured - see check_storage_writable) - Gemma
    reachability last, since it's the one real network call."""
    return [
        check_config(settings),
        check_storage_writable(settings),
        check_gemma_reachable(settings),
    ]
