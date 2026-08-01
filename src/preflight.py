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


def check_gemma_reachable(settings: Settings) -> CheckResult:
    client = GemmaClient(settings=settings)
    try:
        client.generate(prompt="Reply with the single word: ok", timeout=15)
    except GemmaClientError as exc:
        return CheckResult("Gemma reachable", False, str(exc))

    backend = client.last_backend_used
    note = "" if backend == settings.gemma_backend else f" (fell back from {settings.gemma_backend})"
    return CheckResult("Gemma reachable", True, f"responded via {backend}{note}")


def run_all_checks(settings: Settings = default_settings) -> list[CheckResult]:
    """Config and filesystem checks first (fast, no network) - Gemma
    reachability last, since it's the one real network call."""
    return [
        check_config(settings),
        check_log_dir_writable(settings),
        check_gemma_reachable(settings),
    ]
