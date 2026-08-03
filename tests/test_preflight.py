"""Tests for src/preflight.py. Gemma connectivity is mocked - no real
network calls. Postgres reachability IS exercised for real (see the
tests below using TEST_DATABASE_URL) when a local Postgres is reachable,
matching test_postgres_logging.py's own real-not-mocked discipline for
storage correctness - skipped cleanly, not failed, otherwise."""
import os
from unittest.mock import patch

import pytest

from src.config import Settings
from src.gemma_client import GemmaClientError
from src.preflight import (
    check_config,
    check_gemma_reachable,
    check_log_dir_writable,
    check_postgres_reachable,
    check_storage_writable,
    run_all_checks,
)

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://localhost/build_with_gemma")


def _postgres_reachable() -> bool:
    psycopg = pytest.importorskip("psycopg")
    try:
        with psycopg.connect(TEST_DATABASE_URL, connect_timeout=2):
            return True
    except psycopg.OperationalError:
        return False


def _settings(**overrides) -> Settings:
    defaults = dict(
        gemma_backend="ollama",
        gemma_model="gemma4:e4b",
        ollama_host="http://localhost:11434",
        gemma_api_key="",
        gemma_model_api="gemma-4-26b-a4b-it",
        log_dir="./logs",
        delta_v_budget_m_s=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_check_config_ok_for_ollama_backend():
    result = check_config(_settings(gemma_backend="ollama"))
    assert result.ok is True


def test_check_config_ok_for_api_backend_with_key():
    result = check_config(_settings(gemma_backend="api", gemma_api_key="some-key"))
    assert result.ok is True


def test_check_config_fails_for_api_backend_without_key():
    result = check_config(_settings(gemma_backend="api", gemma_api_key=""))
    assert result.ok is False
    assert "GEMMA_API_KEY" in result.detail


def test_check_config_fails_for_unknown_backend():
    result = check_config(_settings(gemma_backend="carrier-pigeon"))
    assert result.ok is False


def test_check_log_dir_writable_succeeds_and_cleans_up_probe_file(tmp_path):
    log_dir = tmp_path / "logs"
    result = check_log_dir_writable(_settings(log_dir=str(log_dir)))

    assert result.ok is True
    assert log_dir.exists()
    assert not (log_dir / ".preflight_write_test").exists()


def test_check_gemma_reachable_success():
    with patch("src.preflight.GemmaClient") as MockClient:
        instance = MockClient.return_value
        instance.generate.return_value = "ok"
        instance.last_backend_used = "ollama"

        result = check_gemma_reachable(_settings(gemma_backend="ollama"))

    assert result.ok is True
    assert "ollama" in result.detail


def test_check_gemma_reachable_notes_fallback_when_backend_differs():
    with patch("src.preflight.GemmaClient") as MockClient:
        instance = MockClient.return_value
        instance.generate.return_value = "ok"
        instance.last_backend_used = "api"  # configured as ollama, actually answered via api

        result = check_gemma_reachable(_settings(gemma_backend="ollama"))

    assert result.ok is True
    assert "fell back from ollama" in result.detail


def test_check_gemma_reachable_failure():
    with patch("src.preflight.GemmaClient") as MockClient:
        instance = MockClient.return_value
        instance.generate.side_effect = GemmaClientError("both backends down")

        result = check_gemma_reachable(_settings())

    assert result.ok is False
    assert "both backends down" in result.detail


def test_check_storage_writable_checks_log_dir_when_database_url_unset(tmp_path):
    log_dir = tmp_path / "logs"
    result = check_storage_writable(_settings(log_dir=str(log_dir)))

    assert result.name == "Log directory writable"
    assert result.ok is True


def test_check_storage_writable_checks_postgres_when_database_url_set():
    """Real bug this closes: the old code always checked log_dir, even
    when nothing writes there for a Postgres-backed deployment - and
    never checked the one thing that actually matters for that shape.
    Confirms the dispatch itself (not a real connection - see the
    Postgres-specific tests below for that)."""
    with patch("src.preflight.check_postgres_reachable") as mock_check:
        mock_check.return_value = "sentinel"
        result = check_storage_writable(_settings(database_url="postgresql://example/db"))

    assert result == "sentinel"
    mock_check.assert_called_once()


@pytest.mark.skipif(not _postgres_reachable(), reason=f"no reachable Postgres at {TEST_DATABASE_URL}")
def test_check_postgres_reachable_succeeds_against_a_real_local_postgres():
    result = check_postgres_reachable(_settings(database_url=TEST_DATABASE_URL))

    assert result.ok is True
    assert result.name == "Postgres reachable"


def test_check_postgres_reachable_fails_cleanly_for_an_unreachable_database():
    # Port 1 is never a real Postgres server - a real, fast connection
    # failure, not a mock, to confirm this actually surfaces OperationalError
    # as a clean CheckResult rather than an uncaught exception.
    result = check_postgres_reachable(_settings(database_url="postgresql://localhost:1/nope"))

    assert result.ok is False
    assert result.name == "Postgres reachable"


def test_run_all_checks_returns_three_results():
    with patch("src.preflight.GemmaClient") as MockClient:
        instance = MockClient.return_value
        instance.generate.return_value = "ok"
        instance.last_backend_used = "ollama"

        results = run_all_checks(_settings())

    assert len(results) == 3
    assert [r.name for r in results] == [
        "Config: GEMMA_BACKEND valid", "Log directory writable", "Gemma reachable",
    ]
