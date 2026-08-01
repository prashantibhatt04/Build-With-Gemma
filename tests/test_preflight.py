"""Tests for src/preflight.py. Gemma connectivity is mocked - no real
network calls."""
from unittest.mock import patch

from src.config import Settings
from src.gemma_client import GemmaClientError
from src.preflight import check_config, check_gemma_reachable, check_log_dir_writable, run_all_checks


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
