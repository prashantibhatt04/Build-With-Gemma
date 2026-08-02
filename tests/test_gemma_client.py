"""Unit tests for GemmaClient's retry and cross-backend-fallback behavior.
No real network calls - the backend methods themselves are mocked."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.config import Settings
from src.gemma_client import GemmaClient, GemmaClientError
from src.pipeline import make_decide_node
from src.schemas import AnomalyFinding, Severity, TelemetryEvent


def _settings(**overrides) -> Settings:
    defaults = dict(
        gemma_backend="ollama",
        gemma_model="gemma4:e4b",
        ollama_host="http://localhost:11434",
        gemma_api_key="test-api-key",
        gemma_model_api="gemma-4-26b-a4b-it",
        log_dir="./logs",
        delta_v_budget_m_s=5.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def test_generate_succeeds_on_primary_without_touching_fallback():
    client = GemmaClient(settings=_settings())

    with patch.object(client, "_generate_ollama", return_value="primary ok") as mock_ollama, \
         patch.object(client, "_generate_hosted_api") as mock_api:
        result = client.generate(prompt="test prompt")

    assert result == "primary ok"
    assert mock_ollama.call_count == 1
    mock_api.assert_not_called()
    assert client.last_backend_used == "ollama"


def test_generate_falls_back_to_other_backend_after_primary_exhausts_retry():
    client = GemmaClient(settings=_settings(gemma_backend="ollama"))

    with patch.object(client, "_generate_ollama", side_effect=GemmaClientError("ollama down")) as mock_ollama, \
         patch.object(client, "_generate_hosted_api", return_value="fallback response") as mock_api:
        result = client.generate(prompt="test prompt")

    assert result == "fallback response"
    assert mock_ollama.call_count == 2  # original attempt + same-backend retry
    assert mock_api.call_count == 1
    assert client.last_backend_used == "api"


def test_generate_raises_primary_error_if_other_backend_unconfigured():
    client = GemmaClient(settings=_settings(gemma_backend="ollama", gemma_api_key=""))

    with patch.object(client, "_generate_ollama", side_effect=GemmaClientError("ollama down")) as mock_ollama, \
         patch.object(client, "_generate_hosted_api") as mock_api:
        with pytest.raises(GemmaClientError, match="ollama down"):
            client.generate(prompt="test prompt")

    assert mock_ollama.call_count == 2
    mock_api.assert_not_called()


def test_generate_raises_combined_error_naming_both_failures_if_fallback_also_fails():
    """Regression test: generate() used to raise ONLY the primary's error
    when both backends failed, silently discarding the fallback's actual
    failure reason - usually the more useful one, since the primary is
    often deliberately/expectedly down (e.g. this project's own failover
    demo step intentionally breaks the local backend to test the cloud
    fallback; if the cloud call then also failed for a real reason, that
    reason was invisible)."""
    client = GemmaClient(settings=_settings(gemma_backend="ollama"))

    with patch.object(client, "_generate_ollama", side_effect=GemmaClientError("ollama down")), \
         patch.object(client, "_generate_hosted_api", side_effect=GemmaClientError("api also down")):
        with pytest.raises(GemmaClientError) as exc_info:
            client.generate(prompt="test prompt")

    assert "ollama down" in str(exc_info.value)
    assert "api also down" in str(exc_info.value)
    assert client.last_backend_used is None


def _make_conjunction_event(min_distance_km: float) -> TelemetryEvent:
    return TelemetryEvent(
        event_id="conj-test",
        timestamp=datetime.now(timezone.utc),
        source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "A",
            "object_b_id": "2", "object_b_name": "B",
            "min_distance_km": min_distance_km,
            "time_of_closest_approach": "2026-08-02T00:00:00+00:00",
            "relative_velocity_km_s": 5.0,
        },
    )


def test_rationale_provenance_reflects_cross_backend_fallback():
    """End-to-end through decide_node: when the primary backend fails and
    GemmaClient falls over to the other one, GemmaProvenance.model_used
    should say so, not silently report the configured-but-failing backend."""
    client = GemmaClient(settings=_settings(gemma_backend="ollama"))
    event = _make_conjunction_event(50.0)
    finding = AnomalyFinding(
        event_id=event.event_id, severity=Severity.WATCH,
        description="Test finding.", confidence=0.8,
    )
    decide_node = make_decide_node(client)

    with patch.object(client, "_generate_ollama", side_effect=GemmaClientError("ollama down")), \
         patch.object(client, "_generate_hosted_api", return_value="Recommendation: continue."):
        result_state = decide_node({
            "telemetry": event, "finding": finding, "decision": None, "log_path": None,
        })

    provenance = result_state["rationale_provenance"]
    assert provenance.source == "gemma"
    assert "api" in provenance.model_used
    assert "ollama" in provenance.model_used
    assert result_state["decision"].rationale == "Recommendation: continue."


def _mock_embed_response(embeddings: list[list[float]]):
    response = MagicMock()
    response.json.return_value = {"embeddings": embeddings}
    response.raise_for_status = MagicMock()
    return response


@patch("src.gemma_client.requests.post")
def test_embed_posts_to_ollama_embed_endpoint_and_returns_vectors(mock_post):
    mock_post.return_value = _mock_embed_response([[0.1, 0.2], [0.3, 0.4]])
    client = GemmaClient(settings=_settings())

    result = client.embed(["first text", "second text"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    call_kwargs = mock_post.call_args.kwargs
    assert mock_post.call_args.args[0] == "http://localhost:11434/api/embed"
    assert call_kwargs["json"] == {"model": "nomic-embed-text", "input": ["first text", "second text"]}


@patch("src.gemma_client.requests.post")
def test_embed_raises_on_network_failure(mock_post):
    mock_post.side_effect = requests.RequestException("connection refused")
    client = GemmaClient(settings=_settings())

    with pytest.raises(GemmaClientError, match="Ollama embeddings unreachable"):
        client.embed(["text"])


@patch("src.gemma_client.requests.post")
def test_embed_raises_on_unexpected_response_shape(mock_post):
    response = MagicMock()
    response.json.return_value = {"unexpected": "shape"}
    response.raise_for_status = MagicMock()
    mock_post.return_value = response
    client = GemmaClient(settings=_settings())

    with pytest.raises(GemmaClientError, match="Unexpected Ollama embeddings response shape"):
        client.embed(["text"])


def _mock_generate_response(text: str):
    response = MagicMock()
    response.json.return_value = {"response": text}
    response.raise_for_status = MagicMock()
    return response


@patch("src.gemma_client.requests.post")
def test_generate_passes_format_through_to_ollama_payload(mock_post):
    mock_post.return_value = _mock_generate_response('{"verdict": "GO", "reason": "ok"}')
    client = GemmaClient(settings=_settings(gemma_backend="ollama"))
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}

    client.generate(prompt="test", format=schema)

    assert mock_post.call_args.kwargs["json"]["format"] == schema


@patch("src.gemma_client.requests.post")
def test_generate_omits_format_key_when_not_requested(mock_post):
    mock_post.return_value = _mock_generate_response("plain text")
    client = GemmaClient(settings=_settings(gemma_backend="ollama"))

    client.generate(prompt="test")

    assert "format" not in mock_post.call_args.kwargs["json"]


def test_generate_hosted_api_ignores_format_instead_of_raising():
    """format is Ollama-only (see GemmaClient.generate's docstring) - the
    hosted API path must silently ignore it, not raise, so a call that
    starts on Ollama and cross-backend-falls-over to the hosted API (see
    test_generate_falls_back_to_other_backend_after_primary_exhausts_retry)
    degrades to unstructured text instead of failing outright."""
    client = GemmaClient(settings=_settings(gemma_backend="api"))
    schema = {"type": "object"}

    with patch.object(client, "_generate_hosted_api", return_value="plain text") as mock_api:
        result = client.generate(prompt="test", format=schema)

    assert result == "plain text"
    assert mock_api.call_args.args[-1] == schema  # received it, didn't error
