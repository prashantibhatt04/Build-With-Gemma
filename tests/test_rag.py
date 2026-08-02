"""Tests for src/rag.py. No real network calls - GemmaClient.embed() and
.generate() are both duck-typed/faked, but the cosine-similarity ranking
and prompt-construction logic itself runs for real.
"""
from datetime import datetime, timezone
from types import SimpleNamespace

from src.gemma_client import GemmaClientError
from src.rag import (
    GEMMA_UNREACHABLE_FALLBACK,
    NO_ENTRIES_ANSWER,
    _cosine_similarity,
    _entry_to_text,
    answer_question,
    get_entry_embeddings,
    retrieve_relevant_entries,
)
from src.schemas import AnomalyFinding, Decision, DecisionLogEntry, GemmaProvenance, Severity, TelemetryEvent


def _entry(event_id: str, rationale: str = "Test rationale.") -> DecisionLogEntry:
    telemetry = TelemetryEvent(
        event_id=event_id, timestamp=datetime.now(timezone.utc), source="celestrak",
        raw_data={
            "object_a_id": "1", "object_a_name": "SAT-A", "object_b_id": "2", "object_b_name": "SAT-B",
            "min_distance_km": 50.0,
        },
    )
    finding = AnomalyFinding(event_id=event_id, severity=Severity.WATCH, description="Test finding.", confidence=0.8)
    decision = Decision(action="continue", rationale=rationale, made_at=datetime.now(timezone.utc))
    return DecisionLogEntry(
        telemetry=telemetry, finding=finding, decision=decision,
        rationale_provenance=GemmaProvenance(source="gemma", model_used="fake-model", latency_ms=1.0),
    )


class FakeEmbedClient:
    """Duck-types GemmaClient's embed()/generate()/settings without any
    network calls. embed() returns entry_vectors[event_id] for any text
    starting with "event_id: <id>" (how _entry_to_text always formats
    entries), or query_vector for anything else (i.e. the raw query
    string passed to retrieve_relevant_entries)."""

    def __init__(self, entry_vectors, query_vector, generate_response="Real answer.", raise_on_generate=False):
        self.settings = SimpleNamespace(
            gemma_model="fake-model", gemma_backend="ollama", gemma_embed_model="fake-embed-model",
        )
        self.entry_vectors = entry_vectors
        self.query_vector = query_vector
        self.generate_response = generate_response
        self.raise_on_generate = raise_on_generate
        self.embed_calls: list[list[str]] = []

    def embed(self, texts):
        self.embed_calls.append(list(texts))
        vectors = []
        for text in texts:
            event_id = next(
                (line[len("event_id: "):] for line in text.splitlines() if line.startswith("event_id: ")), None,
            )
            vectors.append(self.entry_vectors[event_id] if event_id is not None else self.query_vector)
        return vectors

    def generate(self, prompt, system=None, timeout=60):
        if self.raise_on_generate:
            raise GemmaClientError("simulated failure")
        return self.generate_response


def test_entry_to_text_includes_key_conjunction_fields():
    text = _entry_to_text(_entry("event-1"))

    assert "event_id: event-1" in text
    assert "severity: watch" in text
    assert "objects: SAT-A vs SAT-B" in text
    assert "min_distance_km: 50.0" in text
    assert "rationale: Test rationale." in text


def test_cosine_similarity_identical_orthogonal_opposite():
    import numpy as np

    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert _cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert round(_cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])), 5) == -1.0
    assert _cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0  # zero vector - no crash


def test_get_entry_embeddings_caches_and_skips_already_embedded_entries(tmp_path):
    entries = [_entry("a"), _entry("b")]
    client = FakeEmbedClient(entry_vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]}, query_vector=[1.0, 0.0])

    first = get_entry_embeddings(entries, client, cache_dir=tmp_path)
    assert len(client.embed_calls) == 1  # one batch call for both new entries
    assert list(first["a"]) == [1.0, 0.0]

    second = get_entry_embeddings(entries, client, cache_dir=tmp_path)
    assert len(client.embed_calls) == 1  # still 1 - nothing new to embed
    assert list(second["b"]) == [0.0, 1.0]


def test_get_entry_embeddings_recomputes_when_embed_model_changes(tmp_path):
    entries = [_entry("a")]
    client_v1 = FakeEmbedClient(entry_vectors={"a": [1.0, 0.0]}, query_vector=[1.0, 0.0])
    get_entry_embeddings(entries, client_v1, cache_dir=tmp_path)

    client_v2 = FakeEmbedClient(entry_vectors={"a": [0.0, 1.0]}, query_vector=[0.0, 1.0])
    client_v2.settings.gemma_embed_model = "a-different-embed-model"
    result = get_entry_embeddings(entries, client_v2, cache_dir=tmp_path)

    assert len(client_v2.embed_calls) == 1  # had to recompute, not reuse the v1 cache
    assert list(result["a"]) == [0.0, 1.0]


def test_retrieve_relevant_entries_ranks_by_cosine_similarity(tmp_path):
    entries = [_entry("closest"), _entry("medium"), _entry("far")]
    client = FakeEmbedClient(
        entry_vectors={"closest": [1.0, 0.0], "medium": [0.5, 0.5], "far": [0.0, 1.0]},
        query_vector=[1.0, 0.0],
    )

    results = retrieve_relevant_entries("some question", entries, client, top_k=2, cache_dir=tmp_path)

    assert [r.entry.telemetry.event_id for r in results] == ["closest", "medium"]
    assert results[0].similarity > results[1].similarity


def test_retrieve_relevant_entries_empty_log_returns_empty(tmp_path):
    client = FakeEmbedClient(entry_vectors={}, query_vector=[1.0, 0.0])
    assert retrieve_relevant_entries("anything", [], client, cache_dir=tmp_path) == []


def test_answer_question_grounds_prompt_in_only_the_retrieved_entries(tmp_path):
    entries = [_entry("closest", rationale="Closest rationale."), _entry("far", rationale="Far rationale.")]
    client = FakeEmbedClient(
        entry_vectors={"closest": [1.0, 0.0], "far": [0.0, 1.0]}, query_vector=[1.0, 0.0],
        generate_response="Real narrated answer.",
    )

    result = answer_question("why?", entries, client, top_k=1, cache_dir=tmp_path)

    assert result["answer"] == "Real narrated answer."
    assert result["retrieved_event_ids"] == ["closest"]
    assert result["provenance"].source == "gemma"


def test_answer_question_falls_back_gracefully_if_gemma_generation_fails(tmp_path):
    entries = [_entry("a")]
    client = FakeEmbedClient(
        entry_vectors={"a": [1.0, 0.0]}, query_vector=[1.0, 0.0], raise_on_generate=True,
    )

    result = answer_question("why?", entries, client, cache_dir=tmp_path)

    assert result["answer"] == GEMMA_UNREACHABLE_FALLBACK
    assert result["retrieved_event_ids"] == ["a"]  # real retrieval still succeeded
    assert result["provenance"].source == "fallback"


def test_answer_question_with_no_logged_entries():
    client = FakeEmbedClient(entry_vectors={}, query_vector=[1.0, 0.0])

    result = answer_question("anything", [], client)

    assert result == {"answer": NO_ENTRIES_ANSWER, "retrieved_event_ids": [], "provenance": None}
