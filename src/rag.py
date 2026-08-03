"""Retrieval-augmented question answering over this project's own real
audit log - "ask about the mission log in plain English" instead of
clicking through the decision table.

Local-only by design: embeddings come from Ollama's own /api/embed
endpoint (nomic-embed-text by default - see config.gemma_embed_model),
regardless of which GEMMA_BACKEND is configured for narration elsewhere
in this project. The hosted Gemini-style API has no embedding endpoint
wired up here, and mixing "local embeddings, hosted generation" would add
a second real dependency for no benefit at this log size.

Retrieval, not fine-tuning, and still narration-only: every logged
entry's real fields (subject, severity, action, rationale) get embedded
and ranked by cosine similarity against the query, and only the top-K
real entries are handed to Gemma as context - Gemma is explicitly
instructed to answer ONLY from those entries, never from outside
knowledge, the same "Gemma explains, never invents" principle as
everywhere else in this project. Nothing here feeds back into severity/
action classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gemma_client import GemmaClient
from .pipeline import _call_gemma_with_provenance
from .schemas import DecisionLogEntry

DEFAULT_CACHE_DIR = Path("data/rag_cache")

ANSWER_SYSTEM_PROMPT = (
    "You are a mission-log assistant for a real spacecraft collision-avoidance "
    "and orbital-hazard triage system. Answer the operator's question using "
    "ONLY the real logged entries provided below as context - every fact in "
    "your answer must come from those entries. If they don't contain enough "
    "information to answer, say so explicitly rather than guessing or using "
    "outside knowledge. Reference event_id values when citing a specific entry. "
    "Keep the answer to 2-4 sentences."
)

NO_ENTRIES_ANSWER = "No logged decisions yet to search."
GEMMA_UNREACHABLE_FALLBACK = (
    "Gemma is currently unreachable, so no narrated answer could be generated - "
    "but the most relevant logged entries were still found by real search; see "
    "the retrieved event_ids below."
)


@dataclass
class RetrievedEntry:
    entry: DecisionLogEntry
    similarity: float


def _entry_to_text(entry: DecisionLogEntry) -> str:
    """A concise, information-dense text representation of one logged
    decision - deliberately its own formatter, not a reuse of display.py's
    or dashboard_data.py's terser one-line "subject" strings, since
    embedding quality benefits from more real content (severity, action,
    rationale, approval state), not just a display-width-constrained
    label."""
    raw = entry.telemetry.raw_data
    lines = [
        f"event_id: {entry.telemetry.event_id}",
        f"timestamp: {entry.telemetry.timestamp.isoformat()}",
        f"source: {entry.telemetry.source}",
        f"severity: {entry.finding.severity.value}",
        f"finding: {entry.finding.description}",
        f"action: {entry.decision.action}",
        f"rationale: {entry.decision.rationale}",
    ]
    if "object_a_name" in raw:
        lines.append(f"objects: {raw['object_a_name']} vs {raw['object_b_name']}")
        lines.append(f"min_distance_km: {raw.get('min_distance_km')}")
    elif "perigee_altitude_km" in raw:
        # Branches on the actual distinguishing field, not just
        # "object_name" - the attitude hazard (Phase 18) also has a
        # single object_name but a different shape entirely (see
        # display.py's identical discipline). A real QA-found bug: the
        # old single "elif object_name" branch always rendered
        # perigee_altitude_km even for attitude entries, which don't
        # have one - "perigee_altitude_km: None" for every attitude
        # entry, with pointing_error_deg (the actual diagnostic number
        # for that hazard) never mentioned at all.
        lines.append(f"object: {raw['object_name']}")
        lines.append(f"perigee_altitude_km: {raw.get('perigee_altitude_km')}")
    elif "pointing_error_deg" in raw:
        lines.append(f"object: {raw['object_name']}")
        lines.append(f"pointing_error_deg: {raw.get('pointing_error_deg')}")
        lines.append(f"angular_rate_deg_s: {raw.get('angular_rate_deg_s')}")
    approval = entry.decision.maneuver_approval
    if approval is not None:
        lines.append(
            f"maneuver_approval: mode={approval.mode} approved={approval.approved} "
            f"approved_by={approval.approved_by}"
        )
    if entry.decision.budget_insufficient:
        lines.append("budget_insufficient: true")
    return "\n".join(lines)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _load_cache(cache_path: Path, model: str) -> dict:
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("model") == model:
            return cache
    return {"model": model, "embeddings": {}}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))


def get_entry_embeddings(
    entries: list[DecisionLogEntry], client: GemmaClient, cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, np.ndarray]:
    """Real embeddings for every entry, computed once and cached on disk
    (keyed by event_id) so repeated searches over an unchanged log don't
    re-embed everything on every query - entries this project logs are
    append-only/immutable in practice, so a cache keyed on event_id alone
    is safe. Invalidated wholesale if gemma_embed_model changes (the
    cached vectors would otherwise silently mix embeddings from two
    different models, which aren't comparable)."""
    cache_path = Path(cache_dir) / "embeddings.json"
    cache = _load_cache(cache_path, client.settings.gemma_embed_model)

    missing = [e for e in entries if e.telemetry.event_id not in cache["embeddings"]]
    if missing:
        vectors = client.embed([_entry_to_text(e) for e in missing])
        for entry, vector in zip(missing, vectors):
            cache["embeddings"][entry.telemetry.event_id] = vector
        _save_cache(cache_path, cache)

    return {
        entry.telemetry.event_id: np.array(cache["embeddings"][entry.telemetry.event_id])
        for entry in entries
    }


def retrieve_relevant_entries(
    query: str, entries: list[DecisionLogEntry], client: GemmaClient,
    top_k: int = 5, cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> list[RetrievedEntry]:
    """Real cosine-similarity search: embeds `query` and every entry (via
    the disk cache above), ranks entries by similarity, returns the top_k.
    Raises GemmaClientError if Ollama's embedding endpoint is unreachable -
    callers should catch this and explain that mission-log search needs a
    reachable local Ollama regardless of the configured GEMMA_BACKEND."""
    if not entries:
        return []
    embeddings = get_entry_embeddings(entries, client, cache_dir)
    query_vector = np.array(client.embed([query])[0])
    scored = [
        RetrievedEntry(entry=entry, similarity=_cosine_similarity(query_vector, embeddings[entry.telemetry.event_id]))
        for entry in entries
    ]
    scored.sort(key=lambda r: r.similarity, reverse=True)
    return scored[:top_k]


def answer_question(
    query: str, entries: list[DecisionLogEntry], client: GemmaClient,
    top_k: int = 5, cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict:
    """Retrieves the top_k most relevant real logged entries for `query`
    and asks Gemma to answer using ONLY those entries as context. Returns
    {"answer", "retrieved_event_ids", "provenance"} - retrieved_event_ids
    is always populated (even on a Gemma failure) so the UI can show real
    grounding sources regardless of whether narration succeeded, matching
    this project's "nothing hidden" audit-log design elsewhere."""
    if not entries:
        return {"answer": NO_ENTRIES_ANSWER, "retrieved_event_ids": [], "provenance": None}

    retrieved = retrieve_relevant_entries(query, entries, client, top_k, cache_dir)
    context = "\n\n---\n\n".join(_entry_to_text(r.entry) for r in retrieved)
    prompt = f"Real logged mission entries:\n\n{context}\n\nOperator question: {query}"

    answer, provenance = _call_gemma_with_provenance(
        client, prompt=prompt, system=ANSWER_SYSTEM_PROMPT, fallback_text=GEMMA_UNREACHABLE_FALLBACK,
    )
    return {
        "answer": answer,
        "retrieved_event_ids": [r.entry.telemetry.event_id for r in retrieved],
        "provenance": provenance,
    }
