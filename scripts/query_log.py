#!/usr/bin/env python3
"""Ask a plain-English question about the real mission log.

Retrieval-augmented, not fine-tuned: embeds every logged decision via a
local Ollama embedding model, ranks by real cosine similarity against the
question, and asks Gemma to answer using ONLY the retrieved real entries
as context - see src/rag.py for the full design rationale.

Requires a reachable local Ollama for embeddings (GEMMA_EMBED_MODEL,
default nomic-embed-text) regardless of which GEMMA_BACKEND is configured
for narration - the hosted API has no embedding endpoint wired up here.

Usage: python scripts/query_log.py "why was conj-33765-33818 vetoed?"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.gemma_client import GemmaClient, GemmaClientError
from src.logging_utils import DecisionLogger
from src.rag import answer_question


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a plain-English question about the real mission log.")
    parser.add_argument("question", help="e.g. 'which CRITICAL events were vetoed and why?'")
    parser.add_argument("--top-k", type=int, default=5, help="how many logged entries to retrieve as context")
    args = parser.parse_args()

    entries = DecisionLogger(settings=settings).load_all_entries()
    if not entries:
        print("No logged decisions yet - run the demo or dashboard first to generate some.")
        return 0

    client = GemmaClient(settings=settings)
    try:
        result = answer_question(args.question, entries, client, top_k=args.top_k)
    except GemmaClientError as exc:
        print(f"ERROR: mission-log search needs a reachable local Ollama for embeddings: {exc}")
        return 1

    print(result["answer"])
    print(f"\nGrounded in {len(result['retrieved_event_ids'])} real logged entries: "
          f"{', '.join(result['retrieved_event_ids'])}")
    if result["provenance"] is not None:
        print(f"(source: {result['provenance'].source}, model: {result['provenance'].model_used})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
