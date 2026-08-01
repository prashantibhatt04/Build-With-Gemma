#!/usr/bin/env python3
"""Standalone connectivity check for whichever Gemma backend is configured.

Usage: python scripts/check_gemma.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.gemma_client import GemmaClient, GemmaClientError


def main() -> int:
    print(f"Checking Gemma backend: {settings.gemma_backend} (model={settings.gemma_model})")

    client = GemmaClient()
    try:
        reply = client.generate(prompt="Reply with the single word: OK")
    except GemmaClientError as exc:
        print(f"UNREACHABLE: {exc}")
        print("This is expected if no backend is running yet - failing gracefully.")
        return 0

    print("REACHABLE")
    print(f"Response: {reply!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
