#!/usr/bin/env python3
"""Mark a logged decision as human-reviewed.

Usage: python scripts/mark_reviewed.py <event_id> <reviewed_by>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import DecisionLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark a logged decision as human-reviewed.")
    parser.add_argument("event_id", help="telemetry.event_id of the decision to mark reviewed")
    parser.add_argument("reviewed_by", help="name or identifier of the reviewer")
    args = parser.parse_args()

    logger = DecisionLogger()
    try:
        updated = logger.mark_reviewed(args.event_id, args.reviewed_by)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Marked {args.event_id!r} as reviewed by {args.reviewed_by!r} at {updated.human_reviewed_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
