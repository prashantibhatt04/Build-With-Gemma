#!/usr/bin/env python3
"""Real retention cleanup for the Postgres-backed audit log - a
post-Phase-6 improvement (ROADMAP_TO_PRODUCT.md).

Only applies to the Postgres backend (DATABASE_URL configured) - the
JSONL backend already has a trivially simple retention story of its own
(one file per real day, e.g. logs/decisions-2026-08-01.jsonl - just
delete the old date-stamped files directly, no code needed for that).

Deletes rows by `created_at` (when a decision was actually inserted, set
by Postgres itself - see src/postgres_logging.py), not any timestamp
inside the entry's own payload - a historical replay's real 2009
collision timestamp must not make that row look 16+ years old for
retention purposes when it was actually logged today.

Deliberately requires an explicit, positive --days (or RETENTION_DAYS)
value - no default retention window is applied automatically. A cron job
that silently deletes a real audit trail because someone forgot to set a
value would be a genuinely dangerous default; refusing to run without
one is the safe failure mode here, the same "explicit, not silent"
principle every other optional feature in this project already follows.

Usage:
    python scripts/retention_cleanup.py --days 365
    RETENTION_DAYS=365 python scripts/retention_cleanup.py
    python scripts/retention_cleanup.py --days 365 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.postgres_logging import PostgresDecisionLogStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Delete decisions logged more than this many days ago. Falls back to RETENTION_DAYS if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many rows WOULD be deleted, without actually deleting them.",
    )
    args = parser.parse_args()

    days = args.days if args.days is not None else int(os.getenv("RETENTION_DAYS", "0"))
    if days <= 0:
        print(
            "No positive retention window configured - pass --days N or set RETENTION_DAYS. "
            "Refusing to run with an implicit default (deleting audit history is not something "
            "to do by accident)."
        )
        sys.exit(1)

    if not settings.database_url:
        print(
            "DATABASE_URL is not configured - this project is using the JSONL backend, which has "
            "its own simple retention story: delete old logs/decisions-YYYY-MM-DD.jsonl files "
            "directly. Nothing for this script to do."
        )
        sys.exit(0)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    store = PostgresDecisionLogStore(settings.database_url)

    if args.dry_run:
        count = store.count_older_than(cutoff)
        print(f"Dry run: {count} real rows older than {cutoff.isoformat()} would be deleted.")
        return

    deleted = store.delete_older_than(cutoff)
    print(f"Deleted {deleted} real rows older than {cutoff.isoformat()} (retention: {days} days).")


if __name__ == "__main__":
    main()
