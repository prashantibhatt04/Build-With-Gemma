#!/usr/bin/env bash
# Real Postgres backup - a post-Phase-6 improvement (ROADMAP_TO_PRODUCT.md).
#
# A real pg_dump wrapper, not a placeholder - only applies to the Postgres
# backend (DATABASE_URL configured); the JSONL backend's own "backup" is
# just copying the logs/ directory, which needs no dedicated tooling.
#
# Uses pg_dump's custom format (-Fc): compressed, and restorable with
# pg_restore in a way that supports partial/selective restore - a plain
# SQL dump would work too but loses that flexibility for no real benefit.
#
# Usage:
#   ./scripts/backup_postgres.sh                       # writes to ./backups/
#   BACKUP_DIR=/mnt/backups ./scripts/backup_postgres.sh
#
# Restore:
#   pg_restore --dbname="$DATABASE_URL" --clean --if-exists backups/decisions-2026-08-03T120000Z.dump
#
# Real environment note, found live-verifying this script: if pg_dump
# fails with "aborting because of server version mismatch", your PATH's
# pg_dump is older than the actual Postgres server (e.g. Homebrew
# installed both postgresql@16 and postgresql@17 and the default `pg_dump`
# symlink points at the older one) - point PATH at the matching version's
# bin directory, e.g. PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH".
set -euo pipefail

if [ -z "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL is not set - this project is using the JSONL backend, which needs no" >&2
  echo "database backup: back up the logs/ directory directly (e.g. rsync/cp/tar it)." >&2
  exit 0
fi

BACKUP_DIR="${BACKUP_DIR:-./backups}"
mkdir -p "$BACKUP_DIR"

TIMESTAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUTPUT_FILE="$BACKUP_DIR/decisions-$TIMESTAMP.dump"

echo "Backing up $DATABASE_URL to $OUTPUT_FILE ..."
pg_dump --format=custom --dbname="$DATABASE_URL" --file="$OUTPUT_FILE"

REAL_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "Done: $OUTPUT_FILE ($REAL_SIZE)"
