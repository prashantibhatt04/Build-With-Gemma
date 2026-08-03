"""Real Postgres-backed DecisionLogStore - ROADMAP_TO_PRODUCT.md Phase 4,
with retention support added as a post-Phase-6 improvement.

Why this exists: the original JSONL-file store (src/logging_utils.py)
scans and rewrites whole files for find()/update() - fine for a demo's
occasional clicks, but not safe under the concurrent writes a real
continuous scheduler (scripts/scheduler.py) and multiple dashboard/CLI
operators would produce at the same time against the same log. A real
database gives real concurrent-write safety (row-level locking, not
whole-file rewrites) and real indexed lookup instead of a linear file
scan per find()/update() call.

Uses psycopg (v3) directly - no ORM. Each DecisionLogEntry is stored as
one row: the full entry as JSONB (so the schema can evolve without a
migration for every new field, matching how the JSONL store never needed
one either) plus an indexed event_id column for real fast lookup, and a
SERIAL id for stable insertion-order ("oldest first") semantics matching
JSONLDecisionLogStore's own load_all() ordering exactly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import psycopg
from psycopg.rows import dict_row

from .logging_utils import DecisionLogStore
from .schemas import DecisionLogEntry

DEFAULT_TABLE_NAME = "decision_log_entries"


class PostgresDecisionLogStore(DecisionLogStore):
    def __init__(self, database_url: str, table_name: str = DEFAULT_TABLE_NAME):
        # table_name is never end-user input (it's a deployment-time
        # config choice, same trust level as database_url itself), so
        # direct interpolation below is safe - not building a query from
        # untrusted request data.
        self.database_url = database_url
        self.table_name = table_name
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    entry_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS {self.table_name}_event_id_idx
                    ON {self.table_name} (event_id);
                """
            )
            conn.commit()

    def append(self, entry: DecisionLogEntry) -> str:
        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                f"INSERT INTO {self.table_name} (event_id, entry_json) VALUES (%s, %s)",
                (entry.telemetry.event_id, entry.model_dump_json()),
            )
            conn.commit()
        return f"postgres:{self.table_name}:{entry.telemetry.event_id}"

    def find(self, event_id: str) -> Optional[DecisionLogEntry]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                f"SELECT entry_json FROM {self.table_name} WHERE event_id = %s "
                "ORDER BY id ASC LIMIT 1",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        # psycopg auto-deserializes a JSONB column into a Python dict, not
        # the raw JSON string model_validate_json expects.
        return DecisionLogEntry.model_validate(row["entry_json"])

    def update(self, event_id: str, updated_entry: DecisionLogEntry) -> None:
        # Matches the FIRST occurrence, same semantics JSONLDecisionLogStore
        # already established (and every adapter's run_id scheme exists
        # specifically to keep event_ids unique per real scan - see
        # celestrak_adapter.py) - picks the earliest-inserted row by id,
        # not an arbitrary one.
        with psycopg.connect(self.database_url) as conn:
            target_id = conn.execute(
                f"SELECT id FROM {self.table_name} WHERE event_id = %s ORDER BY id ASC LIMIT 1",
                (event_id,),
            ).fetchone()
            if target_id is None:
                raise ValueError(f"No logged decision found with event_id={event_id!r}")
            conn.execute(
                f"UPDATE {self.table_name} SET entry_json = %s WHERE id = %s",
                (updated_entry.model_dump_json(), target_id[0]),
            )
            conn.commit()

    def update_if(
        self,
        event_id: str,
        guard: Callable[[DecisionLogEntry], None],
        mutate: Callable[[DecisionLogEntry], DecisionLogEntry],
    ) -> DecisionLogEntry:
        # SELECT ... FOR UPDATE takes a real row lock for the rest of
        # this transaction - a concurrent update_if() on the same row
        # blocks here until this one commits (below), then re-reads the
        # now-committed row before its own guard() runs. That's what
        # closes the race: two callers can no longer both read the
        # pre-update state, both pass guard(), and both write - the
        # second one now genuinely sees the first's result.
        with psycopg.connect(self.database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                f"SELECT id, entry_json FROM {self.table_name} WHERE event_id = %s "
                "ORDER BY id ASC LIMIT 1 FOR UPDATE",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No logged decision found with event_id={event_id!r}")
            entry = DecisionLogEntry.model_validate(row["entry_json"])
            guard(entry)
            updated = mutate(entry)
            conn.execute(
                f"UPDATE {self.table_name} SET entry_json = %s WHERE id = %s",
                (updated.model_dump_json(), row["id"]),
            )
            conn.commit()
        return updated

    def load_all(self) -> list[DecisionLogEntry]:
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT entry_json FROM {self.table_name} ORDER BY id ASC",
            ).fetchall()
        return [DecisionLogEntry.model_validate(row[0]) for row in rows]

    def count_older_than(self, cutoff: datetime) -> int:
        """Real, read-only count of rows delete_older_than(cutoff) would
        remove - used by scripts/retention_cleanup.py's --dry-run, so a
        dry run reports a real number instead of just "would delete
        something." Same created_at semantics as delete_older_than."""
        with psycopg.connect(self.database_url) as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name} WHERE created_at < %s",
                (cutoff,),
            ).fetchone()
        return row[0]

    def delete_older_than(self, cutoff: datetime) -> int:
        """Real, DB-side retention deletion - see scripts/retention_cleanup.py.
        Uses `created_at` (when the row was actually inserted, set by
        Postgres itself at INSERT time - see the table schema above), not
        any timestamp inside entry_json, so retention behavior is
        unaffected by whatever real-world event time a decision's own
        payload happens to carry (a historical replay's real 2009
        timestamp, for instance, must not make that row look 16+ years
        old for retention purposes - it was inserted recently, same as
        any other real entry).

        Returns the real number of rows actually deleted, so a caller
        (scripts/retention_cleanup.py) can report what happened rather
        than assuming success silently.
        """
        with psycopg.connect(self.database_url) as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE created_at < %s",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()
        return deleted
