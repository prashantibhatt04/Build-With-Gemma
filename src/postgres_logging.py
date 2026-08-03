"""Real Postgres-backed DecisionLogStore - ROADMAP_TO_PRODUCT.md Phase 4.

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

from typing import Optional

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

    def load_all(self) -> list[DecisionLogEntry]:
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(
                f"SELECT entry_json FROM {self.table_name} ORDER BY id ASC",
            ).fetchall()
        return [DecisionLogEntry.model_validate(row[0]) for row in rows]
