"""Audit log for every decision the pipeline makes.

DecisionLogger is a thin business-logic wrapper: `mark_reviewed` and
`approve_maneuver` contain the actual audit-log rules (what changes,
what's rejected as invalid), while raw persistence is delegated to a
`DecisionLogStore` (ROADMAP_TO_PRODUCT.md Phase 4). This split exists so
storage can be swapped - JSONL files (the original, default, zero-setup
backend) or a real Postgres database (src/postgres_logging.py, for
continuous/scheduled operation - see scripts/scheduler.py) - without
DecisionLogger's own public interface or any of its callers (dashboard,
RAG, trends, alerting, pipeline.make_log_node) changing at all.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings, settings as default_settings
from .maneuver import verify_maneuver
from .schemas import DecisionLogEntry, ManeuverApproval


class DecisionLogStore(ABC):
    """Storage backend interface DecisionLogger delegates to. Every
    implementation must preserve the same real semantics the original
    JSONL file logic established: append-only ordering from `load_all`
    (oldest first), and `update` only ever touches the one entry matching
    `event_id` - already-append-only files/rows for every other entry."""

    @abstractmethod
    def append(self, entry: DecisionLogEntry) -> str:
        """Persists a new entry. Returns an implementation-defined,
        non-empty identifier string (a file path for JSONL, a connection
        description for Postgres) - callers treat it as opaque."""
        raise NotImplementedError

    @abstractmethod
    def find(self, event_id: str) -> Optional[DecisionLogEntry]:
        raise NotImplementedError

    @abstractmethod
    def update(self, event_id: str, updated_entry: DecisionLogEntry) -> None:
        """Replaces the persisted entry matching event_id. Raises
        ValueError if no such entry exists."""
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[DecisionLogEntry]:
        raise NotImplementedError


class JSONLDecisionLogStore(DecisionLogStore):
    """The original backend: each DecisionLogEntry as one JSON line in a
    dated file under log_dir. Append-only by design - update() is the one
    intentional exception, rewriting a single matched line in place."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"decisions-{date_str}.jsonl")

    def append(self, entry: DecisionLogEntry) -> str:
        path = self._log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return path

    def _find_with_location(self, event_id: str) -> Optional[tuple[Path, int, DecisionLogEntry]]:
        for path in sorted(Path(self.log_dir).glob("decisions-*.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                entry = DecisionLogEntry.model_validate_json(line)
                if entry.telemetry.event_id == event_id:
                    return path, i, entry
        return None

    def find(self, event_id: str) -> Optional[DecisionLogEntry]:
        found = self._find_with_location(event_id)
        return found[2] if found else None

    def update(self, event_id: str, updated_entry: DecisionLogEntry) -> None:
        found = self._find_with_location(event_id)
        if found is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")
        path, line_index, _ = found
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[line_index] = updated_entry.model_dump_json()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load_all(self) -> list[DecisionLogEntry]:
        entries = []
        for path in sorted(Path(self.log_dir).glob("decisions-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                entries.append(DecisionLogEntry.model_validate_json(line))
        return entries


def _default_store(settings: Settings) -> DecisionLogStore:
    if settings.database_url:
        # Imported lazily so the common (JSONL) path never requires
        # psycopg to be installed at all - only paid for when
        # DATABASE_URL is actually configured.
        from .postgres_logging import PostgresDecisionLogStore
        return PostgresDecisionLogStore(settings.database_url)
    return JSONLDecisionLogStore(settings.log_dir)


class DecisionLogger:
    """Public interface every caller in this project uses - unchanged
    across both storage backends. See module docstring."""

    def __init__(self, settings: Settings = default_settings, store: Optional[DecisionLogStore] = None):
        self.settings = settings
        self._store = store or _default_store(settings)

    def log(self, entry: DecisionLogEntry) -> str:
        return self._store.append(entry)

    def find_entry(self, event_id: str) -> Optional[DecisionLogEntry]:
        return self._store.find(event_id)

    def load_all_entries(self) -> list[DecisionLogEntry]:
        """Reads every persisted decision, oldest first - used by
        scripts/dashboard.py for the live risk board."""
        return self._store.load_all()

    def mark_reviewed(self, event_id: str, reviewed_by: str) -> DecisionLogEntry:
        """Sets human_reviewed/human_reviewed_at/reviewed_by on the logged
        entry matching event_id. Raises ValueError if no entry with this
        event_id is found."""
        entry = self.find_entry(event_id)
        if entry is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")

        updated = entry.model_copy(update={
            "human_reviewed": True,
            "human_reviewed_at": datetime.now(timezone.utc),
            "reviewed_by": reviewed_by,
        })
        self._store.update(event_id, updated)
        return updated

    def approve_maneuver(self, event_id: str, approved: bool, approved_by: str) -> DecisionLogEntry:
        """Resolves a CRITICAL-severity maneuver that's awaiting human
        approval (cloud/api backend - see pipeline.decide_node and
        schemas.ManeuverApproval).

        If approved: actually verifies the maneuver now (nothing was
        applied while it was pending) and records who approved it. If
        rejected: records that: verified_clearance stays None, nothing was
        executed. Either way, awaiting_human_approval flips to False and
        maneuver_approval gets populated.

        Raises ValueError if no matching entry is found, or if it isn't
        actually awaiting approval (already resolved, budget-blocked
        instead, or not CRITICAL at all).
        """
        entry = self.find_entry(event_id)
        if entry is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")

        if not entry.decision.awaiting_human_approval:
            raise ValueError(
                f"Decision for event_id={event_id!r} is not awaiting human approval "
                f"(awaiting_human_approval={entry.decision.awaiting_human_approval})"
            )

        now = datetime.now(timezone.utc)
        if approved:
            verified_clearance = verify_maneuver(
                entry.telemetry.raw_data["min_distance_km"], entry.decision.maneuver_plan,
            )
            status_note = f"[HUMAN DECISION: APPROVED by {approved_by}]"
        else:
            verified_clearance = None
            status_note = f"[HUMAN DECISION: REJECTED by {approved_by}]"

        approval = ManeuverApproval(
            mode="human",
            approved=approved,
            approved_by=approved_by,
            approved_at=now,
            reason=f"{'Approved' if approved else 'Rejected'} via approve_maneuver (CLI/demo prompt).",
        )
        updated_decision = entry.decision.model_copy(update={
            "awaiting_human_approval": False,
            "verified_clearance": verified_clearance,
            "maneuver_approval": approval,
            "rationale": f"{entry.decision.rationale} {status_note}",
        })
        updated_entry = entry.model_copy(update={"decision": updated_decision})

        self._store.update(event_id, updated_entry)
        return updated_entry
