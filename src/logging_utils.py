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

import fcntl
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from .config import Settings, settings as default_settings
from .maneuver import verify_maneuver
from .schemas import DecisionLogEntry, ManeuverApproval


def _atomic_write_lines(path: Path, lines: list[str]) -> None:
    """Writes lines to path atomically via write-to-a-temp-file-then-
    os.replace, so a process kill mid-write (SIGKILL, OOM-kill, disk
    full, container restart - all realistic for a real 24/7
    scripts/scheduler.py or a Dockerized dashboard) can never leave path
    itself in a torn/truncated state. os.replace's underlying rename()
    is atomic at the filesystem level: any concurrent reader sees either
    the complete old content or the complete new content, never a
    half-written mix - unlike the previous seek(0)+write()+truncate()
    in-place rewrite, which could leave one corrupted JSON line that
    poisoned every OTHER, perfectly intact entry in the same file (every
    reader here validates every line unconditionally - see
    _parse_lines_tolerantly below for the other half of this fix)."""
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _safe_parse_line(path: Path, index: int, line: str) -> Optional[DecisionLogEntry]:
    """Parses one line, returning None (with a printed warning, not a
    crash) if it fails to validate - real defense against a real
    corrupted-line failure mode: a torn write from before the atomic-
    write fix above, a hand-edited file, or any other single-line
    corruption. Without this, DecisionLogEntry.model_validate_json
    raising on ONE bad line took down `load_all_entries` (the dashboard's
    main table, the alert-cooldown history fetch in pipeline.py, every
    read endpoint in scripts/api.py) and `find`/`update_if` for every
    OTHER, perfectly intact entry in that same date-file - a total
    outage of one day's audit log from a single bad line, not a lost
    record."""
    try:
        return DecisionLogEntry.model_validate_json(line)
    except ValidationError as exc:
        print(f"WARNING: skipping corrupted log line {index} in {path}: {exc}")
        return None


def _parse_lines_tolerantly(path: Path, lines: list[str]) -> list[DecisionLogEntry]:
    entries = []
    for i, line in enumerate(lines):
        entry = _safe_parse_line(path, i, line)
        if entry is not None:
            entries.append(entry)
    return entries


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
    def update_if(
        self,
        event_id: str,
        guard: Callable[[DecisionLogEntry], None],
        mutate: Callable[[DecisionLogEntry], DecisionLogEntry],
    ) -> DecisionLogEntry:
        """Atomic read-check-write: finds the entry matching event_id,
        calls guard(entry) (which must raise ValueError if the entry
        isn't in a state this update applies to), then persists
        mutate(entry) - all as one operation a concurrent caller can't
        interleave with (a real Postgres row lock, a real flock'd JSONL
        file). Unlike find()+update() called separately, two callers
        racing on the same event_id can't both pass the same guard check
        and silently overwrite each other - see approve_maneuver, the
        real bug this exists to close (an approve and a reject racing on
        the same CRITICAL event could otherwise both "succeed" while only
        one write actually survived). Raises ValueError if no entry with
        this event_id exists, or if guard(entry) raises."""
        raise NotImplementedError

    @abstractmethod
    def load_all(self) -> list[DecisionLogEntry]:
        raise NotImplementedError


class JSONLDecisionLogStore(DecisionLogStore):
    """The original backend: each DecisionLogEntry as one JSON line in a
    dated file under log_dir. Append-only by design - update()/update_if()
    are the one intentional exception, rewriting a single matched line in
    place - but always via _atomic_write_lines (write-to-temp-then-rename),
    never an in-place seek+write+truncate, so a process kill mid-rewrite
    can't corrupt the file. Every read here also tolerates a single
    already-corrupted line (from before this real bug was fixed, or any
    other cause) via _parse_lines_tolerantly/_safe_parse_line, rather than
    letting one bad line take down every other, perfectly intact entry in
    the same file."""

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
                entry = _safe_parse_line(path, i, line)
                if entry is not None and entry.telemetry.event_id == event_id:
                    return path, i, entry
        return None

    def find(self, event_id: str) -> Optional[DecisionLogEntry]:
        found = self._find_with_location(event_id)
        return found[2] if found else None

    def update(self, event_id: str, updated_entry: DecisionLogEntry) -> None:
        self.update_if(event_id, guard=lambda entry: None, mutate=lambda entry: updated_entry)

    def update_if(
        self,
        event_id: str,
        guard: Callable[[DecisionLogEntry], None],
        mutate: Callable[[DecisionLogEntry], DecisionLogEntry],
    ) -> DecisionLogEntry:
        # Locks a STABLE, never-replaced ".lock" sidecar file, not the
        # data file itself - real cross-process AND cross-thread mutual
        # exclusion (flock is scoped to the open file description). This
        # has to be a separate file: the data file's own content is
        # replaced atomically below via _atomic_write_lines (os.replace,
        # i.e. a real rename), and flock-ing a path that might get
        # renamed out from under a concurrent waiter is a real, subtle
        # bug of its own - a second caller that already opened the OLD
        # inode before the rename would win its lock against a now-
        # orphaned file and silently lose its write, never visible via
        # the real path again. The lock file's own inode never changes,
        # so that race can't happen here.
        for path in sorted(Path(self.log_dir).glob("decisions-*.jsonl")):
            lock_path = path.with_suffix(path.suffix + ".lock")
            with open(lock_path, "a", encoding="utf-8") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                lines = path.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    entry = _safe_parse_line(path, i, line)
                    if entry is not None and entry.telemetry.event_id == event_id:
                        guard(entry)
                        updated = mutate(entry)
                        lines[i] = updated.model_dump_json()
                        _atomic_write_lines(path, lines)
                        return updated
                # Lock released when the `with` block exits below, before
                # moving on to check the next file.
        raise ValueError(f"No logged decision found with event_id={event_id!r}")

    def load_all(self) -> list[DecisionLogEntry]:
        entries = []
        for path in sorted(Path(self.log_dir).glob("decisions-*.jsonl")):
            entries.extend(_parse_lines_tolerantly(path, path.read_text(encoding="utf-8").splitlines()))
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
        def mutate(entry: DecisionLogEntry) -> DecisionLogEntry:
            return entry.model_copy(update={
                "human_reviewed": True,
                "human_reviewed_at": datetime.now(timezone.utc),
                "reviewed_by": reviewed_by,
            })

        return self._store.update_if(event_id, guard=lambda entry: None, mutate=mutate)

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
        instead, or not CRITICAL at all) - checked atomically against the
        stored entry, so two calls racing on the same event_id (e.g. an
        approve and a reject arriving at nearly the same time) can't both
        pass this check and silently overwrite each other; the loser gets
        this ValueError instead of appearing to succeed.
        """
        def guard(entry: DecisionLogEntry) -> None:
            if not entry.decision.awaiting_human_approval:
                raise ValueError(
                    f"Decision for event_id={event_id!r} is not awaiting human approval "
                    f"(awaiting_human_approval={entry.decision.awaiting_human_approval})"
                )

        def mutate(entry: DecisionLogEntry) -> DecisionLogEntry:
            now = datetime.now(timezone.utc)
            if approved:
                verified_clearance = verify_maneuver(
                    entry.telemetry.raw_data["min_distance_km"], entry.decision.maneuver_plan,
                    self.settings.conjunction_critical_km,
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
            return entry.model_copy(update={"decision": updated_decision})

        return self._store.update_if(event_id, guard=guard, mutate=mutate)
