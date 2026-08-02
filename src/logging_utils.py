"""Append-only JSON-lines audit log for every decision the pipeline makes."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings, settings as default_settings
from .maneuver import verify_maneuver
from .schemas import DecisionLogEntry, ManeuverApproval


class DecisionLogger:
    """Writes each DecisionLogEntry as one JSON line to a dated file under LOG_DIR."""

    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        os.makedirs(self.settings.log_dir, exist_ok=True)

    def _log_path(self) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(self.settings.log_dir, f"decisions-{date_str}.jsonl")

    def log(self, entry: DecisionLogEntry) -> str:
        path = self._log_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return path

    def find_entry(self, event_id: str) -> Optional[tuple[Path, int, DecisionLogEntry]]:
        """Searches every decisions-*.jsonl file under log_dir for an entry
        whose telemetry.event_id matches. Returns (file_path, line_index,
        entry) for the first match, or None."""
        for path in sorted(Path(self.settings.log_dir).glob("decisions-*.jsonl")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                entry = DecisionLogEntry.model_validate_json(line)
                if entry.telemetry.event_id == event_id:
                    return path, i, entry
        return None

    def load_all_entries(self) -> list[DecisionLogEntry]:
        """Reads every persisted decision across all decisions-*.jsonl
        files under log_dir, oldest file first, in on-disk line order
        within each file. Read-only, no interpretation beyond parsing -
        used by scripts/dashboard.py for the live risk board."""
        entries = []
        for path in sorted(Path(self.settings.log_dir).glob("decisions-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                entries.append(DecisionLogEntry.model_validate_json(line))
        return entries

    @staticmethod
    def _rewrite_line(path: Path, line_index: int, updated_entry: DecisionLogEntry) -> None:
        """Rewrites a single line of an existing log file in place. The log
        is append-only by design (see class docstring) - mark_reviewed and
        approve_maneuver are the two intentional exceptions, since the
        fields they set already exist on the schema and are otherwise
        permanently inert."""
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[line_index] = updated_entry.model_dump_json()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def mark_reviewed(self, event_id: str, reviewed_by: str) -> DecisionLogEntry:
        """Sets human_reviewed/human_reviewed_at/reviewed_by on the logged
        entry matching event_id and rewrites that one line in place. Raises
        ValueError if no entry with this event_id is found."""
        found = self.find_entry(event_id)
        if found is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")
        path, line_index, entry = found

        updated = entry.model_copy(update={
            "human_reviewed": True,
            "human_reviewed_at": datetime.now(timezone.utc),
            "reviewed_by": reviewed_by,
        })
        self._rewrite_line(path, line_index, updated)
        return updated

    def approve_maneuver(self, event_id: str, approved: bool, approved_by: str) -> DecisionLogEntry:
        """Resolves a CRITICAL-severity maneuver that's awaiting human
        approval (cloud/api backend - see pipeline.decide_node and
        schemas.ManeuverApproval).

        If approved: actually verifies the maneuver now (nothing was
        applied while it was pending) and records who approved it. If
        rejected: records that: verified_clearance stays None, nothing was
        executed. Either way, awaiting_human_approval flips to False and
        maneuver_approval gets populated. Rewrites the matching log line in
        place, same mechanism as mark_reviewed.

        Raises ValueError if no matching entry is found, or if it isn't
        actually awaiting approval (already resolved, budget-blocked
        instead, or not CRITICAL at all).
        """
        found = self.find_entry(event_id)
        if found is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")
        path, line_index, entry = found

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

        self._rewrite_line(path, line_index, updated_entry)
        return updated_entry
