"""Append-only JSON-lines audit log for every decision the pipeline makes."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Settings, settings as default_settings
from .schemas import DecisionLogEntry


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

    def mark_reviewed(self, event_id: str, reviewed_by: str) -> DecisionLogEntry:
        """Sets human_reviewed/human_reviewed_at/reviewed_by on the logged
        entry matching event_id and rewrites that one line in place.

        The log is append-only by design (see class docstring) - this is
        the one intentional exception, since human_reviewed/human_reviewed_at
        already exist on DecisionLogEntry and are otherwise permanently
        inert. Raises ValueError if no entry with this event_id is found.
        """
        found = self.find_entry(event_id)
        if found is None:
            raise ValueError(f"No logged decision found with event_id={event_id!r}")
        path, line_index, entry = found

        updated = entry.model_copy(update={
            "human_reviewed": True,
            "human_reviewed_at": datetime.now(timezone.utc),
            "reviewed_by": reviewed_by,
        })

        lines = path.read_text(encoding="utf-8").splitlines()
        lines[line_index] = updated.model_dump_json()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return updated
