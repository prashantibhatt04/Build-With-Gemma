"""Tests for scripts/healthcheck_scheduler.py - the Docker HEALTHCHECK
CMD for the scheduler service (docker-compose.yml). Exercises main()'s
real exit-code contract (0 healthy / 1 unhealthy) against a real
tmp_path heartbeat file - Docker itself isn't involved, matching this
project's practice of testing the logic a script runs rather than
Docker's own invocation of it (that part is covered by the live
`docker compose up` verification in ROADMAP_TO_PRODUCT.md instead).
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.healthcheck_scheduler as healthcheck_scheduler


def test_main_returns_0_for_a_fresh_heartbeat(tmp_path, monkeypatch):
    path = tmp_path / "heartbeat.json"
    heartbeat = {"timestamp": datetime.now(timezone.utc).isoformat(), "interval_seconds": 60.0, "tick_number": 5}
    path.write_text(json.dumps(heartbeat))
    monkeypatch.setattr(healthcheck_scheduler, "HEARTBEAT_PATH", path)

    assert healthcheck_scheduler.main() == 0


def test_main_returns_1_when_heartbeat_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(healthcheck_scheduler, "HEARTBEAT_PATH", tmp_path / "does-not-exist.json")

    assert healthcheck_scheduler.main() == 1


def test_main_returns_1_for_a_stale_heartbeat(tmp_path, monkeypatch):
    path = tmp_path / "heartbeat.json"
    stale_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
    heartbeat = {"timestamp": stale_timestamp.isoformat(), "interval_seconds": 60.0, "tick_number": 1}
    path.write_text(json.dumps(heartbeat))
    monkeypatch.setattr(healthcheck_scheduler, "HEARTBEAT_PATH", path)

    assert healthcheck_scheduler.main() == 1


def test_main_returns_1_for_malformed_json(tmp_path, monkeypatch):
    path = tmp_path / "heartbeat.json"
    path.write_text("not valid json")
    monkeypatch.setattr(healthcheck_scheduler, "HEARTBEAT_PATH", path)

    assert healthcheck_scheduler.main() == 1
