#!/usr/bin/env python3
"""Docker HEALTHCHECK CMD for the scheduler service (docker-compose.yml) -
a post-Phase-6 improvement (ROADMAP_TO_PRODUCT.md).

scripts/scheduler.py has no HTTP server to probe (unlike the dashboard's
Streamlit endpoint or the API's /health) - real liveness here comes from
reading the heartbeat file it writes at startup and after every tick
(see write_heartbeat/heartbeat_is_fresh in scheduler.py) and checking
it's recent. Exits 0 (healthy) or 1 (unhealthy) - the exit code Docker's
HEALTHCHECK actually reads, nothing printed is consumed by Docker but
stderr is useful when running this by hand to debug a real unhealthy
container (`docker compose exec scheduler python scripts/healthcheck_scheduler.py`).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scheduler import HEARTBEAT_PATH, heartbeat_is_fresh


def main() -> int:
    if not HEARTBEAT_PATH.exists():
        print(f"unhealthy: no heartbeat file at {HEARTBEAT_PATH} yet", file=sys.stderr)
        return 1

    try:
        heartbeat = json.loads(HEARTBEAT_PATH.read_text())
        fresh = heartbeat_is_fresh(heartbeat, datetime.now(timezone.utc))
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"unhealthy: malformed heartbeat file: {exc}", file=sys.stderr)
        return 1

    if not fresh:
        print(f"unhealthy: heartbeat is stale: {heartbeat}", file=sys.stderr)
        return 1

    print(f"healthy: {heartbeat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
