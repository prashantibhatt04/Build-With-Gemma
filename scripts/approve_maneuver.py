#!/usr/bin/env python3
"""Approve or reject a maneuver that's awaiting human approval - logged
when the configured Gemma backend is "api" (cloud), treated in this system
as "ground control is reachable" (see src/schemas.py:ManeuverApproval).

Usage:
    python scripts/approve_maneuver.py <event_id> <approver_name>            # approve
    python scripts/approve_maneuver.py <event_id> <approver_name> --reject   # reject
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_utils import DecisionLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve or reject a maneuver awaiting human approval.")
    parser.add_argument("event_id", help="telemetry.event_id of the pending decision")
    parser.add_argument("approver_name", help="name or identifier of the approver")
    parser.add_argument("--reject", action="store_true", help="Reject instead of approve.")
    args = parser.parse_args()

    logger = DecisionLogger()
    try:
        updated = logger.approve_maneuver(
            args.event_id, approved=not args.reject, approved_by=args.approver_name,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    verdict = "APPROVED" if updated.decision.maneuver_approval.approved else "REJECTED"
    print(f"{verdict} maneuver for {args.event_id!r} by {args.approver_name!r}")
    if updated.decision.verified_clearance is not None:
        vc = updated.decision.verified_clearance
        print(f"Verified new separation: {vc.new_min_distance_km:.2f}km (cleared={vc.cleared})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
