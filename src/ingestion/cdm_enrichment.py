"""Enriches already-screened conjunction TelemetryEvents with a real
Space-Track CDM's collision probability, when one exists for that exact
object pair - ROADMAP_TO_PRODUCT.md Phase 2.

Deliberately a separate, explicit, opt-in step - NOT folded into
SpaceTrackAdapter.fetch_batch or CelesTrakAdapter.fetch_batch. Two real
reasons:

1. Rate limits. SpaceTrackClient.fetch_recent_cdms() is already a single
   bounded call (not per-pair) specifically to respect Space-Track's tight
   CDM quota (see its own docstring) - baking it into every screening
   call would make that limit trivial to blow through on a second click.
2. Honesty about what's likely to actually happen. Per src/pc_severity.py's
   module docstring, a generic (non-owner/operator) Space-Track account
   will most likely never receive a real CDM for an arbitrary pair - so
   this step will, in the overwhelming majority of real runs, enrich
   nothing at all and every event will fall through to the existing
   distance-threshold severity path. That's the CORRECT behavior, not a
   bug to work around - same "real data rarely reaches the interesting
   case" pattern already established for CRITICAL conjunctions (Phase 5)
   and CRITICAL decay risk (Phase 14). Keeping it a separate, visible step
   makes that honest, rather than silently attempting (and always
   failing) a lookup on every single screening call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..pc_severity import extract_cdm_summary
from ..schemas import TelemetryEvent

if TYPE_CHECKING:
    from .spacetrack_client import SpaceTrackClient


def _pair_key(id_a: str, id_b: str) -> frozenset[str]:
    return frozenset((str(id_a), str(id_b)))


def enrich_conjunction_events_with_pc(
    events: list[TelemetryEvent], client: "SpaceTrackClient", cdm_limit: int = 10,
) -> list[TelemetryEvent]:
    """Fetches the account's most recent real CDMs ONCE, then merges a
    matching CDM's collision_probability (and supporting fields - see
    pc_severity.extract_cdm_summary) into any event whose object pair it
    covers. Events with no matching CDM are returned unchanged - pipeline
    analyze_node then correctly falls back to the distance threshold for
    those, exactly as designed.

    Only conjunction-shaped events (object_a_id/object_b_id present) are
    ever eligible - decay/attitude events pass through untouched, same
    "check the actual shape, not just the source label" discipline
    analyze_node itself already uses.
    """
    cdm_rows = client.fetch_recent_cdms(limit=cdm_limit)
    cdm_by_pair: dict[frozenset[str], dict] = {}
    for row in cdm_rows:
        summary = extract_cdm_summary(row)
        if summary is None:
            continue
        key = _pair_key(summary["sat1_object_designator"], summary["sat2_object_designator"])
        cdm_by_pair[key] = summary

    if not cdm_by_pair:
        return events

    enriched = []
    for event in events:
        raw = event.raw_data
        object_a_id = raw.get("object_a_id")
        object_b_id = raw.get("object_b_id")
        if object_a_id is None or object_b_id is None:
            enriched.append(event)
            continue
        match = cdm_by_pair.get(_pair_key(object_a_id, object_b_id))
        if match is None:
            enriched.append(event)
            continue
        enriched.append(event.model_copy(update={"raw_data": {**raw, **match}}))

    return enriched
