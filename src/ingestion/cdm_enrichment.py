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

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from ..pc_severity import extract_cdm_summary
from ..schemas import TelemetryEvent

if TYPE_CHECKING:
    from .spacetrack_client import SpaceTrackClient

# How close a CDM's own TCA must be to the freshly-screened event's own
# time_of_closest_approach to treat them as describing the same real
# encounter. Matching on object pair alone isn't enough: two objects with
# repeatedly close passes (common for fragments of the same debris
# field) can have a real CDM published for one specific pass still sit
# among the 10 most recent when a geometrically unrelated pass between
# the same two objects gets screened later - applying that stale Pc
# would misclassify a genuinely different encounter. 24h is generous
# relative to how often two catalog objects plausibly re-conjunct (not
# sub-hour for the vast majority of real objects), while still tolerant
# of a freshly re-propagated TCA drifting a bit from the CDM's own
# predicted one due to a newer TLE.
MAX_TCA_DIFFERENCE = timedelta(hours=24)


def _pair_key(id_a: str, id_b: str) -> frozenset[str]:
    return frozenset((str(id_a), str(id_b)))


def _parse_tca(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _tca_close_enough(cdm_tca_value, event_tca_value) -> bool:
    """True only if both TCAs parse and land within MAX_TCA_DIFFERENCE of
    each other - never on missing/unparseable data (fails closed: no
    match found is the existing safe fallback to the distance-threshold
    path, same as any other non-matching event)."""
    cdm_tca = _parse_tca(cdm_tca_value)
    event_tca = _parse_tca(event_tca_value)
    if cdm_tca is None or event_tca is None:
        return False
    # Space-Track's own CDM TCA has no UTC offset; this project's event
    # TCAs are timezone-aware. Normalize both to naive UTC before
    # subtracting so a real match isn't missed (or a mismatch missed)
    # purely because of an aware/naive comparison error.
    if cdm_tca.tzinfo is not None:
        cdm_tca = cdm_tca.astimezone(timezone.utc).replace(tzinfo=None)
    if event_tca.tzinfo is not None:
        event_tca = event_tca.astimezone(timezone.utc).replace(tzinfo=None)
    return abs(cdm_tca - event_tca) <= MAX_TCA_DIFFERENCE


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
        if match is None or not _tca_close_enough(match.get("cdm_tca"), raw.get("time_of_closest_approach")):
            enriched.append(event)
            continue
        enriched.append(event.model_copy(update={"raw_data": {**raw, **match}}))

    return enriched
