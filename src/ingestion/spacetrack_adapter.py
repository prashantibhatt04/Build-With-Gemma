"""Space-Track.org conjunction adapter - ROADMAP_TO_PRODUCT.md Phase 1.

An alternative to CelesTrakAdapter, not a replacement for it: CelesTrak
stays the always-available, no-credentials-required default (see
PROJECT_OVERVIEW.md). SpaceTrackAdapter requires a real free Space-Track
account (SPACETRACK_USERNAME/SPACETRACK_PASSWORD - see src/config.py) and
exists specifically to unblock Phase 2's real probability-of-collision
severity, which needs Conjunction Data Messages (and their covariance
data) that only Space-Track provides - CelesTrak's public TLEs alone
can't produce a real Pc.

Implemented as a thin subclass of CelesTrakAdapter rather than a
reimplementation: the entire cross-group screening algorithm (coarse-pass
ranking, fine-pass refinement, refine_top_k, min_cross_group_refine) is
identical regardless of which real data source the raw TLE text came
from - only the fetch mechanism and the `source` label written into
TelemetryEvent differ. This mirrors Phase 14's own reasoning for
extracting tle_source.py: a genuine second real consumer justifies
sharing code instead of duplicating it, not writing a second copy "to be
safe."

Live-verified against a real Space-Track account (ROADMAP_TO_PRODUCT.md
Phase 1's progress-log entry) - real named objects, real sensible
distances, zero NaN, after two real bugs (a missing-name-line format
mismatch, decayed-object propagation failures) were found and fixed
during that verification. See tle_source.fetch_spacetrack_group_text's
own docstring for the details.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from .base_adapter import DataSourceAdapter
from .cdm_enrichment import enrich_conjunction_events_with_pc
from .celestrak_adapter import CelesTrakAdapter, DEFAULT_EXCLUDE_WITHIN_GROUP
from .spacetrack_client import SpaceTrackClient
from .tle_source import DEFAULT_CACHE_DIR, fetch_spacetrack_group_text
from ..schemas import TelemetryEvent

# Space-Track has no server-side "named group" concept the way CelesTrak
# does - the closest real equivalent is an OBJECT_NAME `~~` ("like")
# filter. These patterns are chosen to match the exact same real-world
# populations CelesTrakAdapter's own DEFAULT_GROUPS cover ("stations":
# crewed stations and docked vehicles; "cosmos-2251-debris": real
# fragments from the 2009 Cosmos 2251/Iridium 33 collision), so a
# Space-Track scan and a CelesTrak scan are answering the same real
# question from two different real sources, not two different questions.
DEFAULT_GROUP_NAME_PATTERNS: Mapping[str, str] = {
    "stations": "ISS",
    "cosmos-2251-debris": "COSMOS 2251 DEB",
}


class SpaceTrackAdapter(CelesTrakAdapter):
    SOURCE_LABEL = "spacetrack"

    def __init__(
        self,
        client: SpaceTrackClient,
        group_name_patterns: Mapping[str, str] = DEFAULT_GROUP_NAME_PATTERNS,
        sample_size_per_group: int = 100,
        refine_top_k: int = 80,
        min_cross_group_refine: int = 20,
        lookahead_hours: int = 48,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        exclude_within_group: Sequence[str] = DEFAULT_EXCLUDE_WITHIN_GROUP,
        run_id: Optional[str] = None,
    ):
        self._group_name_patterns = dict(group_name_patterns)

        def _fetch(group: str, cache_dir: Path) -> str:
            pattern = self._group_name_patterns[group]
            return fetch_spacetrack_group_text(pattern, client, cache_dir, group_slug=group)

        super().__init__(
            groups=list(group_name_patterns.keys()),
            sample_size_per_group=sample_size_per_group,
            refine_top_k=refine_top_k,
            min_cross_group_refine=min_cross_group_refine,
            lookahead_hours=lookahead_hours,
            cache_dir=cache_dir,
            exclude_within_group=exclude_within_group,
            run_id=run_id,
            fetch_group_fn=_fetch,
        )


class EnrichedSpaceTrackAdapter(DataSourceAdapter):
    """SpaceTrackAdapter's own conjunction screening, plus a real attempt
    to enrich each result with a real Pc from Space-Track's `cdm_public`
    class (Phase 2 - see src/pc_severity.py and cdm_enrichment.py) before
    the pipeline ever sees it.

    A thin composition, not a subclass - deliberately duck-types
    DataSourceAdapter (fetch_batch(limit) -> list[TelemetryEvent]) so it
    drops into run_once/scripts/scheduler.py exactly like any other real
    adapter, with no changes needed to pipeline.py. Live-verified
    end-to-end: real conjunctions fetched, real CDM lookup attempted, 0
    matches in that specific live test (cdm_public covers the whole real
    catalog, not just the narrow stations/cosmos-2251-debris groups this
    screens - see ROADMAP_TO_PRODUCT.md Phase 2) - every event correctly
    fell through to the distance-threshold path with no errors, exactly
    the intended honest-degrade behavior.
    """

    def __init__(self, client: SpaceTrackClient, **spacetrack_adapter_kwargs):
        self._client = client
        self._adapter = SpaceTrackAdapter(client=client, **spacetrack_adapter_kwargs)

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        events = self._adapter.fetch_batch(limit)
        return enrich_conjunction_events_with_pc(events, self._client)

    @property
    def last_scan_stats(self) -> Optional[dict]:
        return self._adapter.last_scan_stats
