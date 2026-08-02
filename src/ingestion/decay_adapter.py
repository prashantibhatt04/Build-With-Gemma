"""Real orbital decay/re-entry risk screening (Phase 14) - a second real
hazard type alongside conjunctions, using the exact same real TLE data
this project already fetches from CelesTrak. Assesses EACH object
individually (not pairs, unlike CelesTrakAdapter) using
src/decay.assess_decay_risk, ranked by ascending perigee altitude (lowest
- most at-risk - first).

source="celestrak-decay" (not "celestrak") so this is always clearly
distinguishable from conjunction-pair data everywhere it surfaces
downstream (audit log, dashboard table/orbit-plot availability check) -
the raw_data shape also differs (object_id/object_name, singular; no
min_distance_km), which is what pipeline.analyze_node actually keys off
to route a finding to conjunction vs. decay handling.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from skyfield.api import EarthSatellite, load

from ..decay import assess_decay_risk
from ..orbital import parse_norad_id, parse_tle_epoch
from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter
from .tle_source import DEFAULT_CACHE_DIR, fetch_tle_group_text, parse_tle_blocks

# Real fragments from the 2009 Cosmos 2251/Iridium 33 collision - the same
# real debris field CelesTrakAdapter's default already screens for
# conjunctions. Debris fields naturally span a spread of perigee altitudes
# as fragments disperse post-breakup, unlike a tightly-maintained active
# constellation - a reasonable, already-established real source for this,
# not a new one.
DEFAULT_GROUP = "cosmos-2251-debris"


class DecayRiskAdapter(DataSourceAdapter):
    def __init__(
        self,
        group: str = DEFAULT_GROUP,
        sample_size: int = 200,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        run_id: Optional[str] = None,
    ):
        self.group = group
        self.sample_size = sample_size
        self.cache_dir = Path(cache_dir)
        # Same uniqueness reasoning as CelesTrakAdapter/SyntheticCriticalAdapter/
        # HistoricalReplayAdapter - included in every event_id so repeat
        # scans of the same real object don't collide on DecisionLogger's
        # first-match-wins semantics.
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.ts = load.timescale()

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        text = fetch_tle_group_text(self.group, self.cache_dir)
        blocks = parse_tle_blocks(text)[: self.sample_size]

        start_time = datetime.now(timezone.utc)
        assessments = []
        for name, l1, l2 in blocks:
            sat = EarthSatellite(l1, l2, name, self.ts)
            object_id = parse_norad_id(l1)
            epoch_age_hours = (start_time - parse_tle_epoch(l1)).total_seconds() / 3600
            assessments.append(assess_decay_risk(object_id, name, sat, epoch_age_hours))

        # Lowest (most at-risk) perigee first, matching CelesTrakAdapter's
        # closest-first ranking convention for conjunctions.
        assessments.sort(key=lambda a: a["perigee_altitude_km"])

        now = datetime.now(timezone.utc)
        return [
            TelemetryEvent(
                event_id=f"decay-{a['object_id']}-{self.run_id}",
                timestamp=now,
                source="celestrak-decay",
                raw_data=a,
            )
            for a in assessments[:limit]
        ]
