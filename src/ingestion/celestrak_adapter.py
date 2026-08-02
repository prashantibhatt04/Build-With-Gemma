"""CelesTrak TLE adapter: real orbital data in, conjunction candidates out."""
from __future__ import annotations

import itertools
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from skyfield.api import EarthSatellite, load

from ..orbital import build_coarse_times, coarse_min_distance, compute_coarse_positions, parse_norad_id, parse_tle_epoch, refine_closest_approach
from ..schemas import TelemetryEvent
from .base_adapter import DataSourceAdapter
from .tle_source import DEFAULT_CACHE_DIR, fetch_tle_group_text, parse_tle_blocks

# Two real, meaningfully different CelesTrak groups by default: crewed
# stations (the asset actually worth protecting) and a real debris field
# (cosmos-2251-debris - fragments from the 2009 Cosmos 2251/Iridium 33
# collision). Screening across both, not just within one, is the point of
# Phase 10 - "is a real active spacecraft at risk from real tracked
# debris?" is the actually motivating collision-avoidance question, not
# debris-vs-debris alone.
DEFAULT_GROUPS = ("stations", "cosmos-2251-debris")

# "stations" contains crewed stations AND their currently-docked visiting
# vehicles (Soyuz, Progress, Cygnus, Crew Dragon, ...) - live-verified
# during development: these sit at ~0km separation from each other since
# they're intentionally, physically attached, which is real, correct
# physics but not a conjunction risk in the operational sense, and it
# crowds out the actually-interesting stations-vs-debris results at the
# top of the ranking. Pairs within the same excluded group are skipped
# entirely rather than screened and then filtered post-hoc, so they never
# cost fine-pass refinement budget either. Cross-group pairs (an asset vs.
# real debris) and within-debris-group pairs are unaffected and still
# fully screened - debris fragments are genuinely dispersed, not docked.
DEFAULT_EXCLUDE_WITHIN_GROUP = ("stations",)

COARSE_STEP_MINUTES = 5
FINE_STEP_SECONDS = 10


class CelesTrakAdapter(DataSourceAdapter):
    """Fetches real TLE data from one or more CelesTrak groups and screens
    every pairwise conjunction across the combined pool - real objects,
    not staged ones.

    Naively running the full two-pass closest-approach search
    (orbital.find_closest_approach) on every pair does not scale: it
    recomputes each satellite's coarse-pass propagation from scratch for
    every pair it appears in, which is O(pairs) expensive propagation
    calls instead of O(satellites). Measured on a real 60-object sample,
    that took ~9.6s for 1770 pairs - already too slow for a live demo step,
    and scaling to hundreds of real objects would take minutes. Instead:

    1. Compute each satellite's coarse-pass position ONCE
       (orbital.compute_coarse_positions) and reuse it for every pair it's
       in - this alone cut the same 60-object/1770-pair coarse screen from
       ~9.6s to ~0.02s in testing.
    2. Rank ALL pairs by that cheap coarse distance. The coarse pass can
       only ever OVERESTIMATE the true closest approach (see orbital.py's
       module docstring), so this ranking is a sound proxy for the true
       ranking - it will not miss a genuinely close pair.
    3. Only run the expensive fine-pass refinement
       (orbital.refine_closest_approach) on the `refine_top_k` closest
       candidates by coarse distance, not all of them - there's no reason
       to spend precision effort on pairs that are obviously nowhere near
       each other.

    Live-verified during development: a dense single-origin debris field
    (e.g. cosmos-2251-debris - many fragments from one breakup, naturally
    close to each other) can fill the entire refine_top_k ranking on its
    own, crowding out cross-group pairs even though answering "is my real
    asset at risk from real debris" - the actual motivating question here
    - specifically needs at least some of those refined. `min_cross_group_refine`
    guarantees that many cross-group pairs (by the same coarse ranking)
    get refined too, as a union with the top-refine_top_k overall, rather
    than leaving cross-group representation to chance.

    The raw TLE text for each group is cached to disk for up to an hour
    (CelesTrak asks users to go easy on request volume, and there's no
    reason to refetch on every call during dev/testing).
    """

    def __init__(
        self,
        groups: Sequence[str] = DEFAULT_GROUPS,
        sample_size_per_group: int = 100,
        refine_top_k: int = 80,
        min_cross_group_refine: int = 20,
        lookahead_hours: int = 48,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        exclude_within_group: Sequence[str] = DEFAULT_EXCLUDE_WITHIN_GROUP,
        run_id: Optional[str] = None,
    ):
        self.groups = list(groups)
        self.sample_size_per_group = sample_size_per_group
        self.refine_top_k = refine_top_k
        self.min_cross_group_refine = min_cross_group_refine
        self.lookahead_hours = lookahead_hours
        self.cache_dir = Path(cache_dir)
        self.exclude_within_group = set(exclude_within_group)
        # Included in every event_id this instance produces (see
        # fetch_batch) - the SAME real conjunction (same object pair) can
        # easily recur across separate scans (e.g. two dashboard clicks
        # within the same 1h TLE cache window), and DecisionLogger's
        # find_entry/mark_reviewed/approve_maneuver match an event_id's
        # FIRST logged occurrence - without this, a second scan's entry
        # would be indistinguishable from an already-resolved earlier one,
        # so approving/reviewing "this scan's" pending entry could silently
        # resolve a stale one instead. Same fix already applied to
        # SyntheticCriticalAdapter and HistoricalReplayAdapter.
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.ts = load.timescale()
        # Populated by _rank_conjunctions() - lets callers (e.g.
        # scripts/run_demo.py) report what was actually screened without
        # threading extra return values through fetch_batch's interface.
        self.last_scan_stats: Optional[dict] = None

    def _fetch_sample(self) -> list[tuple[str, str, str, str]]:
        """Returns (name, tle_line1, tle_line2, source_group) for up to
        sample_size_per_group real objects from each configured group."""
        sample = []
        for group in self.groups:
            text = fetch_tle_group_text(group, self.cache_dir)
            group_sample = parse_tle_blocks(text)[: self.sample_size_per_group]
            sample.extend((name, l1, l2, group) for name, l1, l2 in group_sample)
        return sample

    def _rank_conjunctions(self) -> list[dict]:
        sample = self._fetch_sample()
        satellites = [EarthSatellite(l1, l2, name, self.ts) for name, l1, l2, _ in sample]
        norad_ids = [parse_norad_id(l1) for _, l1, _, _ in sample]
        names = [name for name, _, _, _ in sample]
        epochs = [parse_tle_epoch(l1) for _, l1, _, _ in sample]
        source_groups = [group for _, _, _, group in sample]

        start_time = datetime.now(timezone.utc)
        coarse_times = build_coarse_times(start_time, self.lookahead_hours, COARSE_STEP_MINUTES)
        # The expensive step, done ONCE per satellite - see class docstring.
        coarse_positions = [
            compute_coarse_positions(sat, self.ts, coarse_times) for sat in satellites
        ]

        all_pairs = [
            (i, j) for i, j in itertools.combinations(range(len(satellites)), 2)
            # Skip pairs within an excluded same-role group (default:
            # "stations", whose docked vehicles sit at ~0km on purpose -
            # see DEFAULT_EXCLUDE_WITHIN_GROUP) before they ever cost
            # coarse-ranking or refinement effort.
            if not (source_groups[i] == source_groups[j] and source_groups[i] in self.exclude_within_group)
        ]
        pairs_by_coarse_distance = sorted(
            (
                (i, j, *coarse_min_distance(coarse_positions[i], coarse_positions[j]))
                for i, j in all_pairs
            ),
            key=lambda row: row[2],  # coarse distance km, ascending
        )

        # Union of the globally closest refine_top_k pairs and the closest
        # min_cross_group_refine CROSS-group pairs specifically - a dense
        # single-origin debris field can otherwise fill the entire overall
        # ranking on its own (see class docstring), silently starving the
        # "asset vs. real debris" pairs that are the actual point here.
        top_overall = pairs_by_coarse_distance[: self.refine_top_k]
        top_cross_group = [
            row for row in pairs_by_coarse_distance if source_groups[row[0]] != source_groups[row[1]]
        ][: self.min_cross_group_refine]
        seen_pairs: set[tuple[int, int]] = set()
        to_refine = []
        for row in itertools.chain(top_overall, top_cross_group):
            pair_key = (row[0], row[1])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                to_refine.append(row)

        conjunctions = []
        for i, j, _coarse_distance_km, coarse_min_idx in to_refine:
            result = refine_closest_approach(
                satellites[i], satellites[j], self.ts, coarse_times[coarse_min_idx],
                COARSE_STEP_MINUTES, FINE_STEP_SECONDS,
            )
            # How stale the older of the two TLEs is, in hours - a real
            # (if simplified) signal for AnomalyFinding.confidence rather
            # than a flat constant. See pipeline.compute_confidence.
            epoch_age_hours = max(
                (start_time - epochs[i]).total_seconds() / 3600,
                (start_time - epochs[j]).total_seconds() / 3600,
            )
            conjunctions.append({
                "object_a_id": norad_ids[i],
                "object_a_name": names[i],
                "object_a_group": source_groups[i],
                "object_b_id": norad_ids[j],
                "object_b_name": names[j],
                "object_b_group": source_groups[j],
                "min_distance_km": result["min_distance_km"],
                "time_of_closest_approach": result["time_of_closest_approach"].isoformat(),
                "relative_velocity_km_s": result["relative_velocity_km_s"],
                "tle_epoch_age_hours": epoch_age_hours,
            })

        conjunctions.sort(key=lambda c: c["min_distance_km"])
        self.last_scan_stats = {
            "groups": list(self.groups),
            "total_objects": len(satellites),
            "total_pairs_screened": len(all_pairs),
            "pairs_refined": len(to_refine),
            "cross_group_pairs_refined": sum(
                1 for i, j, *_ in to_refine if source_groups[i] != source_groups[j]
            ),
        }
        return conjunctions

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        conjunctions = self._rank_conjunctions()
        now = datetime.now(timezone.utc)
        return [
            TelemetryEvent(
                event_id=f"conj-{c['object_a_id']}-{c['object_b_id']}-{self.run_id}",
                timestamp=now,
                source="celestrak",
                raw_data=c,
            )
            for c in conjunctions[:limit]
        ]
