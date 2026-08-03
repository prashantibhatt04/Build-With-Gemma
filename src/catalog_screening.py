"""Apogee/perigee altitude-range overlap filter - ROADMAP_TO_PRODUCT.md
Phase 3: a fast geometric pre-filter so conjunction screening can scale
past a curated sample toward a real full catalog (Space-Track's is
40,000+ objects and growing).

The problem this solves: CelesTrakAdapter/SpaceTrackAdapter's existing
screening (Phase 10) already avoids re-propagating each pair from
scratch, but it still enumerates and coarse-ranks EVERY pair
(itertools.combinations - O(n^2) pairs). That's fine at ~100-200 objects
(a few thousand pairs), but at real catalog scale it's not just slow, the
pair COUNT itself becomes infeasible to even generate (40,000 objects is
~800 million pairs).

The fix is a real, standard technique used by actual conjunction-
screening systems (including CelesTrak's own SOCRATES) as a first pass:
two objects can only possibly conjunct if their orbital altitude ranges
(perigee to apogee) overlap at all - an object confined to a 400-420km
LEO orbit and one confined to GEO (~35,786km) can never be close to each
other, regardless of where either is along its orbit right now, so
there's no reason to spend even a cheap coarse-distance calculation on
that pair. This is a NECESSARY, not sufficient, condition - it only
ELIMINATES pairs that provably can never conjunct; every pair it keeps
still goes through the existing real coarse/fine propagation search
unchanged (see src/orbital.py) to actually confirm anything.

Implemented as an interval-overlap sweep (sort by perigee, sweep with an
active set keyed by apogee), not a nested loop - O(n log n + k) where k
is the number of surviving (altitude-overlapping) pairs, instead of
O(n^2). This is the actual algorithmic change that makes scaling past a
curated sample possible; the physics and severity logic from Phases 1-2
are completely untouched.
"""
from __future__ import annotations

import heapq


def apogee_perigee_overlap_pairs(
    altitude_ranges: list[tuple[float, float]], margin_km: float = 0.0,
) -> list[tuple[int, int]]:
    """altitude_ranges[i] = (perigee_km, apogee_km) for object i (see
    orbital.perigee_apogee_altitude_km). Returns (i, j) index pairs
    (i < j) whose altitude ranges overlap - candidates for the existing
    coarse/fine propagation search, everything else is provably
    unreachable and skipped.

    margin_km pads every range on both ends before checking overlap - a
    real deployment screening over a multi-hour/day lookahead window may
    want a small positive margin (perigee/apogee at TLE epoch aren't
    perfectly constant - they drift slightly under real perturbations
    over that window), trading a few extra false-positive candidate pairs
    for safety against a false negative. Defaults to 0.0 (maximally
    strict/fast) since this project's own lookahead window is a modest 48
    hours and every surviving pair still goes through the real, precise
    propagation search regardless - a margin only matters if drift over
    the window could be large enough to close an already-tight gap, which
    is a judgment call for whoever configures a real deployment, not
    something this function should assume for them.

    Two intervals [a1, a2] and [b1, b2] overlap iff a1 <= b2 and b1 <= a2.
    Sorting by start (perigee) and sweeping with a min-heap of active
    intervals keyed by end (apogee) finds every overlapping pair without
    ever comparing a pair whose altitude ranges are provably disjoint.
    """
    padded = [(perigee - margin_km, apogee + margin_km) for perigee, apogee in altitude_ranges]
    order = sorted(range(len(padded)), key=lambda i: padded[i][0])

    pairs: list[tuple[int, int]] = []
    active: list[tuple[float, int]] = []  # min-heap of (apogee, index)

    for i in order:
        perigee_i, apogee_i = padded[i]
        # Active intervals whose apogee is already below this interval's
        # perigee can never overlap this or any later interval (later
        # intervals only have an equal or greater perigee, since `order`
        # is sorted) - drop them for good rather than re-checking them.
        while active and active[0][0] < perigee_i:
            heapq.heappop(active)
        for _, j in active:
            pairs.append((j, i) if j < i else (i, j))
        heapq.heappush(active, (apogee_i, i))

    return pairs
