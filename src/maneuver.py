"""Deterministic avoidance-maneuver calculator for CRITICAL conjunctions.

This is intentionally simplified displacement math, not real astrodynamics.
analyze_node only gives us a scalar miss distance and a scalar relative
velocity at closest approach - no position/velocity state vectors, no
epoch, no along-track geometry. So compute_avoidance_maneuver() can't model
a real burn; instead it estimates the delta-v that, applied radially
outward and left to act for an assumed maneuver lead time, would grow the
miss distance past the CRITICAL threshold and into a comfortable target
clearance. Good enough for a deterministic "here's a plausible action and
why" signal for the demo; not flight software.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .schemas import ManeuverPlan, VerifiedClearance

# Below this, analyze_node's classify_conjunction_severity() calls it
# CRITICAL. A maneuver is only ever computed for conjunctions already in
# this range.
CRITICAL_THRESHOLD_KM = 5.0

# How far past the threshold to aim for. 30km sits comfortably clear of the
# 25km WARNING/WATCH boundary, so a successful maneuver reads unambiguously
# as "no longer urgent" rather than landing right on a boundary.
BASE_TARGET_CLEARANCE_KM = 30.0

# Simplified safety margin: faster relative closing velocity gets a little
# extra target clearance, since the same absolute miss distance is riskier
# when it's closing faster. Not derived from real covariance/uncertainty
# modeling - just a rough, monotonic heuristic.
VELOCITY_MARGIN_KM_PER_KM_S = 1.0

# Displacement-from-delta-v needs a time for the delta-v to act over. We
# don't have a real time-to-closest-approach here (no epoch/state vector),
# so this is a fixed stand-in lead time, documented as a simplification
# rather than derived physics.
ASSUMED_MANEUVER_LEAD_TIME_S = 6 * 60 * 60  # 6 hours

# verify_maneuver's distance recompute (below) is mathematically guaranteed
# to reproduce target_clearance_km exactly whenever it's called with the
# same min_distance_km the plan was built from - the two formulas are
# algebraic inverses of each other, not independent derivations - so on its
# own it can only ever catch an implementation bug (the two formulas
# drifting out of sync), never an actually-bad maneuver. This bound adds a
# check that CAN genuinely fail: a real collision-avoidance burn is
# typically well under a few m/s, so anything demanding more than this is a
# sign of implausible/corrupted input (e.g. a garbage min_distance_km),
# not a maneuver to wave through as "safe" just because the arithmetic
# is self-consistent.
MAX_PLAUSIBLE_DELTA_V_M_S = 50.0


def compute_avoidance_maneuver(
    object_a: str,
    object_b: str,
    min_distance_km: float,
    relative_velocity_km_s: float,
) -> ManeuverPlan:
    """Compute a simplified avoidance maneuver for a CRITICAL conjunction.

    object_a/object_b identify the two objects (e.g. NORAD IDs or names)
    for logging/traceability - the simplified scalar-only physics here
    doesn't use per-object state, so they aren't consumed in the
    calculation itself.
    """
    target_clearance_km = BASE_TARGET_CLEARANCE_KM + relative_velocity_km_s * VELOCITY_MARGIN_KM_PER_KM_S
    required_displacement_km = target_clearance_km - min_distance_km

    magnitude_delta_v = (required_displacement_km * 1000) / ASSUMED_MANEUVER_LEAD_TIME_S

    return ManeuverPlan(
        direction="radial-outward",
        magnitude_delta_v=magnitude_delta_v,
        target_clearance_km=target_clearance_km,
        computed_at=datetime.now(timezone.utc),
    )


def verify_maneuver(original_min_distance_km: float, maneuver_plan: ManeuverPlan) -> VerifiedClearance:
    """Re-derives the post-maneuver miss distance and checks it's both
    self-consistent and plausible before calling it cleared. Two distinct
    checks, and only one of them is actually independent:

    1. Recompute the resulting distance forward from the plan's delta-v
       and the same assumed lead time, rather than just echoing
       maneuver_plan.target_clearance_km (the number the plan was *solved
       for*). Honest caveat: since compute_avoidance_maneuver's math is
       linear, this recompute is the algebraic inverse of the original
       solve - given the same original_min_distance_km, it is
       *mathematically guaranteed* to land exactly on target_clearance_km.
       That makes it a regression/consistency guard (it would catch the
       two formulas drifting out of sync after a future edit), not an
       independent safety verification on its own.
    2. MAX_PLAUSIBLE_DELTA_V_M_S (below) is the check that's actually
       independent - it uses a real-world plausibility bound that isn't
       already implied by the plan's own arithmetic, so unlike
       (1), it CAN fail: an implausibly large required delta-v, or a
       nonsensical original_min_distance_km, means the input was probably
       bad, and this refuses to call that "cleared" just because the
       recompute is self-consistent.
    """
    displacement_km = (maneuver_plan.magnitude_delta_v * ASSUMED_MANEUVER_LEAD_TIME_S) / 1000
    new_min_distance_km = original_min_distance_km + displacement_km
    plausible = (
        original_min_distance_km > 0
        and maneuver_plan.magnitude_delta_v <= MAX_PLAUSIBLE_DELTA_V_M_S
    )
    cleared = plausible and new_min_distance_km > CRITICAL_THRESHOLD_KM

    return VerifiedClearance(
        new_min_distance_km=new_min_distance_km,
        cleared=cleared,
        verified_at=datetime.now(timezone.utc),
    )


class DeltaVBudgetTracker:
    """Tracks remaining simplified delta-v budget across maneuvers within a
    single tracker's lifetime (by default, one per pipeline run - see
    pipeline.make_decide_node). Not real fuel/mass modeling - just a running
    counter so repeated CRITICAL events can't silently execute an unlimited
    number of maneuvers without anyone noticing."""

    def __init__(self, starting_budget_m_s: float):
        self.remaining_m_s = starting_budget_m_s

    def consume(self, magnitude_delta_v: float) -> bool:
        """Attempts to spend magnitude_delta_v from the budget. Returns True
        and decrements remaining_m_s if there's enough left; returns False
        (budget left untouched) if not."""
        if magnitude_delta_v > self.remaining_m_s:
            return False
        self.remaining_m_s -= magnitude_delta_v
        return True
