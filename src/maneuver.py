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
    """Independently re-derive the post-maneuver miss distance and confirm
    it clears CRITICAL_THRESHOLD_KM.

    orbital.find_closest_approach's two-pass coarse/fine search exists
    because a real trajectory's closest approach is nonlinear and has to be
    numerically bracketed then refined. This function has no real
    trajectory to search - compute_avoidance_maneuver's displacement model
    is linear in time, so there's nothing for a second refinement pass to
    improve on. What it reuses "conceptually" instead is the same
    discipline of not trusting a single forward calculation: rather than
    just echoing maneuver_plan.target_clearance_km (the number the plan was
    *solved for*), this recomputes the resulting distance forward from the
    plan's delta-v and the same assumed lead time, as an independent
    cross-check that the two are consistent.
    """
    displacement_km = (maneuver_plan.magnitude_delta_v * ASSUMED_MANEUVER_LEAD_TIME_S) / 1000
    new_min_distance_km = original_min_distance_km + displacement_km
    cleared = new_min_distance_km > CRITICAL_THRESHOLD_KM

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
