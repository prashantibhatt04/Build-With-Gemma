"""Unit tests for the simplified deterministic maneuver math. Synthetic
inputs only - no network calls, no real orbital state vectors."""
from datetime import datetime, timezone

import pytest

from src.maneuver import (
    CRITICAL_THRESHOLD_KM,
    MAX_PLAUSIBLE_DELTA_V_M_S,
    DeltaVBudgetTracker,
    compute_avoidance_maneuver,
    verify_maneuver,
)
from src.schemas import ManeuverPlan, VerifiedClearance


def test_compute_avoidance_maneuver_returns_plan_for_critical_range():
    plan = compute_avoidance_maneuver(
        object_a="33765", object_b="33818",
        min_distance_km=3.0, relative_velocity_km_s=5.0,
    )

    assert isinstance(plan, ManeuverPlan)
    assert plan.direction == "radial-outward"
    assert plan.magnitude_delta_v > 0
    assert plan.target_clearance_km > 25.0  # comfortably past WARNING/WATCH boundary
    assert plan.target_clearance_km > 3.0  # clears the starting distance
    assert (datetime.now(timezone.utc) - plan.computed_at).total_seconds() < 5


def test_returns_none_when_min_distance_already_beyond_target_clearance():
    """Real bug this closes: the Pc-based CRITICAL path (pc_severity.py)
    can classify a conjunction CRITICAL even at a large miss distance -
    tight covariance, not proximity, drives it. The old code computed a
    negative required_displacement_km here, which made ManeuverPlan's
    magnitude_delta_v > 0 validation raise uncaught. There's nothing this
    displacement-only model can compute once we're already past its own
    target clearance, so it must return None instead."""
    plan = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=250.0, relative_velocity_km_s=13.788,
    )

    assert plan is None


def test_returns_a_plan_right_up_to_the_boundary():
    """min_distance_km exactly at target_clearance_km is the boundary the
    None-check sits on - confirms it's a real boundary, not an off-by-one
    that also swallows legitimate close-range cases."""
    plan = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=29.999, relative_velocity_km_s=0.0,
    )

    assert plan is not None
    assert plan.magnitude_delta_v > 0


def test_larger_displacement_for_smaller_starting_distance():
    closer = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=0.5, relative_velocity_km_s=5.0,
    )
    farther = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=4.5, relative_velocity_km_s=5.0,
    )

    assert closer.magnitude_delta_v > farther.magnitude_delta_v


def test_larger_displacement_for_faster_relative_velocity():
    slow = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=3.0, relative_velocity_km_s=1.0,
    )
    fast = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=3.0, relative_velocity_km_s=10.0,
    )

    assert fast.magnitude_delta_v > slow.magnitude_delta_v
    assert fast.target_clearance_km > slow.target_clearance_km


def test_verify_maneuver_reports_cleared_with_new_distance():
    original_min_distance_km = 2.0
    plan = compute_avoidance_maneuver(
        object_a="a", object_b="b",
        min_distance_km=original_min_distance_km, relative_velocity_km_s=3.0,
    )

    verified = verify_maneuver(original_min_distance_km, plan)

    assert isinstance(verified, VerifiedClearance)
    assert verified.cleared is True
    assert verified.new_min_distance_km > CRITICAL_THRESHOLD_KM
    # Re-derived forward from delta-v using the same linear model the plan
    # was solved from - given the same original_min_distance_km, this is
    # mathematically guaranteed to land exactly on target_clearance_km (see
    # verify_maneuver's docstring for why that's a consistency guard, not
    # independent verification on its own - MAX_PLAUSIBLE_DELTA_V_M_S below
    # is the part of the check that's actually independent).
    assert verified.new_min_distance_km == pytest.approx(plan.target_clearance_km)
    assert (datetime.now(timezone.utc) - verified.verified_at).total_seconds() < 5


def test_verify_maneuver_flags_implausibly_large_delta_v_as_not_cleared():
    """The distance recompute alone can never independently catch a bad
    maneuver (see docstring/comment above) - this proves the ADDED
    plausibility bound actually can fail. Plan constructed directly,
    bypassing compute_avoidance_maneuver, to simulate a maneuver_plan that
    reached verify_maneuver already corrupted (e.g. from a bug elsewhere)."""
    implausible_plan = ManeuverPlan(
        direction="radial-outward",
        magnitude_delta_v=MAX_PLAUSIBLE_DELTA_V_M_S + 1,
        target_clearance_km=30.0,
        computed_at=datetime.now(timezone.utc),
    )

    verified = verify_maneuver(2.0, implausible_plan)

    assert verified.cleared is False
    # Still reports the (now-meaningless) recomputed distance - cleared is
    # what callers must check, not new_min_distance_km's sign alone.
    assert verified.new_min_distance_km > CRITICAL_THRESHOLD_KM


def test_verify_maneuver_flags_nonpositive_original_distance_as_not_cleared():
    plan = compute_avoidance_maneuver(
        object_a="a", object_b="b", min_distance_km=2.0, relative_velocity_km_s=3.0,
    )

    verified = verify_maneuver(0.0, plan)

    assert verified.cleared is False


@pytest.mark.parametrize("original_min_distance_km", [0.1, 1.0, 2.5, 4.9])
def test_verify_maneuver_new_distance_always_exceeds_original(original_min_distance_km):
    plan = compute_avoidance_maneuver(
        object_a="a", object_b="b",
        min_distance_km=original_min_distance_km, relative_velocity_km_s=4.0,
    )

    verified = verify_maneuver(original_min_distance_km, plan)

    assert verified.new_min_distance_km > original_min_distance_km


def test_budget_tracker_consumes_when_sufficient():
    tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)

    spent = tracker.consume(1.5)

    assert spent is True
    assert tracker.remaining_m_s == pytest.approx(3.5)


def test_budget_tracker_refuses_and_leaves_budget_untouched_when_insufficient():
    tracker = DeltaVBudgetTracker(starting_budget_m_s=1.0)

    spent = tracker.consume(1.5)

    assert spent is False
    assert tracker.remaining_m_s == pytest.approx(1.0)  # untouched, not partially spent


def test_budget_tracker_depletes_across_multiple_maneuvers():
    tracker = DeltaVBudgetTracker(starting_budget_m_s=3.0)

    assert tracker.consume(1.5) is True
    assert tracker.consume(1.4) is True
    # Only 0.1 m/s left - this one should be refused.
    assert tracker.consume(0.5) is False
    assert tracker.remaining_m_s == pytest.approx(0.1)
