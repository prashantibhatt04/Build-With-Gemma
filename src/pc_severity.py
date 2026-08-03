"""Real probability-of-collision (Pc) severity classification -
ROADMAP_TO_PRODUCT.md Phase 2.

Every conjunction severity everywhere else in this project (see
pipeline.classify_conjunction_severity) is a plain distance threshold -
real, but structurally blind to how WELL-TRACKED either object actually
is. A tight, well-tracked close pass can be far less dangerous than a
loose one at the same distance; Pc is what actually captures that,
because it's derived from both objects' real position/velocity
covariance, not just their predicted separation. TLEs don't carry
covariance at all - only a real Conjunction Data Message (CDM), which
this project gets from Space-Track.org's `cdm_public` class (see
src/ingestion/spacetrack_client.py), carries a real, precomputed `PC`
value the 18th Space Defense Squadron itself calculated - this module
does NOT recompute Pc from a covariance matrix itself (that's a real,
separate numerical-integration problem the full, privilege-gated `cdm`
class exposes more detail about via its own COLLISION_PROBABILITY_METHOD
field, not available on the public class this project actually uses),
it just classifies severity from the real number Space-Track already
provides.

Correction to an earlier documented finding, from real live testing
against a real Space-Track account (ROADMAP_TO_PRODUCT.md Phase 2's
verification): the full `cdm` class genuinely does require registered
owner/operator privileges - confirmed live, a generic account gets a
real `{"error":"Your Class Does Not Exist "}` 400 response querying it,
matching what Space-Track's own operator handbook and NOAA's TraCSS
documentation describe. But Space-Track also publishes a SEPARATE,
lower-detail `cdm_public` class - real current conjunction data
(SAT_1_ID/SAT_2_ID/PC/TCA/MIN_RNG), accessible to any registered
account, no owner/operator status required. Confirmed live: a real
recent row for two real objects (CZ-6A R/B vs. CZ-6A DEB) with a real
PC of 0.0016. This project uses `cdm_public` specifically - not the
full `cdm` class the code originally assumed - so the Pc path here is
real and reachable, not permanently dormant as first believed.
"""
from __future__ import annotations

from typing import Optional

from .schemas import Severity

# The two officially documented anchor points: NASA's own published ISS
# conjunction-assessment process uses Pc >= 1e-4 as its "Red" threshold
# (maneuver strongly considered) and Pc >= 1e-5 as its "Yellow" threshold
# (increased monitoring/evaluation) - these two numbers are the real,
# widely-cited ones, not invented for this project. The WATCH/NOMINAL
# split below (1e-6) is this project's own extension to fill out the
# existing 4-tier Severity enum, in the same "simplified, clearly labeled
# as simplified" spirit as src/decay.py's perigee bands - not itself an
# official third threshold.
PC_CRITICAL_THRESHOLD = 1e-4
PC_WARNING_THRESHOLD = 1e-5
PC_WATCH_THRESHOLD = 1e-6


def classify_pc_severity(collision_probability: float) -> Severity:
    """Deterministic Pc-based severity - not Gemma-derived, matching
    every other classify_*_severity function in pipeline.py."""
    if collision_probability >= PC_CRITICAL_THRESHOLD:
        return Severity.CRITICAL
    if collision_probability >= PC_WARNING_THRESHOLD:
        return Severity.WARNING
    if collision_probability >= PC_WATCH_THRESHOLD:
        return Severity.WATCH
    return Severity.NOMINAL


def extract_cdm_summary(cdm_row: dict) -> Optional[dict]:
    """Pulls the real fields this project actually uses out of one raw
    Space-Track `cdm_public` row (JSON format) into a small, clean dict.
    Field names below (PC, SAT_1_ID/SAT_2_ID, TCA, MIN_RNG, CREATED) are
    `cdm_public`'s real, live-confirmed shape - NOT the full `cdm`
    class's field names (COLLISION_PROBABILITY, SAT1_OBJECT_DESIGNATOR,
    etc.), which this project cannot access (see module docstring).
    Space-Track's JSON API returns every field as a string regardless of
    logical type, so numeric fields are explicitly coerced here rather
    than assumed to already be floats.

    Returns None if the row is missing PC or either object's id -
    defensive against a malformed or partial row, same "don't guess,
    skip cleanly" treatment tle_source.parse_tle_blocks already gives a
    malformed TLE block.
    """
    try:
        collision_probability = float(cdm_row["PC"])
        sat1_id = str(cdm_row["SAT_1_ID"])
        sat2_id = str(cdm_row["SAT_2_ID"])
    except (KeyError, TypeError, ValueError):
        return None

    return {
        "collision_probability": collision_probability,
        # cdm_public has no COLLISION_PROBABILITY_METHOD equivalent -
        # that level of detail is only in the privilege-gated full `cdm`
        # class this project can't reach (see module docstring).
        "collision_probability_method": None,
        "cdm_tca": cdm_row.get("TCA"),
        # MIN_RNG's unit isn't documented on cdm_public itself (unlike
        # the full CDM's MISS_DISTANCE_UNIT field) - Space-Track's own
        # convention elsewhere is meters, and a live value of "66" for a
        # real close approach is consistent with meters, not km, but
        # this is inferred, not confirmed by an explicit units field.
        "cdm_miss_distance_m": cdm_row.get("MIN_RNG"),
        "cdm_creation_date": cdm_row.get("CREATED"),
        "sat1_object_designator": sat1_id,
        "sat2_object_designator": sat2_id,
    }
