"""Tests for src/pc_severity.py - pure functions, no network. Row shapes
below match Space-Track's real `cdm_public` class field names, confirmed
live against a real account (ROADMAP_TO_PRODUCT.md Phase 2) - not the
full `cdm` class's field names, which this project cannot reach.
"""
import pytest

from src.pc_severity import classify_pc_severity, extract_cdm_summary
from src.schemas import Severity


@pytest.mark.parametrize(
    "pc,expected",
    [
        (2e-4, Severity.CRITICAL),
        (1e-4, Severity.CRITICAL),  # boundary, inclusive
        (5e-5, Severity.WARNING),
        (1e-5, Severity.WARNING),  # boundary, inclusive
        (5e-6, Severity.WATCH),
        (1e-6, Severity.WATCH),  # boundary, inclusive
        (1e-7, Severity.NOMINAL),
        (0.0, Severity.NOMINAL),
    ],
)
def test_classify_pc_severity(pc, expected):
    assert classify_pc_severity(pc) == expected


def test_extract_cdm_summary_parses_a_real_shaped_row():
    # A real row shape confirmed live: CZ-6A R/B vs. CZ-6A DEB, PC 0.0016.
    row = {
        "CDM_ID": "1618640977",
        "CREATED": "2026-08-03 12:56:18.000000",
        "TCA": "2026-08-04T03:17:38.003000",
        "MIN_RNG": "66",
        "PC": "0.001598517",
        "SAT_1_ID": "61570",
        "SAT_1_NAME": "CZ-6A R/B",
        "SAT_2_ID": "56239",
        "SAT_2_NAME": "CZ-6A DEB",
    }

    summary = extract_cdm_summary(row)

    assert summary["collision_probability"] == pytest.approx(0.001598517)
    assert summary["collision_probability_method"] is None  # not present on cdm_public
    assert summary["sat1_object_designator"] == "61570"
    assert summary["sat2_object_designator"] == "56239"
    assert summary["cdm_miss_distance_m"] == "66"
    assert summary["cdm_tca"] == "2026-08-04T03:17:38.003000"
    assert summary["cdm_creation_date"] == "2026-08-03 12:56:18.000000"


def test_extract_cdm_summary_returns_none_for_missing_collision_probability():
    assert extract_cdm_summary({"SAT_1_ID": "1", "SAT_2_ID": "2"}) is None


def test_extract_cdm_summary_returns_none_for_unparseable_probability():
    row = {"PC": "not-a-number", "SAT_1_ID": "1", "SAT_2_ID": "2"}
    assert extract_cdm_summary(row) is None


def test_extract_cdm_summary_returns_none_for_missing_designator():
    row = {"PC": "1e-5", "SAT_1_ID": "1"}
    assert extract_cdm_summary(row) is None
