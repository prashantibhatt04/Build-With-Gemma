"""Generic telemetry / anomaly / decision schemas shared across the pipeline.

These are intentionally idea-agnostic: they describe the *shape* of a
telemetry event, an anomaly finding, and a decision, without assuming
what the underlying spacecraft, mission, or sensor actually is.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    NOMINAL = "nominal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


class Action(str, Enum):
    CONTINUE = "continue"
    HOLD = "hold"
    ABORT = "abort"


class TelemetryEvent(BaseModel):
    event_id: str
    timestamp: datetime
    source: str
    raw_data: dict


class AnomalyFinding(BaseModel):
    event_id: str
    severity: Severity
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class ManeuverPlan(BaseModel):
    """A deterministic, simplified avoidance-maneuver stand-in computed for
    CRITICAL-severity conjunctions. See src/maneuver.py for the (simplified,
    non-astrodynamics) math behind these numbers."""

    direction: str
    magnitude_delta_v: float = Field(gt=0, description="Simplified delta-v magnitude, in m/s.")
    target_clearance_km: float = Field(gt=0)
    computed_at: datetime


class VerifiedClearance(BaseModel):
    """Independent re-check of a ManeuverPlan's effect: re-derives the
    post-maneuver miss distance forward from the plan's delta-v (rather
    than trusting its target_clearance_km verbatim) and confirms it clears
    the CRITICAL threshold. See src/maneuver.py:verify_maneuver."""

    new_min_distance_km: float
    cleared: bool
    verified_at: datetime


class Decision(BaseModel):
    action: Action
    rationale: str
    made_at: datetime
    # Populated only for CRITICAL severity, where an avoidance maneuver is
    # actually computed and verified rather than just recommended.
    maneuver_plan: Optional[ManeuverPlan] = None
    verified_clearance: Optional[VerifiedClearance] = None
    # True when a CRITICAL maneuver was calculated but NOT executed because
    # it would have exceeded the remaining delta-v budget (see
    # src/maneuver.py:DeltaVBudgetTracker) - in that case verified_clearance
    # stays None, since nothing was actually applied to verify.
    budget_insufficient: bool = False


class GemmaProvenance(BaseModel):
    """Where a piece of generated text actually came from: the real model,
    or the deterministic fallback used when Gemma was unreachable."""

    source: Literal["gemma", "fallback"]
    model_used: str
    latency_ms: float


class DecisionLogEntry(BaseModel):
    telemetry: TelemetryEvent
    finding: AnomalyFinding
    decision: Decision
    # analyze_node's Gemma call for finding.description. None when no call was
    # attempted at all (e.g. DummyAdapter's non-conjunction placeholder path).
    description_provenance: Optional[GemmaProvenance] = None
    # decide_node's Gemma call for decision.rationale. Always attempted.
    rationale_provenance: GemmaProvenance
    human_reviewed: bool = False
    human_reviewed_at: Optional[datetime] = None
    # Who performed the review that set human_reviewed - see
    # src/logging_utils.py:DecisionLogger.mark_reviewed.
    reviewed_by: Optional[str] = None
