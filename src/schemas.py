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


class ManeuverApproval(BaseModel):
    """Records how a CRITICAL maneuver was authorized to proceed.

    mode="autonomous": the configured Gemma backend is "ollama" (local) -
    in this system, local-only is used as a stand-in for "ground control is
    currently unreachable" (e.g. communication blackout, light-delay). A
    deterministic physics check independently verifies the maneuver is
    safe first (see src/maneuver.py:verify_maneuver), then Gemma itself
    gets those same numbers and issues a real GO/NO-GO verdict, standing
    in for the unavailable human (see src/pipeline.py:_maneuver_veto_check).
    Gemma can only make this MORE conservative than the physics check, never
    less: approved=True means physics verified it safe AND Gemma affirmed
    it (approved_by stays None - no human, autonomous by construction);
    approved=False means Gemma vetoed an already-verified-safe maneuver, or
    its response couldn't be parsed into a clear verdict (the fail-safe
    default is NO-GO, not a free pass) - approved_by identifies Gemma as
    the vetoing party in that case. Gemma being unreachable is NOT treated
    as a veto: an LLM outage alone shouldn't block a maneuver the
    deterministic physics has already verified safe, so that case falls
    back to today's physics-only approval (approved=True, approved_by=None).

    mode="human": the configured backend is "api" (cloud) - a stand-in for
    "ground control is reachable" - so a real person must explicitly
    confirm before the maneuver is treated as executed. See
    src/logging_utils.py:DecisionLogger.approve_maneuver.
    """

    mode: Literal["autonomous", "human"]
    approved: bool
    # None for a physics-affirmed autonomous approval (no human involved);
    # identifies Gemma when mode="autonomous" but it vetoed the maneuver
    # (see pipeline._maneuver_veto_check); the human's name when mode="human".
    approved_by: Optional[str] = None
    approved_at: datetime
    reason: str


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
    # True when a CRITICAL maneuver was calculated and the budget allows it,
    # but it has NOT yet been executed/verified because it's waiting on a
    # human to approve it (cloud backend - see ManeuverApproval). While this
    # is True, verified_clearance stays None; once resolved (see
    # DecisionLogger.approve_maneuver), this flips to False and
    # maneuver_approval gets populated either way (approved or rejected).
    awaiting_human_approval: bool = False
    maneuver_approval: Optional[ManeuverApproval] = None


class GemmaProvenance(BaseModel):
    """Where a piece of generated text actually came from: the real model,
    or the deterministic fallback used when Gemma was unreachable."""

    source: Literal["gemma", "fallback"]
    model_used: str
    latency_ms: float
    # The real prompt text sent to Gemma for this call - not just the
    # response, the actual question. None only for provenance built
    # without going through pipeline._call_gemma_with_provenance (e.g.
    # test fixtures that don't care about it); every real Gemma call this
    # project makes populates it, for genuine audit reconstruction of
    # exactly what was asked, not just what came back.
    prompt: Optional[str] = None


class DecisionLogEntry(BaseModel):
    telemetry: TelemetryEvent
    finding: AnomalyFinding
    decision: Decision
    # analyze_node's Gemma call for finding.description. None when no call was
    # attempted at all (e.g. DummyAdapter's non-conjunction placeholder path).
    description_provenance: Optional[GemmaProvenance] = None
    # decide_node's Gemma call for decision.rationale. Always attempted.
    rationale_provenance: GemmaProvenance
    # decide_node's Gemma veto-check call (see ManeuverApproval, mode=
    # "autonomous") for a CRITICAL conjunction on the local/autonomous path.
    # None whenever no veto check was attempted: non-CRITICAL, cloud/api
    # backend (human approval instead), or budget-insufficient.
    veto_provenance: Optional[GemmaProvenance] = None
    human_reviewed: bool = False
    human_reviewed_at: Optional[datetime] = None
    # Who performed the review that set human_reviewed - see
    # src/logging_utils.py:DecisionLogger.mark_reviewed.
    reviewed_by: Optional[str] = None
