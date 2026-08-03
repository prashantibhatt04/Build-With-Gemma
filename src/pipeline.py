"""LangGraph pipeline skeleton: ingest -> analyze -> decide -> log.

analyze_node classifies conjunction severity deterministically from
min_distance_km and uses Gemma only to generate the human-readable
description. decide_node maps severity to action deterministically and
uses Gemma only to explain the recommendation. Non-conjunction
telemetry (e.g. DummyAdapter's generic payload) falls back to
placeholder behavior in both nodes.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .alerting import send_critical_alert
from .gemma_client import GemmaClient, GemmaClientError
from .ingestion.base_adapter import DataSourceAdapter, DummyAdapter
from .logging_utils import DecisionLogger
from .maneuver import DeltaVBudgetTracker, compute_avoidance_maneuver, verify_maneuver
from .pc_severity import classify_pc_severity
from .schemas import (
    Action,
    AnomalyFinding,
    Decision,
    DecisionLogEntry,
    GemmaProvenance,
    ManeuverApproval,
    ManeuverPlan,
    Severity,
    TelemetryEvent,
    VerifiedClearance,
)


class PipelineState(TypedDict):
    telemetry: TelemetryEvent
    finding: Optional[AnomalyFinding]
    decision: Optional[Decision]
    log_path: Optional[str]
    description_provenance: Optional[GemmaProvenance]
    rationale_provenance: Optional[GemmaProvenance]
    veto_provenance: Optional[GemmaProvenance]


def _describe_model_used(client: GemmaClient) -> str:
    """Usually just the configured model name for whichever backend actually
    responded. Ollama and the hosted API use different model-naming schemes
    (see config.Settings), so this looks up the right one rather than
    assuming gemma_model applies to both. If GemmaClient's cross-backend
    fallback kicked in (see gemma_client.py), last_backend_used differs from
    the configured backend - make that visible in provenance rather than
    silently reporting the primary backend that actually failed."""
    backend_used = getattr(client, "last_backend_used", None) or getattr(client.settings, "gemma_backend", None)
    model_by_backend = {
        "ollama": getattr(client.settings, "gemma_model", None),
        "api": getattr(client.settings, "gemma_model_api", None),
    }
    model_name = model_by_backend.get(backend_used) or getattr(client.settings, "gemma_model", "unknown")

    configured_backend = getattr(client.settings, "gemma_backend", None)
    if backend_used and configured_backend and backend_used != configured_backend:
        return f"{model_name} ({backend_used}, fallback from {configured_backend})"
    return model_name


# Below this length, a final line is treated as a throwaway remark (e.g.
# "Ok, let's go.", "Understood.") rather than real content - see
# _extract_final_answer. A real one-sentence rationale/description is
# consistently well over this in practice.
_MIN_SUBSTANTIVE_LINE_LENGTH = 20


def _extract_final_answer(text: str) -> str:
    """Some backends/models respond with their full reasoning trace before
    the actual answer - draft attempts, self-checklists, "wait, let me
    reconsider", labeled sections like "Final Polish:" - instead of just the
    answer (observed with the hosted API; local Ollama's gemma4:e4b responds
    cleanly). The formatting of that reasoning isn't consistent (sometimes
    blank-line-separated paragraphs, sometimes a label and the answer on
    consecutive single-newline-separated lines), but in every case observed,
    the real answer ends up as one of the LAST non-empty lines of the
    response. Occasionally the model's literal last line is a short
    throwaway remark instead (e.g. "Ok, let's go.") with the real answer one
    line above it - so this searches backward for the last line that's
    actually substantive, not just the literal last line. For an
    already-clean single-line response (the normal Ollama case), this is a
    no-op - there's only one line to consider."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return text.strip()
    for line in reversed(lines):
        if len(line) >= _MIN_SUBSTANTIVE_LINE_LENGTH:
            return line
    return lines[-1]  # nothing met the bar - literal last line is the best we have


def _call_gemma_with_provenance(
    client: GemmaClient, prompt: str, system: str, fallback_text: str,
    postprocess: Callable[[str], str] = _extract_final_answer,
    format: Optional[dict] = None,
) -> tuple[str, GemmaProvenance]:
    """Calls Gemma and always returns (text, provenance), never raises -
    falls back to fallback_text on GemmaClientError. postprocess is applied
    to a successful raw response before returning it - defaults to
    _extract_final_answer (strip chain-of-thought down to the real answer).
    The maneuver veto-check caller passes a no-op instead: it needs the
    FULL response (JSON, or free text if format wasn't honored - see
    _maneuver_veto_check) rather than just the last substantive line.
    format, when given, is passed through to GemmaClient.generate() for
    real JSON-schema-constrained decoding (Ollama-only - see
    GemmaClient.generate's docstring for the cross-backend-fallback
    caveat). The returned GemmaProvenance always carries the real prompt
    text that was sent (regardless of whether Gemma actually answered or
    the deterministic fallback kicked in) - the point of logging
    provenance at all is to be able to reconstruct exactly what was
    asked, not just what came back."""
    start = time.monotonic()
    try:
        # format is only passed through when actually requested, not as an
        # always-present format=None kwarg - keeps every OTHER call site
        # (description/rationale/RAG-answer generation, and every
        # duck-typed fake GemmaClient in the test suite that predates
        # format support) working unchanged, since they never opt in.
        generate_kwargs = {"prompt": prompt, "system": system}
        if format is not None:
            generate_kwargs["format"] = format
        text = postprocess(client.generate(**generate_kwargs))
        source = "gemma"
        model_used = _describe_model_used(client)
    except GemmaClientError:
        text = fallback_text
        source = "fallback"
        # Report the model for whichever backend was actually CONFIGURED
        # (both attempts failed, so nothing "responded", but the fallback
        # text is standing in for that backend specifically) - not always
        # gemma_model, which is only the Ollama tag and is simply wrong
        # for a fallback that occurred while GEMMA_BACKEND=api.
        if getattr(client.settings, "gemma_backend", None) == "api":
            model_used = getattr(client.settings, "gemma_model_api", None) or client.settings.gemma_model
        else:
            model_used = client.settings.gemma_model
    latency_ms = (time.monotonic() - start) * 1000
    provenance = GemmaProvenance(
        source=source, model_used=model_used, latency_ms=latency_ms, prompt=prompt,
    )
    return text, provenance


def compute_confidence(raw: dict) -> float:
    """Deterministic confidence heuristic. Uses TLE epoch age (how stale the
    tracking data is) as a real, if simplified, staleness signal when the
    adapter provides one - not a real orbital covariance/uncertainty model,
    just a monotonic, bounded stand-in so confidence stops being a flat
    constant for conjunction-shaped events. Falls back to a clearly labeled
    placeholder for events with no epoch signal at all (e.g. adapters or
    test fixtures that don't carry tle_epoch_age_hours)."""
    age_hours = raw.get("tle_epoch_age_hours")
    if age_hours is None:
        return 0.8  # PLACEHOLDER: no staleness signal available for this event
    if age_hours <= 24:
        return 0.9
    if age_hours <= 72:
        return 0.8
    if age_hours <= 168:  # 1 week
        return 0.6
    return 0.4


def classify_conjunction_severity(min_distance_km: float) -> Severity:
    """Deterministic distance-based severity. Not Gemma-derived - this
    needs to be reliable, so it's a plain threshold check."""
    if min_distance_km < 5:
        return Severity.CRITICAL
    if min_distance_km < 25:
        return Severity.WARNING
    if min_distance_km < 100:
        return Severity.WATCH
    return Severity.NOMINAL


def classify_conjunction_finding(raw: dict) -> tuple[Severity, str]:
    """Dispatches a conjunction to real Pc-based severity when a real
    Space-Track CDM was matched to it (src/ingestion/cdm_enrichment.py
    populates `collision_probability` in raw_data when that happens),
    otherwise falls back to the existing distance threshold - see
    ROADMAP_TO_PRODUCT.md Phase 2. Returns (severity, severity_source) so
    callers can record which path actually produced the severity, never
    silently blending the two."""
    collision_probability = raw.get("collision_probability")
    if collision_probability is not None:
        return classify_pc_severity(collision_probability), "probability-of-collision"
    return classify_conjunction_severity(raw["min_distance_km"]), "distance-threshold"


def classify_decay_severity(perigee_altitude_km: float) -> Severity:
    """Deterministic perigee-altitude-based severity for the decay hazard
    (see src/decay.py) - not Gemma-derived, matching
    classify_conjunction_severity's design. Real, well-established LEO
    decay bands: below these altitudes, real objects reliably reenter
    within roughly the stated timeframe, regardless of other factors -
    not a precise reentry-time predictor, just a reliable threshold."""
    if perigee_altitude_km < 200:
        return Severity.CRITICAL  # days to weeks
    if perigee_altitude_km < 300:
        return Severity.WARNING  # weeks to months
    if perigee_altitude_km < 500:
        return Severity.WATCH  # months to a few years
    return Severity.NOMINAL


def classify_attitude_severity(pointing_error_deg: float) -> Severity:
    """Deterministic pointing-error-based severity for the attitude/
    pointing-loss hazard (see src/ingestion/attitude_adapter.py - this
    hazard type is synthetic-only, no real data source exists) - not
    Gemma-derived, matching classify_conjunction_severity/
    classify_decay_severity's design. A few degrees of pointing error is
    within normal control-loop tolerance for most missions; tens of
    degrees indicates the spacecraft has likely dropped into a safe/
    tumble mode and lost fine attitude control."""
    if pointing_error_deg >= 45:
        return Severity.CRITICAL
    if pointing_error_deg >= 15:
        return Severity.WARNING
    if pointing_error_deg >= 5:
        return Severity.WATCH
    return Severity.NOMINAL


def _conjunction_description(client: GemmaClient, raw: dict) -> tuple[str, GemmaProvenance]:
    prompt = (
        "Summarize this satellite conjunction (close approach) for a mission "
        "controller in 1-2 plain-language sentences:\n"
        f"Object A: {raw['object_a_name']} (NORAD {raw['object_a_id']})\n"
        f"Object B: {raw['object_b_name']} (NORAD {raw['object_b_id']})\n"
        f"Minimum distance: {raw['min_distance_km']:.3f} km\n"
        f"Relative velocity: {raw['relative_velocity_km_s']:.3f} km/s\n"
        f"Time of closest approach: {raw['time_of_closest_approach']}"
    )
    fallback_text = (
        f"{raw['object_a_name']} and {raw['object_b_name']} approach within "
        f"{raw['min_distance_km']:.3f} km at {raw['time_of_closest_approach']} "
        f"(relative velocity {raw['relative_velocity_km_s']:.3f} km/s). "
        "Gemma was unreachable, so this is a fallback summary."
    )
    return _call_gemma_with_provenance(
        client,
        prompt=prompt,
        system="You are assisting a mission controller by summarizing orbital "
               "conjunction risk in plain language. Be concise and factual. "
               "Respond in plain prose only - no LaTeX, no markdown formatting, "
               "no special notation.",
        fallback_text=fallback_text,
    )


def _decay_description(client: GemmaClient, raw: dict) -> tuple[str, GemmaProvenance]:
    prompt = (
        "Summarize this satellite's orbital decay risk for a mission "
        "controller in 1-2 plain-language sentences:\n"
        f"Object: {raw['object_name']} (NORAD {raw['object_id']})\n"
        f"Perigee altitude: {raw['perigee_altitude_km']:.1f} km\n"
        f"Apogee altitude: {raw['apogee_altitude_km']:.1f} km\n"
        f"BSTAR drag term: {raw['bstar']:.3e}"
    )
    fallback_text = (
        f"{raw['object_name']} has a perigee altitude of "
        f"{raw['perigee_altitude_km']:.1f} km (apogee {raw['apogee_altitude_km']:.1f} km). "
        "Gemma was unreachable, so this is a fallback summary."
    )
    return _call_gemma_with_provenance(
        client,
        prompt=prompt,
        system="You are assisting a mission controller by summarizing orbital "
               "decay/re-entry risk in plain language. Be concise and factual - "
               "this is a simplified, perigee-altitude-based assessment, not a "
               "precise reentry-time prediction, so don't state a specific "
               "reentry date. Respond in plain prose only - no LaTeX, no "
               "markdown formatting, no special notation.",
        fallback_text=fallback_text,
    )


def _attitude_description(client: GemmaClient, raw: dict) -> tuple[str, GemmaProvenance]:
    prompt = (
        "Summarize this spacecraft's attitude/pointing status for a mission "
        "controller in 1-2 plain-language sentences:\n"
        f"Object: {raw['object_name']} (NORAD {raw['object_id']})\n"
        f"Pointing error: {raw['pointing_error_deg']:.1f} deg from commanded attitude\n"
        f"Angular rate: {raw['angular_rate_deg_s']:.2f} deg/s\n"
        f"Solar panel power output: {raw['solar_panel_power_pct']:.0f}% of nominal"
    )
    fallback_text = (
        f"{raw['object_name']} has a pointing error of "
        f"{raw['pointing_error_deg']:.1f} deg (angular rate "
        f"{raw['angular_rate_deg_s']:.2f} deg/s, power at "
        f"{raw['solar_panel_power_pct']:.0f}% of nominal). "
        "Gemma was unreachable, so this is a fallback summary."
    )
    return _call_gemma_with_provenance(
        client,
        prompt=prompt,
        system="You are assisting a mission controller by summarizing "
               "spacecraft attitude/pointing status in plain language. Be "
               "concise and factual - this is a simplified assessment "
               "based on pointing error, angular rate, and solar panel "
               "power output, not a full attitude-determination-and-"
               "control analysis. Respond in plain prose only - no LaTeX, "
               "no markdown formatting, no special notation.",
        fallback_text=fallback_text,
    )


def make_analyze_node(client: GemmaClient):
    def analyze_node(state: PipelineState) -> PipelineState:
        event = state["telemetry"]
        raw = event.raw_data
        min_distance_km = raw.get("min_distance_km")
        perigee_altitude_km = raw.get("perigee_altitude_km")
        pointing_error_deg = raw.get("pointing_error_deg")

        severity_source = None
        if min_distance_km is not None:
            severity, severity_source = classify_conjunction_finding(raw)
            description, description_provenance = _conjunction_description(client, raw)
            confidence = compute_confidence(raw)
        elif perigee_altitude_km is not None:
            severity = classify_decay_severity(perigee_altitude_km)
            description, description_provenance = _decay_description(client, raw)
            confidence = compute_confidence(raw)
        elif pointing_error_deg is not None:
            severity = classify_attitude_severity(pointing_error_deg)
            description, description_provenance = _attitude_description(client, raw)
            confidence = compute_confidence(raw)
        else:
            # Non-conjunction, non-decay telemetry (e.g. DummyAdapter's
            # generic payload) - no real detection logic for this shape
            # yet, so no Gemma call is attempted at all (nothing to
            # attribute provenance to).
            severity = Severity.NOMINAL
            description = "Placeholder anomaly check - no real detection logic yet."
            description_provenance = None
            confidence = 0.5  # PLACEHOLDER: no detection logic exists yet to derive this from

        finding = AnomalyFinding(
            event_id=event.event_id,
            severity=severity,
            description=description,
            confidence=confidence,
            severity_source=severity_source,
        )
        return {**state, "finding": finding, "description_provenance": description_provenance}

    return analyze_node


ACTION_BY_SEVERITY = {
    Severity.NOMINAL: Action.CONTINUE,
    Severity.WATCH: Action.CONTINUE,
    Severity.WARNING: Action.HOLD,
    Severity.CRITICAL: Action.ABORT,
}


def _decision_rationale(
    client: GemmaClient,
    finding: AnomalyFinding,
    action: Action,
    raw: dict,
    maneuver_plan: Optional[ManeuverPlan] = None,
    verified_clearance: Optional[VerifiedClearance] = None,
    budget_insufficient: bool = False,
    awaiting_human_approval: bool = False,
    maneuver_vetoed: bool = False,
) -> tuple[str, GemmaProvenance]:
    lines = [
        f"Recommended action: {action.value}",
        f"Severity: {finding.severity.value}",
        f"Finding: {finding.description}",
    ]
    if raw.get("min_distance_km") is not None:
        lines += [
            f"Object A: {raw['object_a_name']} (NORAD {raw['object_a_id']})",
            f"Object B: {raw['object_b_name']} (NORAD {raw['object_b_id']})",
            f"Minimum distance: {raw['min_distance_km']:.3f} km",
            f"Relative velocity: {raw['relative_velocity_km_s']:.3f} km/s",
            f"Time of closest approach: {raw['time_of_closest_approach']}",
        ]
    elif raw.get("perigee_altitude_km") is not None:
        lines += [
            f"Object: {raw['object_name']} (NORAD {raw['object_id']})",
            f"Perigee altitude: {raw['perigee_altitude_km']:.1f} km",
            f"Apogee altitude: {raw['apogee_altitude_km']:.1f} km",
            f"BSTAR drag term: {raw['bstar']:.3e}",
        ]
    elif raw.get("pointing_error_deg") is not None:
        lines += [
            f"Object: {raw['object_name']} (NORAD {raw['object_id']})",
            f"Pointing error: {raw['pointing_error_deg']:.1f} deg",
            f"Angular rate: {raw['angular_rate_deg_s']:.2f} deg/s",
            f"Solar panel power: {raw['solar_panel_power_pct']:.0f}% of nominal",
        ]

    if finding.severity == Severity.CRITICAL and maneuver_plan is not None and budget_insufficient:
        # Maneuver was calculated but NOT executed - insufficient remaining
        # delta-v budget (see DeltaVBudgetTracker). This must not be narrated
        # as a completed or successful action.
        lines += [
            f"Maneuver calculated but NOT executed: {maneuver_plan.direction}, "
            f"~{maneuver_plan.magnitude_delta_v:.2f} m/s delta-v required",
            "Reason not executed: insufficient remaining delta-v budget",
        ]
        instruction = (
            "An avoidance maneuver was calculated but COULD NOT be executed "
            "because it would have exceeded the remaining delta-v budget. "
            "State plainly, as your first sentence, that a maneuver was "
            "calculated but not executed due to insufficient budget, and that "
            "this requires immediate human review (e.g. \"Maneuver required "
            "but budget insufficient - escalating for human review.\"). Do "
            "NOT say or imply that a maneuver was executed, or that clearance "
            "was achieved, or that the situation is resolved."
        )
    elif finding.severity == Severity.CRITICAL and maneuver_plan is not None and awaiting_human_approval:
        # Cloud backend ("ground control reachable" in this system's
        # model) - maneuver is calculated and affordable, but must not be
        # narrated as executed until a human explicitly approves it via
        # DecisionLogger.approve_maneuver.
        lines += [
            f"Proposed maneuver: {maneuver_plan.direction}, "
            f"~{maneuver_plan.magnitude_delta_v:.2f} m/s delta-v "
            f"(target clearance {maneuver_plan.target_clearance_km:.1f} km)",
            "Status: awaiting human approval before execution",
        ]
        instruction = (
            "An avoidance maneuver has been CALCULATED and is affordable, "
            "but has NOT been executed - it is a PROPOSAL awaiting explicit "
            "human approval before it can proceed (ground control is "
            "reachable via the cloud backend, so this system does not act "
            "alone). State plainly, as your first sentence, that a maneuver "
            "has been proposed and is awaiting human confirmation (e.g. "
            "\"Maneuver proposed: awaiting human approval before "
            "execution.\"). Briefly summarize the proposed direction and "
            "why it's needed. Do NOT say or imply that it has already been "
            "executed, verified, or that the situation is resolved - it is "
            "not resolved until a human approves it."
        )
    elif finding.severity == Severity.CRITICAL and maneuver_plan is not None and maneuver_vetoed:
        # Autonomous (local) path only - a deterministic physics check
        # already verified this maneuver was safe, but Gemma's own veto
        # check (see _maneuver_veto_check) said NO-GO, standing in for the
        # unavailable human. It was NOT applied.
        lines += [
            f"Proposed maneuver: {maneuver_plan.direction}, "
            f"~{maneuver_plan.magnitude_delta_v:.2f} m/s delta-v "
            f"(target clearance {maneuver_plan.target_clearance_km:.1f} km)",
            "Status: VETOED by autonomous safety review before execution",
        ]
        instruction = (
            "An avoidance maneuver was calculated and was affordable, and a "
            "deterministic physics check already verified it as safe - but a "
            "separate autonomous safety review (Gemma, standing in for the "
            "unavailable human) VETOED it before it could execute. It was NOT "
            "applied. State plainly, as your first sentence, that the "
            "maneuver was vetoed and is blocked pending review (e.g. "
            "\"Maneuver vetoed: blocked pending review.\"). Do NOT say or "
            "imply that it was executed, that clearance was achieved, or "
            "that the situation is resolved."
        )
    elif finding.severity == Severity.CRITICAL and maneuver_plan is not None and verified_clearance is not None:
        # CRITICAL is the one severity where an action has already been
        # autonomously executed (not just recommended) by the time Gemma is
        # asked to explain it - see maneuver.py. The rationale needs to
        # narrate that as a completed fact, not as advice to a human.
        lines += [
            f"Maneuver executed: {maneuver_plan.direction}, "
            f"~{maneuver_plan.magnitude_delta_v:.2f} m/s delta-v",
            f"Verified post-maneuver separation: {verified_clearance.new_min_distance_km:.2f} km "
            f"(cleared={verified_clearance.cleared})",
        ]
        instruction = (
            "An autonomous avoidance maneuver has ALREADY BEEN EXECUTED and "
            "verified - this is a completed action, not a pending "
            "recommendation, and not up for reconsideration. State plainly, as "
            "your first sentence, that autonomous action was taken (e.g. "
            "\"Autonomous action taken: executed a "
            f"{maneuver_plan.direction} avoidance maneuver.\"). Then state the "
            "verification result: the new separation distance and that it "
            "clears the CRITICAL threshold. Narrate both in past tense, as "
            "things that already happened. Do not phrase either as a "
            "suggestion, recommendation, or something a human should do next, "
            "and do not hedge on whether the maneuver was necessary."
        )
    else:
        instruction = (
            f"The action has already been decided deterministically: {action.value}. "
            f"This is settled and not up for reconsideration. State it plainly as "
            f"your first sentence (e.g. \"Recommendation: {action.value}.\"), then "
            "give exactly one supporting reason in 1-2 more sentences. Do not hedge, "
            "second-guess, or contradict the action or the severity classification "
            "that produced it - do not say the action 'may not be required' or "
            "similar."
        )
        if action == Action.HOLD and raw.get("perigee_altitude_km") is not None:
            instruction += (
                " Then, as a natural continuation of that same recommendation (not "
                "a contradiction of it), suggest in general terms only (no precise "
                "burn math or delta-v figures) a plausible next step given the "
                "decaying orbit - e.g. an increased tracking/monitoring cadence, or "
                "planning ahead for a reboost or controlled-deorbit decision as the "
                "perigee continues to drop."
            )
        elif action == Action.HOLD and raw.get("pointing_error_deg") is not None:
            instruction += (
                " Then, as a natural continuation of that same recommendation (not "
                "a contradiction of it), suggest in general terms only (no precise "
                "control-law or torque figures) a plausible next step given the "
                "pointing error - e.g. increased attitude telemetry monitoring, "
                "or preparing a detumble/safe-mode-recovery procedure if the "
                "pointing error or angular rate continues to increase."
            )
        elif action == Action.HOLD:
            instruction += (
                " Then, as a natural continuation of that same recommendation (not "
                "a contradiction of it), suggest in general terms only (no precise "
                "burn math or delta-v figures) a plausible avoidance direction - "
                "e.g. a small along-track burn to shift the crossing time, or a "
                "radial burn to increase separation."
            )

    prompt = "\n".join(lines) + "\n\n" + instruction
    fallback_text = f"Action: {action.value}. Automated fallback -- Gemma explanation unavailable."

    return _call_gemma_with_provenance(
        client,
        prompt=prompt,
        system="You are assisting a mission controller by explaining a "
               "spacecraft-safety action (collision avoidance, orbital "
               "decay/re-entry risk, or attitude/pointing loss) in plain "
               "language. Be concise, "
               "factual, decisive, and avoid precise maneuver figures beyond "
               "what you're explicitly given. The action has already been "
               "decided deterministically before you are called - for "
               "CRITICAL cases it may already be executed (autonomous, no "
               "human in the loop), awaiting a human's explicit approval, "
               "blocked by insufficient budget, or vetoed by a separate "
               "autonomous safety review; follow the specific instructions "
               "below for which applies. "
               "Your job is to explain the current state accurately, not "
               "decide or second-guess it. Never hedge on or contradict the "
               "action or severity you are given. Respond in plain prose "
               "only - no LaTeX, no markdown formatting, no special notation.",
        fallback_text=fallback_text,
    )


_MANEUVER_VETO_SYSTEM = (
    "You are a final autonomous safety reviewer for an avoidance maneuver "
    "that a deterministic physics check has already computed and "
    "independently verified as safe, on a spacecraft where ground control "
    "is currently unreachable. You are the only check left before this "
    "maneuver executes with no human in the loop. Respond with a JSON "
    "object containing \"verdict\" (exactly the string \"GO\" or \"NO-GO\") "
    "and \"reason\" (a short plain-English explanation, no LaTeX or "
    "markdown). Only answer NO-GO if you see a concrete problem with the "
    "numbers you are given - do not second-guess a maneuver the numbers "
    "already show is safe just to be cautious."
)

# Real JSON-schema-constrained decoding for the veto verdict (Ollama's
# own /api/generate `format` param - see GemmaClient._generate_ollama),
# not a post-hoc parsing hint: confirmed directly against this project's
# own model (gemma4:e4b) that Ollama's structured-output support forces
# the response to actually match this shape, eliminating the free-text
# parsing ambiguity _parse_veto_verdict below exists to handle. Kept as
# a schema constant (not a Pydantic model) since it's passed straight
# through to Ollama's HTTP API, which expects raw JSON Schema.
_VETO_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["GO", "NO-GO"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def _maneuver_veto_prompt(
    raw: dict, maneuver_plan: ManeuverPlan, verified_clearance: VerifiedClearance,
) -> str:
    return (
        "A CRITICAL conjunction's avoidance maneuver has been computed and "
        "independently verified:\n"
        f"Object A: {raw['object_a_name']} (NORAD {raw['object_a_id']})\n"
        f"Object B: {raw['object_b_name']} (NORAD {raw['object_b_id']})\n"
        f"Current minimum distance: {raw['min_distance_km']:.3f} km\n"
        f"Relative velocity: {raw['relative_velocity_km_s']:.3f} km/s\n"
        f"Proposed maneuver: {maneuver_plan.direction}, "
        f"{maneuver_plan.magnitude_delta_v:.2f} m/s delta-v\n"
        f"Independently verified post-maneuver separation: "
        f"{verified_clearance.new_min_distance_km:.2f} km "
        f"(clears the 5km CRITICAL threshold: {verified_clearance.cleared})\n\n"
        "What is your verdict?"
    )


# Matches "GO", "NO-GO", "NO GO", "NOGO" as whole words/phrases, case
# insensitive. NO-GO variants are checked before bare GO in the pattern so
# a "NO-GO" response can't register as a false-positive "GO" match.
_VETO_VERDICT_RE = re.compile(r"\bNO[\s-]?GO\b|\bGO\b", re.IGNORECASE)


def _parse_veto_verdict(text: str) -> Optional[bool]:
    """Parses a GO/NO-GO verdict from a veto-check response.

    Checks the FIRST token authoritatively before falling back to a scan:
    the prompt instructs the verdict to come first, and scanning the whole
    response for the LAST match can misread a later NEGATED mention of the
    other token - e.g. "GO - ... this is clearly not a NO-GO situation, so
    proceed." leads with an affirmed GO, but naively taking the last match
    anywhere in the text would find the "NO-GO" substring inside that
    later sentence and misread it as a veto. Falls back to scanning for
    the LAST match only when the response doesn't lead with a clear token
    - mirroring this project's other chain-of-thought handling
    (_extract_final_answer): a model that reasons before its verdict tends
    to converge on the real answer last, even though this prompt asks for
    the verdict first. Returns None if no verdict token appears anywhere -
    the caller treats that as a fail-safe NO-GO, not a free pass.
    """
    stripped = text.strip()
    first_token = _VETO_VERDICT_RE.match(stripped)
    if first_token:
        return not first_token.group(0).upper().startswith("NO")

    matches = _VETO_VERDICT_RE.findall(stripped)
    if not matches:
        return None
    return not matches[-1].upper().startswith("NO")


def _parse_veto_json(text: str) -> Optional[tuple[bool, str]]:
    """Parses the {"verdict": "GO"|"NO-GO", "reason": str} response
    Ollama's schema-constrained generation produces for _VETO_JSON_SCHEMA
    - the preferred path: deterministic by construction, no free-text
    scanning needed. Returns None (not "NO-GO") for anything that isn't
    valid, schema-matching JSON - e.g. a cross-backend fallback response
    from the hosted API, which doesn't honor `format` (see
    GemmaClient._generate_hosted_api) - so the caller can fall back to
    _parse_veto_verdict's free-text scan instead of misreading a parse
    failure as an actual veto."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    if verdict not in ("GO", "NO-GO"):
        return None
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    return verdict == "GO", reason


def _maneuver_veto_check(
    client: GemmaClient, raw: dict, maneuver_plan: ManeuverPlan, verified_clearance: VerifiedClearance,
) -> tuple[bool, str, GemmaProvenance]:
    """Calls Gemma for a real GO/NO-GO verdict on an already-verified-safe
    maneuver, standing in for the unavailable human on the autonomous
    (local) path only (see make_decide_node). Gemma can only make the
    outcome MORE conservative than the physics check, never less - it never
    gets to approve something that hasn't already been independently
    verified safe.

    Returns (approved, detail, provenance). detail is a short, already-
    labeled explanation of how the verdict was reached, meant to be
    embedded directly into ManeuverApproval.reason:
      - Gemma unreachable: NOT treated as a veto (an LLM outage alone
        shouldn't block a maneuver the deterministic physics has already
        verified safe) - approved=True, provenance.source="fallback".
      - Gemma responds but gives no parseable GO/NO-GO verdict: fail-safe
        default is NO-GO, not a free pass - approved=False.
      - Gemma explicitly says NO-GO: approved=False.
      - Gemma explicitly says GO: approved=True.

    Requests real JSON-schema-constrained output (_VETO_JSON_SCHEMA) and
    prefers parsing that structured response (_parse_veto_json) over the
    older free-text scan (_parse_veto_verdict) - the schema constraint is
    Ollama-only, so the free-text scan stays as a fallback for the one
    case structured output can't cover: generate()'s own cross-backend
    fallback landing on the hosted API mid-call, which doesn't honor
    `format` and returns plain text instead.
    """
    prompt = _maneuver_veto_prompt(raw, maneuver_plan, verified_clearance)
    fallback_text = "Gemma unreachable; defaulting to physics-only autonomous approval (fail-safe)."
    text, provenance = _call_gemma_with_provenance(
        client, prompt=prompt, system=_MANEUVER_VETO_SYSTEM, fallback_text=fallback_text,
        postprocess=lambda raw_text: raw_text.strip(),
        format=_VETO_JSON_SCHEMA,
    )
    if provenance.source == "fallback":
        return True, fallback_text, provenance

    structured = _parse_veto_json(text)
    if structured is not None:
        verdict, reason = structured
        detail_text = reason
    else:
        verdict = _parse_veto_verdict(text)
        detail_text = text

    if verdict is None:
        return False, f"Gemma's response had no parseable GO/NO-GO verdict - defaulted to NO-GO (fail-safe): \"{text}\"", provenance
    if verdict:
        return True, f"affirmed by Gemma's autonomous safety review: \"{detail_text}\"", provenance
    return False, f"vetoed by Gemma's autonomous safety review: \"{detail_text}\"", provenance


def make_decide_node(client: GemmaClient, budget_tracker: Optional[DeltaVBudgetTracker] = None):
    # One tracker per decide_node closure by default, so its remaining
    # budget persists across every event processed by this pipeline
    # instance (see build_pipeline) rather than resetting per event.
    if budget_tracker is None:
        starting_budget = getattr(client.settings, "delta_v_budget_m_s", 5.0)
        budget_tracker = DeltaVBudgetTracker(starting_budget)

    def decide_node(state: PipelineState) -> PipelineState:
        finding = state["finding"]
        assert finding is not None, "decide_node requires a finding from analyze_node"

        action = ACTION_BY_SEVERITY[finding.severity]
        raw = state["telemetry"].raw_data

        maneuver_plan: Optional[ManeuverPlan] = None
        verified_clearance: Optional[VerifiedClearance] = None
        maneuver_approval: Optional[ManeuverApproval] = None
        veto_provenance: Optional[GemmaProvenance] = None
        budget_insufficient = False
        awaiting_human_approval = False
        maneuver_vetoed = False
        # CRITICAL can now come from any of three hazard types (conjunctions
        # via classify_conjunction_severity, orbital decay via
        # classify_decay_severity, or attitude/pointing loss via
        # classify_attitude_severity - see analyze_node) - the maneuver
        # machinery below is conjunction-specific (an avoidance burn makes
        # no sense for "your perigee is too low" or "you're tumbling"), so
        # it's gated on the raw_data actually being conjunction-shaped, not
        # just on severity. A CRITICAL decay or attitude finding still gets
        # a real deterministic action (Action.ABORT) and real Gemma
        # narration - just no maneuver/budget/veto/approval machinery,
        # which is conjunction-only scope (see PHASE_PROGRESS.md Phases 14
        # and 18).
        if finding.severity == Severity.CRITICAL and "object_a_id" in raw:
            maneuver_plan = compute_avoidance_maneuver(
                object_a=raw["object_a_id"],
                object_b=raw["object_b_id"],
                min_distance_km=raw["min_distance_km"],
                relative_velocity_km_s=raw["relative_velocity_km_s"],
            )
            # None here means min_distance_km is already beyond this
            # model's target clearance - real for the Pc-based CRITICAL
            # path (see pc_severity.classify_pc_severity), where tight
            # covariance can classify a conjunction CRITICAL at a large
            # miss distance the displacement-maneuver model has nothing
            # to compute for. Falls through with no maneuver/budget/
            # veto/approval machinery, the same as CRITICAL decay/
            # attitude findings - _decision_rationale's generic branch
            # still narrates the deterministic action correctly.
            if maneuver_plan is not None:
                if budget_tracker.consume(maneuver_plan.magnitude_delta_v):
                    # Backend choice doubles as "is ground control reachable"
                    # in this system's model (see ManeuverApproval docstring):
                    # local (ollama) -> treat as unreachable -> self-approve
                    # based on the deterministic checks already done above.
                    # api (or anything else/unrecognized) -> treat as reachable
                    # -> require an explicit human approval before executing;
                    # unrecognized defaults to requiring a human as the safer
                    # fail-safe, not to full autonomy.
                    configured_backend = getattr(client.settings, "gemma_backend", None)
                    if configured_backend == "ollama":
                        # A deterministic physics check verifies safety first,
                        # then Gemma itself gets the same numbers and issues a
                        # real GO/NO-GO, standing in for the unavailable human
                        # (see _maneuver_veto_check). Gemma can only make this
                        # MORE conservative than the physics check, never less.
                        candidate_clearance = verify_maneuver(raw["min_distance_km"], maneuver_plan)
                        veto_go, veto_detail, veto_provenance = _maneuver_veto_check(
                            client, raw, maneuver_plan, candidate_clearance,
                        )
                        if veto_go:
                            verified_clearance = candidate_clearance
                            maneuver_approval = ManeuverApproval(
                                mode="autonomous",
                                approved=True,
                                approved_by=None,
                                approved_at=datetime.now(timezone.utc),
                                reason=(
                                    "Local backend - ground control treated as unreachable; "
                                    "approved autonomously based on deterministic severity/physics "
                                    f"checks, {veto_detail}"
                                ),
                            )
                        else:
                            # Not executed - verified_clearance stays None, same
                            # invariant as budget_insufficient/awaiting_human_approval.
                            maneuver_vetoed = True
                            maneuver_approval = ManeuverApproval(
                                mode="autonomous",
                                approved=False,
                                approved_by="Gemma (autonomous safety review)",
                                approved_at=datetime.now(timezone.utc),
                                reason=(
                                    "Maneuver was independently verified safe by "
                                    f"deterministic physics, but {veto_detail}"
                                ),
                            )
                    else:
                        awaiting_human_approval = True
                else:
                    # Plan stays visible (mission control can see what would've
                    # been needed) but nothing was actually applied, so there's
                    # nothing to verify and no approval to seek yet.
                    budget_insufficient = True

        rationale, rationale_provenance = _decision_rationale(
            client, finding, action, raw, maneuver_plan, verified_clearance,
            budget_insufficient, awaiting_human_approval, maneuver_vetoed,
        )

        decision = Decision(
            action=action,
            rationale=rationale,
            made_at=datetime.now(timezone.utc),
            maneuver_plan=maneuver_plan,
            verified_clearance=verified_clearance,
            budget_insufficient=budget_insufficient,
            awaiting_human_approval=awaiting_human_approval,
            maneuver_approval=maneuver_approval,
        )
        return {
            **state, "decision": decision, "rationale_provenance": rationale_provenance,
            "veto_provenance": veto_provenance,
        }

    return decide_node


def make_log_node(logger: DecisionLogger):
    def log_node(state: PipelineState) -> PipelineState:
        finding = state["finding"]
        decision = state["decision"]
        assert finding is not None and decision is not None

        entry = DecisionLogEntry(
            telemetry=state["telemetry"],
            finding=finding,
            decision=decision,
            description_provenance=state.get("description_provenance"),
            rationale_provenance=state["rationale_provenance"],
            veto_provenance=state.get("veto_provenance"),
        )
        log_path = logger.log(entry)
        # Real-time alert, not just a passive log entry - see
        # src/alerting.py. A no-op unless ALERT_WEBHOOK_URL is configured
        # (logger.settings, the same Settings this pipeline run already
        # uses); never raises, so a webhook outage can't block logging.
        send_critical_alert(entry, logger.settings)
        return {**state, "log_path": log_path}

    return log_node


def build_pipeline(
    client: Optional[GemmaClient] = None,
    logger: Optional[DecisionLogger] = None,
    budget_tracker: Optional[DeltaVBudgetTracker] = None,
):
    client = client or GemmaClient()
    logger = logger or DecisionLogger()

    graph = StateGraph(PipelineState)
    graph.add_node("analyze", make_analyze_node(client))
    graph.add_node("decide", make_decide_node(client, budget_tracker=budget_tracker))
    graph.add_node("log", make_log_node(logger))

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "decide")
    graph.add_edge("decide", "log")
    graph.add_edge("log", END)

    return graph.compile()


def run_once(
    adapter: Optional[DataSourceAdapter] = None,
    client: Optional[GemmaClient] = None,
    logger: Optional[DecisionLogger] = None,
    budget_tracker: Optional[DeltaVBudgetTracker] = None,
    limit: int = 1,
) -> list[DecisionLogEntry]:
    """Fetch a batch of telemetry and run each event through the pipeline."""
    adapter = adapter or DummyAdapter()
    compiled = build_pipeline(client=client, logger=logger, budget_tracker=budget_tracker)

    entries: list[DecisionLogEntry] = []
    for event in adapter.fetch_batch(limit):
        result = compiled.invoke({
            "telemetry": event,
            "finding": None,
            "decision": None,
            "log_path": None,
            "description_provenance": None,
            "rationale_provenance": None,
            "veto_provenance": None,
        })
        entries.append(
            DecisionLogEntry(
                telemetry=result["telemetry"],
                finding=result["finding"],
                decision=result["decision"],
                description_provenance=result.get("description_provenance"),
                rationale_provenance=result["rationale_provenance"],
                veto_provenance=result.get("veto_provenance"),
            )
        )
    return entries


if __name__ == "__main__":
    from .display import render_entries

    results = run_once(limit=3)
    render_entries(results)
