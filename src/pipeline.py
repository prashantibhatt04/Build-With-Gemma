"""LangGraph pipeline skeleton: ingest -> analyze -> decide -> log.

analyze_node classifies conjunction severity deterministically from
min_distance_km and uses Gemma only to generate the human-readable
description. decide_node maps severity to action deterministically and
uses Gemma only to explain the recommendation. Non-conjunction
telemetry (e.g. DummyAdapter's generic payload) falls back to
placeholder behavior in both nodes.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from .gemma_client import GemmaClient, GemmaClientError
from .ingestion.base_adapter import DataSourceAdapter, DummyAdapter
from .logging_utils import DecisionLogger
from .maneuver import DeltaVBudgetTracker, compute_avoidance_maneuver, verify_maneuver
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
) -> tuple[str, GemmaProvenance]:
    """Calls Gemma and always returns (text, provenance), never raises -
    falls back to fallback_text on GemmaClientError."""
    start = time.monotonic()
    try:
        text = _extract_final_answer(client.generate(prompt=prompt, system=system))
        source = "gemma"
        model_used = _describe_model_used(client)
    except GemmaClientError:
        text = fallback_text
        source = "fallback"
        model_used = client.settings.gemma_model
    latency_ms = (time.monotonic() - start) * 1000
    provenance = GemmaProvenance(
        source=source, model_used=model_used, latency_ms=latency_ms,
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


def make_analyze_node(client: GemmaClient):
    def analyze_node(state: PipelineState) -> PipelineState:
        event = state["telemetry"]
        raw = event.raw_data
        min_distance_km = raw.get("min_distance_km")

        if min_distance_km is not None:
            severity = classify_conjunction_severity(min_distance_km)
            description, description_provenance = _conjunction_description(client, raw)
            confidence = compute_confidence(raw)
        else:
            # Non-conjunction telemetry (e.g. DummyAdapter's generic payload) -
            # no real detection logic for this shape yet, so no Gemma call is
            # attempted at all (nothing to attribute provenance to).
            severity = Severity.NOMINAL
            description = "Placeholder anomaly check - no real detection logic yet."
            description_provenance = None
            confidence = 0.5  # PLACEHOLDER: no detection logic exists yet to derive this from

        finding = AnomalyFinding(
            event_id=event.event_id,
            severity=severity,
            description=description,
            confidence=confidence,
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
        if action == Action.HOLD:
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
               "collision-avoidance action in plain language. Be concise, "
               "factual, decisive, and avoid precise maneuver figures beyond "
               "what you're explicitly given. The action has already been "
               "decided deterministically before you are called - for "
               "CRITICAL cases it may already be executed (autonomous, no "
               "human in the loop) or awaiting a human's explicit approval; "
               "follow the specific instructions below for which applies. "
               "Your job is to explain the current state accurately, not "
               "decide or second-guess it. Never hedge on or contradict the "
               "action or severity you are given. Respond in plain prose "
               "only - no LaTeX, no markdown formatting, no special notation.",
        fallback_text=fallback_text,
    )


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
        budget_insufficient = False
        awaiting_human_approval = False
        if finding.severity == Severity.CRITICAL:
            # CRITICAL is only ever produced from conjunction-shaped raw_data
            # (see classify_conjunction_severity), so these keys are present.
            maneuver_plan = compute_avoidance_maneuver(
                object_a=raw["object_a_id"],
                object_b=raw["object_b_id"],
                min_distance_km=raw["min_distance_km"],
                relative_velocity_km_s=raw["relative_velocity_km_s"],
            )
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
                    verified_clearance = verify_maneuver(raw["min_distance_km"], maneuver_plan)
                    maneuver_approval = ManeuverApproval(
                        mode="autonomous",
                        approved=True,
                        approved_by=None,
                        approved_at=datetime.now(timezone.utc),
                        reason=(
                            "Local backend - ground control treated as unreachable; "
                            "approved autonomously based on deterministic severity/physics checks."
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
            budget_insufficient, awaiting_human_approval,
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
        return {**state, "decision": decision, "rationale_provenance": rationale_provenance}

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
        )
        log_path = logger.log(entry)
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
        })
        entries.append(
            DecisionLogEntry(
                telemetry=result["telemetry"],
                finding=result["finding"],
                decision=result["decision"],
                description_provenance=result.get("description_provenance"),
                rationale_provenance=result["rationale_provenance"],
            )
        )
    return entries


if __name__ == "__main__":
    from .display import render_entries

    results = run_once(limit=3)
    render_entries(results)
