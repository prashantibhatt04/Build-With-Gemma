"""Environment-driven configuration for the Track 2 scaffolding."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemma_backend: str
    # Local Ollama model tag (e.g. "gemma4:e4b"). Ollama and the hosted API
    # use different model-naming schemes entirely - a tag valid for one is
    # not valid for the other - so each backend gets its own field rather
    # than sharing gemma_model.
    gemma_model: str
    ollama_host: str
    gemma_api_key: str
    # Hosted Gemini-style API model id (e.g. "gemma-4-26b-a4b-it") - see
    # gemma_model's docstring above for why this is separate.
    gemma_model_api: str
    log_dir: str
    # Starting simplified delta-v budget for CRITICAL avoidance maneuvers,
    # in m/s. Not a real fuel/mass model - just a bound so repeated CRITICAL
    # events can't silently execute an unlimited number of maneuvers. See
    # src/maneuver.py:DeltaVBudgetTracker.
    delta_v_budget_m_s: float
    # Ollama embedding model used ONLY by src/rag.py's mission-log search -
    # always via Ollama regardless of gemma_backend, since the hosted
    # Gemini-style API has no embedding endpoint wired up here. Defaulted
    # (unlike gemma_model/gemma_model_api) so every existing Settings(...)
    # call site in this codebase keeps working unchanged.
    gemma_embed_model: str = "nomic-embed-text"
    # Webhook URL for real-time CRITICAL-event alerts (see src/alerting.py).
    # Slack Incoming Webhook compatible ({"text": ...} payload) - also
    # works with Discord/Microsoft Teams via a compatible endpoint, or any
    # custom HTTP receiver. Empty string (the default) disables alerting
    # entirely - a no-op, not an error, so existing tests/demos are
    # unaffected unless someone explicitly configures one.
    alert_webhook_url: str = ""
    # Space-Track.org credentials (see ROADMAP_TO_PRODUCT.md Phase 1) - a
    # real free account, registered by a human at
    # https://www.space-track.org/auth/createAccount, not something this
    # project can obtain automatically. Both default empty: SpaceTrackAdapter
    # (src/ingestion/spacetrack_adapter.py) raises a clear configuration
    # error if constructed without them, same "absent = explicit, not a
    # silent no-op" treatment as GEMMA_API_KEY when GEMMA_BACKEND=api -
    # CelesTrakAdapter stays the always-available, no-credentials default.
    spacetrack_username: str = ""
    spacetrack_password: str = ""
    # Postgres connection string (see ROADMAP_TO_PRODUCT.md Phase 4), e.g.
    # "postgresql://user:pass@host:5432/dbname". Empty (the default) keeps
    # DecisionLogger on its original append-only JSONL-file backend -
    # same "absent = fallback to the existing default, not an error"
    # pattern as every other optional integration in this project. Only
    # storage changes when this is set - DecisionLogger's public interface
    # (log/find_entry/load_all_entries/mark_reviewed/approve_maneuver) is
    # identical either way, so nothing downstream (dashboard, RAG, trends,
    # alerting) needs to know or care which backend is active.
    database_url: str = ""
    # Real operator authentication for scripts/dashboard.py (see
    # src/auth.py) - "name:token,name2:token2". Empty (the default)
    # leaves the dashboard's operator field as free text, same as before
    # this phase - configuring this is what actually closes the "anyone
    # can claim to be anyone" gap for approve/reject/mark-reviewed.
    operator_tokens: dict[str, str] = field(default_factory=dict)
    # Real rate limit for scripts/api.py (see src/rate_limit.py) - requests
    # per minute per client, always on by default (unlike most settings in
    # this project, a baseline protection shouldn't require opting in). 0
    # disables it entirely, e.g. for local dev convenience.
    api_rate_limit_per_minute: int = 120
    # Real, deterministic hazard-severity thresholds - not Gemma-derived,
    # see src/pipeline.py's classify_*_severity functions for exactly how
    # each is used. Defaults match this project's original hardcoded
    # values exactly (zero behavior change for anyone who doesn't set
    # these) - configurable per deployment because different real
    # operators have different risk tolerance and different assets: a
    # maneuverable, high-value satellite reasonably wants a bigger
    # CRITICAL buffer than a defunct cubesat, and there's no one correct
    # answer this project should hardcode for every real customer.
    # conjunction_critical_km is also the one src/maneuver.py's
    # compute_avoidance_maneuver/verify_maneuver read (via decide_node
    # passing it through explicitly) - a single source of truth, not a
    # second hardcoded constant that could silently drift out of sync
    # with what actually got classified CRITICAL in the first place.
    conjunction_critical_km: float = 5.0
    conjunction_warning_km: float = 25.0
    conjunction_watch_km: float = 100.0
    decay_critical_perigee_km: float = 200.0
    decay_warning_perigee_km: float = 300.0
    decay_watch_perigee_km: float = 500.0
    attitude_critical_deg: float = 45.0
    attitude_warning_deg: float = 15.0
    attitude_watch_deg: float = 5.0
    # A real operator's own asset(s), by NORAD catalog ID - "25544,48274".
    # Empty (the default) keeps every existing demo/screening button
    # pointed at CelesTrak's own curated "stations" group, exactly as
    # before this setting existed. This is what actually lets a real
    # customer point this project at THEIR satellite: without it, there
    # was no way to screen anything but CelesTrak's own pre-picked group
    # as the "asset" side of a conjunction, which for nearly every real
    # satellite that isn't a crewed station means no way to monitor it at
    # all (see CelesTrakAdapter/SpaceTrackAdapter's watched_norad_ids
    # param, and ROADMAP_TO_PRODUCT.md).
    watched_norad_ids: tuple[str, ...] = ()


def _parse_watched_norad_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def load_settings() -> Settings:
    # Local import to avoid a circular import at module load time
    # (src/auth.py has no reason to depend on config.py itself, but
    # config.py is imported very early by nearly everything else).
    from .auth import parse_operator_tokens

    return Settings(
        gemma_backend=os.getenv("GEMMA_BACKEND", "ollama"),
        gemma_model=os.getenv("GEMMA_MODEL", "gemma4:e4b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        gemma_api_key=os.getenv("GEMMA_API_KEY", ""),
        gemma_model_api=os.getenv("GEMMA_MODEL_API", "gemma-4-26b-a4b-it"),
        log_dir=os.getenv("LOG_DIR", "./logs"),
        delta_v_budget_m_s=float(os.getenv("DELTA_V_BUDGET_M_S", "5.0")),
        gemma_embed_model=os.getenv("GEMMA_EMBED_MODEL", "nomic-embed-text"),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
        spacetrack_username=os.getenv("SPACETRACK_USERNAME", ""),
        spacetrack_password=os.getenv("SPACETRACK_PASSWORD", ""),
        database_url=os.getenv("DATABASE_URL", ""),
        operator_tokens=parse_operator_tokens(os.getenv("OPERATOR_TOKENS", "")),
        api_rate_limit_per_minute=int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "120")),
        watched_norad_ids=_parse_watched_norad_ids(os.getenv("WATCHED_NORAD_IDS", "")),
        conjunction_critical_km=float(os.getenv("CONJUNCTION_CRITICAL_KM", "5.0")),
        conjunction_warning_km=float(os.getenv("CONJUNCTION_WARNING_KM", "25.0")),
        conjunction_watch_km=float(os.getenv("CONJUNCTION_WATCH_KM", "100.0")),
        decay_critical_perigee_km=float(os.getenv("DECAY_CRITICAL_PERIGEE_KM", "200.0")),
        decay_warning_perigee_km=float(os.getenv("DECAY_WARNING_PERIGEE_KM", "300.0")),
        decay_watch_perigee_km=float(os.getenv("DECAY_WATCH_PERIGEE_KM", "500.0")),
        attitude_critical_deg=float(os.getenv("ATTITUDE_CRITICAL_DEG", "45.0")),
        attitude_warning_deg=float(os.getenv("ATTITUDE_WARNING_DEG", "15.0")),
        attitude_watch_deg=float(os.getenv("ATTITUDE_WATCH_DEG", "5.0")),
    )


settings = load_settings()
