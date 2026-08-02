"""Environment-driven configuration for the Track 2 scaffolding."""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def load_settings() -> Settings:
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
    )


settings = load_settings()
