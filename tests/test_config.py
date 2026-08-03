"""Tests for src/config.py's env-var parsing helpers."""
from src.config import Settings, _parse_watched_norad_ids, load_settings


def test_parse_watched_norad_ids_splits_on_commas():
    assert _parse_watched_norad_ids("25544,48274") == ("25544", "48274")


def test_parse_watched_norad_ids_strips_whitespace():
    assert _parse_watched_norad_ids(" 25544 , 48274 ") == ("25544", "48274")


def test_parse_watched_norad_ids_drops_empty_items():
    assert _parse_watched_norad_ids("25544,,48274,") == ("25544", "48274")


def test_parse_watched_norad_ids_empty_string_is_empty_tuple():
    assert _parse_watched_norad_ids("") == ()


def test_parse_watched_norad_ids_single_id():
    assert _parse_watched_norad_ids("25544") == ("25544",)


def test_settings_severity_thresholds_default_to_the_original_hardcoded_values():
    """Real behavior this guards: a deployment that never configures any
    of the new *_KM/*_DEG env vars must see EXACTLY this project's
    original hardcoded thresholds - zero behavior change for anyone who
    doesn't opt in."""
    settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir="./logs",
        delta_v_budget_m_s=5.0,
    )

    assert settings.conjunction_critical_km == 5.0
    assert settings.conjunction_warning_km == 25.0
    assert settings.conjunction_watch_km == 100.0
    assert settings.decay_critical_perigee_km == 200.0
    assert settings.decay_warning_perigee_km == 300.0
    assert settings.decay_watch_perigee_km == 500.0
    assert settings.attitude_critical_deg == 45.0
    assert settings.attitude_warning_deg == 15.0
    assert settings.attitude_watch_deg == 5.0


def test_load_settings_reads_real_configured_severity_thresholds_from_env(monkeypatch):
    """Real feature this tests: a real operator can actually override
    these via .env/the real process environment, not just construct a
    Settings object directly in Python."""
    monkeypatch.setenv("CONJUNCTION_CRITICAL_KM", "2.5")
    monkeypatch.setenv("CONJUNCTION_WARNING_KM", "10.0")
    monkeypatch.setenv("DECAY_CRITICAL_PERIGEE_KM", "250.0")
    monkeypatch.setenv("ATTITUDE_CRITICAL_DEG", "30.0")

    settings = load_settings()

    assert settings.conjunction_critical_km == 2.5
    assert settings.conjunction_warning_km == 10.0
    assert settings.decay_critical_perigee_km == 250.0
    assert settings.attitude_critical_deg == 30.0
