"""Tests for scripts/scheduler.py. Real Gemma/network calls are never
made here - a duck-typed fake client (same convention as
test_pipeline_smoke.py's FakeGemmaClient) and DummyAdapter stand in, so
these run fast and offline. The scheduling loop itself (interval sleep,
Ctrl-C handling) isn't exercised directly - run_tick is the real unit of
behavior worth testing without actually sleeping in a test.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.ingestion.base_adapter import DummyAdapter
from src.logging_utils import DecisionLogger
from src.maneuver import DeltaVBudgetTracker
from scripts.scheduler import (
    CONSECUTIVE_FAILURE_ALERT_THRESHOLD,
    default_adapters,
    heartbeat_is_fresh,
    run_tick,
    update_consecutive_failures,
    write_heartbeat,
)


class FakeGemmaClient:
    def __init__(self):
        self.settings = SimpleNamespace(gemma_model="fake-model", gemma_backend="ollama")

    def generate(self, prompt, system=None, timeout=60, format=None):
        return "Stubbed narration: nothing to report."


def _settings(tmp_path) -> Settings:
    return Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir=str(tmp_path),
        delta_v_budget_m_s=5.0,
    )


def test_run_tick_screens_every_configured_adapter_and_logs_results(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    client = FakeGemmaClient()
    budget_tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)
    adapters = [DummyAdapter(source_name="fake-a"), DummyAdapter(source_name="fake-b")]

    entries = run_tick(client, logger, budget_tracker, tick_number=1, adapters=adapters, limit_per_adapter=2)

    # 2 adapters x 2 events each.
    assert len(entries) == 4
    assert {e.telemetry.source for e in entries} == {"fake-a", "fake-b"}
    # Actually persisted, not just returned in memory.
    assert len(logger.load_all_entries()) == 4


def test_run_tick_reuses_the_same_client_and_budget_tracker_across_calls(tmp_path):
    """The whole point of constructing these ONCE in main() rather than
    per tick - confirms run_tick doesn't secretly reconstruct either,
    which would silently reset delta-v budget / defeat the circuit
    breaker exactly like the bug PHASE_PROGRESS.md's QA pass already
    found and fixed for the dashboard/run_demo.py."""
    logger = DecisionLogger(settings=_settings(tmp_path))
    client = FakeGemmaClient()
    budget_tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)

    run_tick(client, logger, budget_tracker, tick_number=1, adapters=[DummyAdapter()], limit_per_adapter=1)
    run_tick(client, logger, budget_tracker, tick_number=2, adapters=[DummyAdapter()], limit_per_adapter=1)

    # Same object identity across both calls - the test's own local
    # variables, not reconstructed inside run_tick.
    assert budget_tracker.remaining_m_s == 5.0  # DummyAdapter never triggers a maneuver


def test_run_tick_returns_empty_list_when_an_adapter_yields_nothing(tmp_path):
    logger = DecisionLogger(settings=_settings(tmp_path))
    client = FakeGemmaClient()
    budget_tracker = DeltaVBudgetTracker(starting_budget_m_s=5.0)

    entries = run_tick(
        client, logger, budget_tracker, tick_number=1,
        adapters=[DummyAdapter()], limit_per_adapter=0,
    )

    assert entries == []


def test_default_adapters_returns_the_real_conjunction_and_decay_sources():
    from src.ingestion.celestrak_adapter import CelesTrakAdapter
    from src.ingestion.decay_adapter import DecayRiskAdapter

    adapters = default_adapters()

    assert len(adapters) == 2
    assert isinstance(adapters[0], CelesTrakAdapter)
    assert isinstance(adapters[1], DecayRiskAdapter)
    assert adapters[0].watched_norad_ids == []


def test_default_adapters_watches_the_real_configured_asset_when_set(monkeypatch):
    """Real feature this tests: continuous unattended monitoring - the
    whole point of running this scheduler at all - must watch a real
    operator's own configured satellite when WATCHED_NORAD_IDS is set,
    not always fall back to CelesTrak's "stations" demo placeholder."""
    import scripts.scheduler as scheduler_module

    watching_settings = Settings(
        gemma_backend="ollama", gemma_model="gemma4:e4b", ollama_host="http://localhost:11434",
        gemma_api_key="", gemma_model_api="gemma-4-26b-a4b-it", log_dir="./logs",
        delta_v_budget_m_s=5.0, watched_norad_ids=("25544",),
    )
    monkeypatch.setattr(scheduler_module, "settings", watching_settings)

    adapters = scheduler_module.default_adapters()

    assert len(adapters) == 2
    assert adapters[0].watched_norad_ids == ["25544"]
    assert adapters[0].groups == ["cosmos-2251-debris"]
    assert adapters[1].catalog_ids == ["25544"]


def test_update_consecutive_failures_resets_to_zero_on_success():
    assert update_consecutive_failures(True, consecutive_failures=2) == (0, False)


def test_update_consecutive_failures_increments_and_stays_silent_below_threshold():
    count, should_alert = update_consecutive_failures(False, consecutive_failures=0)
    assert count == 1
    assert should_alert is False


def test_update_consecutive_failures_alerts_exactly_once_at_the_threshold():
    count = 0
    alerts = []
    for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD + 2):
        count, should_alert = update_consecutive_failures(False, count)
        alerts.append(should_alert)

    # Exactly one True, at the threshold - not before, not repeated after.
    assert alerts.count(True) == 1
    assert alerts[CONSECUTIVE_FAILURE_ALERT_THRESHOLD - 1] is True


def test_update_consecutive_failures_can_alert_again_after_a_recovery():
    count, _ = update_consecutive_failures(False, 0)
    for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD - 1):
        count, _ = update_consecutive_failures(False, count)
    count, _ = update_consecutive_failures(True, count)  # recovers

    alerted_again = False
    for _ in range(CONSECUTIVE_FAILURE_ALERT_THRESHOLD):
        count, should_alert = update_consecutive_failures(False, count)
        alerted_again = alerted_again or should_alert

    assert alerted_again is True


def test_write_heartbeat_creates_parent_dirs_and_writes_real_json(tmp_path):
    path = tmp_path / "nested" / "heartbeat.json"

    write_heartbeat(path, interval_seconds=60.0, tick_number=3)

    heartbeat = json.loads(path.read_text())
    assert heartbeat["interval_seconds"] == 60.0
    assert heartbeat["tick_number"] == 3
    assert "timestamp" in heartbeat


def test_write_heartbeat_overwrites_the_previous_heartbeat(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, interval_seconds=60.0, tick_number=1)

    write_heartbeat(path, interval_seconds=60.0, tick_number=2)

    assert json.loads(path.read_text())["tick_number"] == 2


def test_heartbeat_is_fresh_true_for_a_recent_heartbeat():
    now = datetime.now(timezone.utc)
    heartbeat = {"timestamp": now.isoformat(), "interval_seconds": 60.0}

    assert heartbeat_is_fresh(heartbeat, now) is True


def test_heartbeat_is_fresh_false_once_past_2x_interval_plus_grace():
    now = datetime.now(timezone.utc)
    stale_timestamp = now - timedelta(seconds=60 * 2 + 5 * 60 + 1)
    heartbeat = {"timestamp": stale_timestamp.isoformat(), "interval_seconds": 60.0}

    assert heartbeat_is_fresh(heartbeat, now) is False


def test_heartbeat_is_fresh_tolerates_one_slow_tick_without_going_stale():
    """A single tick that overruns its own interval (e.g. a slow
    Space-Track response) must not itself look like a hang - only
    genuinely missing multiple cycles should."""
    now = datetime.now(timezone.utc)
    slightly_late = now - timedelta(seconds=60 * 1.5)
    heartbeat = {"timestamp": slightly_late.isoformat(), "interval_seconds": 60.0}

    assert heartbeat_is_fresh(heartbeat, now) is True


def test_heartbeat_is_fresh_scales_with_a_real_longer_configured_interval():
    """The default 8-hour cadence must not need a hardcoded threshold -
    the heartbeat's own interval_seconds drives the freshness window."""
    now = datetime.now(timezone.utc)
    eight_hours = 8 * 60 * 60
    just_under_threshold = now - timedelta(seconds=eight_hours * 2 + 5 * 60 - 30)
    heartbeat = {"timestamp": just_under_threshold.isoformat(), "interval_seconds": eight_hours}

    assert heartbeat_is_fresh(heartbeat, now) is True
