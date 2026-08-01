"""Data-source adapter pattern for pulling telemetry into the pipeline."""
from __future__ import annotations

import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ..schemas import TelemetryEvent


class DataSourceAdapter(ABC):
    """Base class for anything that can produce a batch of telemetry events.

    Concrete adapters (a CSV replay, a live sensor feed, a mission API,
    etc.) implement fetch_batch(). The rest of the pipeline only ever
    talks to this interface.
    """

    @abstractmethod
    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        raise NotImplementedError


class DummyAdapter(DataSourceAdapter):
    """Generates fake telemetry events, useful for smoke tests and demos."""

    def __init__(self, source_name: str = "dummy-sensor"):
        self.source_name = source_name

    def fetch_batch(self, limit: int) -> list[TelemetryEvent]:
        return [
            TelemetryEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                source=self.source_name,
                raw_data={"value": round(random.uniform(0, 100), 2)},
            )
            for _ in range(limit)
        ]
