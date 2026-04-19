from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class OrchestratorTelemetryEvent:
    event_name: str
    payload: dict[str, Any]


class OrchestratorTelemetrySink(Protocol):
    def emit(self, event_name: str, payload: Mapping[str, Any]) -> None:
        pass


class NoopOrchestratorTelemetrySink:
    def emit(self, event_name: str, payload: Mapping[str, Any]) -> None:
        return


class InMemoryOrchestratorTelemetrySink:
    def __init__(self) -> None:
        self._events: list[OrchestratorTelemetryEvent] = []

    @property
    def events(self) -> list[OrchestratorTelemetryEvent]:
        return list(self._events)

    def emit(self, event_name: str, payload: Mapping[str, Any]) -> None:
        self._events.append(
            OrchestratorTelemetryEvent(
                event_name=event_name,
                payload=dict(payload),
            )
        )
