from __future__ import annotations

import unittest

from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorRequest
from finance_agent.orchestration import OrchestratorService
from finance_agent.orchestration.telemetry import InMemoryOrchestratorTelemetrySink
from finance_agent.workflow import WorkflowExecutionResult


class _StubWorkflowEngine:
    def __init__(self, result: WorkflowExecutionResult) -> None:
        self._result = result

    def execute(self, **kwargs: object) -> WorkflowExecutionResult:
        return self._result


class OrchestratorTelemetryTests(unittest.TestCase):
    def test_service_emits_lifecycle_events_with_correlation(self) -> None:
        sink = InMemoryOrchestratorTelemetrySink()
        service = OrchestratorService(
            request_classifier=lambda _: RequestClass.STRATEGY_PARSING,
            workflow_engine=_StubWorkflowEngine(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Нужны уточнения",
                    details=["workflow=strategy_parsing"],
                    evidence=OrchestratorEvidence(coverage=1.0, unsupported_criteria=[]),
                )
            ),
            telemetry_sink=sink,
        )
        request = OrchestratorRequest(message="Обнови стратегию", correlation_id="corr-telemetry")

        response = service.handle(request)

        self.assertEqual(response.correlation_id, "corr-telemetry")
        event_names = [event.event_name for event in sink.events]
        self.assertEqual(
            event_names,
            [
                "intake_started",
                "context_loaded",
                "workflow_executed",
                "validation_finished",
                "synthesis_finished",
                "persist_finished",
                "response_ready",
            ],
        )

        event_corr = {event.payload.get("correlation_id") for event in sink.events}
        self.assertEqual(event_corr, {"corr-telemetry"})


if __name__ == "__main__":
    unittest.main()
