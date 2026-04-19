from __future__ import annotations

import unittest

from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorRequest
from finance_agent.observability.metrics import InMemoryOrchestratorMetricsSink
from finance_agent.orchestration import OrchestratorService
from finance_agent.orchestration.intent import IntentClassification
from finance_agent.workflow import WorkflowExecutionResult


class _StubWorkflowEngine:
    def __init__(self, result: WorkflowExecutionResult) -> None:
        self._result = result

    def execute(self, **kwargs: object) -> WorkflowExecutionResult:
        return self._result


class _StubIntentClassifier:
    def classify(self, *, message_text: str, correlation_id: str) -> IntentClassification:
        del message_text, correlation_id
        return IntentClassification(
            request_class=RequestClass.STRATEGY_PARSING,
            model_id="intent-model",
            llm_tokens_in=11,
            llm_tokens_out=7,
            retry_count=1,
        )


class _StubResponseSynthesizer:
    def synthesize(self, **kwargs: object) -> WorkflowExecutionResult:
        workflow_result: WorkflowExecutionResult = kwargs["workflow_result"]
        return WorkflowExecutionResult(
            status=workflow_result.status,
            summary=workflow_result.summary,
            details=list(workflow_result.details),
            evidence=workflow_result.evidence,
            errors=list(workflow_result.errors),
            tool_latency_ms=321,
            llm_tokens_in=5,
            llm_tokens_out=3,
            retry_count=2,
        )


class OrchestratorMetricsIntegrationTests(unittest.TestCase):
    def test_service_records_request_metric(self) -> None:
        sink = InMemoryOrchestratorMetricsSink()
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
            metrics_sink=sink,
        )

        response = service.handle(OrchestratorRequest(message="Обнови стратегию"))

        self.assertEqual(response.status, ResponseStatus.PARTIAL)
        self.assertEqual(len(sink.records), 1)
        metric = sink.records[0]
        self.assertEqual(metric.request_class, RequestClass.STRATEGY_PARSING)
        self.assertEqual(metric.status, ResponseStatus.PARTIAL)
        self.assertGreaterEqual(metric.request_latency_ms, 0)

    def test_service_aggregates_llm_and_retry_metrics(self) -> None:
        sink = InMemoryOrchestratorMetricsSink()
        service = OrchestratorService(
            intent_classifier=_StubIntentClassifier(),
            workflow_engine=_StubWorkflowEngine(
                WorkflowExecutionResult(
                    status=ResponseStatus.OK,
                    summary="ok",
                    details=["workflow=strategy_parsing"],
                    evidence=OrchestratorEvidence(coverage=1.0, unsupported_criteria=[]),
                )
            ),
            response_synthesizer=_StubResponseSynthesizer(),
            metrics_sink=sink,
        )

        response = service.handle(OrchestratorRequest(message="Обнови стратегию"))

        self.assertEqual(response.status, ResponseStatus.OK)
        self.assertEqual(len(sink.records), 1)
        metric = sink.records[0]
        self.assertEqual(metric.tool_latency_ms, 321)
        self.assertEqual(metric.llm_tokens_in, 16)
        self.assertEqual(metric.llm_tokens_out, 10)
        self.assertEqual(metric.retry_count, 3)


if __name__ == "__main__":
    unittest.main()
