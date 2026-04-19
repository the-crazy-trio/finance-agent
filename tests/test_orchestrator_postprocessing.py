from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any

from finance_agent.contracts.common import (
    AuthMode,
    CorrelationContext,
    RequestClass,
    ResponseStatus,
)
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorRequest
from finance_agent.llm import LlmCompletion, LlmProviderError, LlmUsage
from finance_agent.orchestration.persistence import OrchestratorPersistenceStage
from finance_agent.orchestration.synthesis import OrchestratorResponseSynthesizer
from finance_agent.retriever.storage import InMemoryMemoryStorage
from finance_agent.workflow import WorkflowExecutionResult


def build_context() -> CorrelationContext:
    return CorrelationContext(
        correlation_id="corr-1",
        principal_id="local_user",
        auth_mode=AuthMode.LOCAL,
        request_id="req-1",
    )


class _StubLlmClient:
    def __init__(
        self,
        completion: LlmCompletion | None = None,
        error: Exception | None = None,
    ) -> None:
        self._completion = completion
        self._error = error

    def complete(self, **_: Any) -> LlmCompletion:
        if self._error is not None:
            raise self._error
        if self._completion is None:
            raise AssertionError("stub completion is not configured")
        return self._completion


class _StubModelRouting:
    def synthesis_model(self, request_class: RequestClass) -> str:
        del request_class
        return "synthesis-model-v1"


class OrchestratorPostprocessingTests(unittest.TestCase):
    def test_synthesizer_appends_disclaimer_and_stage_markers(self) -> None:
        synthesizer = OrchestratorResponseSynthesizer()
        request = OrchestratorRequest(message="Покажи портфель")
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="Портфель загружен.",
            details=["workflow=portfolio_analysis"],
            evidence=OrchestratorEvidence(coverage=1.0, unsupported_criteria=[]),
        )

        synthesized = synthesizer.synthesize(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(synthesized.status, ResponseStatus.OK)
        self.assertIn("synthesis=deterministic_stub", synthesized.details)
        self.assertIn("prompt_version=1.0.0", synthesized.details)
        self.assertIn("disclaimer=Не является инвестиционной рекомендацией.", synthesized.details)

    def test_synthesizer_uses_llm_when_client_is_configured(self) -> None:
        synthesizer = OrchestratorResponseSynthesizer(
            llm_client=_StubLlmClient(
                completion=LlmCompletion(
                    text="LLM summary",
                    model_id="synthesis-model-v1",
                    usage=LlmUsage(prompt_tokens=120, completion_tokens=30),
                    retries_used=1,
                )
            ),
            model_routing=_StubModelRouting(),
        )
        request = OrchestratorRequest(message="Покажи портфель")
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="Портфель загружен.",
            details=["workflow=portfolio_analysis"],
            evidence=OrchestratorEvidence(coverage=1.0, unsupported_criteria=[]),
        )

        synthesized = synthesizer.synthesize(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(synthesized.summary, "LLM summary")
        self.assertIn("synthesis=llm", synthesized.details)
        self.assertIn("synthesis.model_id=synthesis-model-v1", synthesized.details)
        self.assertEqual(synthesized.llm_tokens_in, 120)
        self.assertEqual(synthesized.llm_tokens_out, 30)
        self.assertEqual(synthesized.retry_count, 1)

    def test_synthesizer_falls_back_to_deterministic_on_llm_error(self) -> None:
        synthesizer = OrchestratorResponseSynthesizer(
            llm_client=_StubLlmClient(error=LlmProviderError("timeout")),
            model_routing=_StubModelRouting(),
        )
        request = OrchestratorRequest(message="Покажи портфель")
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="Портфель загружен.",
            details=["workflow=portfolio_analysis"],
            evidence=OrchestratorEvidence(coverage=1.0, unsupported_criteria=[]),
        )

        synthesized = synthesizer.synthesize(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(synthesized.summary, "Портфель загружен.")
        self.assertIn("synthesis=deterministic_stub", synthesized.details)

    def test_persistence_stage_saves_summary_record_in_memory(self) -> None:
        memory_storage = InMemoryMemoryStorage()
        persistence = OrchestratorPersistenceStage(
            memory_storage=memory_storage,
            now_fn=lambda: datetime(2026, 4, 18, 12, 0, tzinfo=UTC),
        )
        request = OrchestratorRequest(
            message="Покажи портфель",
            session_id="session-1",
        )
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="Частичный ответ",
            details=["workflow=portfolio_analysis"],
            evidence=OrchestratorEvidence(coverage=0.8, unsupported_criteria=[]),
        )

        persisted = persistence.persist(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertIn("persist=memory_saved", persisted.details)
        saved_records = memory_storage.get_workflow_summaries("local_user")
        self.assertEqual(len(saved_records), 1)
        self.assertEqual(saved_records[0]["request_class"], "portfolio_analysis")
        self.assertEqual(saved_records[0]["status"], "partial")


if __name__ == "__main__":
    unittest.main()
