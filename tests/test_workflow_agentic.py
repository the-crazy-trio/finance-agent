from __future__ import annotations

import unittest

from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass
from finance_agent.llm import LlmCompletion, LlmProviderError, LlmUsage
from finance_agent.workflow.agentic import DeterministicWorkflowAgent, LlmWorkflowAgent


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
        "DEFAULT_MODEL_INTENT": "intent-model",
    }


class _FakeLlmClient:
    def __init__(
        self,
        completion: LlmCompletion | None = None,
        error: Exception | None = None,
    ) -> None:
        self._completion = completion
        self._error = error

    def complete(self, **_: object) -> LlmCompletion:
        if self._error is not None:
            raise self._error
        if self._completion is None:
            raise AssertionError("fake completion is not configured")
        return self._completion


class WorkflowAgenticTests(unittest.TestCase):
    def test_deterministic_snapshot_prefers_collect_for_refresh_requests(self) -> None:
        agent = DeterministicWorkflowAgent()

        directive = agent.snapshot_directive(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            message_text="Обнови и пересчитай портфель",
            correlation_id="corr-1",
        )

        self.assertEqual(directive.primary_tool, "portfolio_collect")

    def test_deterministic_portfolio_qa_interprets_bond_share(self) -> None:
        agent = DeterministicWorkflowAgent()

        directive = agent.portfolio_qa_directive(
            message_text="Какая доля облигаций в портфеле?",
            correlation_id="corr-2",
        )

        self.assertEqual(directive.intent.kind, "class_share")
        self.assertEqual(directive.intent.asset_class, "bond")
        self.assertEqual(directive.primary_tool, "portfolio_show")

    def test_llm_portfolio_qa_uses_llm_payload(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        agent = LlmWorkflowAgent(
            llm_client=_FakeLlmClient(
                completion=LlmCompletion(
                    text='{"kind":"position_count","asset_class":null,'
                    '"primary_tool":"portfolio_collect","prefer_aggregate":true}',
                    model_id="intent-model",
                    usage=LlmUsage(prompt_tokens=12, completion_tokens=6),
                    retries_used=1,
                )
            ),
            model_routing=ModelRoutingPolicy(runtime_config),
            deterministic_fallback=DeterministicWorkflowAgent(),
            tokens_max=runtime_config.tokens_intent_max,
            llm_timeout_ms=runtime_config.llm_timeout_ms,
        )

        directive = agent.portfolio_qa_directive(
            message_text="Сколько бумаг в портфеле?",
            correlation_id="corr-3",
        )

        self.assertEqual(directive.intent.kind, "position_count")
        self.assertEqual(directive.primary_tool, "portfolio_collect")
        self.assertTrue(directive.prefer_aggregate)
        self.assertEqual(directive.signal.mode, "llm")
        self.assertEqual(directive.signal.llm_tokens_in, 12)

    def test_llm_fallback_to_deterministic_when_provider_fails(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        agent = LlmWorkflowAgent(
            llm_client=_FakeLlmClient(error=LlmProviderError("timeout")),
            model_routing=ModelRoutingPolicy(runtime_config),
            deterministic_fallback=DeterministicWorkflowAgent(),
            tokens_max=runtime_config.tokens_intent_max,
            llm_timeout_ms=runtime_config.llm_timeout_ms,
        )

        directive = agent.asset_analysis_directive(
            message_text="Сделай анализ актива SBER",
            correlation_id="corr-4",
        )

        self.assertEqual(directive.ticker, "SBER")
        self.assertEqual(directive.signal.mode, "deterministic_fallback")
        self.assertIsNotNone(directive.signal.fallback_reason)


if __name__ == "__main__":
    unittest.main()
