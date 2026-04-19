from __future__ import annotations

import unittest

from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass
from finance_agent.llm import LlmCompletion, LlmProviderError, LlmUsage
from finance_agent.orchestration.intent import DeterministicIntentClassifier, LlmIntentClassifier
from finance_agent.workflow import determine_request_class


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
        "DEFAULT_MODEL_INTENT": "intent-model-v1",
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


class LlmIntentClassifierTests(unittest.TestCase):
    def test_returns_llm_classification_when_payload_is_valid(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        classifier = LlmIntentClassifier(
            llm_client=_FakeLlmClient(
                completion=LlmCompletion(
                    text='{"request_class":"strategy_fit"}',
                    model_id="intent-model-v1",
                    usage=LlmUsage(prompt_tokens=100, completion_tokens=10),
                    retries_used=1,
                )
            ),
            model_routing=ModelRoutingPolicy(runtime_config),
            fallback_classifier=DeterministicIntentClassifier(determine_request_class),
            tokens_max=runtime_config.tokens_intent_max,
            llm_timeout_ms=runtime_config.llm_timeout_ms,
        )

        result = classifier.classify(
            message_text="Проверь соответствие портфеля стратегии",
            correlation_id="corr-intent-1",
        )

        self.assertEqual(result.request_class, RequestClass.STRATEGY_FIT)
        self.assertEqual(result.model_id, "intent-model-v1")
        self.assertEqual(result.llm_tokens_in, 100)
        self.assertEqual(result.llm_tokens_out, 10)
        self.assertEqual(result.retry_count, 1)
        self.assertIsNone(result.fallback_reason)

    def test_falls_back_to_deterministic_classifier_on_llm_error(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        classifier = LlmIntentClassifier(
            llm_client=_FakeLlmClient(error=LlmProviderError("upstream timeout")),
            model_routing=ModelRoutingPolicy(runtime_config),
            fallback_classifier=DeterministicIntentClassifier(determine_request_class),
            tokens_max=runtime_config.tokens_intent_max,
            llm_timeout_ms=runtime_config.llm_timeout_ms,
        )

        result = classifier.classify(
            message_text="Обнови мою инвестиционную стратегию",
            correlation_id="corr-intent-2",
        )

        self.assertEqual(result.request_class, RequestClass.STRATEGY_PARSING)
        self.assertEqual(result.model_id, "intent-model-v1")
        self.assertEqual(result.llm_tokens_in, 0)
        self.assertEqual(result.llm_tokens_out, 0)
        self.assertIsNotNone(result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
