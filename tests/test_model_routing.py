import unittest

from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


class ModelRoutingPolicyTests(unittest.TestCase):
    def test_intent_model_is_used_for_intent_purpose(self) -> None:
        env = build_minimum_env()
        env["DEFAULT_MODEL_INTENT"] = "intent-model"
        env["DEFAULT_MODEL_SYNTHESIS"] = "synthesis-model"
        policy = ModelRoutingPolicy(runtime_config=RuntimeConfig.from_env(env))

        for request_class in RequestClass:
            with self.subTest(request_class=request_class.value):
                self.assertEqual(
                    policy.model_for_request(request_class=request_class, purpose="intent"),
                    "intent-model",
                )

    def test_synthesis_model_is_used_for_synthesis_purpose(self) -> None:
        env = build_minimum_env()
        env["DEFAULT_MODEL_INTENT"] = "intent-model"
        env["DEFAULT_MODEL_SYNTHESIS"] = "synthesis-model"
        policy = ModelRoutingPolicy(runtime_config=RuntimeConfig.from_env(env))

        self.assertEqual(
            policy.model_for_request(
                request_class=RequestClass.STRATEGY_PARSING,
                purpose="synthesis",
            ),
            "intent-model",
        )
        for request_class in (
            RequestClass.PORTFOLIO_ANALYSIS,
            RequestClass.STRATEGY_FIT,
            RequestClass.ASSET_ANALYSIS,
            RequestClass.PORTFOLIO_QA,
        ):
            with self.subTest(request_class=request_class.value):
                self.assertEqual(
                    policy.model_for_request(request_class=request_class, purpose="synthesis"),
                    "synthesis-model",
                )

    def test_workflow_model_uses_request_class_matrix(self) -> None:
        env = build_minimum_env()
        env["DEFAULT_MODEL_INTENT"] = "intent-model"
        env["DEFAULT_MODEL_SYNTHESIS"] = "synthesis-model"
        policy = ModelRoutingPolicy(runtime_config=RuntimeConfig.from_env(env))

        self.assertEqual(
            policy.model_for_request(
                request_class=RequestClass.STRATEGY_PARSING,
                purpose="workflow",
            ),
            "intent-model",
        )
        for request_class in (
            RequestClass.PORTFOLIO_ANALYSIS,
            RequestClass.STRATEGY_FIT,
            RequestClass.ASSET_ANALYSIS,
            RequestClass.PORTFOLIO_QA,
        ):
            with self.subTest(request_class=request_class.value):
                self.assertEqual(
                    policy.model_for_request(request_class=request_class, purpose="workflow"),
                    "synthesis-model",
                )


if __name__ == "__main__":
    unittest.main()
