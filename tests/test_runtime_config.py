import unittest

from pydantic import ValidationError

from finance_agent.config.runtime import RuntimeConfig


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


class RuntimeConfigTests(unittest.TestCase):
    def test_loads_from_complete_env(self) -> None:
        config = RuntimeConfig.from_env(build_minimum_env())
        self.assertEqual(config.app_env.value, "dev")
        self.assertEqual(config.max_retries_transient, 2)

    def test_raises_for_missing_required(self) -> None:
        env = build_minimum_env()
        env.pop("OPENROUTER_API_KEY")

        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            RuntimeConfig.from_env(env)

    def test_raises_for_invalid_coverage_range(self) -> None:
        env = build_minimum_env()
        env["STRATEGY_FIT_MIN_COVERAGE"] = "1.2"

        with self.assertRaises(ValidationError):
            RuntimeConfig.from_env(env)

    def test_uses_defaults_for_non_secret_runtime_values(self) -> None:
        config = RuntimeConfig.from_env(build_minimum_env())
        self.assertEqual(config.request_timeout_ms, 120_000)
        self.assertEqual(config.default_model_intent, "openai/gpt-4.1-mini")

    def test_blank_feature_flags_fall_back_to_default(self) -> None:
        env = build_minimum_env()
        env["ENABLE_GUARDED_TEXT2SQL"] = " "
        env["ENABLE_GUARDED_TEXT2API"] = ""
        env["ENABLE_OFFLINE_EVAL_EXPORT"] = "\t"

        config = RuntimeConfig.from_env(env)

        self.assertFalse(config.enable_guarded_text2sql)
        self.assertFalse(config.enable_guarded_text2api)
        self.assertFalse(config.enable_offline_eval_export)


if __name__ == "__main__":
    unittest.main()
