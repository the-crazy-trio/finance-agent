from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.harness.skill_runner import (
    SkillRunner,
    build_local_skill_runner,
    run_harness_tool,
)
from finance_agent.retriever.gateway import GuardedToolGateway
from finance_agent.retriever.storage import InMemoryMemoryStorage, InMemoryUserStorage
from finance_agent.retriever.tinvest_adapter import TInvestReadonlyAdapter


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


def build_strategy_payload() -> dict[str, object]:
    return {
        "client_id": "client-1",
        "base_currency": "RUB",
        "tax_residency": "RU",
        "jurisdiction_constraints": {"allowed_markets": ["MOEX"]},
        "knowledge_experience": {"experience_level": "basic"},
        "financial_profile": {"monthly_income": 100000},
        "goals": [{"goal_id": "retirement"}],
        "risk_preferences": {"risk_tolerance_self_assessed": "balanced"},
        "investable_universe": {"allowed_instruments": ["stocks", "bonds"]},
        "portfolio_policy": {"benchmark": "custom"},
        "explainability_preferences": {"wants_short_reports": True},
    }


class FakeTInvestClient:
    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> dict[str, object]:
        return {
            "account_id": account_id,
            "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
            "positions": [],
            "portfolio_value": 0.0,
            "coverage": 1.0,
        }


def build_skill_runner() -> SkillRunner:
    runtime_config = RuntimeConfig.from_env(build_minimum_env())
    user_storage = InMemoryUserStorage()
    memory_storage = InMemoryMemoryStorage()
    gateway = GuardedToolGateway(
        runtime_config=runtime_config,
        user_storage=user_storage,
        memory_storage=memory_storage,
        portfolio_adapter=TInvestReadonlyAdapter(
            client=FakeTInvestClient(),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
        ),
    )
    return SkillRunner(gateway=gateway)


class SkillRunnerTests(unittest.TestCase):
    def test_lists_required_tool_contracts(self) -> None:
        runner = build_skill_runner()

        tool_names = [item["name"] for item in runner.list_tools()]
        self.assertEqual(
            tool_names,
            [
                "issuer_indicators_show",
                "macro_indicators_show",
                "portfolio_accounts",
                "portfolio_collect",
                "portfolio_show",
                "strategy_fit",
                "strategy_save",
            ],
        )

    def test_generates_correlation_id_when_missing(self) -> None:
        runner = build_skill_runner()

        result = runner.run_tool(
            tool_name="strategy_save",
            arguments={"strategy": build_strategy_payload()},
        )

        self.assertTrue(result["meta"]["correlation_id"].startswith("corr_"))

    def test_uses_incoming_correlation_id(self) -> None:
        runner = build_skill_runner()

        result = runner.run_tool(
            tool_name="strategy_save",
            arguments={"strategy": build_strategy_payload()},
            correlation_id="corr_from_harness",
        )

        self.assertEqual(result["meta"]["correlation_id"], "corr_from_harness")

    def test_does_not_trust_principal_id_from_payload(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        memory_storage = InMemoryMemoryStorage()
        user_storage.grant_account_access("other_user", "acc-1")
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(),
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )
        runner = SkillRunner(gateway=gateway)

        result = runner.run_tool(
            tool_name="portfolio_collect",
            arguments={"account_id": "acc-1", "principal_id": "other_user"},
            correlation_id="corr-scope",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "UNAUTHORIZED_SCOPE")

    def test_run_harness_tool_payload_contract(self) -> None:
        runner = build_skill_runner()

        result = run_harness_tool(
            {
                "tool_name": "strategy_save",
                "arguments": {"strategy": build_strategy_payload()},
                "correlation_id": "corr-payload",
            },
            runner=runner,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["meta"]["correlation_id"], "corr-payload")

    def test_build_local_skill_runner_supports_strategy_save(self) -> None:
        runner = build_local_skill_runner(env=build_minimum_env())

        result = runner.run_tool(
            tool_name="strategy_save",
            arguments={"strategy": build_strategy_payload()},
            correlation_id="corr-local-builder",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["meta"]["correlation_id"], "corr-local-builder")

    def test_build_local_skill_runner_returns_scoped_accounts_when_configured(self) -> None:
        runner = build_local_skill_runner(
            env=build_minimum_env(),
            allowed_account_ids=["acc-1", "acc-2"],
        )

        result = runner.run_tool(
            tool_name="portfolio_accounts",
            arguments={},
            correlation_id="corr-local-accounts",
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["source"], "scope")
        self.assertEqual(result["data"]["account_ids"], ["acc-1", "acc-2"])

    def test_build_local_skill_runner_returns_upstream_error_for_collect_without_client(
        self,
    ) -> None:
        runner = build_local_skill_runner(
            env=build_minimum_env(),
            use_env_tinvest_client=False,
        )

        result = runner.run_tool(
            tool_name="portfolio_collect",
            arguments={},
            correlation_id="corr-no-client",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "UPSTREAM_UNAVAILABLE")

    def test_run_harness_tool_returns_validation_error_without_tool_name(self) -> None:
        runner = build_skill_runner()

        result = run_harness_tool({}, runner=runner)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
