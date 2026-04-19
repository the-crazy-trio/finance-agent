from __future__ import annotations

import unittest

from finance_agent.harness.mcp_server import (
    _build_tinvest_client,
    build_default_mcp_server,
)
from finance_agent.retriever.tinvest_sdk_client import TInvestSdkReadonlyClient


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


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


class McpServerTests(unittest.TestCase):
    def test_initialize_options_work_without_runtime_secrets_in_env(self) -> None:
        server = build_default_mcp_server(env={})
        options = server.initialization_options()

        self.assertEqual(options.server_name, "finance-agent")
        self.assertEqual(options.server_version, "0.1.0")

    def test_initialize_options_expose_tools_capabilities(self) -> None:
        server = build_default_mcp_server(env=build_minimum_env())
        options = server.initialization_options()

        self.assertIsNotNone(options.capabilities.tools)
        self.assertFalse(options.capabilities.tools.listChanged)

    def test_tools_list_contains_guarded_tools(self) -> None:
        server = build_default_mcp_server(env=build_minimum_env())
        tools = server.list_tools()
        tool_names = [tool.name for tool in tools]
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

    def test_tools_call_returns_structured_content(self) -> None:
        server = build_default_mcp_server(env=build_minimum_env())

        result = server.call_tool(
            name="strategy_save",
            arguments={"strategy": build_strategy_payload()},
            meta=None,
        )

        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["status"], "ok")
        self.assertIn("meta", result.structuredContent)

    def test_tools_call_accepts_meta_correlation_id(self) -> None:
        server = build_default_mcp_server(env=build_minimum_env())

        result = server.call_tool(
            name="strategy_save",
            arguments={"strategy": build_strategy_payload()},
            meta={"correlation_id": "corr_mcp_123"},
        )

        self.assertEqual(
            result.structuredContent["meta"]["correlation_id"],
            "corr_mcp_123",
        )

    def test_tools_call_requires_non_empty_name(self) -> None:
        server = build_default_mcp_server(env=build_minimum_env())

        result = server.call_tool(
            name="   ",
            arguments={"strategy": build_strategy_payload()},
            meta=None,
        )

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["status"], "error")

    def test_build_tinvest_client_uses_token_only_when_present(self) -> None:
        self.assertIsNone(_build_tinvest_client({}))

        client = _build_tinvest_client({"TINVEST_READONLY_TOKEN": "token-1"})
        self.assertIsNotNone(client)
        self.assertIsInstance(client, TInvestSdkReadonlyClient)

    def test_build_tinvest_client_supports_grpc_target_override(self) -> None:
        client = _build_tinvest_client(
            {
                "TINVEST_READONLY_TOKEN": "token-1",
                "TINVEST_GRPC_TARGET": "invest-public-api.tinkoff.ru:443",
            }
        )
        self.assertIsNotNone(client)

    def test_default_mcp_server_without_token_keeps_unconfigured_client(self) -> None:
        server = build_default_mcp_server(env={})

        result = server.call_tool(
            name="portfolio_collect",
            arguments={},
            meta={"correlation_id": "corr-no-token"},
        )

        self.assertTrue(result.isError)
        self.assertEqual(result.structuredContent["errors"][0]["code"], "UPSTREAM_UNAVAILABLE")
        self.assertIn(
            "not configured",
            result.structuredContent["errors"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()
