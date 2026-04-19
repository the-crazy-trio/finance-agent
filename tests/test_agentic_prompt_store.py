from __future__ import annotations

import unittest

from finance_agent.agentic import SubagentKind, build_default_prompt_store
from finance_agent.contracts.common import RequestClass


class AgenticPromptStoreTests(unittest.TestCase):
    def test_default_store_has_workflow_agent_prompt_per_request_class(self) -> None:
        store = build_default_prompt_store()

        for request_class in RequestClass:
            template = store.resolve(
                kind=SubagentKind.WORKFLOW_AGENT,
                request_class=request_class,
            )
            self.assertEqual(template.subagent_id, request_class.value)
            self.assertTrue(template.guidelines)

    def test_response_synthesis_uses_specific_template_when_available(self) -> None:
        store = build_default_prompt_store()

        strategy_template = store.resolve(
            kind=SubagentKind.RESPONSE_SYNTHESIS,
            request_class=RequestClass.STRATEGY_PARSING,
        )
        portfolio_template = store.resolve(
            kind=SubagentKind.RESPONSE_SYNTHESIS,
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
        )

        self.assertEqual(strategy_template.subagent_id, RequestClass.STRATEGY_PARSING.value)
        self.assertEqual(portfolio_template.subagent_id, "global")

    def test_portfolio_workflow_prompt_mentions_account_discovery_step(self) -> None:
        store = build_default_prompt_store()

        template = store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
        )

        combined = "\n".join(template.guidelines)
        self.assertIn("portfolio_accounts", combined)
        self.assertIn("account_id", combined)


if __name__ == "__main__":
    unittest.main()
