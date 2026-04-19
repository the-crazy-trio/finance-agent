from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.contracts.common import AuthMode, CorrelationContext, RequestClass
from finance_agent.contracts.entities import (
    FreshnessMetadata,
    PortfolioPosition,
    PortfolioSnapshot,
    StructuredStrategy,
)
from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.orchestration.context_loader import OrchestratorContextLoader
from finance_agent.retriever.storage import InMemoryMemoryStorage


def build_context() -> CorrelationContext:
    return CorrelationContext(
        correlation_id="corr-ctx",
        principal_id="local_user",
        auth_mode=AuthMode.LOCAL,
        request_id="req-ctx",
    )


def build_strategy() -> StructuredStrategy:
    return StructuredStrategy(
        client_id="local_user",
        base_currency="RUB",
        tax_residency="RU",
        jurisdiction_constraints={"allowed_markets": ["MOEX"]},
        knowledge_experience={"experience_level": "basic"},
        financial_profile={"monthly_income": 100000},
        goals=[{"goal_id": "retirement"}],
        risk_preferences={"risk_tolerance_self_assessed": "balanced"},
        investable_universe={"allowed_instruments": ["stocks", "bonds"]},
        portfolio_policy={"benchmark": "custom"},
        explainability_preferences={"wants_short_reports": True},
    )


def build_snapshot() -> PortfolioSnapshot:
    as_of_ts = datetime(2026, 4, 18, 11, 0, tzinfo=UTC)
    return PortfolioSnapshot(
        snapshot_id="psnap-1",
        user_id="local_user",
        account_id="acc-1",
        as_of_ts=as_of_ts,
        positions=[
            PortfolioPosition(ticker="SBER", quantity=1.0, market_value=100.0, asset_class="equity")
        ],
        portfolio_value=100.0,
        freshness=FreshnessMetadata(
            source_ts=as_of_ts,
            age_minutes=5,
            max_age_minutes=60,
            is_stale=False,
        ),
        coverage=1.0,
        unknown_share=0.0,
        partial=False,
    )


class OrchestratorContextLoaderTests(unittest.TestCase):
    def test_noop_loader_without_memory_storage(self) -> None:
        loader = OrchestratorContextLoader()
        request = OrchestratorRequest(message="Покажи портфель", account_id="acc-1")

        loaded = loader.load(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
        )

        self.assertEqual(loaded.details, ["context_load=noop"])
        self.assertFalse(loaded.strategy_available)
        self.assertFalse(loaded.snapshot_available)

    def test_loader_reads_strategy_and_snapshot_metadata(self) -> None:
        storage = InMemoryMemoryStorage()
        storage.save_strategy("local_user", build_strategy())
        storage.save_portfolio_snapshot(build_snapshot())

        loader = OrchestratorContextLoader(memory_storage=storage)
        request = OrchestratorRequest(message="Покажи портфель", account_id="acc-1")

        loaded = loader.load(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=build_context(),
        )

        self.assertTrue(loaded.strategy_available)
        self.assertTrue(loaded.snapshot_available)
        self.assertEqual(loaded.snapshot_account_id, "acc-1")
        self.assertEqual(loaded.snapshot_as_of_ts, datetime(2026, 4, 18, 11, 0, tzinfo=UTC))
        self.assertIn("context.strategy=available", loaded.details)
        self.assertIn("context.snapshot=available", loaded.details)


if __name__ == "__main__":
    unittest.main()
