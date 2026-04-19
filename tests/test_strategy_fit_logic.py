from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.analytics import build_analytics_snapshot, evaluate_strategy_fit
from finance_agent.contracts.entities import (
    FreshnessMetadata,
    PortfolioPosition,
    PortfolioSnapshot,
    StructuredStrategy,
)


def build_snapshot(
    *,
    positions: list[PortfolioPosition],
    portfolio_value: float = 1000.0,
    coverage: float = 1.0,
    unknown_share: float = 0.0,
) -> PortfolioSnapshot:
    as_of_ts = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    freshness = FreshnessMetadata(
        source_ts=as_of_ts,
        age_minutes=5,
        max_age_minutes=60,
        is_stale=False,
    )
    return PortfolioSnapshot(
        snapshot_id="psnap_fit_1",
        user_id="local_user",
        account_id="acc-1",
        as_of_ts=as_of_ts,
        positions=positions,
        portfolio_value=portfolio_value,
        freshness=freshness,
        coverage=coverage,
        unknown_share=unknown_share,
    )


def build_strategy(
    *,
    include_max_single: bool = True,
    include_max_sector: bool = True,
    include_target_allocation: bool = True,
    include_drift: bool = True,
    max_single_name_weight: float = 0.7,
    max_sector_weight: float = 0.8,
) -> dict[str, object]:
    risk_preferences: dict[str, object] = {
        "risk_tolerance_self_assessed": "balanced",
    }
    if include_max_single:
        risk_preferences["max_single_name_weight"] = max_single_name_weight
    if include_max_sector:
        risk_preferences["max_sector_weight"] = max_sector_weight

    portfolio_policy: dict[str, object] = {
        "benchmark": "custom",
    }
    if include_target_allocation:
        portfolio_policy["target_asset_allocation"] = {
            "equity": 0.6,
            "bond": 0.4,
        }
    if include_drift:
        portfolio_policy["rebalance_policy"] = {
            "frequency": "quarterly",
            "drift_threshold_pct": 10,
        }

    return {
        "client_id": "client-1",
        "base_currency": "RUB",
        "tax_residency": "RU",
        "jurisdiction_constraints": {"allowed_markets": ["MOEX"]},
        "knowledge_experience": {"experience_level": "basic"},
        "financial_profile": {"monthly_income": 100000},
        "goals": [{"goal_id": "retirement"}],
        "risk_preferences": risk_preferences,
        "investable_universe": {"allowed_instruments": ["stocks", "bonds"]},
        "portfolio_policy": portfolio_policy,
        "explainability_preferences": {"wants_short_reports": True},
    }


class StrategyFitLogicTests(unittest.TestCase):
    def test_coverage_below_threshold_returns_unknown(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=600.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=400.0,
                    asset_class="bond",
                ),
            ],
            coverage=0.5,
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        strategy = StructuredStrategy.model_validate(build_strategy())

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "unknown")
        self.assertEqual(assessment.unsupported_criteria, ["insufficient_coverage"])
        self.assertEqual(diagnostics["evaluated_criteria"], [])

    def test_portfolio_within_limits_returns_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=600.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=400.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        strategy = StructuredStrategy.model_validate(build_strategy())

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "fit")
        self.assertEqual(assessment.unsupported_criteria, [])
        self.assertEqual(diagnostics["violations"], [])

    def test_single_name_breach_returns_not_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=900.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=100.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        strategy = StructuredStrategy.model_validate(build_strategy(max_single_name_weight=0.5))

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "not_fit")
        self.assertIn("max_single_name_weight", diagnostics["violations"])

    def test_missing_criteria_returns_partial_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=600.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=400.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        strategy = StructuredStrategy.model_validate(
            build_strategy(
                include_max_sector=False,
                include_target_allocation=False,
                include_drift=False,
            )
        )

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "partial_fit")
        self.assertIn("max_sector_weight", assessment.unsupported_criteria)
        self.assertIn("target_asset_allocation", assessment.unsupported_criteria)
        self.assertIn("drift_threshold_pct", assessment.unsupported_criteria)
        self.assertEqual(diagnostics["violations"], [])

    def test_hard_rule_one_off_dividend_flag_returns_not_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=600.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=400.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        analytics_snapshot.metrics["hard_rules"] = {
            "one_off_dividend_extrapolated": True,
        }
        strategy = StructuredStrategy.model_validate(build_strategy())

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "not_fit")
        self.assertIn(
            "hard_rule_one_off_dividend_extrapolated",
            diagnostics["hard_rule_violations"],
        )

    def test_hard_rule_cross_country_without_adjustments_returns_not_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=600.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=400.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)
        analytics_snapshot.metrics["hard_rules"] = {
            "cross_country_without_adjustments": True,
        }
        strategy = StructuredStrategy.model_validate(build_strategy())

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "not_fit")
        self.assertIn(
            "hard_rule_cross_country_without_adjustments",
            diagnostics["hard_rule_violations"],
        )

    def test_hard_rule_market_access_violation_returns_not_fit(self) -> None:
        snapshot = build_snapshot(
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=900.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=100.0,
                    asset_class="bond",
                ),
            ],
        )
        analytics_snapshot = build_analytics_snapshot(snapshot)

        strategy_payload = build_strategy(max_single_name_weight=0.95)
        strategy_payload["investable_universe"]["allowed_instruments"] = ["bonds"]
        strategy_payload["investable_universe"]["disallowed_instruments"] = ["stocks"]
        strategy = StructuredStrategy.model_validate(strategy_payload)

        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=0.7,
        )

        self.assertEqual(assessment.verdict, "not_fit")
        self.assertIn(
            "hard_rule_inaccessible_market_primary_driver",
            diagnostics["hard_rule_violations"],
        )


if __name__ == "__main__":
    unittest.main()
