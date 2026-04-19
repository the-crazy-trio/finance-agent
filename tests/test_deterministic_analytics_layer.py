from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.analytics import build_analytics_snapshot, compute_portfolio_metrics
from finance_agent.contracts.entities import (
    FreshnessMetadata,
    PortfolioPosition,
    PortfolioSnapshot,
)


def build_snapshot(
    *,
    portfolio_value: float,
    positions: list[PortfolioPosition],
    coverage: float = 1.0,
    unknown_share: float = 0.0,
    partial: bool = False,
) -> PortfolioSnapshot:
    as_of_ts = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
    freshness = FreshnessMetadata(
        source_ts=as_of_ts,
        age_minutes=5,
        max_age_minutes=60,
        is_stale=False,
    )
    return PortfolioSnapshot(
        snapshot_id="psnap_1",
        user_id="local_user",
        account_id="acc-1",
        as_of_ts=as_of_ts,
        positions=positions,
        portfolio_value=portfolio_value,
        freshness=freshness,
        coverage=coverage,
        unknown_share=unknown_share,
        partial=partial,
    )


class DeterministicAnalyticsLayerTests(unittest.TestCase):
    def test_weights_hhi_and_top_n_are_computed_from_snapshot(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[
                PortfolioPosition(ticker="AAA", quantity=1, market_value=500.0),
                PortfolioPosition(ticker="BBB", quantity=1, market_value=300.0),
                PortfolioPosition(ticker="CCC", quantity=1, market_value=200.0),
            ],
        )

        metrics = compute_portfolio_metrics(snapshot, top_n=2)

        self.assertEqual(metrics["position_count"], 3)
        self.assertAlmostEqual(metrics["top_n_concentration"], 0.8)
        self.assertAlmostEqual(metrics["hhi"], 0.38)
        self.assertEqual(metrics["weights"][0]["instrument_key"], "AAA")
        self.assertAlmostEqual(metrics["weights"][0]["weight"], 0.5)

    def test_class_exposure_aggregates_by_asset_class(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
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
                    market_value=300.0,
                    asset_class="bond",
                ),
                PortfolioPosition(
                    ticker="CCC",
                    quantity=1,
                    market_value=100.0,
                    asset_class="equity",
                ),
            ],
        )

        metrics = compute_portfolio_metrics(snapshot)

        self.assertAlmostEqual(metrics["class_exposure"]["equity"], 0.7)
        self.assertAlmostEqual(metrics["class_exposure"]["bond"], 0.3)

    def test_unknown_asset_class_is_grouped_as_unknown(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[
                PortfolioPosition(ticker="AAA", quantity=1, market_value=400.0, asset_class=None),
                PortfolioPosition(ticker="BBB", quantity=1, market_value=600.0, asset_class="  "),
            ],
        )

        metrics = compute_portfolio_metrics(snapshot)

        self.assertAlmostEqual(metrics["class_exposure"]["unknown"], 1.0)

    def test_zero_portfolio_value_returns_zeroed_metrics(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=0.0,
            positions=[
                PortfolioPosition(
                    ticker="AAA",
                    quantity=1,
                    market_value=300.0,
                    asset_class="equity",
                ),
                PortfolioPosition(
                    ticker="BBB",
                    quantity=1,
                    market_value=700.0,
                    asset_class="bond",
                ),
            ],
        )

        metrics = compute_portfolio_metrics(snapshot)

        self.assertAlmostEqual(metrics["top_n_concentration"], 0.0)
        self.assertAlmostEqual(metrics["hhi"], 0.0)
        self.assertAlmostEqual(metrics["class_exposure"]["equity"], 0.0)
        self.assertAlmostEqual(metrics["class_exposure"]["bond"], 0.0)

    def test_empty_positions_return_empty_weights_and_zero_concentration(self) -> None:
        snapshot = build_snapshot(portfolio_value=0.0, positions=[])

        metrics = compute_portfolio_metrics(snapshot)

        self.assertEqual(metrics["weights"], [])
        self.assertEqual(metrics["class_exposure"], {})
        self.assertAlmostEqual(metrics["top_n_concentration"], 0.0)
        self.assertAlmostEqual(metrics["hhi"], 0.0)

    def test_weights_are_sorted_descending(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[
                PortfolioPosition(ticker="SMALL", quantity=1, market_value=100.0),
                PortfolioPosition(ticker="BIG", quantity=1, market_value=600.0),
                PortfolioPosition(ticker="MID", quantity=1, market_value=300.0),
            ],
        )

        metrics = compute_portfolio_metrics(snapshot)
        ordered_keys = [item["instrument_key"] for item in metrics["weights"]]

        self.assertEqual(ordered_keys, ["BIG", "MID", "SMALL"])

    def test_build_analytics_snapshot_carries_identity_and_context_fields(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[PortfolioPosition(ticker="AAA", quantity=1, market_value=1000.0)],
            coverage=0.95,
        )

        analytics_snapshot = build_analytics_snapshot(snapshot)

        self.assertTrue(analytics_snapshot.snapshot_id.startswith("an_"))
        self.assertEqual(analytics_snapshot.user_id, snapshot.user_id)
        self.assertEqual(analytics_snapshot.account_id, snapshot.account_id)
        self.assertEqual(analytics_snapshot.as_of_ts, snapshot.as_of_ts)
        self.assertEqual(analytics_snapshot.coverage, snapshot.coverage)
        self.assertEqual(analytics_snapshot.freshness, snapshot.freshness)
        self.assertIsNot(analytics_snapshot.freshness, snapshot.freshness)

    def test_partial_flag_is_true_when_coverage_below_one(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[PortfolioPosition(ticker="AAA", quantity=1, market_value=1000.0)],
            coverage=0.8,
        )

        analytics_snapshot = build_analytics_snapshot(snapshot)

        self.assertTrue(analytics_snapshot.partial)

    def test_partial_flag_is_true_when_unknown_share_positive(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[PortfolioPosition(ticker="AAA", quantity=1, market_value=1000.0)],
            unknown_share=0.1,
        )

        analytics_snapshot = build_analytics_snapshot(snapshot)

        self.assertTrue(analytics_snapshot.partial)

    def test_partial_flag_respects_existing_portfolio_partial(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[PortfolioPosition(ticker="AAA", quantity=1, market_value=1000.0)],
            partial=True,
        )

        analytics_snapshot = build_analytics_snapshot(snapshot)

        self.assertTrue(analytics_snapshot.partial)

    def test_metric_calculation_does_not_mutate_input_snapshot(self) -> None:
        snapshot = build_snapshot(
            portfolio_value=1000.0,
            positions=[
                PortfolioPosition(ticker="AAA", quantity=1, market_value=600.0),
                PortfolioPosition(ticker="BBB", quantity=1, market_value=400.0),
            ],
            coverage=0.9,
            unknown_share=0.1,
            partial=True,
        )
        before = snapshot.model_dump(mode="json")

        compute_portfolio_metrics(snapshot)

        after = snapshot.model_dump(mode="json")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
