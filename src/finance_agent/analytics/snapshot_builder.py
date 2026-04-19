from __future__ import annotations

from uuid import uuid4

from finance_agent.analytics.portfolio_metrics import compute_portfolio_metrics
from finance_agent.contracts.entities import AnalyticsSnapshot, PortfolioSnapshot


def build_analytics_snapshot(snapshot: PortfolioSnapshot) -> AnalyticsSnapshot:
    metrics = compute_portfolio_metrics(snapshot)
    partial = snapshot.partial or snapshot.coverage < 1.0 or snapshot.unknown_share > 0.0

    return AnalyticsSnapshot(
        snapshot_id=f"an_{uuid4().hex}",
        user_id=snapshot.user_id,
        account_id=snapshot.account_id,
        as_of_ts=snapshot.as_of_ts,
        metrics=metrics,
        freshness=snapshot.freshness.model_copy(deep=True),
        coverage=snapshot.coverage,
        partial=partial,
    )
