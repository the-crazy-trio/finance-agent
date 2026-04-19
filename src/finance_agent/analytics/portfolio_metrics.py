from __future__ import annotations

from collections import defaultdict
from typing import Any

from finance_agent.contracts.entities import PortfolioPosition, PortfolioSnapshot


def compute_portfolio_metrics(
    snapshot: PortfolioSnapshot,
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    if top_n < 0:
        raise ValueError("top_n must be greater than or equal to 0")

    weights: list[dict[str, float | str]] = []
    class_exposure: dict[str, float] = defaultdict(float)

    for position in snapshot.positions:
        instrument_key = _resolve_instrument_key(position)
        weight = _safe_weight(position.market_value, snapshot.portfolio_value)
        weights.append({"instrument_key": instrument_key, "weight": weight})

        asset_class = _normalize_asset_class(position.asset_class)
        class_exposure[asset_class] += weight

    sorted_weights = sorted(
        weights,
        key=lambda item: (-float(item["weight"]), str(item["instrument_key"])),
    )

    top_n_concentration = sum(float(item["weight"]) for item in sorted_weights[:top_n])
    hhi = sum(float(item["weight"]) ** 2 for item in sorted_weights)

    return {
        "weights": sorted_weights,
        "top_n_concentration": top_n_concentration,
        "hhi": hhi,
        "class_exposure": dict(class_exposure),
        "position_count": len(snapshot.positions),
        "unknown_share": snapshot.unknown_share,
    }


def _safe_weight(position_value: float, portfolio_value: float) -> float:
    if portfolio_value <= 0.0:
        return 0.0
    return position_value / portfolio_value


def _normalize_asset_class(asset_class: str | None) -> str:
    if asset_class is None:
        return "unknown"
    normalized = asset_class.strip()
    if not normalized:
        return "unknown"
    return normalized


def _resolve_instrument_key(position: PortfolioPosition) -> str:
    if position.instrument_uid:
        return position.instrument_uid
    if position.figi:
        return position.figi
    if position.ticker:
        return position.ticker

    raise ValueError("at least one instrument identifier must be present")
