from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AssetAnalysisResult:
    ticker: str
    weight: float
    market_value: float


def extract_ticker(message_text: str) -> str | None:
    candidates = re.findall(r"\b[A-Z]{2,12}\b", message_text.upper())
    if not candidates:
        return None
    return candidates[-1]


def analyze_asset_in_snapshot(
    *,
    snapshot: Mapping[str, Any],
    ticker: str,
) -> AssetAnalysisResult | None:
    positions = _positions(snapshot)
    portfolio_total = _portfolio_total(snapshot, positions)
    if portfolio_total <= 0:
        portfolio_total = 0.0

    for position in positions:
        position_ticker = str(position.get("ticker") or "").strip().upper()
        if position_ticker != ticker.upper():
            continue

        market_value = _to_float(position.get("market_value"))
        weight = market_value / portfolio_total if portfolio_total > 0 else 0.0
        return AssetAnalysisResult(
            ticker=ticker.upper(),
            weight=weight,
            market_value=market_value,
        )

    return None


def _positions(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_positions = snapshot.get("positions")
    if not isinstance(raw_positions, list):
        return []

    positions: list[Mapping[str, Any]] = []
    for item in raw_positions:
        if isinstance(item, Mapping):
            positions.append(item)
    return positions


def _portfolio_total(snapshot: Mapping[str, Any], positions: list[Mapping[str, Any]]) -> float:
    raw_portfolio_value = snapshot.get("portfolio_value")
    portfolio_value = _to_float(raw_portfolio_value)
    if portfolio_value > 0:
        return portfolio_value
    return sum(_to_float(item.get("market_value")) for item in positions)


def _to_float(raw_value: Any) -> float:
    if isinstance(raw_value, bool) or raw_value is None:
        return 0.0
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0
