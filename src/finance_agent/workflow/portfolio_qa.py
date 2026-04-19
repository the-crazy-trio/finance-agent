from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PortfolioQaIntent:
    kind: str
    asset_class: str | None = None


def interpret_portfolio_question(message_text: str) -> PortfolioQaIntent:
    normalized = message_text.strip().lower()

    if _contains_any(normalized, ("сколько", "количество")) and _contains_any(
        normalized,
        ("позиц", "бумаг", "тикер"),
    ):
        return PortfolioQaIntent(kind="position_count")

    if _contains_any(normalized, ("доля", "share")):
        if _contains_any(normalized, ("облигац", "bond")):
            return PortfolioQaIntent(kind="class_share", asset_class="bond")
        if _contains_any(normalized, ("акци", "equity", "stock")):
            return PortfolioQaIntent(kind="class_share", asset_class="equity")
        if _contains_any(normalized, ("кэш", "кеш", "cash", "money")):
            return PortfolioQaIntent(kind="class_share", asset_class="cash")

    return PortfolioQaIntent(kind="unknown")


def answer_portfolio_question(
    *,
    snapshot: Mapping[str, Any],
    intent: PortfolioQaIntent,
) -> str | None:
    if intent.kind == "position_count":
        count = _positions_count(snapshot)
        return f"Количество позиций в портфеле: {count}."

    if intent.kind == "class_share" and intent.asset_class is not None:
        share = _class_share(snapshot=snapshot, target_class=intent.asset_class)
        label = {
            "bond": "облигаций",
            "equity": "акций",
            "cash": "кэша",
        }.get(intent.asset_class, intent.asset_class)
        return f"Доля {label} в портфеле: {share * 100:.1f}%."

    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _positions_count(snapshot: Mapping[str, Any]) -> int:
    raw_positions = snapshot.get("positions")
    if isinstance(raw_positions, list):
        return len(raw_positions)
    return 0


def _class_share(*, snapshot: Mapping[str, Any], target_class: str) -> float:
    positions = _positions(snapshot)
    total_value = _portfolio_total(snapshot, positions)
    if total_value <= 0:
        return 0.0

    target_value = 0.0
    for position in positions:
        asset_class = _normalize_asset_class(position.get("asset_class"))
        if asset_class != target_class:
            continue
        target_value += _to_float(position.get("market_value"))

    return target_value / total_value


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
    snapshot_total = _to_float(snapshot.get("portfolio_value"))
    if snapshot_total > 0:
        return snapshot_total
    return sum(_to_float(position.get("market_value")) for position in positions)


def _normalize_asset_class(raw_value: Any) -> str:
    text = str(raw_value or "").strip().lower()
    if text in {"bond", "bonds", "облигации", "облигация"}:
        return "bond"
    if text in {"equity", "stock", "stocks", "акции", "акция"}:
        return "equity"
    if text in {"cash", "money_market", "money_market_funds", "кэш", "кеш"}:
        return "cash"
    return text


def _to_float(raw_value: Any) -> float:
    if isinstance(raw_value, bool):
        return 0.0
    if raw_value is None:
        return 0.0
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0
