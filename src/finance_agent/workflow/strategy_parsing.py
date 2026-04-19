from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "equity": ("акци", "stocks", "stock", "equity"),
    "bond": ("облигац", "bonds", "bond"),
    "cash": ("кэш", "кеш", "cash", "налич"),
}

_RISK_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("aggressive", ("aggressive", "агрессив", "высокий риск")),
    ("balanced", ("balanced", "сбаланс", "умерен")),
    ("growth", ("growth", "рост", "growth-")),
    ("conservative", ("conservative", "консерват", "низкий риск")),
)


@dataclass(frozen=True, slots=True)
class StrategyParsingDraft:
    strategy_payload: dict[str, Any] | None
    risk_tolerance: str | None
    target_asset_allocation: dict[str, float]
    missing_slots: list[str]
    clarification_question: str | None


def parse_strategy_message(
    *,
    message_text: str,
    principal_id: str,
) -> StrategyParsingDraft:
    normalized = _normalize(message_text)
    risk_tolerance = _parse_risk_tolerance(normalized)
    target_asset_allocation = _parse_target_asset_allocation(normalized)

    return build_strategy_draft(
        principal_id=principal_id,
        risk_tolerance=risk_tolerance,
        target_asset_allocation=target_asset_allocation,
    )


def build_strategy_draft(
    *,
    principal_id: str,
    risk_tolerance: str | None,
    target_asset_allocation: dict[str, float],
) -> StrategyParsingDraft:
    normalized_allocation = _normalize_allocation(target_asset_allocation)

    missing_slots: list[str] = []
    if risk_tolerance is None:
        missing_slots.append("risk_tolerance_self_assessed")
    if not normalized_allocation:
        missing_slots.append("target_asset_allocation")

    clarification_question = _build_clarification_question(missing_slots)
    if clarification_question is not None or risk_tolerance is None:
        return StrategyParsingDraft(
            strategy_payload=None,
            risk_tolerance=risk_tolerance,
            target_asset_allocation=normalized_allocation,
            missing_slots=missing_slots,
            clarification_question=clarification_question,
        )

    return StrategyParsingDraft(
        strategy_payload=build_strategy_payload(
            principal_id=principal_id,
            risk_tolerance=risk_tolerance,
            target_asset_allocation=normalized_allocation,
        ),
        risk_tolerance=risk_tolerance,
        target_asset_allocation=normalized_allocation,
        missing_slots=[],
        clarification_question=None,
    )


def build_strategy_payload(
    *,
    principal_id: str,
    risk_tolerance: str,
    target_asset_allocation: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "client_id": principal_id,
        "base_currency": "RUB",
        "tax_residency": "RU",
        "jurisdiction_constraints": {
            "allowed_markets": ["MOEX"],
            "forbidden_markets": [],
            "sanctions_or_broker_access_notes": "",
        },
        "knowledge_experience": {
            "experience_level": "basic",
            "instruments_used": ["stocks", "bonds", "money_market_funds"],
            "understands_leverage": False,
            "understands_duration_risk": True,
        },
        "financial_profile": {
            "monthly_income": 0,
            "monthly_expenses": 0,
            "liquid_reserve_months": 6,
            "emergency_fund_ready": True,
            "debt_load_level": "low",
        },
        "goals": [
            {
                "goal_id": "long_term_growth",
                "goal_name": "Долгосрочный капитал",
                "priority": 1,
                "target_amount": 0,
                "target_date": None,
                "horizon_months": 120,
                "contribution_plan": {
                    "initial_amount": 0,
                    "monthly_contribution": 0,
                },
                "required_liquidity_windows": [],
                "target_return_nominal": None,
                "target_return_real": None,
                "max_drawdown_pct": 25,
                "acceptable_volatility_pct": 18,
                "loss_tolerance_rule": "готов терпеть просадку до 25% без продажи",
                "income_need": False,
                "dividend_cash_need": False,
            }
        ],
        "risk_preferences": {
            "risk_tolerance_self_assessed": risk_tolerance,
            "risk_capacity_model_score": 0.0,
            "max_single_name_weight": 0.1,
            "max_sector_weight": 0.25,
            "max_country_weight": 1.0,
            "max_fx_unhedged_weight": 0.3,
            "illiquid_assets_allowed": False,
            "leverage_allowed": False,
        },
        "investable_universe": {
            "allowed_instruments": ["stocks", "bonds", "money_market_funds"],
            "disallowed_instruments": ["options", "structured_products"],
            "preferred_sectors": [],
            "excluded_sectors": [],
            "dividend_preference": "balanced",
            "esg_or_personal_restrictions": [],
        },
        "portfolio_policy": {
            "benchmark": "custom",
            "target_asset_allocation": target_asset_allocation,
            "rebalance_policy": {
                "frequency": "quarterly",
                "drift_threshold_pct": 5,
            },
            "buy_rules": {
                "min_expected_return_spread_vs_cash_pct": 4,
                "required_thesis_confidence": 0.65,
            },
            "sell_rules": {
                "thesis_broken": True,
                "better_opportunity_threshold": 0.15,
                "max_position_review_days": 90,
            },
        },
        "explainability_preferences": {
            "wants_short_reports": False,
            "wants_scenarios": True,
            "wants_probabilities": True,
        },
    }


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_risk_tolerance(text: str) -> str | None:
    for risk_tolerance, markers in _RISK_MARKERS:
        if any(marker in text for marker in markers):
            return risk_tolerance
    return None


def _parse_target_asset_allocation(text: str) -> dict[str, float]:
    allocation: dict[str, float] = {}
    for asset_class, aliases in _ASSET_ALIASES.items():
        weight = _extract_weight_for_aliases(text, aliases)
        if weight is not None:
            allocation[asset_class] = weight
    return allocation


def _extract_weight_for_aliases(text: str, aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        weight = _extract_weight_with_alias(text, alias)
        if weight is not None:
            return weight
    return None


def _extract_weight_with_alias(text: str, alias: str) -> float | None:
    escaped = re.escape(alias)
    patterns = (
        rf"{escaped}[^\d]{{0,20}}(\d{{1,3}}(?:[\.,]\d+)?)\s*%?",
        rf"(\d{{1,3}}(?:[\.,]\d+)?)\s*%?[^\d]{{0,20}}{escaped}",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        raw_value = match.group(1).replace(",", ".")
        try:
            numeric_value = float(raw_value)
        except ValueError:
            continue
        return _to_weight(numeric_value)
    return None


def _to_weight(raw_value: float) -> float | None:
    if raw_value < 0:
        return None
    if raw_value <= 1:
        return raw_value
    if raw_value <= 100:
        return raw_value / 100.0
    return None


def _normalize_allocation(target_asset_allocation: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in ("equity", "bond", "cash"):
        raw_value = target_asset_allocation.get(key)
        if isinstance(raw_value, bool):
            continue
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            continue
        weight = _to_weight(numeric)
        if weight is None:
            continue
        normalized[key] = weight
    return normalized


def _build_clarification_question(missing_slots: list[str]) -> str | None:
    missing_set = set(missing_slots)
    if not missing_set:
        return None
    if missing_set == {"risk_tolerance_self_assessed", "target_asset_allocation"}:
        return (
            "Уточните риск-профиль (conservative|balanced|growth|aggressive) "
            "и целевую аллокацию (акции/облигации/кэш)."
        )
    if "risk_tolerance_self_assessed" in missing_set:
        return "Уточните риск-профиль (conservative|balanced|growth|aggressive)."
    return "Уточните целевую аллокацию (акции/облигации/кэш)."
