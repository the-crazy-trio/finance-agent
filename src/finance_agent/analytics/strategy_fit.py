from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from finance_agent.contracts.entities import (
    AnalyticsSnapshot,
    StrategyFitAssessment,
    StrategyFitVerdict,
    StructuredStrategy,
)


def evaluate_strategy_fit(
    *,
    analytics_snapshot: AnalyticsSnapshot,
    strategy: StructuredStrategy,
    min_coverage: float,
) -> tuple[StrategyFitAssessment, dict[str, Any]]:
    hard_rule_violations: list[str] = []

    if analytics_snapshot.coverage < min_coverage:
        return (
            StrategyFitAssessment(
                verdict=StrategyFitVerdict.UNKNOWN,
                unsupported_criteria=["insufficient_coverage"],
                reason="Coverage is below configured threshold",
            ),
            {
                "evaluated_criteria": [],
                "violations": [],
                "hard_rule_violations": [],
            },
        )

    weights = _extract_weights(analytics_snapshot.metrics)
    class_exposure = _extract_class_exposure(analytics_snapshot.metrics)

    unsupported: list[str] = []
    evaluated: list[str] = []
    violations: list[str] = []

    max_single_name = _read_float(strategy.risk_preferences.get("max_single_name_weight"))
    if max_single_name is None:
        unsupported.append("max_single_name_weight")
    else:
        evaluated.append("max_single_name_weight")
        max_observed_weight = max(weights, default=0.0)
        if max_observed_weight > max_single_name:
            violations.append("max_single_name_weight")

    max_sector_weight = _read_float(strategy.risk_preferences.get("max_sector_weight"))
    if max_sector_weight is None:
        unsupported.append("max_sector_weight")
    else:
        evaluated.append("max_sector_weight")
        max_observed_sector = max(class_exposure.values(), default=0.0)
        if max_observed_sector > max_sector_weight:
            violations.append("max_sector_weight")

    target_allocation = _extract_target_asset_allocation(strategy.portfolio_policy)
    drift_threshold_pct = _extract_drift_threshold_pct(strategy.portfolio_policy)
    if target_allocation is None:
        unsupported.append("target_asset_allocation")
    if drift_threshold_pct is None:
        unsupported.append("drift_threshold_pct")

    if target_allocation is not None and drift_threshold_pct is not None:
        evaluated.append("target_asset_allocation")
        drift_limit = drift_threshold_pct / 100.0
        if _allocation_drift_exceeds_limit(
            actual=class_exposure,
            target=target_allocation,
            drift_limit=drift_limit,
        ):
            violations.append("target_asset_allocation")

    strategy_violations = list(violations)

    hard_rule_violations.extend(
        _extract_explicit_hard_rule_violations(analytics_snapshot.metrics)
    )
    hard_rule_violations.extend(
        _evaluate_market_access_hard_rule(
            class_exposure=class_exposure,
            strategy=strategy,
        )
    )
    violations.extend(hard_rule_violations)

    if not evaluated:
        return (
            StrategyFitAssessment(
                verdict=StrategyFitVerdict.UNKNOWN,
                unsupported_criteria=unsupported,
                reason="No supported strategy-fit criteria are configured",
            ),
            {
                "evaluated_criteria": evaluated,
                "violations": violations,
                "hard_rule_violations": hard_rule_violations,
            },
        )

    if violations:
        return (
            StrategyFitAssessment(
                verdict=StrategyFitVerdict.NOT_FIT,
                unsupported_criteria=unsupported,
                reason=_not_fit_reason(
                    has_strategy_violations=bool(strategy_violations),
                    has_hard_rule_violations=bool(hard_rule_violations),
                ),
            ),
            {
                "evaluated_criteria": evaluated,
                "violations": violations,
                "hard_rule_violations": hard_rule_violations,
            },
        )

    if unsupported:
        return (
            StrategyFitAssessment(
                verdict=StrategyFitVerdict.PARTIAL_FIT,
                unsupported_criteria=unsupported,
                reason="Supported criteria pass, but some criteria are unsupported",
            ),
            {
                "evaluated_criteria": evaluated,
                "violations": violations,
                "hard_rule_violations": hard_rule_violations,
            },
        )

    return (
        StrategyFitAssessment(
            verdict=StrategyFitVerdict.FIT,
            unsupported_criteria=[],
            reason="Portfolio satisfies configured strategy criteria",
        ),
        {
            "evaluated_criteria": evaluated,
            "violations": violations,
            "hard_rule_violations": hard_rule_violations,
        },
    )


def _not_fit_reason(*, has_strategy_violations: bool, has_hard_rule_violations: bool) -> str:
    if has_hard_rule_violations and has_strategy_violations:
        return "Portfolio violates hard business rules and strategy criteria"
    if has_hard_rule_violations:
        return "Portfolio violates hard business rules"
    return "Portfolio violates configured strategy criteria"


def _read_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _extract_weights(metrics: Mapping[str, Any]) -> list[float]:
    raw_weights = metrics.get("weights")
    if not isinstance(raw_weights, list):
        return []

    weights: list[float] = []
    for item in raw_weights:
        if not isinstance(item, Mapping):
            continue
        parsed = _read_float(item.get("weight"))
        if parsed is None:
            continue
        weights.append(parsed)
    return weights


def _extract_class_exposure(metrics: Mapping[str, Any]) -> dict[str, float]:
    raw_exposure = metrics.get("class_exposure")
    if not isinstance(raw_exposure, Mapping):
        return {}

    exposure: dict[str, float] = {}
    for raw_key, raw_value in raw_exposure.items():
        key = str(raw_key).strip()
        if not key:
            continue
        parsed = _read_float(raw_value)
        if parsed is None:
            continue
        exposure[key] = parsed
    return exposure


def _extract_target_asset_allocation(
    portfolio_policy: Mapping[str, Any],
) -> dict[str, float] | None:
    raw_value = portfolio_policy.get("target_asset_allocation")
    if not isinstance(raw_value, Mapping):
        return None

    result: dict[str, float] = {}
    for raw_key, raw_weight in raw_value.items():
        key = str(raw_key).strip()
        parsed_weight = _read_float(raw_weight)
        if not key or parsed_weight is None:
            continue
        result[key] = parsed_weight

    if not result:
        return None
    return result


def _extract_drift_threshold_pct(portfolio_policy: Mapping[str, Any]) -> float | None:
    raw_rebalance_policy = portfolio_policy.get("rebalance_policy")
    if not isinstance(raw_rebalance_policy, Mapping):
        return None

    return _read_float(raw_rebalance_policy.get("drift_threshold_pct"))


def _allocation_drift_exceeds_limit(
    *,
    actual: Mapping[str, float],
    target: Mapping[str, float],
    drift_limit: float,
) -> bool:
    for asset_class, target_weight in target.items():
        observed_weight = actual.get(asset_class, 0.0)
        if abs(observed_weight - target_weight) > drift_limit:
            return True
    return False


def _extract_explicit_hard_rule_violations(metrics: Mapping[str, Any]) -> list[str]:
    raw_hard_rules = metrics.get("hard_rules")
    if not isinstance(raw_hard_rules, Mapping):
        return []

    checks: tuple[tuple[str, str], ...] = (
        ("one_off_dividend_extrapolated", "hard_rule_one_off_dividend_extrapolated"),
        ("narrative_without_fundamentals", "hard_rule_narrative_without_fundamentals"),
        (
            "cheap_multiples_only_capital_intensive",
            "hard_rule_cheap_multiples_only_capital_intensive",
        ),
        (
            "cross_country_without_adjustments",
            "hard_rule_cross_country_without_adjustments",
        ),
    )

    violations: list[str] = []
    for input_key, violation_key in checks:
        if _read_bool(raw_hard_rules.get(input_key)):
            violations.append(violation_key)
    return violations


def _evaluate_market_access_hard_rule(
    *,
    class_exposure: Mapping[str, float],
    strategy: StructuredStrategy,
) -> list[str]:
    dominant_instrument = _dominant_instrument_class(class_exposure)
    if dominant_instrument is None:
        return []

    investable_universe = strategy.investable_universe
    allowed = _normalized_instrument_set(investable_universe.get("allowed_instruments"))
    disallowed = _normalized_instrument_set(investable_universe.get("disallowed_instruments"))

    if dominant_instrument in disallowed:
        return ["hard_rule_inaccessible_market_primary_driver"]
    if allowed and dominant_instrument not in allowed:
        return ["hard_rule_inaccessible_market_primary_driver"]
    return []


def _dominant_instrument_class(class_exposure: Mapping[str, float]) -> str | None:
    if not class_exposure:
        return None

    asset_class, share = max(class_exposure.items(), key=lambda item: item[1])
    if share < 0.5:
        return None

    normalized_asset = asset_class.strip().lower()
    return _map_asset_class_to_instrument(normalized_asset)


def _map_asset_class_to_instrument(asset_class: str) -> str | None:
    mapping = {
        "equity": "stocks",
        "stock": "stocks",
        "stocks": "stocks",
        "bond": "bonds",
        "bonds": "bonds",
        "etf": "etf",
        "fx": "fx",
        "futures": "futures",
        "money_market": "money_market_funds",
        "cash": "money_market_funds",
    }
    return mapping.get(asset_class)


def _normalized_instrument_set(raw_value: Any) -> set[str]:
    if not isinstance(raw_value, list):
        return set()

    values: set[str] = set()
    for item in raw_value:
        candidate = str(item).strip().lower()
        if candidate:
            values.add(candidate)
    return values


def _read_bool(raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return False
