from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FreshnessMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ts: datetime | None = None
    age_minutes: float | None = Field(default=None, ge=0)
    max_age_minutes: int | None = Field(default=None, ge=1)
    is_stale: bool = False


class StructuredStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    client_id: str = Field(min_length=1)
    base_currency: str = Field(min_length=1)
    tax_residency: str = Field(min_length=1)
    jurisdiction_constraints: dict[str, Any]
    knowledge_experience: dict[str, Any]
    financial_profile: dict[str, Any]
    goals: list[dict[str, Any]]
    risk_preferences: dict[str, Any]
    investable_universe: dict[str, Any]
    portfolio_policy: dict[str, Any]
    explainability_preferences: dict[str, Any]

    @model_validator(mode="after")
    def validate_strategy_shape(self) -> "StructuredStrategy":
        _validate_semver(self.schema_version)
        _validate_goals(self.goals)
        _validate_risk_preferences(self.risk_preferences)
        _validate_portfolio_policy(self.portfolio_policy)
        _validate_investable_universe(self.investable_universe)
        return self


class PortfolioPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instrument_uid: str | None = None
    figi: str | None = None
    ticker: str | None = None
    quantity: float = Field(ge=0)
    market_value: float = Field(ge=0)
    currency: str | None = None
    asset_class: str | None = None

    @model_validator(mode="after")
    def ensure_any_identifier(self) -> "PortfolioPosition":
        if not (self.instrument_uid or self.figi or self.ticker):
            raise ValueError("at least one position identifier is required")
        return self


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    snapshot_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    account_id: str | None = None
    as_of_ts: datetime
    positions: list[PortfolioPosition]
    portfolio_value: float = Field(ge=0)
    freshness: FreshnessMetadata = Field(default_factory=FreshnessMetadata)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    unknown_share: float = Field(default=0.0, ge=0.0, le=1.0)
    partial: bool = False


class StrategyFitVerdict(StrEnum):
    FIT = "fit"
    PARTIAL_FIT = "partial_fit"
    NOT_FIT = "not_fit"
    UNKNOWN = "unknown"


class StrategyFitAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: StrategyFitVerdict
    unsupported_criteria: list[str] = Field(default_factory=list)
    reason: str | None = None


class AnalyticsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    snapshot_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    account_id: str | None = None
    as_of_ts: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)
    strategy_fit: StrategyFitAssessment | None = None
    freshness: FreshnessMetadata = Field(default_factory=FreshnessMetadata)
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    partial: bool = False


def _validate_semver(value: str) -> None:
    parts = value.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError("schema_version must follow semantic versioning format (x.y.z)")


def _validate_goals(goals: list[dict[str, Any]]) -> None:
    if not goals:
        raise ValueError("goals must contain at least one goal")
    for goal in goals:
        goal_id = str(goal.get("goal_id") or "").strip()
        if not goal_id:
            raise ValueError("each goal must include non-empty goal_id")


def _validate_risk_preferences(risk_preferences: dict[str, Any]) -> None:
    tolerance = str(risk_preferences.get("risk_tolerance_self_assessed") or "").strip()
    supported = {"conservative", "balanced", "growth", "aggressive"}
    if tolerance not in supported:
        raise ValueError(
            "risk_preferences.risk_tolerance_self_assessed must be one of "
            "conservative|balanced|growth|aggressive"
        )

    for key in ("max_single_name_weight", "max_sector_weight", "max_fx_unhedged_weight"):
        if key not in risk_preferences:
            continue
        value = _to_float(risk_preferences.get(key))
        if value is None or not (0.0 <= value <= 1.0):
            raise ValueError(f"risk_preferences.{key} must be a float in range [0, 1]")


def _validate_portfolio_policy(portfolio_policy: dict[str, Any]) -> None:
    benchmark = str(portfolio_policy.get("benchmark") or "").strip()
    if not benchmark:
        raise ValueError("portfolio_policy.benchmark must not be blank")

    raw_allocation = portfolio_policy.get("target_asset_allocation")
    if raw_allocation is None:
        return
    if not isinstance(raw_allocation, dict):
        raise ValueError("portfolio_policy.target_asset_allocation must be an object")
    if not raw_allocation:
        raise ValueError("portfolio_policy.target_asset_allocation must not be empty")

    total = 0.0
    for key, raw_value in raw_allocation.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("portfolio_policy.target_asset_allocation contains blank key")
        value = _to_float(raw_value)
        if value is None or not (0.0 <= value <= 1.0):
            raise ValueError(
                "portfolio_policy.target_asset_allocation values must be floats in range [0, 1]"
            )
        total += value

    if total > 1.05:
        raise ValueError("portfolio_policy.target_asset_allocation total weight must be <= 1.05")


def _validate_investable_universe(investable_universe: dict[str, Any]) -> None:
    for key in ("allowed_instruments", "disallowed_instruments"):
        if key not in investable_universe:
            continue
        value = investable_universe.get(key)
        if not isinstance(value, list):
            raise ValueError(f"investable_universe.{key} must be a list")
        for item in value:
            if not str(item).strip():
                raise ValueError(f"investable_universe.{key} must not contain blank values")


def _to_float(raw_value: Any) -> float | None:
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None
