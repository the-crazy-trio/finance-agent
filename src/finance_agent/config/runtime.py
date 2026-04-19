from __future__ import annotations

from enum import StrEnum
from os import environ
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field


class AppEnv(StrEnum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_env: AppEnv = AppEnv.DEV
    openrouter_api_key: str = Field(min_length=1)
    tinvest_readonly_token: str = Field(min_length=1)
    default_model_intent: str = Field(default="openai/gpt-4.1-mini", min_length=1)
    default_model_synthesis: str = Field(default="openai/gpt-4.1", min_length=1)
    request_timeout_ms: int = Field(default=120_000, gt=0)
    tool_timeout_ms: int = Field(default=60_000, gt=0)
    llm_timeout_ms: int = Field(default=90_000, gt=0)
    max_retries_transient: int = Field(default=2, ge=0)
    tokens_intent_max: int = Field(default=1_500, gt=0)
    tokens_synthesis_max: int = Field(default=4_000, gt=0)
    portfolio_snapshot_max_age_min: int = Field(default=60, gt=0)
    analytics_snapshot_max_age_min: int = Field(default=120, gt=0)
    strategy_fit_min_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_guarded_text2sql: bool = False
    enable_guarded_text2api: bool = False
    enable_offline_eval_export: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RuntimeConfig":
        source = env if env is not None else environ
        required_secrets = ("OPENROUTER_API_KEY", "TINVEST_READONLY_TOKEN")
        missing = [key for key in required_secrets if not _read_required(source, key)]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"Missing required environment variables: {joined}")

        payload = {
            "app_env": _parse_app_env_or_default(source.get("APP_ENV"), AppEnv.DEV),
            "openrouter_api_key": _read_required(source, "OPENROUTER_API_KEY"),
            "tinvest_readonly_token": _read_required(source, "TINVEST_READONLY_TOKEN"),
            "default_model_intent": _read_optional(
                source,
                "DEFAULT_MODEL_INTENT",
                "openai/gpt-4.1-mini",
            ),
            "default_model_synthesis": _read_optional(
                source,
                "DEFAULT_MODEL_SYNTHESIS",
                "openai/gpt-4.1",
            ),
            "request_timeout_ms": _parse_int_or_default(source, "REQUEST_TIMEOUT_MS", 120_000),
            "tool_timeout_ms": _parse_int_or_default(source, "TOOL_TIMEOUT_MS", 60_000),
            "llm_timeout_ms": _parse_int_or_default(source, "LLM_TIMEOUT_MS", 90_000),
            "max_retries_transient": _parse_int_or_default(source, "MAX_RETRIES_TRANSIENT", 2),
            "tokens_intent_max": _parse_int_or_default(source, "TOKENS_INTENT_MAX", 1_500),
            "tokens_synthesis_max": _parse_int_or_default(
                source,
                "TOKENS_SYNTHESIS_MAX",
                4_000,
            ),
            "portfolio_snapshot_max_age_min": _parse_int_or_default(
                source,
                "PORTFOLIO_SNAPSHOT_MAX_AGE_MIN",
                60,
            ),
            "analytics_snapshot_max_age_min": _parse_int_or_default(
                source,
                "ANALYTICS_SNAPSHOT_MAX_AGE_MIN",
                120,
            ),
            "strategy_fit_min_coverage": _parse_float_or_default(
                source,
                "STRATEGY_FIT_MIN_COVERAGE",
                0.7,
            ),
            "enable_guarded_text2sql": _parse_bool(source, "ENABLE_GUARDED_TEXT2SQL", False),
            "enable_guarded_text2api": _parse_bool(source, "ENABLE_GUARDED_TEXT2API", False),
            "enable_offline_eval_export": _parse_bool(
                source,
                "ENABLE_OFFLINE_EVAL_EXPORT",
                False,
            ),
        }
        return cls(**payload)


def _read_required(source: Mapping[str, str], key: str) -> str:
    value = source.get(key)
    if value is None:
        return ""
    return value.strip()


def _read_optional(source: Mapping[str, str], key: str, default: str) -> str:
    value = source.get(key)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _parse_app_env_or_default(raw_value: str | None, default: AppEnv) -> AppEnv:
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    try:
        return AppEnv(normalized)
    except ValueError as error:
        supported = ", ".join(item.value for item in AppEnv)
        raise ValueError(f"APP_ENV must be one of: {supported}") from error


def _parse_int_or_default(source: Mapping[str, str], key: str, default: int) -> int:
    raw_value = source.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value.strip())
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error


def _parse_float_or_default(source: Mapping[str, str], key: str, default: float) -> float:
    raw_value = source.get(key)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    try:
        return float(normalized)
    except ValueError as error:
        raise ValueError(f"{key} must be a float") from error


def _parse_bool(source: Mapping[str, str], key: str, default: bool) -> bool:
    raw_value = source.get(key)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value")
