from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import ErrorCode
from finance_agent.contracts.tooling import ToolMeta, ToolResult
from finance_agent.retriever.providers import CbrApiProvider, MoexApiProvider
from finance_agent.retriever.providers.http import ProviderHttpError

_DEFAULT_MACRO_KEYS: tuple[str, ...] = (
    "key_rate",
    "inflation",
    "gdp",
    "imoex",
    "rts",
    "usd_rub",
)


class ExternalMarketDataAdapter:
    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        cbr_provider: CbrApiProvider | None = None,
        moex_provider: MoexApiProvider | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._cbr = cbr_provider or CbrApiProvider()
        self._moex = moex_provider or MoexApiProvider()
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def show_macro_indicators(
        self,
        *,
        keys: list[str] | None,
        correlation_id: str,
    ) -> ToolResult:
        start_ts = self._now_fn()
        resolved_keys, invalid_keys = _normalize_keys(keys)
        if invalid_keys:
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="macro indicator keys contain unsupported values",
                details={"invalid_keys": invalid_keys},
                meta=ToolMeta(
                    latency_ms=_latency_ms(start_ts, self._now_fn()),
                    correlation_id=correlation_id,
                ),
            )

        indicators: list[dict[str, Any]] = []
        missing: list[str] = []
        errors: list[dict[str, Any]] = []

        cbr_daily: dict[str, Any] = {}
        cbr_rates: dict[str, Any] = {}
        cbr_requested = any(item in {"key_rate", "inflation", "gdp"} for item in resolved_keys)
        if cbr_requested:
            try:
                cbr_daily = self._cbr.get_daily_info()
            except Exception as error:
                errors.append(
                    {
                        "source": "cbr_api",
                        "message": str(error),
                    }
                )

        cbr_rates_requested = "usd_rub" in resolved_keys
        if cbr_rates_requested:
            try:
                cbr_rates = self._cbr.get_daily_rates()
            except Exception as error:
                errors.append(
                    {
                        "source": "cbr_rates",
                        "message": str(error),
                    }
                )

        for key in resolved_keys:
            row = self._resolve_macro_indicator(
                key=key,
                cbr_daily=cbr_daily,
                cbr_rates=cbr_rates,
            )
            if row is None:
                missing.append(key)
                continue
            indicators.append(row)

        latency_ms = _latency_ms(start_ts, self._now_fn())
        if not indicators:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="No macro indicators are available from providers",
                details={"missing": missing, "provider_errors": errors},
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )

        coverage = len(indicators) / float(len(resolved_keys)) if resolved_keys else 0.0
        payload = {
            "indicators": indicators,
            "missing": missing,
            "is_partial": bool(missing) or bool(errors),
            "provider_errors": errors,
        }
        meta = ToolMeta(
            latency_ms=latency_ms,
            source_ts=self._now_fn(),
            coverage=coverage,
            correlation_id=correlation_id,
        )
        if missing or errors:
            return ToolResult.partial(data=payload, meta=meta)
        return ToolResult.ok(data=payload, meta=meta)

    def show_issuer_indicators(
        self,
        *,
        ticker: str,
        correlation_id: str,
    ) -> ToolResult:
        start_ts = self._now_fn()
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="ticker is required",
                meta=ToolMeta(
                    latency_ms=_latency_ms(start_ts, self._now_fn()),
                    correlation_id=correlation_id,
                ),
            )

        try:
            payload = self._moex.resolve_security(security=normalized_ticker)
        except ProviderHttpError as error:
            error_code = (
                ErrorCode.TIMEOUT
                if error.status_code == 408
                else ErrorCode.UPSTREAM_UNAVAILABLE
            )
            return ToolResult.error(
                code=error_code,
                message=f"MOEX issuer lookup failed: {error}",
                retriable=True,
                meta=ToolMeta(
                    latency_ms=_latency_ms(start_ts, self._now_fn()),
                    correlation_id=correlation_id,
                ),
            )
        except Exception as error:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message=f"MOEX issuer lookup failed: {error}",
                retriable=True,
                meta=ToolMeta(
                    latency_ms=_latency_ms(start_ts, self._now_fn()),
                    correlation_id=correlation_id,
                ),
            )

        latency_ms = _latency_ms(start_ts, self._now_fn())
        if payload is None:
            return ToolResult.error(
                code=ErrorCode.MAPPING_NOT_FOUND,
                message="Ticker is not found in MOEX markets",
                details={"ticker": normalized_ticker},
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )

        indicator_payload = _normalize_issuer_payload(payload)
        coverage = _issuer_coverage(indicator_payload)
        meta = ToolMeta(
            latency_ms=latency_ms,
            source_ts=self._now_fn(),
            coverage=coverage,
            correlation_id=correlation_id,
        )
        response_data = {
            "ticker": normalized_ticker,
            "issuer_indicators": indicator_payload,
            "is_partial": coverage < 1.0,
        }
        if coverage < 1.0:
            return ToolResult.partial(data=response_data, meta=meta)
        return ToolResult.ok(data=response_data, meta=meta)

    def _resolve_macro_indicator(
        self,
        *,
        key: str,
        cbr_daily: Mapping[str, Any],
        cbr_rates: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if key == "key_rate":
            return self._cbr.extract_key_rate(dict(cbr_daily))
        if key == "inflation":
            return self._cbr.extract_inflation(dict(cbr_daily))
        if key == "gdp":
            return self._cbr.extract_gdp(dict(cbr_daily))
        if key == "imoex":
            return _moex_indicator("imoex", self._moex.get_imoex())
        if key == "rts":
            return _moex_indicator("rts", self._moex.get_rtsi())
        if key == "usd_rub":
            moex_value = _moex_indicator("usd_rub", self._moex.get_usd_rub())
            if moex_value is not None:
                return moex_value
            return self._cbr.extract_usd_rub(dict(cbr_rates))
        return None


def _normalize_keys(keys: list[str] | None) -> tuple[list[str], list[str]]:
    if keys is None:
        return list(_DEFAULT_MACRO_KEYS), []

    values: list[str] = []
    invalid: list[str] = []
    allowed = set(_DEFAULT_MACRO_KEYS)
    for item in keys:
        normalized = str(item).strip().lower()
        if not normalized:
            continue
        if normalized not in allowed:
            invalid.append(normalized)
            continue
        if normalized not in values:
            values.append(normalized)

    if not values:
        values = list(_DEFAULT_MACRO_KEYS)
    return values, invalid


def _moex_indicator(key: str, payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    value = (
        _to_float(payload.get("last"))
        or _to_float(payload.get("close"))
        or _to_float(payload.get("open"))
    )
    if value is None:
        return None
    return {
        "key": key,
        "value": value,
        "as_of": payload.get("source_timestamp"),
        "signal_type": "market_based",
        "source_name": "moex_iss",
        "source_type": "market",
        "coverage_status": "full",
        "freshness_status": "fresh",
    }


def _normalize_issuer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "security": payload.get("security"),
        "short_name": payload.get("short_name"),
        "engine": payload.get("engine"),
        "market": payload.get("market"),
        "last": _to_float(payload.get("last")),
        "close": _to_float(payload.get("close")),
        "open": _to_float(payload.get("open")),
        "high": _to_float(payload.get("high")),
        "low": _to_float(payload.get("low")),
        "volume": _to_float(payload.get("volume")),
        "value": _to_float(payload.get("value")),
        "trading_status": payload.get("trading_status"),
        "source_name": payload.get("source_name", "moex_iss"),
        "source_type": payload.get("source_type", "market"),
        "as_of": payload.get("source_timestamp"),
    }


def _issuer_coverage(payload: Mapping[str, Any]) -> float:
    required = ("last", "close", "open", "high", "low")
    filled = sum(1 for key in required if _to_float(payload.get(key)) is not None)
    return filled / float(len(required))


def _to_float(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _latency_ms(start_ts: datetime, end_ts: datetime) -> int:
    delta = end_ts - start_ts
    latency = int(delta.total_seconds() * 1000)
    return latency if latency >= 0 else 0
