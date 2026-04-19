from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Callable

from finance_agent.retriever.providers.http import ProviderHttpError, fetch_json, fetch_text


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CbrApiProvider:
    def __init__(
        self,
        *,
        base_url: str = "https://cbrapi.andrewfromtver.ru",
        fetcher: Callable[..., dict[str, Any]] = fetch_json,
        text_fetcher: Callable[..., str] = fetch_text,
        cbr_public_base_url: str = "https://www.cbr.ru",
        cbr_daily_json_url: str = "https://www.cbr-xml-daily.ru/daily_json.js",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._cbr_public_base_url = cbr_public_base_url.rstrip("/")
        self._cbr_daily_json_url = cbr_daily_json_url
        self._fetcher = fetcher
        self._text_fetcher = text_fetcher

    def get_daily_info(self) -> dict[str, Any]:
        try:
            return self._fetcher(
                f"{self._base_url}/api/cbr/daily_info/all_data_info",
                params={"output": "json"},
            )
        except ProviderHttpError:
            fallback = self._scrape_key_indicators()
            if fallback:
                return fallback
            raise

    def get_daily_rates(self) -> dict[str, Any]:
        try:
            return self._fetcher(
                f"{self._base_url}/api/cbr/currency/get_daily_rates",
                params={"output": "json"},
            )
        except ProviderHttpError:
            return self._fetcher(self._cbr_daily_json_url)

    def extract_key_rate(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        value = _find_first_numeric(
            payload,
            [
                "key_rate",
                "keyRate",
                "KeyRate",
                "rates.key_rate",
                "rates.KeyRate",
            ],
        )
        if value is None:
            return None
        return _indicator("key_rate", value, signal_type="policy_based")

    def extract_inflation(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        value = _find_first_numeric(
            payload,
            [
                "inflation",
                "Inflation",
                "macro.inflation",
                "rates.inflation",
                "cpi",
                "CPI",
            ],
        )
        if value is None:
            return None
        return _indicator("inflation", value, signal_type="confirming")

    def extract_gdp(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        value = _find_first_numeric(payload, ["gdp", "GDP", "macro.gdp"])
        if value is None:
            return None
        return _indicator("gdp", value, signal_type="lagging")

    def extract_usd_rub(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        value = _find_first_numeric(
            payload,
            [
                "usd_rub",
                "USD_RUB",
                "rates.usd_rub",
                "rates.USD",
                "Valute.USD.Value",
                "Valute.USD.Previous",
            ],
        )
        if value is None:
            return None
        return _indicator("usd_rub", value, signal_type="market_based")

    def _scrape_key_indicators(self) -> dict[str, Any]:
        html = self._text_fetcher(f"{self._cbr_public_base_url}/key-indicators/")
        key_rate = _extract_percent_value(
            html,
            marker_pattern=r'href="/hd_base/KeyRate/"',
        )
        inflation = _extract_percent_value(
            html,
            marker_pattern=r">\s*Инфляция\s*<",
        )
        payload: dict[str, Any] = {}
        if key_rate is not None:
            payload["key_rate"] = key_rate
        if inflation is not None:
            payload["inflation"] = inflation
        return payload


def _indicator(key: str, value: float, *, signal_type: str) -> dict[str, Any]:
    return {
        "key": key,
        "value": value,
        "as_of": _now_iso(),
        "signal_type": signal_type,
        "source_name": "cbr_api",
        "source_type": "official",
        "coverage_status": "full",
        "freshness_status": "fresh",
    }


def _find_first_numeric(payload: dict[str, Any], paths: list[str]) -> float | None:
    for path in paths:
        value = _lookup(payload, path)
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _lookup(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def _to_float(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _extract_percent_value(html: str, *, marker_pattern: str) -> float | None:
    pattern = re.compile(
        marker_pattern + r".*?<div class=\"value\">\s*([0-9]+(?:[\.,][0-9]+)?)%",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(html)
    if match is None:
        return None
    return _to_float(match.group(1).replace(",", "."))
