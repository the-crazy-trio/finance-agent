from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from finance_agent.retriever.providers.http import fetch_json


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _table_to_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    columns = table.get("columns") if isinstance(table, dict) else None
    data = table.get("data") if isinstance(table, dict) else None
    if not isinstance(columns, list) or not isinstance(data, list):
        return []
    normalized_columns = [str(item) for item in columns]
    rows: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, list):
            continue
        rows.append(
            {
                normalized_columns[index]: row[index]
                for index in range(min(len(normalized_columns), len(row)))
            }
        )
    return rows


_BOARD_PRIORITY: tuple[str, ...] = (
    "CETS",
    "TQBR",
    "TQOB",
    "SNDX",
    "RTSI",
)


class MoexApiProvider:
    def __init__(
        self,
        *,
        base_url: str = "https://iss.moex.com/iss",
        fetcher: Callable[..., dict[str, Any]] = fetch_json,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._fetcher = fetcher

    def get_security_market_data(
        self,
        *,
        security: str,
        engine: str,
        market: str,
    ) -> dict[str, Any] | None:
        payload = self._fetcher(
            f"{self._base_url}/engines/{engine}/markets/{market}/securities/{security}.json",
            params={"iss.meta": "off"},
        )
        market_rows = _table_to_rows(payload.get("marketdata", {}))
        security_rows = _table_to_rows(payload.get("securities", {}))
        if not market_rows and not security_rows:
            return None

        market_row = _select_best_market_row(market_rows)
        security_row = security_rows[0] if security_rows else {}
        return {
            "security": security,
            "engine": engine,
            "market": market,
            "short_name": security_row.get("SHORTNAME"),
            "board_id": market_row.get("BOARDID"),
            "last": _first_numeric(market_row, "LAST", "CURRENTVALUE", "LASTVALUE"),
            "close": _first_numeric(
                market_row,
                "CLOSE",
                "CLOSEPRICE",
                "LASTVALUE",
            ),
            "open": _first_numeric(market_row, "OPEN", "OPENVALUE", "LOPENPRICE"),
            "high": _as_optional_float(market_row.get("HIGH")),
            "low": _as_optional_float(market_row.get("LOW")),
            "volume": _first_numeric(market_row, "VOLUME", "VOLTODAY", "QTY"),
            "value": _first_numeric(market_row, "VALUE", "VALTODAY", "VALUE_USD"),
            "trading_status": market_row.get("TRADINGSTATUS"),
            "source_timestamp": _now_iso(),
            "source_name": "moex_iss",
            "source_type": "market",
        }

    def resolve_security(self, *, security: str) -> dict[str, Any] | None:
        routes: tuple[tuple[str, str], ...] = (
            ("stock", "shares"),
            ("stock", "bonds"),
            ("stock", "index"),
            ("currency", "selt"),
        )
        normalized_security = security.strip().upper()
        if not normalized_security:
            return None
        for engine, market in routes:
            payload = self.get_security_market_data(
                security=normalized_security,
                engine=engine,
                market=market,
            )
            if payload is not None:
                return payload
        return None

    def get_imoex(self) -> dict[str, Any] | None:
        return self.get_security_market_data(
            security="IMOEX",
            engine="stock",
            market="index",
        )

    def get_rtsi(self) -> dict[str, Any] | None:
        return self.get_security_market_data(
            security="RTSI",
            engine="stock",
            market="index",
        )

    def get_usd_rub(self) -> dict[str, Any] | None:
        return self.get_security_market_data(
            security="USD000UTSTOM",
            engine="currency",
            market="selt",
        )


def _as_optional_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _as_optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _select_best_market_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        board_id = str(row.get("BOARDID") or "")
        board_rank = len(_BOARD_PRIORITY)
        if board_id in _BOARD_PRIORITY:
            board_rank = _BOARD_PRIORITY.index(board_id)

        price_score = 0
        for key in ("LAST", "CURRENTVALUE", "LASTVALUE", "CLOSE", "CLOSEPRICE"):
            if _as_optional_float(row.get(key)) is not None:
                price_score += 1
        active_score = 1 if str(row.get("TRADINGSTATUS") or "") == "N" else 0
        return (-price_score, board_rank, -active_score)

    return sorted(rows, key=score)[0]
