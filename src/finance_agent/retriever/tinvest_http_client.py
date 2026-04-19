from __future__ import annotations

import json
import socket
import ssl
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory

_OPERATIONS_GET_PORTFOLIO_PATH = (
    "/rest/tinkoff.public.invest.api.contract.v1.OperationsService/GetPortfolio"
)
_USERS_GET_ACCOUNTS_PATH = "/rest/tinkoff.public.invest.api.contract.v1.UsersService/GetAccounts"
_DEFAULT_BASE_URL = "https://invest-public-api.tbank.ru"

# request_fn signature:
# (url, method, body, headers, timeout_sec) -> response_payload
JsonRequestFn = Callable[
    [str, str, Mapping[str, Any] | None, Mapping[str, str], float],
    Mapping[str, Any],
]


class TInvestHttpReadonlyClient:
    def __init__(
        self,
        *,
        token: str,
        base_url: str = _DEFAULT_BASE_URL,
        request_fn: JsonRequestFn | None = None,
        now_fn: Callable[[], datetime] | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        normalized_token = token.strip()
        if not normalized_token:
            raise ValueError("token must not be blank")

        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("base_url must not be blank")

        self._token = normalized_token
        self._base_url = normalized_base_url
        self._request_fn = request_fn or _build_default_json_request(ssl_context)
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        if account_id is None:
            return self._get_aggregated_portfolio(
                correlation_id=correlation_id,
                timeout_ms=timeout_ms,
            )
        return self._get_portfolio_for_account(
            account_id=account_id,
            correlation_id=correlation_id,
            timeout_ms=timeout_ms,
        )

    def list_account_ids(self, *, correlation_id: str, timeout_ms: int) -> list[str]:
        try:
            payload = self._post(
                path=_USERS_GET_ACCOUNTS_PATH,
                body={},
                correlation_id=correlation_id,
                timeout_ms=timeout_ms,
            )
        except TInvestClientError as error:
            if not _is_http_405_error(error):
                raise
            payload = self._get(
                path=_USERS_GET_ACCOUNTS_PATH,
                correlation_id=correlation_id,
                timeout_ms=timeout_ms,
            )

        raw_accounts = payload.get("accounts")
        if not isinstance(raw_accounts, list):
            raise TInvestClientError(
                category=TInvestErrorCategory.INTERNAL,
                message="T-Invest accounts response is malformed",
            )

        result: list[str] = []
        for item in raw_accounts:
            if not isinstance(item, Mapping):
                continue
            account_status = _normalize_text(item.get("status"))
            if account_status is not None and "closed" in account_status.lower():
                continue

            raw_id = item.get("id")
            if raw_id is None:
                raw_id = item.get("accountId")
            account_id = _normalize_text(raw_id)
            if account_id is not None and account_id not in result:
                result.append(account_id)

        if not result:
            raise TInvestClientError(
                category=TInvestErrorCategory.ACCOUNT_NOT_FOUND,
                message="No accessible broker accounts were found",
            )
        return result

    def _get_aggregated_portfolio(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        account_ids = self.list_account_ids(correlation_id=correlation_id, timeout_ms=timeout_ms)
        payloads = [
            self._get_portfolio_for_account(
                account_id=account_id,
                correlation_id=correlation_id,
                timeout_ms=timeout_ms,
            )
            for account_id in account_ids
        ]
        return _merge_portfolio_payloads(payloads)

    def _get_portfolio_for_account(
        self,
        *,
        account_id: str,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        payload = self._post(
            path=_OPERATIONS_GET_PORTFOLIO_PATH,
            body={"accountId": account_id},
            correlation_id=correlation_id,
            timeout_ms=timeout_ms,
        )
        return _normalize_portfolio_payload(payload, account_id=account_id, now_ts=self._now_fn())

    def _post(
        self,
        *,
        path: str,
        body: Mapping[str, Any],
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        return self._request(
            path=path,
            method="POST",
            body=body,
            correlation_id=correlation_id,
            timeout_ms=timeout_ms,
        )

    def _get(
        self,
        *,
        path: str,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        return self._request(
            path=path,
            method="GET",
            body=None,
            correlation_id=correlation_id,
            timeout_ms=timeout_ms,
        )

    def _request(
        self,
        *,
        path: str,
        method: str,
        body: Mapping[str, Any] | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        timeout_sec = max(float(timeout_ms) / 1000.0, 0.001)
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-correlation-id": correlation_id,
        }
        try:
            payload = self._request_fn(url, method, body, headers, timeout_sec)
        except TInvestClientError:
            raise
        except Exception as error:
            raise TInvestClientError(
                category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
                message=f"T-Invest request failed: {error}",
            ) from error

        if not isinstance(payload, Mapping):
            raise TInvestClientError(
                category=TInvestErrorCategory.INTERNAL,
                message="T-Invest response must be an object",
            )
        return payload


def _default_json_request(
    url: str,
    method: str,
    body: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout_sec: float,
    ssl_context: ssl.SSLContext | None,
) -> Mapping[str, Any]:
    normalized_method = method.strip().upper() or "GET"
    encoded: bytes | None = None
    if body is not None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(
        url=url,
        data=encoded,
        headers=dict(headers),
        method=normalized_method,
    )
    try:
        with urlopen(request, timeout=timeout_sec, context=ssl_context) as response:
            raw_response = response.read().decode("utf-8")
    except HTTPError as error:
        raw_body = _read_http_error_body(error)
        raise _map_http_error(status=error.code, raw_body=raw_body) from error
    except URLError as error:
        raise _map_url_error(error) from error
    except TimeoutError as error:
        raise TInvestClientError(
            category=TInvestErrorCategory.TIMEOUT,
            message="T-Invest request timed out",
        ) from error

    if not raw_response.strip():
        return {}

    try:
        decoded = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise TInvestClientError(
            category=TInvestErrorCategory.INTERNAL,
            message="T-Invest response is not valid JSON",
        ) from error

    if not isinstance(decoded, Mapping):
        raise TInvestClientError(
            category=TInvestErrorCategory.INTERNAL,
            message="T-Invest response JSON must be an object",
        )
    return decoded


def _build_default_json_request(ssl_context: ssl.SSLContext | None) -> JsonRequestFn:
    def _request_fn(
        url: str,
        method: str,
        body: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        timeout_sec: float,
    ) -> Mapping[str, Any]:
        return _default_json_request(
            url=url,
            method=method,
            body=body,
            headers=headers,
            timeout_sec=timeout_sec,
            ssl_context=ssl_context,
        )

    return _request_fn


def _map_url_error(error: URLError) -> TInvestClientError:
    reason = error.reason
    if isinstance(reason, TimeoutError | socket.timeout):
        return TInvestClientError(
            category=TInvestErrorCategory.TIMEOUT,
            message="T-Invest request timed out",
        )
    return TInvestClientError(
        category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        message=f"T-Invest request failed: {reason}",
    )


def _is_http_405_error(error: TInvestClientError) -> bool:
    return "HTTP 405" in str(error)


def _read_http_error_body(error: HTTPError) -> str:
    try:
        raw = error.read()
    except Exception:
        return ""
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _map_http_error(*, status: int, raw_body: str) -> TInvestClientError:
    payload: Mapping[str, Any] = {}
    if raw_body.strip():
        try:
            decoded = json.loads(raw_body)
            if isinstance(decoded, Mapping):
                payload = decoded
        except json.JSONDecodeError:
            payload = {}

    grpc_category = _category_from_grpc_code(payload.get("code"))
    category = grpc_category or _category_from_http_status(status)
    message = _extract_error_message(payload) or f"T-Invest API request failed with HTTP {status}"
    return TInvestClientError(category=category, message=message)


def _category_from_grpc_code(raw_code: Any) -> TInvestErrorCategory | None:
    if isinstance(raw_code, bool):
        return None
    if isinstance(raw_code, str):
        raw_code = raw_code.strip()
        if not raw_code:
            return None
    try:
        code = int(raw_code)
    except (TypeError, ValueError):
        return None

    mapping = {
        3: TInvestErrorCategory.INVALID_ARGUMENT,
        4: TInvestErrorCategory.TIMEOUT,
        5: TInvestErrorCategory.NOT_FOUND,
        7: TInvestErrorCategory.PERMISSION_DENIED,
        8: TInvestErrorCategory.RATE_LIMIT,
        13: TInvestErrorCategory.INTERNAL,
        14: TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        16: TInvestErrorCategory.UNAUTHENTICATED,
    }
    return mapping.get(code)


def _category_from_http_status(status: int) -> TInvestErrorCategory:
    if status == 400:
        return TInvestErrorCategory.INVALID_ARGUMENT
    if status == 401:
        return TInvestErrorCategory.UNAUTHENTICATED
    if status == 403:
        return TInvestErrorCategory.PERMISSION_DENIED
    if status == 404:
        return TInvestErrorCategory.NOT_FOUND
    if status == 408:
        return TInvestErrorCategory.TIMEOUT
    if status == 429:
        return TInvestErrorCategory.RATE_LIMIT
    if status >= 500:
        return TInvestErrorCategory.UPSTREAM_UNAVAILABLE
    return TInvestErrorCategory.INTERNAL


def _extract_error_message(payload: Mapping[str, Any]) -> str | None:
    candidates = ("message", "details", "error", "description")
    for key in candidates:
        value = _normalize_text(payload.get(key))
        if value is not None:
            return value
    return None


def _normalize_portfolio_payload(
    payload: Mapping[str, Any],
    *,
    account_id: str,
    now_ts: datetime,
) -> Mapping[str, Any]:
    positions: list[dict[str, Any]] = []
    known_value_total = 0.0

    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list):
        raw_positions = []

    for idx, item in enumerate(raw_positions):
        if not isinstance(item, Mapping):
            continue
        position_payload, market_value_is_known = _normalize_position(item, index=idx)
        positions.append(position_payload)
        if market_value_is_known:
            known_value_total += position_payload["market_value"]

    portfolio_value = _money_value_to_float(payload.get("totalAmountPortfolio"))
    if portfolio_value is None:
        portfolio_value = known_value_total

    unknown_share = 0.0
    coverage = 1.0
    if portfolio_value > 0.0:
        unknown_value = max(portfolio_value - known_value_total, 0.0)
        unknown_share = min(unknown_value / portfolio_value, 1.0)
        coverage = max(0.0, min(1.0, 1.0 - unknown_share))

    as_of_ts = _normalize_text(
        payload.get("as_of_ts")
        or payload.get("asOfTs")
        or payload.get("asOf")
        or payload.get("as_of")
    )
    if as_of_ts is None:
        as_of_ts = now_ts.isoformat()

    return {
        "account_id": account_id,
        "as_of_ts": as_of_ts,
        "positions": positions,
        "portfolio_value": max(portfolio_value, 0.0),
        "coverage": coverage,
        "unknown_share": unknown_share,
        "partial": unknown_share > 0.0,
    }


def _normalize_position(item: Mapping[str, Any], *, index: int) -> tuple[dict[str, Any], bool]:
    instrument_uid = _normalize_text(item.get("instrumentUid") or item.get("positionUid"))
    figi = _normalize_text(item.get("figi"))
    ticker = _normalize_text(item.get("ticker"))
    if instrument_uid is None and figi is None and ticker is None:
        instrument_uid = f"unknown_{index + 1}"

    quantity = _quotation_to_float(item.get("quantity"))
    if quantity is None:
        quantity = _quotation_to_float(item.get("quantityLots"))
    if quantity is None:
        quantity = 0.0
    quantity = max(quantity, 0.0)

    current_price = _money_value_to_float(item.get("currentPrice"))
    average_price = _money_value_to_float(item.get("averagePositionPrice"))
    explicit_market_value = _money_value_to_float(item.get("marketValue"))

    market_value = explicit_market_value
    market_value_is_known = market_value is not None
    if market_value is None and current_price is not None:
        market_value = current_price * quantity
        market_value_is_known = True
    if market_value is None and average_price is not None:
        market_value = average_price * quantity
        market_value_is_known = True
    if market_value is None:
        market_value = 0.0

    currency = _read_currency(item)
    asset_class = _map_asset_class(item.get("instrumentType"))

    position = {
        "instrument_uid": instrument_uid,
        "figi": figi,
        "ticker": ticker,
        "quantity": quantity,
        "market_value": max(market_value, 0.0),
        "currency": currency,
        "asset_class": asset_class,
    }
    return position, market_value_is_known


def _merge_portfolio_payloads(payloads: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not payloads:
        raise TInvestClientError(
            category=TInvestErrorCategory.ACCOUNT_NOT_FOUND,
            message="No portfolio snapshots are available for accessible accounts",
        )

    if len(payloads) == 1:
        return dict(payloads[0])

    positions: list[Mapping[str, Any]] = []
    portfolio_value = 0.0
    weighted_coverage = 0.0
    weighted_unknown_share = 0.0
    account_ids: list[str] = []
    as_of_candidates: list[datetime] = []
    partial = False

    for payload in payloads:
        raw_positions = payload.get("positions")
        if isinstance(raw_positions, list):
            positions.extend(raw_positions)

        value = _to_non_negative_float(payload.get("portfolio_value"))
        coverage = _clamp01(_to_non_negative_float(payload.get("coverage"), default=1.0))
        unknown_share = _clamp01(_to_non_negative_float(payload.get("unknown_share"), default=0.0))
        portfolio_value += value
        weighted_coverage += coverage * value
        weighted_unknown_share += unknown_share * value
        partial = partial or bool(payload.get("partial", False))

        account_id = _normalize_text(payload.get("account_id"))
        if account_id is not None:
            account_ids.append(account_id)

        as_of_ts = _normalize_text(payload.get("as_of_ts"))
        if as_of_ts is not None:
            parsed = _parse_iso_datetime(as_of_ts)
            if parsed is not None:
                as_of_candidates.append(parsed)

    if portfolio_value > 0.0:
        coverage = _clamp01(weighted_coverage / portfolio_value)
        unknown_share = _clamp01(weighted_unknown_share / portfolio_value)
    else:
        coverage = 1.0
        unknown_share = 0.0

    as_of_ts = (
        min(as_of_candidates).isoformat() if as_of_candidates else datetime.now(UTC).isoformat()
    )
    return {
        "account_id": None,
        "account_ids": account_ids,
        "as_of_ts": as_of_ts,
        "positions": positions,
        "portfolio_value": portfolio_value,
        "coverage": coverage,
        "unknown_share": unknown_share,
        "partial": partial or unknown_share > 0.0,
    }


def _money_value_to_float(raw_value: Any) -> float | None:
    if isinstance(raw_value, Mapping):
        units = _to_float(raw_value.get("units"), default=0.0)
        nano = _to_float(raw_value.get("nano"), default=0.0)
        return units + nano / 1_000_000_000.0
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _quotation_to_float(raw_value: Any) -> float | None:
    if not isinstance(raw_value, Mapping):
        return None
    units = _to_float(raw_value.get("units"), default=0.0)
    nano = _to_float(raw_value.get("nano"), default=0.0)
    return units + nano / 1_000_000_000.0


def _read_currency(item: Mapping[str, Any]) -> str | None:
    money_candidates = (
        item.get("currentPrice"),
        item.get("averagePositionPrice"),
        item.get("marketValue"),
    )
    for candidate in money_candidates:
        if not isinstance(candidate, Mapping):
            continue
        currency = _normalize_text(candidate.get("currency"))
        if currency is not None:
            return currency
    return None


def _map_asset_class(raw_instrument_type: Any) -> str | None:
    instrument_type = _normalize_text(raw_instrument_type)
    if instrument_type is None:
        return None
    normalized = instrument_type.lower()
    mapping = {
        "share": "equity",
        "stock": "equity",
        "bond": "bond",
        "currency": "cash",
        "etf": "etf",
        "future": "future",
        "futures": "future",
        "option": "option",
    }
    return mapping.get(normalized, normalized)


def _normalize_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    if not value:
        return None
    return value


def _to_float(raw_value: Any, *, default: float) -> float:
    if raw_value is None or isinstance(raw_value, bool):
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _to_non_negative_float(raw_value: Any, *, default: float = 0.0) -> float:
    value = _to_float(raw_value, default=default)
    return value if value >= 0.0 else 0.0


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_iso_datetime(raw_value: str) -> datetime | None:
    candidate = raw_value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
