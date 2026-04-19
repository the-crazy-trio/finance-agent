from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from grpc import StatusCode
from t_tech.invest import Client
from t_tech.invest.exceptions import RequestError, UnauthenticatedError

from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory
from finance_agent.retriever.tinvest_http_client import (
    _merge_portfolio_payloads,
    _normalize_portfolio_payload,
)

ServicesCall = Callable[[Any], Any]


class TInvestSdkReadonlyClient:
    def __init__(
        self,
        *,
        token: str,
        target: str | None = None,
        ca_bundle_path: str | None = None,
        client_factory: Callable[..., Any] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_token = token.strip()
        if not normalized_token:
            raise ValueError("token must not be blank")

        normalized_target = target.strip() if isinstance(target, str) else ""
        normalized_bundle = ca_bundle_path.strip() if isinstance(ca_bundle_path, str) else ""

        self._token = normalized_token
        self._target = normalized_target or None
        self._ca_bundle_path = normalized_bundle or None
        self._client_factory = client_factory or Client
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        del correlation_id, timeout_ms
        if account_id is None:
            return self._get_aggregated_portfolio()
        return self._get_portfolio_for_account(account_id)

    def list_account_ids(self, *, correlation_id: str, timeout_ms: int) -> list[str]:
        del correlation_id, timeout_ms
        response = self._invoke(lambda services: services.users.get_accounts())

        raw_accounts = getattr(response, "accounts", None)
        if not isinstance(raw_accounts, list):
            raise TInvestClientError(
                category=TInvestErrorCategory.INTERNAL,
                message="T-Invest accounts response is malformed",
            )

        account_ids: list[str] = []
        for account in raw_accounts:
            account_id = _normalize_text(getattr(account, "id", None))
            if account_id is None:
                continue

            status_name = _normalize_status_name(getattr(account, "status", None))
            if "CLOSED" in status_name:
                continue

            if account_id not in account_ids:
                account_ids.append(account_id)

        if not account_ids:
            raise TInvestClientError(
                category=TInvestErrorCategory.ACCOUNT_NOT_FOUND,
                message="No accessible broker accounts were found",
            )
        return account_ids

    def _get_aggregated_portfolio(self) -> Mapping[str, Any]:
        account_ids = self.list_account_ids(correlation_id="sdk-aggregate", timeout_ms=0)
        payloads = [self._get_portfolio_for_account(account_id) for account_id in account_ids]
        return _merge_portfolio_payloads(payloads)

    def _get_portfolio_for_account(self, account_id: str) -> Mapping[str, Any]:
        response = self._invoke(
            lambda services: services.operations.get_portfolio(account_id=account_id)
        )
        raw_payload = _portfolio_response_to_payload(response)
        return _normalize_portfolio_payload(
            raw_payload,
            account_id=account_id,
            now_ts=self._now_fn(),
        )

    def _invoke(self, callback: ServicesCall) -> Any:
        try:
            with self._grpc_roots_context():
                kwargs: dict[str, Any] = {}
                if self._target is not None:
                    kwargs["target"] = self._target
                with self._client_factory(self._token, **kwargs) as services:
                    return callback(services)
        except UnauthenticatedError as error:
            raise TInvestClientError(
                category=TInvestErrorCategory.UNAUTHENTICATED,
                message=_read_error_details(error),
            ) from error
        except RequestError as error:
            category = _category_from_status_code(error.code, _read_error_details(error))
            raise TInvestClientError(
                category=category,
                message=_read_error_details(error),
            ) from error
        except TInvestClientError:
            raise
        except Exception as error:
            raise TInvestClientError(
                category=TInvestErrorCategory.INTERNAL,
                message=f"T-Invest SDK call failed: {error}",
            ) from error

    @contextmanager
    def _grpc_roots_context(self):
        if self._ca_bundle_path is None:
            yield
            return

        env_key = "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"
        previous_value = os.environ.get(env_key)
        os.environ[env_key] = self._ca_bundle_path
        try:
            yield
        finally:
            if previous_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = previous_value


def _portfolio_response_to_payload(response: Any) -> dict[str, Any]:
    raw_positions = getattr(response, "positions", []) or []
    normalized_positions = [_portfolio_position_to_payload(item) for item in raw_positions]
    return {
        "positions": normalized_positions,
        "totalAmountPortfolio": _money_value_to_payload(
            getattr(response, "total_amount_portfolio", None)
        ),
    }


def _portfolio_position_to_payload(position: Any) -> dict[str, Any]:
    return {
        "instrumentUid": _normalize_text(getattr(position, "instrument_uid", None)),
        "positionUid": _normalize_text(getattr(position, "position_uid", None)),
        "figi": _normalize_text(getattr(position, "figi", None)),
        "ticker": _normalize_text(getattr(position, "ticker", None)),
        "instrumentType": _normalize_text(getattr(position, "instrument_type", None)),
        "quantity": _quotation_to_payload(getattr(position, "quantity", None)),
        "quantityLots": _quotation_to_payload(getattr(position, "quantity_lots", None)),
        "currentPrice": _money_value_to_payload(getattr(position, "current_price", None)),
        "averagePositionPrice": _money_value_to_payload(
            getattr(position, "average_position_price", None)
        ),
    }


def _money_value_to_payload(value: Any) -> dict[str, Any] | None:
    units = getattr(value, "units", None)
    nano = getattr(value, "nano", None)
    if units is None and nano is None:
        return None

    currency = _normalize_text(getattr(value, "currency", None))
    payload: dict[str, Any] = {
        "units": int(units or 0),
        "nano": int(nano or 0),
    }
    if currency is not None:
        payload["currency"] = currency
    return payload


def _quotation_to_payload(value: Any) -> dict[str, Any] | None:
    units = getattr(value, "units", None)
    nano = getattr(value, "nano", None)
    if units is None and nano is None:
        return None
    return {
        "units": int(units or 0),
        "nano": int(nano or 0),
    }


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _normalize_status_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).strip().upper()


def _category_from_status_code(status_code: Any, details: str) -> TInvestErrorCategory:
    mapping = {
        StatusCode.INVALID_ARGUMENT: TInvestErrorCategory.INVALID_ARGUMENT,
        StatusCode.UNAUTHENTICATED: TInvestErrorCategory.UNAUTHENTICATED,
        StatusCode.PERMISSION_DENIED: TInvestErrorCategory.PERMISSION_DENIED,
        StatusCode.NOT_FOUND: TInvestErrorCategory.NOT_FOUND,
        StatusCode.RESOURCE_EXHAUSTED: TInvestErrorCategory.RATE_LIMIT,
        StatusCode.DEADLINE_EXCEEDED: TInvestErrorCategory.TIMEOUT,
        StatusCode.UNAVAILABLE: TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        StatusCode.INTERNAL: TInvestErrorCategory.INTERNAL,
        StatusCode.UNIMPLEMENTED: TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
    }
    if status_code in mapping:
        return mapping[status_code]

    details_text = details.lower()
    if "account" in details_text and "not found" in details_text:
        return TInvestErrorCategory.ACCOUNT_NOT_FOUND
    return TInvestErrorCategory.INTERNAL


def _read_error_details(error: Any) -> str:
    details = _normalize_text(getattr(error, "details", None))
    if details is not None:
        return details
    message = _normalize_text(str(error))
    return message or "T-Invest SDK request failed"
