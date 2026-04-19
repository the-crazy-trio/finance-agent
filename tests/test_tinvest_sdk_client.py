from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from grpc import StatusCode
from t_tech.invest.exceptions import RequestError, UnauthenticatedError

from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory
from finance_agent.retriever.tinvest_sdk_client import TInvestSdkReadonlyClient


class _ClientContext:
    def __init__(self, services: Any) -> None:
        self._services = services

    def __enter__(self) -> Any:
        return self._services

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _ClientFactory:
    def __init__(self, services: Any) -> None:
        self._services = services
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, token: str, *, target: str | None = None) -> _ClientContext:
        self.calls.append((token, target))
        return _ClientContext(self._services)


class _UsersService:
    def __init__(
        self,
        *,
        accounts: list[Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._accounts = accounts or []
        self._error = error

    def get_accounts(self) -> Any:
        if self._error is not None:
            raise self._error
        return SimpleNamespace(accounts=list(self._accounts))


class _OperationsService:
    def __init__(
        self,
        *,
        portfolios_by_account: dict[str, Any] | None = None,
    ) -> None:
        self._portfolios_by_account = portfolios_by_account or {}

    def get_portfolio(self, *, account_id: str = "") -> Any:
        payload = self._portfolios_by_account.get(account_id)
        if payload is None:
            raise RuntimeError(f"missing test payload for account_id={account_id}")
        return payload


def _build_account(account_id: str, status_name: str) -> Any:
    return SimpleNamespace(id=account_id, status=SimpleNamespace(name=status_name))


def _money(units: int, *, currency: str = "RUB") -> Any:
    return SimpleNamespace(units=units, nano=0, currency=currency)


def _quotation(units: int) -> Any:
    return SimpleNamespace(units=units, nano=0)


def _position(*, ticker: str, quantity: int, price: int, uid: str, instrument_type: str) -> Any:
    return SimpleNamespace(
        instrument_uid=uid,
        position_uid=f"{uid}-position",
        figi=None,
        ticker=ticker,
        instrument_type=instrument_type,
        quantity=_quotation(quantity),
        quantity_lots=_quotation(quantity),
        current_price=_money(price),
        average_position_price=_money(price),
    )


class TInvestSdkReadonlyClientTests(unittest.TestCase):
    def test_list_account_ids_filters_closed_accounts(self) -> None:
        services = SimpleNamespace(
            users=_UsersService(
                accounts=[
                    _build_account("acc-1", "ACCOUNT_STATUS_OPEN"),
                    _build_account("acc-2", "ACCOUNT_STATUS_CLOSED"),
                    _build_account("acc-3", "ACCOUNT_STATUS_NEW"),
                ]
            ),
            operations=_OperationsService(),
        )
        factory = _ClientFactory(services)
        client = TInvestSdkReadonlyClient(token="token-1", client_factory=factory)

        account_ids = client.list_account_ids(correlation_id="corr-1", timeout_ms=5000)

        self.assertEqual(account_ids, ["acc-1", "acc-3"])
        self.assertEqual(factory.calls[0], ("token-1", None))

    def test_get_portfolio_without_account_id_aggregates_all_accounts(self) -> None:
        services = SimpleNamespace(
            users=_UsersService(
                accounts=[
                    _build_account("acc-1", "ACCOUNT_STATUS_OPEN"),
                    _build_account("acc-2", "ACCOUNT_STATUS_OPEN"),
                ]
            ),
            operations=_OperationsService(
                portfolios_by_account={
                    "acc-1": SimpleNamespace(
                        positions=[
                            _position(
                                ticker="SBER",
                                quantity=1,
                                price=100,
                                uid="uid-sber",
                                instrument_type="share",
                            )
                        ],
                        total_amount_portfolio=_money(100),
                    ),
                    "acc-2": SimpleNamespace(
                        positions=[
                            _position(
                                ticker="OFZ",
                                quantity=2,
                                price=100,
                                uid="uid-ofz",
                                instrument_type="bond",
                            )
                        ],
                        total_amount_portfolio=_money(200),
                    ),
                }
            ),
        )
        client = TInvestSdkReadonlyClient(
            token="token-1",
            client_factory=_ClientFactory(services),
            now_fn=lambda: datetime(2026, 4, 19, 12, 0, tzinfo=UTC),
        )

        result = client.get_portfolio(
            account_id=None,
            correlation_id="corr-aggregate",
            timeout_ms=5000,
        )

        self.assertEqual(result["account_id"], None)
        self.assertEqual(result["account_ids"], ["acc-1", "acc-2"])
        self.assertEqual(result["portfolio_value"], 300.0)
        self.assertEqual(len(result["positions"]), 2)

    def test_list_account_ids_maps_grpc_error_to_normalized_category(self) -> None:
        services = SimpleNamespace(
            users=_UsersService(
                error=RequestError(
                    StatusCode.UNAVAILABLE,
                    "upstream unavailable",
                    None,
                )
            ),
            operations=_OperationsService(),
        )
        client = TInvestSdkReadonlyClient(token="token-1", client_factory=_ClientFactory(services))

        with self.assertRaises(TInvestClientError) as context:
            client.list_account_ids(correlation_id="corr-error", timeout_ms=5000)

        self.assertEqual(context.exception.category, TInvestErrorCategory.UPSTREAM_UNAVAILABLE)

    def test_list_account_ids_maps_unauthenticated_error(self) -> None:
        services = SimpleNamespace(
            users=_UsersService(
                error=UnauthenticatedError(
                    StatusCode.UNAUTHENTICATED,
                    "invalid token",
                    None,
                )
            ),
            operations=_OperationsService(),
        )
        client = TInvestSdkReadonlyClient(token="token-1", client_factory=_ClientFactory(services))

        with self.assertRaises(TInvestClientError) as context:
            client.list_account_ids(correlation_id="corr-auth", timeout_ms=5000)

        self.assertEqual(context.exception.category, TInvestErrorCategory.UNAUTHENTICATED)


if __name__ == "__main__":
    unittest.main()
