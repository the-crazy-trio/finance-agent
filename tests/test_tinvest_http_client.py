from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any

from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory
from finance_agent.retriever.tinvest_http_client import TInvestHttpReadonlyClient


class TInvestHttpReadonlyClientTests(unittest.TestCase):
    def test_get_portfolio_normalizes_single_account_response(self) -> None:
        captured_requests: list[tuple[str, dict[str, Any]]] = []

        def request_fn(
            url: str,
            method: str,
            body: dict[str, Any] | None,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> dict[str, Any]:
            self.assertIn("Authorization", headers)
            self.assertGreater(timeout_sec, 0)
            self.assertEqual(method, "POST")
            self.assertIsNotNone(body)
            captured_requests.append((url, body))
            return {
                "positions": [
                    {
                        "instrumentUid": "uid-sber",
                        "figi": "figi-sber",
                        "instrumentType": "share",
                        "quantity": {"units": "2", "nano": 0},
                        "currentPrice": {"units": "100", "nano": 0, "currency": "RUB"},
                    }
                ],
                "totalAmountPortfolio": {"units": "200", "nano": 0, "currency": "RUB"},
            }

        client = TInvestHttpReadonlyClient(
            token="token-1",
            request_fn=request_fn,
            now_fn=lambda: datetime(2026, 4, 19, 10, 0, tzinfo=UTC),
        )

        result = client.get_portfolio(
            account_id="acc-1",
            correlation_id="corr-1",
            timeout_ms=12345,
        )

        self.assertEqual(len(captured_requests), 1)
        self.assertTrue(captured_requests[0][0].endswith("OperationsService/GetPortfolio"))
        self.assertEqual(captured_requests[0][1], {"accountId": "acc-1"})
        self.assertEqual(result["account_id"], "acc-1")
        self.assertEqual(result["portfolio_value"], 200.0)
        self.assertEqual(result["coverage"], 1.0)
        self.assertEqual(result["unknown_share"], 0.0)
        self.assertEqual(len(result["positions"]), 1)
        self.assertEqual(result["positions"][0]["market_value"], 200.0)
        self.assertEqual(result["positions"][0]["asset_class"], "equity")

    def test_get_portfolio_without_account_id_aggregates_multiple_accounts(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def request_fn(
            url: str,
            method: str,
            body: dict[str, Any] | None,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> dict[str, Any]:
            del headers, timeout_sec
            self.assertEqual(method, "POST")
            self.assertIsNotNone(body)
            calls.append((url, body))
            if url.endswith("UsersService/GetAccounts"):
                return {
                    "accounts": [
                        {"id": "acc-1", "status": "ACCOUNT_STATUS_OPEN"},
                        {"id": "acc-2", "status": "ACCOUNT_STATUS_OPEN"},
                    ]
                }

            account_id = body["accountId"]
            if account_id == "acc-1":
                return {
                    "asOf": "2026-04-19T10:00:00+00:00",
                    "positions": [
                        {
                            "instrumentUid": "uid-1",
                            "instrumentType": "share",
                            "quantity": {"units": "1", "nano": 0},
                            "currentPrice": {"units": "100", "nano": 0, "currency": "RUB"},
                        }
                    ],
                    "totalAmountPortfolio": {"units": "100", "nano": 0, "currency": "RUB"},
                }
            return {
                "asOf": "2026-04-19T10:05:00+00:00",
                "positions": [
                    {
                        "instrumentUid": "uid-2",
                        "instrumentType": "bond",
                        "quantity": {"units": "2", "nano": 0},
                        "currentPrice": {"units": "100", "nano": 0, "currency": "RUB"},
                    }
                ],
                "totalAmountPortfolio": {"units": "200", "nano": 0, "currency": "RUB"},
            }

        client = TInvestHttpReadonlyClient(token="token-1", request_fn=request_fn)

        result = client.get_portfolio(
            account_id=None,
            correlation_id="corr-agg",
            timeout_ms=5000,
        )

        self.assertEqual(result["account_id"], None)
        self.assertEqual(result["account_ids"], ["acc-1", "acc-2"])
        self.assertEqual(result["portfolio_value"], 300.0)
        self.assertEqual(len(result["positions"]), 2)
        self.assertEqual(result["positions"][1]["asset_class"], "bond")
        self.assertTrue(calls[0][0].endswith("UsersService/GetAccounts"))

    def test_list_account_ids_raises_when_no_active_accounts(self) -> None:
        def request_fn(
            url: str,
            method: str,
            body: dict[str, Any] | None,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> dict[str, Any]:
            self.assertEqual(method, "POST")
            self.assertIsNotNone(body)
            del url, body, headers, timeout_sec
            return {"accounts": [{"id": "acc-closed", "status": "ACCOUNT_STATUS_CLOSED"}]}

        client = TInvestHttpReadonlyClient(token="token-1", request_fn=request_fn)

        with self.assertRaises(TInvestClientError) as context:
            client.list_account_ids(correlation_id="corr-none", timeout_ms=5000)

        self.assertEqual(context.exception.category, TInvestErrorCategory.ACCOUNT_NOT_FOUND)

    def test_list_account_ids_retries_with_get_when_post_returns_405(self) -> None:
        calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def request_fn(
            url: str,
            method: str,
            body: dict[str, Any] | None,
            headers: dict[str, str],
            timeout_sec: float,
        ) -> dict[str, Any]:
            del headers, timeout_sec
            calls.append((url, method, body))
            if method == "POST" and url.endswith("UsersService/GetAccounts"):
                raise TInvestClientError(
                    category=TInvestErrorCategory.INTERNAL,
                    message="T-Invest API request failed with HTTP 405",
                )
            if method == "GET" and url.endswith("UsersService/GetAccounts"):
                return {
                    "accounts": [
                        {"id": "acc-1", "status": "ACCOUNT_STATUS_OPEN"},
                    ]
                }
            raise RuntimeError("unexpected request")

        client = TInvestHttpReadonlyClient(token="token-1", request_fn=request_fn)

        account_ids = client.list_account_ids(correlation_id="corr-retry-get", timeout_ms=5000)

        self.assertEqual(account_ids, ["acc-1"])
        self.assertEqual(calls[0][1], "POST")
        self.assertEqual(calls[1][1], "GET")


if __name__ == "__main__":
    unittest.main()
