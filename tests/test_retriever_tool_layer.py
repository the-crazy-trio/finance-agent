from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import ErrorCode, ToolStatus
from finance_agent.contracts.tooling import ToolMeta, ToolResult
from finance_agent.retriever.errors import (
    TInvestClientError,
    TInvestErrorCategory,
    normalize_tinvest_error,
)
from finance_agent.retriever.gateway import GuardedToolGateway
from finance_agent.retriever.storage import InMemoryMemoryStorage, InMemoryUserStorage
from finance_agent.retriever.tinvest_adapter import TInvestReadonlyAdapter


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
        "PORTFOLIO_SNAPSHOT_MAX_AGE_MIN": "60",
    }


def build_strategy_payload() -> dict[str, object]:
    return {
        "client_id": "client-1",
        "base_currency": "RUB",
        "tax_residency": "RU",
        "jurisdiction_constraints": {"allowed_markets": ["MOEX"]},
        "knowledge_experience": {"experience_level": "basic"},
        "financial_profile": {"monthly_income": 100000},
        "goals": [{"goal_id": "retirement"}],
        "risk_preferences": {"risk_tolerance_self_assessed": "balanced"},
        "investable_universe": {"allowed_instruments": ["stocks", "bonds"]},
        "portfolio_policy": {"benchmark": "custom"},
        "explainability_preferences": {"wants_short_reports": True},
    }


class FakeTInvestClient:
    def __init__(
        self,
        response: dict[str, object] | None = None,
        error: TInvestClientError | None = None,
        account_ids: list[str] | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self._account_ids = account_ids or []
        self.last_call: dict[str, object] | None = None

    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> dict[str, object]:
        self.last_call = {
            "account_id": account_id,
            "correlation_id": correlation_id,
            "timeout_ms": timeout_ms,
        }
        if self._error is not None:
            raise self._error
        if self._response is None:
            raise RuntimeError("test client requires response")
        return self._response

    def list_account_ids(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> list[str]:
        if self._error is not None:
            raise self._error
        del correlation_id, timeout_ms
        return list(self._account_ids)


class MultiAccountFakeTInvestClient:
    def __init__(
        self,
        responses_by_account: dict[str, dict[str, object]],
        failing_accounts: set[str] | None = None,
    ) -> None:
        self._responses_by_account = responses_by_account
        self._failing_accounts = failing_accounts or set()
        self.calls: list[str] = []

    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> dict[str, object]:
        if account_id is None:
            raise TInvestClientError(
                category=TInvestErrorCategory.INVALID_ARGUMENT,
                message="account_id is required for test client",
            )
        self.calls.append(account_id)
        if account_id in self._failing_accounts:
            raise TInvestClientError(
                category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
                message=f"upstream failed for {account_id}",
            )
        payload = self._responses_by_account.get(account_id)
        if payload is None:
            raise RuntimeError(f"missing test payload for account_id={account_id}")
        return payload

    def list_account_ids(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> list[str]:
        del correlation_id, timeout_ms
        return sorted(self._responses_by_account)


class _FakeMarketDataAdapter:
    def show_macro_indicators(
        self,
        *,
        keys: list[str] | None,
        correlation_id: str,
    ) -> ToolResult:
        del keys
        return ToolResult.ok(
            data={
                "indicators": [
                    {
                        "key": "key_rate",
                        "value": 16.0,
                    }
                ],
                "missing": [],
                "is_partial": False,
            },
            meta=ToolMeta(correlation_id=correlation_id, coverage=1.0),
        )

    def show_issuer_indicators(self, *, ticker: str, correlation_id: str) -> ToolResult:
        return ToolResult.ok(
            data={
                "ticker": ticker,
                "issuer_indicators": {
                    "last": 250.0,
                    "close": 248.0,
                },
                "is_partial": False,
            },
            meta=ToolMeta(correlation_id=correlation_id, coverage=1.0),
        )


class RetrieverToolLayerTests(unittest.TestCase):
    def test_error_mapping_timeout_and_rate_limit_flags_retriable(self) -> None:
        timeout_error = normalize_tinvest_error(TInvestErrorCategory.TIMEOUT, "deadline exceeded")
        rate_limited_error = normalize_tinvest_error(
            TInvestErrorCategory.RATE_LIMIT,
            "too many requests",
        )

        self.assertEqual(timeout_error.code, ErrorCode.TIMEOUT)
        self.assertTrue(timeout_error.retriable)
        self.assertEqual(rate_limited_error.code, ErrorCode.RATE_LIMITED)
        self.assertTrue(rate_limited_error.retriable)

    def test_error_mapping_permission_denied_is_non_retriable_scope_error(self) -> None:
        mapped_error = normalize_tinvest_error(
            TInvestErrorCategory.PERMISSION_DENIED,
            "forbidden",
        )

        self.assertEqual(mapped_error.code, ErrorCode.UNAUTHORIZED_SCOPE)
        self.assertFalse(mapped_error.retriable)

    def test_gateway_rejects_unknown_tool_name(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=InMemoryUserStorage(),
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={
                        "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                        "positions": [],
                        "portfolio_value": 0.0,
                    }
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_delete",
            principal_id="local_user",
            correlation_id="corr-1",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.VALIDATION_ERROR)

    def test_gateway_exposes_required_tool_allowlist(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=InMemoryUserStorage(),
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={
                        "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                        "positions": [],
                        "portfolio_value": 0.0,
                    }
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
        )

        self.assertEqual(
            gateway.list_tools(),
            (
                "issuer_indicators_show",
                "macro_indicators_show",
                "portfolio_accounts",
                "portfolio_collect",
                "portfolio_show",
                "strategy_fit",
                "strategy_save",
            ),
        )

    def test_macro_indicators_tool_uses_market_adapter(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=InMemoryUserStorage(),
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={
                        "as_of_ts": datetime.now(UTC),
                        "positions": [],
                    }
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
            market_data_adapter=_FakeMarketDataAdapter(),
        )

        result = gateway.execute(
            tool_name="macro_indicators_show",
            principal_id="local_user",
            correlation_id="corr-macro-1",
            payload={"keys": ["key_rate"]},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.data["indicators"][0]["key"], "key_rate")

    def test_issuer_indicators_tool_returns_validation_error_without_ticker(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=InMemoryUserStorage(),
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={
                        "as_of_ts": datetime.now(UTC),
                        "positions": [],
                    }
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
            market_data_adapter=_FakeMarketDataAdapter(),
        )

        result = gateway.execute(
            tool_name="issuer_indicators_show",
            principal_id="local_user",
            correlation_id="corr-issuer-1",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.VALIDATION_ERROR)

    def test_portfolio_accounts_returns_scoped_accounts(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-2")
        user_storage.grant_account_access("local_user", "acc-1")
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(response={"as_of_ts": datetime.now(UTC), "positions": []}),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_accounts",
            principal_id="local_user",
            correlation_id="corr-accounts-scope",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.data["source"], "scope")
        self.assertEqual(result.data["account_ids"], ["acc-1", "acc-2"])

    def test_portfolio_accounts_discovers_and_grants_scope_from_upstream(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={"as_of_ts": datetime.now(UTC), "positions": []},
                    account_ids=["acc-1", "acc-2"],
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_accounts",
            principal_id="local_user",
            correlation_id="corr-accounts-upstream",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.data["source"], "upstream")
        self.assertEqual(result.data["account_ids"], ["acc-1", "acc-2"])
        self.assertTrue(user_storage.has_account_access("local_user", "acc-1"))
        self.assertTrue(user_storage.has_account_access("local_user", "acc-2"))

    def test_gateway_blocks_account_outside_principal_scope(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        client = FakeTInvestClient(
            response={
                "account_id": "acc-2",
                "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                "positions": [],
                "portfolio_value": 0.0,
            }
        )
        memory_storage = InMemoryMemoryStorage()
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=client,
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-2",
            payload={"account_id": "acc-2"},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.UNAUTHORIZED_SCOPE)

    def test_gateway_autodiscovers_scope_for_known_account_id(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        client = FakeTInvestClient(
            response={
                "account_id": "acc-2",
                "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                "positions": [],
                "portfolio_value": 0.0,
            },
            account_ids=["acc-1", "acc-2"],
        )
        memory_storage = InMemoryMemoryStorage()
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=client,
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-scope-discover",
            payload={"account_id": "acc-2"},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertTrue(user_storage.has_account_access("local_user", "acc-2"))

    def test_gateway_returns_validation_error_for_non_string_account_id(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=InMemoryUserStorage(),
            memory_storage=InMemoryMemoryStorage(),
            portfolio_adapter=TInvestReadonlyAdapter(
                client=FakeTInvestClient(
                    response={
                        "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                        "positions": [],
                        "portfolio_value": 0.0,
                    }
                ),
                runtime_config=runtime_config,
                memory_storage=InMemoryMemoryStorage(),
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-3",
            payload={"account_id": 42},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.VALIDATION_ERROR)

    def test_portfolio_collect_propagates_correlation_id_to_tool_meta(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        client = FakeTInvestClient(
            response={
                "account_id": "acc-1",
                "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                "positions": [
                    {
                        "ticker": "SBER",
                        "quantity": 1,
                        "market_value": 300.0,
                        "currency": "RUB",
                        "asset_class": "equity",
                    }
                ],
                "portfolio_value": 300.0,
                "coverage": 1.0,
            }
        )
        memory_storage = InMemoryMemoryStorage()
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=client,
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-collect-1",
            payload={"account_id": "acc-1"},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.meta.correlation_id, "corr-collect-1")
        self.assertEqual(client.last_call["correlation_id"], "corr-collect-1")

    def test_collect_uses_fresh_snapshot_fallback_on_upstream_failure(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        seed_client = FakeTInvestClient(
            response={
                "account_id": "acc-1",
                "as_of_ts": now - timedelta(minutes=15),
                "positions": [
                    {
                        "ticker": "GAZP",
                        "quantity": 2,
                        "market_value": 340.0,
                        "currency": "RUB",
                        "asset_class": "equity",
                    }
                ],
                "portfolio_value": 340.0,
            }
        )
        seed_adapter = TInvestReadonlyAdapter(
            client=seed_client,
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        seed_adapter.collect_portfolio(
            principal_id="local_user",
            account_id="acc-1",
            correlation_id="corr-seed",
        )

        failing_adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                error=TInvestClientError(
                    category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
                    message="upstream down",
                )
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=failing_adapter,
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-fallback",
            payload={"account_id": "acc-1"},
        )

        self.assertEqual(result.status, ToolStatus.PARTIAL)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)
        self.assertIn("snapshot", result.data)

    def test_portfolio_collect_aggregates_accessible_accounts_without_account_id(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")
        user_storage.grant_account_access("local_user", "acc-2")

        client = MultiAccountFakeTInvestClient(
            responses_by_account={
                "acc-1": {
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "SBER",
                            "quantity": 2,
                            "market_value": 400.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        }
                    ],
                    "portfolio_value": 400.0,
                    "coverage": 1.0,
                },
                "acc-2": {
                    "account_id": "acc-2",
                    "as_of_ts": datetime(2026, 4, 17, 12, 5, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "OFZ",
                            "quantity": 3,
                            "market_value": 600.0,
                            "currency": "RUB",
                            "asset_class": "bond",
                        }
                    ],
                    "portfolio_value": 600.0,
                    "coverage": 0.8,
                },
            }
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=client,
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-aggregate",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(client.calls, ["acc-1", "acc-2"])
        self.assertEqual(result.data["account_ids"], ["acc-1", "acc-2"])
        self.assertEqual(result.data["snapshot"]["account_id"], None)
        self.assertEqual(len(result.data["snapshot"]["positions"]), 2)
        self.assertEqual(result.data["snapshot"]["portfolio_value"], 1000.0)
        self.assertAlmostEqual(result.data["snapshot"]["coverage"], 0.88)

    def test_portfolio_collect_aggregate_returns_partial_if_one_account_fails(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")
        user_storage.grant_account_access("local_user", "acc-2")

        client = MultiAccountFakeTInvestClient(
            responses_by_account={
                "acc-1": {
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "SBER",
                            "quantity": 1,
                            "market_value": 500.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        }
                    ],
                    "portfolio_value": 500.0,
                    "coverage": 1.0,
                },
                "acc-2": {
                    "account_id": "acc-2",
                    "as_of_ts": datetime(2026, 4, 17, 12, 1, tzinfo=UTC),
                    "positions": [],
                    "portfolio_value": 0.0,
                    "coverage": 1.0,
                },
            },
            failing_accounts={"acc-2"},
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=TInvestReadonlyAdapter(
                client=client,
                runtime_config=runtime_config,
                memory_storage=memory_storage,
            ),
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-aggregate-partial",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.PARTIAL)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)
        self.assertEqual(result.data["account_ids"], ["acc-1"])
        self.assertEqual(result.data["snapshot"]["portfolio_value"], 500.0)

    def test_portfolio_show_without_account_id_returns_aggregate_snapshot(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")
        user_storage.grant_account_access("local_user", "acc-2")
        now = datetime(2026, 4, 17, 12, 10, tzinfo=UTC)

        client = MultiAccountFakeTInvestClient(
            responses_by_account={
                "acc-1": {
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "SBER",
                            "quantity": 1,
                            "market_value": 400.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        }
                    ],
                    "portfolio_value": 400.0,
                    "coverage": 1.0,
                },
                "acc-2": {
                    "account_id": "acc-2",
                    "as_of_ts": datetime(2026, 4, 17, 12, 1, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "LKOH",
                            "quantity": 1,
                            "market_value": 600.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        }
                    ],
                    "portfolio_value": 600.0,
                    "coverage": 1.0,
                },
            }
        )
        adapter = TInvestReadonlyAdapter(
            client=client,
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=adapter,
        )

        gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-seed-a1",
            payload={"account_id": "acc-1"},
        )
        gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-seed-a2",
            payload={"account_id": "acc-2"},
        )

        result = gateway.execute(
            tool_name="portfolio_show",
            principal_id="local_user",
            correlation_id="corr-show-all",
            payload={},
        )

        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.data["snapshot"]["account_id"], None)
        self.assertEqual(len(result.data["snapshot"]["positions"]), 2)
        self.assertEqual(result.data["snapshot"]["portfolio_value"], 1000.0)

    def test_collect_returns_error_when_only_stale_snapshot_exists(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        seed_adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                response={
                    "account_id": "acc-1",
                    "as_of_ts": now - timedelta(minutes=120),
                    "positions": [],
                    "portfolio_value": 0.0,
                }
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        seed_adapter.collect_portfolio(
            principal_id="local_user",
            account_id="acc-1",
            correlation_id="corr-seed-stale",
        )

        failing_adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                error=TInvestClientError(
                    category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
                    message="still down",
                )
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=failing_adapter,
        )

        result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-no-fallback",
            payload={"account_id": "acc-1"},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)

    def test_portfolio_show_returns_error_when_snapshot_is_stale(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        memory_storage = InMemoryMemoryStorage()
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                response={
                    "account_id": "acc-1",
                    "as_of_ts": now - timedelta(minutes=90),
                    "positions": [],
                    "portfolio_value": 0.0,
                }
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
            now_fn=lambda: now,
        )
        adapter.collect_portfolio(
            principal_id="local_user",
            account_id="acc-1",
            correlation_id="corr-seed-show",
        )

        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=adapter,
        )
        result = gateway.execute(
            tool_name="portfolio_show",
            principal_id="local_user",
            correlation_id="corr-show-stale",
            payload={"account_id": "acc-1"},
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)

    def test_strategy_save_and_fit_are_available_via_gateway(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        memory_storage = InMemoryMemoryStorage()
        adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                response={
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [],
                    "portfolio_value": 0.0,
                    "coverage": 0.4,
                }
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=adapter,
        )

        save_result = gateway.execute(
            tool_name="strategy_save",
            principal_id="local_user",
            correlation_id="corr-strategy-save",
            payload={"strategy": build_strategy_payload()},
        )
        self.assertEqual(save_result.status, ToolStatus.OK)

        user_storage.grant_account_access("local_user", "acc-1")
        gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-collect-fit",
            payload={"account_id": "acc-1"},
        )

        fit_result = gateway.execute(
            tool_name="strategy_fit",
            principal_id="local_user",
            correlation_id="corr-fit",
            payload={"account_id": "acc-1"},
        )
        self.assertEqual(fit_result.status, ToolStatus.OK)
        self.assertEqual(fit_result.data["strategy_fit"]["verdict"], "unknown")

    def test_strategy_fit_returns_not_fit_when_constraints_are_violated(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        memory_storage = InMemoryMemoryStorage()
        adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                response={
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "SBER",
                            "quantity": 1,
                            "market_value": 900.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        },
                        {
                            "ticker": "OFZ",
                            "quantity": 1,
                            "market_value": 100.0,
                            "currency": "RUB",
                            "asset_class": "bond",
                        },
                    ],
                    "portfolio_value": 1000.0,
                    "coverage": 1.0,
                }
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=adapter,
        )

        strategy_payload = build_strategy_payload()
        strategy_payload["risk_preferences"]["max_single_name_weight"] = 0.5
        strategy_payload["risk_preferences"]["max_sector_weight"] = 0.8
        strategy_payload["portfolio_policy"]["target_asset_allocation"] = {
            "equity": 0.6,
            "bond": 0.4,
        }
        strategy_payload["portfolio_policy"]["rebalance_policy"] = {
            "frequency": "quarterly",
            "drift_threshold_pct": 10,
        }

        save_result = gateway.execute(
            tool_name="strategy_save",
            principal_id="local_user",
            correlation_id="corr-strategy-save-strict",
            payload={"strategy": strategy_payload},
        )
        self.assertEqual(save_result.status, ToolStatus.OK)

        collect_result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-collect-fit-strict",
            payload={"account_id": "acc-1"},
        )
        self.assertEqual(collect_result.status, ToolStatus.OK)

        fit_result = gateway.execute(
            tool_name="strategy_fit",
            principal_id="local_user",
            correlation_id="corr-fit-strict",
            payload={"account_id": "acc-1"},
        )
        self.assertEqual(fit_result.status, ToolStatus.OK)
        self.assertEqual(fit_result.data["strategy_fit"]["verdict"], "not_fit")
        self.assertIn("max_single_name_weight", fit_result.data["diagnostics"]["violations"])

    def test_strategy_fit_returns_not_fit_on_market_access_hard_rule(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        user_storage = InMemoryUserStorage()
        user_storage.grant_account_access("local_user", "acc-1")

        memory_storage = InMemoryMemoryStorage()
        adapter = TInvestReadonlyAdapter(
            client=FakeTInvestClient(
                response={
                    "account_id": "acc-1",
                    "as_of_ts": datetime(2026, 4, 17, 12, 0, tzinfo=UTC),
                    "positions": [
                        {
                            "ticker": "SBER",
                            "quantity": 1,
                            "market_value": 900.0,
                            "currency": "RUB",
                            "asset_class": "equity",
                        },
                        {
                            "ticker": "OFZ",
                            "quantity": 1,
                            "market_value": 100.0,
                            "currency": "RUB",
                            "asset_class": "bond",
                        },
                    ],
                    "portfolio_value": 1000.0,
                    "coverage": 1.0,
                }
            ),
            runtime_config=runtime_config,
            memory_storage=memory_storage,
        )
        gateway = GuardedToolGateway(
            runtime_config=runtime_config,
            user_storage=user_storage,
            memory_storage=memory_storage,
            portfolio_adapter=adapter,
        )

        strategy_payload = build_strategy_payload()
        strategy_payload["risk_preferences"]["max_single_name_weight"] = 0.95
        strategy_payload["investable_universe"]["allowed_instruments"] = ["bonds"]
        strategy_payload["investable_universe"]["disallowed_instruments"] = ["stocks"]

        save_result = gateway.execute(
            tool_name="strategy_save",
            principal_id="local_user",
            correlation_id="corr-strategy-save-hard-rule",
            payload={"strategy": strategy_payload},
        )
        self.assertEqual(save_result.status, ToolStatus.OK)

        collect_result = gateway.execute(
            tool_name="portfolio_collect",
            principal_id="local_user",
            correlation_id="corr-collect-hard-rule",
            payload={"account_id": "acc-1"},
        )
        self.assertEqual(collect_result.status, ToolStatus.OK)

        fit_result = gateway.execute(
            tool_name="strategy_fit",
            principal_id="local_user",
            correlation_id="corr-fit-hard-rule",
            payload={"account_id": "acc-1"},
        )
        self.assertEqual(fit_result.status, ToolStatus.OK)
        self.assertEqual(fit_result.data["strategy_fit"]["verdict"], "not_fit")
        self.assertIn(
            "hard_rule_inaccessible_market_primary_driver",
            fit_result.data["diagnostics"]["hard_rule_violations"],
        )


if __name__ == "__main__":
    unittest.main()
