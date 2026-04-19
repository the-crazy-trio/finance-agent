from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from finance_agent.analytics import build_analytics_snapshot, evaluate_strategy_fit
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import ErrorCode, ErrorDetail, ToolStatus
from finance_agent.contracts.entities import StructuredStrategy
from finance_agent.contracts.tooling import ToolMeta, ToolResult
from finance_agent.retriever.market_adapter import ExternalMarketDataAdapter
from finance_agent.retriever.storage import InMemoryMemoryStorage, InMemoryUserStorage
from finance_agent.retriever.tinvest_adapter import TInvestReadonlyAdapter


class GuardedToolGateway:
    ALLOWLIST = frozenset(
        {
            "portfolio_accounts",
            "portfolio_collect",
            "portfolio_show",
            "macro_indicators_show",
            "issuer_indicators_show",
            "strategy_save",
            "strategy_fit",
        }
    )

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        user_storage: InMemoryUserStorage,
        memory_storage: InMemoryMemoryStorage,
        portfolio_adapter: TInvestReadonlyAdapter,
        market_data_adapter: ExternalMarketDataAdapter | None = None,
    ) -> None:
        self._runtime_config = runtime_config
        self._user_storage = user_storage
        self._memory_storage = memory_storage
        self._portfolio_adapter = portfolio_adapter
        self._market_data_adapter = market_data_adapter or ExternalMarketDataAdapter(
            runtime_config=runtime_config
        )

    def execute(
        self,
        *,
        tool_name: str,
        principal_id: str,
        correlation_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        if tool_name not in self.ALLOWLIST:
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message=f"Unsupported tool: {tool_name}",
            )
        if not principal_id.strip():
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR, message="principal_id is required"
            )
        if not correlation_id.strip():
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR, message="correlation_id is required"
            )

        normalized_payload = dict(payload or {})

        if tool_name == "portfolio_accounts":
            return self._tool_portfolio_accounts(
                principal_id=principal_id,
                correlation_id=correlation_id,
            )
        if tool_name == "portfolio_collect":
            return self._tool_portfolio_collect(
                principal_id=principal_id,
                correlation_id=correlation_id,
                payload=normalized_payload,
            )
        if tool_name == "portfolio_show":
            return self._tool_portfolio_show(
                principal_id=principal_id,
                correlation_id=correlation_id,
                payload=normalized_payload,
            )
        if tool_name == "strategy_save":
            return self._tool_strategy_save(
                principal_id=principal_id,
                correlation_id=correlation_id,
                payload=normalized_payload,
            )
        if tool_name == "macro_indicators_show":
            return self._tool_macro_indicators_show(
                principal_id=principal_id,
                correlation_id=correlation_id,
                payload=normalized_payload,
            )
        if tool_name == "issuer_indicators_show":
            return self._tool_issuer_indicators_show(
                principal_id=principal_id,
                correlation_id=correlation_id,
                payload=normalized_payload,
            )
        return self._tool_strategy_fit(
            principal_id=principal_id,
            correlation_id=correlation_id,
            payload=normalized_payload,
        )

    def list_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self.ALLOWLIST))

    def _tool_portfolio_collect(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        account_id, account_error = _extract_account_id(payload, correlation_id)
        if account_error is not None:
            return account_error
        if account_id is not None:
            if not self._ensure_account_scope(
                principal_id=principal_id,
                account_id=account_id,
                correlation_id=correlation_id,
            ):
                return ToolResult.error(
                    code=ErrorCode.UNAUTHORIZED_SCOPE,
                    message="principal_id has no access to account_id",
                    details={"account_id": account_id},
                    meta=ToolMeta(correlation_id=correlation_id),
                )

            return self._portfolio_adapter.collect_portfolio(
                principal_id=principal_id,
                account_id=account_id,
                correlation_id=correlation_id,
            )

        scoped_accounts = self._user_storage.list_account_access(principal_id)
        if not scoped_accounts:
            return self._portfolio_adapter.collect_portfolio(
                principal_id=principal_id,
                account_id=None,
                correlation_id=correlation_id,
            )

        collected_errors: list[ErrorDetail] = []
        successful_account_ids: list[str] = []
        had_partial_result = False

        for scoped_account_id in scoped_accounts:
            result = self._portfolio_adapter.collect_portfolio(
                principal_id=principal_id,
                account_id=scoped_account_id,
                correlation_id=correlation_id,
            )
            if result.status == ToolStatus.ERROR:
                collected_errors.extend(result.errors)
                continue

            successful_account_ids.append(scoped_account_id)
            if result.status == ToolStatus.PARTIAL:
                had_partial_result = True
                collected_errors.extend(result.errors)

        if not successful_account_ids:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="Failed to collect portfolio snapshots for accessible accounts",
                details={
                    "account_ids": list(scoped_accounts),
                    "errors": [item.model_dump(mode="json") for item in collected_errors],
                },
                meta=ToolMeta(correlation_id=correlation_id),
            )

        merged_snapshot = self._memory_storage.get_latest_portfolio_snapshot_for_accounts(
            principal_id,
            successful_account_ids,
        )
        if merged_snapshot is None:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="No collected portfolio snapshot is available",
                details={"account_ids": successful_account_ids},
                meta=ToolMeta(correlation_id=correlation_id),
            )

        payload_data = {
            "snapshot": merged_snapshot.model_dump(mode="json"),
            "account_ids": successful_account_ids,
        }
        meta = ToolMeta(
            source_ts=merged_snapshot.as_of_ts,
            coverage=merged_snapshot.coverage,
            correlation_id=correlation_id,
        )
        if had_partial_result or collected_errors:
            return ToolResult.partial(
                data=payload_data,
                errors=collected_errors,
                meta=meta,
            )
        return ToolResult.ok(data=payload_data, meta=meta)

    def _tool_portfolio_accounts(
        self,
        *,
        principal_id: str,
        correlation_id: str,
    ) -> ToolResult:
        scoped_accounts = list(self._user_storage.list_account_access(principal_id))
        if scoped_accounts:
            return ToolResult.ok(
                data={"account_ids": scoped_accounts, "source": "scope"},
                meta=ToolMeta(correlation_id=correlation_id),
            )

        accounts_result = self._portfolio_adapter.list_accounts(correlation_id=correlation_id)
        if accounts_result.status == ToolStatus.ERROR:
            return accounts_result

        account_ids = _read_account_ids(accounts_result)
        if not account_ids:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="No accessible broker accounts are available",
                meta=ToolMeta(correlation_id=correlation_id),
            )
        for account_id in account_ids:
            self._user_storage.grant_account_access(principal_id, account_id)

        return ToolResult.ok(
            data={"account_ids": account_ids, "source": "upstream"},
            meta=accounts_result.meta,
        )

    def _tool_portfolio_show(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        account_id, account_error = _extract_account_id(payload, correlation_id)
        if account_error is not None:
            return account_error
        if not self._ensure_account_scope(
            principal_id=principal_id,
            account_id=account_id,
            correlation_id=correlation_id,
        ):
            return ToolResult.error(
                code=ErrorCode.UNAUTHORIZED_SCOPE,
                message="principal_id has no access to account_id",
                details={"account_id": account_id},
                meta=ToolMeta(correlation_id=correlation_id),
            )

        snapshot = self._memory_storage.get_latest_portfolio_snapshot(principal_id, account_id)
        if snapshot is None:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="No portfolio snapshot is available",
                meta=ToolMeta(correlation_id=correlation_id),
            )
        if not self._portfolio_adapter.is_snapshot_fresh(snapshot):
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="Portfolio snapshot is stale and requires refresh",
                details={"max_age_minutes": self._runtime_config.portfolio_snapshot_max_age_min},
                meta=ToolMeta(
                    source_ts=snapshot.as_of_ts,
                    coverage=snapshot.coverage,
                    correlation_id=correlation_id,
                ),
            )

        return ToolResult.ok(
            data={"snapshot": snapshot.model_dump(mode="json")},
            meta=ToolMeta(
                source_ts=snapshot.as_of_ts,
                coverage=snapshot.coverage,
                correlation_id=correlation_id,
            ),
        )

    def _tool_strategy_save(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        raw_strategy = payload.get("strategy")
        if not isinstance(raw_strategy, Mapping):
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="strategy payload is required",
                meta=ToolMeta(correlation_id=correlation_id),
            )

        try:
            strategy = StructuredStrategy.model_validate(dict(raw_strategy))
        except ValidationError as error:
            return ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="strategy payload is invalid",
                details={"validation_error": str(error)},
                meta=ToolMeta(correlation_id=correlation_id),
            )

        self._memory_storage.save_strategy(principal_id, strategy)
        return ToolResult.ok(
            data={"strategy": strategy.model_dump(mode="json")},
            meta=ToolMeta(correlation_id=correlation_id),
        )

    def _tool_strategy_fit(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        account_id, account_error = _extract_account_id(payload, correlation_id)
        if account_error is not None:
            return account_error
        if not self._ensure_account_scope(
            principal_id=principal_id,
            account_id=account_id,
            correlation_id=correlation_id,
        ):
            return ToolResult.error(
                code=ErrorCode.UNAUTHORIZED_SCOPE,
                message="principal_id has no access to account_id",
                details={"account_id": account_id},
                meta=ToolMeta(correlation_id=correlation_id),
            )
        strategy = self._memory_storage.get_strategy(principal_id)
        snapshot = self._memory_storage.get_latest_portfolio_snapshot(principal_id, account_id)

        if strategy is None:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="Structured strategy is not available",
                meta=ToolMeta(correlation_id=correlation_id),
            )
        if snapshot is None:
            return ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message="Portfolio snapshot is not available",
                meta=ToolMeta(correlation_id=correlation_id),
            )

        analytics_snapshot = build_analytics_snapshot(snapshot)
        assessment, diagnostics = evaluate_strategy_fit(
            analytics_snapshot=analytics_snapshot,
            strategy=strategy,
            min_coverage=self._runtime_config.strategy_fit_min_coverage,
        )

        return ToolResult.ok(
            data={
                "strategy_fit": assessment.model_dump(mode="json"),
                "coverage": analytics_snapshot.coverage,
                "diagnostics": diagnostics,
            },
            meta=ToolMeta(
                source_ts=snapshot.as_of_ts,
                coverage=analytics_snapshot.coverage,
                correlation_id=correlation_id,
            ),
        )

    def _tool_macro_indicators_show(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        del principal_id
        keys, validation_error = _extract_macro_keys(payload=payload, correlation_id=correlation_id)
        if validation_error is not None:
            return validation_error
        return self._market_data_adapter.show_macro_indicators(
            keys=keys,
            correlation_id=correlation_id,
        )

    def _tool_issuer_indicators_show(
        self,
        *,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, Any],
    ) -> ToolResult:
        del principal_id
        ticker, validation_error = _extract_ticker(payload=payload, correlation_id=correlation_id)
        if validation_error is not None:
            return validation_error
        return self._market_data_adapter.show_issuer_indicators(
            ticker=ticker,
            correlation_id=correlation_id,
        )

    def _ensure_account_scope(
        self,
        *,
        principal_id: str,
        account_id: str | None,
        correlation_id: str,
    ) -> bool:
        if self._user_storage.has_account_access(principal_id, account_id):
            return True
        if account_id is None:
            return True

        accounts_result = self._portfolio_adapter.list_accounts(correlation_id=correlation_id)
        if accounts_result.status == ToolStatus.ERROR:
            return False
        account_ids = _read_account_ids(accounts_result)
        if not account_ids or account_id not in account_ids:
            return False
        for item in account_ids:
            self._user_storage.grant_account_access(principal_id, item)
        return True


def _normalize_account_id(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError("account_id must be a string")
    candidate = raw_value.strip()
    return candidate or None


def _extract_account_id(
    payload: dict[str, Any],
    correlation_id: str,
) -> tuple[str | None, ToolResult | None]:
    try:
        return _normalize_account_id(payload.get("account_id")), None
    except ValueError:
        return None, ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="account_id must be a string",
            meta=ToolMeta(correlation_id=correlation_id),
        )


def _extract_ticker(
    *,
    payload: dict[str, Any],
    correlation_id: str,
) -> tuple[str, ToolResult | None]:
    raw_ticker = payload.get("ticker")
    if not isinstance(raw_ticker, str):
        return "", ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="ticker must be a string",
            meta=ToolMeta(correlation_id=correlation_id),
        )
    ticker = raw_ticker.strip().upper()
    if not ticker:
        return "", ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="ticker must not be blank",
            meta=ToolMeta(correlation_id=correlation_id),
        )
    return ticker, None


def _extract_macro_keys(
    *,
    payload: dict[str, Any],
    correlation_id: str,
) -> tuple[list[str] | None, ToolResult | None]:
    raw_keys = payload.get("keys")
    if raw_keys is None:
        return None, None
    if not isinstance(raw_keys, list):
        return None, ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="keys must be a list",
            meta=ToolMeta(correlation_id=correlation_id),
        )
    values: list[str] = []
    for item in raw_keys:
        if not isinstance(item, str):
            return None, ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="keys must contain strings only",
                meta=ToolMeta(correlation_id=correlation_id),
            )
        values.append(item)
    return values, None


def _read_account_ids(tool_result: ToolResult) -> list[str]:
    raw_value = tool_result.data.get("account_ids")
    if not isinstance(raw_value, list):
        return []

    values: list[str] = []
    for item in raw_value:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values
