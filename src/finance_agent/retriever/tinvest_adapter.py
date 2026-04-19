from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.entities import FreshnessMetadata, PortfolioPosition, PortfolioSnapshot
from finance_agent.contracts.tooling import ToolMeta, ToolResult
from finance_agent.retriever.errors import (
    TInvestClientError,
    TInvestErrorCategory,
    normalize_tinvest_error,
)
from finance_agent.retriever.storage import InMemoryMemoryStorage


class TInvestReadonlyClient(Protocol):
    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        pass

    def list_account_ids(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> list[str]:
        pass


class TInvestReadonlyAdapter:
    def __init__(
        self,
        *,
        client: TInvestReadonlyClient,
        runtime_config: RuntimeConfig,
        memory_storage: InMemoryMemoryStorage,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._runtime_config = runtime_config
        self._memory_storage = memory_storage
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def collect_portfolio(
        self,
        *,
        principal_id: str,
        account_id: str | None,
        correlation_id: str,
    ) -> ToolResult:
        start_ts = self._now_fn()
        try:
            raw_payload = self._client.get_portfolio(
                account_id=account_id,
                correlation_id=correlation_id,
                timeout_ms=self._runtime_config.tool_timeout_ms,
            )
            snapshot = self._build_snapshot(
                principal_id=principal_id,
                account_id=account_id,
                payload=raw_payload,
            )
            self._memory_storage.save_portfolio_snapshot(snapshot)
            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.ok(
                data={"snapshot": snapshot.model_dump(mode="json")},
                meta=ToolMeta(
                    latency_ms=latency_ms,
                    source_ts=snapshot.as_of_ts,
                    coverage=snapshot.coverage,
                    correlation_id=correlation_id,
                ),
            )
        except TInvestClientError as error:
            normalized_error = normalize_tinvest_error(error.category, str(error))
            fallback_snapshot = self._memory_storage.get_latest_portfolio_snapshot(
                principal_id,
                account_id,
            )
            if fallback_snapshot is not None and self.is_snapshot_fresh(fallback_snapshot):
                latency_ms = _latency_ms(start_ts, self._now_fn())
                return ToolResult.partial(
                    data={"snapshot": fallback_snapshot.model_dump(mode="json")},
                    errors=[normalized_error],
                    meta=ToolMeta(
                        latency_ms=latency_ms,
                        source_ts=fallback_snapshot.as_of_ts,
                        coverage=fallback_snapshot.coverage,
                        correlation_id=correlation_id,
                    ),
                )

            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.error(
                code=normalized_error.code,
                message=normalized_error.message,
                retriable=normalized_error.retriable,
                details=normalized_error.details,
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )

    def list_accounts(self, *, correlation_id: str) -> ToolResult:
        start_ts = self._now_fn()
        list_method = getattr(self._client, "list_account_ids", None)
        if not callable(list_method):
            normalized_error = normalize_tinvest_error(
                TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
                "T-Invest account listing is not configured",
            )
            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.error(
                code=normalized_error.code,
                message=normalized_error.message,
                retriable=normalized_error.retriable,
                details=normalized_error.details,
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )

        try:
            account_ids = list_method(
                correlation_id=correlation_id,
                timeout_ms=self._runtime_config.tool_timeout_ms,
            )
            normalized_ids = _normalize_account_ids(account_ids)
            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.ok(
                data={"account_ids": normalized_ids},
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )
        except TInvestClientError as error:
            normalized_error = normalize_tinvest_error(error.category, str(error))
            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.error(
                code=normalized_error.code,
                message=normalized_error.message,
                retriable=normalized_error.retriable,
                details=normalized_error.details,
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )
        except ValueError as error:
            normalized_error = normalize_tinvest_error(
                TInvestErrorCategory.INTERNAL,
                str(error),
            )
            latency_ms = _latency_ms(start_ts, self._now_fn())
            return ToolResult.error(
                code=normalized_error.code,
                message="T-Invest accounts payload is invalid",
                retriable=normalized_error.retriable,
                details=normalized_error.details,
                meta=ToolMeta(latency_ms=latency_ms, correlation_id=correlation_id),
            )

    def is_snapshot_fresh(self, snapshot: PortfolioSnapshot) -> bool:
        age_minutes = _age_minutes(snapshot.as_of_ts, self._now_fn())
        return age_minutes <= float(self._runtime_config.portfolio_snapshot_max_age_min)

    def _build_snapshot(
        self,
        *,
        principal_id: str,
        account_id: str | None,
        payload: Mapping[str, Any],
    ) -> PortfolioSnapshot:
        as_of_ts = _parse_datetime(payload.get("as_of_ts"))
        effective_account_id = payload.get("account_id")
        if isinstance(effective_account_id, str):
            effective_account_id = effective_account_id.strip() or None
        else:
            effective_account_id = None
        if effective_account_id is None:
            effective_account_id = account_id

        positions_raw = payload.get("positions", [])
        if not isinstance(positions_raw, list):
            raise ValueError("positions payload must be a list")

        positions = [PortfolioPosition.model_validate(item) for item in positions_raw]
        portfolio_value = _to_float(
            payload.get("portfolio_value"), default=_portfolio_total(positions)
        )
        coverage = _to_float(payload.get("coverage"), default=1.0)
        unknown_share = _to_float(payload.get("unknown_share"), default=0.0)

        age_minutes = _age_minutes(as_of_ts, self._now_fn())
        max_age = self._runtime_config.portfolio_snapshot_max_age_min
        freshness = FreshnessMetadata(
            source_ts=as_of_ts,
            age_minutes=age_minutes,
            max_age_minutes=max_age,
            is_stale=age_minutes > float(max_age),
        )

        partial = bool(payload.get("partial", False)) or coverage < 1.0 or unknown_share > 0.0
        return PortfolioSnapshot(
            snapshot_id=f"psnap_{uuid4().hex}",
            user_id=principal_id,
            account_id=effective_account_id,
            as_of_ts=as_of_ts,
            positions=positions,
            portfolio_value=portfolio_value,
            freshness=freshness,
            coverage=coverage,
            unknown_share=unknown_share,
            partial=partial,
        )


def _parse_datetime(raw_value: Any) -> datetime:
    if isinstance(raw_value, datetime):
        return raw_value
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        if candidate:
            return datetime.fromisoformat(candidate)
    raise ValueError("as_of_ts is required and must be a valid datetime")


def _to_float(raw_value: Any, *, default: float) -> float:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        raise ValueError("boolean values are not valid numeric payloads")
    return float(raw_value)


def _portfolio_total(positions: list[PortfolioPosition]) -> float:
    return sum(position.market_value for position in positions)


def _age_minutes(source_ts: datetime, now_ts: datetime) -> float:
    if source_ts.tzinfo is None:
        source_ts = source_ts.replace(tzinfo=UTC)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=UTC)
    delta = now_ts - source_ts
    return max(delta.total_seconds() / 60.0, 0.0)


def _latency_ms(start_ts: datetime, end_ts: datetime) -> int:
    delta = end_ts - start_ts
    latency = int(delta.total_seconds() * 1000)
    return latency if latency >= 0 else 0


def _normalize_account_ids(raw_account_ids: Any) -> list[str]:
    if not isinstance(raw_account_ids, list):
        raise ValueError("account_ids payload must be a list")

    values: list[str] = []
    for item in raw_account_ids:
        if not isinstance(item, str):
            raise ValueError("account_id must be a string")
        normalized = item.strip()
        if normalized and normalized not in values:
            values.append(normalized)
    return values
