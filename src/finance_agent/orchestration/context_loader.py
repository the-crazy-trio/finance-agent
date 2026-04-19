from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from finance_agent.contracts.common import CorrelationContext, RequestClass
from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.retriever.storage import InMemoryMemoryStorage


@dataclass(frozen=True, slots=True)
class LoadedRequestContext:
    strategy_available: bool = False
    snapshot_available: bool = False
    snapshot_account_id: str | None = None
    snapshot_as_of_ts: datetime | None = None
    details: list[str] = field(default_factory=list)


class OrchestratorContextLoader:
    def __init__(self, *, memory_storage: InMemoryMemoryStorage | None = None) -> None:
        self._memory_storage = memory_storage

    def load(
        self,
        *,
        request_class: RequestClass,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> LoadedRequestContext:
        if self._memory_storage is None:
            return LoadedRequestContext(details=["context_load=noop"])

        strategy = self._memory_storage.get_strategy(context.principal_id)
        snapshot = self._memory_storage.get_latest_portfolio_snapshot(
            context.principal_id,
            request.account_id,
        )

        details = [
            "context_load=memory",
            f"context.request_class={request_class.value}",
            "context.strategy=available" if strategy is not None else "context.strategy=missing",
            "context.snapshot=available" if snapshot is not None else "context.snapshot=missing",
        ]

        return LoadedRequestContext(
            strategy_available=strategy is not None,
            snapshot_available=snapshot is not None,
            snapshot_account_id=snapshot.account_id if snapshot is not None else None,
            snapshot_as_of_ts=snapshot.as_of_ts if snapshot is not None else None,
            details=details,
        )
