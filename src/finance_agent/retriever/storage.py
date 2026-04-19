from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from finance_agent.contracts.entities import (
    FreshnessMetadata,
    PortfolioSnapshot,
    StructuredStrategy,
)


class InMemoryUserStorage:
    def __init__(self) -> None:
        self._account_scope: dict[str, set[str]] = {}

    def grant_account_access(self, principal_id: str, account_id: str) -> None:
        principal_key = principal_id.strip()
        account_key = account_id.strip()
        if not principal_key or not account_key:
            raise ValueError("principal_id and account_id must not be blank")

        if principal_key not in self._account_scope:
            self._account_scope[principal_key] = set()
        self._account_scope[principal_key].add(account_key)

    def has_account_access(self, principal_id: str, account_id: str | None) -> bool:
        if account_id is None:
            return True

        principal_key = principal_id.strip()
        account_key = account_id.strip()
        if not principal_key or not account_key:
            return False
        return account_key in self._account_scope.get(principal_key, set())

    def list_account_access(self, principal_id: str) -> tuple[str, ...]:
        principal_key = principal_id.strip()
        if not principal_key:
            return ()
        values = self._account_scope.get(principal_key, set())
        return tuple(sorted(values))


class InMemoryMemoryStorage:
    def __init__(self) -> None:
        self._portfolio_snapshots: dict[tuple[str, str | None], PortfolioSnapshot] = {}
        self._strategies: dict[str, StructuredStrategy] = {}
        self._workflow_summaries: dict[str, list[dict[str, Any]]] = {}

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        key = (snapshot.user_id, snapshot.account_id)
        self._portfolio_snapshots[key] = snapshot

    def get_latest_portfolio_snapshot(
        self,
        principal_id: str,
        account_id: str | None = None,
    ) -> PortfolioSnapshot | None:
        if account_id is not None:
            return self._portfolio_snapshots.get((principal_id, account_id))

        principal_snapshots = self._snapshots_for_principal(principal_id)
        if not principal_snapshots:
            return None
        if len(principal_snapshots) == 1:
            return principal_snapshots[0]
        return _merge_portfolio_snapshots(principal_id, principal_snapshots)

    def get_latest_portfolio_snapshot_for_accounts(
        self,
        principal_id: str,
        account_ids: Iterable[str],
    ) -> PortfolioSnapshot | None:
        snapshots: list[PortfolioSnapshot] = []
        for account_id in account_ids:
            account_key = account_id.strip()
            if not account_key:
                continue
            snapshot = self._portfolio_snapshots.get((principal_id, account_key))
            if snapshot is not None:
                snapshots.append(snapshot)

        if not snapshots:
            return None
        if len(snapshots) == 1:
            return snapshots[0]
        return _merge_portfolio_snapshots(principal_id, snapshots)

    def save_strategy(self, principal_id: str, strategy: StructuredStrategy) -> None:
        self._strategies[principal_id] = strategy

    def get_strategy(self, principal_id: str) -> StructuredStrategy | None:
        return self._strategies.get(principal_id)

    def save_workflow_summary(self, principal_id: str, summary: dict[str, Any]) -> None:
        principal_key = principal_id.strip()
        if not principal_key:
            raise ValueError("principal_id must not be blank")
        if principal_key not in self._workflow_summaries:
            self._workflow_summaries[principal_key] = []
        self._workflow_summaries[principal_key].append(dict(summary))

    def get_workflow_summaries(self, principal_id: str) -> list[dict[str, Any]]:
        principal_key = principal_id.strip()
        if not principal_key:
            return []
        values = self._workflow_summaries.get(principal_key, [])
        return [dict(item) for item in values]

    def _snapshots_for_principal(self, principal_id: str) -> list[PortfolioSnapshot]:
        values: Iterable[PortfolioSnapshot] = (
            snapshot
            for (owner_id, _), snapshot in self._portfolio_snapshots.items()
            if owner_id == principal_id
        )
        return list(values)


class InMemoryAnalyticsStorage:
    def __init__(self) -> None:
        self._instrument_mapping: dict[str, str] = {}

    def put_instrument_mapping(self, source_key: str, instrument_uid: str) -> None:
        key = source_key.strip()
        value = instrument_uid.strip()
        if not key or not value:
            raise ValueError("source_key and instrument_uid must not be blank")
        self._instrument_mapping[key] = value

    def get_instrument_uid(self, source_key: str) -> str | None:
        key = source_key.strip()
        if not key:
            return None
        return self._instrument_mapping.get(key)


class FileBackedMemoryStorage(InMemoryMemoryStorage):
    def __init__(self, *, state_path: str) -> None:
        super().__init__()
        normalized = state_path.strip()
        if not normalized:
            raise ValueError("state_path must not be blank")
        self._state_path = Path(normalized)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._reload()

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self._reload()
        super().save_portfolio_snapshot(snapshot)
        self._flush()

    def get_latest_portfolio_snapshot(
        self,
        principal_id: str,
        account_id: str | None = None,
    ) -> PortfolioSnapshot | None:
        self._reload()
        return super().get_latest_portfolio_snapshot(principal_id, account_id)

    def get_latest_portfolio_snapshot_for_accounts(
        self,
        principal_id: str,
        account_ids: Iterable[str],
    ) -> PortfolioSnapshot | None:
        self._reload()
        return super().get_latest_portfolio_snapshot_for_accounts(principal_id, account_ids)

    def save_strategy(self, principal_id: str, strategy: StructuredStrategy) -> None:
        self._reload()
        super().save_strategy(principal_id, strategy)
        self._flush()

    def get_strategy(self, principal_id: str) -> StructuredStrategy | None:
        self._reload()
        return super().get_strategy(principal_id)

    def save_workflow_summary(self, principal_id: str, summary: dict[str, Any]) -> None:
        self._reload()
        super().save_workflow_summary(principal_id, summary)
        self._flush()

    def get_workflow_summaries(self, principal_id: str) -> list[dict[str, Any]]:
        self._reload()
        return super().get_workflow_summaries(principal_id)

    def _reload(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return

        self._portfolio_snapshots = _decode_portfolio_snapshots(payload.get("portfolio_snapshots"))
        self._strategies = _decode_strategies(payload.get("strategies"))
        self._workflow_summaries = _decode_workflow_summaries(payload.get("workflow_summaries"))

    def _flush(self) -> None:
        payload = {
            "portfolio_snapshots": [
                snapshot.model_dump(mode="json")
                for snapshot in self._portfolio_snapshots.values()
            ],
            "strategies": {
                principal_id: strategy.model_dump(mode="json")
                for principal_id, strategy in self._strategies.items()
            },
            "workflow_summaries": {
                principal_id: list(summaries)
                for principal_id, summaries in self._workflow_summaries.items()
            },
        }
        temp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self._state_path)


def _merge_portfolio_snapshots(
    principal_id: str,
    snapshots: list[PortfolioSnapshot],
) -> PortfolioSnapshot:
    ordered = sorted(snapshots, key=lambda item: item.as_of_ts)
    oldest = ordered[0]
    positions = [position for snapshot in ordered for position in snapshot.positions]
    portfolio_value = sum(snapshot.portfolio_value for snapshot in ordered)
    coverage = _weighted_mean(
        values=[snapshot.coverage for snapshot in ordered],
        weights=[snapshot.portfolio_value for snapshot in ordered],
    )
    unknown_share = _weighted_mean(
        values=[snapshot.unknown_share for snapshot in ordered],
        weights=[snapshot.portfolio_value for snapshot in ordered],
    )
    freshness = FreshnessMetadata(
        source_ts=oldest.as_of_ts,
        age_minutes=max(
            (snapshot.freshness.age_minutes or 0.0)
            for snapshot in ordered
            if snapshot.freshness.age_minutes is not None
        )
        if any(snapshot.freshness.age_minutes is not None for snapshot in ordered)
        else None,
        max_age_minutes=min(
            snapshot.freshness.max_age_minutes
            for snapshot in ordered
            if snapshot.freshness.max_age_minutes is not None
        )
        if any(snapshot.freshness.max_age_minutes is not None for snapshot in ordered)
        else None,
        is_stale=any(snapshot.freshness.is_stale for snapshot in ordered),
    )
    partial = (
        any(snapshot.partial for snapshot in ordered)
        or coverage < 1.0
        or unknown_share > 0.0
    )
    return PortfolioSnapshot(
        snapshot_id=f"psnap_{uuid4().hex}",
        user_id=principal_id,
        account_id=None,
        as_of_ts=oldest.as_of_ts,
        positions=positions,
        portfolio_value=portfolio_value,
        freshness=freshness,
        coverage=coverage,
        unknown_share=unknown_share,
        partial=partial,
    )


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for weight in weights if weight > 0)
    if total_weight <= 0:
        return sum(values) / float(len(values))
    numerator = sum(value * weight for value, weight in zip(values, weights, strict=False))
    return numerator / total_weight


def _decode_portfolio_snapshots(raw_value: Any) -> dict[tuple[str, str | None], PortfolioSnapshot]:
    if not isinstance(raw_value, list):
        return {}
    items: dict[tuple[str, str | None], PortfolioSnapshot] = {}
    for row in raw_value:
        if not isinstance(row, dict):
            continue
        try:
            snapshot = PortfolioSnapshot.model_validate(row)
        except Exception:
            continue
        items[(snapshot.user_id, snapshot.account_id)] = snapshot
    return items


def _decode_strategies(raw_value: Any) -> dict[str, StructuredStrategy]:
    if not isinstance(raw_value, dict):
        return {}
    items: dict[str, StructuredStrategy] = {}
    for principal_id, payload in raw_value.items():
        if not isinstance(principal_id, str) or not isinstance(payload, dict):
            continue
        try:
            strategy = StructuredStrategy.model_validate(payload)
        except Exception:
            continue
        items[principal_id] = strategy
    return items


def _decode_workflow_summaries(raw_value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw_value, dict):
        return {}
    items: dict[str, list[dict[str, Any]]] = {}
    for principal_id, payload in raw_value.items():
        if not isinstance(principal_id, str) or not isinstance(payload, list):
            continue
        entries: list[dict[str, Any]] = []
        for row in payload:
            if isinstance(row, dict):
                entries.append(dict(row))
        items[principal_id] = entries
    return items
