from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Protocol

from finance_agent.contracts.common import RequestClass, ResponseStatus


@dataclass(frozen=True, slots=True)
class RequestMetricRecord:
    request_class: RequestClass
    status: ResponseStatus
    request_latency_ms: int
    tool_latency_ms: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    retry_count: int = 0


@dataclass(frozen=True, slots=True)
class RequestMetricsSnapshot:
    total_requests: int
    request_latency_p50_ms: float
    request_latency_p95_ms: float
    tool_latency_avg_ms: float
    llm_tokens_in_total: int
    llm_tokens_out_total: int
    retry_count_total: int
    error_rate: float
    partial_response_rate: float


class OrchestratorMetricsSink(Protocol):
    def record_request(self, metric: RequestMetricRecord) -> None:
        pass


class NoopOrchestratorMetricsSink:
    def record_request(self, metric: RequestMetricRecord) -> None:
        return


class InMemoryOrchestratorMetricsSink:
    def __init__(self) -> None:
        self._records: list[RequestMetricRecord] = []

    @property
    def records(self) -> list[RequestMetricRecord]:
        return list(self._records)

    def record_request(self, metric: RequestMetricRecord) -> None:
        self._records.append(metric)

    def snapshot(self) -> RequestMetricsSnapshot:
        if not self._records:
            return RequestMetricsSnapshot(
                total_requests=0,
                request_latency_p50_ms=0.0,
                request_latency_p95_ms=0.0,
                tool_latency_avg_ms=0.0,
                llm_tokens_in_total=0,
                llm_tokens_out_total=0,
                retry_count_total=0,
                error_rate=0.0,
                partial_response_rate=0.0,
            )

        total = len(self._records)
        request_latencies = [record.request_latency_ms for record in self._records]
        tool_latencies = [record.tool_latency_ms for record in self._records]

        errors = sum(1 for record in self._records if record.status == ResponseStatus.ERROR)
        partials = sum(1 for record in self._records if record.status == ResponseStatus.PARTIAL)
        return RequestMetricsSnapshot(
            total_requests=total,
            request_latency_p50_ms=_percentile(request_latencies, 50),
            request_latency_p95_ms=_percentile(request_latencies, 95),
            tool_latency_avg_ms=sum(tool_latencies) / total,
            llm_tokens_in_total=sum(record.llm_tokens_in for record in self._records),
            llm_tokens_out_total=sum(record.llm_tokens_out for record in self._records),
            retry_count_total=sum(record.retry_count for record in self._records),
            error_rate=errors / total,
            partial_response_rate=partials / total,
        )


def _percentile(values: Sequence[int], percentile: int) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return float(min(values))
    if percentile >= 100:
        return float(max(values))

    sorted_values = sorted(values)
    rank = ceil((percentile / 100) * len(sorted_values))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return float(sorted_values[index])
