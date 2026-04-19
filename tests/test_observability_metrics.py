from __future__ import annotations

import unittest

from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.observability.metrics import (
    InMemoryOrchestratorMetricsSink,
    RequestMetricRecord,
)


class ObservabilityMetricsTests(unittest.TestCase):
    def test_metrics_snapshot_calculates_latency_and_rates(self) -> None:
        sink = InMemoryOrchestratorMetricsSink()
        sink.record_request(
            RequestMetricRecord(
                request_class=RequestClass.PORTFOLIO_ANALYSIS,
                status=ResponseStatus.OK,
                request_latency_ms=100,
            )
        )
        sink.record_request(
            RequestMetricRecord(
                request_class=RequestClass.STRATEGY_FIT,
                status=ResponseStatus.ERROR,
                request_latency_ms=300,
            )
        )
        sink.record_request(
            RequestMetricRecord(
                request_class=RequestClass.STRATEGY_PARSING,
                status=ResponseStatus.PARTIAL,
                request_latency_ms=200,
            )
        )

        snapshot = sink.snapshot()

        self.assertEqual(snapshot.total_requests, 3)
        self.assertAlmostEqual(snapshot.request_latency_p50_ms, 200.0)
        self.assertAlmostEqual(snapshot.request_latency_p95_ms, 300.0)
        self.assertAlmostEqual(snapshot.error_rate, 1 / 3)
        self.assertAlmostEqual(snapshot.partial_response_rate, 1 / 3)

    def test_snapshot_is_zeroed_when_no_records(self) -> None:
        snapshot = InMemoryOrchestratorMetricsSink().snapshot()

        self.assertEqual(snapshot.total_requests, 0)
        self.assertEqual(snapshot.request_latency_p50_ms, 0.0)
        self.assertEqual(snapshot.request_latency_p95_ms, 0.0)
        self.assertEqual(snapshot.error_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
