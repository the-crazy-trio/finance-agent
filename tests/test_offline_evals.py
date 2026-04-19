from __future__ import annotations

import unittest

from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorResponse
from finance_agent.evals.offline import OfflineEvalCase, run_offline_eval


class _FakeService:
    def __init__(self, responses: dict[str, OrchestratorResponse]) -> None:
        self._responses = responses

    def handle(self, request, auth_metadata=None):  # noqa: ANN001
        return self._responses[request.message_text]


def build_response(
    *,
    request_class: RequestClass,
    status: ResponseStatus,
    summary: str,
    unsupported_criteria: list[str] | None = None,
) -> OrchestratorResponse:
    return OrchestratorResponse(
        request_id="req-1",
        correlation_id="corr-1",
        status=status,
        summary=summary,
        details=["workflow=test"],
        evidence=OrchestratorEvidence(
            coverage=1.0,
            unsupported_criteria=unsupported_criteria or [],
        ),
        request_class=request_class,
    )


class OfflineEvalsTests(unittest.TestCase):
    def test_offline_eval_computes_pass_rate_and_quality_metrics(self) -> None:
        service = _FakeService(
            responses={
                "case-1": build_response(
                    request_class=RequestClass.STRATEGY_PARSING,
                    status=ResponseStatus.PARTIAL,
                    summary="Уточните риск-профиль",
                    unsupported_criteria=["risk_tolerance_self_assessed"],
                ),
                "case-2": build_response(
                    request_class=RequestClass.PORTFOLIO_ANALYSIS,
                    status=ResponseStatus.OK,
                    summary="Портфель загружен",
                ),
            }
        )
        report = run_offline_eval(
            service=service,
            cases=[
                OfflineEvalCase(
                    case_id="c1",
                    message_text="case-1",
                    expected_request_class=RequestClass.STRATEGY_PARSING,
                    expected_statuses=(ResponseStatus.PARTIAL,),
                ),
                OfflineEvalCase(
                    case_id="c2",
                    message_text="case-2",
                    expected_request_class=RequestClass.PORTFOLIO_ANALYSIS,
                    expected_statuses=(ResponseStatus.OK,),
                ),
            ],
        )

        self.assertEqual(report.total_cases, 2)
        self.assertEqual(report.passed_cases, 2)
        self.assertAlmostEqual(report.pass_rate, 1.0)
        self.assertAlmostEqual(report.partial_response_rate, 0.5)
        self.assertAlmostEqual(report.unsupported_disclosure_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
