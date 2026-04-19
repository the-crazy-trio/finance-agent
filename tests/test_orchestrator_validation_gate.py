from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from finance_agent.contracts.common import (
    AuthMode,
    CorrelationContext,
    ErrorCode,
    ErrorDetail,
    RequestClass,
    ResponseStatus,
)
from finance_agent.contracts.request_response import OrchestratorEvidence
from finance_agent.orchestration.validation import OrchestratorValidationGate
from finance_agent.workflow import WorkflowExecutionResult


def build_context() -> CorrelationContext:
    return CorrelationContext(
        correlation_id="corr-test",
        principal_id="local_user",
        auth_mode=AuthMode.LOCAL,
        request_id="req-test",
    )


class OrchestratorValidationGateTests(unittest.TestCase):
    def test_returns_unavailable_when_evidence_is_stale(self) -> None:
        now = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
        gate = OrchestratorValidationGate(
            portfolio_snapshot_max_age_min=60,
            now_fn=lambda: now,
        )
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="ok",
            evidence=OrchestratorEvidence(
                as_of_ts=now - timedelta(minutes=90),
                coverage=1.0,
                unsupported_criteria=[],
            ),
        )

        validated = gate.validate(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.UNAVAILABLE)
        self.assertEqual(validated.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)
        self.assertIn("reason=stale_evidence", validated.details)

    def test_returns_validation_error_when_data_workflow_missing_as_of(self) -> None:
        gate = OrchestratorValidationGate()
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="ok",
            evidence=OrchestratorEvidence(
                as_of_ts=None,
                coverage=1.0,
                unsupported_criteria=[],
            ),
        )

        validated = gate.validate(
            request_class=RequestClass.PORTFOLIO_QA,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.ERROR)
        self.assertEqual(validated.errors[0].code, ErrorCode.VALIDATION_ERROR)

    def test_returns_scope_error_when_unauthorized_scope_present(self) -> None:
        gate = OrchestratorValidationGate()
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="partial",
            errors=[
                ErrorDetail(
                    code=ErrorCode.UNAUTHORIZED_SCOPE,
                    message="principal has no access",
                )
            ],
        )

        validated = gate.validate(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.ERROR)
        self.assertEqual(validated.errors[0].code, ErrorCode.UNAUTHORIZED_SCOPE)
        self.assertIn("reason=principal_scope_violation", validated.details)

    def test_skips_freshness_for_non_data_workflow(self) -> None:
        gate = OrchestratorValidationGate()
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="needs clarification",
            evidence=OrchestratorEvidence(
                as_of_ts=None,
                coverage=1.0,
                unsupported_criteria=["risk_tolerance_self_assessed"],
            ),
        )

        validated = gate.validate(
            request_class=RequestClass.STRATEGY_PARSING,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.PARTIAL)
        self.assertEqual(validated.summary, "needs clarification")

    def test_skips_freshness_for_account_selection_partial(self) -> None:
        gate = OrchestratorValidationGate()
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="Найдено несколько счетов. Выберите account_id.",
            details=[
                "workflow=asset_analysis",
                "reason=account_selection_required",
                "account_options=acc-1,acc-2",
            ],
            evidence=OrchestratorEvidence(
                as_of_ts=None,
                coverage=1.0,
                unsupported_criteria=["account_id_missing"],
            ),
        )

        validated = gate.validate(
            request_class=RequestClass.ASSET_ANALYSIS,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.PARTIAL)
        self.assertEqual(validated.summary, "Найдено несколько счетов. Выберите account_id.")

    def test_skips_freshness_for_missing_ticker_partial(self) -> None:
        gate = OrchestratorValidationGate()
        workflow_result = WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="Для анализа актива нужен тикер.",
            details=[
                "workflow=asset_analysis",
                "reason=ticker_missing",
                "clarification=Укажите тикер.",
            ],
            evidence=OrchestratorEvidence(
                as_of_ts=None,
                coverage=1.0,
                unsupported_criteria=["asset_ticker_missing"],
            ),
        )

        validated = gate.validate(
            request_class=RequestClass.ASSET_ANALYSIS,
            context=build_context(),
            workflow_result=workflow_result,
        )

        self.assertEqual(validated.status, ResponseStatus.PARTIAL)
        self.assertEqual(validated.summary, "Для анализа актива нужен тикер.")


if __name__ == "__main__":
    unittest.main()
