from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from finance_agent.contracts.common import (
    CorrelationContext,
    ErrorCode,
    ErrorDetail,
    RequestClass,
    ResponseStatus,
)
from finance_agent.workflow.engine import WorkflowExecutionResult

_DATA_FRESHNESS_REQUEST_CLASSES = {
    RequestClass.PORTFOLIO_ANALYSIS,
    RequestClass.STRATEGY_FIT,
    RequestClass.ASSET_ANALYSIS,
    RequestClass.PORTFOLIO_QA,
}

_NON_EVIDENCE_PARTIAL_REASON_MARKERS = {
    "reason=account_selection_required",
    "reason=ticker_missing",
}


class OrchestratorValidationGate:
    def __init__(
        self,
        *,
        portfolio_snapshot_max_age_min: int = 60,
        analytics_snapshot_max_age_min: int = 120,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._portfolio_snapshot_max_age_min = portfolio_snapshot_max_age_min
        self._analytics_snapshot_max_age_min = analytics_snapshot_max_age_min
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def validate(
        self,
        *,
        request_class: RequestClass,
        context: CorrelationContext,
        workflow_result: WorkflowExecutionResult,
    ) -> WorkflowExecutionResult:
        unauthorized_scope = self._find_unauthorized_scope(workflow_result)
        if unauthorized_scope is not None:
            return WorkflowExecutionResult(
                status=ResponseStatus.ERROR,
                summary="Доступ к данным ограничен для текущего principal scope.",
                details=[
                    f"workflow={request_class.value}",
                    "reason=principal_scope_violation",
                    f"principal_id={context.principal_id}",
                ],
                evidence=workflow_result.evidence,
                errors=list(workflow_result.errors),
                tool_latency_ms=workflow_result.tool_latency_ms,
                llm_tokens_in=workflow_result.llm_tokens_in,
                llm_tokens_out=workflow_result.llm_tokens_out,
                retry_count=workflow_result.retry_count,
            )

        if workflow_result.status == ResponseStatus.OK and workflow_result.errors:
            return self._validation_error(
                workflow_result=workflow_result,
                request_class=request_class,
                reason="ok_status_with_errors",
                message="Workflow returned status=ok with non-empty errors",
            )

        if not self._requires_freshness_check(
            request_class,
            workflow_result.status,
            workflow_result,
        ):
            return workflow_result

        if workflow_result.evidence.as_of_ts is None:
            return self._validation_error(
                workflow_result=workflow_result,
                request_class=request_class,
                reason="missing_evidence_as_of_ts",
                message="Missing as_of_ts evidence for freshness-aware workflow",
            )

        max_age_minutes = self._max_age_minutes_for(request_class)
        age_minutes = _age_minutes(workflow_result.evidence.as_of_ts, self._now_fn())
        if age_minutes > float(max_age_minutes):
            return WorkflowExecutionResult(
                status=ResponseStatus.UNAVAILABLE,
                summary="Данные устарели и не проходят freshness policy.",
                details=[
                    f"workflow={request_class.value}",
                    "reason=stale_evidence",
                    f"age_minutes={age_minutes:.1f}",
                    f"max_age_minutes={max_age_minutes}",
                ],
                evidence=workflow_result.evidence,
                errors=[
                    ErrorDetail(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="Evidence freshness exceeds configured max-age",
                        retriable=True,
                        details={
                            "age_minutes": age_minutes,
                            "max_age_minutes": max_age_minutes,
                        },
                    )
                ],
                tool_latency_ms=workflow_result.tool_latency_ms,
                llm_tokens_in=workflow_result.llm_tokens_in,
                llm_tokens_out=workflow_result.llm_tokens_out,
                retry_count=workflow_result.retry_count,
            )

        return workflow_result

    def _requires_freshness_check(
        self,
        request_class: RequestClass,
        status: ResponseStatus,
        workflow_result: WorkflowExecutionResult,
    ) -> bool:
        if request_class not in _DATA_FRESHNESS_REQUEST_CLASSES:
            return False
        if _has_detail_marker(workflow_result, "mode=classification_only"):
            return False
        if _has_detail_marker(workflow_result, "reason=tool_gateway_not_configured"):
            return False
        if any(
            _has_detail_marker(workflow_result, marker)
            for marker in _NON_EVIDENCE_PARTIAL_REASON_MARKERS
        ):
            return False
        return status in {ResponseStatus.OK, ResponseStatus.PARTIAL}

    def _max_age_minutes_for(self, request_class: RequestClass) -> int:
        if request_class == RequestClass.STRATEGY_FIT:
            return self._analytics_snapshot_max_age_min
        return self._portfolio_snapshot_max_age_min

    def _find_unauthorized_scope(
        self,
        workflow_result: WorkflowExecutionResult,
    ) -> ErrorDetail | None:
        for error in workflow_result.errors:
            if error.code == ErrorCode.UNAUTHORIZED_SCOPE:
                return error
        return None

    def _validation_error(
        self,
        *,
        workflow_result: WorkflowExecutionResult,
        request_class: RequestClass,
        reason: str,
        message: str,
    ) -> WorkflowExecutionResult:
        return WorkflowExecutionResult(
            status=ResponseStatus.ERROR,
            summary="Результат workflow не прошел validation gate.",
            details=[
                f"workflow={request_class.value}",
                f"reason={reason}",
            ],
            evidence=workflow_result.evidence,
            errors=[
                ErrorDetail(
                    code=ErrorCode.VALIDATION_ERROR,
                    message=message,
                    retriable=False,
                )
            ],
            tool_latency_ms=workflow_result.tool_latency_ms,
            llm_tokens_in=workflow_result.llm_tokens_in,
            llm_tokens_out=workflow_result.llm_tokens_out,
            retry_count=workflow_result.retry_count,
        )


def _age_minutes(source_ts: datetime, now_ts: datetime) -> float:
    if source_ts.tzinfo is None:
        source_ts = source_ts.replace(tzinfo=UTC)
    if now_ts.tzinfo is None:
        now_ts = now_ts.replace(tzinfo=UTC)
    delta = now_ts - source_ts
    return max(delta.total_seconds() / 60.0, 0.0)


def _has_detail_marker(workflow_result: WorkflowExecutionResult, marker: str) -> bool:
    return marker in workflow_result.details
