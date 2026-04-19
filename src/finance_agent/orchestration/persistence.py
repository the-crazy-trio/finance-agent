from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from finance_agent.contracts.common import CorrelationContext, RequestClass
from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.retriever.storage import InMemoryMemoryStorage
from finance_agent.workflow.engine import WorkflowExecutionResult


class OrchestratorPersistenceStage:
    def __init__(
        self,
        *,
        memory_storage: InMemoryMemoryStorage | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._memory_storage = memory_storage
        self._now_fn = now_fn or (lambda: datetime.now(UTC))

    def persist(
        self,
        *,
        request_class: RequestClass,
        request: OrchestratorRequest,
        context: CorrelationContext,
        workflow_result: WorkflowExecutionResult,
    ) -> WorkflowExecutionResult:
        details = list(workflow_result.details)

        if self._memory_storage is None:
            details.append("persist=noop")
            return WorkflowExecutionResult(
                status=workflow_result.status,
                summary=workflow_result.summary,
                details=details,
                evidence=workflow_result.evidence,
                errors=list(workflow_result.errors),
                tool_latency_ms=workflow_result.tool_latency_ms,
                llm_tokens_in=workflow_result.llm_tokens_in,
                llm_tokens_out=workflow_result.llm_tokens_out,
                retry_count=workflow_result.retry_count,
            )

        summary_record: dict[str, Any] = {
            "request_class": request_class.value,
            "request_id": context.request_id,
            "correlation_id": context.correlation_id,
            "session_id": request.session_id,
            "status": workflow_result.status.value,
            "summary": workflow_result.summary,
            "coverage": workflow_result.evidence.coverage,
            "saved_at": self._now_fn().isoformat(),
        }
        self._memory_storage.save_workflow_summary(context.principal_id, summary_record)
        details.append("persist=memory_saved")

        return WorkflowExecutionResult(
            status=workflow_result.status,
            summary=workflow_result.summary,
            details=details,
            evidence=workflow_result.evidence,
            errors=list(workflow_result.errors),
            tool_latency_ms=workflow_result.tool_latency_ms,
            llm_tokens_in=workflow_result.llm_tokens_in,
            llm_tokens_out=workflow_result.llm_tokens_out,
            retry_count=workflow_result.retry_count,
        )
