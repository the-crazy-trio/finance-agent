from __future__ import annotations

from uuid import uuid4

from finance_agent.contracts.common import AuthMode, CorrelationContext
from finance_agent.contracts.request_response import OrchestratorRequest


def generate_correlation_id() -> str:
    return f"corr_{uuid4().hex}"


def generate_request_id() -> str:
    return f"req_{uuid4().hex}"


def resolve_correlation_id(correlation_id: str | None) -> str:
    candidate = (correlation_id or "").strip()
    if candidate:
        return candidate
    return generate_correlation_id()


def resolve_request_id(request_id: str | None) -> str:
    candidate = (request_id or "").strip()
    if candidate:
        return candidate
    return generate_request_id()


def build_correlation_context(
    request: OrchestratorRequest,
    principal_id: str,
    auth_mode: AuthMode = AuthMode.LOCAL,
) -> CorrelationContext:
    return CorrelationContext(
        correlation_id=resolve_correlation_id(request.correlation_id),
        request_id=resolve_request_id(request.request_id),
        principal_id=principal_id,
        auth_mode=auth_mode,
        session_id=request.session_id,
    )
