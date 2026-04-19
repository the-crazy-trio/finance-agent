from __future__ import annotations

from typing import Any

from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse
from finance_agent.orchestration import build_local_orchestrator_service


def run_harness_request(payload: dict[str, Any]) -> dict[str, Any]:
    request = OrchestratorRequest.model_validate(payload)
    service = build_local_orchestrator_service()
    response = service.handle(request=request)
    return OrchestratorResponse.model_validate(response).model_dump(mode="json")
