from typing import Any

import uvicorn
from fastapi import FastAPI, Header

from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse
from finance_agent.orchestration import build_local_orchestrator_service

app = FastAPI()
orchestrator_service = build_local_orchestrator_service()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/orchestrate", response_model=OrchestratorResponse)
async def orchestrate(
    request: OrchestratorRequest,
    x_principal_id: str | None = Header(default=None),
    x_correlation_id: str | None = Header(default=None),
) -> OrchestratorResponse:
    if x_correlation_id and request.correlation_id is None:
        request = request.model_copy(update={"correlation_id": x_correlation_id.strip()})

    auth_metadata: dict[str, Any] | None = None
    if x_principal_id:
        auth_metadata = {"x-principal-id": x_principal_id}
    return orchestrator_service.handle(request=request, auth_metadata=auth_metadata)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
