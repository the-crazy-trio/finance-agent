from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse
from finance_agent.orchestration import OrchestratorService, build_local_orchestrator_service


def run_cli_request(
    *,
    message_text: str,
    session_id: str | None = None,
    account_id: str | None = None,
    correlation_id: str | None = None,
    local_state_path: str | None = None,
    service: OrchestratorService | None = None,
) -> dict[str, Any]:
    request = OrchestratorRequest(
        message=message_text,
        session_id=session_id,
        channel="cli",
        account_id=account_id,
        correlation_id=correlation_id,
    )
    if service is not None:
        orchestrator = service
    elif local_state_path is None:
        orchestrator = build_local_orchestrator_service()
    else:
        env = dict(os.environ)
        env["LOCAL_STATE_PATH"] = local_state_path
        orchestrator = build_local_orchestrator_service(env=env)
    response = orchestrator.handle(request)
    return OrchestratorResponse.model_validate(response).model_dump(mode="json")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local Financial Assistant request")
    parser.add_argument("message", help="User message text")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--correlation-id", default=None)
    parser.add_argument(
        "--state-path",
        default=".finance_agent_state.json",
        help="Path to local persisted state for CLI continuity",
    )
    args = parser.parse_args(argv)

    result = run_cli_request(
        message_text=args.message,
        session_id=args.session_id,
        account_id=args.account_id,
        correlation_id=args.correlation_id,
        local_state_path=args.state_path,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
