from __future__ import annotations

from collections.abc import Mapping
from os import environ
from typing import Any

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.orchestration.context_loader import OrchestratorContextLoader
from finance_agent.orchestration.identity import LocalIdentityResolver
from finance_agent.orchestration.persistence import OrchestratorPersistenceStage
from finance_agent.orchestration.service import OrchestratorService
from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory
from finance_agent.retriever.gateway import GuardedToolGateway
from finance_agent.retriever.storage import (
    FileBackedMemoryStorage,
    InMemoryMemoryStorage,
    InMemoryUserStorage,
)
from finance_agent.retriever.tinvest_adapter import TInvestReadonlyAdapter, TInvestReadonlyClient
from finance_agent.retriever.tinvest_client_factory import build_tinvest_readonly_client_from_env
from finance_agent.workflow import WorkflowEngine
from finance_agent.workflow.agentic import build_workflow_agent


def build_local_orchestrator_service(
    *,
    env: Mapping[str, str] | None = None,
    principal_id: str = "local_user",
) -> OrchestratorService:
    source = env if env is not None else environ
    runtime_config = RuntimeConfig.from_env(_with_local_runtime_defaults(source))
    memory_storage = _build_memory_storage(source)
    workflow_engine = _build_local_workflow_engine(
        env=source,
        runtime_config=runtime_config,
        memory_storage=memory_storage,
    )
    return OrchestratorService(
        identity_resolver=LocalIdentityResolver(principal_id=principal_id),
        workflow_engine=workflow_engine,
        context_loader=OrchestratorContextLoader(memory_storage=memory_storage),
        persistence_stage=OrchestratorPersistenceStage(memory_storage=memory_storage),
        runtime_config=runtime_config,
    )


def _with_local_runtime_defaults(source: Mapping[str, str]) -> dict[str, str]:
    payload = dict(source)
    if not payload.get("OPENROUTER_API_KEY", "").strip():
        payload["OPENROUTER_API_KEY"] = "local-orchestrator-placeholder-openrouter"
    if not payload.get("TINVEST_READONLY_TOKEN", "").strip():
        payload["TINVEST_READONLY_TOKEN"] = "local-orchestrator-placeholder-tinvest"
    return payload


def _build_memory_storage(source: Mapping[str, str]) -> InMemoryMemoryStorage:
    state_path = source.get("LOCAL_STATE_PATH", "").strip()
    if not state_path:
        return InMemoryMemoryStorage()
    return FileBackedMemoryStorage(state_path=state_path)


def _build_local_workflow_engine(
    *,
    env: Mapping[str, str],
    runtime_config: RuntimeConfig,
    memory_storage: InMemoryMemoryStorage,
) -> WorkflowEngine:
    workflow_agent = build_workflow_agent(runtime_config)
    if not _has_real_tinvest_token(env):
        return WorkflowEngine(workflow_agent=workflow_agent)

    user_storage = InMemoryUserStorage()
    tinvest_client = _build_tinvest_client(env)
    adapter = TInvestReadonlyAdapter(
        client=tinvest_client,
        runtime_config=runtime_config,
        memory_storage=memory_storage,
    )
    gateway = GuardedToolGateway(
        runtime_config=runtime_config,
        user_storage=user_storage,
        memory_storage=memory_storage,
        portfolio_adapter=adapter,
    )
    return WorkflowEngine(tool_gateway=gateway, workflow_agent=workflow_agent)


def _has_real_tinvest_token(env: Mapping[str, str]) -> bool:
    candidate = env.get("TINVEST_READONLY_TOKEN", "").strip().lower()
    if not candidate:
        return False
    return "placeholder" not in candidate and not candidate.startswith("local-")


def _build_tinvest_client(env: Mapping[str, str]) -> TInvestReadonlyClient:
    return build_tinvest_readonly_client_from_env(env) or _UnconfiguredTInvestClient()


class _UnconfiguredTInvestClient:
    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> dict[str, Any]:
        del account_id, correlation_id, timeout_ms
        raise TInvestClientError(
            category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
            message="T-Invest client is not configured for orchestrator",
        )

    def list_account_ids(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> list[str]:
        del correlation_id, timeout_ms
        raise TInvestClientError(
            category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
            message="T-Invest client is not configured for orchestrator",
        )
