from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import environ
from typing import Any

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import ErrorCode
from finance_agent.contracts.tooling import ToolResult
from finance_agent.orchestration import (
    IdentityResolver,
    LocalIdentityResolver,
    resolve_correlation_id,
)
from finance_agent.retriever.errors import TInvestClientError, TInvestErrorCategory
from finance_agent.retriever.gateway import GuardedToolGateway
from finance_agent.retriever.storage import InMemoryMemoryStorage, InMemoryUserStorage
from finance_agent.retriever.tinvest_adapter import (
    TInvestReadonlyAdapter,
    TInvestReadonlyClient,
)
from finance_agent.retriever.tinvest_client_factory import build_tinvest_readonly_client_from_env

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "portfolio_accounts": (
        "Lists available account_ids; call this first before portfolio analysis tools"
    ),
    "portfolio_collect": (
        "Collects snapshots after account selection; omitting account_id uses aggregate"
    ),
    "portfolio_show": (
        "Shows latest snapshot after account selection; omitting account_id uses aggregate"
    ),
    "macro_indicators_show": (
        "Shows macro indicators from CBR/MOEX; accepts optional keys list"
    ),
    "issuer_indicators_show": "Shows MOEX issuer indicators for ticker",
    "strategy_save": "Saves normalized structured strategy payload",
    "strategy_fit": (
        "Builds strategy-fit after account selection; omitting account_id uses aggregate"
    ),
}

_TOOL_INPUT_SCHEMA: dict[str, dict[str, Any]] = {
    "portfolio_accounts": {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    },
    "portfolio_collect": {
        "type": "object",
        "properties": {
            "account_id": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "portfolio_show": {
        "type": "object",
        "properties": {
            "account_id": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "strategy_save": {
        "type": "object",
        "required": ["strategy"],
        "properties": {
            "strategy": {"type": "object"},
        },
        "additionalProperties": True,
    },
    "macro_indicators_show": {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": True,
    },
    "issuer_indicators_show": {
        "type": "object",
        "required": ["ticker"],
        "properties": {
            "ticker": {"type": "string"},
        },
        "additionalProperties": True,
    },
    "strategy_fit": {
        "type": "object",
        "properties": {
            "account_id": {"type": "string"},
        },
        "additionalProperties": True,
    },
}


class SkillRunner:
    def __init__(
        self,
        *,
        gateway: GuardedToolGateway,
        identity_resolver: IdentityResolver | None = None,
    ) -> None:
        self._gateway = gateway
        self._identity_resolver = identity_resolver or LocalIdentityResolver()

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool_name in self._gateway.list_tools():
            tools.append(
                {
                    "name": tool_name,
                    "description": _TOOL_DESCRIPTIONS[tool_name],
                    "input_schema": _TOOL_INPUT_SCHEMA[tool_name],
                }
            )
        return tools

    def run_tool(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        auth_metadata: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        identity = self._identity_resolver.resolve(auth_metadata)
        resolved_correlation_id = resolve_correlation_id(correlation_id)

        result = self._gateway.execute(
            tool_name=tool_name,
            principal_id=identity.principal_id,
            correlation_id=resolved_correlation_id,
            payload=dict(arguments or {}),
        )
        return result.model_dump(mode="json")


def build_local_skill_runner(
    *,
    env: Mapping[str, str] | None = None,
    tinvest_client: TInvestReadonlyClient | None = None,
    use_env_tinvest_client: bool = True,
    principal_id: str = "local_user",
    allowed_account_ids: Sequence[str] | None = None,
) -> SkillRunner:
    source = env if env is not None else environ
    runtime_config = RuntimeConfig.from_env(source)
    memory_storage = InMemoryMemoryStorage()
    user_storage = InMemoryUserStorage()

    resolved_tinvest_client = tinvest_client
    if resolved_tinvest_client is None and use_env_tinvest_client:
        resolved_tinvest_client = build_tinvest_readonly_client_from_env(source)

    adapter = TInvestReadonlyAdapter(
        client=resolved_tinvest_client or _UnconfiguredTInvestClient(),
        runtime_config=runtime_config,
        memory_storage=memory_storage,
    )
    gateway = GuardedToolGateway(
        runtime_config=runtime_config,
        user_storage=user_storage,
        memory_storage=memory_storage,
        portfolio_adapter=adapter,
    )
    for account_id in allowed_account_ids or []:
        user_storage.grant_account_access(principal_id, account_id)
    return SkillRunner(
        gateway=gateway,
        identity_resolver=LocalIdentityResolver(principal_id=principal_id),
    )


def run_harness_tool(
    payload: Mapping[str, Any],
    *,
    runner: SkillRunner,
) -> dict[str, Any]:
    try:
        tool_name = _read_required_string(payload, "tool_name")
    except ValueError as error:
        return ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message=str(error),
        ).model_dump(mode="json")

    raw_arguments = payload.get("arguments")
    if raw_arguments is None:
        arguments: Mapping[str, Any] | None = None
    elif isinstance(raw_arguments, Mapping):
        arguments = raw_arguments
    else:
        return ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="arguments must be an object",
        ).model_dump(mode="json")

    raw_auth = payload.get("auth_metadata")
    if raw_auth is None:
        auth_metadata: Mapping[str, str] | None = None
    elif isinstance(raw_auth, Mapping):
        auth_metadata = {str(key): str(value) for key, value in raw_auth.items()}
    else:
        return ToolResult.error(
            code=ErrorCode.VALIDATION_ERROR,
            message="auth_metadata must be an object",
        ).model_dump(mode="json")

    raw_correlation_id = payload.get("correlation_id")
    if raw_correlation_id is None:
        correlation_id = None
    else:
        correlation_id = str(raw_correlation_id)

    return runner.run_tool(
        tool_name=tool_name,
        arguments=arguments,
        correlation_id=correlation_id,
        auth_metadata=auth_metadata,
    )


def _read_required_string(payload: Mapping[str, Any], key: str) -> str:
    raw_value = payload.get(key)
    if raw_value is None:
        raise ValueError(f"{key} is required")
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"{key} must not be blank")
    return value


class _UnconfiguredTInvestClient:
    def get_portfolio(
        self,
        *,
        account_id: str | None,
        correlation_id: str,
        timeout_ms: int,
    ) -> dict[str, Any]:
        raise TInvestClientError(
            category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
            message="T-Invest client is not configured for skill runner",
        )

    def list_account_ids(
        self,
        *,
        correlation_id: str,
        timeout_ms: int,
    ) -> list[str]:
        raise TInvestClientError(
            category=TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
            message="T-Invest client is not configured for skill runner",
        )
