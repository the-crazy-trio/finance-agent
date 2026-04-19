from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from os import environ
from typing import Any

import mcp.types as mcp_types
from mcp.server import InitializationOptions, NotificationOptions, Server
from mcp.server.stdio import stdio_server

from finance_agent.contracts.common import ErrorCode
from finance_agent.contracts.tooling import ToolResult
from finance_agent.harness.skill_runner import SkillRunner, build_local_skill_runner
from finance_agent.retriever.tinvest_adapter import TInvestReadonlyClient
from finance_agent.retriever.tinvest_client_factory import build_tinvest_readonly_client_from_env


class FinanceMcpServer:
    def __init__(
        self,
        *,
        runner: SkillRunner,
        server_name: str = "finance-agent",
        server_version: str = "0.1.0",
    ) -> None:
        self._runner = runner
        self._server_name = server_name
        self._server_version = server_version
        self._server = Server(self._server_name)
        self._register_handlers()

    def initialization_options(self) -> InitializationOptions:
        capabilities = self._server.get_capabilities(
            notification_options=NotificationOptions(tools_changed=False),
            experimental_capabilities={},
        )
        return InitializationOptions(
            server_name=self._server_name,
            server_version=self._server_version,
            capabilities=capabilities,
        )

    def list_tools(self) -> list[mcp_types.Tool]:
        tools: list[mcp_types.Tool] = []
        for tool in self._runner.list_tools():
            tools.append(
                mcp_types.Tool(
                    name=tool["name"],
                    description=tool["description"],
                    inputSchema=tool["input_schema"],
                )
            )
        return tools

    def call_tool(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any] | None,
        meta: Mapping[str, Any] | None,
    ) -> mcp_types.CallToolResult:
        normalized_name = name.strip()
        if not normalized_name:
            payload = ToolResult.error(
                code=ErrorCode.VALIDATION_ERROR,
                message="tools/call requires non-empty 'name'",
            ).model_dump(mode="json")
            return _build_call_tool_result(payload)

        correlation_id = _extract_correlation_id(meta)

        try:
            result = self._runner.run_tool(
                tool_name=normalized_name,
                arguments=arguments,
                correlation_id=correlation_id,
            )
        except Exception as error:  # pragma: no cover - guard for runtime failures
            result = ToolResult.error(
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                message=f"tool execution failed: {error}",
            ).model_dump(mode="json")

        return _build_call_tool_result(result)

    async def serve_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self.initialization_options(),
            )

    def _register_handlers(self) -> None:
        @self._server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:
            return self.list_tools()

        @self._server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
            request_meta = _to_mapping_or_none(self._server.request_context.meta)
            return self.call_tool(name=name, arguments=arguments, meta=request_meta)


def build_default_mcp_server(
    *,
    env: Mapping[str, str] | None = None,
) -> FinanceMcpServer:
    source = env if env is not None else environ
    runtime_env = _with_local_runtime_defaults(source)
    allowed_accounts = _parse_allowed_accounts(source.get("LOCAL_ALLOWED_ACCOUNT_IDS"))
    tinvest_client = _build_tinvest_client(source)
    runner = build_local_skill_runner(
        env=runtime_env,
        allowed_account_ids=allowed_accounts,
        tinvest_client=tinvest_client,
        use_env_tinvest_client=False,
    )
    return FinanceMcpServer(runner=runner)


async def serve_stdio(server: FinanceMcpServer) -> None:
    await server.serve_stdio()


def _extract_correlation_id(meta: Mapping[str, Any] | None) -> str | None:
    if meta is not None:
        value = meta.get("correlation_id")
        if value is not None:
            candidate = str(value).strip()
            if candidate:
                return candidate
    return None


def _build_call_tool_result(result: Mapping[str, Any]) -> mcp_types.CallToolResult:
    content_text = json.dumps(result, ensure_ascii=False)
    return mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(
                type="text",
                text=content_text,
            )
        ],
        structuredContent=dict(result),
        isError=result.get("status") == "error",
    )


def _to_mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python", exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _parse_allowed_accounts(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    values = [item.strip() for item in raw_value.split(",")]
    return [item for item in values if item]


def _build_tinvest_client(source: Mapping[str, str]) -> TInvestReadonlyClient | None:
    return build_tinvest_readonly_client_from_env(source)


def _with_local_runtime_defaults(source: Mapping[str, str]) -> dict[str, str]:
    payload = dict(source)
    if not payload.get("OPENROUTER_API_KEY", "").strip():
        payload["OPENROUTER_API_KEY"] = "local-mcp-placeholder-openrouter"
    if not payload.get("TINVEST_READONLY_TOKEN", "").strip():
        payload["TINVEST_READONLY_TOKEN"] = "local-mcp-placeholder-tinvest"
    return payload


def main() -> None:
    server = build_default_mcp_server()
    asyncio.run(serve_stdio(server))


if __name__ == "__main__":
    main()
