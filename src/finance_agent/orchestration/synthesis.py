from __future__ import annotations

import json

from finance_agent.agentic import SubagentKind, build_default_prompt_store
from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import CorrelationContext, RequestClass
from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.llm import (
    LlmClient,
    LlmProviderError,
    OpenRouterChatClient,
    is_placeholder_api_key,
)
from finance_agent.workflow.engine import WorkflowExecutionResult

_DISCLAIMER_TEXT = "Не является инвестиционной рекомендацией."


class OrchestratorResponseSynthesizer:
    def __init__(
        self,
        *,
        llm_client: LlmClient | None = None,
        model_routing: ModelRoutingPolicy | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self._prompt_store = build_default_prompt_store()
        self._runtime_config = runtime_config
        if llm_client is not None:
            self._llm_client = llm_client
            self._model_routing = model_routing
            return

        if runtime_config is None or is_placeholder_api_key(runtime_config.openrouter_api_key):
            self._llm_client = None
            self._model_routing = None
            return

        self._llm_client = OpenRouterChatClient(
            api_key=runtime_config.openrouter_api_key,
            max_retries=runtime_config.max_retries_transient,
        )
        self._model_routing = model_routing or ModelRoutingPolicy(runtime_config)

    def synthesize(
        self,
        *,
        request_class: RequestClass,
        request: OrchestratorRequest,
        context: CorrelationContext,
        workflow_result: WorkflowExecutionResult,
    ) -> WorkflowExecutionResult:
        prompt_template = self._prompt_store.resolve(
            kind=SubagentKind.RESPONSE_SYNTHESIS,
            request_class=request_class,
        )
        base_details = list(workflow_result.details)
        metadata_details = [
            f"request_class={request_class.value}",
            f"prompt_id={prompt_template.prompt_id}",
            f"prompt_version={prompt_template.version}",
            f"prompt_subagent_id={prompt_template.subagent_id}",
            "language=ru",
            f"disclaimer={_DISCLAIMER_TEXT}",
        ]

        if self._llm_client is not None and self._model_routing is not None:
            llm_result = self._synthesize_with_llm(
                request_class=request_class,
                request=request,
                context=context,
                workflow_result=workflow_result,
                prompt_id=prompt_template.prompt_id,
                prompt_version=prompt_template.version,
                system_prompt=_build_system_prompt(
                    prompt_template.system_prompt,
                    prompt_template.guidelines,
                ),
            )
            if llm_result is not None:
                summary, model_id, llm_tokens_in, llm_tokens_out, retries_used = llm_result
                details = base_details + [
                    "synthesis=llm",
                    f"synthesis.model_id={model_id}",
                ]
                return WorkflowExecutionResult(
                    status=workflow_result.status,
                    summary=summary,
                    details=details + metadata_details,
                    evidence=workflow_result.evidence,
                    errors=list(workflow_result.errors),
                    tool_latency_ms=workflow_result.tool_latency_ms,
                    llm_tokens_in=workflow_result.llm_tokens_in + llm_tokens_in,
                    llm_tokens_out=workflow_result.llm_tokens_out + llm_tokens_out,
                    retry_count=workflow_result.retry_count + retries_used,
                )

        return WorkflowExecutionResult(
            status=workflow_result.status,
            summary=workflow_result.summary,
            details=base_details + ["synthesis=deterministic_stub"] + metadata_details,
            evidence=workflow_result.evidence,
            errors=list(workflow_result.errors),
            tool_latency_ms=workflow_result.tool_latency_ms,
            llm_tokens_in=workflow_result.llm_tokens_in,
            llm_tokens_out=workflow_result.llm_tokens_out,
            retry_count=workflow_result.retry_count,
        )

    def _synthesize_with_llm(
        self,
        *,
        request_class: RequestClass,
        request: OrchestratorRequest,
        context: CorrelationContext,
        workflow_result: WorkflowExecutionResult,
        prompt_id: str,
        prompt_version: str,
        system_prompt: str,
    ) -> tuple[str, str, int, int, int] | None:
        if self._llm_client is None or self._model_routing is None:
            return None

        runtime = self._runtime_config
        max_tokens = runtime.tokens_synthesis_max if runtime is not None else 1_200
        timeout_ms = runtime.llm_timeout_ms if runtime is not None else 90_000
        model_id = self._model_routing.synthesis_model(request_class)

        try:
            completion = self._llm_client.complete(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=_build_user_prompt(
                    request=request,
                    context=context,
                    workflow_result=workflow_result,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                ),
                max_tokens=max_tokens,
                timeout_ms=timeout_ms,
                correlation_id=context.correlation_id,
            )
            summary = completion.text.strip()
            if not summary:
                return None
            return (
                summary,
                completion.model_id,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                completion.retries_used,
            )
        except (LlmProviderError, ValueError):
            return None


def _build_system_prompt(system_prompt: str, guidelines: tuple[str, ...]) -> str:
    joined_guidelines = "\n".join(f"- {item}" for item in guidelines)
    return (
        f"{system_prompt}\n"
        "Follow these rules:\n"
        f"{joined_guidelines}\n"
        "Keep response concise, grounded in provided evidence, and in Russian language."
    )


def _build_user_prompt(
    *,
    request: OrchestratorRequest,
    context: CorrelationContext,
    workflow_result: WorkflowExecutionResult,
    prompt_id: str,
    prompt_version: str,
) -> str:
    payload = {
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "request": {
            "request_id": context.request_id,
            "correlation_id": context.correlation_id,
            "message_text": request.message_text,
            "account_id": request.account_id,
            "channel": request.channel,
        },
        "validated_workflow_result": {
            "status": workflow_result.status.value,
            "summary": workflow_result.summary,
            "details": workflow_result.details,
            "evidence": workflow_result.evidence.model_dump(mode="json"),
            "errors": [item.model_dump(mode="json") for item in workflow_result.errors],
        },
    }
    return (
        "Compose final user-facing summary based only on validated_workflow_result. "
        "Do not invent numbers or claims. Mention unsupported criteria when present.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
