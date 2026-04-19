from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Protocol

from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import CorrelationContext, RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse
from finance_agent.llm import OpenRouterChatClient, is_placeholder_api_key
from finance_agent.observability.metrics import (
    NoopOrchestratorMetricsSink,
    OrchestratorMetricsSink,
    RequestMetricRecord,
)
from finance_agent.orchestration.context import build_correlation_context
from finance_agent.orchestration.context_loader import OrchestratorContextLoader
from finance_agent.orchestration.identity import IdentityResolver, LocalIdentityResolver
from finance_agent.orchestration.intent import (
    DeterministicIntentClassifier,
    IntentClassification,
    LlmIntentClassifier,
)
from finance_agent.orchestration.persistence import OrchestratorPersistenceStage
from finance_agent.orchestration.synthesis import OrchestratorResponseSynthesizer
from finance_agent.orchestration.telemetry import (
    NoopOrchestratorTelemetrySink,
    OrchestratorTelemetrySink,
)
from finance_agent.orchestration.validation import OrchestratorValidationGate
from finance_agent.workflow import WorkflowEngine, determine_request_class


class IntentClassifier(Protocol):
    def classify(self, *, message_text: str, correlation_id: str) -> IntentClassification:
        pass


class OrchestratorService:
    def __init__(
        self,
        identity_resolver: IdentityResolver | None = None,
        request_classifier: Callable[[str], RequestClass] | None = None,
        intent_classifier: IntentClassifier | None = None,
        workflow_engine: WorkflowEngine | None = None,
        validation_gate: OrchestratorValidationGate | None = None,
        response_synthesizer: OrchestratorResponseSynthesizer | None = None,
        persistence_stage: OrchestratorPersistenceStage | None = None,
        context_loader: OrchestratorContextLoader | None = None,
        telemetry_sink: OrchestratorTelemetrySink | None = None,
        metrics_sink: OrchestratorMetricsSink | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self._identity_resolver = identity_resolver or LocalIdentityResolver()
        deterministic_classifier = DeterministicIntentClassifier(
            request_classifier or determine_request_class
        )
        self._intent_classifier = intent_classifier or _build_default_intent_classifier(
            runtime_config=runtime_config,
            deterministic_classifier=deterministic_classifier,
        )
        self._workflow_engine = workflow_engine or WorkflowEngine()
        if validation_gate is not None:
            self._validation_gate = validation_gate
        elif runtime_config is not None:
            self._validation_gate = OrchestratorValidationGate(
                portfolio_snapshot_max_age_min=runtime_config.portfolio_snapshot_max_age_min,
                analytics_snapshot_max_age_min=runtime_config.analytics_snapshot_max_age_min,
            )
        else:
            self._validation_gate = OrchestratorValidationGate()
        self._response_synthesizer = response_synthesizer or OrchestratorResponseSynthesizer(
            runtime_config=runtime_config,
            model_routing=(
                ModelRoutingPolicy(runtime_config) if runtime_config is not None else None
            ),
        )
        self._persistence_stage = persistence_stage or OrchestratorPersistenceStage()
        self._context_loader = context_loader or OrchestratorContextLoader()
        self._telemetry_sink = telemetry_sink or NoopOrchestratorTelemetrySink()
        self._metrics_sink = metrics_sink or NoopOrchestratorMetricsSink()
        self._workflow_version = "1.0.0"
        self._schema_versions = {
            "structured_strategy": "1.0.0",
            "portfolio_snapshot": "1.0.0",
            "analytics_snapshot": "1.0.0",
        }

    def handle(
        self,
        request: OrchestratorRequest,
        auth_metadata: Mapping[str, str] | None = None,
    ) -> OrchestratorResponse:
        started = perf_counter()
        identity = self._identity_resolver.resolve(auth_metadata)
        context = build_correlation_context(
            request,
            principal_id=identity.principal_id,
            auth_mode=identity.auth_mode,
        )
        intent_result = self._intent_classifier.classify(
            message_text=request.message_text,
            correlation_id=context.correlation_id,
        )
        request_class = intent_result.request_class
        self._emit(
            "intake_started",
            context=context,
            request_class=request_class,
            channel=request.channel,
            has_account_id=request.account_id is not None,
            model_id=intent_result.model_id,
            prompt_version=intent_result.prompt_version,
            fallback_reason=intent_result.fallback_reason,
        )
        loaded_context = self._context_loader.load(
            request_class=request_class,
            request=request,
            context=context,
        )
        self._emit(
            "context_loaded",
            context=context,
            request_class=request_class,
            strategy_available=loaded_context.strategy_available,
            snapshot_available=loaded_context.snapshot_available,
            snapshot_account_id=loaded_context.snapshot_account_id,
        )

        workflow_result = self._workflow_engine.execute(
            request_class=request_class,
            request=request,
            context=context,
        )
        self._emit(
            "workflow_executed",
            context=context,
            request_class=request_class,
            status=workflow_result.status.value,
        )
        validated_result = self._validation_gate.validate(
            request_class=request_class,
            context=context,
            workflow_result=workflow_result,
        )
        self._emit(
            "validation_finished",
            context=context,
            request_class=request_class,
            status=validated_result.status.value,
        )
        synthesized_result = self._response_synthesizer.synthesize(
            request_class=request_class,
            request=request,
            context=context,
            workflow_result=validated_result,
        )
        self._emit(
            "synthesis_finished",
            context=context,
            request_class=request_class,
            status=synthesized_result.status.value,
        )
        persisted_result = self._persistence_stage.persist(
            request_class=request_class,
            request=request,
            context=context,
            workflow_result=synthesized_result,
        )
        self._emit(
            "persist_finished",
            context=context,
            request_class=request_class,
            status=persisted_result.status.value,
        )
        details = [
            f"channel={request.channel or 'unknown'}",
            f"principal_id={context.principal_id}",
            f"auth_mode={context.auth_mode.value}",
            f"intent.prompt_version={intent_result.prompt_version}",
            f"intent.model_id={intent_result.model_id or 'deterministic'}",
        ] + loaded_context.details + persisted_result.details
        if intent_result.fallback_reason is not None:
            details.append(f"intent.fallback_reason={intent_result.fallback_reason}")

        self._emit(
            "response_ready",
            context=context,
            request_class=request_class,
            status=persisted_result.status.value,
        )

        response = OrchestratorResponse(
            request_id=context.request_id,
            correlation_id=context.correlation_id,
            status=persisted_result.status,
            summary=persisted_result.summary,
            details=details,
            evidence=persisted_result.evidence,
            errors=persisted_result.errors,
            request_class=request_class,
        )
        self._record_metrics(
            request_class=request_class,
            status=response.status,
            started=started,
            tool_latency_ms=persisted_result.tool_latency_ms,
            llm_tokens_in=intent_result.llm_tokens_in + persisted_result.llm_tokens_in,
            llm_tokens_out=intent_result.llm_tokens_out + persisted_result.llm_tokens_out,
            retry_count=intent_result.retry_count + persisted_result.retry_count,
        )
        return response

    def _emit(
        self,
        event_name: str,
        *,
        context: CorrelationContext,
        request_class: RequestClass,
        **payload: object,
    ) -> None:
        event_payload = {
            "correlation_id": context.correlation_id,
            "request_id": context.request_id,
            "principal_id": context.principal_id,
            "request_class": request_class.value,
            "workflow_version": self._workflow_version,
            "schema_versions": dict(self._schema_versions),
            **payload,
        }
        self._telemetry_sink.emit(event_name, event_payload)

    def _record_metrics(
        self,
        *,
        request_class: RequestClass,
        status: ResponseStatus,
        started: float,
        tool_latency_ms: int,
        llm_tokens_in: int,
        llm_tokens_out: int,
        retry_count: int,
    ) -> None:
        latency_ms = int((perf_counter() - started) * 1000)
        self._metrics_sink.record_request(
            RequestMetricRecord(
                request_class=request_class,
                status=status,
                request_latency_ms=latency_ms if latency_ms >= 0 else 0,
                tool_latency_ms=tool_latency_ms if tool_latency_ms >= 0 else 0,
                llm_tokens_in=llm_tokens_in if llm_tokens_in >= 0 else 0,
                llm_tokens_out=llm_tokens_out if llm_tokens_out >= 0 else 0,
                retry_count=retry_count if retry_count >= 0 else 0,
            )
        )


def _build_default_intent_classifier(
    *,
    runtime_config: RuntimeConfig | None,
    deterministic_classifier: DeterministicIntentClassifier,
) -> IntentClassifier:
    if runtime_config is None:
        return deterministic_classifier
    if is_placeholder_api_key(runtime_config.openrouter_api_key):
        return deterministic_classifier

    llm_client = OpenRouterChatClient(
        api_key=runtime_config.openrouter_api_key,
        max_retries=runtime_config.max_retries_transient,
    )
    return LlmIntentClassifier(
        llm_client=llm_client,
        model_routing=ModelRoutingPolicy(runtime_config),
        fallback_classifier=deterministic_classifier,
        tokens_max=runtime_config.tokens_intent_max,
        llm_timeout_ms=runtime_config.llm_timeout_ms,
    )
