from __future__ import annotations

import json
from dataclasses import dataclass

from finance_agent.agentic import SubagentKind, build_default_prompt_store
from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass
from finance_agent.llm import (
    LlmClient,
    LlmProviderError,
    OpenRouterChatClient,
    is_placeholder_api_key,
)
from finance_agent.workflow.asset_analysis import extract_ticker
from finance_agent.workflow.portfolio_qa import PortfolioQaIntent, interpret_portfolio_question
from finance_agent.workflow.strategy_parsing import (
    StrategyParsingDraft,
    build_strategy_draft,
    parse_strategy_message,
)


@dataclass(frozen=True, slots=True)
class WorkflowAgentSignal:
    mode: str
    model_id: str | None
    prompt_id: str
    prompt_version: str
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    retry_count: int = 0
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SnapshotDirective:
    primary_tool: str
    prefer_aggregate: bool
    signal: WorkflowAgentSignal


@dataclass(frozen=True, slots=True)
class StrategyFitDirective:
    prefer_aggregate: bool
    signal: WorkflowAgentSignal


@dataclass(frozen=True, slots=True)
class PortfolioQaDirective:
    intent: PortfolioQaIntent
    primary_tool: str
    prefer_aggregate: bool
    signal: WorkflowAgentSignal


@dataclass(frozen=True, slots=True)
class AssetAnalysisDirective:
    ticker: str | None
    primary_tool: str
    macro_keys: tuple[str, ...]
    prefer_aggregate: bool
    signal: WorkflowAgentSignal


@dataclass(frozen=True, slots=True)
class StrategyParsingDirective:
    draft: StrategyParsingDraft
    signal: WorkflowAgentSignal


class DeterministicWorkflowAgent:
    def __init__(self) -> None:
        self._prompt_store = build_default_prompt_store()

    def snapshot_directive(
        self,
        *,
        request_class: RequestClass,
        message_text: str,
        correlation_id: str,
    ) -> SnapshotDirective:
        del correlation_id
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=request_class,
        )
        normalized = message_text.strip().lower()
        primary_tool = "portfolio_show"
        if any(marker in normalized for marker in ("обнов", "refresh", "актуал", "пересчитай")):
            primary_tool = "portfolio_collect"
        return SnapshotDirective(
            primary_tool=primary_tool,
            prefer_aggregate=_prefer_aggregate_mode(message_text),
            signal=WorkflowAgentSignal(
                mode="deterministic",
                model_id=None,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
            ),
        )

    def strategy_fit_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> StrategyFitDirective:
        del correlation_id
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=RequestClass.STRATEGY_FIT,
        )
        return StrategyFitDirective(
            prefer_aggregate=_prefer_aggregate_mode(message_text),
            signal=WorkflowAgentSignal(
                mode="deterministic",
                model_id=None,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
            ),
        )

    def portfolio_qa_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> PortfolioQaDirective:
        snapshot = self.snapshot_directive(
            request_class=RequestClass.PORTFOLIO_QA,
            message_text=message_text,
            correlation_id=correlation_id,
        )
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=RequestClass.PORTFOLIO_QA,
        )
        return PortfolioQaDirective(
            intent=interpret_portfolio_question(message_text),
            primary_tool=snapshot.primary_tool,
            prefer_aggregate=snapshot.prefer_aggregate,
            signal=WorkflowAgentSignal(
                mode="deterministic",
                model_id=None,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
            ),
        )

    def asset_analysis_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> AssetAnalysisDirective:
        snapshot = self.snapshot_directive(
            request_class=RequestClass.ASSET_ANALYSIS,
            message_text=message_text,
            correlation_id=correlation_id,
        )
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=RequestClass.ASSET_ANALYSIS,
        )
        return AssetAnalysisDirective(
            ticker=extract_ticker(message_text),
            primary_tool=snapshot.primary_tool,
            macro_keys=("key_rate", "inflation", "imoex", "usd_rub"),
            prefer_aggregate=snapshot.prefer_aggregate,
            signal=WorkflowAgentSignal(
                mode="deterministic",
                model_id=None,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
            ),
        )

    def strategy_parsing_directive(
        self,
        *,
        message_text: str,
        principal_id: str,
        correlation_id: str,
    ) -> StrategyParsingDirective:
        del correlation_id
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=RequestClass.STRATEGY_PARSING,
        )
        return StrategyParsingDirective(
            draft=parse_strategy_message(message_text=message_text, principal_id=principal_id),
            signal=WorkflowAgentSignal(
                mode="deterministic",
                model_id=None,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
            ),
        )


class LlmWorkflowAgent:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        model_routing: ModelRoutingPolicy,
        deterministic_fallback: DeterministicWorkflowAgent,
        tokens_max: int,
        llm_timeout_ms: int,
    ) -> None:
        self._llm_client = llm_client
        self._model_routing = model_routing
        self._fallback = deterministic_fallback
        self._tokens_max = tokens_max
        self._llm_timeout_ms = llm_timeout_ms
        self._prompt_store = build_default_prompt_store()

    def snapshot_directive(
        self,
        *,
        request_class: RequestClass,
        message_text: str,
        correlation_id: str,
    ) -> SnapshotDirective:
        fallback = self._fallback.snapshot_directive(
            request_class=request_class,
            message_text=message_text,
            correlation_id=correlation_id,
        )
        payload, signal = self._complete_json(
            request_class=request_class,
            correlation_id=correlation_id,
            user_prompt=(
                "Decide retrieval plan for this request. "
                "Return JSON object with fields primary_tool and prefer_aggregate. "
                "primary_tool must be portfolio_show or portfolio_collect.\n"
                f"message: {message_text}"
            ),
        )
        if payload is None:
            return SnapshotDirective(
                primary_tool=fallback.primary_tool,
                prefer_aggregate=fallback.prefer_aggregate,
                signal=self._with_fallback(signal=signal, fallback=fallback.signal),
            )

        primary_tool = _text(payload.get("primary_tool"))
        if primary_tool not in {"portfolio_show", "portfolio_collect"}:
            primary_tool = fallback.primary_tool
        prefer_aggregate = _read_bool(payload.get("prefer_aggregate"), fallback.prefer_aggregate)
        return SnapshotDirective(
            primary_tool=primary_tool,
            prefer_aggregate=prefer_aggregate,
            signal=signal,
        )

    def strategy_fit_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> StrategyFitDirective:
        fallback = self._fallback.strategy_fit_directive(
            message_text=message_text,
            correlation_id=correlation_id,
        )
        payload, signal = self._complete_json(
            request_class=RequestClass.STRATEGY_FIT,
            correlation_id=correlation_id,
            user_prompt=(
                "Decide account-scope preference for this strategy-fit request. "
                "Return JSON object with field prefer_aggregate (boolean).\n"
                f"message: {message_text}"
            ),
        )
        if payload is None:
            return StrategyFitDirective(
                prefer_aggregate=fallback.prefer_aggregate,
                signal=self._with_fallback(signal=signal, fallback=fallback.signal),
            )
        return StrategyFitDirective(
            prefer_aggregate=_read_bool(payload.get("prefer_aggregate"), fallback.prefer_aggregate),
            signal=signal,
        )

    def portfolio_qa_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> PortfolioQaDirective:
        fallback = self._fallback.portfolio_qa_directive(
            message_text=message_text,
            correlation_id=correlation_id,
        )
        payload, signal = self._complete_json(
            request_class=RequestClass.PORTFOLIO_QA,
            correlation_id=correlation_id,
            user_prompt=(
                "Classify portfolio question intent. Return JSON with fields kind, asset_class, "
                "primary_tool, prefer_aggregate. kind must be one of: "
                "position_count, class_share, unknown. "
                "primary_tool must be portfolio_show or portfolio_collect. "
                "asset_class allowed values: bond, equity, cash, or null.\n"
                f"message: {message_text}"
            ),
        )
        if payload is None:
            return PortfolioQaDirective(
                intent=fallback.intent,
                primary_tool=fallback.primary_tool,
                prefer_aggregate=fallback.prefer_aggregate,
                signal=self._with_fallback(signal=signal, fallback=fallback.signal),
            )

        kind = _text(payload.get("kind")) or fallback.intent.kind
        asset_class = _text(payload.get("asset_class"))
        primary_tool = _text(payload.get("primary_tool")) or fallback.primary_tool
        if primary_tool not in {"portfolio_show", "portfolio_collect"}:
            primary_tool = fallback.primary_tool
        if kind == "class_share" and asset_class not in {"bond", "equity", "cash"}:
            kind = "unknown"
            asset_class = None
        if kind not in {"position_count", "class_share", "unknown"}:
            kind = fallback.intent.kind
            asset_class = fallback.intent.asset_class
        return PortfolioQaDirective(
            intent=PortfolioQaIntent(kind=kind, asset_class=asset_class),
            primary_tool=primary_tool,
            prefer_aggregate=_read_bool(payload.get("prefer_aggregate"), fallback.prefer_aggregate),
            signal=signal,
        )

    def asset_analysis_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> AssetAnalysisDirective:
        fallback = self._fallback.asset_analysis_directive(
            message_text=message_text,
            correlation_id=correlation_id,
        )
        payload, signal = self._complete_json(
            request_class=RequestClass.ASSET_ANALYSIS,
            correlation_id=correlation_id,
            user_prompt=(
                "Extract ticker and account-scope preference from asset-analysis request. "
                "Return JSON with fields ticker, primary_tool, macro_keys, and "
                "prefer_aggregate. "
                "primary_tool must be portfolio_show or portfolio_collect. "
                "macro_keys allowed: key_rate, inflation, gdp, imoex, rts, usd_rub. "
                "ticker should be uppercase symbol or null.\n"
                f"message: {message_text}"
            ),
        )
        if payload is None:
            return AssetAnalysisDirective(
                ticker=fallback.ticker,
                primary_tool=fallback.primary_tool,
                macro_keys=fallback.macro_keys,
                prefer_aggregate=fallback.prefer_aggregate,
                signal=self._with_fallback(signal=signal, fallback=fallback.signal),
            )

        ticker = _text(payload.get("ticker"))
        if ticker is not None:
            ticker = ticker.upper()
        primary_tool = _text(payload.get("primary_tool")) or fallback.primary_tool
        if primary_tool not in {"portfolio_show", "portfolio_collect"}:
            primary_tool = fallback.primary_tool
        macro_keys = _parse_macro_keys(payload.get("macro_keys"), fallback.macro_keys)
        return AssetAnalysisDirective(
            ticker=ticker or fallback.ticker,
            primary_tool=primary_tool,
            macro_keys=macro_keys,
            prefer_aggregate=_read_bool(payload.get("prefer_aggregate"), fallback.prefer_aggregate),
            signal=signal,
        )

    def strategy_parsing_directive(
        self,
        *,
        message_text: str,
        principal_id: str,
        correlation_id: str,
    ) -> StrategyParsingDirective:
        fallback = self._fallback.strategy_parsing_directive(
            message_text=message_text,
            principal_id=principal_id,
            correlation_id=correlation_id,
        )
        payload, signal = self._complete_json(
            request_class=RequestClass.STRATEGY_PARSING,
            correlation_id=correlation_id,
            user_prompt=(
                "Extract strategy slots. Return JSON with fields risk_tolerance and "
                "target_asset_allocation. risk_tolerance values: conservative|balanced|growth|"
                "aggressive|null. target_asset_allocation keys: equity, bond, cash with "
                "weights 0..1.\n"
                f"message: {message_text}"
            ),
        )
        if payload is None:
            return StrategyParsingDirective(
                draft=fallback.draft,
                signal=self._with_fallback(signal=signal, fallback=fallback.signal),
            )

        risk_tolerance = _text(payload.get("risk_tolerance"))
        if risk_tolerance not in {"conservative", "balanced", "growth", "aggressive"}:
            risk_tolerance = fallback.draft.risk_tolerance
        allocation = _parse_allocation(payload.get("target_asset_allocation"))
        if not allocation:
            allocation = dict(fallback.draft.target_asset_allocation)

        draft = build_strategy_draft(
            principal_id=principal_id,
            risk_tolerance=risk_tolerance,
            target_asset_allocation=allocation,
        )
        return StrategyParsingDirective(draft=draft, signal=signal)

    def _complete_json(
        self,
        *,
        request_class: RequestClass,
        correlation_id: str,
        user_prompt: str,
    ) -> tuple[dict[str, object] | None, WorkflowAgentSignal]:
        template = self._prompt_store.resolve(
            kind=SubagentKind.WORKFLOW_AGENT,
            request_class=request_class,
        )
        model_id = self._model_routing.workflow_model(request_class)
        system_prompt = _compose_system_prompt(
            system_prompt=template.system_prompt,
            guidelines=template.guidelines,
        )
        try:
            completion = self._llm_client.complete(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self._tokens_max,
                timeout_ms=self._llm_timeout_ms,
                correlation_id=correlation_id,
                response_format={"type": "json_object"},
            )
            payload = json.loads(completion.text)
            if not isinstance(payload, dict):
                raise ValueError("workflow-agent payload must be object")
            return payload, WorkflowAgentSignal(
                mode="llm",
                model_id=completion.model_id,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
                llm_tokens_in=completion.usage.prompt_tokens,
                llm_tokens_out=completion.usage.completion_tokens,
                retry_count=completion.retries_used,
            )
        except (LlmProviderError, ValueError, json.JSONDecodeError) as error:
            return None, WorkflowAgentSignal(
                mode="llm_fallback",
                model_id=model_id,
                prompt_id=template.prompt_id,
                prompt_version=template.version,
                fallback_reason=str(error),
            )

    def _with_fallback(
        self,
        *,
        signal: WorkflowAgentSignal,
        fallback: WorkflowAgentSignal,
    ) -> WorkflowAgentSignal:
        return WorkflowAgentSignal(
            mode="deterministic_fallback",
            model_id=signal.model_id,
            prompt_id=signal.prompt_id,
            prompt_version=signal.prompt_version,
            llm_tokens_in=signal.llm_tokens_in,
            llm_tokens_out=signal.llm_tokens_out,
            retry_count=signal.retry_count,
            fallback_reason=signal.fallback_reason or fallback.fallback_reason,
        )


def build_workflow_agent(
    runtime_config: RuntimeConfig,
) -> DeterministicWorkflowAgent | LlmWorkflowAgent:
    deterministic = DeterministicWorkflowAgent()
    if is_placeholder_api_key(runtime_config.openrouter_api_key):
        return deterministic
    llm_client = OpenRouterChatClient(
        api_key=runtime_config.openrouter_api_key,
        max_retries=runtime_config.max_retries_transient,
    )
    return LlmWorkflowAgent(
        llm_client=llm_client,
        model_routing=ModelRoutingPolicy(runtime_config),
        deterministic_fallback=deterministic,
        tokens_max=runtime_config.tokens_intent_max,
        llm_timeout_ms=runtime_config.llm_timeout_ms,
    )


def _compose_system_prompt(*, system_prompt: str, guidelines: tuple[str, ...]) -> str:
    rules = "\n".join(f"- {item}" for item in guidelines)
    return f"{system_prompt}\nFollow these rules:\n{rules}\nReturn strict JSON only."


def _text(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    return normalized


def _read_bool(raw_value: object, default: bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return default


def _parse_allocation(raw_value: object) -> dict[str, float]:
    if not isinstance(raw_value, dict):
        return {}
    result: dict[str, float] = {}
    for key in ("equity", "bond", "cash"):
        value = raw_value.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 <= parsed <= 1.0:
            result[key] = parsed
    return result


def _parse_macro_keys(raw_value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw_value, list):
        return default
    allowed = {"key_rate", "inflation", "gdp", "imoex", "rts", "usd_rub"}
    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower()
        if normalized in allowed and normalized not in values:
            values.append(normalized)
    if not values:
        return default
    return tuple(values)


def _prefer_aggregate_mode(message_text: str) -> bool:
    normalized = message_text.strip().lower()
    aggregate_markers = (
        "по всем счет",
        "все счета",
        "all accounts",
        "aggregate",
    )
    if any(marker in normalized for marker in aggregate_markers):
        return True

    single_account_markers = (
        "по конкретному счет",
        "по конкретному счёт",
        "по счету",
        "по счёту",
        "на счете",
        "на счёте",
        "single account",
    )
    if any(marker in normalized for marker in single_account_markers):
        return False

    return True
