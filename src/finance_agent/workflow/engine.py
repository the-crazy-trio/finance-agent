from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from finance_agent.contracts.common import (
    CorrelationContext,
    ErrorCode,
    ErrorDetail,
    RequestClass,
    ResponseStatus,
    ToolStatus,
)
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorRequest
from finance_agent.contracts.tooling import ToolResult
from finance_agent.workflow.agentic import (
    AssetAnalysisDirective,
    DeterministicWorkflowAgent,
    PortfolioQaDirective,
    SnapshotDirective,
    StrategyFitDirective,
    StrategyParsingDirective,
    WorkflowAgentSignal,
)
from finance_agent.workflow.asset_analysis import (
    AssetAnalysisResult,
    analyze_asset_in_snapshot,
)
from finance_agent.workflow.portfolio_qa import (
    PortfolioQaIntent,
    answer_portfolio_question,
)
from finance_agent.workflow.strategy_parsing import StrategyParsingDraft, parse_strategy_message

_TRANSIENT_ERROR_CODES = {
    ErrorCode.TIMEOUT,
    ErrorCode.UPSTREAM_UNAVAILABLE,
    ErrorCode.RATE_LIMITED,
}

_BOUNDED_PLANNER_VERSION = "bounded_v1"

_WORKFLOW_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "portfolio_analysis": frozenset({"portfolio_accounts", "portfolio_show", "portfolio_collect"}),
    "strategy_fit": frozenset(
        {
            "portfolio_accounts",
            "portfolio_collect",
            "strategy_save",
            "strategy_fit",
        }
    ),
    "asset_analysis": frozenset(
        {
            "portfolio_accounts",
            "portfolio_show",
            "portfolio_collect",
            "macro_indicators_show",
            "issuer_indicators_show",
        }
    ),
    "portfolio_qa": frozenset({"portfolio_accounts", "portfolio_show", "portfolio_collect"}),
}


class ToolGateway(Protocol):
    def execute(
        self,
        *,
        tool_name: str,
        principal_id: str,
        correlation_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        pass


class WorkflowAgent(Protocol):
    def snapshot_directive(
        self,
        *,
        request_class: RequestClass,
        message_text: str,
        correlation_id: str,
    ) -> SnapshotDirective:
        pass

    def strategy_fit_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> StrategyFitDirective:
        pass

    def portfolio_qa_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> PortfolioQaDirective:
        pass

    def asset_analysis_directive(
        self,
        *,
        message_text: str,
        correlation_id: str,
    ) -> AssetAnalysisDirective:
        pass

    def strategy_parsing_directive(
        self,
        *,
        message_text: str,
        principal_id: str,
        correlation_id: str,
    ) -> StrategyParsingDirective:
        pass


@dataclass(slots=True)
class WorkflowExecutionResult:
    status: ResponseStatus
    summary: str
    details: list[str] = field(default_factory=list)
    evidence: OrchestratorEvidence = field(default_factory=OrchestratorEvidence)
    errors: list[ErrorDetail] = field(default_factory=list)
    tool_latency_ms: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    retry_count: int = 0


@dataclass(slots=True)
class SnapshotPlanExecution:
    account_id: str | None
    snapshot_result: ToolResult | None = None
    used_collect: bool = False
    tool_latency_ms: int = 0
    plan_details: list[str] = field(default_factory=list)
    terminal_result: WorkflowExecutionResult | None = None


@dataclass(slots=True)
class StrategyFitPlanExecution:
    account_id: str | None
    fit_result: ToolResult | None = None
    tool_latency_ms: int = 0
    plan_details: list[str] = field(default_factory=list)
    terminal_result: WorkflowExecutionResult | None = None


@dataclass(slots=True)
class AssetSignalCollection:
    macro_indicators: dict[str, float] = field(default_factory=dict)
    issuer_indicators: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)
    unsupported_criteria: list[str] = field(default_factory=list)
    errors: list[ErrorDetail] = field(default_factory=list)
    additional_latency_ms: int = 0


class WorkflowEngine:
    def __init__(
        self,
        tool_gateway: ToolGateway | None = None,
        workflow_agent: WorkflowAgent | None = None,
        max_planner_steps: int = 6,
    ) -> None:
        self._tool_gateway = tool_gateway
        self._workflow_agent = workflow_agent or DeterministicWorkflowAgent()
        self._max_planner_steps = max(max_planner_steps, 1)

    def execute(
        self,
        *,
        request_class: RequestClass,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        if request_class == RequestClass.PORTFOLIO_ANALYSIS:
            return self._run_portfolio_analysis(request=request, context=context)
        if request_class == RequestClass.STRATEGY_FIT:
            return self._run_strategy_fit(request=request, context=context)
        if request_class == RequestClass.STRATEGY_PARSING:
            return self._run_strategy_parsing(request=request, context=context)
        if request_class == RequestClass.ASSET_ANALYSIS:
            return self._run_asset_analysis(request=request, context=context)
        return self._run_portfolio_qa(request=request, context=context)

    def _run_portfolio_analysis(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        directive = self._workflow_agent.snapshot_directive(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            message_text=request.message_text,
            correlation_id=context.correlation_id,
        )
        if self._tool_gateway is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.OK,
                    summary="Запрос классифицирован как анализ портфеля.",
                    details=[
                        "workflow=portfolio_analysis",
                        "mode=classification_only",
                    ],
                ),
                directive.signal,
            )

        plan = self._execute_snapshot_plan(
            request=request,
            context=context,
            workflow_name="portfolio_analysis",
            show_success_statuses={ToolStatus.OK},
            primary_tool=directive.primary_tool,
            prefer_aggregate=directive.prefer_aggregate,
        )
        if plan.terminal_result is not None:
            return _apply_workflow_agent_signal(
                _append_plan_details(plan.terminal_result, plan.plan_details),
                directive.signal,
            )

        tool_result = plan.snapshot_result
        if tool_result is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.ERROR,
                    summary="Workflow planner завершился без результата snapshot.",
                    details=[
                        "workflow=portfolio_analysis",
                        "reason=planner_snapshot_result_missing",
                    ]
                    + plan.plan_details,
                ),
                directive.signal,
            )

        result = _build_portfolio_result(
            tool_result=tool_result,
            collected=plan.used_collect,
            additional_tool_latency_ms=_additional_latency(plan.tool_latency_ms, tool_result),
        )
        return _apply_workflow_agent_signal(
            _append_plan_details(result, plan.plan_details),
            directive.signal,
        )

    def _run_strategy_fit(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        directive = self._workflow_agent.strategy_fit_directive(
            message_text=request.message_text,
            correlation_id=context.correlation_id,
        )
        if self._tool_gateway is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.UNAVAILABLE,
                    summary="Strategy-fit workflow пока не подключен к retriever/tool gateway.",
                    details=[
                        "workflow=strategy_fit",
                        "reason=tool_gateway_not_configured",
                    ],
                ),
                directive.signal,
            )

        plan = self._execute_strategy_fit_plan(
            request=request,
            context=context,
            workflow_name="strategy_fit",
            prefer_aggregate=directive.prefer_aggregate,
        )
        if plan.terminal_result is not None:
            return _apply_workflow_agent_signal(
                _append_plan_details(plan.terminal_result, plan.plan_details),
                directive.signal,
            )

        fit_result = plan.fit_result
        if fit_result is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.ERROR,
                    summary="Workflow planner завершился без strategy-fit результата.",
                    details=[
                        "workflow=strategy_fit",
                        "reason=planner_strategy_fit_result_missing",
                    ]
                    + plan.plan_details,
                ),
                directive.signal,
            )

        result = _build_strategy_fit_result(
            fit_result,
            additional_tool_latency_ms=_additional_latency(plan.tool_latency_ms, fit_result),
        )
        return _apply_workflow_agent_signal(
            _append_plan_details(result, plan.plan_details),
            directive.signal,
        )

    def _run_strategy_parsing(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        directive = self._workflow_agent.strategy_parsing_directive(
            message_text=request.message_text,
            principal_id=context.principal_id,
            correlation_id=context.correlation_id,
        )
        draft = directive.draft
        if draft.clarification_question is not None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Для сохранения стратегии нужны уточнения.",
                    details=[
                        "workflow=strategy_parsing",
                        "reason=missing_required_slots",
                        f"missing_slots={','.join(draft.missing_slots)}",
                        f"clarification={draft.clarification_question}",
                    ],
                    evidence=OrchestratorEvidence(
                        coverage=1.0,
                        unsupported_criteria=list(draft.missing_slots),
                    ),
                ),
                directive.signal,
            )

        if self._tool_gateway is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Стратегия распознана, но сохранение сейчас недоступно.",
                    details=[
                        "workflow=strategy_parsing",
                        "reason=tool_gateway_not_configured",
                        f"risk_tolerance={draft.risk_tolerance}",
                    ],
                ),
                directive.signal,
            )

        save_result = self._tool_gateway.execute(
            tool_name="strategy_save",
            principal_id=context.principal_id,
            correlation_id=context.correlation_id,
            payload={"strategy": draft.strategy_payload},
        )

        if save_result.status in {ToolStatus.OK, ToolStatus.PARTIAL}:
            return _apply_workflow_agent_signal(
                _build_strategy_parsing_result(save_result, draft),
                directive.signal,
            )

        return _apply_workflow_agent_signal(
            _build_failure_result(
                tool_result=save_result,
                workflow_name="strategy_parsing",
                reason="strategy_save_failed",
            ),
            directive.signal,
        )

    def _run_asset_analysis(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        directive = self._workflow_agent.asset_analysis_directive(
            message_text=request.message_text,
            correlation_id=context.correlation_id,
        )
        ticker = directive.ticker
        if ticker is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Для анализа актива нужен тикер.",
                    details=[
                        "workflow=asset_analysis",
                        "reason=ticker_missing",
                        "clarification=Укажите тикер, например: 'Сделай анализ актива SBER'.",
                    ],
                    evidence=OrchestratorEvidence(
                        coverage=1.0,
                        unsupported_criteria=["asset_ticker_missing"],
                    ),
                ),
                directive.signal,
            )

        if self._tool_gateway is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Asset analysis workflow пока не подключен к guarded retrieval.",
                    details=[
                        "workflow=asset_analysis",
                        "reason=tool_gateway_not_configured",
                        f"ticker={ticker}",
                    ],
                ),
                directive.signal,
            )

        plan = self._execute_snapshot_plan(
            request=request,
            context=context,
            workflow_name="asset_analysis",
            show_success_statuses={ToolStatus.OK, ToolStatus.PARTIAL},
            primary_tool=directive.primary_tool,
            prefer_aggregate=directive.prefer_aggregate,
        )
        if plan.terminal_result is not None:
            return _apply_workflow_agent_signal(
                _append_plan_details(plan.terminal_result, plan.plan_details),
                directive.signal,
            )

        tool_result = plan.snapshot_result
        if tool_result is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.ERROR,
                    summary="Workflow planner завершился без результата snapshot.",
                    details=[
                        "workflow=asset_analysis",
                        "reason=planner_snapshot_result_missing",
                    ]
                    + plan.plan_details,
                ),
                directive.signal,
            )

        result = _build_asset_analysis_result(
            tool_result=tool_result,
            ticker=ticker,
            signal_collection=self._collect_asset_signals(
                context=context,
                ticker=ticker,
                macro_keys=list(directive.macro_keys),
            ),
            additional_tool_latency_ms=_additional_latency(plan.tool_latency_ms, tool_result),
        )
        return _apply_workflow_agent_signal(
            _append_plan_details(result, plan.plan_details),
            directive.signal,
        )

    def _collect_asset_signals(
        self,
        *,
        context: CorrelationContext,
        ticker: str,
        macro_keys: list[str],
    ) -> AssetSignalCollection:
        if self._tool_gateway is None:
            return AssetSignalCollection()

        details: list[str] = []
        unsupported_criteria: list[str] = []
        errors: list[ErrorDetail] = []
        macro_indicators: dict[str, float] = {}
        issuer_indicators: dict[str, float] = {}
        additional_latency_ms = 0

        if self._supports_tool("macro_indicators_show"):
            macro_result = self._execute_planned_tool(
                workflow_name="asset_analysis",
                tool_name="macro_indicators_show",
                context=context,
                payload={"keys": macro_keys},
            )
            if isinstance(macro_result, ToolResult):
                additional_latency_ms += macro_result.meta.latency_ms
                errors.extend(list(macro_result.errors))
                macro_indicators = _extract_macro_values(macro_result)
                if macro_indicators:
                    details.append(
                        "macro_indicators="
                        + ",".join(
                            f"{key}:{value:.4f}" for key, value in sorted(macro_indicators.items())
                        )
                    )
                else:
                    unsupported_criteria.append("macro_indicators_unavailable")
            else:
                unsupported_criteria.append("macro_indicators_unavailable")

        if self._supports_tool("issuer_indicators_show"):
            issuer_result = self._execute_planned_tool(
                workflow_name="asset_analysis",
                tool_name="issuer_indicators_show",
                context=context,
                payload={"ticker": ticker},
            )
            if isinstance(issuer_result, ToolResult):
                additional_latency_ms += issuer_result.meta.latency_ms
                errors.extend(list(issuer_result.errors))
                issuer_indicators = _extract_issuer_values(issuer_result)
                if issuer_indicators:
                    details.append(
                        "issuer_indicators="
                        + ",".join(
                            f"{key}:{value:.4f}" for key, value in sorted(issuer_indicators.items())
                        )
                    )
                else:
                    unsupported_criteria.append("issuer_indicators_unavailable")
            else:
                unsupported_criteria.append("issuer_indicators_unavailable")

        return AssetSignalCollection(
            macro_indicators=macro_indicators,
            issuer_indicators=issuer_indicators,
            details=details,
            unsupported_criteria=unsupported_criteria,
            errors=errors,
            additional_latency_ms=additional_latency_ms,
        )

    def _run_portfolio_qa(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
    ) -> WorkflowExecutionResult:
        directive = self._workflow_agent.portfolio_qa_directive(
            message_text=request.message_text,
            correlation_id=context.correlation_id,
        )
        if self._tool_gateway is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.PARTIAL,
                    summary="Portfolio Q&A workflow пока не подключен к guarded retrieval.",
                    details=[
                        "workflow=portfolio_qa",
                        "reason=tool_gateway_not_configured",
                        (
                            "clarification=Спросите про долю акций, облигаций, "
                            "кэша или количество позиций."
                        ),
                    ],
                    evidence=OrchestratorEvidence(
                        coverage=1.0,
                        unsupported_criteria=["qa_question_not_supported"],
                    ),
                ),
                directive.signal,
            )

        plan = self._execute_snapshot_plan(
            request=request,
            context=context,
            workflow_name="portfolio_qa",
            show_success_statuses={ToolStatus.OK, ToolStatus.PARTIAL},
            primary_tool=directive.primary_tool,
            prefer_aggregate=directive.prefer_aggregate,
        )
        if plan.terminal_result is not None:
            return _apply_workflow_agent_signal(
                _append_plan_details(plan.terminal_result, plan.plan_details),
                directive.signal,
            )

        tool_result = plan.snapshot_result
        if tool_result is None:
            return _apply_workflow_agent_signal(
                WorkflowExecutionResult(
                    status=ResponseStatus.ERROR,
                    summary="Workflow planner завершился без результата snapshot.",
                    details=[
                        "workflow=portfolio_qa",
                        "reason=planner_snapshot_result_missing",
                    ]
                    + plan.plan_details,
                ),
                directive.signal,
            )

        result = _build_portfolio_qa_result(
            tool_result=tool_result,
            intent=directive.intent,
            additional_tool_latency_ms=_additional_latency(plan.tool_latency_ms, tool_result),
        )
        return _apply_workflow_agent_signal(
            _append_plan_details(result, plan.plan_details),
            directive.signal,
        )

    def _execute_snapshot_plan(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
        workflow_name: str,
        show_success_statuses: set[ToolStatus],
        primary_tool: str,
        prefer_aggregate: bool,
    ) -> SnapshotPlanExecution:
        if self._tool_gateway is None:
            return SnapshotPlanExecution(account_id=request.account_id)

        plan_details = [
            f"planner={_BOUNDED_PLANNER_VERSION}",
            f"planner.workflow={workflow_name}",
            f"planner.max_steps={self._max_planner_steps}",
        ]
        steps_used = 0
        tool_latency_ms = 0
        account_id = request.account_id

        if account_id is not None:
            plan_details.append("planner.account_source=request")
        else:
            if steps_used >= self._max_planner_steps:
                return SnapshotPlanExecution(
                    account_id=None,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            steps_used += 1
            plan_details.append(f"planner.step={steps_used}:portfolio_accounts")

            accounts_result = self._execute_planned_tool(
                workflow_name=workflow_name,
                tool_name="portfolio_accounts",
                context=context,
                payload={},
            )
            if isinstance(accounts_result, WorkflowExecutionResult):
                return SnapshotPlanExecution(
                    account_id=None,
                    plan_details=plan_details,
                    terminal_result=accounts_result,
                )

            tool_latency_ms += accounts_result.meta.latency_ms
            if accounts_result.status == ToolStatus.ERROR:
                return SnapshotPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=_build_failure_result(
                        tool_result=accounts_result,
                        workflow_name=workflow_name,
                        reason="account_discovery_failed",
                    ),
                )

            account_ids = _read_account_ids(accounts_result)
            if len(account_ids) > 1 and not prefer_aggregate:
                options = ", ".join(account_ids)
                return SnapshotPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=WorkflowExecutionResult(
                        status=ResponseStatus.PARTIAL,
                        summary=(
                            "Найдено несколько счетов. Выберите account_id для точного анализа "
                            "или явно запросите агрегированный режим."
                        ),
                        details=[
                            f"workflow={workflow_name}",
                            "reason=account_selection_required",
                            f"account_options={options}",
                            (
                                "clarification=Укажите account_id из списка или добавьте в запрос "
                                "'по всем счетам'."
                            ),
                        ],
                        evidence=OrchestratorEvidence(
                            as_of_ts=accounts_result.meta.source_ts,
                            coverage=_resolve_coverage(accounts_result),
                            unsupported_criteria=["account_id_missing"],
                        ),
                        errors=list(accounts_result.errors),
                        tool_latency_ms=tool_latency_ms,
                    ),
                )

            if account_ids:
                account_id = account_ids[0]
                plan_details.append(f"planner.account_selected={account_id}")
            else:
                plan_details.append("planner.account_selected=aggregate")

            if len(account_ids) > 1 and prefer_aggregate:
                account_id = None
                plan_details.append("planner.account_selected=aggregate_preferred")

        if primary_tool not in {"portfolio_show", "portfolio_collect"}:
            primary_tool = "portfolio_show"

        if steps_used >= self._max_planner_steps:
            return SnapshotPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=self._planner_step_limit_result(workflow_name),
            )
        steps_used += 1
        plan_details.append(f"planner.step={steps_used}:{primary_tool}")
        primary_result = self._execute_planned_tool(
            workflow_name=workflow_name,
            tool_name=primary_tool,
            context=context,
            payload={"account_id": account_id},
        )
        if isinstance(primary_result, WorkflowExecutionResult):
            return SnapshotPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=primary_result,
            )

        tool_latency_ms += primary_result.meta.latency_ms
        primary_success_statuses = show_success_statuses
        if primary_tool == "portfolio_collect":
            primary_success_statuses = {ToolStatus.OK, ToolStatus.PARTIAL}
        if primary_result.status in primary_success_statuses:
            return SnapshotPlanExecution(
                account_id=account_id,
                snapshot_result=primary_result,
                used_collect=primary_tool == "portfolio_collect",
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
            )

        fallback_tool = (
            "portfolio_collect" if primary_tool == "portfolio_show" else "portfolio_show"
        )

        if steps_used >= self._max_planner_steps:
            return SnapshotPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=self._planner_step_limit_result(workflow_name),
            )
        steps_used += 1
        plan_details.append(f"planner.step={steps_used}:{fallback_tool}")
        collect_result = self._execute_planned_tool(
            workflow_name=workflow_name,
            tool_name=fallback_tool,
            context=context,
            payload={"account_id": account_id},
        )
        if isinstance(collect_result, WorkflowExecutionResult):
            return SnapshotPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=collect_result,
            )

        tool_latency_ms += collect_result.meta.latency_ms
        fallback_success_statuses = {ToolStatus.OK, ToolStatus.PARTIAL}
        if fallback_tool == "portfolio_show":
            fallback_success_statuses = show_success_statuses
        if collect_result.status in fallback_success_statuses:
            return SnapshotPlanExecution(
                account_id=account_id,
                snapshot_result=collect_result,
                used_collect=fallback_tool == "portfolio_collect",
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
            )

        reason = "portfolio_data_unavailable"
        if workflow_name == "asset_analysis":
            reason = "asset_data_unavailable"
        if workflow_name == "portfolio_qa":
            reason = "portfolio_qa_data_unavailable"
        return SnapshotPlanExecution(
            account_id=account_id,
            tool_latency_ms=tool_latency_ms,
            plan_details=plan_details,
            terminal_result=_build_failure_result(
                tool_result=collect_result,
                workflow_name=workflow_name,
                reason=reason,
            ),
        )

    def _execute_strategy_fit_plan(
        self,
        *,
        request: OrchestratorRequest,
        context: CorrelationContext,
        workflow_name: str,
        prefer_aggregate: bool,
    ) -> StrategyFitPlanExecution:
        if self._tool_gateway is None:
            return StrategyFitPlanExecution(account_id=request.account_id)

        plan_details = [
            f"planner={_BOUNDED_PLANNER_VERSION}",
            f"planner.workflow={workflow_name}",
            f"planner.max_steps={self._max_planner_steps}",
        ]
        steps_used = 0
        tool_latency_ms = 0
        account_id = request.account_id

        if account_id is not None:
            plan_details.append("planner.account_source=request")
        else:
            if steps_used >= self._max_planner_steps:
                return StrategyFitPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            steps_used += 1
            plan_details.append(f"planner.step={steps_used}:portfolio_accounts")

            accounts_result = self._execute_planned_tool(
                workflow_name=workflow_name,
                tool_name="portfolio_accounts",
                context=context,
                payload={},
            )
            if isinstance(accounts_result, WorkflowExecutionResult):
                return StrategyFitPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=accounts_result,
                )

            tool_latency_ms += accounts_result.meta.latency_ms
            if accounts_result.status == ToolStatus.ERROR:
                return StrategyFitPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=_build_failure_result(
                        tool_result=accounts_result,
                        workflow_name=workflow_name,
                        reason="account_discovery_failed",
                    ),
                )

            account_ids = _read_account_ids(accounts_result)
            if len(account_ids) > 1 and not prefer_aggregate:
                options = ", ".join(account_ids)
                return StrategyFitPlanExecution(
                    account_id=None,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=WorkflowExecutionResult(
                        status=ResponseStatus.PARTIAL,
                        summary=(
                            "Найдено несколько счетов. Выберите account_id для точного анализа "
                            "или явно запросите агрегированный режим."
                        ),
                        details=[
                            f"workflow={workflow_name}",
                            "reason=account_selection_required",
                            f"account_options={options}",
                            (
                                "clarification=Укажите account_id из списка или добавьте в запрос "
                                "'по всем счетам'."
                            ),
                        ],
                        evidence=OrchestratorEvidence(
                            as_of_ts=accounts_result.meta.source_ts,
                            coverage=_resolve_coverage(accounts_result),
                            unsupported_criteria=["account_id_missing"],
                        ),
                        errors=list(accounts_result.errors),
                        tool_latency_ms=tool_latency_ms,
                    ),
                )

            if account_ids:
                account_id = account_ids[0]
                plan_details.append(f"planner.account_selected={account_id}")
            else:
                plan_details.append("planner.account_selected=aggregate")

            if len(account_ids) > 1 and prefer_aggregate:
                account_id = None
                plan_details.append("planner.account_selected=aggregate_preferred")

        fit_payload = {"account_id": account_id}

        def run_step(
            *,
            tool_name: str,
            payload: dict[str, Any],
        ) -> tuple[ToolResult | None, WorkflowExecutionResult | None]:
            nonlocal steps_used, tool_latency_ms
            if steps_used >= self._max_planner_steps:
                return None, self._planner_step_limit_result(workflow_name)
            steps_used += 1
            plan_details.append(f"planner.step={steps_used}:{tool_name}")
            result = self._execute_planned_tool(
                workflow_name=workflow_name,
                tool_name=tool_name,
                context=context,
                payload=payload,
            )
            if isinstance(result, WorkflowExecutionResult):
                return None, result
            tool_latency_ms += result.meta.latency_ms
            return result, None

        fit_result, terminal = run_step(tool_name="strategy_fit", payload=fit_payload)
        if terminal is not None:
            return StrategyFitPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=terminal,
            )
        if fit_result is None:
            return StrategyFitPlanExecution(
                account_id=account_id,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
                terminal_result=self._planner_step_limit_result(workflow_name),
            )
        if fit_result.status in {ToolStatus.OK, ToolStatus.PARTIAL}:
            return StrategyFitPlanExecution(
                account_id=account_id,
                fit_result=fit_result,
                tool_latency_ms=tool_latency_ms,
                plan_details=plan_details,
            )

        if _is_missing_strategy_error(fit_result):
            draft = parse_strategy_message(
                message_text=request.message_text,
                principal_id=context.principal_id,
            )
            if draft.strategy_payload is None:
                clarification = draft.clarification_question or (
                    "Опишите риск-профиль и целевую аллокацию (акции/облигации/кэш)."
                )
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=WorkflowExecutionResult(
                        status=ResponseStatus.UNAVAILABLE,
                        summary="Стратегия не сохранена и не может быть восстановлена из запроса.",
                        details=[
                            "workflow=strategy_fit",
                            "reason=strategy_missing",
                            f"missing_slots={','.join(draft.missing_slots)}",
                            f"clarification={clarification}",
                        ],
                        evidence=OrchestratorEvidence(
                            coverage=1.0,
                            unsupported_criteria=list(draft.missing_slots),
                        ),
                        errors=list(fit_result.errors),
                        tool_latency_ms=tool_latency_ms,
                    ),
                )

            save_result, terminal = run_step(
                tool_name="strategy_save",
                payload={"strategy": draft.strategy_payload},
            )
            if terminal is not None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=terminal,
                )
            if save_result is None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            if save_result.status not in {ToolStatus.OK, ToolStatus.PARTIAL}:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=_build_failure_result(
                        tool_result=save_result,
                        workflow_name=workflow_name,
                        reason="strategy_save_failed",
                        additional_tool_latency_ms=_additional_latency(
                            tool_latency_ms,
                            save_result,
                        ),
                    ),
                )

            fit_result, terminal = run_step(tool_name="strategy_fit", payload=fit_payload)
            if terminal is not None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=terminal,
                )
            if fit_result is None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            if fit_result.status in {ToolStatus.OK, ToolStatus.PARTIAL}:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    fit_result=fit_result,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                )

        if _is_missing_snapshot_error(fit_result):
            collect_result, terminal = run_step(
                tool_name="portfolio_collect",
                payload={"account_id": account_id},
            )
            if terminal is not None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=terminal,
                )
            if collect_result is None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            if collect_result.status == ToolStatus.ERROR:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=_build_failure_result(
                        tool_result=collect_result,
                        workflow_name=workflow_name,
                        reason="strategy_fit_snapshot_collect_failed",
                        additional_tool_latency_ms=(
                            _additional_latency(tool_latency_ms, collect_result)
                        ),
                    ),
                )

            fit_result, terminal = run_step(tool_name="strategy_fit", payload=fit_payload)
            if terminal is not None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=terminal,
                )
            if fit_result is None:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                    terminal_result=self._planner_step_limit_result(workflow_name),
                )
            if fit_result.status in {ToolStatus.OK, ToolStatus.PARTIAL}:
                return StrategyFitPlanExecution(
                    account_id=account_id,
                    fit_result=fit_result,
                    tool_latency_ms=tool_latency_ms,
                    plan_details=plan_details,
                )

        return StrategyFitPlanExecution(
            account_id=account_id,
            tool_latency_ms=tool_latency_ms,
            plan_details=plan_details,
            terminal_result=_build_failure_result(
                tool_result=fit_result,
                workflow_name=workflow_name,
                reason="strategy_fit_tool_failed",
                additional_tool_latency_ms=_additional_latency(tool_latency_ms, fit_result),
            ),
        )

    def _execute_planned_tool(
        self,
        *,
        workflow_name: str,
        tool_name: str,
        context: CorrelationContext,
        payload: dict[str, Any],
    ) -> ToolResult | WorkflowExecutionResult:
        if self._tool_gateway is None:
            return WorkflowExecutionResult(
                status=ResponseStatus.ERROR,
                summary="Tool gateway is not configured for planned execution.",
                details=[
                    f"workflow={workflow_name}",
                    "reason=planner_tool_gateway_missing",
                ],
            )

        allowed_tools = _WORKFLOW_TOOL_ALLOWLIST.get(workflow_name, frozenset())
        if tool_name not in allowed_tools:
            return WorkflowExecutionResult(
                status=ResponseStatus.ERROR,
                summary="Planner attempted to use a non-allowlisted tool.",
                details=[
                    f"workflow={workflow_name}",
                    "reason=planner_action_not_allowlisted",
                    f"tool_name={tool_name}",
                ],
                errors=[
                    ErrorDetail(
                        code=ErrorCode.VALIDATION_ERROR,
                        message="planner action tool is not allowlisted",
                        retriable=False,
                        details={
                            "workflow": workflow_name,
                            "tool_name": tool_name,
                        },
                    )
                ],
            )

        return self._tool_gateway.execute(
            tool_name=tool_name,
            principal_id=context.principal_id,
            correlation_id=context.correlation_id,
            payload=payload,
        )

    def _supports_tool(self, tool_name: str) -> bool:
        if self._tool_gateway is None:
            return False
        list_tools = getattr(self._tool_gateway, "list_tools", None)
        if not callable(list_tools):
            return False
        try:
            available = list_tools()
        except Exception:
            return False
        if not isinstance(available, (list, tuple)):
            return False
        return tool_name in available

    def _planner_step_limit_result(self, workflow_name: str) -> WorkflowExecutionResult:
        return WorkflowExecutionResult(
            status=ResponseStatus.ERROR,
            summary="Workflow planner достиг лимита шагов.",
            details=[
                f"workflow={workflow_name}",
                "reason=planner_step_limit_reached",
            ],
            errors=[
                ErrorDetail(
                    code=ErrorCode.VALIDATION_ERROR,
                    message="workflow planner step limit reached",
                    retriable=False,
                    details={"max_planner_steps": self._max_planner_steps},
                )
            ],
        )


def _build_portfolio_result(
    *,
    tool_result: ToolResult,
    collected: bool,
    additional_tool_latency_ms: int = 0,
) -> WorkflowExecutionResult:
    snapshot = _read_snapshot(tool_result)
    positions_count, top3_concentration, hhi, class_exposure, unknown_share = (
        _portfolio_analysis_metrics(snapshot)
    )
    coverage = _resolve_coverage(tool_result)
    exposure_preview = _class_exposure_preview(class_exposure)

    status = ResponseStatus.OK if tool_result.status == ToolStatus.OK else ResponseStatus.PARTIAL
    if collected:
        source_phrase = "обновлен"
    else:
        source_phrase = "загружен из snapshot"
    summary = (
        f"Портфель {source_phrase}: позиций {positions_count}, "
        f"концентрация top-3 {top3_concentration * 100:.1f}%, "
        f"HHI {hhi:.4f}."
    )
    if exposure_preview:
        summary += f" Структура: {exposure_preview}."

    details = [
        "workflow=portfolio_analysis",
        f"positions_count={positions_count}",
        f"top3_concentration={top3_concentration:.4f}",
        f"hhi={hhi:.6f}",
        f"unknown_share={unknown_share:.4f}",
    ]
    for asset_class, weight in sorted(class_exposure.items()):
        details.append(f"class_exposure.{asset_class}={weight:.4f}")
    if collected:
        details.append("source=portfolio_collect")
    else:
        details.append("source=portfolio_show")

    return WorkflowExecutionResult(
        status=status,
        summary=summary,
        details=details,
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=coverage,
            unsupported_criteria=[],
        ),
        errors=list(tool_result.errors),
        tool_latency_ms=additional_tool_latency_ms + tool_result.meta.latency_ms,
    )


def _build_strategy_fit_result(
    tool_result: ToolResult,
    *,
    additional_tool_latency_ms: int = 0,
) -> WorkflowExecutionResult:
    strategy_fit_payload = tool_result.data.get("strategy_fit")
    if not isinstance(strategy_fit_payload, Mapping):
        return WorkflowExecutionResult(
            status=ResponseStatus.ERROR,
            summary="Strategy-fit вернул некорректный payload.",
            details=[
                "workflow=strategy_fit",
                "reason=invalid_strategy_fit_payload",
            ],
        )

    verdict = str(strategy_fit_payload.get("verdict", "unknown"))
    reason = _optional_text(strategy_fit_payload.get("reason"))
    unsupported_criteria = _read_string_list(strategy_fit_payload.get("unsupported_criteria"))
    coverage = _resolve_coverage(tool_result)

    status = ResponseStatus.OK if tool_result.status == ToolStatus.OK else ResponseStatus.PARTIAL
    details = [
        "workflow=strategy_fit",
        f"verdict={verdict}",
    ]
    if reason:
        details.append(f"reason={reason}")

    return WorkflowExecutionResult(
        status=status,
        summary=_strategy_fit_summary(verdict),
        details=details,
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=coverage,
            unsupported_criteria=unsupported_criteria,
        ),
        errors=list(tool_result.errors),
        tool_latency_ms=additional_tool_latency_ms + tool_result.meta.latency_ms,
    )


def _build_strategy_parsing_result(
    tool_result: ToolResult,
    draft: StrategyParsingDraft,
) -> WorkflowExecutionResult:
    status = ResponseStatus.OK if tool_result.status == ToolStatus.OK else ResponseStatus.PARTIAL
    allocation_keys = sorted(draft.target_asset_allocation)
    return WorkflowExecutionResult(
        status=status,
        summary="Инвестиционная стратегия сохранена.",
        details=[
            "workflow=strategy_parsing",
            f"risk_tolerance={draft.risk_tolerance}",
            f"allocation_keys={','.join(allocation_keys)}",
        ],
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=_resolve_coverage(tool_result),
            unsupported_criteria=[],
        ),
        errors=list(tool_result.errors),
        tool_latency_ms=tool_result.meta.latency_ms,
    )


def _build_asset_analysis_result(
    *,
    tool_result: ToolResult,
    ticker: str,
    signal_collection: AssetSignalCollection,
    additional_tool_latency_ms: int = 0,
) -> WorkflowExecutionResult:
    snapshot = _read_snapshot(tool_result)
    analysis = analyze_asset_in_snapshot(snapshot=snapshot, ticker=ticker)
    if analysis is None:
        return WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary=f"Актив {ticker} не найден в текущем портфеле.",
            details=[
                "workflow=asset_analysis",
                "reason=asset_not_in_portfolio",
                "clarification=Проверьте тикер или запросите анализ актива из портфеля.",
            ],
            evidence=OrchestratorEvidence(
                as_of_ts=tool_result.meta.source_ts,
                coverage=_resolve_coverage(tool_result),
                unsupported_criteria=["asset_not_in_portfolio"]
                + list(signal_collection.unsupported_criteria),
            ),
            errors=list(tool_result.errors) + list(signal_collection.errors),
            tool_latency_ms=(
                additional_tool_latency_ms
                + signal_collection.additional_latency_ms
                + tool_result.meta.latency_ms
            ),
        )

    status = ResponseStatus.OK if tool_result.status == ToolStatus.OK else ResponseStatus.PARTIAL
    if signal_collection.unsupported_criteria:
        status = ResponseStatus.PARTIAL
    details = [
        "workflow=asset_analysis",
        f"ticker={analysis.ticker}",
        "stage=macro",
        "stage=sector",
        "stage=issuer",
        "stage=portfolio_fit",
    ] + list(signal_collection.details)

    summary = _asset_summary(
        analysis,
        macro_indicators=signal_collection.macro_indicators,
        issuer_indicators=signal_collection.issuer_indicators,
    )
    return WorkflowExecutionResult(
        status=status,
        summary=summary,
        details=details,
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=_resolve_coverage(tool_result),
            unsupported_criteria=list(signal_collection.unsupported_criteria),
        ),
        errors=list(tool_result.errors) + list(signal_collection.errors),
        tool_latency_ms=(
            additional_tool_latency_ms
            + signal_collection.additional_latency_ms
            + tool_result.meta.latency_ms
        ),
    )


def _build_portfolio_qa_result(
    *,
    tool_result: ToolResult,
    intent: PortfolioQaIntent,
    additional_tool_latency_ms: int = 0,
) -> WorkflowExecutionResult:
    snapshot = _read_snapshot(tool_result)
    answer = answer_portfolio_question(snapshot=snapshot, intent=intent)
    if answer is None:
        return WorkflowExecutionResult(
            status=ResponseStatus.PARTIAL,
            summary="Пока поддерживаются только базовые вопросы по структуре портфеля.",
            details=[
                "workflow=portfolio_qa",
                "reason=qa_question_not_supported",
                "clarification=Спросите про долю акций, облигаций, кэша или количество позиций.",
            ],
            evidence=OrchestratorEvidence(
                as_of_ts=tool_result.meta.source_ts,
                coverage=_resolve_coverage(tool_result),
                unsupported_criteria=["qa_question_not_supported"],
            ),
            errors=list(tool_result.errors),
            tool_latency_ms=additional_tool_latency_ms + tool_result.meta.latency_ms,
        )

    status = ResponseStatus.OK if tool_result.status == ToolStatus.OK else ResponseStatus.PARTIAL
    return WorkflowExecutionResult(
        status=status,
        summary=answer,
        details=[
            "workflow=portfolio_qa",
            f"question_kind={intent.kind}",
        ],
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=_resolve_coverage(tool_result),
            unsupported_criteria=[],
        ),
        errors=list(tool_result.errors),
        tool_latency_ms=additional_tool_latency_ms + tool_result.meta.latency_ms,
    )


def _asset_summary(
    analysis: AssetAnalysisResult,
    *,
    macro_indicators: Mapping[str, float],
    issuer_indicators: Mapping[str, float],
) -> str:
    parts = [
        (
            f"Анализ {analysis.ticker}: доля в портфеле {analysis.weight * 100:.1f}%, "
            f"рыночная стоимость {analysis.market_value:.2f}."
        )
    ]
    if macro_indicators:
        fragments = []
        for key in ("key_rate", "inflation", "imoex", "usd_rub"):
            value = macro_indicators.get(key)
            if value is None:
                continue
            fragments.append(f"{key}={value:.4f}")
        if fragments:
            parts.append("Макроиндикаторы: " + ", ".join(fragments) + ".")

    if issuer_indicators:
        last = issuer_indicators.get("last")
        close = issuer_indicators.get("close")
        if last is not None:
            sentence = f"MOEX last={last:.4f}"
            if close is not None:
                sentence += f", close={close:.4f}"
            parts.append(sentence + ".")
    return " ".join(parts)


def _build_failure_result(
    *,
    tool_result: ToolResult,
    workflow_name: str,
    reason: str,
    additional_tool_latency_ms: int = 0,
) -> WorkflowExecutionResult:
    status = _map_failure_status(tool_result.errors)
    summary = (
        "Workflow временно недоступен из-за внешних ограничений."
        if status == ResponseStatus.UNAVAILABLE
        else "Workflow завершился ошибкой валидации или доступа."
    )
    return WorkflowExecutionResult(
        status=status,
        summary=summary,
        details=[
            f"workflow={workflow_name}",
            f"reason={reason}",
        ],
        evidence=OrchestratorEvidence(
            as_of_ts=tool_result.meta.source_ts,
            coverage=_resolve_coverage(tool_result),
            unsupported_criteria=[],
        ),
        errors=list(tool_result.errors),
        tool_latency_ms=additional_tool_latency_ms + tool_result.meta.latency_ms,
    )


def _append_plan_details(
    result: WorkflowExecutionResult,
    plan_details: list[str],
) -> WorkflowExecutionResult:
    if not plan_details:
        return result
    return WorkflowExecutionResult(
        status=result.status,
        summary=result.summary,
        details=list(result.details) + list(plan_details),
        evidence=result.evidence,
        errors=list(result.errors),
        tool_latency_ms=result.tool_latency_ms,
        llm_tokens_in=result.llm_tokens_in,
        llm_tokens_out=result.llm_tokens_out,
        retry_count=result.retry_count,
    )


def _apply_workflow_agent_signal(
    result: WorkflowExecutionResult,
    signal: WorkflowAgentSignal,
) -> WorkflowExecutionResult:
    details = list(result.details)
    details.append(f"workflow_agent.mode={signal.mode}")
    details.append(f"workflow_agent.prompt_id={signal.prompt_id}")
    details.append(f"workflow_agent.prompt_version={signal.prompt_version}")
    details.append(f"workflow_agent.model_id={signal.model_id or 'deterministic'}")
    if signal.fallback_reason:
        details.append(f"workflow_agent.fallback_reason={signal.fallback_reason}")

    return WorkflowExecutionResult(
        status=result.status,
        summary=result.summary,
        details=details,
        evidence=result.evidence,
        errors=list(result.errors),
        tool_latency_ms=result.tool_latency_ms,
        llm_tokens_in=result.llm_tokens_in + signal.llm_tokens_in,
        llm_tokens_out=result.llm_tokens_out + signal.llm_tokens_out,
        retry_count=result.retry_count + signal.retry_count,
    )


def _additional_latency(total_tool_latency_ms: int, tool_result: ToolResult) -> int:
    value = total_tool_latency_ms - tool_result.meta.latency_ms
    return value if value > 0 else 0


def _map_failure_status(errors: list[ErrorDetail]) -> ResponseStatus:
    if errors and all(error.code in _TRANSIENT_ERROR_CODES for error in errors):
        return ResponseStatus.UNAVAILABLE
    return ResponseStatus.ERROR


def _read_snapshot(tool_result: ToolResult) -> Mapping[str, Any]:
    raw_snapshot = tool_result.data.get("snapshot")
    if isinstance(raw_snapshot, Mapping):
        return raw_snapshot
    return {}


def _extract_macro_values(tool_result: ToolResult) -> dict[str, float]:
    raw_indicators = tool_result.data.get("indicators")
    if not isinstance(raw_indicators, list):
        return {}

    values: dict[str, float] = {}
    for item in raw_indicators:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "").strip().lower()
        value = _to_float(item.get("value"))
        if not key or value is None:
            continue
        values[key] = value
    return values


def _extract_issuer_values(tool_result: ToolResult) -> dict[str, float]:
    raw_payload = tool_result.data.get("issuer_indicators")
    if not isinstance(raw_payload, Mapping):
        return {}
    values: dict[str, float] = {}
    for key in ("last", "close", "open", "high", "low", "volume", "value"):
        value = _to_float(raw_payload.get(key))
        if value is None:
            continue
        values[key] = value
    return values


def _to_float(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool) or raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _is_missing_strategy_error(tool_result: ToolResult) -> bool:
    if tool_result.status != ToolStatus.ERROR:
        return False
    for error in tool_result.errors:
        if error.code != ErrorCode.UPSTREAM_UNAVAILABLE:
            continue
        message = error.message.strip().lower()
        if "structured strategy is not available" in message:
            return True
    return False


def _is_missing_snapshot_error(tool_result: ToolResult) -> bool:
    if tool_result.status != ToolStatus.ERROR:
        return False
    for error in tool_result.errors:
        if error.code != ErrorCode.UPSTREAM_UNAVAILABLE:
            continue
        message = error.message.strip().lower()
        if "portfolio snapshot is not available" in message:
            return True
    return False


def _positions_count(snapshot: Mapping[str, Any]) -> int:
    raw_positions = snapshot.get("positions")
    if isinstance(raw_positions, list):
        return len(raw_positions)
    return 0


def _portfolio_analysis_metrics(
    snapshot: Mapping[str, Any],
) -> tuple[int, float, float, dict[str, float], float]:
    raw_positions = snapshot.get("positions")
    positions: list[Mapping[str, Any]] = []
    if isinstance(raw_positions, list):
        for item in raw_positions:
            if isinstance(item, Mapping):
                positions.append(item)

    position_values = [
        _to_float(position.get("market_value")) or 0.0
        for position in positions
    ]
    explicit_total = _to_float(snapshot.get("portfolio_value"))
    portfolio_total = explicit_total if explicit_total is not None and explicit_total > 0 else 0.0
    if portfolio_total <= 0.0:
        portfolio_total = sum(position_values)

    weights: list[float] = []
    class_exposure: dict[str, float] = {}
    for idx, position in enumerate(positions):
        value = position_values[idx] if idx < len(position_values) else 0.0
        weight = value / portfolio_total if portfolio_total > 0 else 0.0
        weights.append(weight)

        asset_class = _normalize_asset_class(position.get("asset_class"))
        class_exposure[asset_class] = class_exposure.get(asset_class, 0.0) + weight

    sorted_weights = sorted(weights, reverse=True)
    top3_concentration = sum(sorted_weights[:3])
    hhi = sum(item * item for item in weights)
    unknown_share = _to_float(snapshot.get("unknown_share")) or 0.0
    if unknown_share < 0.0:
        unknown_share = 0.0
    if unknown_share > 1.0:
        unknown_share = 1.0
    return len(positions), top3_concentration, hhi, class_exposure, unknown_share


def _normalize_asset_class(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if not normalized:
        return "unknown"
    mapping = {
        "stock": "equity",
        "stocks": "equity",
        "акции": "equity",
        "акция": "equity",
        "bond": "bond",
        "bonds": "bond",
        "облигации": "bond",
        "облигация": "bond",
        "cash": "cash",
        "money_market": "cash",
        "money_market_funds": "cash",
        "кэш": "cash",
        "кеш": "cash",
    }
    return mapping.get(normalized, normalized)


def _class_exposure_preview(class_exposure: Mapping[str, float]) -> str:
    if not class_exposure:
        return ""
    preferred_order = ("equity", "bond", "cash")
    values: list[str] = []
    for key in preferred_order:
        weight = class_exposure.get(key)
        if weight is None:
            continue
        values.append(f"{key} {weight * 100:.1f}%")
    for key in sorted(class_exposure):
        if key in preferred_order:
            continue
        values.append(f"{key} {class_exposure[key] * 100:.1f}%")
    return ", ".join(values)


def _resolve_coverage(tool_result: ToolResult) -> float:
    candidates = [
        tool_result.meta.coverage,
        tool_result.data.get("coverage"),
    ]
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int | float):
            value = float(candidate)
            if 0.0 <= value <= 1.0:
                return value
    return 1.0


def _optional_text(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    return normalized


def _read_string_list(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []

    values: list[str] = []
    for item in raw_value:
        text = str(item).strip()
        if text:
            values.append(text)
    return values


def _read_account_ids(tool_result: ToolResult) -> list[str]:
    raw_value = tool_result.data.get("account_ids")
    if not isinstance(raw_value, list):
        return []

    values: list[str] = []
    for item in raw_value:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _strategy_fit_summary(verdict: str) -> str:
    if verdict == "fit":
        return "Портфель соответствует заданной стратегии."
    if verdict == "partial_fit":
        return "Портфель частично соответствует стратегии."
    if verdict == "not_fit":
        return "Портфель не соответствует стратегии."
    return "Недостаточно данных для окончательного strategy-fit verdict."
