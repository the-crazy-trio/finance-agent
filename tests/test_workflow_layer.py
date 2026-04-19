from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.contracts.common import ErrorCode, RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.contracts.tooling import ToolMeta, ToolResult
from finance_agent.orchestration.context import build_correlation_context
from finance_agent.workflow import WorkflowEngine, determine_request_class


class _FakeGateway:
    def __init__(self, responses: dict[str, list[ToolResult]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        *,
        tool_name: str,
        principal_id: str,
        correlation_id: str,
        payload: dict[str, object] | None = None,
    ) -> ToolResult:
        self.calls.append(
            {
                "tool_name": tool_name,
                "principal_id": principal_id,
                "correlation_id": correlation_id,
                "payload": payload or {},
            }
        )
        tool_responses = self._responses.get(tool_name, [])
        if not tool_responses:
            raise AssertionError(f"No fake response configured for {tool_name}")
        return tool_responses.pop(0)

    def list_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._responses))


class WorkflowLayerTests(unittest.TestCase):
    def test_request_class_mapping_is_one_to_one_with_workflows(self) -> None:
        self.assertEqual(
            determine_request_class("Проверь соответствие портфеля стратегии"),
            RequestClass.STRATEGY_FIT,
        )
        self.assertEqual(
            determine_request_class("Обнови мою инвестиционную стратегию"),
            RequestClass.STRATEGY_PARSING,
        )
        self.assertEqual(
            determine_request_class("Сделай анализ актива SBER"),
            RequestClass.ASSET_ANALYSIS,
        )
        self.assertEqual(
            determine_request_class("Какая доля облигаций в портфеле?"),
            RequestClass.PORTFOLIO_QA,
        )
        self.assertEqual(
            determine_request_class("Покажи общий анализ портфеля"),
            RequestClass.PORTFOLIO_ANALYSIS,
        )

    def test_portfolio_analysis_workflow_uses_snapshot_tool(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [{"ticker": "SBER"}, {"ticker": "GAZP"}],
                                "portfolio_value": 1200.0,
                            }
                        },
                        meta=ToolMeta(
                            source_ts=datetime(2026, 4, 18, 10, 0, tzinfo=UTC),
                            coverage=0.9,
                            correlation_id="corr-portfolio",
                        ),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Покажи общий анализ портфеля",
            account_id="acc-1",
            correlation_id="corr-portfolio",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertAlmostEqual(result.evidence.coverage, 0.9)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_show")

    def test_strategy_fit_workflow_maps_tool_errors_to_unavailable(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_fit": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="strategy data is unavailable",
                        meta=ToolMeta(correlation_id="corr-fit"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Проверь соответствие портфеля стратегии",
            account_id="acc-1",
            correlation_id="corr-fit",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_FIT,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.UNAVAILABLE)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)

    def test_strategy_fit_bootstraps_strategy_from_message_and_retries(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_fit": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="Structured strategy is not available",
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-strategy"),
                    ),
                    ToolResult.ok(
                        data={"strategy_fit": {"verdict": "fit", "unsupported_criteria": []}},
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-strategy"),
                    ),
                ],
                "strategy_save": [
                    ToolResult.ok(
                        data={"strategy": {"client_id": "local_user"}},
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-strategy"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message=(
                "Проверь соответствие портфеля стратегии: риск balanced, "
                "акции 60%, облигации 30%, кэш 10%"
            ),
            account_id="acc-1",
            correlation_id="corr-fit-bootstrap-strategy",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_FIT,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(
            [item["tool_name"] for item in gateway.calls],
            ["strategy_fit", "strategy_save", "strategy_fit"],
        )

    def test_strategy_fit_collects_snapshot_when_missing_and_retries(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_fit": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="Portfolio snapshot is not available",
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-snapshot"),
                    ),
                    ToolResult.ok(
                        data={
                            "strategy_fit": {
                                "verdict": "partial_fit",
                                "unsupported_criteria": ["coverage_below_threshold"],
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-snapshot"),
                    ),
                ],
                "portfolio_collect": [
                    ToolResult.ok(
                        data={"snapshot": {"positions": [], "portfolio_value": 0.0}},
                        meta=ToolMeta(correlation_id="corr-fit-bootstrap-snapshot"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Проверь соответствие портфеля стратегии",
            account_id="acc-1",
            correlation_id="corr-fit-bootstrap-snapshot",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_FIT,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(
            [item["tool_name"] for item in gateway.calls],
            ["strategy_fit", "portfolio_collect", "strategy_fit"],
        )

    def test_strategy_parsing_saves_strategy_when_required_slots_present(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_save": [
                    ToolResult.ok(
                        data={"strategy": {"client_id": "local_user"}},
                        meta=ToolMeta(correlation_id="corr-strategy-1"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message=(
                "Обнови стратегию: риск сбалансированный, "
                "акции 60%, облигации 30%, кэш 10%"
            ),
            correlation_id="corr-strategy-1",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_PARSING,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["tool_name"], "strategy_save")

        payload = gateway.calls[0]["payload"]
        self.assertEqual(payload["strategy"]["client_id"], "local_user")
        self.assertEqual(
            payload["strategy"]["risk_preferences"]["risk_tolerance_self_assessed"],
            "balanced",
        )
        self.assertAlmostEqual(
            payload["strategy"]["portfolio_policy"]["target_asset_allocation"]["equity"],
            0.6,
        )

    def test_strategy_parsing_returns_single_clarification_when_slots_missing(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_save": [
                    ToolResult.ok(
                        data={"strategy": {"client_id": "local_user"}},
                        meta=ToolMeta(correlation_id="corr-strategy-2"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Обнови стратегию",
            correlation_id="corr-strategy-2",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_PARSING,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.PARTIAL)
        clarifications = [item for item in result.details if item.startswith("clarification=")]
        self.assertEqual(len(clarifications), 1)
        self.assertEqual(len(gateway.calls), 0)

    def test_strategy_parsing_maps_save_errors_to_unavailable(self) -> None:
        gateway = _FakeGateway(
            responses={
                "strategy_save": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="strategy storage unavailable",
                        meta=ToolMeta(correlation_id="corr-strategy-3"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message=(
                "Стратегия growth, акции 70, облигации 20, cash 10. "
                "Сохрани пожалуйста"
            ),
            correlation_id="corr-strategy-3",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_PARSING,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.UNAVAILABLE)
        self.assertEqual(result.errors[0].code, ErrorCode.UPSTREAM_UNAVAILABLE)

    def test_portfolio_qa_answers_bond_share_via_portfolio_show(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [
                                    {
                                        "ticker": "SBER",
                                        "asset_class": "equity",
                                        "market_value": 700.0,
                                    },
                                    {
                                        "ticker": "OFZ",
                                        "asset_class": "bond",
                                        "market_value": 300.0,
                                    },
                                ],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(
                            source_ts=datetime(2026, 4, 18, 10, 0, tzinfo=UTC),
                            coverage=1.0,
                            correlation_id="corr-qa-1",
                        ),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Какая доля облигаций в портфеле?",
            account_id="acc-1",
            correlation_id="corr-qa-1",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_QA,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertIn("30.0%", result.summary)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_show")

    def test_portfolio_qa_falls_back_to_collect_when_show_fails(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="snapshot is stale",
                        meta=ToolMeta(correlation_id="corr-qa-2"),
                    )
                ],
                "portfolio_collect": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [
                                    {
                                        "ticker": "SBER",
                                        "asset_class": "equity",
                                        "market_value": 600.0,
                                    },
                                    {
                                        "ticker": "OFZ",
                                        "asset_class": "bond",
                                        "market_value": 400.0,
                                    },
                                ],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(
                            source_ts=datetime(2026, 4, 18, 10, 5, tzinfo=UTC),
                            coverage=0.95,
                            correlation_id="corr-qa-2",
                        ),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Какая доля облигаций в портфеле?",
            account_id="acc-1",
            correlation_id="corr-qa-2",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_QA,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertIn("40.0%", result.summary)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_show")
        self.assertEqual(gateway.calls[1]["tool_name"], "portfolio_collect")

    def test_portfolio_qa_returns_single_clarification_for_unknown_question(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [
                                    {
                                        "ticker": "SBER",
                                        "asset_class": "equity",
                                        "market_value": 600.0,
                                    },
                                    {
                                        "ticker": "OFZ",
                                        "asset_class": "bond",
                                        "market_value": 400.0,
                                    },
                                ],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-qa-3"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Какая текущая дюрация?",
            account_id="acc-1",
            correlation_id="corr-qa-3",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_QA,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.PARTIAL)
        self.assertEqual(result.evidence.unsupported_criteria, ["qa_question_not_supported"])
        clarifications = [item for item in result.details if item.startswith("clarification=")]
        self.assertEqual(len(clarifications), 1)

    def test_portfolio_qa_maps_collect_failure_to_unavailable(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="snapshot missing",
                        meta=ToolMeta(correlation_id="corr-qa-4"),
                    )
                ],
                "portfolio_collect": [
                    ToolResult.error(
                        code=ErrorCode.TIMEOUT,
                        message="upstream timeout",
                        retriable=True,
                        meta=ToolMeta(correlation_id="corr-qa-4"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Какая доля облигаций в портфеле?",
            account_id="acc-1",
            correlation_id="corr-qa-4",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_QA,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.UNAVAILABLE)
        self.assertEqual(result.errors[0].code, ErrorCode.TIMEOUT)

    def test_asset_analysis_returns_weight_for_ticker_from_snapshot(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [
                                    {"ticker": "SBER", "market_value": 650.0},
                                    {"ticker": "OFZ", "market_value": 350.0},
                                ],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-asset-1", coverage=0.9),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Сделай анализ актива SBER",
            account_id="acc-1",
            correlation_id="corr-asset-1",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.ASSET_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertIn("SBER", result.summary)
        self.assertIn("65.0%", result.summary)
        self.assertIn("stage=macro", result.details)
        self.assertIn("stage=sector", result.details)
        self.assertIn("stage=issuer", result.details)
        self.assertIn("stage=portfolio_fit", result.details)

    def test_asset_analysis_includes_macro_and_issuer_indicators_when_available(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [
                                    {"ticker": "SBER", "market_value": 500.0},
                                    {"ticker": "OFZ", "market_value": 500.0},
                                ],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-asset-signals"),
                    )
                ],
                "macro_indicators_show": [
                    ToolResult.ok(
                        data={
                            "indicators": [
                                {"key": "key_rate", "value": 16.0},
                                {"key": "imoex", "value": 3000.0},
                            ]
                        },
                        meta=ToolMeta(correlation_id="corr-asset-signals"),
                    )
                ],
                "issuer_indicators_show": [
                    ToolResult.ok(
                        data={
                            "issuer_indicators": {
                                "last": 250.0,
                                "close": 248.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-asset-signals"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Сделай анализ актива SBER с макро контекстом",
            account_id="acc-1",
            correlation_id="corr-asset-signals",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.ASSET_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertIn("key_rate", result.summary)
        self.assertIn("MOEX last", result.summary)
        self.assertIn("macro_indicators=imoex:3000.0000,key_rate:16.0000", result.details)
        self.assertIn("issuer_indicators=close:248.0000,last:250.0000", result.details)

    def test_asset_analysis_asks_single_clarification_when_ticker_missing(self) -> None:
        engine = WorkflowEngine(tool_gateway=_FakeGateway(responses={}))
        request = OrchestratorRequest(
            message="Сделай анализ актива",
            correlation_id="corr-asset-2",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.ASSET_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.PARTIAL)
        clarifications = [item for item in result.details if item.startswith("clarification=")]
        self.assertEqual(len(clarifications), 1)

    def test_asset_analysis_maps_data_unavailable_to_unavailable(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_show": [
                    ToolResult.error(
                        code=ErrorCode.UPSTREAM_UNAVAILABLE,
                        message="snapshot is stale",
                        meta=ToolMeta(correlation_id="corr-asset-3"),
                    )
                ],
                "portfolio_collect": [
                    ToolResult.error(
                        code=ErrorCode.TIMEOUT,
                        message="timeout",
                        retriable=True,
                        meta=ToolMeta(correlation_id="corr-asset-3"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Сделай анализ актива SBER",
            account_id="acc-1",
            correlation_id="corr-asset-3",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.ASSET_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.UNAVAILABLE)
        self.assertEqual(result.errors[0].code, ErrorCode.TIMEOUT)

    def test_portfolio_analysis_uses_aggregate_mode_by_default_when_many_accounts(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_accounts": [
                    ToolResult.ok(
                        data={"account_ids": ["acc-1", "acc-2"]},
                        meta=ToolMeta(correlation_id="corr-portfolio-accounts"),
                    )
                ],
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [{"ticker": "SBER"}, {"ticker": "OFZ"}],
                                "portfolio_value": 1500.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-portfolio-accounts"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Покажи общий анализ портфеля",
            correlation_id="corr-portfolio-accounts",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_accounts")
        self.assertEqual(gateway.calls[1]["tool_name"], "portfolio_show")
        self.assertEqual(gateway.calls[1]["payload"], {"account_id": None})

    def test_portfolio_analysis_requires_selection_for_specific_account_request(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_accounts": [
                    ToolResult.ok(
                        data={"account_ids": ["acc-1", "acc-2"]},
                        meta=ToolMeta(correlation_id="corr-portfolio-specific"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Покажи анализ портфеля по конкретному счету",
            correlation_id="corr-portfolio-specific",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.PARTIAL)
        self.assertIn("reason=account_selection_required", result.details)
        self.assertIn("account_id_missing", result.evidence.unsupported_criteria)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_accounts")

    def test_portfolio_analysis_uses_single_discovered_account_when_missing(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_accounts": [
                    ToolResult.ok(
                        data={"account_ids": ["acc-main"]},
                        meta=ToolMeta(correlation_id="corr-portfolio-single"),
                    )
                ],
                "portfolio_show": [
                    ToolResult.ok(
                        data={
                            "snapshot": {
                                "positions": [{"ticker": "SBER"}],
                                "portfolio_value": 1000.0,
                            }
                        },
                        meta=ToolMeta(correlation_id="corr-portfolio-single"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Покажи общий анализ портфеля",
            correlation_id="corr-portfolio-single",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_accounts")
        self.assertEqual(gateway.calls[1]["tool_name"], "portfolio_show")
        self.assertEqual(gateway.calls[1]["payload"], {"account_id": "acc-main"})

    def test_strategy_fit_uses_account_discovery_before_tool_when_missing_account(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_accounts": [
                    ToolResult.ok(
                        data={"account_ids": ["acc-main"]},
                        meta=ToolMeta(correlation_id="corr-fit-accounts"),
                    )
                ],
                "strategy_fit": [
                    ToolResult.ok(
                        data={"strategy_fit": {"verdict": "fit", "unsupported_criteria": []}},
                        meta=ToolMeta(correlation_id="corr-fit-accounts"),
                    )
                ],
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway)
        request = OrchestratorRequest(
            message="Проверь соответствие портфеля стратегии",
            correlation_id="corr-fit-accounts",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.STRATEGY_FIT,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(len(gateway.calls), 2)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_accounts")
        self.assertEqual(gateway.calls[1]["tool_name"], "strategy_fit")
        self.assertEqual(gateway.calls[1]["payload"], {"account_id": "acc-main"})
        self.assertIn("planner=bounded_v1", result.details)

    def test_portfolio_analysis_returns_error_when_planner_step_limit_reached(self) -> None:
        gateway = _FakeGateway(
            responses={
                "portfolio_accounts": [
                    ToolResult.ok(
                        data={"account_ids": ["acc-main"]},
                        meta=ToolMeta(correlation_id="corr-planner-limit"),
                    )
                ]
            }
        )
        engine = WorkflowEngine(tool_gateway=gateway, max_planner_steps=1)
        request = OrchestratorRequest(
            message="Покажи общий анализ портфеля",
            correlation_id="corr-planner-limit",
        )
        context = build_correlation_context(request, principal_id="local_user")

        result = engine.execute(
            request_class=RequestClass.PORTFOLIO_ANALYSIS,
            request=request,
            context=context,
        )

        self.assertEqual(result.status, ResponseStatus.ERROR)
        self.assertIn("reason=planner_step_limit_reached", result.details)
        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(gateway.calls[0]["tool_name"], "portfolio_accounts")


if __name__ == "__main__":
    unittest.main()
