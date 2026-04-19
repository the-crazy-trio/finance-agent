from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse


class OrchestratorHandle(Protocol):
    def handle(self, request: OrchestratorRequest) -> OrchestratorResponse:
        pass


@dataclass(frozen=True, slots=True)
class OfflineEvalCase:
    case_id: str
    message_text: str
    expected_request_class: RequestClass
    expected_statuses: tuple[ResponseStatus, ...]


@dataclass(frozen=True, slots=True)
class OfflineEvalResult:
    case_id: str
    passed: bool
    expected_request_class: RequestClass
    actual_request_class: RequestClass | None
    expected_statuses: tuple[ResponseStatus, ...]
    actual_status: ResponseStatus
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class OfflineEvalReport:
    total_cases: int
    passed_cases: int
    pass_rate: float
    partial_response_rate: float
    unsupported_disclosure_rate: float
    results: list[OfflineEvalResult]


def run_offline_eval(
    *,
    service: OrchestratorHandle,
    cases: Sequence[OfflineEvalCase],
) -> OfflineEvalReport:
    results: list[OfflineEvalResult] = []
    partial_count = 0
    unsupported_disclosure_count = 0

    for case in cases:
        response = service.handle(OrchestratorRequest(message=case.message_text, channel="eval"))
        class_ok = response.request_class == case.expected_request_class
        status_ok = response.status in case.expected_statuses
        passed = class_ok and status_ok

        reason: str | None = None
        if not class_ok:
            reason = "request_class_mismatch"
        elif not status_ok:
            reason = "status_mismatch"

        if response.status == ResponseStatus.PARTIAL:
            partial_count += 1
        if response.evidence.unsupported_criteria:
            unsupported_disclosure_count += 1

        results.append(
            OfflineEvalResult(
                case_id=case.case_id,
                passed=passed,
                expected_request_class=case.expected_request_class,
                actual_request_class=response.request_class,
                expected_statuses=case.expected_statuses,
                actual_status=response.status,
                failure_reason=reason,
            )
        )

    total = len(cases)
    passed_cases = sum(1 for item in results if item.passed)
    return OfflineEvalReport(
        total_cases=total,
        passed_cases=passed_cases,
        pass_rate=(passed_cases / total) if total else 0.0,
        partial_response_rate=(partial_count / total) if total else 0.0,
        unsupported_disclosure_rate=(unsupported_disclosure_count / total) if total else 0.0,
        results=results,
    )


def build_default_offline_cases() -> list[OfflineEvalCase]:
    return [
        OfflineEvalCase(
            case_id="strategy_parsing_clarification",
            message_text="Обнови стратегию",
            expected_request_class=RequestClass.STRATEGY_PARSING,
            expected_statuses=(ResponseStatus.PARTIAL,),
        ),
        OfflineEvalCase(
            case_id="portfolio_analysis_basic",
            message_text="Покажи общий анализ портфеля",
            expected_request_class=RequestClass.PORTFOLIO_ANALYSIS,
            expected_statuses=(ResponseStatus.OK, ResponseStatus.PARTIAL),
        ),
        OfflineEvalCase(
            case_id="strategy_fit_no_gateway",
            message_text="Проверь соответствие портфеля стратегии",
            expected_request_class=RequestClass.STRATEGY_FIT,
            expected_statuses=(ResponseStatus.UNAVAILABLE,),
        ),
        OfflineEvalCase(
            case_id="portfolio_qa_basic",
            message_text="Какая доля облигаций в портфеле?",
            expected_request_class=RequestClass.PORTFOLIO_QA,
            expected_statuses=(ResponseStatus.PARTIAL,),
        ),
    ]
