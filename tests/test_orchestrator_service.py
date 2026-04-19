import unittest
from datetime import UTC, datetime

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass, ResponseStatus
from finance_agent.contracts.request_response import OrchestratorEvidence, OrchestratorRequest
from finance_agent.orchestration import (
    HeaderIdentityResolver,
    OrchestratorService,
    OrchestratorValidationGate,
)
from finance_agent.workflow import WorkflowExecutionResult


class _StubWorkflowEngine:
    def __init__(self, result: WorkflowExecutionResult) -> None:
        self._result = result

    def execute(self, **kwargs: object) -> WorkflowExecutionResult:
        return self._result


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


class OrchestratorServiceTests(unittest.TestCase):
    def test_local_mode_without_auth_metadata(self) -> None:
        service = OrchestratorService()
        request = OrchestratorRequest(message="Привет", channel="cli")

        response = service.handle(request)

        self.assertEqual(response.status.value, "ok")
        self.assertTrue(response.correlation_id.startswith("corr_"))
        self.assertIn("auth_mode=local", response.details)
        self.assertIn("synthesis=deterministic_stub", response.details)
        self.assertIn("persist=noop", response.details)
        self.assertIsNotNone(response.request_class)

    def test_api_mode_with_header_identity(self) -> None:
        service = OrchestratorService(identity_resolver=HeaderIdentityResolver())
        request = OrchestratorRequest(message="Привет", channel="api")

        response = service.handle(request, auth_metadata={"x-principal-id": "u-1"})

        self.assertEqual(response.status.value, "ok")
        self.assertIn("principal_id=u-1", response.details)
        self.assertIn("auth_mode=api", response.details)
        self.assertIsNotNone(response.request_class)

    def test_validation_gate_is_applied_in_service(self) -> None:
        now = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
        stale_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="ok",
            evidence=OrchestratorEvidence(
                as_of_ts=datetime(2026, 4, 18, 9, 0, tzinfo=UTC),
                coverage=1.0,
                unsupported_criteria=[],
            ),
        )
        service = OrchestratorService(
            request_classifier=lambda _: RequestClass.PORTFOLIO_ANALYSIS,
            workflow_engine=_StubWorkflowEngine(stale_result),
            validation_gate=OrchestratorValidationGate(
                portfolio_snapshot_max_age_min=60,
                now_fn=lambda: now,
            ),
        )
        request = OrchestratorRequest(message="Покажи портфель", channel="cli")

        response = service.handle(request)

        self.assertEqual(response.status.value, "unavailable")
        self.assertEqual(response.errors[0].code.value, "UPSTREAM_UNAVAILABLE")

    def test_runtime_config_drives_default_validation_gate(self) -> None:
        runtime_config = RuntimeConfig.from_env(
            {
                **build_minimum_env(),
                "PORTFOLIO_SNAPSHOT_MAX_AGE_MIN": "1",
            }
        )
        stale_result = WorkflowExecutionResult(
            status=ResponseStatus.OK,
            summary="ok",
            evidence=OrchestratorEvidence(
                as_of_ts=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
                coverage=1.0,
                unsupported_criteria=[],
            ),
        )
        service = OrchestratorService(
            request_classifier=lambda _: RequestClass.PORTFOLIO_ANALYSIS,
            workflow_engine=_StubWorkflowEngine(stale_result),
            runtime_config=runtime_config,
        )
        request = OrchestratorRequest(message="Покажи портфель", channel="cli")

        response = service.handle(request)

        self.assertEqual(response.status.value, "unavailable")
        self.assertEqual(response.errors[0].code.value, "UPSTREAM_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
