import unittest

from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.orchestration.context import build_correlation_context


class OrchestrationContextTests(unittest.TestCase):
    def test_uses_incoming_correlation_id_when_present(self) -> None:
        request = OrchestratorRequest(
            session_id="session-1",
            message="Обнови аналитику",
            request_id="req-1",
            correlation_id="corr-1",
        )

        context = build_correlation_context(request, principal_id="local_user")
        self.assertEqual(context.correlation_id, "corr-1")
        self.assertEqual(context.request_id, "req-1")
        self.assertEqual(context.principal_id, "local_user")

    def test_generates_ids_when_absent(self) -> None:
        request = OrchestratorRequest(
            session_id="session-2",
            message="Покажи fit",
        )

        context = build_correlation_context(request, principal_id="local_user")
        self.assertTrue(context.correlation_id.startswith("corr_"))
        self.assertTrue(context.request_id.startswith("req_"))


if __name__ == "__main__":
    unittest.main()
