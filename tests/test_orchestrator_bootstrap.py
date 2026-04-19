from __future__ import annotations

import unittest

from finance_agent.contracts.request_response import OrchestratorRequest
from finance_agent.orchestration import build_local_orchestrator_service


class OrchestratorBootstrapTests(unittest.TestCase):
    def test_local_builder_works_without_runtime_secrets(self) -> None:
        service = build_local_orchestrator_service(env={})
        response = service.handle(OrchestratorRequest(message="Привет", channel="cli"))

        self.assertEqual(response.status.value, "ok")
        self.assertIn("principal_id=local_user", response.details)

    def test_local_builder_wires_context_loading_and_persistence(self) -> None:
        service = build_local_orchestrator_service(env={})
        response = service.handle(OrchestratorRequest(message="Покажи портфель", channel="cli"))

        self.assertIn("context_load=memory", response.details)
        self.assertIn("persist=memory_saved", response.details)


if __name__ == "__main__":
    unittest.main()
