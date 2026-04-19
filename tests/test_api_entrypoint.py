import asyncio
import unittest

from finance_agent.contracts.request_response import OrchestratorRequest
from main import orchestrate


class ApiEntrypointTests(unittest.TestCase):
    def test_header_correlation_id_is_used_when_body_missing(self) -> None:
        request = OrchestratorRequest(message="Покажи портфель", channel="api")

        response = asyncio.run(
            orchestrate(
                request=request,
                x_principal_id=None,
                x_correlation_id="corr_header_123",
            )
        )

        self.assertEqual(response.correlation_id, "corr_header_123")


if __name__ == "__main__":
    unittest.main()
