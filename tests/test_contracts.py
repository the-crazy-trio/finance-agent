import unittest

from pydantic import ValidationError

from finance_agent.contracts.common import ErrorCode, ErrorDetail, ResponseStatus, ToolStatus
from finance_agent.contracts.entities import PortfolioPosition
from finance_agent.contracts.request_response import OrchestratorRequest, OrchestratorResponse
from finance_agent.contracts.tooling import ToolMeta, ToolResult


class ContractsTests(unittest.TestCase):
    def test_request_message_alias_is_supported(self) -> None:
        request = OrchestratorRequest(
            message="Покажи портфель",
        )
        self.assertEqual(request.message_text, "Покажи портфель")

    def test_partial_response_requires_reason(self) -> None:
        with self.assertRaises(ValidationError):
            OrchestratorResponse(
                request_id="req-1",
                status=ResponseStatus.PARTIAL,
                summary="Частичный ответ",
                correlation_id="corr-1",
            )

    def test_tool_error_helper_builds_error_payload(self) -> None:
        result = ToolResult.error(
            code=ErrorCode.TIMEOUT,
            message="Tool timeout",
            retriable=True,
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.TIMEOUT)

    def test_portfolio_position_requires_identifier(self) -> None:
        with self.assertRaises(ValidationError):
            PortfolioPosition(quantity=1, market_value=100)

    def test_error_detail_rejects_blank_message(self) -> None:
        with self.assertRaises(ValidationError):
            ErrorDetail(code=ErrorCode.TIMEOUT, message="   ")

    def test_tool_meta_rejects_blank_correlation_id(self) -> None:
        with self.assertRaises(ValidationError):
            ToolMeta(correlation_id="   ")


if __name__ == "__main__":
    unittest.main()
