from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from finance_agent.client.cli import run_cli_request


class CliInterfaceTests(unittest.TestCase):
    def test_run_cli_request_returns_unified_response_schema(self) -> None:
        result = run_cli_request(
            message_text="Привет",
            session_id="session-1",
            correlation_id="corr-cli-1",
        )

        self.assertEqual(result["correlation_id"], "corr-cli-1")
        self.assertIn(result["status"], {"ok", "partial", "unavailable", "error"})
        self.assertIn("summary", result)
        self.assertIn("details", result)
        self.assertIn("evidence", result)
        self.assertIn("errors", result)

    def test_run_cli_request_persists_state_when_state_path_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "cli_state.json")
            first_result = run_cli_request(
                message_text="Привет",
                local_state_path=state_path,
            )
            self.assertEqual(first_result["status"], "ok")
            self.assertTrue(os.path.exists(state_path))

            stored_payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
            first_count = len(stored_payload.get("workflow_summaries", {}).get("local_user", []))
            self.assertGreaterEqual(first_count, 1)

            second_result = run_cli_request(
                message_text="Привет ещё раз",
                local_state_path=state_path,
            )
            self.assertEqual(second_result["status"], "ok")

            stored_payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
            second_count = len(stored_payload.get("workflow_summaries", {}).get("local_user", []))
            self.assertGreaterEqual(second_count, first_count + 1)


if __name__ == "__main__":
    unittest.main()
