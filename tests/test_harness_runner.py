import unittest

from finance_agent.harness.local_runner import run_harness_request


class HarnessRunnerTests(unittest.TestCase):
    def test_harness_runner_uses_single_user_profile(self) -> None:
        result = run_harness_request({"message": "Проверь портфель", "channel": "mcp"})

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["correlation_id"].startswith("corr_"))
        self.assertIn("principal_id=local_user", result["details"])


if __name__ == "__main__":
    unittest.main()
