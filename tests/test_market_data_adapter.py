from __future__ import annotations

import unittest
from datetime import UTC, datetime

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import ErrorCode, ToolStatus
from finance_agent.retriever.market_adapter import ExternalMarketDataAdapter


def build_minimum_env() -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "openrouter-key",
        "TINVEST_READONLY_TOKEN": "readonly-token",
    }


class _FakeCbrProvider:
    def get_daily_info(self) -> dict[str, object]:
        return {
            "key_rate": 16.0,
            "inflation": 7.5,
            "gdp": 2.1,
        }

    def extract_key_rate(self, payload: dict[str, object]) -> dict[str, object] | None:
        value = payload.get("key_rate")
        if not isinstance(value, (int, float)):
            return None
        return {"key": "key_rate", "value": float(value)}

    def extract_inflation(self, payload: dict[str, object]) -> dict[str, object] | None:
        value = payload.get("inflation")
        if not isinstance(value, (int, float)):
            return None
        return {"key": "inflation", "value": float(value)}

    def extract_gdp(self, payload: dict[str, object]) -> dict[str, object] | None:
        value = payload.get("gdp")
        if not isinstance(value, (int, float)):
            return None
        return {"key": "gdp", "value": float(value)}


class _FakeMoexProvider:
    def get_imoex(self) -> dict[str, object] | None:
        return {
            "last": 3000.0,
            "source_timestamp": datetime(2026, 4, 19, 10, 0, tzinfo=UTC).isoformat(),
        }

    def get_rtsi(self) -> dict[str, object] | None:
        return {
            "last": 1100.0,
            "source_timestamp": datetime(2026, 4, 19, 10, 0, tzinfo=UTC).isoformat(),
        }

    def get_usd_rub(self) -> dict[str, object] | None:
        return {
            "last": 95.0,
            "source_timestamp": datetime(2026, 4, 19, 10, 0, tzinfo=UTC).isoformat(),
        }

    def resolve_security(self, *, security: str) -> dict[str, object] | None:
        if security == "UNKNOWN":
            return None
        return {
            "security": security,
            "last": 250.0,
            "close": 248.0,
            "open": 249.0,
            "high": 252.0,
            "low": 247.5,
            "volume": 1000.0,
            "value": 250000.0,
            "source_timestamp": datetime(2026, 4, 19, 10, 0, tzinfo=UTC).isoformat(),
            "source_name": "moex_iss",
            "source_type": "market",
        }


class MarketDataAdapterTests(unittest.TestCase):
    def test_macro_indicators_returns_partial_when_some_keys_missing(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        adapter = ExternalMarketDataAdapter(
            runtime_config=runtime_config,
            cbr_provider=_FakeCbrProvider(),
            moex_provider=_FakeMoexProvider(),
        )

        result = adapter.show_macro_indicators(
            keys=["key_rate", "imoex", "unknown_metric"],
            correlation_id="corr-macro-test",
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.VALIDATION_ERROR)

    def test_macro_indicators_returns_ok_for_default_keys(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        adapter = ExternalMarketDataAdapter(
            runtime_config=runtime_config,
            cbr_provider=_FakeCbrProvider(),
            moex_provider=_FakeMoexProvider(),
        )

        result = adapter.show_macro_indicators(keys=None, correlation_id="corr-macro-default")

        self.assertIn(result.status, {ToolStatus.OK, ToolStatus.PARTIAL})
        self.assertIn("indicators", result.data)
        self.assertGreaterEqual(len(result.data["indicators"]), 3)

    def test_issuer_indicators_returns_ok_for_known_ticker(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        adapter = ExternalMarketDataAdapter(
            runtime_config=runtime_config,
            cbr_provider=_FakeCbrProvider(),
            moex_provider=_FakeMoexProvider(),
        )

        result = adapter.show_issuer_indicators(
            ticker="SBER",
            correlation_id="corr-issuer-ok",
        )

        self.assertIn(result.status, {ToolStatus.OK, ToolStatus.PARTIAL})
        self.assertEqual(result.data["ticker"], "SBER")
        self.assertIn("issuer_indicators", result.data)

    def test_issuer_indicators_returns_mapping_not_found_for_unknown_ticker(self) -> None:
        runtime_config = RuntimeConfig.from_env(build_minimum_env())
        adapter = ExternalMarketDataAdapter(
            runtime_config=runtime_config,
            cbr_provider=_FakeCbrProvider(),
            moex_provider=_FakeMoexProvider(),
        )

        result = adapter.show_issuer_indicators(
            ticker="UNKNOWN",
            correlation_id="corr-issuer-missing",
        )

        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.errors[0].code, ErrorCode.MAPPING_NOT_FOUND)


if __name__ == "__main__":
    unittest.main()
