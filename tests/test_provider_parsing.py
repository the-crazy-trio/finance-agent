from __future__ import annotations

import unittest
from typing import Any

from finance_agent.retriever.providers.cbr_api import CbrApiProvider
from finance_agent.retriever.providers.http import ProviderHttpError
from finance_agent.retriever.providers.moex_api import MoexApiProvider


class ProviderParsingTests(unittest.TestCase):
    def test_cbr_daily_info_falls_back_to_html_scrape(self) -> None:
        def fake_json_fetcher(url: str, **_: Any) -> dict[str, Any]:
            raise ProviderHttpError(f"unavailable: {url}")

        html = """
        <div class="indicator">
            <div class="title"><a href="/statistics/inf">Инфляция</a></div>
            <div class="value">5,9%</div>
        </div>
        <div class="indicator">
            <div class="title"><a href="/hd_base/KeyRate/">Ключевая ставка</a></div>
            <div class="value">15,00%</div>
        </div>
        """

        provider = CbrApiProvider(
            fetcher=fake_json_fetcher,
            text_fetcher=lambda *_args, **_kwargs: html,
        )

        payload = provider.get_daily_info()

        self.assertAlmostEqual(float(payload["key_rate"]), 15.0)
        self.assertAlmostEqual(float(payload["inflation"]), 5.9)
        self.assertIsNotNone(provider.extract_key_rate(payload))
        self.assertIsNotNone(provider.extract_inflation(payload))

    def test_moex_selects_best_market_row_when_first_row_is_empty(self) -> None:
        def fake_fetcher(url: str, **_: Any) -> dict[str, Any]:
            if "USD000UTSTOM" not in url:
                raise AssertionError("unexpected security request")
            return {
                "marketdata": {
                    "columns": [
                        "BOARDID",
                        "LAST",
                        "CURRENTVALUE",
                        "LASTVALUE",
                        "OPEN",
                        "HIGH",
                        "LOW",
                        "TRADINGSTATUS",
                    ],
                    "data": [
                        ["AUCB", None, None, None, None, None, None, "C"],
                        ["CETS", 75.65, None, None, 75.9, 76.2, 75.5, "N"],
                    ],
                },
                "securities": {
                    "columns": ["SHORTNAME"],
                    "data": [["USDRUB_TOM"]],
                },
            }

        provider = MoexApiProvider(fetcher=fake_fetcher)
        payload = provider.get_usd_rub()

        self.assertIsNotNone(payload)
        self.assertEqual(payload["board_id"], "CETS")
        self.assertAlmostEqual(float(payload["last"]), 75.65)

    def test_moex_uses_currentvalue_when_last_is_missing_for_index(self) -> None:
        def fake_fetcher(url: str, **_: Any) -> dict[str, Any]:
            if "IMOEX" not in url:
                raise AssertionError("unexpected security request")
            return {
                "marketdata": {
                    "columns": ["BOARDID", "CURRENTVALUE", "LASTVALUE", "OPENVALUE"],
                    "data": [["SNDX", 2723.94, 2741.14, 2741.81]],
                },
                "securities": {
                    "columns": ["SHORTNAME"],
                    "data": [["Индекс МосБиржи"]],
                },
            }

        provider = MoexApiProvider(fetcher=fake_fetcher)
        payload = provider.get_imoex()

        self.assertIsNotNone(payload)
        self.assertAlmostEqual(float(payload["last"]), 2723.94)
        self.assertAlmostEqual(float(payload["close"]), 2741.14)
        self.assertAlmostEqual(float(payload["open"]), 2741.81)


if __name__ == "__main__":
    unittest.main()
