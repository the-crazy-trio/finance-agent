from __future__ import annotations

import re

from finance_agent.contracts.common import RequestClass

_STRATEGY_FIT_MARKERS = (
    "strategy fit",
    "соответ",
    "подходит",
    "fit",
)

_STRATEGY_PARSING_MARKERS = (
    "стратег",
    "strategy",
    "риск-проф",
    "аллокац",
)

_ASSET_ANALYSIS_MARKERS = (
    "актив",
    "бумаг",
    "тикер",
    "issuer",
    "эмитент",
)

_PORTFOLIO_MARKERS = (
    "портфел",
    "portfolio",
)

_QUESTION_MARKERS = (
    "почему",
    "зачем",
    "какая",
    "какой",
    "какие",
    "сколько",
    "когда",
    "где",
)


def determine_request_class(message_text: str) -> RequestClass:
    normalized = _normalize(message_text)

    if _contains_any(normalized, _STRATEGY_FIT_MARKERS):
        return RequestClass.STRATEGY_FIT

    if _contains_any(normalized, _STRATEGY_PARSING_MARKERS):
        return RequestClass.STRATEGY_PARSING

    if _contains_any(normalized, _ASSET_ANALYSIS_MARKERS):
        return RequestClass.ASSET_ANALYSIS

    has_portfolio_marker = _contains_any(normalized, _PORTFOLIO_MARKERS)
    if has_portfolio_marker and _looks_like_question(normalized):
        return RequestClass.PORTFOLIO_QA

    if has_portfolio_marker:
        return RequestClass.PORTFOLIO_ANALYSIS

    return RequestClass.PORTFOLIO_ANALYSIS


def _normalize(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return collapsed


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_question(text: str) -> bool:
    if "?" in text:
        return True
    return _contains_any(text, _QUESTION_MARKERS)
