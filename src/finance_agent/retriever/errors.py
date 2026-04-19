from __future__ import annotations

from enum import StrEnum

from finance_agent.contracts.common import ErrorCode, ErrorDetail


class TInvestErrorCategory(StrEnum):
    INVALID_ARGUMENT = "invalid_argument"
    UNAUTHENTICATED = "unauthenticated"
    PERMISSION_DENIED = "permission_denied"
    ACCOUNT_NOT_FOUND = "account_not_found"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INTERNAL = "internal"


class TInvestClientError(RuntimeError):
    def __init__(self, category: TInvestErrorCategory | str, message: str) -> None:
        self.category = _coerce_category(category)
        super().__init__(message)


def normalize_tinvest_error(
    category: TInvestErrorCategory | str,
    message: str,
) -> ErrorDetail:
    normalized_category = _coerce_category(category)
    code, retriable = _category_mapping(normalized_category)
    safe_message = message.strip() or "T-Invest request failed"

    return ErrorDetail(
        code=code,
        message=safe_message,
        retriable=retriable,
        details={"upstream_category": normalized_category.value},
    )


def _coerce_category(category: TInvestErrorCategory | str) -> TInvestErrorCategory:
    if isinstance(category, TInvestErrorCategory):
        return category

    normalized = category.strip().lower()
    aliases = {
        "invalid argument": TInvestErrorCategory.INVALID_ARGUMENT,
        "bad request": TInvestErrorCategory.INVALID_ARGUMENT,
        "invalid_argument": TInvestErrorCategory.INVALID_ARGUMENT,
        "unauthenticated": TInvestErrorCategory.UNAUTHENTICATED,
        "invalid token": TInvestErrorCategory.UNAUTHENTICATED,
        "permission denied": TInvestErrorCategory.PERMISSION_DENIED,
        "permission_denied": TInvestErrorCategory.PERMISSION_DENIED,
        "account not found": TInvestErrorCategory.ACCOUNT_NOT_FOUND,
        "account_not_found": TInvestErrorCategory.ACCOUNT_NOT_FOUND,
        "not found": TInvestErrorCategory.NOT_FOUND,
        "not_found": TInvestErrorCategory.NOT_FOUND,
        "rate limit": TInvestErrorCategory.RATE_LIMIT,
        "too many requests": TInvestErrorCategory.RATE_LIMIT,
        "rate_limit": TInvestErrorCategory.RATE_LIMIT,
        "deadline exceeded": TInvestErrorCategory.TIMEOUT,
        "timeout": TInvestErrorCategory.TIMEOUT,
        "upstream unavailable": TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        "unavailable": TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        "upstream_unavailable": TInvestErrorCategory.UPSTREAM_UNAVAILABLE,
        "internal": TInvestErrorCategory.INTERNAL,
        "unknown": TInvestErrorCategory.INTERNAL,
    }
    return aliases.get(normalized, TInvestErrorCategory.INTERNAL)


def _category_mapping(category: TInvestErrorCategory) -> tuple[ErrorCode, bool]:
    mapping = {
        TInvestErrorCategory.INVALID_ARGUMENT: (ErrorCode.VALIDATION_ERROR, False),
        TInvestErrorCategory.UNAUTHENTICATED: (ErrorCode.UNAUTHORIZED_SCOPE, False),
        TInvestErrorCategory.PERMISSION_DENIED: (ErrorCode.UNAUTHORIZED_SCOPE, False),
        TInvestErrorCategory.ACCOUNT_NOT_FOUND: (ErrorCode.UPSTREAM_UNAVAILABLE, False),
        TInvestErrorCategory.NOT_FOUND: (ErrorCode.UPSTREAM_UNAVAILABLE, False),
        TInvestErrorCategory.RATE_LIMIT: (ErrorCode.RATE_LIMITED, True),
        TInvestErrorCategory.TIMEOUT: (ErrorCode.TIMEOUT, True),
        TInvestErrorCategory.UPSTREAM_UNAVAILABLE: (ErrorCode.UPSTREAM_UNAVAILABLE, True),
        TInvestErrorCategory.INTERNAL: (ErrorCode.UPSTREAM_UNAVAILABLE, True),
    }
    return mapping[category]
