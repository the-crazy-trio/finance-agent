from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestClass(StrEnum):
    STRATEGY_PARSING = "strategy_parsing"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    STRATEGY_FIT = "strategy_fit"
    ASSET_ANALYSIS = "asset_analysis"
    PORTFOLIO_QA = "portfolio_qa"


class ResponseStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ToolStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED_SCOPE = "UNAUTHORIZED_SCOPE"
    TIMEOUT = "TIMEOUT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    MAPPING_NOT_FOUND = "MAPPING_NOT_FOUND"


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str = Field(min_length=1)
    retriable: bool = False
    details: dict[str, Any] | None = None

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class AuthMode(StrEnum):
    LOCAL = "local"
    API = "api"


class CorrelationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    auth_mode: AuthMode = AuthMode.LOCAL
    request_id: str | None = None
    session_id: str | None = None

    @field_validator("correlation_id", "principal_id", "request_id", "session_id")
    @classmethod
    def validate_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped
