from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from finance_agent.contracts.common import ErrorCode, ErrorDetail, ToolStatus


class ToolMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms: int = Field(default=0, ge=0)
    source_ts: datetime | None = None
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    correlation_id: str | None = None

    @field_validator("correlation_id")
    @classmethod
    def validate_correlation_id_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[ErrorDetail] = Field(default_factory=list)
    meta: ToolMeta = Field(default_factory=ToolMeta)

    @model_validator(mode="after")
    def validate_error_payload(self) -> "ToolResult":
        if self.status == ToolStatus.ERROR and not self.errors:
            raise ValueError("error status requires at least one error item")
        return self

    @classmethod
    def ok(cls, data: dict[str, Any], meta: ToolMeta | None = None) -> "ToolResult":
        return cls(status=ToolStatus.OK, data=data, meta=meta or ToolMeta())

    @classmethod
    def partial(
        cls,
        data: dict[str, Any],
        errors: list[ErrorDetail] | None = None,
        meta: ToolMeta | None = None,
    ) -> "ToolResult":
        return cls(
            status=ToolStatus.PARTIAL,
            data=data,
            errors=errors or [],
            meta=meta or ToolMeta(),
        )

    @classmethod
    def error(
        cls,
        code: ErrorCode,
        message: str,
        retriable: bool = False,
        details: dict[str, Any] | None = None,
        meta: ToolMeta | None = None,
    ) -> "ToolResult":
        error = ErrorDetail(
            code=code,
            message=message,
            retriable=retriable,
            details=details,
        )
        return cls(
            status=ToolStatus.ERROR,
            data={},
            errors=[error],
            meta=meta or ToolMeta(),
        )
