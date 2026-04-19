from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from finance_agent.contracts.common import ErrorDetail, RequestClass, ResponseStatus


class OrchestratorEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of_ts: datetime | None = None
    coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_criteria: list[str] = Field(default_factory=list)


class OrchestratorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message_text: str = Field(min_length=1, alias="message")
    session_id: str | None = None
    channel: str | None = None
    account_id: str | None = None
    correlation_id: str | None = None
    request_id: str | None = None

    @field_validator(
        "message_text",
        "session_id",
        "channel",
        "account_id",
        "correlation_id",
        "request_id",
    )
    @classmethod
    def validate_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class OrchestratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    status: ResponseStatus
    summary: str = Field(min_length=1)
    details: list[str] = Field(default_factory=list)
    evidence: OrchestratorEvidence = Field(default_factory=OrchestratorEvidence)
    errors: list[ErrorDetail] = Field(default_factory=list)
    correlation_id: str = Field(min_length=1)
    request_class: RequestClass | None = None

    @field_validator("request_id", "summary", "correlation_id")
    @classmethod
    def validate_required_text_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped

    @model_validator(mode="after")
    def validate_partial_and_unavailable_reason(self) -> "OrchestratorResponse":
        if self.status in {ResponseStatus.PARTIAL, ResponseStatus.UNAVAILABLE}:
            has_reason = bool(self.details) or bool(self.errors)
            if not has_reason:
                raise ValueError("partial or unavailable response must include details or errors")
        return self
