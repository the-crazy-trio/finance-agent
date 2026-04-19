from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from finance_agent.config.model_routing import ModelRoutingPolicy
from finance_agent.contracts.common import RequestClass
from finance_agent.llm import LlmClient, LlmProviderError

_INTENT_PROMPT_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class IntentClassification:
    request_class: RequestClass
    model_id: str | None = None
    prompt_version: str = _INTENT_PROMPT_VERSION
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    retry_count: int = 0
    fallback_reason: str | None = None


class DeterministicIntentClassifier:
    def __init__(self, classify_fn: Callable[[str], RequestClass]) -> None:
        self._classify_fn = classify_fn

    def classify(self, *, message_text: str, correlation_id: str) -> IntentClassification:
        del correlation_id
        return IntentClassification(request_class=self._classify_fn(message_text))


class LlmIntentClassifier:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        model_routing: ModelRoutingPolicy,
        fallback_classifier: DeterministicIntentClassifier,
        tokens_max: int,
        llm_timeout_ms: int,
    ) -> None:
        self._llm_client = llm_client
        self._model_routing = model_routing
        self._fallback_classifier = fallback_classifier
        self._tokens_max = tokens_max
        self._llm_timeout_ms = llm_timeout_ms

    def classify(self, *, message_text: str, correlation_id: str) -> IntentClassification:
        model_id = self._model_routing.intent_model()
        try:
            completion = self._llm_client.complete(
                model_id=model_id,
                system_prompt=_intent_system_prompt(),
                user_prompt=_intent_user_prompt(message_text),
                max_tokens=self._tokens_max,
                timeout_ms=self._llm_timeout_ms,
                correlation_id=correlation_id,
                response_format={"type": "json_object"},
            )
            parsed_request_class = _parse_request_class(completion.text)
            if parsed_request_class is None:
                raise ValueError("request_class is missing or invalid")
            return IntentClassification(
                request_class=parsed_request_class,
                model_id=completion.model_id,
                llm_tokens_in=completion.usage.prompt_tokens,
                llm_tokens_out=completion.usage.completion_tokens,
                retry_count=completion.retries_used,
            )
        except (LlmProviderError, ValueError, json.JSONDecodeError) as error:
            fallback = self._fallback_classifier.classify(
                message_text=message_text,
                correlation_id=correlation_id,
            )
            return IntentClassification(
                request_class=fallback.request_class,
                model_id=model_id,
                llm_tokens_in=0,
                llm_tokens_out=0,
                retry_count=0,
                fallback_reason=str(error),
            )


def _intent_system_prompt() -> str:
    classes = ", ".join(item.value for item in RequestClass)
    return (
        "You classify a user request into exactly one workflow class for a financial assistant. "
        "Return strict JSON object with field request_class only. "
        f"Allowed values: {classes}."
    )


def _intent_user_prompt(message_text: str) -> str:
    return (
        "Classify this message into one request_class. "
        "Do not add explanations.\n"
        f"message: {message_text}"
    )


def _parse_request_class(content: str) -> RequestClass | None:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        return None
    raw_value = payload.get("request_class")
    if not isinstance(raw_value, str):
        return None
    candidate = raw_value.strip()
    if not candidate:
        return None
    try:
        return RequestClass(candidate)
    except ValueError:
        return None
