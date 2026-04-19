from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from socket import timeout as SocketTimeout
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    text: str
    model_id: str
    usage: LlmUsage
    retries_used: int = 0


class LlmProviderError(RuntimeError):
    pass


class LlmClient(Protocol):
    def complete(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_ms: int,
        correlation_id: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> LlmCompletion:
        pass


class OpenRouterChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        max_retries: int = 2,
    ) -> None:
        key = api_key.strip()
        if not key:
            raise ValueError("api_key must not be blank")
        self._api_key = key
        self._endpoint = endpoint
        self._max_retries = max_retries if max_retries >= 0 else 0

    def complete(
        self,
        *,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        timeout_ms: int,
        correlation_id: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> LlmCompletion:
        if not model_id.strip():
            raise ValueError("model_id must not be blank")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        payload: dict[str, Any] = {
            "model": model_id,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        request_body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Correlation-Id": correlation_id,
        }

        for attempt in range(self._max_retries + 1):
            try:
                request = Request(
                    self._endpoint,
                    data=request_body,
                    headers=headers,
                    method="POST",
                )
                with urlopen(request, timeout=timeout_ms / 1000.0) as response:  # noqa: S310
                    response_body = response.read().decode("utf-8")
                response_payload = json.loads(response_body)
                return _parse_completion_payload(response_payload, retries_used=attempt)
            except HTTPError as error:
                message = _http_error_message(error)
                if _is_retriable_status(error.code) and attempt < self._max_retries:
                    continue
                raise LlmProviderError(message) from error
            except (URLError, SocketTimeout, TimeoutError, OSError) as error:
                if attempt < self._max_retries:
                    continue
                raise LlmProviderError(f"LLM provider is unavailable: {error}") from error
            except json.JSONDecodeError as error:
                raise LlmProviderError("LLM provider returned invalid JSON") from error

        raise LlmProviderError("LLM provider request failed after retries")


def is_placeholder_api_key(api_key: str) -> bool:
    normalized = api_key.strip().lower()
    if not normalized:
        return True
    return normalized.startswith("local-") or "placeholder" in normalized


def _parse_completion_payload(payload: Mapping[str, Any], *, retries_used: int) -> LlmCompletion:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmProviderError("LLM provider returned empty choices")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise LlmProviderError("LLM provider returned malformed choice payload")

    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise LlmProviderError("LLM provider returned malformed message payload")

    content = _read_content(message.get("content"))
    if not content:
        raise LlmProviderError("LLM provider returned empty content")

    model_id = str(payload.get("model") or "").strip() or "unknown"
    usage_payload = payload.get("usage")
    usage = _parse_usage(usage_payload)
    return LlmCompletion(
        text=content,
        model_id=model_id,
        usage=usage,
        retries_used=retries_used,
    )


def _read_content(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        return raw_value.strip()

    if isinstance(raw_value, list):
        parts: list[str] = []
        for item in raw_value:
            if not isinstance(item, Mapping):
                continue
            part = item.get("text")
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
        return "\n".join(parts).strip()

    return ""


def _parse_usage(raw_value: Any) -> LlmUsage:
    if not isinstance(raw_value, Mapping):
        return LlmUsage()

    prompt_tokens = _to_non_negative_int(raw_value.get("prompt_tokens"))
    completion_tokens = _to_non_negative_int(raw_value.get("completion_tokens"))
    return LlmUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _to_non_negative_int(raw_value: Any) -> int:
    if isinstance(raw_value, bool):
        return 0
    if isinstance(raw_value, int):
        return raw_value if raw_value >= 0 else 0
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return value if value >= 0 else 0


def _is_retriable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _http_error_message(error: HTTPError) -> str:
    body = ""
    try:
        body = error.read().decode("utf-8")
    except Exception:
        body = ""
    message = f"LLM provider request failed with HTTP {error.code}"
    if body.strip():
        return f"{message}: {body.strip()}"
    return message
