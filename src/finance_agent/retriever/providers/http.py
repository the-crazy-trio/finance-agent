from __future__ import annotations

import json
from collections.abc import Mapping
from socket import timeout as SocketTimeout
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderHttpError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def fetch_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    body = fetch_text(url, params=params, timeout_sec=timeout_sec)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderHttpError("provider returned invalid JSON") from error

    if not isinstance(payload, dict):
        raise ProviderHttpError("provider returned non-object JSON payload")
    return payload


def fetch_text(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> str:
    query = f"?{urlencode(params)}" if params else ""
    full_url = f"{url}{query}"
    request = Request(full_url, method="GET")
    try:
        with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="ignore")
    except HTTPError as error:
        raise ProviderHttpError(
            f"provider returned HTTP {error.code}",
            status_code=error.code,
        ) from error
    except (URLError, TimeoutError, SocketTimeout, OSError) as error:
        raise ProviderHttpError(f"provider request failed: {error}") from error
