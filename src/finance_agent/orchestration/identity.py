from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from finance_agent.contracts.common import AuthMode


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    principal_id: str
    auth_mode: AuthMode


class IdentityResolver(Protocol):
    def resolve(self, metadata: Mapping[str, str] | None = None) -> ExecutionIdentity:
        pass


class LocalIdentityResolver:
    def __init__(self, principal_id: str = "local_user") -> None:
        self._principal_id = principal_id

    def resolve(self, metadata: Mapping[str, str] | None = None) -> ExecutionIdentity:
        return ExecutionIdentity(principal_id=self._principal_id, auth_mode=AuthMode.LOCAL)


class HeaderIdentityResolver:
    def __init__(self, header_name: str = "x-principal-id") -> None:
        self._header_name = header_name.lower()

    def resolve(self, metadata: Mapping[str, str] | None = None) -> ExecutionIdentity:
        metadata_map = {k.lower(): v for k, v in (metadata or {}).items()}
        principal_id = metadata_map.get(self._header_name, "").strip()
        if not principal_id:
            raise ValueError("Missing principal id in auth metadata")
        return ExecutionIdentity(principal_id=principal_id, auth_mode=AuthMode.API)
