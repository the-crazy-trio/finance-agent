from __future__ import annotations

from collections.abc import Mapping

from finance_agent.retriever.tinvest_adapter import TInvestReadonlyClient
from finance_agent.retriever.tinvest_sdk_client import TInvestSdkReadonlyClient


def build_tinvest_readonly_client_from_env(
    source: Mapping[str, str],
) -> TInvestReadonlyClient | None:
    token = source.get("TINVEST_READONLY_TOKEN", "").strip()
    if not token:
        return None

    target = source.get("TINVEST_GRPC_TARGET", "").strip() or None
    ca_bundle = source.get("TINVEST_CA_BUNDLE", "").strip() or None
    return TInvestSdkReadonlyClient(
        token=token,
        target=target,
        ca_bundle_path=ca_bundle,
    )
