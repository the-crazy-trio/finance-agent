from __future__ import annotations

from finance_agent.config.runtime import RuntimeConfig
from finance_agent.contracts.common import RequestClass


class ModelRoutingPolicy:
    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self._runtime_config = runtime_config

    def intent_model(self) -> str:
        return self._runtime_config.default_model_intent

    def synthesis_model(self, request_class: RequestClass) -> str:
        if request_class == RequestClass.STRATEGY_PARSING:
            return self._runtime_config.default_model_intent
        return self._runtime_config.default_model_synthesis

    def workflow_model(self, request_class: RequestClass) -> str:
        if request_class == RequestClass.STRATEGY_PARSING:
            return self._runtime_config.default_model_intent
        return self._runtime_config.default_model_synthesis

    def model_for_request(self, request_class: RequestClass, purpose: str) -> str:
        if purpose == "intent":
            return self.intent_model()
        if purpose == "synthesis":
            return self.synthesis_model(request_class)
        if purpose == "workflow":
            return self.workflow_model(request_class)
        raise ValueError("purpose must be 'intent', 'workflow', or 'synthesis'")
