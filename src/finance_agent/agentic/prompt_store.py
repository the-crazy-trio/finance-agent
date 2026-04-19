from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from finance_agent.contracts.common import RequestClass


class SubagentKind(StrEnum):
    WORKFLOW_AGENT = "workflow_agent"
    RESPONSE_SYNTHESIS = "response_synthesis"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    prompt_id: str
    version: str
    subagent_id: str
    kind: SubagentKind
    system_prompt: str
    guidelines: tuple[str, ...]


class PromptTemplateStore:
    def __init__(self, templates: Iterable[PromptTemplate]) -> None:
        index: dict[tuple[SubagentKind, str], PromptTemplate] = {}
        for template in templates:
            key = (template.kind, template.subagent_id)
            if key in index:
                raise ValueError(
                    f"duplicate prompt template for kind={template.kind}"
                    f" subagent_id={template.subagent_id}"
                )
            index[key] = template
        self._index = index

    def resolve(
        self,
        *,
        kind: SubagentKind,
        request_class: RequestClass,
    ) -> PromptTemplate:
        specific_key = (kind, request_class.value)
        if specific_key in self._index:
            return self._index[specific_key]

        global_key = (kind, "global")
        if global_key in self._index:
            return self._index[global_key]

        raise KeyError(
            f"Prompt template is not configured for kind={kind}"
            f" and request_class={request_class.value}"
        )

    def has_template(
        self,
        *,
        kind: SubagentKind,
        subagent_id: str,
    ) -> bool:
        return (kind, subagent_id) in self._index


def build_default_prompt_store() -> PromptTemplateStore:
    templates: list[PromptTemplate] = []

    templates.append(
        PromptTemplate(
            prompt_id="workflow-agent-global",
            version="1.0.0",
            subagent_id="global",
            kind=SubagentKind.WORKFLOW_AGENT,
            system_prompt=(
                "You are a bounded financial workflow agent. Use only guarded tools, "
                "preserve correlation_id, respect principal scope, and stop when "
                "evidence is sufficient or limits are reached."
            ),
            guidelines=(
                "Use only allowlisted guarded tools.",
                "Do not fabricate data; disclose unsupported criteria.",
                (
                    "For portfolio/strategy-fit flows, call portfolio_accounts first, "
                    "show account_ids to user, then continue with selected account_id."
                ),
                "Ask at most one clarification question when required slots are missing.",
                "Respect timeout and max-step stop conditions.",
            ),
        )
    )

    templates.extend(_workflow_agent_templates())

    templates.append(
        PromptTemplate(
            prompt_id="response-synthesis-global",
            version="1.0.0",
            subagent_id="global",
            kind=SubagentKind.RESPONSE_SYNTHESIS,
            system_prompt=(
                "You are a financial response synthesizer. Summarize validated workflow "
                "results in Russian, keep evidence and limitations explicit."
            ),
            guidelines=(
                "Never override validation-gate outcomes.",
                "Keep unsupported criteria visible.",
                "Include not-investment-advice disclaimer.",
            ),
        )
    )

    templates.append(
        PromptTemplate(
            prompt_id="response-synthesis-strategy-parsing",
            version="1.0.0",
            subagent_id=RequestClass.STRATEGY_PARSING.value,
            kind=SubagentKind.RESPONSE_SYNTHESIS,
            system_prompt=(
                "Synthesize strategy parsing outcomes with explicit missing slots or "
                "saved strategy confirmation."
            ),
            guidelines=(
                "If partial, include exactly one clarification question.",
                "If saved, mention risk profile and target allocation summary.",
            ),
        )
    )

    return PromptTemplateStore(templates)


def _workflow_agent_templates() -> list[PromptTemplate]:
    templates: list[PromptTemplate] = []
    for request_class in RequestClass:
        guidelines = [
            "Keep chain-of-thought private; output only decisions and rationale summary.",
            "Do not access raw data outside guarded tools.",
            "Return partial/unavailable with explicit reason when needed.",
        ]
        if request_class in {
            RequestClass.PORTFOLIO_ANALYSIS,
            RequestClass.STRATEGY_FIT,
            RequestClass.ASSET_ANALYSIS,
            RequestClass.PORTFOLIO_QA,
        }:
            guidelines.append(
                (
                    "If account_id is missing, run portfolio_accounts first and present "
                    "available account_ids before continuing."
                )
            )

        templates.append(
            PromptTemplate(
                prompt_id=f"workflow-agent-{request_class.value}",
                version="1.0.0",
                subagent_id=request_class.value,
                kind=SubagentKind.WORKFLOW_AGENT,
                system_prompt=(
                    f"You are the {request_class.value} workflow agent. "
                    "Decide the next guarded tool call using available evidence."
                ),
                guidelines=tuple(guidelines),
            )
        )
    return templates
