from __future__ import annotations

import json

from finance_agent.evals.offline import build_default_offline_cases, run_offline_eval
from finance_agent.orchestration import build_local_orchestrator_service


def main() -> None:
    service = build_local_orchestrator_service()
    report = run_offline_eval(service=service, cases=build_default_offline_cases())
    print(json.dumps(report, default=_json_default, ensure_ascii=False))


def _json_default(value: object) -> object:
    if hasattr(value, "__dict__"):
        return value.__dict__
    if hasattr(value, "value"):
        return value.value
    return str(value)


if __name__ == "__main__":
    main()
