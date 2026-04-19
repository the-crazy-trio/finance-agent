This repository implements an AI Financial Assistant PoC.

## Source of truth

- Always rely on `docs/system-design.md` and `docs/specs/*.md`.
- Keep specs and implementation aligned with `docs/specs/changelog.md` rules.
- Ignore legacy assumptions that are not present in current docs or code.

## Current project layout

- Main package: `src/finance_agent/`
  - `contracts/`: request/response, entities, tool contracts
  - `orchestration/`: context, identity, orchestrator service
  - `retriever/`: guarded tool gateway, storage, T-Invest adapter, error mapping
  - `harness/`: skill runner, local harness runner, MCP stdio server
  - `config/`: runtime config and model routing policy
- API entrypoint: `main.py`
- MCP script entrypoint: `mcp_server = finance_agent.harness.mcp_server:main`
- Tests: `tests/test_*.py`

## Entrypoints to preserve

- `SKILL` path for harness usage via guarded workflow-level tools.
- Local orchestrator path for API/CLI style usage without harness.
- MCP support must stay a thin adapter over existing core logic.

## Architecture constraints

- Keep core architecture stable: contracts, workflows, deterministic analytics, storage, API layer.
- Enforce guarded tools only (no free-form raw data access).
- T-Invest integration is read-only.
- Keep identity boundary via adapters:
  - local PoC mode: `principal_id=local_user`
  - API mode: identity resolved by auth metadata/header adapter
- Never trust identity fields from request payload.
- Propagate `correlation_id` end-to-end across API, orchestrator, and tools.

## Tooling contracts (minimum)

- Required guarded tools:
  - `portfolio_collect`
  - `portfolio_show`
  - `strategy_save`
  - `strategy_fit`
- Keep error normalization aligned with specs: `VALIDATION_ERROR`, `UNAUTHORIZED_SCOPE`,
  `TIMEOUT`, `UPSTREAM_UNAVAILABLE`, `RATE_LIMITED`, `MAPPING_NOT_FOUND`.

## Runtime and config rules

- Python `3.13`, PEP8 compatible style, max line length `100`.
- Use `uv` for dependency management and command execution.
- Required secrets for runtime config:
  - `OPENROUTER_API_KEY`
  - `TINVEST_READONLY_TOKEN`
- In PoC mode, `.env` is for secrets only.

## Quality gates

- Run lint: `uv run ruff check .`
- Run tests: `uv run python -m unittest discover -s tests -p "test_*.py"`

## Engineering preference

Simple > Readable > Effective
