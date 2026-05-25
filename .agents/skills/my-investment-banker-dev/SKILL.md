---
name: my-investment-banker-dev
description: Use when working in the MyInvestmentBanker repo on FastAPI webhooks, LangGraph orchestration, Supabase persistence, Telegram delivery, financial data ingestion, or repo-specific validation. Helps Codex use the right commands, preserve fallback behavior, and make small safe changes in this codebase.
---

# MyInvestmentBanker Dev

Use this skill for tasks in this repository.

## Quick start

- Setup the local environment with `make setup`.
- Run the API with `make run`.
- Run the smoke test with `make verify`.
- Expose local port `8000` for Telegram webhook testing with `make tunnel`.

If the `Makefile` is unavailable for any reason, use:

- `python3 -m venv .venv`
- `. .venv/bin/activate`
- `pip install -r requirements.txt`
- `uvicorn main:app --reload --port 8000`
- `python3 test_pipeline.py`

## Code map

- `main.py`: FastAPI entrypoints, Telegram webhook handling, and scheduled-run dispatch.
- `agents/orchestrator.py`: LangGraph portfolio workflow and opportunity-discovery entrypoints.
- `agents/communication_agent.py`: user-command parsing and bulletin formatting.
- `agents/scout_agent.py`, `agents/cfa_agent.py`, `agents/risk_agent.py`: ingestion, analysis, and risk review.
- `database/supabase_client.py`: portfolio, preferences, chat logs, and discovery persistence.
- `utils/financial_tools.py`: market, macro, filings, and news data helpers.
- `test_pipeline.py`: repo smoke test and fallback-behavior verification.

## Working rules

- Preserve mock-mode and fallback behavior when secrets or upstream data are unavailable.
- Keep changes narrow. Avoid adding new frameworks or orchestration layers unless the task clearly needs them.
- When editing discovery logic, protect duplicate suppression and catalyst checks so automated sweeps stay low-noise.
- When editing webhook or scheduled-run code, preserve background task flow and private-user access checks.
- Prefer validating through the repo entrypoints and `test_pipeline.py` rather than only unit-level reasoning.

## External context

- For OpenAI or Codex setup questions, use the `openai-docs` skill and the `openaiDeveloperDocs` MCP server.
- If a current library API is uncertain and Context7 MCP is configured, prefer it before broad web search.
