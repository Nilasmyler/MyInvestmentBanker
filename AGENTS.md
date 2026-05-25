# Functionality: Core objective / Ultimate goal
This project is to create an actual agentic Wealth Manager / Investment Banker for me personally. The system should know my portfolio, do autonomous, event-driven + scheduled management and research related to the portfolio for the purposes of: notifying me of thesis-changing risks and events that implies that I should sell or reduce a position or consider increasing my position. I should be able to communicate with the system via a communication agent, or with the PM. This should enable me to ask follow questions and perform specific operations.


# Functionality: Scheduled features
- Portfolio update every monday morning: market movements, value change, summary
- Investment opportunities: Based on recent events, news, prices, SEC filings, technological events and whatever else, the system should provide 1-3 investment propositions with an explainable investment hypothesis.

# Functionality: Event driven features
The system (or parts of the system) should monitor movements in markets, news landscape, technology and economy. To look for investment prospects that will trigger the rest or part of the analysis system. 

# Engineering & Coding:
Simplicity is good. Keep code clean, do not overenigneer solutions that should not be. If more complex architectural changes is neccessary or preferred present a detailed plan for implementing these. These could be tools, additional agents, orchestration et cetera - feel free to propose these when relevant.

# Development workflow
- Prefer the stable repo commands in the `Makefile`: `make setup`, `make run`, `make verify`, and `make tunnel`.
- `make verify` is the default smoke test. It exercises imports, fallback parsing, macro aggregation, discovery formatting, and orchestration behavior via `test_pipeline.py`.
- The FastAPI app is started from `main.py`. Local webhook development should run the API on port `8000` and expose it through `ngrok` when Telegram callbacks need a public URL.
- The app intentionally supports mock-mode behavior when `TELEGRAM_BOT_TOKEN` or `TELEGRAM_USER_ID` are missing. Preserve graceful fallbacks instead of failing hard during local development.

# Development priorities
- Optimize for reliable portfolio monitoring, discovery quality, and low-noise alerts before adding new agents or abstractions.
- Preserve duplicate-suppression logic and "fresh catalyst" checks in discovery flows so scheduled sweeps do not spam repeated ideas.
- Keep unauthorized-user checks, background task boundaries, and scheduled-run routing intact when touching webhook or automation flows.
- Prefer incremental changes in `agents/`, `database/`, and `utils/` over broad framework rewrites.

# Codex optimization
- Use the repo skill at `.agents/skills/my-investment-banker-dev/SKILL.md` when working in this codebase.
- For Codex/OpenAI product questions, prefer the `openai-docs` skill and the `openaiDeveloperDocs` MCP server.
- If a current third-party library API is unclear and a docs MCP such as Context7 is configured, prefer it before generic web search.
- Keep the active skill set lean. A small number of high-signal skills works better than installing many generic skills.
