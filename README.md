# MyInvestmentBanker 🤖💼

An automated, cloud-hosted, multi-agent personal wealth manager. Running 24/7 on a hybrid framework using **Gemini 3.5 Flash** as its primary engine, **LangGraph** for stateful agent orchestration, **LLMLingua** for prompt compaction, and **Supabase (PostgreSQL + pgvector)** for persistence and cached research context.

Interactive communication is handled through a secure **Telegram Bot** webhook flow plus scheduled trigger endpoints protected by shared secrets.

---

## 🌟 Key Features

*   **Multi-Agent Coordination (LangGraph)**: Specialized agents (Communication & Portfolio, Data Scout, CFA Analyst, and Risk Officer) collaborate in a deterministic, stateful flow.
*   **Prompt Compression (LLMLingua)**: Middleware compresses raw financial data (SEC filings, news logs) by up to **20x**, ensuring lightning-fast execution and reducing LLM token costs to **<$2/month**.
*   **Official Financial Feeds**: Direct integration with the **SEC EDGAR API**, **FRED** macroeconomic indicators, **Finnhub** company news when configured, and a Google News RSS fallback when Finnhub is unavailable.
*   **Adaptive Research Routing**: Deep discovery no longer sends every candidate through the expensive path. A deterministic planner routes ownership intelligence, analyst-consensus checks, and LLM-heavy review only where the evidence set justifies it.
*   **Long-Term Agentic Memory**: Stores active portfolio snapshots, user preferences, theses, and analyst memos in Supabase so weekly digests can compare against prior baselines without deleting historical context when a position is closed.
*   **Brokerage Portfolio Import**: Can read an existing brokerage account and sync live holdings into the tracked portfolio snapshot. Alpaca is supported first through API keys, with the adapter structured for more brokers later.
*   **Secure Access**: Telegram webhooks require Telegram's secret-token header plus your configured `TELEGRAM_USER_ID`, and `/scheduled-run` requires its own shared secret before it can queue live runs.

---

## 📂 File Architecture

```text
MyInvestmentBanker/
├── .agents/skills/         # Repo-local Codex skills and workflow guidance
├── agents/                  # Multi-agent orchestrations
│   ├── __init__.py
│   ├── communication_agent.py # Chat & Portfolio manager (Front Desk)
│   ├── scout_agent.py        # Ingestion & Noise-filtering agent (Scout)
│   ├── cfa_agent.py          # Deep corporate reporting analyst (CFA)
│   ├── risk_agent.py         # Macro & portfolio risk synthesizer (Risk Officer)
│   ├── ownership_intel_agent.py # Insider / sponsorship / holder-intelligence specialist
│   ├── street_consensus_agent.py # Analyst recommendation and target-trend specialist
│   ├── research_planner_agent.py # Bounded research router and candidate-review budgeter
│   └── orchestrator.py       # Core LangGraph state machine & router
├── database/                # Relational & Vector storage scripts
│   ├── __init__.py
│   ├── schema.sql           # Database tables, triggers, and pgvector indexes
│   └── supabase_client.py   # DB CRUD & vector search utilities
├── integrations/            # External brokerage and data-provider adapters
│   └── brokerage.py         # Brokerage account readers (Alpaca currently supported)
├── services/                # Application service layer
│   └── portfolio_service.py # Portfolio snapshot loading + broker sync helpers
├── utils/                   # Support modules
│   ├── __init__.py
│   ├── financial_tools.py   # FRED, Finnhub, and SEC scrapers
│   └── prompt_compressor.py # Microsoft LLMLingua compression middleware
├── Makefile                # Stable setup, run, verify, and ngrok commands
├── main.py                  # FastAPI server for webhooks & scheduled triggers
├── requirements.txt         # Project dependencies
├── .env.example             # Template for API keys and keys
├── AGENTS.md                # Repo instructions and Codex development guidance
└── Progress.md              # Project initialization & modification logs
```

---

## 🚀 Local Quickstart

### 1. Clone & Set Up Python Environment
```bash
cd Documents/GitHub/MyInvestmentBanker

make setup
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
Configure the Telegram and scheduler secrets before exposing the API publicly:
```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=...
TELEGRAM_WEBHOOK_SECRET=choose-a-random-secret
SCHEDULED_RUN_SECRET=choose-a-different-random-secret
```
*If `TELEGRAM_BOT_TOKEN` or `TELEGRAM_USER_ID` is missing, the app stays in mock-delivery mode and prints outbound bot messages locally instead of sending them to Telegram.*

Optional brokerage sync settings:
```bash
BROKERAGE_PROVIDER=alpaca
BROKERAGE_SYNC_ON_READ=false
BROKERAGE_SYNC_BEFORE_RUNS=true
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
# Use https://api.alpaca.markets for live accounts
ALPACA_API_BASE_URL=https://paper-api.alpaca.markets
```

When configured, `/sync` imports current broker positions into `portfolio_holdings`, retires closed positions from the active view without deleting linked thesis or memo history, and stores a lightweight snapshot baseline for later digest comparisons. Scheduled analysis runs can refresh from the broker automatically before research starts.

### 3. Initialize Database Schema
1. Create a free project on [Supabase](https://supabase.com).
2. Go to your project's **SQL Editor** and execute the contents of `database/schema.sql`.
3. Fill in your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` inside `.env`.

### 4. Run the API locally
```bash
make run
```

### 5. Run the smoke test
```bash
make verify
```

### 6. Expose the local webhook for Telegram development
```bash
make tunnel
```

When you register the Telegram webhook, send the same `TELEGRAM_WEBHOOK_SECRET` as Telegram's secret token. Any caller hitting `/scheduled-run` must include `X-Scheduled-Run-Secret: <your secret>`.

---

## 📄 License

For personal use only. Developed in collaboration with Antigravity.
