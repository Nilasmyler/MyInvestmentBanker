# MyInvestmentBanker 🤖💼

An automated, cloud-hosted, multi-agent personal wealth manager. Running 24/7 on a hybrid framework using **Gemini 3.5 Flash** as its primary engine, **LangGraph** for stateful agent orchestration, **LLMLingua** for prompt compaction, and **Supabase (PostgreSQL + pgvector)** for memory and vector storage.

Interactive communication is handled entirely via a secure, zero-cost **Telegram Bot** interface with custom action buttons and PDF report dispatch.

---

## 🌟 Key Features

*   **Multi-Agent Coordination (LangGraph)**: Specialized agents (Communication & Portfolio, Data Scout, CFA Analyst, and Risk Officer) collaborate in a deterministic, stateful flow.
*   **Prompt Compression (LLMLingua)**: Middleware compresses raw financial data (SEC filings, news logs) by up to **20x**, ensuring lightning-fast execution and reducing LLM token costs to **<$2/month**.
*   **Official Financial Feeds**: Direct, compliant integration with the **SEC EDGAR API** (via `sec-parser`), **FRED** (Federal Reserve macroeconomic indicators), and **Finnhub/Polygon** free tiers.
*   **Long-Term Agentic Memory**: Stores portfolio history, user preferences, and past analysis reports in Supabase to track financial metrics and forecasts longitudinally.
*   **Secure Access**: Webhook verification strictly restricts command execution and report deliveries to *your* personal Telegram user ID.

---

## 📂 File Architecture

```text
MyInvestmentBanker/
├── agents/                  # Multi-agent orchestrations
│   ├── __init__.py
│   ├── communication_agent.py # Chat & Portfolio manager (Front Desk)
│   ├── scout_agent.py        # Ingestion & Noise-filtering agent (Scout)
│   ├── cfa_agent.py          # Deep corporate reporting analyst (CFA)
│   ├── risk_agent.py         # Macro & portfolio risk synthesizer (Risk Officer)
│   └── orchestrator.py       # Core LangGraph state machine & router
├── database/                # Relational & Vector storage scripts
│   ├── __init__.py
│   ├── schema.sql           # Database tables, triggers, and pgvector indexes
│   └── supabase_client.py   # DB CRUD & vector search utilities
├── utils/                   # Support modules
│   ├── __init__.py
│   ├── financial_tools.py   # FRED, Finnhub, and SEC scrapers
│   └── prompt_compressor.py # Microsoft LLMLingua compression middleware
├── main.py                  # FastAPI server for webhooks & scheduled triggers
├── requirements.txt         # Project dependencies
├── .env.example             # Template for API keys and keys
├── Agents.md                # Detailed Agent specs and prompt strategies
└── Progress.md              # Project initialization & modification logs
```

---

## 🚀 Local Quickstart

### 1. Clone & Set Up Python Environment
```bash
# Clone the repository
cd Documents/GitHub/MyInvestmentBanker

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```
*Make sure to configure `TELEGRAM_USER_ID` to restrict the bot to your Telegram account.*

### 3. Initialize Database Schema
1. Create a free project on [Supabase](https://supabase.com).
2. Go to your project's **SQL Editor** and execute the contents of `database/schema.sql`.
3. Fill in your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` inside `.env`.

### 4. Run the API locally
```bash
uvicorn main:app --reload --port 8000
```

---

## 📄 License

For personal use only. Developed in collaboration with Antigravity.
