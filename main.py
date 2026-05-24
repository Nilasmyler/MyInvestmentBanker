import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from agents.communication_agent import CommunicationAgent
from agents.orchestrator import (
    trigger_autonomous_discovery_check,
    trigger_opportunity_discovery,
    trigger_wealth_manager_run,
)
from database.supabase_client import log_chat_message

# Load env variables
load_dotenv()

# Setup logs
logger = logging.getLogger("MyInvestmentBanker.main")
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_USER_ID:
    logger.warning(
        "TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID is not configured in .env. Bot features will run in mock mode."
    )

app = FastAPI(
    title="MyInvestmentBanker API",
    description="Automated multi-agent portfolio risk tracking and investment discovery.",
    version="1.1",
)


async def send_telegram_message(chat_id: str, text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram token missing. Printing message to stdout instead:")
        logger.info(f"\n[Telegram Bot Dispatch to {chat_id}]:\n{text}\n")
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            payload["parse_mode"] = ""
            await client.post(url, json=payload, timeout=10)
            return False
        except Exception as e:
            logger.error(f"Failed to post to Telegram: {e}")
            return False


@app.get("/health")
def health_check():
    return {
        "status": "active",
        "environment": "cloud-hosted",
        "primary_engine": "Gemini 3.5 Flash",
        "orchestration_framework": "LangGraph (Stateful)",
    }


async def background_pipeline_execution(chat_id: str):
    await send_telegram_message(
        chat_id,
        "⚙️ **MyInvestmentBanker**: Starting the portfolio analysis cycle. Gathering market data, SEC filings, and thesis context.",
    )
    bulletin_report = trigger_wealth_manager_run()
    portfolio_follow_up = CommunicationAgent.prepare_portfolio_follow_up(chat_id)
    if portfolio_follow_up:
        bulletin_report = f"{bulletin_report}\n\n{portfolio_follow_up}"
    log_chat_message(chat_id, "assistant", bulletin_report, {"source": "Scheduled/On-Demand Portfolio Run"})
    await send_telegram_message(chat_id, bulletin_report)


async def background_discovery_execution(chat_id: str, run_type: str = "deep"):
    if run_type == "deep":
        await send_telegram_message(
            chat_id,
            "🔍 **MyInvestmentBanker**: Starting a deep theme-led discovery review. Scanning macro context, sector leaders, filings, and material news.",
        )

    discovery_result = (
        trigger_opportunity_discovery(run_type="deep", return_result=True)
        if run_type == "deep"
        else trigger_autonomous_discovery_check(return_result=True)
    )
    if not discovery_result:
        logger.info("Discovery run completed without a bulletin.")
        return

    discovery_bulletin = discovery_result.get("bulletin", "")
    if not discovery_bulletin:
        logger.info("Discovery result did not produce a bulletin.")
        return

    discovery_follow_up = CommunicationAgent.prepare_discovery_follow_up(chat_id, discovery_result)
    if discovery_follow_up:
        discovery_bulletin = f"{discovery_bulletin}\n\n{discovery_follow_up}"

    log_chat_message(chat_id, "assistant", discovery_bulletin, {"source": f"{run_type.title()} Discovery"})
    await send_telegram_message(chat_id, discovery_bulletin)


async def background_single_stock_analysis(chat_id: str, symbol: str, user_context: str = ""):
    await send_telegram_message(
        chat_id,
        f"🔬 **MyInvestmentBanker**: Starting a single-stock review for **{symbol.upper()}**. "
        "I am pulling price context, filings, material news, and your current preference profile.",
    )
    briefing = CommunicationAgent.generate_single_stock_analysis(chat_id, symbol, user_context)
    log_chat_message(chat_id, "assistant", briefing, {"source": "Single Stock Analysis", "symbol": symbol.upper()})
    await send_telegram_message(chat_id, briefing)


@app.post("/telegram-webhook")
async def telegram_webhook_receiver(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    message = body.get("message", {})
    chat = message.get("chat", {})
    chat_id = str(chat.get("id", ""))
    user_info = message.get("from", {})
    sender_id = str(user_info.get("id", ""))
    message_text = message.get("text", "")

    if not chat_id or not sender_id:
        return {"status": "ignored", "reason": "No valid message sender info found."}

    if TELEGRAM_USER_ID and sender_id != TELEGRAM_USER_ID:
        logger.warning(f"Unauthorized access attempt! Sender: {sender_id} (Name: {user_info.get('username')})")
        background_tasks.add_task(
            send_telegram_message,
            chat_id,
            "🔒 **Access Denied**: This wealth manager instance is strictly configured for a single private portfolio.",
        )
        return {"status": "rejected", "reason": "Unauthorized User ID"}

    log_chat_message(chat_id, "user", message_text, {"username": user_info.get("username")})

    if message_text.strip().lower() == "/update":
        background_tasks.add_task(background_pipeline_execution, chat_id)
        return {"status": "processing", "node": "PortfolioDigest"}

    if message_text.strip().lower() == "/discover":
        background_tasks.add_task(background_discovery_execution, chat_id, "deep")
        return {"status": "processing", "node": "ThemeDiscovery"}

    stock_analysis_request = CommunicationAgent.detect_stock_analysis_request(message_text)
    if stock_analysis_request:
        background_tasks.add_task(
            background_single_stock_analysis,
            chat_id,
            stock_analysis_request["symbol"],
            stock_analysis_request.get("context", ""),
        )
        return {
            "status": "processing",
            "node": "SingleStockAnalysis",
            "symbol": stock_analysis_request["symbol"],
        }

    reply = CommunicationAgent.parse_user_command(chat_id, message_text)
    background_tasks.add_task(send_telegram_message, chat_id, reply)
    return {"status": "success", "command_processed": message_text.split()[0] if message_text else "none"}


@app.post("/scheduled-run")
def scheduled_daily_digest(run_type: str = "daily", background_tasks: BackgroundTasks = None):
    if not TELEGRAM_USER_ID:
        raise HTTPException(status_code=400, detail="TELEGRAM_USER_ID is not configured in .env.")

    if run_type in ["weekly_discovery", "monday_deep_discovery"]:
        logger.info("Cron Trigger: Initiating weekly deep discovery run...")
        background_tasks.add_task(background_discovery_execution, TELEGRAM_USER_ID, "deep")
        return {
            "status": "cron_initiated",
            "run_type": run_type,
            "target_user_id": TELEGRAM_USER_ID,
            "workflow": "Theme-Led Deep Discovery",
        }

    if run_type in ["discovery_sweep", "autonomous_discovery", "hourly_news_sweep"]:
        logger.info("Cron Trigger: Initiating weekday discovery sweep...")
        background_tasks.add_task(background_discovery_execution, TELEGRAM_USER_ID, "sweep")
        return {
            "status": "cron_initiated",
            "run_type": run_type,
            "target_user_id": TELEGRAM_USER_ID,
            "workflow": "Theme Discovery Sweep",
        }

    logger.info("Cron Trigger: Initiating scheduled portfolio digest...")
    background_tasks.add_task(background_pipeline_execution, TELEGRAM_USER_ID)
    return {
        "status": "cron_initiated",
        "run_type": run_type,
        "target_user_id": TELEGRAM_USER_ID,
        "workflow": "LangGraph Portfolio Flow",
    }
