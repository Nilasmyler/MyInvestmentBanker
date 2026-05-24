import os
import logging
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import httpx
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Setup logs
logger = logging.getLogger("MyInvestmentBanker.main")
logging.basicConfig(level=logging.INFO)

# Config Keys
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_USER_ID = os.getenv("TELEGRAM_USER_ID")

# Verify keys are set
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_USER_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID is not configured in .env. Bot features will run in mock mode.")

# Import our agents
from agents.communication_agent import CommunicationAgent
from agents.orchestrator import trigger_wealth_manager_run, trigger_opportunity_discovery, trigger_autonomous_discovery_check
from database.supabase_client import log_chat_message

app = FastAPI(
    title="MyInvestmentBanker API",
    description="Automated multi-agent portfolio risk tracking and investment theses evaluation.",
    version="1.0"
)

# ==============================================================================
# Telegram Message Dispatcher Helper
# ==============================================================================

async def send_telegram_message(chat_id: str, text: str) -> bool:
    """
    Dispatches a formatted Markdown message back to the user's secure Telegram chat.
    Uses httpx to ensure non-blocking networking.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram token missing. Printing message to stdout instead:")
        logger.info(f"\n[Telegram Bot Dispatch to {chat_id}]:\n{text}\n")
        return True
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                # Fallback to plain text if Markdown parsing failed (due to unescaped characters)
                payload["parse_mode"] = ""
                await client.post(url, json=payload, timeout=10)
                return False
        except Exception as e:
            logger.error(f"Failed to post to Telegram: {e}")
            return False


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/health")
def health_check():
    """Simple API status checks."""
    return {
        "status": "active",
        "environment": "cloud-hosted",
        "primary_engine": "Gemini 3.5 Flash",
        "orchestration_framework": "LangGraph (Stateful)"
    }


async def background_pipeline_execution(chat_id: str):
    """Asynchronous worker to run the multi-agent graph in the background."""
    await send_telegram_message(chat_id, "⚙️ **MyInvestmentBanker**: Starting multi-agent analysis cycle. Scraping pricing, parsing SEC EDGAR filings, and verifying thesis vectors. Please hold...")
    
    # Run the full LangGraph pipeline
    bulletin_report = trigger_wealth_manager_run()
    
    # Log report memo in Supabase chat histories
    log_chat_message(chat_id, "assistant", bulletin_report, {"source": "Scheduled/On-Demand Run"})
    
    # Dispatch compiled bulletin to user
    await send_telegram_message(chat_id, bulletin_report)


async def background_discovery_execution(chat_id: str):
    """Asynchronous worker to run the autonomous opportunity discovery flow."""
    await send_telegram_message(chat_id, "🔍 **MyInvestmentBanker**: Initiating autonomous opportunity discovery sweep. Screeners are loading candidate stocks. Please hold...")
    
    # Run discovery
    discovery_bulletin = trigger_opportunity_discovery()
    
    # Log in Supabase chat history
    log_chat_message(chat_id, "assistant", discovery_bulletin, {"source": "On-Demand Discovery"})
    
    # Dispatch compiled bulletin to user
    await send_telegram_message(chat_id, discovery_bulletin)


@app.post("/telegram-webhook")
async def telegram_webhook_receiver(request: Request, background_tasks: BackgroundTasks):
    """
    Handles live webhooks from Telegram.
    Verifies that requests originate strictly from the authorized user.
    """
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
        
    # SECURITY GUARDRAIL: Strict User-ID Verification
    if TELEGRAM_USER_ID and sender_id != TELEGRAM_USER_ID:
        logger.warning(f"Unauthorized access attempt! Sender: {sender_id} (Name: {user_info.get('username')})")
        # Send security rejection message
        background_tasks.add_task(
            send_telegram_message, 
            chat_id, 
            "🔒 **Access Denied**: This wealth manager instance is strictly configured for a single private portfolio."
        )
        return {"status": "rejected", "reason": "Unauthorized User ID"}
        
    # Log User command to DB
    log_chat_message(chat_id, "user", message_text, {"username": user_info.get("username")})
    
    # Check if user triggered an on-demand full Multi-Agent Analysis
    if message_text.strip().lower() == "/update":
        background_tasks.add_task(background_pipeline_execution, chat_id)
        return {"status": "processing", "node": "IngestionNode"}
        
    # Check if user triggered an on-demand Opportunity Discovery
    if message_text.strip().lower() == "/discover":
        background_tasks.add_task(background_discovery_execution, chat_id)
        return {"status": "processing", "node": "DiscoveryNode"}
        
    # Parse standard command mutations (portfolio edits, cost basis, thesis updates)
    reply = CommunicationAgent.parse_user_command(chat_id, message_text)
    
    # Dispatch reply back to Telegram
    background_tasks.add_task(send_telegram_message, chat_id, reply)
    
    return {"status": "success", "command_processed": message_text.split()[0] if message_text else "none"}


async def background_autonomous_discovery_check(chat_id: str):
    """Asynchronous worker to execute the 24/7 PM planning poller check."""
    logger.info("Executing 24/7 PM Planning opportunity discovery check...")
    
    # Run autonomous check
    briefing = trigger_autonomous_discovery_check()
    
    if briefing:
        # PM triggered and compiled a recommendation!
        log_chat_message(chat_id, "assistant", briefing, {"source": "Autonomous 24/7 Discovery Event"})
        await send_telegram_message(chat_id, briefing)
    else:
        logger.info("PM decided to stand down today. Silently completing poller run.")


@app.post("/scheduled-run")
def scheduled_daily_digest(run_type: str = "daily", background_tasks: BackgroundTasks = None):
    """
    Scheduled cron endpoint called by Cloud Cron (GitHub Actions, Render scheduler).
    Supports 'daily' portfolio digest and 'autonomous_discovery' (24/7 PM planning check).
    """
    if not TELEGRAM_USER_ID:
        raise HTTPException(status_code=400, detail="TELEGRAM_USER_ID is not configured in .env.")
        
    if run_type in ["autonomous_discovery", "hourly_news_sweep"]:
        logger.info("Cron Trigger: Initiating autonomous 24/7 PM Opportunity Planning check...")
        background_tasks.add_task(background_autonomous_discovery_check, TELEGRAM_USER_ID)
        return {
            "status": "cron_initiated",
            "run_type": run_type,
            "target_user_id": TELEGRAM_USER_ID,
            "workflow": "PM Planning Engine"
        }
    else:
        logger.info("Cron Trigger: Initiating scheduled daily digest...")
        background_tasks.add_task(background_pipeline_execution, TELEGRAM_USER_ID)
        return {
            "status": "cron_initiated",
            "run_type": run_type,
            "target_user_id": TELEGRAM_USER_ID,
            "workflow": "LangGraph Cyclic Flow"
        }
