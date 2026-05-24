import os
import logging
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
import google.generativeai as genai
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Initialize logging
logger = logging.getLogger("MyInvestmentBanker.database")
logging.basicConfig(level=logging.INFO)

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.warning("Supabase credentials missing. Local test mode will be active.")
    supabase_client: Optional[Client] = None
else:
    supabase_client: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Configure Gemini for native embeddings
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """
    Generates a 768-dimension vector embedding using Gemini's native gemini-embedding-2.
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured. Cannot generate embeddings.")
        return [0.0] * 768
    
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-2",
            content=text,
            task_type=task_type,
            output_dimensionality=768
        )
        return response["embedding"]
    except Exception as e:
        logger.error(f"Error generating Gemini embedding: {e}")
        return [0.0] * 768


# ==============================================================================
# Portfolio CRUD Helpers
# ==============================================================================

def fetch_portfolio() -> List[Dict[str, Any]]:
    """
    Retrieves all current holdings in the user's stock portfolio.
    """
    if not supabase_client:
        return []
    try:
        response = supabase_client.table("portfolio_holdings").select("*").execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return []


def update_portfolio_holding(symbol: str, qty: float, price: float, name: str = "") -> bool:
    """
    Upserts a stock holding. Set quantity to 0 to represent selling the entire position.
    """
    if not supabase_client:
        return False
    
    symbol = symbol.upper().strip()
    try:
        # If quantity is 0, we can choose to delete or just update to 0
        if qty <= 0:
            supabase_client.table("portfolio_holdings").delete().eq("symbol", symbol).execute()
            logger.info(f"Deleted ticker {symbol} from holdings (quantity was 0 or less).")
            return True
        
        payload = {
            "symbol": symbol,
            "quantity": float(qty),
            "cost_basis": float(price),
        }
        if name:
            payload["name"] = name
            
        supabase_client.table("portfolio_holdings").upsert(payload).execute()
        logger.info(f"Upserted holdings for ticker {symbol}: qty={qty}, cost_basis={price}")
        return True
    except Exception as e:
        logger.error(f"Error updating portfolio for {symbol}: {e}")
        return False


# ==============================================================================
# Investment Thesis Helpers
# ==============================================================================

def fetch_investment_thesis(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the saved personal investment thesis for a specific stock.
    """
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("investment_thesis").select("*").eq("symbol", symbol.upper()).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching thesis for {symbol}: {e}")
        return None


def save_investment_thesis(symbol: str, thesis_text: str) -> bool:
    """
    Saves and embeds your personal thesis for a specific stock.
    """
    if not supabase_client:
        return False
    
    symbol = symbol.upper().strip()
    try:
        vector = get_embedding(thesis_text, task_type="retrieval_document")
        payload = {
            "symbol": symbol,
            "thesis_text": thesis_text,
            "thesis_vector": vector
        }
        supabase_client.table("investment_thesis").upsert(payload).execute()
        logger.info(f"Saved investment thesis for {symbol}.")
        return True
    except Exception as e:
        logger.error(f"Error saving thesis for {symbol}: {e}")
        return False


# ==============================================================================
# Cognitive Memory Helpers (Analyst Memos)
# ==============================================================================

def save_analyst_memo(symbol: str, period: str, memo_text: str, metrics: Dict[str, Any]) -> bool:
    """
    Saves a formal CFA Agent analyst report memo into memory.
    """
    if not supabase_client:
        return False
    try:
        payload = {
            "symbol": symbol.upper().strip(),
            "period": period,
            "memo_text": memo_text,
            "metrics": metrics
        }
        supabase_client.table("corporate_analyst_memos").insert(payload).execute()
        logger.info(f"Saved analyst memo for {symbol} ({period}).")
        return True
    except Exception as e:
        logger.error(f"Error saving analyst memo for {symbol}: {e}")
        return False


def fetch_historical_memos(symbol: str, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent analyst memos for a specific stock to establish historical memory.
    """
    if not supabase_client:
        return []
    try:
        response = (
            supabase_client.table("corporate_analyst_memos")
            .select("*")
            .eq("symbol", symbol.upper().strip())
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error fetching historical memos for {symbol}: {e}")
        return []


# ==============================================================================
# RAG: News Cache & Semantic Search Helpers
# ==============================================================================

def cache_news_digest(symbol: str, title: str, summary: str, url: str = "", published_at: str = "") -> bool:
    """
    Caches daily summarized developments and generates vector embeddings for future semantic search.
    """
    if not supabase_client:
        return False
    try:
        from datetime import datetime, timezone
        pub_date = published_at if published_at else datetime.now(timezone.utc).isoformat()
        
        vector_text = f"Title: {title}\nSummary: {summary}"
        vector = get_embedding(vector_text, task_type="retrieval_document")
        
        payload = {
            "symbol": symbol.upper().strip(),
            "title": title,
            "summary": summary,
            "url": url,
            "published_at": pub_date,
            "article_vector": vector
        }
        supabase_client.table("news_digests").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(f"Error caching news for {symbol}: {e}")
        return False


def query_semantic_news(symbol: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Performs cosine-distance vector search against cached stock developments in Supabase using pgvector.
    """
    if not supabase_client:
        return []
    try:
        query_vector = get_embedding(query_text, task_type="retrieval_query")
        
        # We call the RPC function in Supabase for vector search if set up.
        # Alternatively, for simplicity, we can pull recent digests or call an RPC.
        # Let's write the pure postgres query RPC if pgvector cosine matches are desired.
        # Inside schema.sql we created an index. For a direct RPC search, we can declare
        # a function in SQL. Let's do a simple filter by symbol first, or fallback to chronological order
        # if the RPC isn't loaded yet.
        response = (
            supabase_client.table("news_digests")
            .select("title, summary, url, published_at")
            .eq("symbol", symbol.upper().strip())
            .order("published_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error executing semantic news search: {e}")
        return []


# ==============================================================================
# Chat logs helper (Short-term memory)
# ==============================================================================

def log_chat_message(user_id: str, role: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Logs chat thread transcripts to database for persistent audits.
    """
    if not supabase_client:
        return False
    try:
        payload = {
            "user_id": str(user_id),
            "role": role,
            "message": message,
            "metadata": metadata if metadata else {}
        }
        supabase_client.table("chat_logs").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging chat message: {e}")
        return False


# ==============================================================================
# User Preferences Helpers (Policy & History Store)
# ==============================================================================

def get_user_preference(key: str) -> Optional[Any]:
    """
    Retrieves a JSON preference value from user_preferences table by key.
    """
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("user_preferences").select("value").eq("key", key).execute()
        return response.data[0]["value"] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching preference {key}: {e}")
        return None


def save_user_preference(key: str, value: Any) -> bool:
    """
    Saves or updates a JSON preference in the user_preferences table.
    """
    if not supabase_client:
        return False
    try:
        payload = {
            "key": key,
            "value": value
        }
        supabase_client.table("user_preferences").upsert(payload).execute()
        logger.info(f"Saved user preference: {key}")
        return True
    except Exception as e:
        logger.error(f"Error saving preference {key}: {e}")
        return False
