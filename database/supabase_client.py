import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from supabase import Client, create_client

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
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

NEWS_DIGEST_UNSUPPORTED_COLUMNS = set()


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").upper().strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(raw_value: str) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_iso_datetime(raw_value: str) -> str:
    parsed = _parse_iso_datetime(raw_value)
    if not parsed:
        return raw_value
    return parsed.replace(microsecond=0).isoformat()


def _get_existing_holding(symbol: str) -> Optional[Dict[str, Any]]:
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("portfolio_holdings").select("*").eq("symbol", symbol).limit(1).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching holding for {symbol}: {e}")
        return None


def _is_legacy_news_digest_schema_error(error: Exception) -> bool:
    error_text = str(error)
    return "news_digests.entity_type" in error_text or "news_digests.entity_key" in error_text


def _extract_missing_news_digest_column(error: Exception) -> Optional[str]:
    error_text = str(error)
    for column in ["entity_type", "entity_key", "metadata", "article_vector"]:
        if f"news_digests.{column}" in error_text or f"'{column}' column of 'news_digests'" in error_text:
            return column
    return None


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
            output_dimensionality=768,
        )
        return response["embedding"]
    except Exception as e:
        logger.error(f"Error generating Gemini embedding: {e}")
        return [0.0] * 768


# ==============================================================================
# Portfolio CRUD Helpers
# ==============================================================================

def fetch_portfolio(include_inactive: bool = False) -> List[Dict[str, Any]]:
    if not supabase_client:
        return []
    try:
        response = supabase_client.table("portfolio_holdings").select("*").execute()
        holdings = response.data or []
        if not include_inactive:
            holdings = [row for row in holdings if _safe_float(row.get("quantity")) > 0]
        holdings.sort(key=lambda row: str(row.get("symbol", "")))
        return holdings
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return []


def update_portfolio_holding(symbol: str, qty: float, price: float, name: str = "") -> bool:
    if not supabase_client:
        return False

    symbol = _normalize_symbol(symbol)
    try:
        if qty <= 0:
            existing_holding = _get_existing_holding(symbol)
            if not existing_holding:
                logger.info(f"No holding row existed for {symbol}; treating retire request as a no-op.")
                return True

            payload = {
                "symbol": symbol,
                "quantity": 0.0,
                "cost_basis": _safe_float(existing_holding.get("cost_basis")),
            }
            if existing_holding.get("name"):
                payload["name"] = existing_holding["name"]

            supabase_client.table("portfolio_holdings").upsert(payload).execute()
            logger.info(f"Retired ticker {symbol} from active holdings while preserving linked history.")
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


def replace_portfolio_holdings(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not supabase_client:
        return {
            "ok": False,
            "persisted": False,
            "removed_symbols": [],
            "upserted_count": 0,
            "reason": "Supabase client is not configured.",
        }

    normalized_holdings: List[Dict[str, Any]] = []
    incoming_symbols = set()
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).upper().strip()
        quantity = float(holding.get("quantity", 0) or 0)
        if not symbol or quantity <= 0:
            continue
        incoming_symbols.add(symbol)
        normalized_holdings.append(
            {
                "symbol": symbol,
                "name": str(holding.get("name", "") or ""),
                "quantity": quantity,
                "cost_basis": float(holding.get("cost_basis", 0) or 0),
            }
        )

    try:
        existing_holdings = {
            _normalize_symbol(row.get("symbol", "")): row
            for row in fetch_portfolio()
            if row.get("symbol")
        }
        existing_symbols = set(existing_holdings.keys())

        if normalized_holdings:
            supabase_client.table("portfolio_holdings").upsert(normalized_holdings).execute()

        removed_symbols = sorted(existing_symbols - incoming_symbols)
        for symbol in removed_symbols:
            prior_row = existing_holdings.get(symbol, {})
            payload = {
                "symbol": symbol,
                "quantity": 0.0,
                "cost_basis": _safe_float(prior_row.get("cost_basis")),
            }
            if prior_row.get("name"):
                payload["name"] = prior_row["name"]
            supabase_client.table("portfolio_holdings").upsert(payload).execute()

        return {
            "ok": True,
            "persisted": True,
            "removed_symbols": removed_symbols,
            "upserted_count": len(normalized_holdings),
        }
    except Exception as e:
        logger.error(f"Error replacing portfolio holdings: {e}")
        return {
            "ok": False,
            "persisted": False,
            "removed_symbols": [],
            "upserted_count": 0,
            "reason": str(e),
        }


# ==============================================================================
# Investment Thesis Helpers
# ==============================================================================

def fetch_investment_thesis(symbol: str) -> Optional[Dict[str, Any]]:
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("investment_thesis").select("*").eq("symbol", _normalize_symbol(symbol)).execute()
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching thesis for {symbol}: {e}")
        return None


def save_investment_thesis(symbol: str, thesis_text: str) -> bool:
    if not supabase_client:
        return False

    symbol = _normalize_symbol(symbol)
    try:
        vector = get_embedding(thesis_text, task_type="retrieval_document")
        payload = {
            "symbol": symbol,
            "thesis_text": thesis_text,
            "thesis_vector": vector,
        }
        supabase_client.table("investment_thesis").upsert(payload).execute()
        logger.info(f"Saved investment thesis for {symbol}.")
        return True
    except Exception as e:
        logger.error(f"Error saving thesis for {symbol}: {e}")
        return False


# ==============================================================================
# Analyst Memo Helpers
# ==============================================================================

def save_analyst_memo(symbol: str, period: str, memo_text: str, metrics: Dict[str, Any]) -> bool:
    if not supabase_client:
        return False
    try:
        symbol = _normalize_symbol(symbol)
        payload = {
            "symbol": symbol,
            "period": period,
            "memo_text": memo_text,
            "metrics": metrics,
        }
        existing = (
            supabase_client.table("corporate_analyst_memos")
            .select("id")
            .eq("symbol", symbol)
            .eq("period", period)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if existing.data:
            supabase_client.table("corporate_analyst_memos").update(payload).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase_client.table("corporate_analyst_memos").insert(payload).execute()
        logger.info(f"Saved analyst memo for {symbol} ({period}).")
        return True
    except Exception as e:
        logger.error(f"Error saving analyst memo for {symbol}: {e}")
        return False


def fetch_historical_memos(symbol: str, limit: int = 2) -> List[Dict[str, Any]]:
    if not supabase_client:
        return []
    try:
        response = (
            supabase_client.table("corporate_analyst_memos")
            .select("*")
            .eq("symbol", _normalize_symbol(symbol))
            .order("created_at", desc=True)
            .limit(max(limit * 4, limit))
            .execute()
        )
        unique_memos = []
        seen_periods = set()
        for row in response.data or []:
            period = str(row.get("period", "")).strip().upper()
            if period in seen_periods:
                continue
            seen_periods.add(period)
            unique_memos.append(row)
            if len(unique_memos) >= limit:
                break
        return unique_memos
    except Exception as e:
        logger.error(f"Error fetching historical memos for {symbol}: {e}")
        return []


# ==============================================================================
# Event Cache Helpers
# ==============================================================================

def cache_news_digest(
    symbol: str,
    title: str,
    summary: str,
    url: str = "",
    published_at: str = "",
    entity_type: str = "symbol",
    entity_key: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    if not supabase_client:
        return False
    try:
        normalized_symbol = _normalize_symbol(symbol)
        normalized_entity_key = _normalize_symbol(entity_key or symbol)
        pub_date = _normalize_iso_datetime(published_at) if published_at else datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        use_legacy_schema = "entity_type" in NEWS_DIGEST_UNSUPPORTED_COLUMNS or "entity_key" in NEWS_DIGEST_UNSUPPORTED_COLUMNS
        if not use_legacy_schema:
            try:
                duplicate_query = (
                    supabase_client.table("news_digests")
                    .select("id, published_at, title, url")
                    .eq("entity_type", entity_type)
                    .eq("entity_key", normalized_entity_key)
                )
                if url:
                    duplicate_query = duplicate_query.eq("url", url).limit(1)
                else:
                    duplicate_query = duplicate_query.eq("title", title).order("published_at", desc=True).limit(5)
                duplicate_rows = duplicate_query.execute().data or []
            except Exception as duplicate_error:
                if not _is_legacy_news_digest_schema_error(duplicate_error):
                    raise
                NEWS_DIGEST_UNSUPPORTED_COLUMNS.update({"entity_type", "entity_key"})
                use_legacy_schema = True

        if use_legacy_schema:
            duplicate_query = supabase_client.table("news_digests").select("id, published_at, title, url").eq(
                "symbol",
                normalized_symbol or normalized_entity_key,
            )
            if url:
                duplicate_query = duplicate_query.eq("url", url).limit(1)
            else:
                duplicate_query = duplicate_query.eq("title", title).order("published_at", desc=True).limit(5)
            duplicate_rows = duplicate_query.execute().data or []

        for row in duplicate_rows:
            same_url = bool(url) and row.get("url") == url
            same_title = not url and str(row.get("title", "")).strip() == title.strip()
            same_published_at = _normalize_iso_datetime(str(row.get("published_at", ""))) == pub_date
            if same_url or (same_title and same_published_at):
                logger.info(f"Skipping duplicate cached news for {normalized_entity_key}: {title}")
                return True

        vector_text = f"Title: {title}\nSummary: {summary}"
        vector = get_embedding(vector_text, task_type="retrieval_document")
        payload = {
            "symbol": normalized_symbol if normalized_symbol else None,
            "entity_type": entity_type,
            "entity_key": normalized_entity_key,
            "title": title,
            "summary": summary,
            "url": url,
            "published_at": pub_date,
            "article_vector": vector,
            "metadata": metadata or {},
        }
        if use_legacy_schema:
            payload.pop("entity_type", None)
            payload.pop("entity_key", None)

        for unsupported_column in list(NEWS_DIGEST_UNSUPPORTED_COLUMNS):
            payload.pop(unsupported_column, None)

        while True:
            try:
                supabase_client.table("news_digests").insert(payload).execute()
                break
            except Exception as insert_error:
                missing_column = _extract_missing_news_digest_column(insert_error)
                if not missing_column:
                    raise
                if missing_column in ["entity_type", "entity_key"]:
                    NEWS_DIGEST_UNSUPPORTED_COLUMNS.update({"entity_type", "entity_key"})
                    payload.pop("entity_type", None)
                    payload.pop("entity_key", None)
                else:
                    NEWS_DIGEST_UNSUPPORTED_COLUMNS.add(missing_column)
                    payload.pop(missing_column, None)
        return True
    except Exception as e:
        logger.error(f"Error caching news for {symbol or entity_key}: {e}")
        return False


def query_semantic_news(symbol: str, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    if not supabase_client:
        return []
    try:
        _ = get_embedding(query_text, task_type="retrieval_query")
        while True:
            include_entity_fields = "entity_type" not in NEWS_DIGEST_UNSUPPORTED_COLUMNS and "entity_key" not in NEWS_DIGEST_UNSUPPORTED_COLUMNS
            include_metadata = "metadata" not in NEWS_DIGEST_UNSUPPORTED_COLUMNS
            select_fields = ["symbol", "title", "summary", "url", "published_at"]
            if include_entity_fields:
                select_fields[1:1] = ["entity_type", "entity_key"]
            if include_metadata:
                select_fields.append("metadata")

            try:
                query = supabase_client.table("news_digests").select(", ".join(select_fields)).order("published_at", desc=True).limit(limit)
                if include_entity_fields:
                    query = query.eq("entity_key", _normalize_symbol(symbol))
                else:
                    query = query.eq("symbol", _normalize_symbol(symbol))
                response = query.execute()
                break
            except Exception as query_error:
                missing_column = _extract_missing_news_digest_column(query_error)
                if missing_column in ["entity_type", "entity_key"]:
                    NEWS_DIGEST_UNSUPPORTED_COLUMNS.update({"entity_type", "entity_key"})
                    continue
                if missing_column == "metadata":
                    NEWS_DIGEST_UNSUPPORTED_COLUMNS.add("metadata")
                    continue
                raise
        return response.data
    except Exception as e:
        logger.error(f"Error executing semantic news search: {e}")
        return []


def fetch_recent_cached_events(entity_key: str, entity_type: str = "symbol", limit: int = 10) -> List[Dict[str, Any]]:
    if not supabase_client:
        return []
    try:
        use_legacy_schema = "entity_type" in NEWS_DIGEST_UNSUPPORTED_COLUMNS or "entity_key" in NEWS_DIGEST_UNSUPPORTED_COLUMNS
        if not use_legacy_schema:
            try:
                response = (
                    supabase_client.table("news_digests")
                    .select("*")
                    .eq("entity_key", entity_key)
                    .eq("entity_type", entity_type)
                    .order("published_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                return response.data
            except Exception as query_error:
                if not _is_legacy_news_digest_schema_error(query_error):
                    raise
                NEWS_DIGEST_UNSUPPORTED_COLUMNS.update({"entity_type", "entity_key"})

        response = (
            supabase_client.table("news_digests")
            .select("*")
            .eq("symbol", _normalize_symbol(entity_key))
            .order("published_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"Error fetching cached events for {entity_type}:{entity_key}: {e}")
        return []


# ==============================================================================
# Discovery Persistence Helpers
# ==============================================================================

def save_discovery_run(
    run_type: str,
    status: str,
    policy_snapshot: Dict[str, Any],
    themes: List[Dict[str, Any]],
    summary_text: str = "",
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> Optional[str]:
    if not supabase_client:
        return None
    try:
        payload = {
            "run_type": run_type,
            "status": status,
            "policy_snapshot": policy_snapshot,
            "themes": themes,
            "summary_text": summary_text,
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "completed_at": completed_at or datetime.now(timezone.utc).isoformat(),
        }
        response = supabase_client.table("discovery_runs").insert(payload).execute()
        if response.data:
            return response.data[0].get("id")
        return None
    except Exception as e:
        logger.error(f"Error saving discovery run: {e}")
        return None


def save_discovery_candidates(run_id: Optional[str], candidates: List[Dict[str, Any]]) -> bool:
    if not supabase_client or not candidates:
        return False
    try:
        payload = []
        for candidate in candidates:
            record = {
                "run_id": run_id,
                "theme_key": candidate.get("theme_key"),
                "symbol": candidate.get("symbol"),
                "source_etf": candidate.get("source_etf"),
                "recommendation_type": candidate.get("recommendation_type"),
                "status": candidate.get("status", "recommended"),
                "evidence": candidate.get("evidence", {}),
                "rationale": candidate.get("rationale", ""),
            }
            payload.append(record)
        supabase_client.table("discovery_candidates").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(f"Error saving discovery candidates: {e}")
        return False


def fetch_recent_discovery_candidates(
    symbol: Optional[str] = None,
    theme_key: Optional[str] = None,
    days: int = 7,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    if not supabase_client:
        return []
    try:
        query = supabase_client.table("discovery_candidates").select("*").order("created_at", desc=True).limit(limit)
        if symbol:
            query = query.eq("symbol", symbol.upper().strip())
        if theme_key:
            query = query.eq("theme_key", theme_key)
        response = query.execute()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_rows = []
        for row in response.data:
            created_at = row.get("created_at")
            if not created_at:
                continue
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_dt >= cutoff:
                    recent_rows.append(row)
            except Exception:
                recent_rows.append(row)
        return recent_rows
    except Exception as e:
        logger.error(f"Error fetching recent discovery candidates: {e}")
        return []


# ==============================================================================
# Chat logs helper
# ==============================================================================

def log_chat_message(user_id: str, role: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    if not supabase_client:
        return False
    try:
        payload = {
            "user_id": str(user_id),
            "role": role,
            "message": message,
            "metadata": metadata if metadata else {},
        }
        supabase_client.table("chat_logs").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging chat message: {e}")
        return False


# ==============================================================================
# User Preferences Helpers
# ==============================================================================

def get_user_preference(key: str) -> Optional[Any]:
    if not supabase_client:
        return None
    try:
        response = supabase_client.table("user_preferences").select("value").eq("key", key).execute()
        return response.data[0]["value"] if response.data else None
    except Exception as e:
        logger.error(f"Error fetching preference {key}: {e}")
        return None


def save_user_preference(key: str, value: Any) -> bool:
    if not supabase_client:
        return False
    try:
        payload = {
            "key": key,
            "value": value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        supabase_client.table("user_preferences").upsert(payload).execute()
        logger.info(f"Saved user preference: {key}")
        return True
    except Exception as e:
        logger.error(f"Error saving preference {key}: {e}")
        return False
