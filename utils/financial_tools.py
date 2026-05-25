import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
import yfinance as yf
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Initialize logging
logger = logging.getLogger("MyInvestmentBanker.financial_tools")
logging.basicConfig(level=logging.INFO)

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "MyInvestmentBanker/1.0 admin@example.com")

ETF_HOLDINGS_FALLBACKS: Dict[str, List[str]] = {
    "SMH": ["NVDA", "TSM", "AVGO", "ASML", "AMD", "QCOM", "AMAT", "LRCX", "MU", "ADI"],
    "IGV": ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "INTU", "PANW", "CRWD", "SNPS", "CDNS"],
    "HACK": ["PANW", "CRWD", "ZS", "FTNT", "OKTA", "CYBR", "S", "TENB"],
    "XBI": ["VRTX", "REGN", "GILD", "BIIB", "MRNA", "ALNY", "INCY", "BMRN"],
    "XLV": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR"],
    "XLI": ["GE", "CAT", "RTX", "ETN", "UNP", "PH", "HON", "DE"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "PSX"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS"],
}

OWNERSHIP_FORM_PREFIXES = ("SC 13D", "SC 13G", "13D", "13G")


def _safe_float(value: Any) -> Optional[float]:
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_pct(value: Any) -> Optional[float]:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return None
    if abs(numeric_value) <= 1:
        numeric_value *= 100
    return round(numeric_value, 2)


def _safe_int(value: Any) -> Optional[int]:
    if value in [None, ""]:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _format_table_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _normalize_table_records(table: Any, limit: int = 5) -> List[Dict[str, Any]]:
    if table is None or getattr(table, "empty", False):
        return []

    try:
        records = table.to_dict("records")
    except Exception:
        return []

    normalized: List[Dict[str, Any]] = []
    for record in records[:limit]:
        holder_name = ""
        for key in ["Holder", "holder", "Organization", "organization"]:
            if record.get(key):
                holder_name = str(record.get(key))
                break

        pct_out = None
        for key in ["% Out", "%Out", "pctHeld", "PctHeld"]:
            if record.get(key) is not None:
                pct_out = _normalize_pct(record.get(key))
                break

        shares = None
        for key in ["Shares", "shares", "Position", "position"]:
            if record.get(key) is not None:
                shares = _safe_int(record.get(key))
                break

        report_date = None
        for key in ["Date Reported", "dateReported", "Report Date", "reportDate"]:
            if record.get(key) is not None:
                report_date = _format_table_value(record.get(key))
                break

        normalized.append(
            {
                "holder": holder_name,
                "shares": shares,
                "pct_out": pct_out,
                "report_date": report_date,
            }
        )

    return normalized


def _fetch_sec_recent_submission_rows(ticker: str) -> List[Dict[str, Any]]:
    ticker = ticker.upper().strip()
    logger.info(f"Querying SEC EDGAR recent submissions for ticker: {ticker}...")

    cik_map = get_sec_cik_mapping()
    cik = cik_map.get(ticker)
    if not cik:
        logger.warning(f"Could not map ticker {ticker} to CIK. SEC Edgar query aborted.")
        return []

    headers = {"User-Agent": SEC_USER_AGENT}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"SEC submissions lookup failed for {ticker}: {response.status_code}")
            return []

        submissions = response.json()
        recent_filings = submissions.get("filings", {}).get("recent", {})
        filing_rows = []
        accession_numbers = recent_filings.get("accessionNumber", [])
        for index in range(len(accession_numbers)):
            form_type = str(recent_filings.get("form", [""])[index]).strip()
            accession_number = str(accession_numbers[index]).strip()
            accession_number_compact = accession_number.replace("-", "")
            document_name = str(recent_filings.get("primaryDocument", [""])[index]).strip()
            filing_date = str(recent_filings.get("filingDate", [""])[index]).strip()
            sec_url = ""
            if accession_number_compact and document_name:
                sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number_compact}/{document_name}"

            filing_rows.append(
                {
                    "form": form_type,
                    "date": filing_date,
                    "report_url": sec_url,
                    "description": recent_filings.get("reportDescription", [""] * len(accession_numbers))[index]
                    if "reportDescription" in recent_filings
                    else f"Form {form_type} filing",
                    "accession_number": accession_number,
                    "primary_document": document_name,
                }
            )
        return filing_rows
    except Exception as e:
        logger.error(f"Error querying SEC submissions for {ticker} (CIK: {cik}): {e}")
        return []


# ==============================================================================
# SEC Edgar Official REST API Wrapper
# ==============================================================================

def get_sec_cik_mapping() -> Dict[str, str]:
    """
    Fetches the official SEC ticker-to-CIK mapping database.
    Returns a dictionary of ticker -> padded 10-digit CIK string.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    url = "https://www.sec.gov/files/company_tickers.json"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            mapping = {}
            for item in data.values():
                ticker = item["ticker"].upper()
                cik = str(item["cik_str"]).zfill(10)
                mapping[ticker] = cik
            return mapping
    except Exception as e:
        logger.error(f"Error fetching SEC CIK mapping: {e}")
    return {}


def fetch_recent_sec_filings(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Directly queries the official SEC EDGAR API for a ticker's recent submissions.
    Returns structured list of filings (10-K, 10-Q, 8-K) with official accession numbers.
    """
    filing_list = []
    for row in _fetch_sec_recent_submission_rows(ticker):
        if row.get("form") not in ["10-K", "10-Q", "8-K"]:
            continue
        filing_list.append(row)
        if len(filing_list) >= limit:
            break
    return filing_list


def fetch_recent_sec_ownership_filings(ticker: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetches recent insider and beneficial ownership filings for a ticker from SEC submissions.
    """
    ownership_rows: List[Dict[str, Any]] = []
    for row in _fetch_sec_recent_submission_rows(ticker):
        form_type = str(row.get("form", "")).strip().upper()
        if form_type in {"3", "4", "5"}:
            ownership_rows.append({**row, "category": "insider"})
        elif form_type.startswith(OWNERSHIP_FORM_PREFIXES):
            ownership_rows.append({**row, "category": "beneficial_ownership"})
        if limit is not None and len(ownership_rows) >= limit:
            break
    return ownership_rows


# ==============================================================================
# FRED (Federal Reserve Macroeconomic Indicators) API
# ==============================================================================

def fetch_macro_indicators() -> Dict[str, Any]:
    """
    Queries FRED for key indicators: Fed Funds Rate, Inflation (CPI), and GDP.
    Includes a static robust fallback if FRED_API_KEY is not set.
    """
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY missing. Loading modern macroeconomic baseline values.")
        return {
            "source": "Static Economic Baseline (2026)",
            "fed_funds_rate": "5.25% - 5.50%",
            "cpi_inflation": "2.8%",
            "gdp_growth_rate": "2.1% (Annualized Q1 2026)",
            "yield_curve_status": "Inverted (10Y minus 2Y at -0.15%)",
            "notes": "Static baseline loaded. Configure FRED_API_KEY in .env for active daily updates.",
        }

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    indicators = {
        "fed_funds": "FEDFUNDS",
        "cpi": "CPIAUCSL",
        "gdp": "A191RL1Q225SBEA",
    }

    results = {}
    for name, series_id in indicators.items():
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                obs = data.get("observations", [])
                if obs:
                    results[name] = {
                        "date": obs[0]["date"],
                        "value": f"{float(obs[0]['value']):.2f}%" if name != "cpi" else obs[0]["value"],
                    }
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")

    return {
        "source": "FRED Live Feeds",
        "fed_funds_rate": results.get("fed_funds", {}).get("value", "N/A"),
        "cpi_inflation_index": results.get("cpi", {}).get("value", "N/A"),
        "gdp_growth_rate": results.get("gdp", {}).get("value", "N/A"),
    }


# ==============================================================================
# Market Data Helpers
# ==============================================================================

def _calculate_change_pct(current_value: Optional[float], prior_value: Optional[float]) -> Optional[float]:
    if current_value in [None, 0] or prior_value in [None, 0]:
        return None
    try:
        return round(((current_value - prior_value) / prior_value) * 100, 2)
    except Exception:
        return None


def _extract_historical_changes(history) -> Dict[str, Optional[float]]:
    if history is None or history.empty:
        return {"five_day_change_pct": None, "thirty_day_change_pct": None}

    closes = history["Close"].dropna().tolist()
    if not closes:
        return {"five_day_change_pct": None, "thirty_day_change_pct": None}

    current_close = closes[-1]
    five_day_close = closes[-6] if len(closes) >= 6 else closes[0]
    thirty_day_close = closes[-22] if len(closes) >= 22 else closes[0]
    return {
        "five_day_change_pct": _calculate_change_pct(current_close, five_day_close),
        "thirty_day_change_pct": _calculate_change_pct(current_close, thirty_day_close),
    }


def _calculate_target_premium_pct(target_price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    if target_price in [None, 0] or current_price in [None, 0]:
        return None
    try:
        return round(((target_price - current_price) / current_price) * 100, 2)
    except Exception:
        return None


def get_stock_price_and_history(ticker: str, retries: int = 3) -> Dict[str, Any]:
    """
    Fetches real-time price, short-horizon momentum, and basic company metadata from Yahoo Finance.
    Implements exponential backoff to reduce transient yfinance failures.
    """
    ticker = ticker.upper().strip()
    delay = 1.0

    for attempt in range(retries):
        try:
            logger.info(f"Querying yfinance for price data ({ticker}) [Attempt {attempt + 1}/{retries}]...")
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            history = stock.history(period="1mo", interval="1d", auto_adjust=False)

            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
            prev_close = info.get("previousClose")
            if current_price is None and history is not None and not history.empty:
                current_price = round(float(history["Close"].dropna().iloc[-1]), 2)
            if prev_close is None and history is not None and len(history.index) >= 2:
                prev_close = round(float(history["Close"].dropna().iloc[-2]), 2)

            price_change = None if current_price is None or prev_close is None else round(current_price - prev_close, 2)
            day_change_pct = _calculate_change_pct(current_price, prev_close)
            historical_changes = _extract_historical_changes(history)

            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "current_price": current_price,
                "previous_close": prev_close,
                "day_change_dollar": price_change,
                "day_change_pct": day_change_pct,
                "five_day_change_pct": historical_changes["five_day_change_pct"],
                "thirty_day_change_pct": historical_changes["thirty_day_change_pct"],
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "beta": info.get("beta"),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0.0,
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("volume"),
            }
        except Exception as e:
            logger.warning(f"yfinance lookup error for {ticker} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                logger.info(f"Sleeping {delay}s before retry...")
                time.sleep(delay)
                delay *= 2.0

    return {
        "ticker": ticker,
        "name": ticker,
        "current_price": None,
        "day_change_pct": None,
        "five_day_change_pct": None,
        "thirty_day_change_pct": None,
        "error": "Data fetch failed. yfinance rate limit or scrape blocks encountered.",
    }


def fetch_analyst_consensus(ticker: str, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Fetches lightweight analyst-consensus metadata from Yahoo Finance.
    """
    ticker = ticker.upper().strip()
    market_data = market_data or {}
    try:
        logger.info(f"Fetching analyst consensus for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        current_price = market_data.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean_price = _safe_float(info.get("targetMeanPrice"))
        recommendation_breakdown: Dict[str, Any] = {}

        try:
            summary_table = getattr(stock, "recommendations_summary", None)
            if summary_table is not None and not summary_table.empty:
                summary_records = summary_table.to_dict("records")
                if summary_records:
                    recommendation_breakdown = {
                        str(key): value
                        for key, value in summary_records[0].items()
                        if str(key).lower() != "period"
                    }
        except Exception as e:
            logger.debug(f"Could not parse recommendations summary for {ticker}: {e}")

        return {
            "symbol": ticker,
            "source": "yfinance",
            "analyst_count": _safe_int(info.get("numberOfAnalystOpinions")) or 0,
            "recommendation_key": str(info.get("recommendationKey", "") or "").lower(),
            "recommendation_mean": _safe_float(info.get("recommendationMean")),
            "target_mean_price": target_mean_price,
            "target_high_price": _safe_float(info.get("targetHighPrice")),
            "target_low_price": _safe_float(info.get("targetLowPrice")),
            "price_target_premium_pct": _calculate_target_premium_pct(target_mean_price, _safe_float(current_price)),
            "recommendation_breakdown": recommendation_breakdown,
        }
    except Exception as e:
        logger.warning(f"Analyst consensus lookup failed for {ticker}: {e}")
        return {
            "symbol": ticker,
            "source": "yfinance",
            "analyst_count": 0,
            "recommendation_key": "",
            "recommendation_mean": None,
            "target_mean_price": None,
            "target_high_price": None,
            "target_low_price": None,
            "price_target_premium_pct": None,
            "recommendation_breakdown": {},
            "error": str(e),
        }


def fetch_institutional_holder_snapshot(ticker: str, limit: int = 5) -> Dict[str, Any]:
    """
    Fetches a lightweight ownership snapshot from Yahoo Finance.
    """
    ticker = ticker.upper().strip()
    try:
        logger.info(f"Fetching ownership snapshot for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        institutional_holders = getattr(stock, "institutional_holders", None)
        top_holders = _normalize_table_records(institutional_holders, limit=limit)

        return {
            "symbol": ticker,
            "source": "yfinance",
            "institutional_ownership_pct": _normalize_pct(info.get("heldPercentInstitutions")),
            "insider_ownership_pct": _normalize_pct(info.get("heldPercentInsiders")),
            "top_holders": top_holders,
            "holder_count": len(top_holders),
        }
    except Exception as e:
        logger.warning(f"Ownership snapshot lookup failed for {ticker}: {e}")
        return {
            "symbol": ticker,
            "source": "yfinance",
            "institutional_ownership_pct": None,
            "insider_ownership_pct": None,
            "top_holders": [],
            "holder_count": 0,
            "error": str(e),
        }


# ==============================================================================
# News Helpers
# ==============================================================================

def _dedupe_news_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("headline"), item.get("url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_google_news_date(raw_date: str) -> str:
    if not raw_date:
        return datetime.now(timezone.utc).isoformat()
    try:
        return datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return raw_date


def _fetch_finnhub_news(symbol: str, days: int, limit: int) -> List[Dict[str, Any]]:
    if not FINNHUB_API_KEY:
        return []

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)
    params = {
        "symbol": symbol.upper().strip(),
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "token": FINNHUB_API_KEY,
    }
    try:
        response = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Finnhub news lookup failed for {symbol}: {response.status_code}")
            return []

        articles = response.json()
        news_items = []
        for item in articles[:limit]:
            news_items.append(
                {
                    "symbol": symbol.upper().strip(),
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "url": item.get("url"),
                    "published_at": datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc).isoformat()
                    if item.get("datetime")
                    else datetime.now(timezone.utc).isoformat(),
                    "source": item.get("source", "Finnhub"),
                }
            )
        return news_items
    except Exception as e:
        logger.warning(f"Finnhub news request failed for {symbol}: {e}")
        return []


def _fetch_google_news(query: str, limit: int) -> List[Dict[str, Any]]:
    if not query:
        return []

    rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(rss_url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Google News RSS lookup failed for query '{query}': {response.status_code}")
            return []

        root = ET.fromstring(response.text)
        items = []
        for item in root.findall(".//item")[:limit]:
            items.append(
                {
                    "headline": (item.findtext("title") or "").strip(),
                    "summary": (item.findtext("description") or "").strip(),
                    "url": (item.findtext("link") or "").strip(),
                    "published_at": _parse_google_news_date((item.findtext("pubDate") or "").strip()),
                    "source": "Google News RSS",
                }
            )
        return items
    except Exception as e:
        logger.warning(f"Google News RSS request failed for query '{query}': {e}")
        return []


def fetch_news(symbol: str, days: int = 7, limit: int = 10, company_name: str = "") -> List[Dict[str, Any]]:
    """
    Fetches recent news for a symbol.
    Finnhub is used when configured; otherwise Google News RSS is used as a fallback.
    """
    symbol = symbol.upper().strip()
    news_items = _fetch_finnhub_news(symbol, days=days, limit=limit)

    if not news_items:
        query = company_name or symbol
        news_items = _fetch_google_news(query, limit=limit)
        for item in news_items:
            item["symbol"] = symbol

    return _dedupe_news_items(news_items[:limit])


# ==============================================================================
# ETF Holdings and Discovery Helpers
# ==============================================================================

def fetch_etf_holdings(etf_symbol: str, limit: int = 10) -> List[str]:
    """
    Fetches the major holding symbols of an ETF using yfinance funds data.
    Falls back to a curated holdings list if live lookup fails.
    """
    etf_symbol = etf_symbol.upper().strip()
    try:
        logger.info(f"Fetching ETF holdings for: {etf_symbol}")
        ticker = yf.Ticker(etf_symbol)
        data = ticker.funds_data
        holdings = data.top_holdings
        if holdings is not None and not holdings.empty:
            tickers = [str(t).upper().strip() for t in holdings.index if t]
            logger.info(f"Discovered {len(tickers)} holdings for {etf_symbol}: {tickers}")
            return tickers[:limit]
    except Exception as e:
        logger.warning(f"Failed to fetch holdings for {etf_symbol}: {e}")

    return ETF_HOLDINGS_FALLBACKS.get(etf_symbol, [])[:limit]


def screen_ticker_fundamentals(ticker: str, retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Retained for compatibility with the previous discovery flow.
    """
    ticker = ticker.upper().strip()
    delay = 1.0

    for attempt in range(retries):
        try:
            logger.info(f"Screening candidate ticker {ticker} [Attempt {attempt + 1}/{retries}]...")
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            market_data = get_stock_price_and_history(ticker, retries=1)

            operating_margins = info.get("operatingMargins")
            debt_to_equity = info.get("debtToEquity")
            free_cashflow = info.get("freeCashflow")

            return {
                "symbol": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "current_price": market_data.get("current_price"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "operating_margin": round(operating_margins * 100, 2) if operating_margins is not None else None,
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
                "free_cashflow": free_cashflow,
                "beta": info.get("beta"),
                "dividend_yield": market_data.get("dividend_yield"),
                "day_change_pct": market_data.get("day_change_pct"),
                "five_day_change_pct": market_data.get("five_day_change_pct"),
                "thirty_day_change_pct": market_data.get("thirty_day_change_pct"),
            }
        except Exception as e:
            logger.warning(f"Error screening candidate {ticker} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2.0

    return None
