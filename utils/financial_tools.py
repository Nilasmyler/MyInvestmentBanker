import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
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
    ticker = ticker.upper().strip()
    logger.info(f"Querying SEC EDGAR for ticker: {ticker}...")
    
    # 1. Fetch CIK
    cik_map = get_sec_cik_mapping()
    cik = cik_map.get(ticker)
    if not cik:
        logger.warning(f"Could not map ticker {ticker} to CIK. SEC Edgar query aborted.")
        return []
        
    # 2. Get Submissions
    headers = {"User-Agent": SEC_USER_AGENT}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            submissions = response.json()
            recent_filings = submissions.get("filings", {}).get("recent", {})
            
            filing_list = []
            for i in range(len(recent_filings.get("accessionNumber", []))):
                form_type = recent_filings["form"][i]
                
                # Filter down to primary disclosures: 10-K, 10-Q, and 8-K
                if form_type not in ["10-K", "10-Q", "8-K"]:
                    continue
                    
                acc_num = recent_filings["accessionNumber"][i].replace("-", "")
                doc_name = recent_filings["primaryDocument"][i]
                filing_date = recent_filings["filingDate"][i]
                
                # Build official direct SEC EDGAR URL
                sec_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num}/{doc_name}"
                
                filing_list.append({
                    "form": form_type,
                    "date": filing_date,
                    "report_url": sec_url,
                    "description": recent_filings["reportDescription"][i] if "reportDescription" in recent_filings else f"Form {form_type} filing"
                })
                
                if len(filing_list) >= limit:
                    break
            return filing_list
    except Exception as e:
        logger.error(f"Error querying SEC submissions for {ticker} (CIK: {cik}): {e}")
        
    return []


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
            "notes": "Static baseline loaded. Configure FRED_API_KEY in .env for active daily updates."
        }
        
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    indicators = {
        "fed_funds": "FEDFUNDS",  # Fed Funds Rate
        "cpi": "CPIAUCSL",       # CPI Inflation
        "gdp": "A191RL1Q225SBEA" # Real GDP Growth %
    }
    
    results = {}
    for name, series_id in indicators.items():
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                obs = data.get("observations", [])
                if obs:
                    results[name] = {
                        "date": obs[0]["date"],
                        "value": f"{float(obs[0]['value']):.2f}%" if name != "cpi" else obs[0]['value']
                    }
        except Exception as e:
            logger.error(f"Failed to fetch FRED series {series_id}: {e}")
            
    # Compile results
    return {
        "source": "FRED Live Feeds",
        "fed_funds_rate": results.get("fed_funds", {}).get("value", "N/A"),
        "cpi_inflation_index": results.get("cpi", {}).get("value", "N/A"),
        "gdp_growth_rate": results.get("gdp", {}).get("value", "N/A")
    }


# ==============================================================================
# yfinance Data Client with Robust Exponential Backoff
# ==============================================================================

def get_stock_price_and_history(ticker: str, retries: int = 3) -> Dict[str, Any]:
    """
    Fetches real-time price, 52-week highs, and basic historical stats from Yahoo Finance.
    Implements robust exponential backoff to handle and prevent 429 throttling.
    """
    ticker = ticker.upper().strip()
    delay = 1.0
    
    for attempt in range(retries):
        try:
            logger.info(f"Querying yfinance for price data ({ticker}) [Attempt {attempt + 1}/{retries}]...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract high-value, clean metrics
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
            prev_close = info.get("previousClose")
            
            price_change = 0.0
            price_change_pct = 0.0
            if current_price and prev_close:
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                
            return {
                "ticker": ticker,
                "current_price": current_price,
                "previous_close": prev_close,
                "day_change_dollar": round(price_change, 2) if current_price else None,
                "day_change_pct": round(price_change_pct, 2) if current_price else None,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0.0,
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("volume")
            }
        except Exception as e:
            logger.warning(f"yfinance lookup error for {ticker} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                logger.info(f"Sleeping {delay}s before retry...")
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff
                
    # Basic API fallback if yfinance completely blocks us
    return {
        "ticker": ticker,
        "current_price": None,
        "error": "Data fetch failed. yfinance rate limit or scrap blocks encountered."
    }


# ==============================================================================
# ETF Holdings and Fundamental Candidate Screening
# ==============================================================================

def fetch_etf_holdings(etf_symbol: str) -> List[str]:
    """
    Fetches the top 10 holding symbols of an ETF using yfinance's funds_data.
    Returns a list of clean uppercase tickers.
    """
    etf_symbol = etf_symbol.upper().strip()
    try:
        logger.info(f"Fetching ETF holdings for: {etf_symbol}")
        ticker = yf.Ticker(etf_symbol)
        data = ticker.funds_data
        holdings = data.top_holdings
        # top_holdings is a pandas DataFrame with index representing the Tickers
        if holdings is not None and not holdings.empty:
            tickers = [str(t).upper().strip() for t in holdings.index if t]
            logger.info(f"Discovered {len(tickers)} holdings for {etf_symbol}: {tickers}")
            return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch holdings for {etf_symbol}: {e}")
    return []


def screen_ticker_fundamentals(ticker: str, retries: int = 2) -> Optional[Dict[str, Any]]:
    """
    Fetches key fundamental screening metrics for a candidate ticker from Yahoo Finance.
    Returns structured data for the discovery screener.
    """
    ticker = ticker.upper().strip()
    delay = 1.0
    
    for attempt in range(retries):
        try:
            logger.info(f"Screening candidate ticker {ticker} [Attempt {attempt + 1}/{retries}]...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamental discovery stats
            operating_margins = info.get("operatingMargins")
            debt_to_equity = info.get("debtToEquity")
            free_cashflow = info.get("freeCashflow")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice")
            
            return {
                "symbol": ticker,
                "name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "current_price": current_price,
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "operating_margin": round(operating_margins * 100, 2) if operating_margins is not None else None,
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
                "free_cashflow": free_cashflow,
                "beta": info.get("beta"),
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else 0.0,
            }
        except Exception as e:
            logger.warning(f"Error screening candidate {ticker} on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2.0
                
    return None
