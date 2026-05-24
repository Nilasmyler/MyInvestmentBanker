import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from utils.financial_tools import get_stock_price_and_history, fetch_recent_sec_filings, fetch_macro_indicators
from utils.prompt_compressor import compress_financial_text
from agents.communication_agent import generate_llm_response

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.scout")
logging.basicConfig(level=logging.INFO)


class ScoutAgent:
    """
    Ingests raw economic indices, pricing data, news, and SEC filing URLs.
    Performs noise filtering and prompt compaction on bulky payloads.
    """
    
    @staticmethod
    def run(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes daily data acquisition.
        For each ticker in holdings:
        1. Fetch price and valuation ratios (yfinance).
        2. Fetch recent official SEC Edgar filing links.
        3. Fetch macroeconomic data (FRED).
        4. Apply LLMLingua / Extractive compaction to bulky filing data.
        """
        if not portfolio:
            logger.info("Portfolio empty. Ingestion aborted.")
            return {"macro": fetch_macro_indicators(), "tickers": {}}
            
        logger.info("Scout Agent: Starting live data acquisition flow...")
        
        # 1. Fetch Macro
        macro_stats = fetch_macro_indicators()
        
        ticker_payloads = {}
        for item in portfolio:
            symbol = item["symbol"].upper()
            
            # Fetch Stock Stats
            market_data = get_stock_price_and_history(symbol)
            
            # Fetch SEC Filings
            recent_filings = fetch_recent_sec_filings(symbol, limit=2)
            
            # If filings exist, we grab metadata. In a live system, we fetch raw report text
            # from the SEC link, compress it using prompt_compressor, and pass it forward.
            # To simulate this, we download and compress the filing metadata & descriptions.
            filings_report = ""
            for f in recent_filings:
                filings_report += f"Form: {f['form']} | Date: {f['date']} | URL: {f['report_url']}\nDescription: {f['description']}\n\n"
                
            # Perform prompt compaction on the filing report text
            compressed_filings = compress_financial_text(
                context=filings_report,
                target_token=300,
                instruction="Filter out generic boilerplate language, saving CIK mappings and key accession IDs."
            )
            
            ticker_payloads[symbol] = {
                "market_data": market_data,
                "recent_filings": compressed_filings,
                "raw_filings_list": recent_filings
            }
            
        logger.info("Scout Agent: Ingestion and compaction tasks complete.")
        
        return {
            "macro": macro_stats,
            "tickers": ticker_payloads
        }

    @staticmethod
    def filter_noise(news_headlines: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """
        Steers Gemini 3.5 Flash to evaluate incoming headlines and filter out noise,
        retaining only highly material market developments.
        """
        if not news_headlines:
            return []
            
        system_instruction = (
            "You are the Data Scout & Noise Filter Agent of MyInvestmentBanker.\n"
            "Your job is to read stock market news and discard non-material noise (price speculation, "
            "generic newsletters, repeating press releases, market sentiment summaries).\n"
            "Retain ONLY developments that directly impact sales, operating margins, leverage, or regulatory moats."
        )
        
        prompt = (
            f"Please read the following news headlines for **{symbol}**.\n"
            f"Filter out any speculative noise, and return a clean, structured bullet-point list of material developments.\n"
            f"News items:\n{news_headlines}"
        )
        
        filtered_text = generate_llm_response(prompt, system_instruction)
        logger.info(f"News filtered successfully for ticker {symbol}.")
        
        return [{"symbol": symbol, "material_summaries": filtered_text}]
