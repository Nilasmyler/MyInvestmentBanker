import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
from database.supabase_client import fetch_historical_memos, save_analyst_memo
from agents.communication_agent import generate_llm_response

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.cfa")
logging.basicConfig(level=logging.INFO)


class CFAAgent:
    """
    Acts as a Chartered Financial Analyst (CFA).
    Performs fundamental balance sheet and cash flow calculations.
    Compares current quarterly disclosures with historical notes retrieved from Supabase.
    """
    
    @staticmethod
    def run(symbol: str, current_filing_data: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes deep-dive fundamental review.
        1. Fetch past two quarters' memos.
        2. Query Gemini 3.5 Flash to compute credit, liquidity, and operational metrics.
        3. Compare current statements against historical numbers to spot trends.
        4. Save the finalized memo to the Supabase database.
        """
        symbol = symbol.upper().strip()
        logger.info(f"CFA Agent: Performing balance sheet audit for {symbol}...")
        
        # 1. Fetch Historical Memos
        past_memos = fetch_historical_memos(symbol, limit=2)
        historical_context = ""
        if past_memos:
            for idx, memo in enumerate(past_memos):
                historical_context += f"--- Historical Analyst Memo {idx+1} ({memo['period']}) ---\n{memo['memo_text']}\n\n"
        else:
            historical_context = "No previous analyst memos exist for this asset in the database. This is a baseline analysis."
            
        # 2. Steer Gemini 3.5 Flash to perform quantitative auditing
        system_instruction = (
            "You are the Chartered Financial Analyst (CFA) Agent of MyInvestmentBanker.\n"
            "Your tone is strictly quantitative, objective, and precise.\n"
            "- Isolate all math calculations into explicit steps (Chain-of-Thought).\n"
            "- Compute FCF yields, debt-to-equity, operational margins, and interest coverage.\n"
            "- Cross-reference current metrics against the historical analyst memos provided.\n"
            "- Identify if inventory turns or accounts receivable collection speeds have worsened."
        )
        
        prompt = (
            f"Please conduct a comprehensive fundamental audit for **{symbol}**.\n\n"
            f"=== 1. Current SEC Filing & Accession Data ===\n{current_filing_data}\n\n"
            f"=== 2. Market Pricing & Valuation Data ===\n{market_data}\n\n"
            f"=== 3. Historical Analyst Memory (Past Quarters) ===\n{historical_context}\n\n"
            f"Structure your response strictly as a structured analyst report containing:\n"
            f"- **Period Under Review**: (e.g. Q1 2026 or Event Volatility Review)\n"
            f"- **Mathematical Core Calculations**: (Show Step-by-Step Chain-of-Thought for Liquidity and Credit ratios)\n"
            f"- **Longitudinal Trends**: (Highlight changes compared to historical memos)\n"
            f"- **CFA Synthesis & Verdict**: (Objective strength of operational performance)"
        )
        
        memo_text = generate_llm_response(prompt, system_instruction)
        logger.info(f"CFA Agent: Audit memo generated for {symbol}.")
        
        # 3. Extract basic structured key-value metrics via a quick LLM parse (to save as JSONB metadata)
        metrics_prompt = (
            f"Based on the analyst memo below, extract a clean JSON dictionary of fundamental metrics.\n"
            f"Keys to extract if available: 'debt_to_equity', 'fcf_margin_pct', 'pe_ratio', 'interest_coverage_ratio'.\n"
            f"Return ONLY raw JSON. No markdown ticks. No extra text.\n\n"
            f"Memo:\n{memo_text}"
        )
        
        raw_json = generate_llm_response(metrics_prompt, "Extract exact JSON key-values.")
        
        # Parse or clean JSON string
        import json
        metrics_dict = {}
        try:
            # Strip potential markdown wrapping
            cleaned_json = raw_json.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-3]
            metrics_dict = json.loads(cleaned_json.strip())
        except Exception as e:
            logger.warning(f"Could not parse metrics JSON: {e}. Storing empty dictionary.")
            
        # 4. Save to Database
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        period_str = f"Q{quarter}_{now.year}"
        save_analyst_memo(symbol, period_str, memo_text, metrics_dict)
        
        return {
            "symbol": symbol,
            "period": period_str,
            "memo_text": memo_text,
            "metrics": metrics_dict
        }
