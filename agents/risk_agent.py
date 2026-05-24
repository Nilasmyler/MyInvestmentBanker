import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from database.supabase_client import fetch_investment_thesis
from agents.communication_agent import generate_llm_response

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.risk")
logging.basicConfig(level=logging.INFO)


class RiskAgent:
    """
    Acts as the Macro & Portfolio Risk Officer.
    Analyzes asset correlations, sector concentrations, FRED macro metrics,
    and correlates incoming corporate data against your personal investment thesis.
    """
    
    @staticmethod
    def run(portfolio: List[Dict[str, Any]], 
            macro_data: Dict[str, Any], 
            cfa_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes macro and portfolio allocation auditing.
        1. Parse portfolio allocations (sectors, weightings).
        2. Correlate FRED inflation/rate figures to asset exposures.
        3. Match CFA analyst notes against saved personal investment theses.
        4. Summarize margin of safety risks and opportunism.
        """
        logger.info("Risk Agent: Conducting portfolio threat and thesis correlation audit...")
        
        # 1. Gather Theses Context
        theses_context = ""
        for h in portfolio:
            symbol = h["symbol"].upper()
            thesis = fetch_investment_thesis(symbol)
            if thesis:
                theses_context += f"--- User Thesis on {symbol} ---\n{thesis['thesis_text']}\n\n"
            else:
                theses_context += f"--- User Thesis on {symbol} ---\nNo thesis logged for this asset. Thesis validation checks skipped.\n\n"
                
        # 2. Steer Gemini 3.5 Flash to act as a skeptical Risk Officer
        system_instruction = (
            "You are the Macro & Portfolio Risk Officer of MyInvestmentBanker.\n"
            "Your tone is skeptical, cautious, and highly analytical.\n"
            "- Evaluate interest rate sensitivity (floating debt risk) and inflation vulnerabilities.\n"
            "- Check for sector concentration and overall asset correlation.\n"
            "- Contrast actual quarterly numbers from the CFA reports with the user's saved theses.\n"
            "- Explicitly point out where a user's thesis is validated or broken (thesis-drift)."
        )
        
        prompt = (
            f"Please conduct a portfolio risk and macro alignment review.\n\n"
            f"=== 1. Active Portfolio Holdings ===\n{portfolio}\n\n"
            f"=== 2. Macroeconomic Environment ===\n{macro_data}\n\n"
            f"=== 3. CFA Analyst Quarterly Reports ===\n{cfa_reports}\n\n"
            f"=== 4. User Investment Theses ===\n{theses_context}\n\n"
            f"Output a formal risk memo structured as follows:\n"
            f"- **Macro Threat Alignment**: How macroeconomic trends (interest rates, CPI) specifically pressure or support our holdings.\n"
            f"- **Thesis Validation & Drift**: Flag specific assets where thesis objectives are being missed or drifted from.\n"
            f"- **Margin of Safety & Allocation Audits**: Evaluate leverage metrics across holdings, highlighting structural exposures."
        )
        
        risk_memo = generate_llm_response(prompt, system_instruction)
        logger.info("Risk Agent: Portfolio threat assessment memo generated.")
        
        return {
            "risk_memo": risk_memo
        }

    @staticmethod
    def audit_discovery_candidates(candidates: List[Dict[str, Any]], policy_text: str) -> str:
        """
        Audits a list of screened candidates against the user's investment policy.
        Returns a skeptical, analytical recommendation review outlining why these companies
        pass or fail the standards, and selecting the top 1 or 2 best opportunities.
        """
        logger.info(f"Risk Agent: Conducting policy screening audit on {len(candidates)} candidates...")
        
        system_instruction = (
            "You are the Macro & Portfolio Risk Officer of MyInvestmentBanker.\n"
            "Your tone is skeptical, conservative, and highly analytical.\n"
            "Your job is to audit new investment opportunities against the user's Broad Investment Policy.\n"
            "- Reject companies with low margins, excessive debt, or weak free cash flow unless the policy specifically allows them.\n"
            "- Demand a strong Margin of Safety (valuing actual operating cash flow, not projections).\n"
            "- Recommend the top 1 or 2 strongest prospects, explaining their primary tailwinds and structural risks."
        )
        
        prompt = (
            f"Please conduct an investment opportunity screening audit.\n\n"
            f"=== 1. User Broad Investment Policy ===\n{policy_text}\n\n"
            f"=== 2. Screened Candidate Companies ===\n{candidates}\n\n"
            f"Please evaluate these candidates. Output your final audit memo structured exactly as follows:\n"
            f"- **Policy Fit & Screening Summary**: A brief summary of which candidates were rejected and why.\n"
            f"- **Primary Recommendation(s)**: Highlight the top 1 or 2 candidates that best fit the policy, detailing their business strength.\n"
            f"- **Key Structural Risks**: Skeptically highlight the primary risks/vulnerabilities of the recommended companies (e.g., valuation, sector competition)."
        )
        
        return generate_llm_response(prompt, system_instruction)
