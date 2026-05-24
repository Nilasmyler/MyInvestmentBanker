import os
import logging
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv
from database.supabase_client import (
    fetch_portfolio, 
    update_portfolio_holding, 
    save_investment_thesis, 
    fetch_investment_thesis,
    get_user_preference,
    save_user_preference
)

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.communication")
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Toggle to gemini-3.5-flash / gemini-2.5-flash in .env

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_llm_response(prompt: str, system_instruction: str = "") -> str:
    """Helper to query Gemini with system instructions."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured in .env."
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        return f"Error executing model reasoning step: {e}"


class CommunicationAgent:
    """
    Handles user commands, portfolio mutations, and compiles final financial reports 
    into premium Telegram Markdown format.
    """
    
    @staticmethod
    def parse_user_command(user_id: str, message_text: str) -> str:
        """
        Parses text commands and updates database states accordingly.
        Supported:
        - /portfolio : View entire portfolio
        - /add TICKER PRICE QTY : Log a purchase
        - /remove TICKER : Delete ticker holding
        - /thesis TICKER text : Log an investment thesis
        """
        tokens = message_text.strip().split()
        if not tokens:
            return "Command empty. Send /start or /help to see options."
            
        cmd = tokens[0].lower()
        
        if cmd in ["/start", "/help"]:
            return (
                "👋 **Welcome to MyInvestmentBanker**!\n\n"
                "Your 24/7 personal, cloud-hosted AI wealth manager is active.\n"
                "I track your holdings, digest economic indices, and scan SEC disclosures for threats/opportunities.\n\n"
                "📋 **Available Commands:**\n"
                "• `/portfolio` — View allocations and cost-basis\n"
                "• `/add TICKER PRICE QTY` — e.g. `/add AAPL 175.50 10`\n"
                "• `/remove TICKER` — e.g. `/remove AAPL`\n"
                "• `/thesis TICKER ...` — Log your investment thesis\n"
                "• `/policy` — View your investment policy & target sectors\n"
                "• `/policy update ...` — e.g. `/policy update software and biotech only`\n"
                "• `/discover` — Scan ETFs for new investment opportunities\n"
                "• `/update` — Trigger an on-demand portfolio risk digest\n"
            )
            
        elif cmd == "/portfolio":
            holdings = fetch_portfolio()
            if not holdings:
                return "📂 **Your portfolio is currently empty.** Add holdings with `/add TICKER PRICE QTY`."
                
            report = "💼 **Active Portfolio Holdings:**\n\n"
            total_cost = 0.0
            
            for idx, h in enumerate(holdings):
                ticker = h["symbol"]
                qty = float(h["quantity"])
                price = float(h["cost_basis"])
                cost = qty * price
                total_cost += cost
                
                # Check for active thesis
                thesis = fetch_investment_thesis(ticker)
                thesis_flag = "📝 Has Thesis" if thesis else "❌ No Thesis"
                
                report += f"{idx+1}. **{ticker}**\n"
                report += f"   • Position: `{qty:.2f}` shares @ `{price:.2f}` USD\n"
                report += f"   • Total Cost: `{cost:.2f}` USD | {thesis_flag}\n\n"
                
            report += f"📊 **Aggregate Cost Basis: `{total_cost:.2f}` USD**"
            return report
            
        elif cmd == "/add":
            if len(tokens) < 4:
                return "⚠️ **Usage error:** Use `/add TICKER PRICE QTY` (e.g. `/add MSFT 420.00 5`)"
            
            ticker = tokens[1].upper()
            try:
                price = float(tokens[2])
                qty = float(tokens[3])
            except ValueError:
                return "⚠️ **Format error:** Price and Quantity must be numerical values."
                
            success = update_portfolio_holding(ticker, qty, price)
            if success:
                return f"✅ **Holdings updated:** Added `{qty:.2f}` shares of **{ticker}** @ `{price:.2f}` USD."
            return "❌ **Database write failed.** Check logs."
            
        elif cmd == "/remove":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/remove TICKER` (e.g. `/remove MSFT`)"
                
            ticker = tokens[1].upper()
            success = update_portfolio_holding(ticker, 0, 0) # Set to 0 deletes holding
            if success:
                return f"✅ **Holdings updated:** Completely removed **{ticker}** from your active portfolio."
            return "❌ **Database write failed.** Check logs."
            
        elif cmd == "/thesis":
            if len(tokens) < 3:
                return "⚠️ **Usage error:** Use `/thesis TICKER [detailed thesis text...]`"
                
            ticker = tokens[1].upper()
            thesis_text = " ".join(tokens[2:])
            
            # Check if ticker actually exists in holdings first
            holdings = fetch_portfolio()
            tickers = [h["symbol"] for h in holdings]
            if ticker not in tickers:
                return f"⚠️ **Portfolio mismatch:** Add **{ticker}** to holdings first using `/add` before saving a thesis."
                
            success = save_investment_thesis(ticker, thesis_text)
            if success:
                return f"✅ **Thesis logged:** Successfully stored personal thesis and generated RAG vector for **{ticker}**."
            return "❌ **Database write failed.** Check logs."
            
        elif cmd == "/policy":
            if len(tokens) > 1 and tokens[1].lower() == "update":
                if len(tokens) < 3:
                    return "⚠️ **Usage error:** Use `/policy update [detailed guidelines...]`"
                
                new_policy_text = " ".join(tokens[2:])
                
                system_instruction = (
                    "You are the Portfolio Manager Front Desk of MyInvestmentBanker.\n"
                    "Your job is to parse the user's investment policy update, extract the core rules, "
                    "and select 1 or 2 matching thematic sector ETFs (e.g. SMH for semiconductors, "
                    "IGV for software, XBI for biotech, XLV for healthcare, XLF for finance, XLI for industrials, XLY for consumer discretionary, XLP for consumer staples).\n"
                    "Output a valid JSON matching this schema: {\"policy_text\": \"...\", \"matching_etfs\": [\"ETF1\", \"ETF2\"]}"
                )
                
                prompt = f"Please structure this investment policy update: '{new_policy_text}'"
                structured_response = generate_llm_response(prompt, system_instruction)
                
                try:
                    import json
                    clean_res = structured_response.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_res)
                    
                    policy_text = parsed.get("policy_text", new_policy_text)
                    matching_etfs = parsed.get("matching_etfs", ["SMH", "IGV"])
                    
                    save_user_preference("broad_investment_policy", {"policy_text": policy_text})
                    save_user_preference("active_opportunity_focus", matching_etfs)
                    
                    return (
                        f"✅ **Policy Focus Updated successfully!**\n\n"
                        f"🔍 **New Policy Guidelines:**\n`{policy_text}`\n\n"
                        f"🎯 **Target Screeners (ETFs):** `{', '.join(matching_etfs)}`"
                    )
                except Exception as e:
                    logger.error(f"Error parsing Gemini policy JSON: {e}. Raw response: {structured_response}")
                    save_user_preference("broad_investment_policy", {"policy_text": new_policy_text})
                    save_user_preference("active_opportunity_focus", ["SMH", "IGV"])
                    return f"✅ **Policy Guidelines logged:** Saved raw rules. Default ETFs active."
            
            policy = get_user_preference("broad_investment_policy")
            focus = get_user_preference("active_opportunity_focus")
            
            policy_text = policy.get("policy_text") if isinstance(policy, dict) else policy or "No broad policy set. Using high-margin growth stock defaults."
            focus_etfs = focus if isinstance(focus, list) else ["SMH", "IGV"]
            
            report = "📋 **Active Investment Policy & Thematic Focus:**\n\n"
            report += f"🔍 **Broad Policy Guidelines:**\n`{policy_text}`\n\n"
            report += f"🎯 **Active Screener Focus (ETFs):** `{', '.join(focus_etfs)}`\n\n"
            report += "💡 *To update your investment focus, send* `/policy update [new guidelines...]`"
            return report

        elif cmd == "/discover":
            return "🔍 **MyInvestmentBanker**: Starting autonomous opportunity discovery. Screeners are active. Please hold..."
            
        return "❓ **Unknown command.** Send `/help` to see list of options."

    @staticmethod
    def compile_synthesis_report(portfolio_state: List[Dict[str, Any]], 
                                  macro_data: Dict[str, Any], 
                                  cfa_memos: Dict[str, Any], 
                                  risk_memos: Dict[str, Any]) -> str:
        """
        Synthesizes the output of the CFA and Risk Officer agents into a unified, 
        beautifully-formatted Telegram bulletin report.
        """
        system_instruction = (
            "You are the Lead Portfolio Synthesis Manager of MyInvestmentBanker.\n"
            "Your job is to compile financial insights from your analyst agents into a premium, "
            "actionable personal briefing. Focus on clarity, visual hierarchy, and analytical depth.\n"
            "Always format in clean Telegram-compatible Markdown. Avoid long text blocks; use bulleted summaries."
        )
        
        prompt = (
            f"Please synthesize the following data streams into a professional personal investment briefing.\n\n"
            f"=== 1. Active Portfolio State ===\n{portfolio_state}\n\n"
            f"=== 2. Macroeconomic Indicators ===\n{macro_data}\n\n"
            f"=== 3. CFA Analyst Memos (Filing Reviews) ===\n{cfa_memos}\n\n"
            f"=== 4. Portfolio Risk & Valuation Reviews ===\n{risk_memos}\n\n"
            f"Structure the final output as follows:\n"
            f"1. 📈 **Executive Synthesis**: A high-level overview of the health of the holdings given today's macro conditions.\n"
            f"2. 🛡️ **Critical Vulnerabilities & Dangers**: Highly prioritized risks (debt structures, concentration, regulatory, or margin pressures).\n"
            f"3. 💡 **Opportunities & Valuations**: Highlight target buy ranges (Margin of Safety) or thesis-validation milestones.\n"
            f"4. 📅 **Upcoming Dates**: Notable dates (earnings calendar) for the portfolio."
        )
        
        return generate_llm_response(prompt, system_instruction)

    @staticmethod
    def compile_opportunity_briefing(discovery_memo: str) -> str:
        """
        Compiles the Risk Agent's opportunity screening memo into a premium, beautifully-formatted 
        Telegram opportunity discovery bulletin.
        """
        system_instruction = (
            "You are the Lead Portfolio Synthesis Manager of MyInvestmentBanker.\n"
            "Your job is to format the Risk Officer's raw candidate screening memo into a premium, "
            "actionable Investment Opportunity Discovery briefing for the user.\n"
            "Use Markdown formatting, dynamic emojis, and bulleted lists. Always highlight how the "
            "user can add these candidates to their portfolio or watchlist in one click using commands."
        )
        
        prompt = (
            f"Please compile the following raw Risk Officer screening memo into a professional opportunity discovery bulletin.\n\n"
            f"{discovery_memo}\n\n"
            f"Conclude the bulletin with a clear, visual 'Action Command' block demonstrating how the user can act on these recommendations "
            f"(e.g., using `/add TICKER PRICE QTY` or `/watchlist add TICKER`)."
        )
        
        return generate_llm_response(prompt, system_instruction)

    @staticmethod
    def pm_plan_discovery_run(policy_text: str, current_focus: List[str], market_context: Dict[str, Any]) -> str:
        """
        The PM Planning Engine. Analyzes policy constraints, current thematic focuses,
        and real-time market/macro conditions using Gemini.
        Returns a structured JSON decision explaining whether a deep discovery screen is triggered.
        """
        logger.info("PM Planning: Evaluating market environment for potential discovery runs...")
        
        system_instruction = (
            "You are the Lead Portfolio Manager (PM) of MyInvestmentBanker.\n"
            "Your job is to strategically decide/plan whether to trigger an autonomous opportunity discovery run today.\n"
            "You must prioritize high-conviction thematic alignments, avoiding low-probability noise.\n"
            "Trigger a screening ONLY if:\n"
            "- A major macro indicator (FRED) has shifted in a way that supports a sector (e.g., lower rates supporting growth sectors like Tech/Biotech).\n"
            "- A monitored thematic sector ETF shows strong positive daily momentum (> 2% increase) or notable volume breaks.\n"
            "- The portfolio lacks exposure in a high-potential sector.\n\n"
            "Output your decision in a strict, valid JSON format matching this schema:\n"
            "{\n"
            "  \"trigger_discovery\": true_or_false,\n"
            "  \"target_sectors\": [\"ETF1\", \"ETF2\"],\n"
            "  \"reasoning\": \"A highly analytical, detailed professional explanation of your strategic planning decision.\"\n"
            "}"
        )
        
        prompt = (
            f"Please conduct a strategic PM opportunity planning review.\n\n"
            f"=== 1. Broad Investment Policy ===\n{policy_text}\n\n"
            f"=== 2. Monitored Thematic Focus ETFs ===\n{current_focus}\n\n"
            f"=== 3. Real-Time Market & Macro Context ===\n{market_context}\n\n"
            f"Determine if we should execute a deep discovery screening run today. Respond with the strict JSON payload only."
        )
        
        return generate_llm_response(prompt, system_instruction)
