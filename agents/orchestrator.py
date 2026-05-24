import os
import logging
from typing import TypedDict, List, Dict, Any, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END

# Import our agents
from agents.scout_agent import ScoutAgent
from agents.cfa_agent import CFAAgent
from agents.risk_agent import RiskAgent
from agents.communication_agent import CommunicationAgent
from database.supabase_client import fetch_portfolio

logger = logging.getLogger("MyInvestmentBanker.orchestrator")
logging.basicConfig(level=logging.INFO)


# ==============================================================================
# 1. State Definition
# ==============================================================================

class AgentState(TypedDict):
    """
    Shared, thread-safe memory state passed across agent nodes in LangGraph.
    Uses Reducers (operator.add) to append metrics dynamically without losing context.
    """
    portfolio: List[Dict[str, Any]]
    raw_ingested_data: Dict[str, Any]
    cfa_analyst_memos: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    final_synthesis_report: str
    messages: List[Dict[str, str]]


# ==============================================================================
# 2. Node Functions
# ==============================================================================

def IngestionNode(state: AgentState) -> Dict[str, Any]:
    """Node: Scout Agent fetches pricing, filings, news, and macro stats."""
    logger.info("--- LANGGRAPH: Entering Ingestion Node ---")
    
    # Reload holdings directly from DB to get the most updated portfolio state
    portfolio = fetch_portfolio()
    
    # Run Scout Ingestions
    scout_payload = ScoutAgent.run(portfolio)
    
    return {
        "portfolio": portfolio,
        "raw_ingested_data": scout_payload,
        "cfa_analyst_memos": [], # Clear previous state
    }


def CFAAnalysisNode(state: AgentState) -> Dict[str, Any]:
    """Node: CFA Agent processes balance sheet & credit metrics for all holdings."""
    logger.info("--- LANGGRAPH: Entering CFA Fundamental Review Node ---")
    
    portfolio = state.get("portfolio", [])
    raw_data = state.get("raw_ingested_data", {})
    tickers_data = raw_data.get("tickers", {})
    
    memos_list = []
    for item in portfolio:
        symbol = item["symbol"].upper()
        ticker_stats = tickers_data.get(symbol, {})
        
        filing_text = ticker_stats.get("recent_filings", "No filing context available.")
        market_stats = ticker_stats.get("market_data", {})
        
        # Run deep fundamental calculations
        analyst_memo = CFAAgent.run(symbol, filing_text, market_stats)
        memos_list.append(analyst_memo)
        
    return {
        "cfa_analyst_memos": memos_list
    }


def PortfolioRiskNode(state: AgentState) -> Dict[str, Any]:
    """Node: Risk Officer evaluates allocations, macro exposures, and thesis validation."""
    logger.info("--- LANGGRAPH: Entering Portfolio Threat & Risk Audit Node ---")
    
    portfolio = state.get("portfolio", [])
    macro_stats = state.get("raw_ingested_data", {}).get("macro", {})
    cfa_memos = state.get("cfa_analyst_memos", [])
    
    # Run macro matching and thesis checks
    risk_payload = RiskAgent.run(portfolio, macro_stats, cfa_memos)
    
    return {
        "risk_assessment": risk_payload
    }


def FinalSynthesisNode(state: AgentState) -> Dict[str, Any]:
    """Node: Synthesis Lead formats and dispatches the compiled Telegram bulletin."""
    logger.info("--- LANGGRAPH: Entering Final Synthesis Node ---")
    
    portfolio = state.get("portfolio", [])
    macro_stats = state.get("raw_ingested_data", {}).get("macro", {})
    cfa_memos = state.get("cfa_analyst_memos", [])
    risk_memo = state.get("risk_assessment", {}).get("risk_memo", "")
    
    # Compile final briefing
    telegram_bulletin = CommunicationAgent.compile_synthesis_report(
        portfolio_state=portfolio,
        macro_data=macro_stats,
        cfa_memos=cfa_memos,
        risk_memos=risk_memo
    )
    
    return {
        "final_synthesis_report": telegram_bulletin
    }


# ==============================================================================
# 3. LangGraph Workflow Compilation
# ==============================================================================

def build_multi_agent_workflow():
    """
    Compiles the state transition nodes into a stateful, deterministic graph.
    """
    workflow = StateGraph(AgentState)
    
    # Declare Nodes
    workflow.add_node("Scout_Ingestion", IngestionNode)
    workflow.add_node("CFA_Analysis", CFAAnalysisNode)
    workflow.add_node("Risk_Audit", PortfolioRiskNode)
    workflow.add_node("Synthesis_Lead", FinalSynthesisNode)
    
    # Establish Edges / Transitions
    workflow.set_entry_point("Scout_Ingestion")
    
    workflow.add_edge("Scout_Ingestion", "CFA_Analysis")
    workflow.add_edge("CFA_Analysis", "Risk_Audit")
    workflow.add_edge("Risk_Audit", "Synthesis_Lead")
    workflow.add_edge("Synthesis_Lead", END)
    
    # Compile
    return workflow.compile()


# Instance for global exports
wealth_manager_graph = build_multi_agent_workflow()


def trigger_wealth_manager_run() -> str:
    """
    Triggers a full on-demand multi-agent pipeline run.
    Returns the finalized markdown bulletin.
    """
    logger.info("Starting automated wealth manager run...")
    
    initial_state = {
        "portfolio": [],
        "raw_ingested_data": {},
        "cfa_analyst_memos": [],
        "risk_assessment": {},
        "final_synthesis_report": "",
        "messages": []
    }
    
    try:
        final_state = wealth_manager_graph.invoke(initial_state)
        return final_state.get("final_synthesis_report", "Error: Pipeline failed to generate synthesis report.")
    except Exception as e:
        logger.error(f"Workflow pipeline invocation failed: {e}")
        return f"❌ **System Error during multi-agent analysis cycle:**\n\n`{e}`"


def trigger_opportunity_discovery(focus_override: List[str] = None) -> str:
    """
    Autonomous investment opportunity discovery workflow.
    Screen thematic ETFs matching dynamic active policy preferences,
    filter out current holdings & ignored history, run CFA fundamental screens,
    audit against investment policy using the Risk Agent, and synthesize
    a premium Telegram discovery briefing.
    """
    logger.info("Starting autonomous opportunity discovery run...")
    try:
        from datetime import datetime
        from database.supabase_client import get_user_preference, save_user_preference
        from utils.financial_tools import fetch_etf_holdings, screen_ticker_fundamentals
        
        # 1. Fetch active policy and opportunity focus
        policy = get_user_preference("broad_investment_policy")
        focus = get_user_preference("active_opportunity_focus")
        history = get_user_preference("discovery_history") or []
        
        policy_text = policy.get("policy_text") if isinstance(policy, dict) else policy
        if not policy_text:
            policy_text = (
                "Search for market-leading compounding businesses with strong high-margin defensive profiles. "
                "Minimum operating margin: 15%, stable free cash flow, and low debt leverage (Debt/Equity < 1.5)."
            )
            save_user_preference("broad_investment_policy", {"policy_text": policy_text})
            
        focus_etfs = focus_override if focus_override else (focus if isinstance(focus, list) else ["SMH", "IGV"])
        if not focus and not focus_override:
            save_user_preference("active_opportunity_focus", focus_etfs)
            
        # 2. Extract portfolio holdings to filter out
        portfolio = fetch_portfolio()
        portfolio_tickers = {h["symbol"].upper().strip() for h in portfolio}
        
        # Ignored/recommended history tickers
        history_tickers = {str(h.get("symbol")).upper().strip() for h in history if isinstance(h, dict) and h.get("symbol")}
        
        # 3. Pull candidates from focus ETFs
        logger.info(f"Targeting sector ETFs for screening: {focus_etfs}")
        candidates = []
        for etf in focus_etfs:
            holdings = fetch_etf_holdings(etf)
            for ticker in holdings:
                ticker_clean = ticker.upper().strip()
                if ticker_clean in portfolio_tickers or ticker_clean in history_tickers:
                    logger.info(f"Filtering out candidate {ticker_clean} (already in portfolio or discovery history)")
                    continue
                candidates.append(ticker_clean)
                
        # Remove duplicates
        candidates = list(set(candidates))
        
        if not candidates:
            logger.info("No new candidates found to screen after portfolio/history filtering. Using fallbacks.")
            fallback_map = {
                "SMH": ["NVDA", "ASML", "AMD", "AVGO", "LRCX"],
                "IGV": ["MSFT", "ADBE", "CRM", "NOW", "PANW"]
            }
            for etf in focus_etfs:
                for ticker in fallback_map.get(etf.upper(), []):
                    ticker_clean = ticker.upper().strip()
                    if ticker_clean not in portfolio_tickers and ticker_clean not in history_tickers:
                        candidates.append(ticker_clean)
            candidates = list(set(candidates))
            
        if not candidates:
            return "❌ **Discovery Complete**: No new candidates found. Adjust your active sector focus or reset your history."
            
        # Limit to top 5 candidates to screen to prevent yfinance rate limits
        candidates = candidates[:5]
        
        # 4. CFA Fundamental Screening
        screened_data = []
        for ticker in candidates:
            stats = screen_ticker_fundamentals(ticker)
            if stats:
                screened_data.append(stats)
                
        if not screened_data:
            return "❌ **Discovery Complete**: Screened candidates failed to return fundamental metrics. Please try again."
            
        # 5. Risk Agent Audit
        discovery_memo = RiskAgent.audit_discovery_candidates(screened_data, policy_text)
        
        # 6. Synthesize final briefing
        bulletin = CommunicationAgent.compile_opportunity_briefing(discovery_memo)
        
        # 7. Update discovery history to prevent repeated recommendations
        for candidate in screened_data:
            history.append({
                "symbol": candidate["symbol"],
                "timestamp": str(datetime.now().isoformat()),
                "action": "recommended"
            })
        save_user_preference("discovery_history", history)
        
        return bulletin
    except Exception as e:
        logger.error(f"Error during opportunity discovery run: {e}")
        return f"❌ **System Error during opportunity discovery run:**\n\n`{e}`"


def trigger_autonomous_discovery_check() -> Optional[str]:
    """
    Autonomous 24/7 background poller check.
    Gathers real-time macro indices and sector ETF momentum metrics,
    queries the PM (Communication Agent) to strategically decide whether to trigger
    an opportunity discovery run, and executes the screening if planned.
    Returns the briefing string if triggered, else None.
    """
    logger.info("Starting autonomous 24/7 PM opportunity planning check...")
    try:
        import json
        from database.supabase_client import get_user_preference, save_user_preference
        from utils.financial_tools import fetch_macro_indicators, get_stock_price_and_history
        
        # 1. Fetch Policy & Monitored Focus ETFs
        policy = get_user_preference("broad_investment_policy")
        focus = get_user_preference("active_opportunity_focus")
        
        policy_text = policy.get("policy_text") if isinstance(policy, dict) else policy
        if not policy_text:
            policy_text = (
                "Search for market-leading compounding businesses with strong high-margin defensive profiles. "
                "Minimum operating margin: 15%, stable free cash flow, and low debt leverage (Debt/Equity < 1.5)."
            )
            save_user_preference("broad_investment_policy", {"policy_text": policy_text})
            
        focus_etfs = focus if isinstance(focus, list) else ["SMH", "IGV"]
        if not focus:
            save_user_preference("active_opportunity_focus", focus_etfs)
            
        # 2. Gather market momentum signals
        logger.info("Gathering macroeconomic data and ETF price momentum...")
        macro_stats = fetch_macro_indicators()
        
        etf_metrics = {}
        for etf in focus_etfs:
            stats = get_stock_price_and_history(etf)
            if stats and "error" not in stats:
                etf_metrics[etf] = {
                    "current_price": stats.get("current_price"),
                    "day_change_pct": stats.get("day_change_pct"),
                    "dividend_yield": stats.get("dividend_yield")
                }
                
        market_context = {
            "macroeconomics": macro_stats,
            "etf_price_momentum": etf_metrics
        }
        
        # 3. Query the PM Planning Engine
        pm_decision_raw = CommunicationAgent.pm_plan_discovery_run(policy_text, focus_etfs, market_context)
        
        try:
            # Clean markdown code fences if present
            clean_res = pm_decision_raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(clean_res)
            
            trigger = decision.get("trigger_discovery", False)
            reason = decision.get("reasoning", "No detailed reasoning provided.")
            target_sectors = decision.get("target_sectors", focus_etfs)
            
            logger.info(f"PM 24/7 Planning Review completed. Trigger: {trigger}. Reasoning: {reason}")
            
            if trigger:
                logger.info(f"⚡ PM Planning triggered a deep discovery sweep for: {target_sectors}!")
                # Execute discovery
                briefing_content = trigger_opportunity_discovery(focus_override=target_sectors)
                
                # Prepend the PM's planned strategic reasoning to the alert report
                planned_bulletin = (
                    f"🤖💼 **Autonomous PM Discovery Event**\n\n"
                    f"💡 **PM Strategic Rationale for Trigger:**\n*\"{reason}\"*\n\n"
                    f"--- \n\n"
                    f"{briefing_content}"
                )
                return planned_bulletin
            else:
                logger.info("PM decided to stand down discovery loop for today. Silently exiting background poll.")
                return None
        except Exception as e:
            logger.error(f"Error parsing PM decision JSON: {e}. Raw response: {pm_decision_raw}")
            return None
    except Exception as e:
        logger.error(f"Error in autonomous 24/7 discovery poller check: {e}")
        return None
