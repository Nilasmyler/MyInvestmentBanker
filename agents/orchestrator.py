import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.cfa_agent import CFAAgent
from agents.communication_agent import CommunicationAgent
from agents.research_planner_agent import ResearchPlannerAgent
from agents.risk_agent import RiskAgent
from agents.scout_agent import ScoutAgent
from database.supabase_client import (
    fetch_recent_discovery_candidates,
    get_user_preference,
    save_discovery_candidates,
    save_discovery_run,
)
from services.portfolio_service import get_portfolio_snapshot, should_sync_portfolio_before_analysis
from utils.discovery_support import get_default_policy_profile, normalize_policy_profile

logger = logging.getLogger("MyInvestmentBanker.orchestrator")
logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    portfolio: List[Dict[str, Any]]
    raw_ingested_data: Dict[str, Any]
    cfa_analyst_memos: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    final_synthesis_report: str
    messages: List[Dict[str, str]]


def _confidence_rank(confidence_level: str) -> int:
    if confidence_level == "high":
        return 3
    if confidence_level == "medium":
        return 2
    if confidence_level == "low":
        return 1
    return 0


def IngestionNode(state: AgentState) -> Dict[str, Any]:
    logger.info("--- LANGGRAPH: Entering Ingestion Node ---")
    portfolio = get_portfolio_snapshot(sync_from_broker=should_sync_portfolio_before_analysis())
    scout_payload = ResearchPlannerAgent.run_portfolio_ingestion(portfolio)
    return {
        "portfolio": portfolio,
        "raw_ingested_data": scout_payload,
        "cfa_analyst_memos": [],
    }


def CFAAnalysisNode(state: AgentState) -> Dict[str, Any]:
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
        analyst_memo = CFAAgent.run(symbol, filing_text, market_stats, supplemental_context=ticker_stats)
        memos_list.append(analyst_memo)

    return {"cfa_analyst_memos": memos_list}


def PortfolioRiskNode(state: AgentState) -> Dict[str, Any]:
    logger.info("--- LANGGRAPH: Entering Portfolio Threat & Risk Audit Node ---")
    portfolio = state.get("portfolio", [])
    macro_stats = state.get("raw_ingested_data", {}).get("macro", {})
    cfa_memos = state.get("cfa_analyst_memos", [])
    risk_payload = RiskAgent.run(portfolio, macro_stats, cfa_memos)
    return {"risk_assessment": risk_payload}


def FinalSynthesisNode(state: AgentState) -> Dict[str, Any]:
    logger.info("--- LANGGRAPH: Entering Final Synthesis Node ---")
    portfolio = state.get("portfolio", [])
    macro_stats = state.get("raw_ingested_data", {}).get("macro", {})
    cfa_memos = state.get("cfa_analyst_memos", [])
    risk_memo = state.get("risk_assessment", {})
    telegram_bulletin = CommunicationAgent.compile_synthesis_report(
        portfolio_state=portfolio,
        macro_data=macro_stats,
        cfa_memos=cfa_memos,
        risk_memos=risk_memo,
    )
    return {"final_synthesis_report": telegram_bulletin}


def build_multi_agent_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("Scout_Ingestion", IngestionNode)
    workflow.add_node("CFA_Analysis", CFAAnalysisNode)
    workflow.add_node("Risk_Audit", PortfolioRiskNode)
    workflow.add_node("Synthesis_Lead", FinalSynthesisNode)
    workflow.set_entry_point("Scout_Ingestion")
    workflow.add_edge("Scout_Ingestion", "CFA_Analysis")
    workflow.add_edge("CFA_Analysis", "Risk_Audit")
    workflow.add_edge("Risk_Audit", "Synthesis_Lead")
    workflow.add_edge("Synthesis_Lead", END)
    return workflow.compile()


wealth_manager_graph = build_multi_agent_workflow()


def _load_policy_profile() -> Dict[str, Any]:
    screening_policy = get_user_preference("screening_policy")
    legacy_focus = get_user_preference("active_opportunity_focus")
    if screening_policy:
        return normalize_policy_profile(screening_policy, legacy_focus=legacy_focus)

    broad_policy = get_user_preference("broad_investment_policy")
    if isinstance(broad_policy, dict):
        broad_policy = broad_policy.get("policy_text")

    if broad_policy:
        return normalize_policy_profile({"policy_text": broad_policy}, legacy_focus=legacy_focus)
    return get_default_policy_profile()


def _has_fresh_catalyst(recommendation: Dict[str, Any]) -> bool:
    evidence = recommendation.get("evidence", {})
    return bool(evidence.get("material_news")) or bool(evidence.get("relevant_filings"))


def _is_repeat_alert(recommendation: Dict[str, Any], run_type: str) -> bool:
    if CommunicationAgent.is_suppressed_discovery_idea(
        recommendation.get("symbol", ""),
        recommendation.get("theme_key", ""),
    ):
        return True
    recent = fetch_recent_discovery_candidates(
        symbol=recommendation.get("symbol"),
        theme_key=recommendation.get("theme_key"),
        days=7 if run_type == "sweep" else 14,
        limit=10,
    )
    if not recent:
        return False
    if run_type == "sweep" and not _has_fresh_catalyst(recommendation):
        return True
    return False


def trigger_wealth_manager_run() -> str:
    logger.info("Starting automated wealth manager run...")
    initial_state = {
        "portfolio": [],
        "raw_ingested_data": {},
        "cfa_analyst_memos": [],
        "risk_assessment": {},
        "final_synthesis_report": "",
        "messages": [],
    }
    try:
        final_state = wealth_manager_graph.invoke(initial_state)
        return final_state.get("final_synthesis_report", "Error: Pipeline failed to generate synthesis report.")
    except Exception as e:
        logger.error(f"Workflow pipeline invocation failed: {e}")
        return f"❌ **System Error during multi-agent analysis cycle:**\n\n`{e}`"


def trigger_opportunity_discovery(
    run_type: str = "deep",
    focus_override: List[str] = None,
    return_result: bool = False,
) -> Optional[Any]:
    logger.info(f"Starting theme-led opportunity discovery run ({run_type})...")
    try:
        portfolio = get_portfolio_snapshot(sync_from_broker=should_sync_portfolio_before_analysis())
        policy_profile = _load_policy_profile()
        scout_payload = ScoutAgent.run_theme_discovery(
            policy_profile=policy_profile,
            portfolio=portfolio,
            run_type=run_type,
            focus_override=focus_override,
        )

        themes = ResearchPlannerAgent.rank_themes(scout_payload.get("themes", []))
        llm_theme_keys = set(ResearchPlannerAgent.select_llm_theme_keys(themes, run_type=run_type))
        remaining_llm_candidate_slots = ResearchPlannerAgent.llm_candidate_budget(run_type)
        if not themes:
            summary_text = "No sector or theme became compelling enough to investigate further."
            discovery_run = {
                "run_type": run_type,
                "recommendations": [],
                "themes": [],
                "policy_profile": policy_profile,
                "summary_text": summary_text,
            }
            save_discovery_run(
                run_type=run_type,
                status="no_opportunity",
                policy_snapshot=policy_profile,
                themes=[],
                summary_text=summary_text,
            )
            if run_type == "sweep":
                return None
            discovery_run["bulletin"] = CommunicationAgent.compile_discovery_briefing(discovery_run)
            return discovery_run if return_result else discovery_run["bulletin"]

        recommendations: List[Dict[str, Any]] = []
        for theme in themes:
            use_llm_for_theme = theme.get("theme_key") in llm_theme_keys and remaining_llm_candidate_slots > 0
            candidate_expressions = ResearchPlannerAgent.expand_theme_candidates(
                theme,
                portfolio,
                run_type=run_type,
                candidate_limit=ResearchPlannerAgent.candidate_limit_for_theme(run_type, use_llm_for_theme),
            )
            per_theme_llm_budget = min(2 if use_llm_for_theme else 0, remaining_llm_candidate_slots)
            llm_review_symbols = set(
                ResearchPlannerAgent.select_llm_review_symbols(
                    candidate_expressions,
                    run_type,
                    budget_override=per_theme_llm_budget,
                )
            )
            remaining_llm_candidate_slots -= len(llm_review_symbols)
            candidate_reviews = [
                CFAAgent.review_discovery_candidate(
                    theme,
                    candidate_expression,
                    policy_profile,
                    use_llm=candidate_expression.get("symbol") in llm_review_symbols,
                )
                for candidate_expression in candidate_expressions
            ]
            audited_theme = RiskAgent.audit_discovery_theme(
                theme=theme,
                candidate_reviews=candidate_reviews,
                policy_profile=policy_profile,
                max_recommendations=2 if run_type == "sweep" else 3,
                use_llm=use_llm_for_theme,
            )
            review_by_symbol = {review.get("symbol"): review for review in candidate_reviews if review.get("symbol")}
            for recommendation in audited_theme.get("recommendations", []):
                matched_review = review_by_symbol.get(recommendation.get("symbol"), {})
                recommendation["triage_score"] = matched_review.get("triage_score", 0)
                recommendation["theme_confidence_level"] = theme.get("confidence_level", "low")
                if not _is_repeat_alert(recommendation, run_type=run_type):
                    recommendations.append(recommendation)

        recommendations.sort(
            key=lambda item: (
                _confidence_rank(str(item.get("theme_confidence_level", "low"))),
                float(item.get("triage_score", 0) or 0),
                len((item.get("evidence") or {}).get("material_news", [])),
                len((item.get("evidence") or {}).get("relevant_filings", [])),
            ),
            reverse=True,
        )
        recommendations = recommendations[:3]
        if not recommendations and run_type == "sweep":
            return None

        summary_text = (
            f"Scout produced `{len(themes)}` theme hypothesis/hypotheses and "
            f"`{len(recommendations)}` final recommendation(s)."
        )
        discovery_run = {
            "run_type": run_type,
            "recommendations": recommendations,
            "themes": themes,
            "policy_profile": policy_profile,
            "summary_text": summary_text,
        }

        run_id = save_discovery_run(
            run_type=run_type,
            status="completed" if recommendations else "no_opportunity",
            policy_snapshot=policy_profile,
            themes=themes,
            summary_text=summary_text,
        )
        save_discovery_candidates(run_id, recommendations)

        if not recommendations:
            if run_type == "sweep":
                return None
            discovery_run["bulletin"] = CommunicationAgent.compile_discovery_briefing(discovery_run)
            return discovery_run if return_result else discovery_run["bulletin"]

        discovery_run["bulletin"] = CommunicationAgent.compile_discovery_briefing(discovery_run)
        return discovery_run if return_result else discovery_run["bulletin"]
    except Exception as e:
        logger.error(f"Error during opportunity discovery run: {e}")
        error_text = f"❌ **System Error during opportunity discovery run:**\n\n`{e}`"
        if return_result:
            return {
                "run_type": run_type,
                "recommendations": [],
                "themes": [],
                "policy_profile": _load_policy_profile(),
                "summary_text": error_text,
                "bulletin": error_text,
                "error": True,
            }
        return error_text


def trigger_autonomous_discovery_check(return_result: bool = False) -> Optional[Any]:
    """
    Weekday light sweep alias retained for compatibility.
    """
    logger.info("Starting autonomous discovery sweep...")
    return trigger_opportunity_discovery(run_type="sweep", return_result=return_result)
