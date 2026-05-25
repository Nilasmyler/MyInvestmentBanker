import logging
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv

from agents.ownership_intel_agent import OwnershipIntelAgent
from agents.scout_agent import ScoutAgent
from agents.street_consensus_agent import StreetConsensusAgent
from utils.discovery_support import CandidateExpression, PolicyProfile, ThemeHypothesis
from utils.prompt_compressor import compress_financial_text

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.research_planner")
logging.basicConfig(level=logging.INFO)


class ResearchPlan(TypedDict, total=False):
    symbol: str
    run_type: str
    include_ownership_intel: bool
    include_street_consensus: bool
    reasons: List[str]
    priority: str


class ResearchPlannerAgent:
    """
    Deterministic research planner that routes specialist data collection and caps LLM-heavy follow-on work.
    """

    @staticmethod
    def build_symbol_plan(
        symbol: str,
        base_research: Dict[str, Any],
        run_type: str,
        is_existing_position: bool = False,
    ) -> ResearchPlan:
        market_data = base_research.get("market_data", {})
        price_move = abs(market_data.get("five_day_change_pct") or 0)
        news_count = len(base_research.get("material_news", []))
        filing_count = len(base_research.get("recent_filings", []))

        include_ownership_intel = is_existing_position or filing_count > 0 or news_count > 0 or price_move >= 6
        include_street_consensus = is_existing_position or run_type == "deep" or market_data.get("current_price") is not None

        reasons: List[str] = []
        if is_existing_position:
            reasons.append("existing_position")
        if filing_count:
            reasons.append("fresh_filings")
        if news_count:
            reasons.append("material_news")
        if price_move >= 6:
            reasons.append("sharp_price_move")
        if run_type == "deep":
            reasons.append("deep_run")

        priority = "low"
        if is_existing_position or len(reasons) >= 3:
            priority = "high"
        elif reasons:
            priority = "medium"

        return {
            "symbol": symbol.upper().strip(),
            "run_type": run_type,
            "include_ownership_intel": include_ownership_intel,
            "include_street_consensus": include_street_consensus,
            "reasons": reasons,
            "priority": priority,
        }

    @staticmethod
    def collect_symbol_research(
        symbol: str,
        theme_context: str = "",
        run_type: str = "deep",
        is_existing_position: bool = False,
    ) -> Dict[str, Any]:
        base_research = ScoutAgent.collect_symbol_research(symbol, theme_context=theme_context)
        plan = ResearchPlannerAgent.build_symbol_plan(
            symbol,
            base_research,
            run_type=run_type,
            is_existing_position=is_existing_position,
        )

        enriched_research = {**base_research, "research_plan": plan}
        if plan.get("include_ownership_intel"):
            enriched_research["ownership_intel"] = OwnershipIntelAgent.collect_symbol_intel(symbol)
        if plan.get("include_street_consensus"):
            enriched_research["street_consensus"] = StreetConsensusAgent.collect_symbol_consensus(
                symbol,
                market_data=base_research.get("market_data", {}),
            )
        return enriched_research

    @staticmethod
    def _compress_filings(recent_filings: List[Dict[str, Any]]) -> str:
        filings_report = ""
        for filing in recent_filings:
            filings_report += (
                f"Form: {filing.get('form')} | Date: {filing.get('date')} | URL: {filing.get('report_url')}\n"
                f"Description: {filing.get('description')}\n\n"
            )
        return compress_financial_text(
            context=filings_report,
            target_token=300,
            instruction="Filter out boilerplate, preserving filing type, date, and material descriptors.",
        )

    @staticmethod
    def run_portfolio_ingestion(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not portfolio:
            logger.info("Portfolio empty. Planned ingestion aborted.")
            return {"macro": ScoutAgent.collect_macro_regime_research(), "tickers": {}}

        macro_stats = ScoutAgent.collect_macro_regime_research()
        ticker_payloads: Dict[str, Dict[str, Any]] = {}
        for item in portfolio:
            symbol = item["symbol"].upper()
            research = ResearchPlannerAgent.collect_symbol_research(
                symbol,
                run_type="portfolio_update",
                is_existing_position=True,
            )
            recent_filings = research.get("recent_filings", [])
            ticker_payloads[symbol] = {
                "market_data": research.get("market_data", {}),
                "material_news": research.get("material_news", []),
                "recent_filings": ResearchPlannerAgent._compress_filings(recent_filings),
                "raw_filings_list": recent_filings,
                "ownership_intel": research.get("ownership_intel", {}),
                "street_consensus": research.get("street_consensus", {}),
                "research_plan": research.get("research_plan", {}),
            }

        return {"macro": macro_stats, "tickers": ticker_payloads}

    @staticmethod
    def score_candidate_expression(candidate_expression: CandidateExpression) -> float:
        score = float(candidate_expression.get("signal_count", 0) or 0) * 4.0
        if candidate_expression.get("is_existing_position"):
            score += 3.0
        score += min(len(candidate_expression.get("material_news", [])), 2) * 1.5
        score += min(len(candidate_expression.get("relevant_filings", [])), 2) * 1.5

        ownership_intel = candidate_expression.get("ownership_intel", {})
        street_consensus = candidate_expression.get("street_consensus", {})
        if ownership_intel.get("signal_strength") == "high":
            score += 2.5
        elif ownership_intel.get("signal_strength") == "medium":
            score += 1.0

        if street_consensus.get("signal_strength") == "high":
            score += 2.5
        elif street_consensus.get("signal_strength") == "medium":
            score += 1.0

        return round(score, 2)

    @staticmethod
    def expand_theme_candidates(
        theme: ThemeHypothesis,
        portfolio: List[Dict[str, Any]],
        run_type: str = "deep",
        candidate_limit: int = 6,
    ) -> List[CandidateExpression]:
        portfolio_symbols = {holding["symbol"].upper().strip() for holding in portfolio}
        candidate_expressions: List[CandidateExpression] = []
        for symbol in theme.get("candidate_symbols", [])[:candidate_limit]:
            research = ResearchPlannerAgent.collect_symbol_research(
                symbol,
                theme_context=theme.get("theme_name", ""),
                run_type=run_type,
                is_existing_position=symbol in portfolio_symbols,
            )
            candidate_expression = ScoutAgent.build_candidate_expression(theme, research, portfolio_symbols)
            signal_count = candidate_expression.get("signal_count", 0)
            if signal_count == 0 and symbol not in portfolio_symbols:
                continue
            candidate_expression["triage_score"] = ResearchPlannerAgent.score_candidate_expression(candidate_expression)
            candidate_expressions.append(candidate_expression)

        candidate_expressions.sort(
            key=lambda item: (
                item.get("triage_score", 0),
                item.get("signal_count", 0),
                len(item.get("material_news", [])),
                len(item.get("relevant_filings", [])),
            ),
            reverse=True,
        )
        return candidate_expressions[:candidate_limit]

    @staticmethod
    def llm_candidate_budget(run_type: str) -> int:
        return 3 if run_type == "deep" else 2

    @staticmethod
    def llm_theme_budget(run_type: str) -> int:
        return 2 if run_type == "deep" else 0

    @staticmethod
    def candidate_limit_for_theme(run_type: str, llm_enabled: bool) -> int:
        if run_type == "deep":
            return 4 if llm_enabled else 2
        return 2 if llm_enabled else 1

    @staticmethod
    def rank_themes(themes: List[ThemeHypothesis]) -> List[ThemeHypothesis]:
        return sorted(
            themes,
            key=lambda item: (
                item.get("confidence_level") == "high",
                item.get("confidence_level") == "medium",
                len(item.get("trigger_sources", [])),
                len(item.get("portfolio_overlap", [])),
            ),
            reverse=True,
        )

    @staticmethod
    def select_llm_review_symbols(
        candidate_expressions: List[CandidateExpression],
        run_type: str,
        budget_override: Optional[int] = None,
    ) -> List[str]:
        budget = budget_override if budget_override is not None else ResearchPlannerAgent.llm_candidate_budget(run_type)
        return [item.get("symbol", "") for item in candidate_expressions[:budget] if item.get("symbol")]

    @staticmethod
    def select_llm_theme_keys(themes: List[ThemeHypothesis], run_type: str) -> List[str]:
        if ResearchPlannerAgent.llm_theme_budget(run_type) <= 0:
            return []

        ranked_themes = ResearchPlannerAgent.rank_themes(themes)
        budget = ResearchPlannerAgent.llm_theme_budget(run_type)
        return [item.get("theme_key", "") for item in ranked_themes[:budget] if item.get("theme_key")]
