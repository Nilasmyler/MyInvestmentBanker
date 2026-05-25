import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agents.communication_agent import generate_llm_response, llm_available
from database.supabase_client import fetch_historical_memos, save_analyst_memo

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.cfa")
logging.basicConfig(level=logging.INFO)


class CFAAgent:
    """
    Performs company-level analytical review for portfolio monitoring and theme-led discovery.
    """

    @staticmethod
    def _format_supplemental_context(supplemental_context: Optional[Dict[str, Any]] = None) -> str:
        if not supplemental_context:
            return "No supplemental ownership or street-consensus context supplied."

        sections = []
        material_news = supplemental_context.get("material_news", [])
        if material_news:
            sections.append(
                "Recent material news: "
                + "; ".join([item.get("headline", "") for item in material_news[:2] if item.get("headline")])
            )
        ownership_intel = supplemental_context.get("ownership_intel", {})
        if ownership_intel.get("summary"):
            sections.append(f"Ownership intelligence: {ownership_intel.get('summary')}")
        street_consensus = supplemental_context.get("street_consensus", {})
        if street_consensus.get("summary"):
            sections.append(f"Street consensus: {street_consensus.get('summary')}")
        research_plan = supplemental_context.get("research_plan", {})
        if research_plan.get("reasons"):
            sections.append(f"Planner routing reasons: {', '.join(research_plan.get('reasons', []))}")
        return "\n".join(sections) if sections else "Supplemental evidence was thin."

    @staticmethod
    def run(
        symbol: str,
        current_filing_data: str,
        market_data: Dict[str, Any],
        supplemental_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = symbol.upper().strip()
        logger.info(f"CFA Agent: Performing balance sheet audit for {symbol}...")

        past_memos = fetch_historical_memos(symbol, limit=2)
        historical_context = ""
        if past_memos:
            for idx, memo in enumerate(past_memos):
                historical_context += f"--- Historical Analyst Memo {idx + 1} ({memo['period']}) ---\n{memo['memo_text']}\n\n"
        else:
            historical_context = "No previous analyst memos exist for this asset in the database. This is a baseline analysis."

        if not llm_available():
            verdict_parts = []
            if market_data.get("current_price") is not None:
                verdict_parts.append(f"Price context available at `{market_data.get('current_price')}`.")
            if market_data.get("day_change_pct") is not None:
                verdict_parts.append(f"Daily move is `{market_data.get('day_change_pct')}%`.")
            if current_filing_data and current_filing_data != "No filing context available.":
                verdict_parts.append("Recent filing metadata was captured for review.")
            if supplemental_context:
                ownership_summary = (supplemental_context.get("ownership_intel") or {}).get("summary")
                if ownership_summary:
                    verdict_parts.append(str(ownership_summary))
                street_summary = (supplemental_context.get("street_consensus") or {}).get("summary")
                if street_summary:
                    verdict_parts.append(str(street_summary))
            if not verdict_parts:
                verdict_parts.append("Only thin market context was available, so this is a low-confidence baseline review.")

            memo_text = " ".join(verdict_parts)
            metrics_dict = {
                "pe_ratio": market_data.get("pe_ratio"),
                "beta": market_data.get("beta"),
            }
            headline_verdict = f"Baseline review completed for {symbol} with limited non-LLM evidence."
        else:
            system_instruction = (
                "You are the Chartered Financial Analyst (CFA) Agent of MyInvestmentBanker.\n"
                "Your tone is quantitative, objective, and concise.\n"
                "Compute only what can be supported by the supplied context and note any missing data."
            )
            prompt = (
                f"Please conduct a comprehensive fundamental audit for **{symbol}**.\n\n"
                f"=== 1. Current SEC Filing & Accession Data ===\n{current_filing_data}\n\n"
                f"=== 2. Market Pricing & Valuation Data ===\n{market_data}\n\n"
                f"=== 3. Supplemental Ownership / Street Context ===\n{CFAAgent._format_supplemental_context(supplemental_context)}\n\n"
                f"=== 4. Historical Analyst Memory (Past Quarters) ===\n{historical_context}\n\n"
                f"Structure the response as: Period Under Review, Mathematical Core Calculations, Longitudinal Trends, Ownership & Sentiment Check, CFA Synthesis & Verdict."
            )
            memo_text = generate_llm_response(prompt, system_instruction)
            headline_verdict = memo_text.split("\n")[0] if memo_text else f"Review completed for {symbol}."

            metrics_prompt = (
                f"Based on the analyst memo below, extract JSON with these keys when available: "
                f"debt_to_equity, fcf_margin_pct, pe_ratio, interest_coverage_ratio.\n"
                f"Return ONLY JSON.\n\nMemo:\n{memo_text}"
            )
            raw_json = generate_llm_response(metrics_prompt, "Extract exact JSON key-values.")
            try:
                metrics_dict = json.loads(raw_json.strip().replace("```json", "").replace("```", "").strip())
            except Exception:
                metrics_dict = {}

        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        period_str = f"Q{quarter}_{now.year}"
        save_analyst_memo(symbol, period_str, memo_text, metrics_dict)

        return {
            "symbol": symbol,
            "period": period_str,
            "memo_text": memo_text,
            "metrics": metrics_dict,
            "headline_verdict": headline_verdict,
        }

    @staticmethod
    def review_discovery_candidate(
        theme: Dict[str, Any],
        candidate_expression: Dict[str, Any],
        policy_profile: Dict[str, Any],
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        symbol = candidate_expression.get("symbol", "UNKNOWN")
        logger.info(f"CFA Agent: Reviewing discovery candidate {symbol} for theme {theme.get('theme_key')}.")

        market_data = candidate_expression.get("price_context", {})
        catalysts = candidate_expression.get("company_catalysts", [])
        data_gaps = candidate_expression.get("data_gaps", [])
        filings = candidate_expression.get("relevant_filings", [])
        material_news = candidate_expression.get("material_news", [])
        ownership_intel = candidate_expression.get("ownership_intel", {})
        street_consensus = candidate_expression.get("street_consensus", {})

        if not use_llm or not llm_available():
            strengths = []
            cautions = list(data_gaps)
            company_preferences = policy_profile.get("company_preferences", [])

            if catalysts:
                strengths.append("Multiple current catalysts make the company relevant to the theme.")
            if material_news:
                strengths.append("There is recent material news tied to the company.")
            if filings:
                strengths.append("A recent SEC filing gives a fresh corporate event to inspect.")
            if ownership_intel.get("signal_strength") == "high":
                strengths.append("Ownership data shows a higher-signal insider or beneficial-ownership development.")
            elif ownership_intel.get("summary"):
                strengths.append(str(ownership_intel.get("summary")))
            if street_consensus.get("signal_strength") == "high":
                strengths.append("Street consensus is active enough to add a useful external cross-check.")
            elif street_consensus.get("summary"):
                strengths.append(str(street_consensus.get("summary")))
            if theme.get("theme_key") in policy_profile.get("preferred_themes", []):
                strengths.append("The theme already matches the user's stated areas of interest.")
            if market_data.get("five_day_change_pct") not in [None, 0]:
                strengths.append(f"Five-day move: `{market_data.get('five_day_change_pct')}%`.")
            if not strengths:
                strengths.append("The company is a sector leader but the near-term evidence set is thin.")
            if market_data.get("beta") and market_data["beta"] > 2:
                cautions.append("High beta may increase position-sizing risk.")
            if policy_profile.get("risk_profile") == "conservative" and market_data.get("beta") and market_data["beta"] > 1.6:
                cautions.append("This is less aligned with a conservative profile because the stock is likely to swing more than the market.")
            if "crowded valuations" in policy_profile.get("risk_avoidances", []) and market_data.get("pe_ratio") and market_data["pe_ratio"] > 45:
                cautions.append("Valuation already looks crowded relative to the user's stated risk avoidances.")
            if "large-cap stability" in company_preferences and market_data.get("market_cap") and market_data["market_cap"] >= 50_000_000_000:
                strengths.append("The company fits the user's bias toward established, larger-cap businesses.")
            if "small-cap upside" in company_preferences and market_data.get("market_cap") and market_data["market_cap"] <= 10_000_000_000:
                strengths.append("The company fits the user's willingness to look at smaller-cap upside.")

            evidence_strength = "high" if len(strengths) >= 3 else "medium" if len(strengths) == 2 else "low"
            analyst_verdict = (
                f"{symbol} looks like a {evidence_strength}-conviction expression of the "
                f"{theme.get('theme_name', theme.get('theme_key', 'current'))} theme."
            )
            return {
                **candidate_expression,
                "thesis_alignment": candidate_expression.get("why_this_company"),
                "strengths": strengths,
                "cautions": cautions,
                "analyst_verdict": analyst_verdict,
                "confidence_note": (
                    "Built from structured market/news/ownership/consensus evidence without Gemini reasoning."
                    if not use_llm or not llm_available()
                    else "LLM-assisted review."
                ),
                "evidence_strength": evidence_strength,
            }

        prompt_payload = {
            "theme": theme,
            "candidate_expression": candidate_expression,
            "policy_profile": policy_profile,
        }
        system_instruction = (
            "You are the CFA Agent of MyInvestmentBanker.\n"
            "Evaluate whether a company is a good expression of the supplied sector/theme thesis.\n"
            "Return only JSON with keys: thesis_alignment, strengths, cautions, analyst_verdict, confidence_note, evidence_strength."
        )
        response = generate_llm_response(str(prompt_payload), system_instruction)
        try:
            parsed = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
        except Exception:
            parsed = {
                "thesis_alignment": candidate_expression.get("why_this_company"),
                "strengths": catalysts[:3] or ["Evidence set could not be fully parsed."],
                "cautions": data_gaps,
                "analyst_verdict": f"{symbol} remains relevant to the theme, but the structured response was degraded.",
                "confidence_note": "LLM output could not be parsed cleanly.",
                "evidence_strength": "medium",
            }

        return {**candidate_expression, **parsed}
