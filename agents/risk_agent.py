import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agents.communication_agent import generate_llm_response, llm_available
from database.supabase_client import fetch_investment_thesis

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.risk")
logging.basicConfig(level=logging.INFO)


class RiskAgent:
    """
    Acts as the Macro & Portfolio Risk Officer for both portfolio review and discovery.
    """

    @staticmethod
    def run(portfolio: List[Dict[str, Any]], macro_data: Dict[str, Any], cfa_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Risk Agent: Conducting portfolio threat and thesis correlation audit...")

        theses_context = ""
        for holding in portfolio:
            symbol = holding["symbol"].upper()
            thesis = fetch_investment_thesis(symbol)
            if thesis:
                theses_context += f"--- User Thesis on {symbol} ---\n{thesis['thesis_text']}\n\n"
            else:
                theses_context += f"--- User Thesis on {symbol} ---\nNo thesis logged for this asset.\n\n"

        if not llm_available():
            return {
                "risk_memo": (
                    f"Baseline risk review completed across `{len(portfolio)}` holdings. "
                    f"Macro reference: Fed Funds `{macro_data.get('fed_funds_rate', 'N/A')}`, "
                    f"Inflation `{macro_data.get('cpi_inflation', macro_data.get('cpi_inflation_index', 'N/A'))}`. "
                    "Use this run as a structured placeholder until Gemini-backed synthesis is available."
                ),
                "summary": "Fallback risk memo generated without Gemini.",
            }

        system_instruction = (
            "You are the Macro & Portfolio Risk Officer of MyInvestmentBanker.\n"
            "Your tone is skeptical, cautious, and analytical.\n"
            "Evaluate macro sensitivity, thesis drift, and structural risk."
        )
        prompt = (
            f"Please conduct a portfolio risk and macro alignment review.\n\n"
            f"=== 1. Active Portfolio Holdings ===\n{portfolio}\n\n"
            f"=== 2. Macroeconomic Environment ===\n{macro_data}\n\n"
            f"=== 3. CFA Analyst Quarterly Reports ===\n{cfa_reports}\n\n"
            f"=== 4. User Investment Theses ===\n{theses_context}\n\n"
            f"Output a formal risk memo with Macro Threat Alignment, Thesis Validation & Drift, and Margin of Safety & Allocation Audits."
        )
        risk_memo = generate_llm_response(prompt, system_instruction)
        return {"risk_memo": risk_memo, "summary": risk_memo.split("\n")[0] if risk_memo else ""}

    @staticmethod
    def _sanitize_llm_audit_result(
        theme: Dict[str, Any],
        parsed: Any,
        candidate_reviews: List[Dict[str, Any]],
        max_recommendations: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return None

        recommendations = parsed.get("recommendations")
        rejected = parsed.get("rejected")
        if not isinstance(recommendations, list) or not isinstance(rejected, list):
            return None

        review_by_symbol = {
            str(review.get("symbol", "")).upper().strip(): review
            for review in candidate_reviews
            if review.get("symbol")
        }
        sanitized_recommendations = []
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            review = review_by_symbol.get(symbol)
            if not review:
                continue
            deterministic_shell = {
                "symbol": review.get("symbol"),
                "recommendation_type": "increase_existing_position" if review.get("is_existing_position") else "new_position",
                "theme_name": theme.get("theme_name"),
                "investment_hypothesis": review.get("thesis_alignment") or review.get("why_this_company"),
                "why_now": review.get("analyst_verdict"),
                "key_risks": review.get("cautions", [])[:3] or ["Evidence is still developing."],
                "what_invalidates_it": theme.get("invalidators", [])[:2],
                "confidence_note": review.get("confidence_note", "LLM-assisted review."),
                "source_etf": review.get("source_etf"),
                "theme_key": theme.get("theme_key"),
                "status": "recommended",
                "evidence": {
                    "catalysts": review.get("company_catalysts", []),
                    "material_news": review.get("material_news", []),
                    "relevant_filings": review.get("relevant_filings", []),
                    "ownership_intel": review.get("ownership_intel", {}),
                    "street_consensus": review.get("street_consensus", {}),
                },
                "rationale": review.get("analyst_verdict", ""),
            }
            sanitized_recommendations.append(
                {
                    **deterministic_shell,
                    **item,
                    "symbol": review.get("symbol"),
                    "source_etf": review.get("source_etf"),
                    "theme_key": theme.get("theme_key"),
                    "theme_name": theme.get("theme_name"),
                    "evidence": deterministic_shell["evidence"],
                }
            )
            if len(sanitized_recommendations) >= max_recommendations:
                break

        sanitized_rejected = [item for item in rejected if isinstance(item, dict)]
        risk_summary = str(parsed.get("risk_summary", "") or "").strip()
        if not risk_summary:
            risk_summary = (
                f"Selected `{len(sanitized_recommendations)}` recommendation(s) from the reviewed candidate set."
            )

        return {
            "recommendations": sanitized_recommendations,
            "rejected": sanitized_rejected,
            "risk_summary": risk_summary,
        }

    @staticmethod
    def _build_fallback_recommendations(
        theme: Dict[str, Any],
        candidate_reviews: List[Dict[str, Any]],
        policy_profile: Optional[Dict[str, Any]] = None,
        max_recommendations: int = 3,
    ) -> Dict[str, Any]:
        policy_profile = policy_profile or {}
        ranked = sorted(
            candidate_reviews,
            key=lambda item: (
                item.get("theme_key") in policy_profile.get("preferred_themes", []),
                item.get("evidence_strength") == "high",
                item.get("is_existing_position", False),
                item.get("ownership_intel", {}).get("signal_strength") == "high",
                item.get("street_consensus", {}).get("signal_strength") == "high",
                len(item.get("company_catalysts", [])),
                len(item.get("material_news", [])),
                len(item.get("relevant_filings", [])),
            ),
            reverse=True,
        )

        recommendations = []
        rejected = []
        for review in ranked:
            if theme.get("theme_key") in policy_profile.get("excluded_themes", []):
                rejected.append(
                    {
                        "symbol": review.get("symbol"),
                        "reason": "The theme is currently excluded by the user's preference profile.",
                    }
                )
                continue

            signal_quality = (
                len(review.get("company_catalysts", []))
                + len(review.get("material_news", []))
                + len(review.get("relevant_filings", []))
                + (1 if review.get("ownership_intel", {}).get("signal_strength") in ["medium", "high"] else 0)
                + (1 if review.get("street_consensus", {}).get("signal_strength") in ["medium", "high"] else 0)
            )
            if signal_quality <= 0:
                rejected.append(
                    {
                        "symbol": review.get("symbol"),
                        "reason": "Evidence set was too thin for a conviction recommendation.",
                    }
                )
                continue

            market_data = review.get("price_context", {})
            if (
                policy_profile.get("risk_profile") == "conservative"
                and market_data.get("beta")
                and market_data["beta"] > 1.8
                and not review.get("is_existing_position")
            ):
                rejected.append(
                    {
                        "symbol": review.get("symbol"),
                        "reason": "The stock looks too volatile for the user's current conservative risk posture.",
                    }
                )
                continue

            recommendation_type = "increase_existing_position" if review.get("is_existing_position") else "new_position"
            recommendations.append(
                {
                    "symbol": review.get("symbol"),
                    "recommendation_type": recommendation_type,
                    "theme_name": theme.get("theme_name"),
                    "investment_hypothesis": review.get("thesis_alignment") or review.get("why_this_company"),
                    "why_now": review.get("analyst_verdict"),
                    "key_risks": review.get("cautions", [])[:3] or ["Evidence is still developing."],
                    "what_invalidates_it": theme.get("invalidators", [])[:2],
                    "confidence_note": review.get("confidence_note", "Fallback review without Gemini."),
                    "source_etf": review.get("source_etf"),
                    "theme_key": theme.get("theme_key"),
                    "status": "recommended",
                    "evidence": {
                        "catalysts": review.get("company_catalysts", []),
                        "material_news": review.get("material_news", []),
                        "relevant_filings": review.get("relevant_filings", []),
                        "ownership_intel": review.get("ownership_intel", {}),
                        "street_consensus": review.get("street_consensus", {}),
                    },
                    "rationale": review.get("analyst_verdict", ""),
                }
            )
            if len(recommendations) >= max_recommendations:
                break

        return {
            "theme_key": theme.get("theme_key"),
            "theme_name": theme.get("theme_name"),
            "recommendations": recommendations,
            "rejected": rejected,
            "risk_summary": (
                f"Selected `{len(recommendations)}` recommendation(s) from the "
                f"{theme.get('theme_name', theme.get('theme_key', 'current'))} theme."
            ),
        }

    @staticmethod
    def audit_discovery_theme(
        theme: Dict[str, Any],
        candidate_reviews: List[Dict[str, Any]],
        policy_profile: Dict[str, Any],
        max_recommendations: int = 3,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        logger.info(
            f"Risk Agent: Auditing {len(candidate_reviews)} candidates for theme {theme.get('theme_key', 'unknown')}."
        )
        if not candidate_reviews:
            return {
                "theme_key": theme.get("theme_key"),
                "theme_name": theme.get("theme_name"),
                "recommendations": [],
                "rejected": [],
                "risk_summary": "No candidates were reviewed for this theme.",
            }

        if not use_llm or not llm_available():
            return RiskAgent._build_fallback_recommendations(
                theme,
                candidate_reviews,
                policy_profile=policy_profile,
                max_recommendations=max_recommendations,
            )

        prompt_payload = {
            "theme": theme,
            "candidate_reviews": candidate_reviews,
            "policy_profile": policy_profile,
            "max_recommendations": max_recommendations,
        }
        system_instruction = (
            "You are the Risk Officer of MyInvestmentBanker.\n"
            "Choose the best company expressions of the supplied theme.\n"
            "Return only JSON with keys: recommendations, rejected, risk_summary."
        )
        response = generate_llm_response(str(prompt_payload), system_instruction)
        try:
            raw_parsed = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
            parsed = RiskAgent._sanitize_llm_audit_result(theme, raw_parsed, candidate_reviews, max_recommendations)
            if parsed is None:
                parsed = RiskAgent._build_fallback_recommendations(
                    theme,
                    candidate_reviews,
                    policy_profile=policy_profile,
                    max_recommendations=max_recommendations,
                )
        except Exception:
            parsed = RiskAgent._build_fallback_recommendations(
                theme,
                candidate_reviews,
                policy_profile=policy_profile,
                max_recommendations=max_recommendations,
            )

        parsed["theme_key"] = theme.get("theme_key")
        parsed["theme_name"] = theme.get("theme_name")
        return parsed
