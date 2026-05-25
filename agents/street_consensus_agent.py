import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from utils.discovery_support import StreetConsensus
from utils.financial_tools import fetch_analyst_consensus

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.street_consensus")
logging.basicConfig(level=logging.INFO)


def _rating_bucket(recommendation_key: str, recommendation_mean: Optional[float]) -> str:
    key = (recommendation_key or "").strip().lower()
    if key in {"strong_buy", "strong-buy", "buy", "outperform", "overweight"}:
        return "positive"
    if key in {"sell", "underperform", "underweight", "strong_sell", "strong-sell"}:
        return "negative"
    if recommendation_mean is not None:
        if recommendation_mean <= 2.0:
            return "positive"
        if recommendation_mean >= 3.2:
            return "negative"
    if key in {"hold", "neutral"}:
        return "neutral"
    return "mixed"


class StreetConsensusAgent:
    """
    Collects analyst-consensus and price-target signals for a symbol.
    """

    @staticmethod
    def collect_symbol_consensus(symbol: str, market_data: Optional[Dict[str, Any]] = None) -> StreetConsensus:
        symbol = symbol.upper().strip()
        consensus = fetch_analyst_consensus(symbol, market_data=market_data)

        analyst_count = int(consensus.get("analyst_count", 0) or 0)
        recommendation_key = str(consensus.get("recommendation_key", "") or "")
        recommendation_mean = consensus.get("recommendation_mean")
        target_premium_pct = consensus.get("price_target_premium_pct")
        rating_bucket = _rating_bucket(recommendation_key, recommendation_mean)
        consensus_error = consensus.get("error")

        alerts: List[str] = []
        if analyst_count:
            alerts.append(f"Analyst snapshot includes roughly `{analyst_count}` opinion(s).")
        if recommendation_key:
            alerts.append(f"Current recommendation label is `{recommendation_key}`.")
        if target_premium_pct is not None:
            alerts.append(f"Mean street target implies `{target_premium_pct}%` upside/downside versus the latest price.")
        if analyst_count and analyst_count <= 3:
            alerts.append("Analyst coverage is thin, so consensus should be treated cautiously.")

        signal_strength = "low"

        summary = "Street-consensus coverage was thin in this pass."
        if signal_strength in ["medium", "high"] and alerts:
            if rating_bucket == "negative":
                summary = f"Street snapshot leans cautious. {' '.join(alerts[:3])}"
            elif rating_bucket == "positive":
                summary = f"Street snapshot leans constructive. {' '.join(alerts[:3])}"
            else:
                summary = " ".join(alerts[:3])
        elif analyst_count or recommendation_key or target_premium_pct is not None:
            summary = "Analyst snapshot is available, but it does not include a fresh revision signal."

        result: StreetConsensus = {
            "symbol": symbol,
            "summary": summary,
            "signal_strength": signal_strength,
            "analyst_count": analyst_count,
            "recommendation_key": recommendation_key,
            "recommendation_mean": recommendation_mean,
            "target_mean_price": consensus.get("target_mean_price"),
            "target_high_price": consensus.get("target_high_price"),
            "target_low_price": consensus.get("target_low_price"),
            "price_target_premium_pct": target_premium_pct,
            "recommendation_breakdown": consensus.get("recommendation_breakdown", {}),
            "alerts": alerts,
        }
        if consensus_error:
            result["error"] = str(consensus_error)
        return result
