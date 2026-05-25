import logging
from typing import Any, Dict, List

from database.supabase_client import cache_news_digest
from dotenv import load_dotenv
from utils.discovery_support import (
    CandidateExpression,
    MaterialNewsItem,
    OwnershipIntel,
    PolicyProfile,
    StreetConsensus,
    ThemeHypothesis,
    get_theme_records_for_etfs,
)
from utils.financial_tools import (
    fetch_etf_holdings,
    fetch_macro_indicators,
    fetch_news,
    fetch_recent_sec_filings,
    get_stock_price_and_history,
)
from utils.prompt_compressor import compress_financial_text

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.scout")
logging.basicConfig(level=logging.INFO)

MATERIAL_KEYWORDS = {
    "earnings": ["guidance", "earnings", "revenue", "margin", "forecast"],
    "product": ["launch", "release", "roadmap", "chip", "platform", "model"],
    "regulation": ["regulation", "regulatory", "fda", "approval", "investigation", "antitrust"],
    "capital": ["funding", "buyback", "acquisition", "merger", "capex", "contract"],
    "security": ["breach", "security", "hack", "incident", "zero trust"],
}


class ScoutAgent:
    """
    Gathers market, filing, news, and macro evidence.
    Acts as a coordinator for multiple discovery research stages.
    """

    @staticmethod
    def run(portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Existing portfolio-ingestion workflow retained for the `/update` pipeline.
        """
        if not portfolio:
            logger.info("Portfolio empty. Ingestion aborted.")
            return {"macro": fetch_macro_indicators(), "tickers": {}}

        logger.info("Scout Agent: Starting live data acquisition flow...")
        macro_stats = fetch_macro_indicators()

        ticker_payloads = {}
        for item in portfolio:
            symbol = item["symbol"].upper()
            market_data = get_stock_price_and_history(symbol)
            recent_filings = fetch_recent_sec_filings(symbol, limit=2)

            filings_report = ""
            for filing in recent_filings:
                filings_report += (
                    f"Form: {filing['form']} | Date: {filing['date']} | URL: {filing['report_url']}\n"
                    f"Description: {filing['description']}\n\n"
                )

            compressed_filings = compress_financial_text(
                context=filings_report,
                target_token=300,
                instruction="Filter out boilerplate, preserving filing type, date, and material descriptors.",
            )

            ticker_payloads[symbol] = {
                "market_data": market_data,
                "recent_filings": compressed_filings,
                "raw_filings_list": recent_filings,
            }

        logger.info("Scout Agent: Ingestion and compaction tasks complete.")
        return {"macro": macro_stats, "tickers": ticker_payloads}

    @staticmethod
    def collect_macro_regime_research() -> Dict[str, Any]:
        return fetch_macro_indicators()

    @staticmethod
    def collect_market_action_research(etf_symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        market_context = {}
        for etf in etf_symbols:
            market_context[etf] = get_stock_price_and_history(etf)
        return market_context

    @staticmethod
    def _infer_materiality(item: Dict[str, Any], symbol: str, theme_context: str = "") -> MaterialNewsItem:
        text_blob = f"{item.get('headline', '')} {item.get('summary', '')} {theme_context}".lower()
        event_type = "development"
        why_material = "Relevant because it may affect the company's current thesis."

        for candidate_type, keywords in MATERIAL_KEYWORDS.items():
            if any(keyword in text_blob for keyword in keywords):
                event_type = candidate_type
                break

        if event_type == "earnings":
            why_material = "Touches sales, guidance, margins, or operating performance."
        elif event_type == "product":
            why_material = "Signals product cadence or demand drivers relevant to the theme."
        elif event_type == "regulation":
            why_material = "Introduces regulatory or approval implications for the company or sector."
        elif event_type == "capital":
            why_material = "Signals capital allocation, customer demand, or strategic positioning."
        elif event_type == "security":
            why_material = "Signals urgent enterprise or regulatory demand for security-related products."

        return {
            "symbol": symbol,
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "url": item.get("url", ""),
            "published_at": item.get("published_at", ""),
            "source": item.get("source", ""),
            "event_type": event_type,
            "why_material": why_material,
        }

    @staticmethod
    def filter_noise(news_headlines: List[Dict[str, Any]], symbol: str, theme_context: str = "") -> List[MaterialNewsItem]:
        if not news_headlines:
            return []

        filtered = []
        seen_urls = set()
        for item in news_headlines:
            headline = (item.get("headline") or "").strip()
            lower_blob = f"{headline} {(item.get('summary') or '')}".lower()
            if not headline or item.get("url") in seen_urls:
                continue
            if any(phrase in lower_blob for phrase in ["newsletter", "market wrap", "stocks to watch", "weekly recap"]):
                continue
            seen_urls.add(item.get("url"))
            filtered.append(ScoutAgent._infer_materiality(item, symbol, theme_context=theme_context))
        return filtered[:5]

    @staticmethod
    def collect_news_research(symbol: str, company_name: str = "", theme_context: str = "") -> List[MaterialNewsItem]:
        raw_news = fetch_news(symbol, days=7, limit=8, company_name=company_name)
        filtered_news = ScoutAgent.filter_noise(raw_news, symbol, theme_context=theme_context)
        for item in filtered_news:
            cache_news_digest(
                symbol=symbol,
                title=item.get("headline", ""),
                summary=item.get("summary", ""),
                url=item.get("url", ""),
                published_at=item.get("published_at", ""),
                entity_type="symbol",
                entity_key=symbol,
                metadata={"event_type": item.get("event_type")},
            )
        return filtered_news

    @staticmethod
    def collect_filing_research(symbol: str) -> List[Dict[str, Any]]:
        return fetch_recent_sec_filings(symbol, limit=2)

    @staticmethod
    def collect_symbol_research(symbol: str, theme_context: str = "") -> Dict[str, Any]:
        market_data = get_stock_price_and_history(symbol)
        material_news = ScoutAgent.collect_news_research(
            symbol,
            company_name=market_data.get("name", ""),
            theme_context=theme_context,
        )
        recent_filings = ScoutAgent.collect_filing_research(symbol)
        return {
            "symbol": symbol,
            "market_data": market_data,
            "material_news": material_news,
            "recent_filings": recent_filings,
        }

    @staticmethod
    def build_candidate_expression(
        theme: ThemeHypothesis,
        research: Dict[str, Any],
        portfolio_symbols: set,
    ) -> CandidateExpression:
        symbol = research.get("symbol", "").upper().strip()
        market_data = research.get("market_data", {})
        material_news = research.get("material_news", [])
        relevant_filings = research.get("recent_filings", [])
        ownership_intel: OwnershipIntel = research.get("ownership_intel", {})
        street_consensus: StreetConsensus = research.get("street_consensus", {})
        ownership_has_signal = ownership_intel.get("signal_strength") in ["medium", "high"] and not ownership_intel.get("error")
        street_has_signal = street_consensus.get("signal_strength") in ["medium", "high"] and not street_consensus.get("error")

        catalysts = []
        if material_news:
            catalysts.extend([item.get("headline", "") for item in material_news[:2] if item.get("headline")])
        if relevant_filings:
            catalysts.append(
                f"{relevant_filings[0].get('form', 'Filing')} filed on {relevant_filings[0].get('date', 'unknown date')}"
            )
        if ownership_has_signal and ownership_intel.get("summary"):
            catalysts.append(str(ownership_intel.get("summary")))
        if street_has_signal and street_consensus.get("summary"):
            catalysts.append(str(street_consensus.get("summary")))
        if abs(market_data.get("five_day_change_pct") or 0) >= 6:
            catalysts.append(
                f"Shares moved `{market_data.get('five_day_change_pct')}%` over five days while the theme gained attention."
            )

        data_gaps = []
        if market_data.get("current_price") is None:
            data_gaps.append("Missing live price context")
        if not material_news:
            data_gaps.append("No material recent news captured")
        if not relevant_filings:
            data_gaps.append("No fresh SEC filing catalyst")
        if not ownership_has_signal:
            data_gaps.append("No fresh ownership catalyst")
        if not street_has_signal:
            data_gaps.append("No strong analyst-consensus signal")

        signal_count = sum(
            1
            for present in [
                bool(material_news),
                bool(relevant_filings),
                ownership_has_signal,
                street_has_signal,
                abs(market_data.get("five_day_change_pct") or 0) >= 6,
            ]
            if present
        )

        company_name = market_data.get("name", symbol)
        return {
            "symbol": symbol,
            "source_etf": (theme.get("mapped_etfs") or [""])[0],
            "is_existing_position": symbol in portfolio_symbols,
            "why_this_company": (
                f"{company_name} appears to be a visible expression of the "
                f"{theme.get('theme_name', theme.get('theme_key', 'current'))} theme."
            ),
            "company_catalysts": catalysts[:3],
            "relevant_filings": relevant_filings,
            "material_news": material_news,
            "price_context": market_data,
            "data_gaps": data_gaps,
            "theme_key": theme.get("theme_key"),
            "theme_name": theme.get("theme_name"),
            "ownership_intel": ownership_intel,
            "street_consensus": street_consensus,
            "signal_count": signal_count,
        }

    @staticmethod
    def _build_theme_why_now(
        theme_name: str,
        etf_symbol: str,
        etf_market_data: Dict[str, Any],
        news_count: int,
        filing_count: int,
        macro_stats: Dict[str, Any],
    ) -> str:
        fragments = [
            f"{theme_name} is in focus through **{etf_symbol}**.",
        ]
        day_change = etf_market_data.get("day_change_pct")
        five_day = etf_market_data.get("five_day_change_pct")
        if day_change is not None:
            fragments.append(f"The ETF moved `{day_change}%` on the day.")
        if five_day is not None:
            fragments.append(f"Five-day momentum is `{five_day}%`.")
        if news_count:
            fragments.append(f"There were `{news_count}` recent material news items across sector leaders.")
        if filing_count:
            fragments.append(f"`{filing_count}` recent SEC filing catalysts were detected.")
        if macro_stats:
            fragments.append(
                f"Macro context reference: Fed Funds `{macro_stats.get('fed_funds_rate', 'N/A')}`, "
                f"Inflation `{macro_stats.get('cpi_inflation', macro_stats.get('cpi_inflation_index', 'N/A'))}`."
            )
        return " ".join(fragments)

    @staticmethod
    def _should_activate_theme(
        etf_market_data: Dict[str, Any],
        news_count: int,
        filing_count: int,
        run_type: str,
    ) -> Dict[str, Any]:
        day_move = abs(etf_market_data.get("day_change_pct") or 0)
        five_day_move = abs(etf_market_data.get("five_day_change_pct") or 0)

        market_trigger = day_move >= 2.5 or five_day_move >= 6.0
        news_trigger = news_count >= 2
        filing_trigger = filing_count >= 1

        trigger_sources = []
        if market_trigger:
            trigger_sources.append("market_action")
        if news_trigger:
            trigger_sources.append("news_cluster")
        if filing_trigger:
            trigger_sources.append("filings")

        if run_type == "sweep":
            active = (market_trigger and (news_trigger or filing_trigger)) or news_trigger or filing_trigger
        else:
            active = market_trigger or news_count > 0 or filing_count > 0

        confidence = "low"
        if len(trigger_sources) >= 3:
            confidence = "high"
        elif len(trigger_sources) >= 2:
            confidence = "medium"

        return {"active": active, "trigger_sources": trigger_sources, "confidence_level": confidence}

    @staticmethod
    def run_theme_discovery(
        policy_profile: PolicyProfile,
        portfolio: List[Dict[str, Any]],
        run_type: str = "deep",
        focus_override: List[str] = None,
    ) -> Dict[str, Any]:
        focus_etfs = focus_override or policy_profile.get("priority_etfs") or ["SMH", "IGV"]
        macro_stats = ScoutAgent.collect_macro_regime_research()
        market_action = ScoutAgent.collect_market_action_research(focus_etfs)
        theme_hypotheses: List[ThemeHypothesis] = []
        portfolio_symbols = {holding["symbol"].upper().strip() for holding in portfolio}

        for theme_record in get_theme_records_for_etfs(focus_etfs):
            etf_symbol = theme_record["etf"]
            holdings = fetch_etf_holdings(etf_symbol, limit=10)
            leader_symbols = holdings[:4]
            leader_research = [
                ScoutAgent.collect_symbol_research(symbol, theme_context=theme_record["theme_name"])
                for symbol in leader_symbols
            ]

            news_count = sum(len(research["material_news"]) for research in leader_research)
            filing_count = sum(len(research["recent_filings"]) for research in leader_research)
            activation = ScoutAgent._should_activate_theme(
                market_action.get(etf_symbol, {}),
                news_count,
                filing_count,
                run_type=run_type,
            )
            if not activation["active"]:
                continue

            evidence_items = []
            if market_action.get(etf_symbol):
                evidence_items.append(
                    {
                        "source": "market_action",
                        "entity": etf_symbol,
                        "details": market_action[etf_symbol],
                    }
                )
            for research in leader_research:
                if research["material_news"]:
                    evidence_items.append(
                        {
                            "source": "news",
                            "entity": research["symbol"],
                            "details": research["material_news"][:2],
                        }
                    )
                if research["recent_filings"]:
                    evidence_items.append(
                        {
                            "source": "filings",
                            "entity": research["symbol"],
                            "details": research["recent_filings"][:1],
                        }
                    )

            theme_hypotheses.append(
                {
                    "theme_key": theme_record["theme_key"],
                    "theme_name": theme_record["theme_name"],
                    "why_now": ScoutAgent._build_theme_why_now(
                        theme_record["theme_name"],
                        etf_symbol,
                        market_action.get(etf_symbol, {}),
                        news_count,
                        filing_count,
                        macro_stats,
                    ),
                    "mapped_etfs": [etf_symbol],
                    "evidence_items": evidence_items,
                    "confidence_level": activation["confidence_level"],
                    "invalidators": theme_record.get("invalidators", []),
                    "candidate_symbols": holdings[:8],
                    "trigger_sources": activation["trigger_sources"],
                    "portfolio_overlap": sorted(portfolio_symbols.intersection(set(holdings[:8]))),
                }
            )

        return {
            "macro": macro_stats,
            "market_action": market_action,
            "themes": theme_hypotheses,
            "policy_focus": focus_etfs,
        }

    @staticmethod
    def expand_theme_candidates(
        theme: ThemeHypothesis,
        portfolio: List[Dict[str, Any]],
        candidate_limit: int = 6,
    ) -> List[CandidateExpression]:
        portfolio_symbols = {holding["symbol"].upper().strip() for holding in portfolio}
        candidate_expressions: List[CandidateExpression] = []
        for symbol in theme.get("candidate_symbols", [])[:candidate_limit]:
            research = ScoutAgent.collect_symbol_research(symbol, theme_context=theme.get("theme_name", ""))
            candidate_expression = ScoutAgent.build_candidate_expression(theme, research, portfolio_symbols)
            signal_count = candidate_expression.get("signal_count", 0)
            if signal_count == 0 and symbol not in portfolio_symbols:
                continue
            candidate_expressions.append(candidate_expression)

        candidate_expressions.sort(
            key=lambda item: (
                item.get("signal_count", 0),
                len(item.get("material_news", [])),
                len(item.get("relevant_filings", [])),
            ),
            reverse=True,
        )
        return candidate_expressions[:candidate_limit]
