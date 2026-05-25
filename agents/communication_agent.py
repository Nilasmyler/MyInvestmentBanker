import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from database.supabase_client import (
    fetch_investment_thesis,
    get_user_preference,
    save_investment_thesis,
    save_user_preference,
    update_portfolio_holding,
)
from services.portfolio_service import (
    build_portfolio_change_summary,
    brokerage_enabled,
    get_portfolio_snapshot,
    should_sync_portfolio_before_analysis,
    should_sync_portfolio_on_read,
    summarize_portfolio_state,
    sync_portfolio_from_brokerage,
)
from utils.discovery_support import (
    build_preference_summary,
    extract_preference_update_from_text,
    get_default_policy_profile,
    get_theme_registry,
    merge_policy_profile,
    normalize_policy_profile,
    parse_policy_text_fallback,
)

load_dotenv()
logger = logging.getLogger("MyInvestmentBanker.agents.communication")
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

PENDING_FOLLOW_UP_PREFIX = "pending_follow_up::"
DISCOVERY_SUPPRESSIONS_KEY = "discovery_suppressions"
POLICY_QUERY_PHRASES = [
    "show my policy",
    "show my preferences",
    "what are my preferences",
    "what do you know about my preferences",
    "summarize my preferences",
    "what is my investment style",
]

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def llm_available() -> bool:
    return bool(GEMINI_API_KEY)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(raw_value: str) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except Exception:
        return None


def _format_human_date(raw_value: str) -> str:
    parsed = _parse_iso(raw_value)
    if not parsed:
        return "the prior snapshot"
    return parsed.strftime("%Y-%m-%d")


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def _normalize_command(raw_token: str) -> str:
    raw_token = (raw_token or "").strip().lower()
    if not raw_token:
        return ""
    return raw_token if raw_token.startswith("/") else f"/{raw_token}"


def _pending_follow_up_key(user_id: str) -> str:
    return f"{PENDING_FOLLOW_UP_PREFIX}{user_id}"


def _build_policy_profile(policy_text: str) -> Dict[str, Any]:
    if not policy_text:
        return get_default_policy_profile()

    if not llm_available():
        return parse_policy_text_fallback(policy_text)

    system_instruction = (
        "You are the Portfolio Manager Front Desk of MyInvestmentBanker.\n"
        "Parse the user's investment policy update into structured JSON.\n"
        "Return only valid JSON with these keys: "
        "{\"policy_text\": str, \"preferred_themes\": [str], \"excluded_themes\": [str], "
        "\"style_bias\": [str], \"risk_avoidances\": [str], \"priority_etfs\": [str], "
        "\"risk_profile\": str, \"time_horizon\": str, \"market_preferences\": [str], "
        "\"company_preferences\": [str], \"preference_summary\": str}"
    )
    prompt = f"Convert this policy into structured JSON: {policy_text}"
    structured_response = generate_llm_response(prompt, system_instruction)
    parsed = _extract_json_payload(structured_response)
    if parsed:
        return normalize_policy_profile(parsed)
    return parse_policy_text_fallback(policy_text)


def generate_llm_response(prompt: str, system_instruction: str = "") -> str:
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured in .env."
    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        return f"Error executing model reasoning step: {e}"


class CommunicationAgent:
    """
    Handles user commands, structured follow-up actions, and plain-language report synthesis.
    """

    @staticmethod
    def _load_policy_profile() -> Dict[str, Any]:
        return normalize_policy_profile(
            get_user_preference("screening_policy"),
            legacy_focus=get_user_preference("active_opportunity_focus"),
        )

    @staticmethod
    def _save_policy_profile(policy_profile: Dict[str, Any]) -> None:
        save_user_preference("screening_policy", policy_profile)
        save_user_preference("broad_investment_policy", {"policy_text": policy_profile["policy_text"]})
        save_user_preference("active_opportunity_focus", policy_profile.get("priority_etfs", []))

    @staticmethod
    def _load_pending_follow_up(user_id: str) -> Optional[Dict[str, Any]]:
        pending = get_user_preference(_pending_follow_up_key(user_id))
        if not isinstance(pending, dict):
            return None
        if not pending.get("kind"):
            return None
        expires_at = _parse_iso(str(pending.get("expires_at", "")))
        if expires_at and expires_at < _now_utc():
            CommunicationAgent.clear_pending_follow_up(user_id)
            return None
        return pending

    @staticmethod
    def _save_pending_follow_up(user_id: str, payload: Dict[str, Any]) -> None:
        save_user_preference(_pending_follow_up_key(user_id), payload)

    @staticmethod
    def clear_pending_follow_up(user_id: str) -> None:
        save_user_preference(_pending_follow_up_key(user_id), {"status": "cleared", "expires_at": _to_iso(_now_utc())})

    @staticmethod
    def _load_discovery_suppressions() -> List[Dict[str, Any]]:
        suppressions = get_user_preference(DISCOVERY_SUPPRESSIONS_KEY)
        if not isinstance(suppressions, list):
            return []
        active_items = []
        now = _now_utc()
        for item in suppressions:
            if not isinstance(item, dict):
                continue
            until = _parse_iso(str(item.get("until", "")))
            if until and until > now:
                active_items.append(item)
        if active_items != suppressions:
            save_user_preference(DISCOVERY_SUPPRESSIONS_KEY, active_items)
        return active_items

    @staticmethod
    def _save_discovery_suppressions(items: List[Dict[str, Any]]) -> None:
        save_user_preference(DISCOVERY_SUPPRESSIONS_KEY, items)

    @staticmethod
    def is_suppressed_discovery_idea(symbol: str, theme_key: Optional[str] = None) -> bool:
        symbol_clean = (symbol or "").upper().strip()
        theme_clean = (theme_key or "").strip().lower()
        for item in CommunicationAgent._load_discovery_suppressions():
            if item.get("symbol", "").upper().strip() != symbol_clean:
                continue
            stored_theme = str(item.get("theme_key", "")).strip().lower()
            if not theme_clean or not stored_theme or stored_theme == theme_clean:
                return True
        return False

    @staticmethod
    def suppress_discovery_idea(symbol: str, theme_key: str = "", days: int = 14) -> None:
        active_items = [
            item
            for item in CommunicationAgent._load_discovery_suppressions()
            if not (
                item.get("symbol", "").upper().strip() == symbol.upper().strip()
                and str(item.get("theme_key", "")).strip().lower() == str(theme_key).strip().lower()
            )
        ]
        active_items.append(
            {
                "symbol": symbol.upper().strip(),
                "theme_key": str(theme_key).strip().lower(),
                "until": _to_iso(_now_utc() + timedelta(days=max(days, 1))),
            }
        )
        CommunicationAgent._save_discovery_suppressions(active_items)

    @staticmethod
    def _resolve_theme_selection(reference_text: str, pending_action: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
        registry = get_theme_registry()
        ref = (reference_text or "").strip()
        ref_lower = ref.lower()

        if pending_action and ref_lower in ["theme", "this theme", "focus theme", "exclude theme", "snooze idea", "idea"]:
            return {
                "theme_key": str(pending_action.get("theme_key", "")),
                "theme_name": str(pending_action.get("theme_name", "")),
                "etf": str(pending_action.get("etf", "")),
                "symbol": str(pending_action.get("symbol", "")),
            }

        ref_upper = ref.upper()
        if ref_upper in registry:
            return {
                "theme_key": registry[ref_upper]["theme_key"],
                "theme_name": registry[ref_upper]["theme_name"],
                "etf": ref_upper,
                "symbol": str(pending_action.get("symbol", "")) if pending_action else "",
            }

        for etf, item in registry.items():
            if ref_lower == item["theme_key"] or ref_lower in item["theme_name"].lower():
                return {
                    "theme_key": item["theme_key"],
                    "theme_name": item["theme_name"],
                    "etf": etf,
                    "symbol": str(pending_action.get("symbol", "")) if pending_action else "",
                }
        return None

    @staticmethod
    def _macro_interpretation(macro_data: Dict[str, Any]) -> str:
        fed_funds = str(macro_data.get("fed_funds_rate", "N/A"))
        inflation = str(macro_data.get("cpi_inflation", macro_data.get("cpi_inflation_index", "N/A")))
        if "5." in fed_funds:
            return (
                f"Borrowing conditions still look fairly tight with Fed Funds around `{fed_funds}`. "
                f"That usually means the market will be more selective, so the bar for new ideas should stay high. "
                f"Inflation is around `{inflation}`."
            )
        return f"Macro conditions look mixed. Fed Funds is `{fed_funds}` and inflation is `{inflation}`."

    @staticmethod
    def _append_follow_up(base_text: str, follow_up_text: str) -> str:
        if not follow_up_text:
            return base_text
        if not base_text:
            return follow_up_text
        return f"{base_text}\n\n{follow_up_text}"

    @staticmethod
    def _format_policy_report(policy_profile: Optional[Dict[str, Any]] = None) -> str:
        profile = normalize_policy_profile(policy_profile or CommunicationAgent._load_policy_profile())
        return (
            "📋 **Active Discovery Policy**\n\n"
            f"🔍 **Working summary:** `{profile.get('preference_summary', profile['policy_text'])}`\n\n"
            f"🧭 **Risk posture:** `{profile.get('risk_profile', 'balanced').replace('_', ' ')}`\n"
            f"⏳ **Time horizon:** `{profile.get('time_horizon', 'long_term').replace('_', ' ')}`\n"
            f"🌍 **Markets:** `{', '.join(profile.get('market_preferences', [])) or 'No explicit market bias stored'}`\n"
            f"🏢 **Company preferences:** `{', '.join(profile.get('company_preferences', [])) or 'None'}`\n"
            f"🎯 **Priority ETFs:** `{', '.join(profile.get('priority_etfs', [])) or 'None'}`\n"
            f"🧠 **Preferred themes:** `{', '.join(profile.get('preferred_themes', [])) or 'None'}`\n"
            f"🚫 **Excluded themes:** `{', '.join(profile.get('excluded_themes', [])) or 'None'}`\n"
            f"🛡️ **Risk avoidances:** `{', '.join(profile.get('risk_avoidances', [])) or 'None'}`\n\n"
            "💡 *You can update this directly with* `/policy update ...` *or just tell me your preferences in plain language.*"
        )

    @staticmethod
    def detect_policy_query(message_text: str) -> bool:
        lower_text = message_text.strip().lower()
        return any(phrase in lower_text for phrase in POLICY_QUERY_PHRASES)

    @staticmethod
    def detect_stock_analysis_request(message_text: str) -> Optional[Dict[str, str]]:
        stripped = message_text.strip()
        if not stripped:
            return None

        patterns = [
            r"^/analyze\s+([A-Za-z][A-Za-z.\-]{0,9})(?:\s+(.*))?$",
            r"^(?:analyze|analyse|review|check|look at|look into)\s+([A-Za-z][A-Za-z.\-]{0,9})(?:\s+(.*))?$",
            r"^(?:what about|thoughts on|can you analyze|can you review|could you analyze|could you review)\s+([A-Za-z][A-Za-z.\-]{0,9})(?:\s+(.*))?$",
        ]

        for pattern in patterns:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if not match:
                continue
            symbol = match.group(1).upper().replace(".", "-").strip()
            if not re.fullmatch(r"[A-Z]{1,6}(?:-[A-Z])?", symbol):
                return None
            return {
                "symbol": symbol,
                "context": (match.group(2) or "").strip(),
            }
        return None

    @staticmethod
    def _format_market_cap(market_cap: Optional[float]) -> str:
        if market_cap in [None, 0]:
            return "unknown size"
        try:
            value = float(market_cap)
        except (TypeError, ValueError):
            return "unknown size"

        if value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f}T USD"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B USD"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M USD"
        return f"{value:.0f} USD"

    @staticmethod
    def _build_portfolio_change_lines(portfolio_state: List[Dict[str, Any]]) -> List[str]:
        snapshot_summary = build_portfolio_change_summary(portfolio_state)
        current_snapshot = snapshot_summary.get("current_snapshot") or {}
        baseline_snapshot = snapshot_summary.get("baseline_snapshot") or {}

        lines = []
        current_market_value = current_snapshot.get("total_market_value")
        if current_market_value is not None:
            lines.append(f"Current tracked market value is `{float(current_market_value):.2f}` USD.")
        else:
            lines.append(
                f"Tracked cost basis across active holdings is `{float(current_snapshot.get('total_cost_basis', 0.0)):.2f}` USD."
            )

        if not baseline_snapshot:
            lines.append("This is the first stored portfolio baseline, so value-change tracking starts from this run.")
            return lines

        baseline_date = _format_human_date(str(baseline_snapshot.get("captured_at", "")))
        elapsed_days = snapshot_summary.get("elapsed_days")
        comparison_label = (
            f"vs the prior weekly baseline from `{baseline_date}`"
            if elapsed_days is not None and elapsed_days >= 6
            else f"vs the prior recorded snapshot from `{baseline_date}`"
        )

        market_value_change = snapshot_summary.get("market_value_change")
        market_value_change_pct = snapshot_summary.get("market_value_change_pct")
        if market_value_change is not None:
            direction = "up" if market_value_change >= 0 else "down"
            change_text = f"{abs(float(market_value_change)):.2f}"
            pct_text = f" ({market_value_change_pct:+.2f}%)" if market_value_change_pct is not None else ""
            lines.append(f"Portfolio value is {direction} `{change_text}` USD{pct_text} {comparison_label}.")
        else:
            cost_basis_change = float(snapshot_summary.get("cost_basis_change", 0.0))
            direction = "higher" if cost_basis_change >= 0 else "lower"
            lines.append(
                f"Cost basis is `{abs(cost_basis_change):.2f}` USD {direction} {comparison_label}, but live market values were not available in both snapshots."
            )

        new_positions = snapshot_summary.get("new_positions", [])
        if new_positions:
            lines.append(f"New active positions since then: `{', '.join(new_positions)}`.")

        removed_positions = snapshot_summary.get("removed_positions", [])
        if removed_positions:
            lines.append(f"Positions that left the active book since then: `{', '.join(removed_positions)}`.")

        largest_position_changes = snapshot_summary.get("largest_position_changes", [])
        if largest_position_changes:
            move_parts = []
            for item in largest_position_changes[:2]:
                pct_text = f" ({item['delta_pct']:+.2f}%)" if item.get("delta_pct") is not None else ""
                move_parts.append(f"{item['symbol']} `{item['delta']:+.2f}` USD{pct_text}")
            lines.append("Largest position-value moves: " + "; ".join(move_parts) + ".")

        return lines

    @staticmethod
    def _infer_theme_from_symbol_research(
        symbol: str,
        research: Dict[str, Any],
        policy_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        registry = get_theme_registry()
        market_data = research.get("market_data", {})
        material_news = research.get("material_news", [])

        lookup_fragments = [
            symbol,
            market_data.get("sector", ""),
            market_data.get("industry", ""),
        ]
        lookup_fragments.extend(item.get("headline", "") for item in material_news[:3])
        lookup_text = " ".join(fragment for fragment in lookup_fragments if fragment).lower()

        matching_etfs = []
        for etf, item in registry.items():
            if any(keyword in lookup_text for keyword in item.get("keywords", [])):
                matching_etfs.append(etf)

        if matching_etfs and matching_etfs[0] in registry:
            record = registry[matching_etfs[0]]
            return {
                "theme_key": record["theme_key"],
                "theme_name": record["theme_name"],
                "mapped_etfs": [matching_etfs[0]],
                "invalidators": record.get("invalidators", []),
            }

        fallback_name = market_data.get("sector") or market_data.get("industry") or f"{symbol} Company Context"
        fallback_key = re.sub(r"[^a-z0-9]+", "_", fallback_name.lower()).strip("_") or "company_specific"
        return {
            "theme_key": fallback_key,
            "theme_name": fallback_name,
            "mapped_etfs": [],
            "invalidators": [],
        }

    @staticmethod
    def _summarize_preference_learning_result(
        learning_result: Dict[str, Any],
    ) -> str:
        profile = learning_result.get("policy_profile", get_default_policy_profile())
        changed = bool(learning_result.get("changed"))
        update = learning_result.get("update", {})

        learned_bits = []
        if update.get("risk_profile"):
            learned_bits.append(f"a `{update['risk_profile'].replace('_', ' ')}` risk posture")
        if update.get("time_horizon"):
            learned_bits.append(f"a `{update['time_horizon'].replace('_', ' ')}` horizon")
        if update.get("market_preferences"):
            learned_bits.append(f"market focus on `{', '.join(update['market_preferences'])}`")
        if update.get("preferred_themes"):
            learned_bits.append(f"theme interest in `{', '.join(update['preferred_themes'])}`")
        if update.get("excluded_themes"):
            learned_bits.append(f"theme exclusions for `{', '.join(update['excluded_themes'])}`")
        if update.get("company_preferences"):
            learned_bits.append(f"company bias toward `{', '.join(update['company_preferences'])}`")

        status_line = "🧭 **Preference Profile Updated**" if changed else "🧭 **Preference Profile Confirmed**"
        detail_line = (
            "I updated your working profile from this conversation."
            if changed
            else "That already fits the working profile I had on file."
        )

        if learned_bits:
            detail_line += " I heard " + "; ".join(learned_bits[:4]) + "."

        return (
            f"{status_line}\n\n"
            f"{detail_line}\n"
            f"🔍 **Working summary:** `{profile.get('preference_summary', build_preference_summary(profile))}`\n"
            "I will apply this to discovery, single-stock analysis, and recommendation tone going forward."
        )

    @staticmethod
    def learn_preferences_from_message(user_id: str, message_text: str) -> Optional[Dict[str, Any]]:
        update = extract_preference_update_from_text(message_text)
        if not update:
            return None

        current_profile = CommunicationAgent._load_policy_profile()
        merged_profile = merge_policy_profile(current_profile, update)
        changed = json.dumps(current_profile, sort_keys=True) != json.dumps(merged_profile, sort_keys=True)

        if changed:
            CommunicationAgent._save_policy_profile(merged_profile)

        return {
            "update": update,
            "policy_profile": merged_profile,
            "changed": changed,
        }

    @staticmethod
    def _build_policy_fit_interpretation(
        policy_profile: Dict[str, Any],
        theme: Dict[str, Any],
        candidate_review: Dict[str, Any],
    ) -> str:
        alignment_notes = []
        theme_key = theme.get("theme_key", "")
        market_data = candidate_review.get("price_context", {})
        beta = market_data.get("beta")

        if theme_key and theme_key in policy_profile.get("preferred_themes", []):
            alignment_notes.append("The theme already sits inside your preferred hunting ground.")
        if theme_key and theme_key in policy_profile.get("excluded_themes", []):
            alignment_notes.append("This theme conflicts with your current exclusions.")
        if policy_profile.get("risk_profile") == "conservative" and beta and beta > 1.6:
            alignment_notes.append("It is less aligned with a conservative posture because the shares likely swing harder than the market.")
        elif policy_profile.get("risk_profile") == "conservative":
            alignment_notes.append("It is reasonably aligned with a conservative posture because the volatility profile does not look extreme.")

        if policy_profile.get("time_horizon") == "long_term":
            alignment_notes.append("This should be judged on multi-quarter execution, not just short-term price action.")
        elif policy_profile.get("time_horizon") == "short_term":
            alignment_notes.append("Near-term catalysts matter more here because your stated horizon is shorter.")

        if not alignment_notes:
            alignment_notes.append("Nothing here clearly conflicts with the current preference profile.")
        return " ".join(alignment_notes[:3])

    @staticmethod
    def prepare_single_stock_follow_up(user_id: str, analysis_result: Dict[str, Any]) -> str:
        theme = analysis_result.get("theme", {})
        symbol = analysis_result.get("symbol", "")
        registry_theme_keys = {item["theme_key"] for item in get_theme_registry().values()}
        if (
            not theme
            or not symbol
            or (not theme.get("mapped_etfs") and theme.get("theme_key") not in registry_theme_keys)
        ):
            return ""

        payload = {
            "kind": "stock_analysis_followup",
            "theme_key": theme.get("theme_key", ""),
            "theme_name": theme.get("theme_name", ""),
            "etf": (theme.get("mapped_etfs") or [""])[0],
            "symbol": symbol,
            "expires_at": _to_iso(_now_utc() + timedelta(days=2)),
        }
        CommunicationAgent._save_pending_follow_up(user_id, payload)

        return (
            "❓ **Follow-up Question**\n"
            f"This review looks most connected to **{theme.get('theme_name', 'this theme')}**. "
            "How would you like to handle this theme and company going forward in your personal discovery policy?\n"
            "You can reply in plain language or use one of these predefined actions:\n"
            "• `focus theme` — keep this theme in active discovery focus\n"
            "• `exclude theme` — stop surfacing this theme\n"
            "• `snooze idea` — suppress this exact idea for 14 days\n"
            "• `skip` — clear this follow-up"
        )

    @staticmethod
    def generate_single_stock_analysis(user_id: str, symbol: str, user_context: str = "") -> str:
        from agents.cfa_agent import CFAAgent
        from agents.research_planner_agent import ResearchPlannerAgent
        from agents.risk_agent import RiskAgent

        symbol = symbol.upper().strip()
        learning_result = CommunicationAgent.learn_preferences_from_message(user_id, user_context) if user_context else None
        policy_profile = CommunicationAgent._load_policy_profile()
        portfolio = get_portfolio_snapshot(sync_from_broker=should_sync_portfolio_before_analysis())
        portfolio_symbols = {holding["symbol"].upper().strip() for holding in portfolio}

        research = ResearchPlannerAgent.collect_symbol_research(
            symbol,
            run_type="deep",
            is_existing_position=symbol in portfolio_symbols,
        )
        theme = CommunicationAgent._infer_theme_from_symbol_research(symbol, research, policy_profile)
        from agents.scout_agent import ScoutAgent

        candidate_expression = ScoutAgent.build_candidate_expression(theme, research, portfolio_symbols)
        candidate_review = CFAAgent.review_discovery_candidate(theme, candidate_expression, policy_profile)
        risk_audit = RiskAgent.audit_discovery_theme(theme, [candidate_review], policy_profile, max_recommendations=1)

        analysis_result = {
            "symbol": symbol,
            "theme": theme,
            "research": research,
            "candidate_review": candidate_review,
            "risk_audit": risk_audit,
            "policy_profile": policy_profile,
        }

        briefing = CommunicationAgent.compile_single_stock_briefing(analysis_result)
        if learning_result:
            briefing = f"{CommunicationAgent._summarize_preference_learning_result(learning_result)}\n\n{briefing}"

        follow_up = CommunicationAgent.prepare_single_stock_follow_up(user_id, analysis_result)
        return CommunicationAgent._append_follow_up(briefing, follow_up)

    @staticmethod
    def prepare_discovery_follow_up(user_id: str, discovery_result: Dict[str, Any]) -> str:
        recommendations = discovery_result.get("recommendations", [])
        themes = discovery_result.get("themes", [])
        if not recommendations or not themes:
            return ""

        primary_recommendation = recommendations[0]
        matching_theme = next(
            (
                theme
                for theme in themes
                if theme.get("theme_key") == primary_recommendation.get("theme_key")
                or theme.get("theme_name") == primary_recommendation.get("theme_name")
            ),
            themes[0],
        )
        etf = (matching_theme.get("mapped_etfs") or [""])[0]
        payload = {
            "kind": "discovery_theme_followup",
            "theme_key": matching_theme.get("theme_key", ""),
            "theme_name": matching_theme.get("theme_name", ""),
            "etf": etf,
            "symbol": primary_recommendation.get("symbol", ""),
            "expires_at": _to_iso(_now_utc() + timedelta(days=2)),
        }
        CommunicationAgent._save_pending_follow_up(user_id, payload)

        theme_name = matching_theme.get("theme_name", "this theme")
        symbol = primary_recommendation.get("symbol", "this idea")
        return (
            "❓ **Follow-up Question**\n"
            f"The strongest thread in this run was **{theme_name}**, with **{symbol}** as the clearest expression. "
            "How does this align with your current thesis or focus? Would you like to adjust your discovery policy for this theme?\n"
            "You can reply in plain language or use one of these predefined actions:\n"
            "• `focus theme` — keep this theme in active discovery focus\n"
            "• `exclude theme` — stop surfacing this theme\n"
            "• `snooze idea` — suppress this exact idea for 14 days\n"
            "• `skip` — clear this follow-up"
        )

    @staticmethod
    def prepare_portfolio_follow_up(user_id: str, portfolio_state: Optional[List[Dict[str, Any]]] = None) -> str:
        holdings = portfolio_state if portfolio_state is not None else get_portfolio_snapshot(sync_from_broker=False)
        missing_thesis_symbol = ""
        for holding in holdings:
            symbol = holding["symbol"].upper()
            if not fetch_investment_thesis(symbol):
                missing_thesis_symbol = symbol
                break

        if not missing_thesis_symbol:
            return ""

        payload = {
            "kind": "portfolio_thesis_followup",
            "symbol": missing_thesis_symbol,
            "expires_at": _to_iso(_now_utc() + timedelta(days=3)),
        }
        CommunicationAgent._save_pending_follow_up(user_id, payload)

        return (
            "❓ **Follow-up Question**\n"
            f"I still do not have your investment thesis logged for **{missing_thesis_symbol}**. "
            "What is your core investment hypothesis or primary reason for holding this position?\n"
            "You can reply in plain language or use one of these predefined actions:\n"
            f"• `thesis {missing_thesis_symbol} [your thesis]` — store your reasoning\n"
            "• `skip` — clear this follow-up"
        )

    @staticmethod
    def handle_pending_follow_up(user_id: str, message_text: str) -> Optional[str]:
        pending_action = CommunicationAgent._load_pending_follow_up(user_id)
        if not pending_action:
            return None

        normalized_text = message_text.strip().lower()
        normalized_cmd = _normalize_command((message_text.strip().split() or [""])[0])

        if normalized_text in ["skip", "/skip", "not now", "later"]:
            CommunicationAgent.clear_pending_follow_up(user_id)
            return "Understood. I cleared the pending follow-up."

        if pending_action.get("kind") in ["discovery_theme_followup", "stock_analysis_followup"]:
            theme_target = CommunicationAgent._resolve_theme_selection("this theme", pending_action)
            if normalized_text in ["focus theme", "/focus theme", "keep focus", "focus it"]:
                policy_profile = CommunicationAgent._load_policy_profile()
                if theme_target and theme_target["etf"] and theme_target["etf"] not in policy_profile["priority_etfs"]:
                    policy_profile["priority_etfs"].append(theme_target["etf"])
                if theme_target and theme_target["theme_key"] and theme_target["theme_key"] not in policy_profile["preferred_themes"]:
                    policy_profile["preferred_themes"].append(theme_target["theme_key"])
                if theme_target and theme_target["theme_key"] in policy_profile["excluded_themes"]:
                    policy_profile["excluded_themes"].remove(theme_target["theme_key"])
                CommunicationAgent._save_policy_profile(policy_profile)
                CommunicationAgent.clear_pending_follow_up(user_id)
                return (
                    f"✅ **Focus updated.** I will keep **{theme_target['theme_name']}** in active discovery focus."
                )

            if normalized_text in ["exclude theme", "/exclude theme", "mute theme", "skip this theme"]:
                policy_profile = CommunicationAgent._load_policy_profile()
                if theme_target and theme_target["theme_key"] and theme_target["theme_key"] not in policy_profile["excluded_themes"]:
                    policy_profile["excluded_themes"].append(theme_target["theme_key"])
                if theme_target and theme_target["theme_key"] in policy_profile["preferred_themes"]:
                    policy_profile["preferred_themes"].remove(theme_target["theme_key"])
                if theme_target and theme_target["etf"] in policy_profile["priority_etfs"]:
                    policy_profile["priority_etfs"].remove(theme_target["etf"])
                CommunicationAgent._save_policy_profile(policy_profile)
                CommunicationAgent.clear_pending_follow_up(user_id)
                return f"✅ **Theme excluded.** I will stop surfacing **{theme_target['theme_name']}** for now."

            if normalized_text in ["snooze idea", "/snooze idea", "snooze it", "mute this idea"]:
                CommunicationAgent.suppress_discovery_idea(
                    symbol=str(pending_action.get("symbol", "")),
                    theme_key=str(pending_action.get("theme_key", "")),
                    days=14,
                )
                CommunicationAgent.clear_pending_follow_up(user_id)
                return (
                    f"✅ **Idea snoozed.** I will suppress **{pending_action.get('symbol', 'this idea')}** "
                    "for 14 days unless a stronger new catalyst appears."
                )

        if pending_action.get("kind") == "portfolio_thesis_followup":
            if normalized_cmd == "/thesis":
                return None
            if normalized_text in ["yes", "/yes"]:
                return (
                    f"Reply with `thesis {pending_action.get('symbol', 'TICKER')} [your thesis]` "
                    "and I will store it."
                )

        return None

    @staticmethod
    def parse_user_command(user_id: str, message_text: str) -> str:
        tokens = message_text.strip().split()
        if not tokens:
            return "Command empty. Send /start or /help to see options."

        pending_result = CommunicationAgent.handle_pending_follow_up(user_id, message_text)
        if pending_result is not None:
            return pending_result

        cmd = _normalize_command(tokens[0])
        stock_analysis_request = CommunicationAgent.detect_stock_analysis_request(message_text)

        if not message_text.strip().startswith("/") and CommunicationAgent.detect_policy_query(message_text):
            return CommunicationAgent._format_policy_report()

        if stock_analysis_request and cmd != "/help":
            return CommunicationAgent.generate_single_stock_analysis(
                user_id,
                stock_analysis_request["symbol"],
                stock_analysis_request.get("context", ""),
            )

        if cmd in ["/start", "/help"]:
            return (
                "👋 **Welcome to MyInvestmentBanker**!\n\n"
                "Your private wealth manager is active.\n"
                "It tracks your holdings, studies sector and company catalysts, and sends risk-aware discovery notes in plain language.\n"
                "You can also tell me things like `I prefer long-term, lower-risk software leaders` and I will learn from that over time.\n\n"
                "📋 **Available Commands:**\n"
                "• `/portfolio` — View allocations and cost-basis\n"
                "• `/sync` — import holdings from the configured brokerage account\n"
                "• `/add TICKER PRICE QTY` — update a holding\n"
                "• `/remove TICKER` — remove a holding\n"
                "• `/thesis TICKER ...` — store your investment thesis\n"
                "• `/analyze TICKER [optional context]` — run a plain-language review of a stock\n"
                "• `/policy` — view your discovery policy\n"
                "• `/policy update ...` — update your discovery guidance\n"
                "• `/discover` — trigger a deep theme-led discovery run\n"
                "• `/update` — trigger a portfolio review\n"
                "• `/focus ETF_OR_THEME` — keep a theme in active discovery focus\n"
                "• `/exclude ETF_OR_THEME` — suppress a theme from future discovery\n"
                "• `/snooze SYMBOL [days]` — suppress an idea temporarily\n"
                "• `/skip` — clear a pending follow-up"
            )

        if cmd == "/skip":
            CommunicationAgent.clear_pending_follow_up(user_id)
            return "Understood. I cleared the pending follow-up."

        if cmd == "/portfolio":
            sync_result = None
            if brokerage_enabled() and should_sync_portfolio_on_read():
                sync_result = sync_portfolio_from_brokerage()
                holdings = sync_result.get("holdings", []) if sync_result.get("ok") else get_portfolio_snapshot(sync_from_broker=False)
            else:
                holdings = get_portfolio_snapshot(sync_from_broker=False)

            if not holdings:
                if brokerage_enabled():
                    return (
                        "📂 **No active holdings are currently loaded.** "
                        "Use `/sync` to import positions from your brokerage account or `/add TICKER PRICE QTY` to log one manually."
                    )
                return "📂 **Your portfolio is currently empty.** Add holdings with `/add TICKER PRICE QTY`."

            report = "💼 **Active Portfolio Holdings**\n\n"
            if sync_result:
                if sync_result.get("ok"):
                    report += (
                        f"Synced from **{sync_result.get('provider', 'brokerage')}** at "
                        f"`{sync_result.get('fetched_at', 'unknown time')}`.\n\n"
                    )
                else:
                    report += (
                        f"Brokerage refresh failed, so I am showing the last stored snapshot instead.\n"
                        f"Reason: `{sync_result.get('reason', 'unknown error')}`\n\n"
                    )

            portfolio_summary = summarize_portfolio_state(holdings)
            for idx, holding in enumerate(holdings):
                ticker = holding["symbol"]
                qty = float(holding["quantity"])
                price = float(holding["cost_basis"])
                market_value = holding.get("market_value")
                current_price = holding.get("current_price")
                thesis = fetch_investment_thesis(ticker)
                thesis_flag = "I have your thesis on file." if thesis else "I still need your thesis for this position."
                report += (
                    f"{idx + 1}. **{ticker}**\n"
                    f"   • You currently have `{qty:.2f}` shares logged at `{price:.2f}` USD.\n"
                    f"   • In plain language: this is the cost basis I will use when I discuss this holding.\n"
                    f"   • Thesis status: {thesis_flag}\n"
                )
                if current_price is not None:
                    report += f"   • Latest price seen from the broker: `{float(current_price):.2f}` USD.\n"
                if market_value is not None:
                    report += f"   • Current market value: `{float(market_value):.2f}` USD.\n"
                report += "\n"
            report += f"📊 **Total cost basis logged:** `{float(portfolio_summary['total_cost_basis']):.2f}` USD"
            if portfolio_summary.get("market_values_available"):
                report += (
                    f"\n📈 **Total market value from broker snapshot:** "
                    f"`{float(portfolio_summary['total_market_value']):.2f}` USD"
                )
            return report

        if cmd == "/sync":
            if not brokerage_enabled():
                return (
                    "⚠️ **Brokerage integration is not configured.** "
                    "Set `BROKERAGE_PROVIDER` and the provider credentials in `.env` first."
                )

            sync_result = sync_portfolio_from_brokerage()
            if not sync_result.get("ok"):
                return (
                    "❌ **Brokerage sync failed.**\n"
                    f"Provider: `{sync_result.get('provider', 'unknown')}`\n"
                    f"Reason: `{sync_result.get('reason', 'unknown error')}`"
                )

            removal_text = ""
            if sync_result.get("removed_symbols"):
                removal_text = f"\n🧹 Removed closed positions from the tracked portfolio: `{', '.join(sync_result['removed_symbols'])}`"

            persistence_text = ""
            if not sync_result.get("persisted"):
                persistence_text = (
                    "\n⚠️ I could read the brokerage account, but I could not persist the snapshot to Supabase. "
                    "The live holdings are still usable for the current read."
                )

            return (
                f"✅ **Brokerage sync complete.** Imported `{sync_result.get('imported_count', 0)}` holding(s) "
                f"from **{sync_result.get('provider', 'brokerage')}** at `{sync_result.get('fetched_at', 'unknown time')}`."
                f"{removal_text}{persistence_text}"
            )

        if cmd == "/add":
            if len(tokens) < 4:
                return "⚠️ **Usage error:** Use `/add TICKER PRICE QTY` (for example `/add MSFT 420.00 5`)."

            ticker = tokens[1].upper()
            try:
                price = float(tokens[2])
                qty = float(tokens[3])
            except ValueError:
                return "⚠️ **Format error:** Price and quantity must both be numbers."

            success = update_portfolio_holding(ticker, qty, price)
            if success:
                broker_note = ""
                if brokerage_enabled():
                    broker_note = " A future brokerage sync can overwrite this local holding snapshot."
                return (
                    f"✅ **Holding updated.** I now have **{ticker}** logged at `{qty:.2f}` shares and "
                    f"`{price:.2f}` USD as the working cost basis.{broker_note}"
                )
            return "❌ **Database write failed.** Check logs."

        if cmd == "/remove":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/remove TICKER` (for example `/remove MSFT`)."

            ticker = tokens[1].upper()
            success = update_portfolio_holding(ticker, 0, 0)
            if success:
                broker_note = ""
                if brokerage_enabled():
                    broker_note = " If the position still exists at the broker, the next sync will add it back."
                return f"✅ **Holding removed.** **{ticker}** is no longer in your active portfolio.{broker_note}"
            return "❌ **Database write failed.** Check logs."

        if cmd == "/thesis":
            if len(tokens) < 3:
                return "⚠️ **Usage error:** Use `/thesis TICKER [your thesis text]`."

            ticker = tokens[1].upper()
            thesis_text = " ".join(tokens[2:])
            holdings = get_portfolio_snapshot(sync_from_broker=False)
            tickers = [holding["symbol"] for holding in holdings]
            if ticker not in tickers:
                return f"⚠️ **Portfolio mismatch:** Add **{ticker}** first using `/add` before saving a thesis."

            success = save_investment_thesis(ticker, thesis_text)
            if success:
                pending_action = CommunicationAgent._load_pending_follow_up(user_id)
                if pending_action and pending_action.get("kind") == "portfolio_thesis_followup" and pending_action.get("symbol") == ticker:
                    CommunicationAgent.clear_pending_follow_up(user_id)
                return f"✅ **Thesis stored.** I now have your reasoning for **{ticker}** on file."
            return "❌ **Database write failed.** Check logs."

        if cmd == "/analyze":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/analyze TICKER [optional context]`."
            ticker = tokens[1].upper().replace(".", "-")
            user_context = " ".join(tokens[2:]) if len(tokens) > 2 else ""
            return CommunicationAgent.generate_single_stock_analysis(user_id, ticker, user_context)

        if cmd == "/focus":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/focus ETF_OR_THEME` or reply `focus theme` to a discovery follow-up."
            pending_action = CommunicationAgent._load_pending_follow_up(user_id)
            theme_target = CommunicationAgent._resolve_theme_selection(" ".join(tokens[1:]), pending_action=pending_action)
            if not theme_target:
                return "⚠️ **Unknown theme reference.** Use a known ETF like `SMH` or a known theme name."
            policy_profile = CommunicationAgent._load_policy_profile()
            if theme_target["etf"] and theme_target["etf"] not in policy_profile["priority_etfs"]:
                policy_profile["priority_etfs"].append(theme_target["etf"])
            if theme_target["theme_key"] and theme_target["theme_key"] not in policy_profile["preferred_themes"]:
                policy_profile["preferred_themes"].append(theme_target["theme_key"])
            if theme_target["theme_key"] in policy_profile["excluded_themes"]:
                policy_profile["excluded_themes"].remove(theme_target["theme_key"])
            CommunicationAgent._save_policy_profile(policy_profile)
            CommunicationAgent.clear_pending_follow_up(user_id)
            return f"✅ **Discovery focus updated.** I will keep **{theme_target['theme_name']}** in active focus."

        if cmd == "/exclude":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/exclude ETF_OR_THEME` or reply `exclude theme` to a discovery follow-up."
            pending_action = CommunicationAgent._load_pending_follow_up(user_id)
            theme_target = CommunicationAgent._resolve_theme_selection(" ".join(tokens[1:]), pending_action=pending_action)
            if not theme_target:
                return "⚠️ **Unknown theme reference.** Use a known ETF like `SMH` or a known theme name."
            policy_profile = CommunicationAgent._load_policy_profile()
            if theme_target["theme_key"] and theme_target["theme_key"] not in policy_profile["excluded_themes"]:
                policy_profile["excluded_themes"].append(theme_target["theme_key"])
            if theme_target["theme_key"] in policy_profile["preferred_themes"]:
                policy_profile["preferred_themes"].remove(theme_target["theme_key"])
            if theme_target["etf"] in policy_profile["priority_etfs"]:
                policy_profile["priority_etfs"].remove(theme_target["etf"])
            CommunicationAgent._save_policy_profile(policy_profile)
            CommunicationAgent.clear_pending_follow_up(user_id)
            return f"✅ **Theme excluded.** I will stop surfacing **{theme_target['theme_name']}** for now."

        if cmd == "/snooze":
            if len(tokens) < 2:
                pending_action = CommunicationAgent._load_pending_follow_up(user_id)
                if pending_action and pending_action.get("kind") == "discovery_theme_followup":
                    CommunicationAgent.suppress_discovery_idea(
                        symbol=str(pending_action.get("symbol", "")),
                        theme_key=str(pending_action.get("theme_key", "")),
                        days=14,
                    )
                    CommunicationAgent.clear_pending_follow_up(user_id)
                    return f"✅ **Idea snoozed.** I will suppress **{pending_action.get('symbol', 'this idea')}** for 14 days."
                return "⚠️ **Usage error:** Use `/snooze SYMBOL [days]` or reply `snooze idea` to a discovery follow-up."

            symbol = tokens[1].upper()
            days = 14
            if len(tokens) >= 3:
                try:
                    days = int(tokens[2])
                except ValueError:
                    return "⚠️ **Format error:** The snooze period must be an integer number of days."
            CommunicationAgent.suppress_discovery_idea(symbol=symbol, theme_key="", days=days)
            CommunicationAgent.clear_pending_follow_up(user_id)
            return f"✅ **Idea snoozed.** I will suppress **{symbol}** for `{days}` day(s)."

        if cmd == "/policy":
            if len(tokens) > 1 and tokens[1].lower() == "update":
                if len(tokens) < 3:
                    return "⚠️ **Usage error:** Use `/policy update [your guidance]`."
                new_policy_text = " ".join(tokens[2:])
                policy_profile = _build_policy_profile(new_policy_text)
                CommunicationAgent._save_policy_profile(policy_profile)
                return (
                    "✅ **Discovery policy updated.**\n\n"
                    f"🔍 **What I will optimize for:** `{policy_profile.get('preference_summary', policy_profile['policy_text'])}`\n"
                    f"🧭 **Risk posture:** `{policy_profile.get('risk_profile', 'balanced').replace('_', ' ')}`\n"
                    f"⏳ **Time horizon:** `{policy_profile.get('time_horizon', 'long_term').replace('_', ' ')}`\n"
                    f"🌍 **Markets:** `{', '.join(policy_profile.get('market_preferences', [])) or 'No explicit market bias stored'}`\n"
                    f"🏢 **Company preferences:** `{', '.join(policy_profile.get('company_preferences', [])) or 'None'}`\n"
                    f"🎯 **Priority ETFs:** `{', '.join(policy_profile.get('priority_etfs', [])) or 'None'}`\n"
                    f"🧭 **Preferred themes:** `{', '.join(policy_profile.get('preferred_themes', [])) or 'None'}`\n"
                    f"🚫 **Excluded themes:** `{', '.join(policy_profile.get('excluded_themes', [])) or 'None'}`\n"
                    f"🛡️ **Risk avoidances:** `{', '.join(policy_profile.get('risk_avoidances', [])) or 'None'}`"
                )

            return CommunicationAgent._format_policy_report()

        if cmd == "/discover":
            return (
                "🔍 **MyInvestmentBanker**: Starting a deep theme-led discovery run. "
                "I am gathering macro context, sector leadership, filings, and material news."
            )

        if cmd == "/update":
            return (
                "⚙️ **MyInvestmentBanker**: Starting a portfolio review. "
                "I am checking your holdings, risk context, and thesis coverage."
            )

        if not message_text.strip().startswith("/"):
            learning_result = CommunicationAgent.learn_preferences_from_message(user_id, message_text)
            if learning_result:
                return CommunicationAgent._summarize_preference_learning_result(learning_result)
            return (
                "I did not treat that as a stored action, but I can still work with plain language.\n\n"
                "If you are describing your investing preferences, just keep talking normally and I will synthesize them into your policy profile.\n"
                "If you want a stock reviewed, use `/analyze TICKER` or say something like `analyze NVDA`."
            )

        return "❓ **Unknown command.** Send `/help` to see list of options."

    @staticmethod
    def compile_single_stock_briefing(analysis_result: Dict[str, Any]) -> str:
        symbol = analysis_result.get("symbol", "UNKNOWN")
        theme = analysis_result.get("theme", {})
        research = analysis_result.get("research", {})
        candidate_review = analysis_result.get("candidate_review", {})
        risk_audit = analysis_result.get("risk_audit", {})
        policy_profile = analysis_result.get("policy_profile", get_default_policy_profile())
        recommendation = (risk_audit.get("recommendations") or [None])[0]

        market_data = research.get("market_data", {})
        material_news = research.get("material_news", [])
        relevant_filings = research.get("recent_filings", [])
        ownership_intel = research.get("ownership_intel", {})
        street_consensus = research.get("street_consensus", {})
        company_name = market_data.get("name", symbol)

        if llm_available():
            system_instruction = (
                "You are the Senior Private Wealth Manager & Investment Banker of MyInvestmentBanker.\n"
                "You are writing a bespoke, highly customized investment briefing directly to a sophisticated private client.\n"
                "Your tone is elite, conversational, analytical, and authoritative.\n"
                "Write a highly concise, punchy memo. The client reads this on a mobile device; keep the entire response under 300 words total.\n"
                "Avoid verbose filler, generic boilerplate, or long intros. Use short, dense paragraphs to deliver high analytical value.\n"
                "Interpret all evidence in context (metrics, ownership, filings) rather than listing raw data, filtering out low-value content farm noise (e.g. Benzinga/SeekingAlpha).\n"
                "Format in clean, Telegram-compatible Markdown with bold terms."
            )
            prompt = (
                f"Please write a highly concise, professional private investment memo for **{symbol}** ({company_name}) based on the following evidence.\n\n"
                f"=== 1. Active Client Discovery Policy ===\n{policy_profile}\n\n"
                f"=== 2. Market Pricing, Valuations & Technicals ===\n{market_data}\n\n"
                f"=== 3. SEC Filings ===\n{relevant_filings}\n\n"
                f"=== 4. Material Corporate News ===\n{material_news}\n\n"
                f"=== 5. Ownership Intel ===\n{ownership_intel}\n\n"
                f"=== 6. Street Analyst Consensus ===\n{street_consensus}\n\n"
                f"=== 7. CFA Analyst Verdict & Theme Context ===\n"
                f"Theme: {theme}\n"
                f"Candidate alignment: {candidate_review}\n"
                f"Risk officer audit recommendation: {recommendation}\n\n"
                f"Structure your response exactly as follows, keeping each section extremely brief (max 2-3 sentences per section, total word count under 300 words):\n"
                f"1. **Bespoke Executive Recommendation** (Refined, actionable recommendation in natural language)\n"
                f"2. **The Fundamental Investment Thesis** (Moat, business model, and sector context)\n"
                f"3. **Material Catalysts & Evidence Synthesis** (Synthesize recent SEC filings, owner shifts, and material news fluidly—do not list raw headlines)\n"
                f"4. **Core Valuation Hurdles & Risk Factors** (Analytical review of P/E, analyst targets, and key invalidators)\n\n"
                f"Ensure the memo is cohesive, deeply analytical, and highly concise."
            )
            narrative = generate_llm_response(prompt, system_instruction)
            if narrative:
                action_suffix = ""
                if recommendation:
                    action_hint = (
                        f"`/add {symbol} PRICE QTY`"
                        if recommendation.get("recommendation_type") == "new_position"
                        else f"`/thesis {symbol} [updated thesis]`"
                    )
                    action_suffix = f"\n\n👉 **Suggested Manual Step**: {action_hint}"
                return f"🔬 **Single-Stock Review: {symbol}**\n\n{narrative}{action_suffix}"

        # FALLBACK: Deterministic reporting block for robust offline/mock execution
        lines = [
            f"🔬 **Single-Stock Review: {symbol}**",
            "",
            f"**Plain-language take:** {recommendation.get('investment_hypothesis') if recommendation else candidate_review.get('analyst_verdict', 'I was able to gather only a thin first-pass evidence set.')}",
            "",
            "🧭 **How It Fits Your Current Preferences**",
            f"• {CommunicationAgent._build_policy_fit_interpretation(policy_profile, theme, candidate_review)}",
            "",
            "🧾 **What The Evidence Says Right Now**",
        ]

        if market_data.get("current_price") is not None:
            price_sentence = f"{company_name} trades around `{market_data.get('current_price')}`"
            if market_data.get("day_change_pct") is not None:
                price_sentence += f", with a `{market_data.get('day_change_pct')}%` move today"
            if market_data.get("five_day_change_pct") is not None:
                price_sentence += f" and `{market_data.get('five_day_change_pct')}%` over the last five trading days"
            price_sentence += "."
            lines.append(f"• {price_sentence}")

        metric_notes = []
        if market_data.get("market_cap"):
            metric_notes.append(
                f"Market cap is roughly `{CommunicationAgent._format_market_cap(market_data.get('market_cap'))}`, which means this is not an early, unproven micro-cap story."
            )
        pe_ratio = market_data.get("pe_ratio")
        if pe_ratio is not None:
            if pe_ratio >= 35:
                metric_notes.append(
                    f"Trailing P/E is `{pe_ratio}`, which means the market is already paying up for future growth."
                )
            elif pe_ratio <= 15:
                metric_notes.append(
                    f"Trailing P/E is `{pe_ratio}`, which means expectations may be more muted or the business may be seen as steadier."
                )
        beta = market_data.get("beta")
        if beta is not None:
            if beta >= 1.6:
                metric_notes.append(f"Beta is `{beta}`, which means the shares likely move more sharply than the broad market.")
            elif beta <= 0.9:
                metric_notes.append(f"Beta is `{beta}`, which suggests the stock may trade with less volatility than the market.")
        for note in metric_notes[:2]:
            lines.append(f"• {note}")

        if material_news:
            lines.append(
                f"• Recent material news item(s):"
            )
            for item in material_news[:2]:
                headline = (item.get("headline") or "").strip()
                url = item.get("url") or ""
                source = item.get("source") or "News"
                why = item.get("why_material") or ""
                
                headline_link = f"[{headline}]({url})" if url else headline
                news_detail = f"  - **{headline_link}** ({source})"
                if why:
                    news_detail += f" | *Materiality:* {why}"
                lines.append(news_detail)
        if relevant_filings:
            lines.append(
                f"• Recent SEC checkpoint(s):"
            )
            for filing in relevant_filings[:1]:
                form = filing.get("form") or "filing"
                date = filing.get("date") or "unknown date"
                url = filing.get("report_url") or ""
                desc = (filing.get("description") or "").strip()
                
                form_link = f"[{form}]({url})" if url else form
                filing_detail = f"  - **{form_link}** on {date}"
                if desc:
                    cropped_desc = desc[:80] + "..." if len(desc) > 80 else desc
                    filing_detail += f": {cropped_desc}"
                lines.append(filing_detail)
        if ownership_intel.get("summary"):
            lines.append(f"• Ownership view: {ownership_intel.get('summary')}")
        if street_consensus.get("summary"):
            lines.append(f"• Street view: {street_consensus.get('summary')}")
        if not market_data.get("current_price") and not material_news and not relevant_filings:
            lines.append("• Live evidence was thin in this pass, so treat this as a low-confidence baseline rather than a conviction call.")

        lines.extend(
            [
                "",
                "💡 **Why This Name, Specifically**",
                f"• {candidate_review.get('thesis_alignment') or candidate_review.get('why_this_company') or 'The company appears relevant, but the company-specific case is still thin.'}",
            ]
        )
        if recommendation and recommendation.get("why_now"):
            lines.append(f"• Timing view: {recommendation.get('why_now')}")
        elif candidate_review.get("company_catalysts"):
            lines.append(f"• Current catalysts: {'; '.join(candidate_review.get('company_catalysts', [])[:2])}.")

        risks = recommendation.get("key_risks") if recommendation else candidate_review.get("cautions", [])
        invalidators = recommendation.get("what_invalidates_it") if recommendation else theme.get("invalidators", [])
        lines.extend(["", "⚠️ **What Could Go Wrong**"])
        if risks:
            for risk in risks[:2]:
                lines.append(f"• {risk}")
        else:
            lines.append("• The evidence set is still incomplete, so confidence should stay measured.")

        if invalidators:
            lines.extend(["", "🛑 **What Would Change The View**"])
            for invalidator in invalidators[:1]:
                lines.append(f"• {invalidator}")

        lines.extend(
            [
                "",
                "🧠 **Bottom Line**",
                f"• {recommendation.get('confidence_note') if recommendation else candidate_review.get('confidence_note', 'First-pass review only.')}",
                "• Interpretation: use this as a research note and a preference check, not as a blind trading instruction.",
            ]
        )

        if recommendation:
            action_hint = (
                f"`/add {symbol} PRICE QTY`"
                if recommendation.get("recommendation_type") == "new_position"
                else f"`/thesis {symbol} [updated thesis]`"
            )
            lines.append(f"• Manual next step if you agree: {action_hint}")

        return "\n".join(lines)

    @staticmethod
    def compile_synthesis_report(
        portfolio_state: List[Dict[str, Any]],
        macro_data: Dict[str, Any],
        cfa_memos: List[Dict[str, Any]],
        risk_memos: Dict[str, Any],
    ) -> str:
        portfolio_change_lines = CommunicationAgent._build_portfolio_change_lines(portfolio_state)
        if not llm_available():
            lines = [
                "📈 **Portfolio Digest**",
                "",
                "📊 **Snapshot Change Summary**",
            ]
            lines.extend([f"• {line}" for line in portfolio_change_lines])
            lines.extend(
                [
                    "",
                    "In plain language:",
                    f"• {CommunicationAgent._macro_interpretation(macro_data)}",
                    f"• I reviewed `{len(portfolio_state)}` holding(s) in this cycle.",
                    "",
                ]
            )
            if cfa_memos:
                lines.append("🧾 **What Stood Out In Individual Names**")
                for memo in cfa_memos[:5]:
                    lines.append(
                        f"• **{memo.get('symbol', 'Unknown')}**: {memo.get('headline_verdict', memo.get('memo_text', 'No memo generated.'))}"
                    )
                lines.append("")
            if risk_memos:
                risk_summary = risk_memos.get("summary") or risk_memos.get("risk_memo") or "Risk memo unavailable."
                lines.append("🛡️ **Portfolio-Level Read**")
                lines.append(f"• {risk_summary}")
                lines.append("• Interpretation: use this as a structured check-in, not a trading instruction on its own.")
            return "\n".join(lines)

        system_instruction = (
            "You are the Lead Portfolio Synthesis Manager of MyInvestmentBanker.\n"
            "Write a highly concise, professional portfolio synthesis briefing.\n"
            "Keep the entire synthesis under 250 words total, suitable for a mobile screen.\n"
            "Use very short, punchy paragraphs and explain any cited metrics immediately without jargon.\n"
            "Avoid metric laundry lists and verbose preamble.\n"
            "Format in clean Telegram-compatible Markdown."
        )
        prompt = (
            f"Please synthesize the following data streams into a highly concise portfolio briefing.\n\n"
            f"=== 1. Active Portfolio State ===\n{portfolio_state}\n\n"
            f"=== 2. Deterministic Portfolio Snapshot Summary ===\n{portfolio_change_lines}\n\n"
            f"=== 3. Macroeconomic Indicators ===\n{macro_data}\n\n"
            f"=== 4. CFA Analyst Memos ===\n{cfa_memos}\n\n"
            f"=== 5. Portfolio Risk Review ===\n{risk_memos}\n\n"
            f"Structure the output exactly as follows, keeping each section to 2-3 dense sentences (total word count under 250 words):\n"
            f"1. **Executive Summary** (Plain-language overview)\n"
            f"2. **Key Portfolio Dynamics** (What changed and why)\n"
            f"3. **Material Risks & Thesis Drift** (Risks worth paying attention to)\n"
            f"4. **Actionable Recommendations** (Opportunities or manual next steps)\n\n"
            f"Keep it extremely concise and direct."
        )
        narrative = generate_llm_response(prompt, system_instruction)
        deterministic_header = ["📈 **Portfolio Digest**", "", "📊 **Snapshot Change Summary**"]
        deterministic_header.extend([f"• {line}" for line in portfolio_change_lines])
        deterministic_header.extend(["", "🧠 **Manager Interpretation**"])
        if narrative:
            deterministic_header.append(narrative)
        return "\n".join(deterministic_header)

    @staticmethod
    def compile_discovery_briefing(discovery_result: Dict[str, Any]) -> str:
        run_type = discovery_result.get("run_type", "deep")
        recommendations = discovery_result.get("recommendations", [])
        themes = discovery_result.get("themes", [])
        policy_profile = discovery_result.get("policy_profile", get_default_policy_profile())
        summary_text = discovery_result.get("summary_text", "")

        if not recommendations:
            if run_type == "sweep":
                return ""
            return (
                "🔎 **Weekly Discovery Review**\n\n"
                "Nothing this week looked compelling enough to turn into a recommendation.\n\n"
                f"In plain language: I looked through the themes tied to `{', '.join(policy_profile.get('priority_etfs', []))}` "
                "and the evidence was either mixed or too weak to justify a high-conviction idea.\n"
                f"🧠 **Run Summary:** {summary_text or 'Signals were too thin to justify action.'}"
            )

        lines = [
            "🔎 **Theme-Led Discovery Briefing**",
            "",
        ]
        if themes:
            lines.append("🌐 **What Changed In The Market's Attention**")
            for theme in themes[:2]:
                lines.append(
                    f"• **{theme.get('theme_name', theme.get('theme_key', 'Theme'))}**: "
                    f"{theme.get('why_now', 'No explanation available.')}"
                )
            lines.append("")

        lines.append("💡 **Ideas Worth A Closer Look**")
        for idx, recommendation in enumerate(recommendations[:3], start=1):
            evidence_sentences = []
            evidence = recommendation.get("evidence") or {}
            if evidence.get("catalysts"):
                evidence_sentences.append(
                    "Current catalysts include " + "; ".join(str(item) for item in evidence["catalysts"][:2]) + "."
                )
            if evidence.get("material_news") and isinstance(evidence.get("material_news"), list):
                news_items = evidence["material_news"]
                news_headlines = []
                for item in news_items[:2]:
                    h = (item.get("headline") or "").strip()
                    u = item.get("url") or ""
                    if h:
                        news_headlines.append(f"[{h}]({u})" if u else h)
                if news_headlines:
                    evidence_sentences.append("Recent news: " + "; ".join(news_headlines) + ".")
            elif evidence.get("material_news"):
                evidence_sentences.append("Recent news is adding support to the thesis.")

            if evidence.get("relevant_filings") and isinstance(evidence.get("relevant_filings"), list):
                filing_items = evidence["relevant_filings"]
                filing_desc = []
                for filing in filing_items[:1]:
                    form = filing.get("form") or "filing"
                    url = filing.get("report_url") or ""
                    date = filing.get("date") or ""
                    form_str = f"[{form}]({url})" if url else form
                    filing_desc.append(f"{form_str} on {date}")
                if filing_desc:
                    evidence_sentences.append("SEC filings: " + "; ".join(filing_desc) + ".")
            elif evidence.get("relevant_filings"):
                evidence_sentences.append("A recent filing gives the story a fresh corporate checkpoint.")

            symbol = recommendation.get('symbol', 'Unknown')
            lines.append(f"{idx}. **{symbol}**")
            lines.append(
                f"   • **Thesis & Timing:** {recommendation.get('investment_hypothesis', 'No company rationale available.')} *Why now:* {recommendation.get('why_now', 'No timing rationale available.')}"
            )
            if evidence_sentences:
                lines.append(f"   • **Evidence:** {' '.join(evidence_sentences)}")
            
            risks = recommendation.get('key_risks', [])
            invalidators = recommendation.get('what_invalidates_it', [])
            risks_str = f"Risks: {', '.join(risks)}" if risks else "Risks: early/incomplete."
            invalidators_str = f"Invalidators: {', '.join(invalidators)}" if invalidators else "Invalidators: catalysts fade."
            lines.append(f"   • **Risk Profile:** {risks_str} | {invalidators_str}")
            lines.append(
                f"   • **Action:** `/add {symbol} PRICE QTY`"
            )
            lines.append("")

        if summary_text:
            lines.append("🧠 **Run Summary**")
            lines.append(f"• {summary_text}")

        return "\n".join(lines)
