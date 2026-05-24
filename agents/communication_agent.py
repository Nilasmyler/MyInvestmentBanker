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
    fetch_portfolio,
    get_user_preference,
    save_investment_thesis,
    save_user_preference,
    update_portfolio_holding,
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

        if not matching_etfs:
            matching_etfs = list(policy_profile.get("priority_etfs", []))[:1]

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
            f"This review looks most connected to **{theme.get('theme_name', 'this theme')}**.\n"
            "Reply with one of these predefined actions:\n"
            "• `focus theme` — keep this theme in active discovery focus\n"
            "• `exclude theme` — stop surfacing this theme\n"
            "• `snooze idea` — suppress this exact idea for 14 days\n"
            "• `skip` — clear this follow-up"
        )

    @staticmethod
    def generate_single_stock_analysis(user_id: str, symbol: str, user_context: str = "") -> str:
        from agents.cfa_agent import CFAAgent
        from agents.risk_agent import RiskAgent
        from agents.scout_agent import ScoutAgent

        symbol = symbol.upper().strip()
        learning_result = CommunicationAgent.learn_preferences_from_message(user_id, user_context) if user_context else None
        policy_profile = CommunicationAgent._load_policy_profile()
        portfolio = fetch_portfolio()
        portfolio_symbols = {holding["symbol"].upper().strip() for holding in portfolio}

        research = ScoutAgent.collect_symbol_research(symbol)
        theme = CommunicationAgent._infer_theme_from_symbol_research(symbol, research, policy_profile)
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
            f"The strongest thread in this run was **{theme_name}**, with **{symbol}** as the clearest expression.\n"
            "Reply with one of these predefined actions:\n"
            "• `focus theme` — keep this theme in active discovery focus\n"
            "• `exclude theme` — stop surfacing this theme\n"
            "• `snooze idea` — suppress this exact idea for 14 days\n"
            "• `skip` — clear this follow-up"
        )

    @staticmethod
    def prepare_portfolio_follow_up(user_id: str, portfolio_state: Optional[List[Dict[str, Any]]] = None) -> str:
        holdings = portfolio_state if portfolio_state is not None else fetch_portfolio()
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
            f"I still do not have your investment thesis for **{missing_thesis_symbol}**.\n"
            "Reply with one of these predefined actions:\n"
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
            holdings = fetch_portfolio()
            if not holdings:
                return "📂 **Your portfolio is currently empty.** Add holdings with `/add TICKER PRICE QTY`."

            report = "💼 **Active Portfolio Holdings**\n\n"
            total_cost = 0.0
            for idx, holding in enumerate(holdings):
                ticker = holding["symbol"]
                qty = float(holding["quantity"])
                price = float(holding["cost_basis"])
                total_cost += qty * price
                thesis = fetch_investment_thesis(ticker)
                thesis_flag = "I have your thesis on file." if thesis else "I still need your thesis for this position."
                report += (
                    f"{idx + 1}. **{ticker}**\n"
                    f"   • You currently have `{qty:.2f}` shares logged at `{price:.2f}` USD.\n"
                    f"   • In plain language: this is the cost basis I will use when I discuss this holding.\n"
                    f"   • Thesis status: {thesis_flag}\n\n"
                )
            report += f"📊 **Total cost basis logged:** `{total_cost:.2f}` USD"
            return report

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
                return (
                    f"✅ **Holding updated.** I now have **{ticker}** logged at `{qty:.2f}` shares and "
                    f"`{price:.2f}` USD as the working cost basis."
                )
            return "❌ **Database write failed.** Check logs."

        if cmd == "/remove":
            if len(tokens) < 2:
                return "⚠️ **Usage error:** Use `/remove TICKER` (for example `/remove MSFT`)."

            ticker = tokens[1].upper()
            success = update_portfolio_holding(ticker, 0, 0)
            if success:
                return f"✅ **Holding removed.** **{ticker}** is no longer in your active portfolio."
            return "❌ **Database write failed.** Check logs."

        if cmd == "/thesis":
            if len(tokens) < 3:
                return "⚠️ **Usage error:** Use `/thesis TICKER [your thesis text]`."

            ticker = tokens[1].upper()
            thesis_text = " ".join(tokens[2:])
            holdings = fetch_portfolio()
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
        company_name = market_data.get("name", symbol)

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
                f"• I captured `{len(material_news)}` recent material news item(s), which suggests the story is still active rather than stale."
            )
        if relevant_filings:
            latest_filing = relevant_filings[0]
            lines.append(
                f"• The latest SEC checkpoint is a `{latest_filing.get('form', 'filing')}` from `{latest_filing.get('date', 'unknown date')}`, which gives a fresh corporate anchor for the thesis."
            )
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
            for risk in risks[:3]:
                lines.append(f"• {risk}")
        else:
            lines.append("• The evidence set is still incomplete, so confidence should stay measured.")

        if invalidators:
            lines.extend(["", "🛑 **What Would Change The View**"])
            for invalidator in invalidators[:2]:
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
        if not llm_available():
            lines = [
                "📈 **Portfolio Digest**",
                "",
                "In plain language:",
                f"• {CommunicationAgent._macro_interpretation(macro_data)}",
                f"• I reviewed `{len(portfolio_state)}` holding(s) in this cycle.",
                "",
            ]
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
            "Write in plain language first. Be specific but understandable.\n"
            "If you mention any metric or valuation datapoint, immediately explain what it means.\n"
            "Avoid metric laundry lists and avoid unexplained jargon.\n"
            "Format in clean Telegram-compatible Markdown."
        )
        prompt = (
            f"Please synthesize the following data streams into a professional personal investment briefing.\n\n"
            f"=== 1. Active Portfolio State ===\n{portfolio_state}\n\n"
            f"=== 2. Macroeconomic Indicators ===\n{macro_data}\n\n"
            f"=== 3. CFA Analyst Memos ===\n{cfa_memos}\n\n"
            f"=== 4. Portfolio Risk Review ===\n{risk_memos}\n\n"
            f"Structure the output as:\n"
            f"1. Plain-language executive summary\n"
            f"2. What changed and why it matters\n"
            f"3. Risks worth paying attention to\n"
            f"4. Opportunities or follow-up items\n"
            f"Keep it concise and explain the meaning of any evidence you cite."
        )
        return generate_llm_response(prompt, system_instruction)

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
            evidence = recommendation.get("evidence", {})
            if evidence.get("catalysts"):
                evidence_sentences.append(
                    "Current catalysts include " + "; ".join(str(item) for item in evidence["catalysts"][:2]) + "."
                )
            if evidence.get("material_news"):
                evidence_sentences.append("Recent news is adding support to the thesis.")
            if evidence.get("relevant_filings"):
                evidence_sentences.append("A recent filing gives the story a fresh corporate checkpoint.")

            lines.append(f"{idx}. **{recommendation.get('symbol', 'Unknown')}**")
            lines.append(
                f"   • **Plain-language view:** {recommendation.get('investment_hypothesis', 'No company rationale available.')}"
            )
            lines.append(
                f"   • **Why now:** {recommendation.get('why_now', 'No timing rationale available.')}"
            )
            if evidence_sentences:
                lines.append(f"   • **Evidence:** {' '.join(evidence_sentences)}")
            lines.append(
                f"   • **What could go wrong:** {', '.join(recommendation.get('key_risks', [])) or 'The evidence is still early or incomplete.'}"
            )
            lines.append(
                f"   • **What would change my mind:** {', '.join(recommendation.get('what_invalidates_it', [])) or 'The thesis would weaken if the catalysts fade.'}"
            )
            lines.append(
                f"   • **How to act manually if you agree:** `/add {recommendation.get('symbol', 'TICKER')} PRICE QTY`"
            )
            lines.append("")

        if summary_text:
            lines.append("🧠 **Run Summary**")
            lines.append(f"• {summary_text}")

        return "\n".join(lines)
