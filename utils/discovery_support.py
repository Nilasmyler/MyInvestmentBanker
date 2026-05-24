import copy
import re
from typing import Any, Dict, List, Optional, TypedDict


class PolicyProfile(TypedDict, total=False):
    policy_text: str
    preference_summary: str
    preferred_themes: List[str]
    excluded_themes: List[str]
    style_bias: List[str]
    risk_avoidances: List[str]
    priority_etfs: List[str]
    risk_profile: str
    time_horizon: str
    market_preferences: List[str]
    company_preferences: List[str]


class MaterialNewsItem(TypedDict, total=False):
    symbol: str
    headline: str
    summary: str
    url: str
    published_at: str
    source: str
    event_type: str
    why_material: str


class ThemeHypothesis(TypedDict, total=False):
    theme_key: str
    theme_name: str
    why_now: str
    mapped_etfs: List[str]
    evidence_items: List[Dict[str, Any]]
    confidence_level: str
    invalidators: List[str]
    candidate_symbols: List[str]
    trigger_sources: List[str]


class CandidateExpression(TypedDict, total=False):
    symbol: str
    source_etf: str
    is_existing_position: bool
    why_this_company: str
    company_catalysts: List[str]
    relevant_filings: List[Dict[str, Any]]
    material_news: List[MaterialNewsItem]
    price_context: Dict[str, Any]
    data_gaps: List[str]
    theme_key: str
    theme_name: str


class DiscoveryRecommendation(TypedDict, total=False):
    symbol: str
    recommendation_type: str
    theme_name: str
    investment_hypothesis: str
    why_now: str
    key_risks: List[str]
    what_invalidates_it: List[str]
    confidence_note: str
    source_etf: str


DEFAULT_THEME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "SMH": {
        "theme_key": "semiconductors",
        "theme_name": "Semiconductor Cycle / AI Infrastructure",
        "keywords": ["semiconductor", "chip", "ai infrastructure", "gpu", "datacenter", "foundry"],
        "invalidators": [
            "AI and datacenter capex spending cools materially",
            "Inventory correction deepens across the semiconductor supply chain",
        ],
    },
    "IGV": {
        "theme_key": "software",
        "theme_name": "Enterprise Software / Digital Productivity",
        "keywords": ["software", "saas", "cloud", "workflow", "enterprise software", "automation"],
        "invalidators": [
            "Enterprise spending guidance weakens broadly",
            "Customers defer large multi-year software contracts",
        ],
    },
    "HACK": {
        "theme_key": "cybersecurity",
        "theme_name": "Cybersecurity Spend",
        "keywords": ["cybersecurity", "security", "identity", "endpoint", "breach", "zero trust"],
        "invalidators": [
            "Security budgets consolidate materially",
            "Large platform competition compresses independent vendor growth",
        ],
    },
    "XBI": {
        "theme_key": "biotech",
        "theme_name": "Biotech Innovation / Drug Pipeline",
        "keywords": ["biotech", "drug", "trial", "fda", "approval", "therapy"],
        "invalidators": [
            "Trial failures or regulatory setbacks spread across leaders",
            "Funding conditions for the sector worsen materially",
        ],
    },
    "XLV": {
        "theme_key": "healthcare",
        "theme_name": "Healthcare Defensives / Medical Innovation",
        "keywords": ["healthcare", "medical", "hospital", "device", "pharma"],
        "invalidators": [
            "Regulatory pricing pressure rises materially",
            "Reimbursement changes impair earnings visibility",
        ],
    },
    "XLI": {
        "theme_key": "industrials",
        "theme_name": "Industrial Automation / Reindustrialization",
        "keywords": ["industrial", "automation", "factory", "logistics", "reindustrialization"],
        "invalidators": [
            "Manufacturing activity rolls over sharply",
            "Order books weaken across capital goods leaders",
        ],
    },
    "XLE": {
        "theme_key": "energy",
        "theme_name": "Energy Cash Flow / Commodity Tightness",
        "keywords": ["energy", "oil", "gas", "lng", "commodity"],
        "invalidators": [
            "Commodity prices fall into oversupply conditions",
            "Political or regulatory changes impair sector economics",
        ],
    },
    "XLF": {
        "theme_key": "financials",
        "theme_name": "Financials / Credit and Rates",
        "keywords": ["bank", "insurance", "credit", "payments", "lending"],
        "invalidators": [
            "Credit losses accelerate",
            "Funding pressure or regulation worsens sector profitability",
        ],
    },
}

DEFAULT_POLICY_PROFILE: PolicyProfile = {
    "policy_text": (
        "Look for explainable, thesis-led investment opportunities in market-leading companies "
        "when a sector or theme is becoming more relevant due to macro shifts, filings, material "
        "news, technology changes, or market action."
    ),
    "preference_summary": (
        "Look for explainable, thesis-led investment opportunities in market-leading companies "
        "with a balanced risk posture and a long-term horizon."
    ),
    "preferred_themes": [],
    "excluded_themes": [],
    "style_bias": ["compounders", "quality leaders"],
    "risk_avoidances": ["unclear thesis", "weak evidence"],
    "priority_etfs": ["SMH", "IGV"],
    "risk_profile": "balanced",
    "time_horizon": "long_term",
    "market_preferences": [],
    "company_preferences": ["quality leaders", "understandable businesses"],
}

THEME_KEYWORD_MAP = {
    "semiconductor": "SMH",
    "chip": "SMH",
    "gpu": "SMH",
    "software": "IGV",
    "saas": "IGV",
    "cloud": "IGV",
    "cyber": "HACK",
    "cybersecurity": "HACK",
    "security": "HACK",
    "biotech": "XBI",
    "drug": "XBI",
    "healthcare": "XLV",
    "medical": "XLV",
    "industrial": "XLI",
    "automation": "XLI",
    "energy": "XLE",
    "oil": "XLE",
    "gas": "XLE",
    "financial": "XLF",
    "bank": "XLF",
    "payments": "XLF",
}

STYLE_KEYWORDS = {
    "compounders": ["compounder", "long-term compounder", "high quality", "durable moat"],
    "defensives": ["defensive", "resilient", "low volatility"],
    "secular growth": ["secular growth", "structural growth", "long runway"],
    "turnarounds": ["turnaround", "special situation", "recovery"],
}

RISK_KEYWORDS = {
    "high leverage": ["high debt", "leverage", "balance sheet risk"],
    "commodity dependence": ["commodity", "oil exposure", "gas exposure"],
    "binary biotech risk": ["binary", "trial risk", "single asset biotech"],
    "crowded valuations": ["expensive", "stretched valuation", "overvalued"],
}

RISK_PROFILE_KEYWORDS = {
    "conservative": [
        "low risk",
        "lower risk",
        "conservative",
        "defensive",
        "capital preservation",
        "sleep at night",
        "less volatile",
        "safer",
    ],
    "balanced": ["balanced", "moderate risk", "middle ground"],
    "aggressive": ["high risk", "higher risk", "aggressive", "speculative", "volatile", "risk-on"],
}

TIME_HORIZON_KEYWORDS = {
    "short_term": ["short term", "near term", "next few months", "trade", "trading", "swing trade"],
    "medium_term": ["medium term", "1-3 years", "one to three years", "2-3 years", "two to three years"],
    "long_term": [
        "long term",
        "long-term",
        "hold for years",
        "patient",
        "five year",
        "5 year",
        "ten year",
        "10 year",
        "compound over time",
    ],
}

MARKET_KEYWORDS = {
    "US": ["us market", "u.s. market", "us stocks", "american companies", "nasdaq", "nyse", "us only"],
    "Europe": ["europe", "european", "eu", "uk", "british", "german", "french", "danish", "nordic", "nordics"],
    "Japan": ["japan", "japanese"],
    "Emerging Markets": ["emerging markets", "em", "india", "brazil", "latin america", "china", "southeast asia"],
    "Global": ["global", "worldwide", "international", "across markets"],
}

COMPANY_PREFERENCE_KEYWORDS = {
    "quality leaders": ["quality leader", "market leader", "category leader", "best in class", "best-in-class"],
    "understandable businesses": ["understandable business", "easy to understand", "simple business", "clear business model"],
    "cash-generative businesses": ["cash generative", "cash-generative", "strong cash flow", "free cash flow", "profitable"],
    "compounders": ["compounder", "compounders", "compounding"],
    "founder-led companies": ["founder led", "founder-led"],
    "large-cap stability": ["large cap", "large-cap", "mega cap", "mega-cap", "established company", "established companies"],
    "small-cap upside": ["small cap", "small-cap", "mid cap", "mid-cap", "smaller companies", "smaller company"],
    "asset-light models": ["asset light", "asset-light", "software-like", "high gross margin"],
}

PREFERENCE_SIGNAL_CUES = [
    "i prefer",
    "i like",
    "i want",
    "i'm looking for",
    "i am looking for",
    "looking for",
    "i avoid",
    "i don't want",
    "i do not want",
    "not interested in",
    "i'm comfortable",
    "i am comfortable",
    "i'm patient",
    "i am patient",
    "for me",
    "my horizon",
    "my risk",
    "focus on",
    "stay away from",
]


def get_theme_registry() -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(DEFAULT_THEME_REGISTRY)


def get_default_policy_profile() -> PolicyProfile:
    return copy.deepcopy(DEFAULT_POLICY_PROFILE)


def _extract_matching_items(text: str, lookup: Dict[str, List[str]]) -> List[str]:
    lower_text = text.lower()
    matches = []
    for label, keywords in lookup.items():
        if any(keyword in lower_text for keyword in keywords):
            matches.append(label)
    return matches


def _extract_first_matching_item(text: str, lookup: Dict[str, List[str]]) -> Optional[str]:
    lower_text = text.lower()
    for label, keywords in lookup.items():
        if any(keyword in lower_text for keyword in keywords):
            return label
    return None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        clean = str(item).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _humanize_theme_key(theme_key: str) -> str:
    registry = get_theme_registry()
    for item in registry.values():
        if item["theme_key"] == theme_key:
            return item["theme_name"]
    return theme_key.replace("_", " ").title()


def infer_priority_etfs(text: str) -> List[str]:
    lower_text = text.lower()
    found_etfs = []
    for keyword, etf in THEME_KEYWORD_MAP.items():
        if keyword in lower_text and etf not in found_etfs:
            found_etfs.append(etf)
    return found_etfs


def extract_preference_update_from_text(policy_text: str) -> Dict[str, Any]:
    if not policy_text:
        return {}

    lower_text = policy_text.lower()
    signal_cue_present = any(cue in lower_text for cue in PREFERENCE_SIGNAL_CUES)

    excluded_matches = re.findall(
        r"(avoid|exclude|no|don't want|do not want|not interested in|stay away from)\s+([a-zA-Z\s/-]+)",
        lower_text,
    )
    excluded_themes = []
    for _, fragment in excluded_matches:
        for keyword, etf in THEME_KEYWORD_MAP.items():
            if keyword in fragment and DEFAULT_THEME_REGISTRY[etf]["theme_key"] not in excluded_themes:
                excluded_themes.append(DEFAULT_THEME_REGISTRY[etf]["theme_key"])

    priority_etfs = infer_priority_etfs(policy_text)
    preferred_themes = [
        DEFAULT_THEME_REGISTRY[etf]["theme_key"]
        for etf in priority_etfs
        if etf in DEFAULT_THEME_REGISTRY and DEFAULT_THEME_REGISTRY[etf]["theme_key"] not in excluded_themes
    ]

    risk_profile = _extract_first_matching_item(policy_text, RISK_PROFILE_KEYWORDS)
    time_horizon = _extract_first_matching_item(policy_text, TIME_HORIZON_KEYWORDS)
    market_preferences = _extract_matching_items(policy_text, MARKET_KEYWORDS)
    company_preferences = _extract_matching_items(policy_text, COMPANY_PREFERENCE_KEYWORDS)
    style_bias = _extract_matching_items(policy_text, STYLE_KEYWORDS)
    risk_avoidances = _extract_matching_items(policy_text, RISK_KEYWORDS)

    signal_count = sum(
        1
        for value in [
            bool(preferred_themes),
            bool(excluded_themes),
            bool(risk_profile),
            bool(time_horizon),
            bool(market_preferences),
            bool(company_preferences),
            bool(style_bias),
            bool(risk_avoidances),
        ]
        if value
    )

    if not signal_cue_present and signal_count < 2:
        return {}

    update: Dict[str, Any] = {
        "preferred_themes": preferred_themes,
        "excluded_themes": excluded_themes,
        "style_bias": style_bias,
        "risk_avoidances": risk_avoidances,
        "priority_etfs": priority_etfs,
        "market_preferences": market_preferences,
        "company_preferences": company_preferences,
    }
    if risk_profile:
        update["risk_profile"] = risk_profile
    if time_horizon:
        update["time_horizon"] = time_horizon
    return update


def build_preference_summary(policy_profile: Optional[Dict[str, Any]]) -> str:
    default_profile = get_default_policy_profile()
    raw_profile = policy_profile if isinstance(policy_profile, dict) else {}
    profile: Dict[str, Any] = copy.deepcopy(default_profile)

    for key in [
        "preferred_themes",
        "excluded_themes",
        "style_bias",
        "risk_avoidances",
        "priority_etfs",
        "market_preferences",
        "company_preferences",
    ]:
        value = raw_profile.get(key)
        if isinstance(value, list):
            profile[key] = _dedupe_preserve_order([str(item).strip() for item in value if str(item).strip()])

    for key in ["risk_profile", "time_horizon"]:
        value = raw_profile.get(key)
        if value:
            profile[key] = str(value).strip().lower().replace(" ", "_")

    parts = ["Look for explainable, thesis-led investment opportunities"]
    if profile.get("company_preferences"):
        parts.append("favoring " + " and ".join(profile["company_preferences"][:2]))

    risk_profile = profile.get("risk_profile", "")
    if risk_profile:
        parts.append(f"with a {risk_profile.replace('_', ' ')} risk posture")

    time_horizon = profile.get("time_horizon", "")
    if time_horizon:
        horizon_text = time_horizon.replace("_", " ")
        if horizon_text == "long term":
            parts.append("over a long-term horizon")
        else:
            parts.append(f"over a {horizon_text} horizon")

    if profile.get("market_preferences"):
        parts.append(f"mainly in {', '.join(profile['market_preferences'][:2])}")

    if profile.get("preferred_themes"):
        themed = [_humanize_theme_key(theme_key) for theme_key in profile["preferred_themes"][:2]]
        parts.append("while leaning toward " + " and ".join(themed))

    avoid_labels = []
    if profile.get("excluded_themes"):
        avoid_labels.extend(_humanize_theme_key(theme_key) for theme_key in profile["excluded_themes"][:2])
    if profile.get("risk_avoidances"):
        avoid_labels.extend(profile["risk_avoidances"][:2])
    if avoid_labels:
        parts.append("and avoiding " + ", ".join(_dedupe_preserve_order(avoid_labels[:3])))

    return " ".join(parts).strip() + "."


def merge_policy_profile(base_profile: Optional[Any], update: Optional[Dict[str, Any]]) -> PolicyProfile:
    merged = normalize_policy_profile(base_profile)
    if not update:
        merged["preference_summary"] = build_preference_summary(merged)
        return merged

    for key in [
        "preferred_themes",
        "excluded_themes",
        "style_bias",
        "risk_avoidances",
        "priority_etfs",
        "market_preferences",
        "company_preferences",
    ]:
        current_items = list(merged.get(key, []))
        update_items = update.get(key, [])
        if isinstance(update_items, list):
            merged[key] = _dedupe_preserve_order(current_items + [str(item).strip() for item in update_items if str(item).strip()])

    for key in ["risk_profile", "time_horizon"]:
        value = update.get(key)
        if value:
            merged[key] = str(value).strip().lower().replace(" ", "_")

    preferred = list(merged.get("preferred_themes", []))
    excluded = set(merged.get("excluded_themes", []))
    merged["preferred_themes"] = [item for item in preferred if item not in excluded]

    registry = get_theme_registry()
    filtered_etfs = [
        etf
        for etf in merged.get("priority_etfs", [])
        if registry.get(etf, {}).get("theme_key") not in excluded
    ]
    merged["priority_etfs"] = filtered_etfs or [
        etf
        for etf in get_default_policy_profile()["priority_etfs"]
        if registry.get(etf, {}).get("theme_key") not in excluded
    ]

    merged["preference_summary"] = build_preference_summary(merged)
    if update.get("policy_text"):
        merged["policy_text"] = str(update["policy_text"]).strip()
    else:
        merged["policy_text"] = merged["preference_summary"]

    return normalize_policy_profile(merged)


def parse_policy_text_fallback(policy_text: str) -> PolicyProfile:
    if not policy_text:
        return get_default_policy_profile()

    parsed = extract_preference_update_from_text(policy_text)
    parsed["policy_text"] = policy_text.strip()
    merged = merge_policy_profile(get_default_policy_profile(), parsed)
    merged["policy_text"] = policy_text.strip()
    merged["preference_summary"] = build_preference_summary(merged)
    return normalize_policy_profile(merged)


def normalize_policy_profile(raw_profile: Optional[Any], legacy_focus: Optional[List[str]] = None) -> PolicyProfile:
    default_profile = get_default_policy_profile()
    if not raw_profile:
        if legacy_focus:
            default_profile["priority_etfs"] = legacy_focus
        return default_profile

    if isinstance(raw_profile, str):
        return parse_policy_text_fallback(raw_profile)

    if not isinstance(raw_profile, dict):
        return default_profile

    normalized = get_default_policy_profile()
    normalized["policy_text"] = str(raw_profile.get("policy_text") or default_profile["policy_text"]).strip()
    normalized["preference_summary"] = str(
        raw_profile.get("preference_summary") or default_profile["preference_summary"]
    ).strip()

    for key in [
        "preferred_themes",
        "excluded_themes",
        "style_bias",
        "risk_avoidances",
        "priority_etfs",
        "market_preferences",
        "company_preferences",
    ]:
        value = raw_profile.get(key)
        if isinstance(value, list):
            normalized[key] = _dedupe_preserve_order([str(item).strip() for item in value if str(item).strip()])

    for key in ["risk_profile", "time_horizon"]:
        value = raw_profile.get(key)
        if value:
            normalized[key] = str(value).strip().lower().replace(" ", "_")

    if not normalized["priority_etfs"]:
        legacy = legacy_focus or raw_profile.get("matching_etfs") or raw_profile.get("active_opportunity_focus")
        if isinstance(legacy, list) and legacy:
            normalized["priority_etfs"] = [str(item).strip().upper() for item in legacy if str(item).strip()]

    if not normalized["priority_etfs"]:
        derived = infer_priority_etfs(normalized["policy_text"])
        normalized["priority_etfs"] = derived or default_profile["priority_etfs"]

    if not normalized["preferred_themes"]:
        normalized["preferred_themes"] = [
            DEFAULT_THEME_REGISTRY[etf]["theme_key"]
            for etf in normalized["priority_etfs"]
            if etf in DEFAULT_THEME_REGISTRY
        ]

    excluded = set(normalized.get("excluded_themes", []))
    normalized["preferred_themes"] = [item for item in normalized["preferred_themes"] if item not in excluded]
    normalized["priority_etfs"] = [
        etf
        for etf in normalized["priority_etfs"]
        if DEFAULT_THEME_REGISTRY.get(etf, {}).get("theme_key") not in excluded
    ] or [
        etf
        for etf in default_profile["priority_etfs"]
        if DEFAULT_THEME_REGISTRY.get(etf, {}).get("theme_key") not in excluded
    ]

    normalized["preference_summary"] = build_preference_summary({**default_profile, **normalized})

    return normalized


def get_theme_records_for_etfs(etfs: List[str]) -> List[Dict[str, Any]]:
    registry = get_theme_registry()
    records = []
    for etf in etfs:
        etf_clean = etf.upper().strip()
        if etf_clean in registry:
            record = registry[etf_clean]
            record["etf"] = etf_clean
            records.append(record)
    return records
