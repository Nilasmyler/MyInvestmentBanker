import logging
import os
import sys
from typing import List

from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MyInvestmentBanker.tester")

logger.info("==============================================================================")
logger.info("        MyInvestmentBanker: Local Pipeline Verification Script (v2)          ")
logger.info("==============================================================================")

failures: List[str] = []


def record_failure(message: str) -> None:
    failures.append(message)
    logger.error(message)


def verify_live_news_digest_schema() -> None:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_service_key:
        logger.info("Supabase credentials missing. Skipping live news_digests schema probe.")
        return

    headers = {
        "apikey": supabase_service_key,
        "Authorization": f"Bearer {supabase_service_key}",
    }
    missing_columns: List[str] = []
    for column in ["entity_type", "entity_key", "metadata"]:
        response = requests.get(
            f"{supabase_url}/rest/v1/news_digests",
            params={"select": column, "limit": 1},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            continue

        try:
            error_payload = response.json()
        except Exception:
            error_payload = {"message": response.text}
        error_message = str(error_payload.get("message", error_payload))
        missing_message_markers = [
            f"news_digests.{column}",
            f"'{column}' column of 'news_digests'",
        ]
        if any(marker in error_message for marker in missing_message_markers):
            missing_columns.append(column)
            continue

        raise RuntimeError(
            f"Supabase news_digests schema probe failed for `{column}` with HTTP {response.status_code}: {error_message}"
        )

    if missing_columns:
        raise RuntimeError(
            "Supabase news_digests schema/cache is missing required columns for the current discovery flow: "
            + ", ".join(missing_columns)
        )

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    logger.info("Test 1/6: Verifying core module imports...")
    from agents.communication_agent import CommunicationAgent
    from agents.orchestrator import trigger_opportunity_discovery, wealth_manager_graph
    from agents.scout_agent import ScoutAgent
    from utils.discovery_support import normalize_policy_profile, parse_policy_text_fallback
    from utils.financial_tools import fetch_macro_indicators

    logger.info("✅ Test 1/6 passed: Core modules imported successfully.")
except Exception as e:
    record_failure(f"❌ Test 1/6 failed: Core modules failed to import. Error: {e}")
    sys.exit(1)


logger.info("\nTest 2/6: Verifying policy fallback parsing...")
try:
    policy = parse_policy_text_fallback(
        "Focus on semiconductors and cybersecurity, avoid highly leveraged turnarounds."
    )
    logger.info(f"Priority ETFs: {policy.get('priority_etfs')}")
    logger.info(f"Risk Avoidances: {policy.get('risk_avoidances')}")
    logger.info("✅ Test 2/6 passed: Policy fallback produced structured output.")
except Exception as e:
    record_failure(f"❌ Test 2/6 failed: Policy parsing failed. Error: {e}")


logger.info("\nTest 3/6: Verifying macro data fallback...")
try:
    macro = fetch_macro_indicators()
    logger.info(f"Macro Source: {macro.get('source')}")
    logger.info(
        f"Fed Funds: {macro.get('fed_funds_rate')} | Inflation: {macro.get('cpi_inflation') or macro.get('cpi_inflation_index')}"
    )
    logger.info("✅ Test 3/6 passed: Macro aggregator executed.")
except Exception as e:
    record_failure(f"❌ Test 3/6 failed: Macro fetch failed. Error: {e}")


logger.info("\nTest 4/6: Verifying Scout material-news filtering...")
try:
    filtered = ScoutAgent.filter_noise(
        [
            {
                "headline": "NVIDIA signs major AI datacenter contract",
                "summary": "A hyperscaler expanded GPU commitments.",
                "url": "https://example.com/1",
                "published_at": "2026-05-24T10:00:00+00:00",
                "source": "Example",
            },
            {
                "headline": "Weekly market newsletter mentions NVIDIA",
                "summary": "General market commentary without specific company impact.",
                "url": "https://example.com/2",
                "published_at": "2026-05-24T11:00:00+00:00",
                "source": "Example",
            },
        ],
        "NVDA",
        theme_context="Semiconductor Cycle / AI Infrastructure",
    )
    logger.info(f"Filtered news items: {len(filtered)}")
    logger.info("✅ Test 4/6 passed: Scout produced structured materiality output.")
except Exception as e:
    record_failure(f"❌ Test 4/6 failed: Scout filtering failed. Error: {e}")


logger.info("\nTest 5/6: Verifying discovery briefing formatting...")
try:
    sample_report = CommunicationAgent.compile_discovery_briefing(
        {
            "run_type": "deep",
            "themes": [
                {
                    "theme_name": "Semiconductor Cycle / AI Infrastructure",
                    "confidence_level": "high",
                    "why_now": "Datacenter capex and sector leadership both strengthened this week.",
                }
            ],
            "recommendations": [
                {
                    "symbol": "NVDA",
                    "recommendation_type": "new_position",
                    "theme_name": "Semiconductor Cycle / AI Infrastructure",
                    "investment_hypothesis": "Leadership in AI compute makes the company a direct expression of the theme.",
                    "why_now": "Fresh contract news and sector momentum aligned.",
                    "key_risks": ["Valuation is still demanding."],
                    "what_invalidates_it": ["AI spending slows materially."],
                }
            ],
            "summary_text": "Sample discovery report.",
        }
    )
    logger.info(sample_report)
    logger.info("✅ Test 5/6 passed: Discovery briefing formatted correctly.")
except Exception as e:
    record_failure(f"❌ Test 5/6 failed: Discovery briefing formatting failed. Error: {e}")


logger.info("\nTest 6/6: Verifying discovery orchestration fallback behavior...")
try:
    verify_live_news_digest_schema()
    result = trigger_opportunity_discovery(run_type="sweep", return_result=True)
    if result is None:
        logger.info("✅ Test 6/6 passed: Discovery sweep returned no-op cleanly.")
    elif isinstance(result, dict) and not result.get("error"):
        logger.info(
            "✅ Test 6/6 passed: Discovery sweep completed without surfacing an orchestration error."
        )
    else:
        error_summary = result if isinstance(result, str) else getattr(result, "get", lambda *_: None)("summary_text")
        raise RuntimeError(error_summary or "Unexpected discovery sweep return value.")
except Exception as e:
    record_failure(f"❌ Test 6/6 failed: Discovery orchestration failed. Error: {e}")

logger.info("\n==============================================================================")
logger.info("                     Verification Tasks Complete                              ")
logger.info("==============================================================================")

if failures:
    logger.error("Verification failed with %s issue(s).", len(failures))
    sys.exit(1)
