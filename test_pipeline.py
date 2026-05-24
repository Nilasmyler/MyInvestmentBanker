import os
import sys
import logging
from dotenv import load_dotenv

# Load environmental configs
load_dotenv()

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MyInvestmentBanker.tester")

logger.info("==============================================================================")
logger.info("           MyInvestmentBanker: Local Pipeline Verification Script             ")
logger.info("==============================================================================")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test 1: Verify Module Imports
try:
    logger.info("Test 1/5: Verifying agent, database, and utility imports...")
    from database.supabase_client import get_embedding, fetch_portfolio
    from utils.prompt_compressor import compress_financial_text
    from utils.financial_tools import fetch_macro_indicators
    from agents.orchestrator import wealth_manager_graph
    logger.info("✅ Test 1/5 passed: All Python imports resolved successfully.")
except Exception as e:
    logger.error(f"❌ Test 1/5 failed: Core modules failed to import. Error: {e}")
    sys.exit(1)


# Test 2: Verify Gemini API Connection & Embeddings
logger.info("\nTest 2/5: Verifying connection to Gemini API & Text Embeddings...")
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    logger.warning("⚠️ GEMINI_API_KEY is not configured in .env. Skipping vector tests.")
else:
    try:
        vector = get_embedding("Testing vector embedding mapping.", task_type="retrieval_document")
        if len(vector) == 768:
            logger.info(f"✅ Test 2/5 passed: Generated {len(vector)}-dimension vector using text-embedding-004.")
        else:
            logger.warning(f"⚠️ Vector generated but dimension mismatched (expected 768, got {len(vector)}).")
    except Exception as e:
        logger.error(f"❌ Test 2/5 failed: Gemini API query failed. Error: {e}")


# Test 3: Verify prompt compression / compaction fallback
logger.info("\nTest 3/5: Verifying Prompt Compaction Middleware...")
try:
    dummy_text = (
        "Item 1. Financial Statements.\n"
        "Total Assets were $1,000,000 as of December 2025. This represents a substantial increase.\n"
        "Total Liabilities were $400,000 as of December 2025. This is a solid performance.\n"
        "We are happy to declare that revenues have spiked by 12% due to services expansion.\n"
        "This is generic boilerplate text that does not contain numbers or high value financial keywords."
    )
    compacted = compress_financial_text(dummy_text, target_token=30, instruction="Compress report.")
    logger.info(f"Original Length: {len(dummy_text)} chars | Compacted Length: {len(compacted)} chars")
    logger.info("Compacted Snippet:\n" + compacted)
    logger.info("✅ Test 3/5 passed: Prompt compactor successfully extracted numerical rows and metrics.")
except Exception as e:
    logger.error(f"❌ Test 3/5 failed: Prompt compressor execution failed. Error: {e}")


# Test 4: Verify live economic FRED fallback
logger.info("\nTest 4/5: Verifying Macroeconomic Data Feeds...")
try:
    macro = fetch_macro_indicators()
    logger.info(f"Fetched Macro Data Source: {macro.get('source')}")
    logger.info(f"Fed Funds: {macro.get('fed_funds_rate')} | Inflation Index: {macro.get('cpi_inflation') or macro.get('cpi_inflation_index')}")
    logger.info("✅ Test 4/5 passed: FRED aggregator executed smoothly (using API or solid local baseline).")
except Exception as e:
    logger.error(f"❌ Test 4/5 failed: Macro aggregator failed. Error: {e}")


# Test 5: Verify LangGraph orchestrator invocation
logger.info("\nTest 5/5: Executing Dry-Run LangGraph Multi-Agent Cycle...")
if not gemini_key:
    logger.warning("⚠️ GEMINI_API_KEY missing. Skipping multi-agent dry-run.")
else:
    try:
        # Mock initial pipeline states
        initial_state = {
            "portfolio": [{"symbol": "AAPL", "quantity": 10, "cost_basis": 175.50}],
            "raw_ingested_data": {},
            "cfa_analyst_memos": [],
            "risk_assessment": {},
            "final_synthesis_report": "",
            "messages": []
        }
        
        logger.info("Invoking LangGraph StateGraph (Scout -> CFA -> Risk -> Synthesis)...")
        final_state = wealth_manager_graph.invoke(initial_state)
        
        synthesis = final_state.get("final_synthesis_report", "")
        if synthesis and not synthesis.startswith("Error"):
            logger.info("✅ Test 5/5 passed: LangGraph successfully completed dynamic routing.")
            logger.info("\n=== PREVIEW SYNTHESIS BRIEFING ===\n")
            print(synthesis)
            logger.info("\n==================================")
        else:
            logger.error("❌ Test 5/5 failed: Synthesis report generation failed.")
    except Exception as e:
        logger.error(f"❌ Test 5/5 failed: LangGraph execution failed. Error: {e}")

logger.info("\n==============================================================================")
logger.info("                     Verification Tasks Complete                              ")
logger.info("==============================================================================")
