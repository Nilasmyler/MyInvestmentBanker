import logging
import re
from typing import List, Union

# Initialize logger
logger = logging.getLogger("MyInvestmentBanker.compressor")
logging.basicConfig(level=logging.INFO)

# Optional LLMLingua imports to prevent container crash on low memory
LLMLINGUA_AVAILABLE = False
compressor_instance = None

try:
    from llmlingua import PromptCompressor
    # Note: Loading PromptCompressor downloads small model weights (~1GB).
    # We load it lazily to avoid heavy start times or memory issues in free hosting.
    LLMLINGUA_AVAILABLE = True
except ImportError:
    logger.warning("LLMLingua package not installed. Extractive fallback will be used.")


def load_compressor():
    """Lazily loads the PromptCompressor to conserve startup memory."""
    global compressor_instance
    if LLMLINGUA_AVAILABLE and compressor_instance is None:
        try:
            logger.info("Initializing Microsoft LLMLingua PromptCompressor (GPT-2)...")
            # We use the default small gpt2 model which is memory efficient
            compressor_instance = PromptCompressor(
                model_name="gpt2", 
                device_map="cpu"
            )
            logger.info("LLMLingua successfully initialized on CPU.")
        except Exception as e:
            logger.error(f"Failed to load LLMLingua (likely low memory): {e}")
            compressor_instance = None


def compress_financial_text(context: str, target_token: int = 600, instruction: str = "") -> str:
    """
    Compresses bulky financial text (e.g., SEC disclosures, earnings transcripts)
    before sending it to Gemini 3.5 Flash.
    
    If LLMLingua is loaded, it compresses text tokens up to 20x.
    If LLMLingua is unavailable or fails due to resource constraints,
    it falls back to a highly optimized extractive text parsing algorithm.
    """
    if LLMLINGUA_AVAILABLE:
        load_compressor()
        
    if compressor_instance:
        try:
            logger.info(f"Compressing context with LLMLingua (Target: {target_token} tokens)...")
            compressed_data = compressor_instance.compress_prompt(
                context=[context],
                instruction=instruction,
                target_token=target_token,
                rank_method="longllmlingua",
                use_deep_llm=False
            )
            compressed_text = compressed_data.get("compressed_prompt", "")
            if compressed_text:
                compression_ratio = len(compressed_text) / max(len(context), 1)
                logger.info(f"LLMLingua completed. Compression Ratio: {compression_ratio:.2%}")
                return compressed_text
        except Exception as e:
            logger.warning(f"LLMLingua compression failed: {e}. Falling back to extractive compiler.")
            
    # ==============================================================================
    # Extractive Prompt Compactor (Fail-Safe Fallback)
    # ==============================================================================
    logger.info("Running Extractive Financial Compactor fallback...")
    return extractive_financial_compaction(context, target_token * 4) # Approx 4 characters per token


def extractive_financial_compaction(text: str, target_chars: int) -> str:
    """
    Extractive summarizer designed specifically for financial disclosures.
    It extracts:
    1. Numerical Tables and Key Balance Sheet rows.
    2. Heading rows & bullet structures.
    3. Sentences containing financial metrics (%, $, billions, increase/decrease).
    """
    if len(text) <= target_chars:
        return text
        
    lines = text.split("\n")
    high_value_lines = []
    
    # Financial keywords indicating crucial fundamental changes
    financial_regex = re.compile(
        r"(\d+(\.\d+)?%)|(\$\s?\d+)|(revenue|net income|operating income|debt|liability|margin|cash flow|liquidity|increase|decrease|risk|acquisition)",
        re.IGNORECASE
    )
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Keep table data structures (often contain pipes or multiple spaces)
        if "|" in line_strip or "   " in line_strip:
            high_value_lines.append(line_strip)
            continue
            
        # Keep headings and bullet points
        if line_strip.startswith(("#", "*", "-", "Item", "ITEM")):
            high_value_lines.append(line_strip)
            continue
            
        # Keep lines rich in financial data or metrics
        if financial_regex.search(line_strip):
            # Limit length of individual line to avoid wrapping bloat
            high_value_lines.append(line_strip[:150])
            
    # Join selected lines
    compacted_text = "\n".join(high_value_lines)
    
    # If still too long, perform hard truncation from the center/end while keeping schema
    if len(compacted_text) > target_chars:
        logger.info(f"Extractive text still exceeds targets. Truncating to {target_chars} chars.")
        return compacted_text[:target_chars] + "\n[... Extractive Truncation Applied due to Prompt Size Limits ...]"
        
    return compacted_text
