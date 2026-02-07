from __future__ import annotations

"""
Cost Estimation Utility for SurvyAI

This module provides cost estimation for LLM API calls based on:
- Model type (GPT-4o, GPT-5-mini, GPT-5.1, etc.)
- Input/output token counts
- Current OpenAI pricing (as of 2024-2025)

Pricing is based on OpenAI's published rates and estimated rates for newer models.
"""

from typing import Any, Dict, Optional, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

# OpenAI Pricing (per 1M tokens) - Updated for 2024-2025
# Source: OpenAI pricing page (prices may vary by region)
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    # GPT-4o family
    "gpt-4o": {
        "input": 2.50,   # $2.50 per 1M input tokens
        "output": 10.00  # $10.00 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,   # $0.15 per 1M input tokens
        "output": 0.60   # $0.60 per 1M output tokens
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00
    },
    
    # GPT-4 Turbo family
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00
    },
    "gpt-4": {
        "input": 30.00,
        "output": 60.00
    },
    
    # GPT-5 family (estimated pricing based on tier)
    "gpt-5-nano": {
        "input": 0.10,   # Estimated: cheaper than mini
        "output": 0.40
    },
    "gpt-5-mini": {
        "input": 0.20,   # Estimated: slightly more than 4o-mini
        "output": 0.80
    },
    "gpt-5": {
        "input": 3.00,   # Estimated: premium pricing
        "output": 12.00
    },
    "gpt-5.1": {
        "input": 5.00,   # Estimated: highest tier
        "output": 20.00
    },
    
    # Fallback for unknown models
    "default": {
        "input": 2.50,
        "output": 10.00
    }
}

# Token estimation: ~4 characters = 1 token, or ~0.75 words = 1 token
# For more accurate: use tiktoken library if available
def estimate_tokens(text: str, method: str = "characters") -> int:
    """
    Estimate token count from text.
    
    Args:
        text: Text to estimate tokens for
        method: "characters" (4 chars = 1 token) or "words" (0.75 words = 1 token)
    
    Returns:
        Estimated token count
    """
    if method == "characters":
        # Rough estimate: 4 characters per token
        return len(text) // 4
    else:
        # Rough estimate: 0.75 words per token
        word_count = len(text.split())
        return int(word_count / 0.75)


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """
    Get pricing for a specific model.
    
    Args:
        model_name: Model name (e.g., "gpt-5-mini", "gpt-4o")
    
    Returns:
        Dict with "input" and "output" prices per 1M tokens
    """
    # Normalize model name
    model_lower = model_name.lower()
    
    # Check exact match first
    if model_lower in OPENAI_PRICING:
        return OPENAI_PRICING[model_lower]
    
    # Check partial matches
    for key in OPENAI_PRICING:
        if key in model_lower or model_lower in key:
            return OPENAI_PRICING[key]
    
    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default pricing")
    return OPENAI_PRICING["default"]


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: Optional[int] = None,
    estimated_output_tokens: int = 2000
) -> Dict[str, Any]:
    """
    Estimate API cost for a request.
    
    Args:
        model_name: Model name (e.g., "gpt-5-mini")
        input_tokens: Number of input tokens
        output_tokens: Actual output tokens (if known)
        estimated_output_tokens: Estimated output tokens if not known
    
    Returns:
        Dict containing:
        - input_cost: Cost for input tokens (USD)
        - output_cost: Cost for output tokens (USD)
        - total_cost: Total cost (USD)
        - input_tokens: Input token count
        - output_tokens: Output token count (estimated or actual)
        - model: Model name used
    """
    pricing = get_model_pricing(model_name)
    
    # Calculate input cost
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    
    # Calculate output cost
    output_token_count = output_tokens if output_tokens is not None else estimated_output_tokens
    output_cost = (output_token_count / 1_000_000) * pricing["output"]
    
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_token_count,
        "model": model_name,
        "pricing_tier": f"${pricing['input']:.2f}/1M input, ${pricing['output']:.2f}/1M output",
        "pricing_source": "static_table(utils/cost_estimator.OPENAI_PRICING)",
    }


def estimate_document_processing_cost(
    model_name: str,
    document_text: str,
    query_text: str = "",
    estimated_iterations: int = 3,
    estimated_output_per_iteration: int = 2000
) -> Dict[str, Any]:
    """
    Estimate total cost for processing a document with multiple iterations.
    
    Args:
        model_name: Model name
        document_text: Full document text
        query_text: User query text
        estimated_iterations: Estimated number of agent-tool iterations
        estimated_output_per_iteration: Estimated output tokens per iteration
    
    Returns:
        Dict with cost breakdown and recommendations
    """
    # Estimate tokens
    doc_tokens = estimate_tokens(document_text)
    query_tokens = estimate_tokens(query_text) if query_text else 500  # Default query size
    
    # System prompt and context overhead (~1000 tokens)
    overhead_tokens = 1000
    
    # Per iteration: document + query + overhead
    input_tokens_per_iteration = doc_tokens + query_tokens + overhead_tokens
    
    # Total input tokens (document sent multiple times potentially)
    total_input_tokens = input_tokens_per_iteration * estimated_iterations
    
    # Total output tokens
    total_output_tokens = estimated_output_per_iteration * estimated_iterations
    
    # Calculate cost
    cost_breakdown = estimate_cost(model_name, total_input_tokens, total_output_tokens)
    
    # Add recommendations
    recommendations = []
    if doc_tokens > 100000:
        recommendations.append("⚠️ Very large document (>100K tokens). Consider using section extraction.")
    if doc_tokens > 200000:
        recommendations.append("⚠️ Extremely large document (>200K tokens). Chunking recommended.")
    if cost_breakdown["total_cost"] > 1.0:
        recommendations.append(f"💰 Estimated cost: ${cost_breakdown['total_cost']:.2f}. Consider using a cheaper model for initial processing.")
    
    cost_breakdown["document_tokens"] = doc_tokens
    cost_breakdown["query_tokens"] = query_tokens
    cost_breakdown["estimated_iterations"] = estimated_iterations
    cost_breakdown["recommendations"] = recommendations
    
    return cost_breakdown


def estimate_document_processing_cost_from_tokens(
    model_name: str,
    document_tokens: int,
    query_tokens: int = 500,
    estimated_iterations: int = 3,
    estimated_output_per_iteration: int = 2000,
    overhead_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Estimate total cost for processing a document using token counts (no large string allocation).
    
    Args:
        model_name: Model name
        document_tokens: Estimated document tokens
        query_tokens: Estimated query tokens
        estimated_iterations: Estimated number of agent-tool iterations
        estimated_output_per_iteration: Estimated output tokens per iteration
        overhead_tokens: System/tool overhead tokens per iteration
    
    Returns:
        Dict with cost breakdown and recommendations
    """
    input_tokens_per_iteration = max(0, int(document_tokens) + int(query_tokens) + int(overhead_tokens))
    total_input_tokens = input_tokens_per_iteration * int(estimated_iterations)
    total_output_tokens = int(estimated_output_per_iteration) * int(estimated_iterations)

    cost_breakdown = estimate_cost(model_name, total_input_tokens, total_output_tokens)

    recommendations: list[str] = []
    if document_tokens > 100000:
        recommendations.append("⚠️ Very large document (>100K tokens). Consider using section extraction.")
    if document_tokens > 200000:
        recommendations.append("⚠️ Extremely large document (>200K tokens). Chunking recommended.")
    if cost_breakdown["total_cost"] > 1.0:
        recommendations.append(
            f"💰 Estimated cost: ${cost_breakdown['total_cost']:.2f}. Consider using a cheaper model for initial processing."
        )

    cost_breakdown["document_tokens"] = int(document_tokens)
    cost_breakdown["query_tokens"] = int(query_tokens)
    cost_breakdown["estimated_iterations"] = int(estimated_iterations)
    cost_breakdown["recommendations"] = recommendations
    return cost_breakdown


def format_cost_summary(cost_data: Dict[str, Any]) -> str:
    """
    Format cost estimation as a readable summary.
    
    Args:
        cost_data: Output from estimate_cost or estimate_document_processing_cost
    
    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 60,
        "💰 COST ESTIMATION",
        "=" * 60,
        f"Model: {cost_data['model']}",
        f"Pricing: {cost_data.get('pricing_tier', 'N/A')}",
        "",
        "Token Breakdown:",
        f"  • Input tokens: {cost_data['input_tokens']:,}",
        f"  • Output tokens: {cost_data['output_tokens']:,}",
        "",
        "Cost Breakdown:",
        f"  • Input cost: ${cost_data['input_cost']:.4f}",
        f"  • Output cost: ${cost_data['output_cost']:.4f}",
        f"  • Total cost: ${cost_data['total_cost']:.4f}",
    ]
    
    if "document_tokens" in cost_data:
        lines.insert(6, f"  • Document tokens: {cost_data['document_tokens']:,}")
        lines.insert(7, f"  • Query tokens: {cost_data['query_tokens']:,}")
        lines.insert(8, f"  • Estimated iterations: {cost_data['estimated_iterations']}")
        lines.insert(9, "")
    
    if cost_data.get("recommendations"):
        lines.append("")
        lines.append("Recommendations:")
        for rec in cost_data["recommendations"]:
            lines.append(f"  {rec}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

