from __future__ import annotations

"""
Cost Estimation Utility for SurvyAI

This module provides cost estimation for LLM API calls based on:
- Model type (GPT-4o, GPT-5-mini, GPT-5.1, etc.)
- Input/output token counts
- Current OpenAI pricing (as of 2024-2025)

Pricing is based on OpenAI's published rates and estimated rates for newer models.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    
    # GPT-5 family — align with platform.openai.com (per 1M tokens, USD)
    "gpt-5-nano": {
        "input": 0.10,
        "output": 0.40,
        "cached_input": 0.025,
    },
    "gpt-5-mini": {
        "input": 0.20,
        "output": 0.80,
        "cached_input": 0.05,
    },
    "gpt-5.4": {
        "input": 2.50,
        "output": 15.00,
        "cached_input": 0.25,
    },
    "gpt-5.2": {
        "input": 2.50,
        "output": 15.00,
        "cached_input": 0.25,
    },
    "gpt-5.1": {
        "input": 2.50,
        "output": 15.00,
        "cached_input": 0.25,
    },
    "gpt-5": {
        "input": 2.50,
        "output": 15.00,
        "cached_input": 0.25,
    },

    # Claude family (approximate published USD rates per 1M tokens)
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 3.00,
    },
    "claude-3-5-sonnet-20240620": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 3.00,
    },
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00,
        "cached_input": 0.80,
    },
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "cached_input": 15.00,
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 3.00,
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,
        "output": 1.25,
        "cached_input": 0.25,
    },

    # Gemini family (approximate blended text pricing per 1M tokens)
    "gemini-2.0-flash": {
        "input": 0.10,
        "output": 0.40,
        "cached_input": 0.10,
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "cached_input": 0.075,
    },
    "gemini-pro-latest": {
        "input": 1.25,
        "output": 5.00,
        "cached_input": 1.25,
    },

    # DeepSeek family
    "deepseek-chat": {
        "input": 0.27,
        "output": 1.10,
        "cached_input": 0.07,
    },
    "deepseek-reasoner": {
        "input": 0.55,
        "output": 2.19,
        "cached_input": 0.14,
    },
    
    # Fallback for unknown models
    "default": {
        "input": 2.50,
        "output": 10.00,
        "cached_input": 0.25,
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
        Dict with "input", "output", and optional "cached_input" prices per 1M tokens
    """
    model_lower = (model_name or "").lower().strip()
    if not model_lower:
        return OPENAI_PRICING["default"]

    if model_lower in OPENAI_PRICING:
        return OPENAI_PRICING[model_lower]

    # Longest key first so "gpt-5.4" matches before "gpt-5"
    best_key: Optional[str] = None
    for key in sorted(OPENAI_PRICING.keys(), key=len, reverse=True):
        if key == "default":
            continue
        if key in model_lower:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key:
        return OPENAI_PRICING[best_key]

    logger.warning(f"Unknown model '{model_name}', using default pricing")
    return OPENAI_PRICING["default"]


def extract_cached_input_tokens(usage: Dict[str, Any]) -> int:
    """Read cached prompt tokens from OpenAI / LangChain usage dicts."""
    if not usage:
        return 0
    for key in (
        "cache_read_input_tokens",
        "cached_input_tokens",
        "prompt_cache_hit_tokens",
    ):
        val = usage.get(key)
        if val is not None:
            return max(0, int(val))
    for details_key in (
        "input_token_details",
        "input_tokens_details",
        "prompt_tokens_details",
    ):
        details = usage.get(details_key)
        if isinstance(details, dict):
            for dk in ("cache_read", "cached_tokens", "cached"):
                v = details.get(dk)
                if v is not None:
                    return max(0, int(v))
    return 0


def extract_message_token_usage(message: Any) -> Optional[Dict[str, int]]:
    """Token usage from one LangChain AIMessage, if the provider reported it."""
    um = getattr(message, "usage_metadata", None) or {}
    if isinstance(um, dict) and um:
        inp = int(
            um.get("input_tokens")
            or um.get("prompt_tokens")
            or um.get("input_token_count")
            or 0
        )
        out = int(
            um.get("output_tokens")
            or um.get("completion_tokens")
            or um.get("output_token_count")
            or 0
        )
        if inp or out:
            return {
                "input_tokens": inp,
                "output_tokens": out,
                "cached_input_tokens": extract_cached_input_tokens(um),
            }

    rm = getattr(message, "response_metadata", None) or {}
    if isinstance(rm, dict):
        tu = rm.get("token_usage") or rm.get("usage") or {}
        if isinstance(tu, dict):
            inp = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            out = int(
                tu.get("completion_tokens") or tu.get("output_tokens") or 0
            )
            if inp or out:
                return {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cached_input_tokens": extract_cached_input_tokens(tu),
                }
    return None


def _infer_cached_input_tokens(
    turn_index: int,
    input_tokens: int,
    explicit_cached: int,
    previous_input_tokens: int,
) -> int:
    """
    When providers omit cache fields, approximate cache hits on repeat context.
    """
    if explicit_cached > 0:
        return min(explicit_cached, input_tokens)
    if turn_index <= 0 or previous_input_tokens <= 0 or input_tokens <= 0:
        return 0
    if input_tokens < 512:
        return 0
    # Later turns usually resend the same prefix; most input is cache-priced.
    overlap = min(input_tokens, previous_input_tokens)
    if overlap < 1024:
        return 0
    return int(min(input_tokens, overlap * 0.92))


def estimate_token_cost_usd(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Provider-style USD cost for one API call (uncached + cached input + output)."""
    pricing = get_model_pricing(model_name)
    inp = max(0, int(input_tokens))
    out = max(0, int(output_tokens))
    cached = max(0, min(int(cached_input_tokens), inp))
    uncached = inp - cached
    cached_rate = float(pricing.get("cached_input", pricing["input"]))
    in_cost = (uncached / 1_000_000) * pricing["input"] + (cached / 1_000_000) * cached_rate
    out_cost = (out / 1_000_000) * pricing["output"]
    return in_cost + out_cost


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: Optional[int] = None,
    estimated_output_tokens: int = 2000,
    cached_input_tokens: int = 0,
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
    output_token_count = output_tokens if output_tokens is not None else estimated_output_tokens
    cached = max(0, min(int(cached_input_tokens), int(input_tokens)))
    uncached = max(0, int(input_tokens) - cached)
    cached_rate = float(pricing.get("cached_input", pricing["input"]))
    input_cost = (uncached / 1_000_000) * pricing["input"] + (cached / 1_000_000) * cached_rate
    output_cost = (output_token_count / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_token_count,
        "cached_input_tokens": cached,
        "model": model_name,
        "pricing_tier": (
            f"${pricing['input']:.2f}/1M input, "
            f"${cached_rate:.2f}/1M cached, "
            f"${pricing['output']:.2f}/1M output"
        ),
        "pricing_source": "static_table(utils/cost_estimator.OPENAI_PRICING)",
    }


def estimate_graph_llm_cost_usd(
    messages: Sequence[Any],
    model_name: str,
    *,
    response_text: str = "",
    initial_messages_token_hint: Optional[int] = None,
    infer_missing_cached: bool = True,
) -> float:
    """
    Sum provider-reported usage across graph AIMessages (per-call, with cache pricing).

    Returns raw USD (before SurvyAI credit markup). Falls back to content-based
    estimates only when no AIMessage includes usage metadata.
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return 0.0

    mn = (model_name or "").strip() or "gpt-5-mini"
    msgs: List[Any] = list(messages or [])
    total_cost = 0.0
    usage_turns = 0
    prev_in = 0
    turn_index = 0

    for m in msgs:
        if not isinstance(m, AIMessage):
            continue

        usage = extract_message_token_usage(m)
        if not usage:
            continue

        inp = usage["input_tokens"]
        explicit_cached = usage["cached_input_tokens"]
        cached = explicit_cached
        if infer_missing_cached:
            cached = _infer_cached_input_tokens(
                turn_index, inp, explicit_cached, prev_in
            )
        total_cost += estimate_token_cost_usd(
            mn,
            inp,
            usage["output_tokens"],
            cached_input_tokens=cached,
        )
        prev_in = inp
        usage_turns += 1
        turn_index += 1

    if usage_turns > 0 and total_cost > 0:
        return round(total_cost, 6)

    ai_turns = sum(1 for m in msgs if isinstance(m, AIMessage))
    out_est = 0
    for m in msgs:
        if isinstance(m, AIMessage):
            content = getattr(m, "content", "") or ""
            if isinstance(content, str):
                out_est += estimate_tokens(content, method="characters")
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        out_est += estimate_tokens(
                            str(part.get("text", "")), method="characters"
                        )
    if not out_est and response_text:
        out_est = estimate_tokens(response_text, method="characters")
    out_est = max(out_est, 400 if ai_turns else 0)

    in_est = max(int(initial_messages_token_hint or 0), 100)
    if ai_turns > 1:
        in_est = min(in_est * ai_turns, 500_000)
    else:
        in_est = min(in_est, 500_000)

    ec = estimate_cost(mn, in_est, output_tokens=min(out_est, 128_000))
    return float(max(0.0, ec.get("total_cost") or 0.0))


def summarize_graph_llm_usage(
    messages: Sequence[Any],
    model_name: str,
    *,
    response_text: str = "",
    initial_messages_token_hint: Optional[int] = None,
    infer_missing_cached: bool = True,
) -> Dict[str, Any]:
    """
    Summarize total LLM token usage for a graph run.

    Returns a dict suitable for telemetry/billing handoff:
    ``model_name``, ``input_tokens``, ``output_tokens``, ``cached_input_tokens``,
    ``cost_usd`` and ``estimated``.
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        AIMessage = None  # type: ignore[assignment]

    mn = (model_name or "").strip() or "gpt-5-mini"
    msgs: List[Any] = list(messages or [])
    total_input = 0
    total_output = 0
    total_cached = 0
    usage_turns = 0
    prev_in = 0
    turn_index = 0

    if AIMessage is not None:
        for m in msgs:
            if not isinstance(m, AIMessage):
                continue
            usage = extract_message_token_usage(m)
            if not usage:
                continue
            inp = int(usage["input_tokens"])
            out = int(usage["output_tokens"])
            explicit_cached = int(usage["cached_input_tokens"])
            cached = explicit_cached
            if infer_missing_cached:
                cached = _infer_cached_input_tokens(
                    turn_index, inp, explicit_cached, prev_in
                )
            total_input += inp
            total_output += out
            total_cached += max(0, min(cached, inp))
            prev_in = inp
            usage_turns += 1
            turn_index += 1

    if usage_turns > 0:
        return {
            "model_name": mn,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cached_input_tokens": total_cached,
            "cost_usd": round(
                estimate_token_cost_usd(
                    mn,
                    total_input,
                    total_output,
                    cached_input_tokens=total_cached,
                ),
                6,
            ),
            "estimated": False,
            "usage_turns": usage_turns,
        }

    out_est = 0
    if AIMessage is not None:
        ai_turns = sum(1 for m in msgs if isinstance(m, AIMessage))
        for m in msgs:
            if not isinstance(m, AIMessage):
                continue
            content = getattr(m, "content", "") or ""
            if isinstance(content, str):
                out_est += estimate_tokens(content, method="characters")
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        out_est += estimate_tokens(
                            str(part.get("text", "")), method="characters"
                        )
    else:
        ai_turns = 0

    if not out_est and response_text:
        out_est = estimate_tokens(response_text, method="characters")
    out_est = max(out_est, 400 if ai_turns else 0)
    in_est = max(int(initial_messages_token_hint or 0), 100)
    if ai_turns > 1:
        in_est = min(in_est * ai_turns, 500_000)
    else:
        in_est = min(in_est, 500_000)

    return {
        "model_name": mn,
        "input_tokens": int(in_est),
        "output_tokens": int(min(out_est, 128_000)),
        "cached_input_tokens": 0,
        "cost_usd": round(
            estimate_token_cost_usd(
                mn,
                int(in_est),
                int(min(out_est, 128_000)),
                cached_input_tokens=0,
            ),
            6,
        ),
        "estimated": True,
        "usage_turns": int(ai_turns),
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

