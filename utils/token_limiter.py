"""
Token Rate Limiting and Chunking Utilities

Handles TPM (Tokens Per Minute) rate limiting by:
1. Estimating tokens before API calls
2. Detecting when requests would exceed limits
3. Chunking large requests with delays
4. Providing cost estimates for user approval
"""

from __future__ import annotations

import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


# TPM limits for different models (Tokens Per Minute)
# These are conservative estimates - actual limits may vary by account tier
MODEL_TPM_LIMITS: Dict[str, int] = {
    # OpenAI models
    "gpt-5.1": 500_000,
    "gpt-5": 500_000,
    "gpt-5-mini": 500_000,
    "gpt-5-nano": 500_000,
    "gpt-4o": 10_000_000,  # Very high limit
    "gpt-4o-mini": 10_000_000,
    "gpt-4-turbo": 1_000_000,
    "gpt-4": 1_000_000,
    
    # Gemini models
    "gemini-2.0-flash": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-pro-latest": 1_000_000,
    "gemini-2.0-flash-exp": 1_000_000,
    
    # Claude models
    "claude-3-5-sonnet-20241022": 1_000_000,
    "claude-3-opus-20240229": 1_000_000,
    "claude-3-5-haiku-20241022": 1_000_000,
    "claude-3-haiku-20240307": 1_000_000,
    
    # DeepSeek
    "deepseek-chat": 1_000_000,
}

# Default TPM if model not found (conservative)
DEFAULT_TPM_LIMIT = 500_000

# Cost per 1M tokens (approximate, varies by model and region)
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "gpt-5.1": {"input": 2.50, "output": 10.00},
    "gpt-5": {"input": 2.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-nano": {"input": 0.05, "output": 0.20},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro-latest": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}

# Default costs if model not found
DEFAULT_COSTS = {"input": 1.00, "output": 2.00}


@dataclass
class TokenEstimate:
    """Token estimation result."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    exceeds_tpm: bool
    tpm_limit: int
    chunks_needed: int


def estimate_tokens_openai(text: str) -> int:
    """
    Estimate tokens for OpenAI models using tiktoken (if available) or approximation.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    try:
        import tiktoken
        # Use cl100k_base encoding (used by GPT-4, GPT-3.5, etc.)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        # This is conservative - actual ratio varies
        return len(text) // 4


def estimate_tokens_generic(text: str) -> int:
    """
    Generic token estimation for non-OpenAI models.
    
    Uses a conservative approximation: 1 token ≈ 3-4 characters for most models.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Conservative estimate: 1 token per 3.5 characters (average across models)
    return int(len(text) / 3.5)


def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Estimate tokens for a given text and model.
    
    Args:
        text: Text to estimate tokens for
        model_name: Optional model name (uses OpenAI estimation if OpenAI model)
        
    Returns:
        Estimated token count
    """
    if model_name and ("gpt" in model_name.lower() or "openai" in model_name.lower()):
        return estimate_tokens_openai(text)
    return estimate_tokens_generic(text)


def estimate_message_tokens(messages: List[Any], model_name: Optional[str] = None) -> Tuple[int, int]:
    """
    Estimate input and output tokens for a list of messages.
    
    Args:
        messages: List of message objects (SystemMessage, HumanMessage, AIMessage, etc.)
        model_name: Optional model name for accurate estimation
        
    Returns:
        Tuple of (input_tokens, output_tokens_estimate)
    """
    total_input = 0
    total_output_estimate = 0
    
    for msg in messages:
        # Extract content from message
        if hasattr(msg, 'content'):
            content = str(msg.content)
        else:
            content = str(msg)
        
        # Estimate tokens for this message
        msg_tokens = estimate_tokens(content, model_name)
        total_input += msg_tokens
        
        # If it's a user message or system message, assume it will generate a response
        # Estimate output as ~30% of input (conservative)
        if hasattr(msg, '__class__'):
            class_name = msg.__class__.__name__
            if class_name in ['HumanMessage', 'SystemMessage']:
                total_output_estimate += int(msg_tokens * 0.3)
    
    return total_input, total_output_estimate


def get_tpm_limit(model_name: Optional[str]) -> int:
    """
    Get TPM limit for a model.
    
    Args:
        model_name: Model name (e.g., "gpt-5.1", "gemini-2.0-flash")
        
    Returns:
        TPM limit for the model
    """
    if not model_name:
        return DEFAULT_TPM_LIMIT
    
    # Try exact match first
    if model_name in MODEL_TPM_LIMITS:
        return MODEL_TPM_LIMITS[model_name]
    
    # Try partial match (e.g., "gpt-5.1" matches "gpt-5.1")
    model_lower = model_name.lower()
    for key, limit in MODEL_TPM_LIMITS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return limit
    
    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default TPM limit: {DEFAULT_TPM_LIMIT}")
    return DEFAULT_TPM_LIMIT


def estimate_cost(input_tokens: int, output_tokens: int, model_name: Optional[str] = None) -> float:
    """
    Estimate cost for a request.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (estimated)
        model_name: Model name
        
    Returns:
        Estimated cost in USD
    """
    if not model_name:
        costs = DEFAULT_COSTS
    else:
        costs = MODEL_COSTS.get(model_name, DEFAULT_COSTS)
    
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    
    return input_cost + output_cost


def check_tpm_limit(
    input_tokens: int,
    output_tokens_estimate: int,
    model_name: Optional[str] = None
) -> TokenEstimate:
    """
    Check if a request would exceed TPM limits and calculate chunking needs.
    
    Args:
        input_tokens: Estimated input tokens
        output_tokens_estimate: Estimated output tokens
        model_name: Model name
        
    Returns:
        TokenEstimate with all relevant information
    """
    total_tokens = input_tokens + output_tokens_estimate
    tpm_limit = get_tpm_limit(model_name)
    
    exceeds_tpm = total_tokens > tpm_limit
    chunks_needed = 1
    
    if exceeds_tpm:
        # Calculate how many chunks we need
        # Leave 10% buffer to account for estimation errors
        safe_limit = int(tpm_limit * 0.9)
        chunks_needed = (total_tokens // safe_limit) + (1 if total_tokens % safe_limit > 0 else 0)
    
    estimated_cost = estimate_cost(input_tokens, output_tokens_estimate, model_name)
    
    return TokenEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens_estimate,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        exceeds_tpm=exceeds_tpm,
        tpm_limit=tpm_limit,
        chunks_needed=chunks_needed
    )


def chunk_messages(
    messages: List[Any],
    max_tokens_per_chunk: int,
    model_name: Optional[str] = None
) -> List[List[Any]]:
    """
    Split messages into chunks that respect TPM limits.
    
    IMPORTANT: This function preserves tool call sequences. It will NOT split
    between an AIMessage with tool_calls and its corresponding ToolMessages.
    
    Args:
        messages: List of messages to chunk
        max_tokens_per_chunk: Maximum tokens per chunk (with buffer)
        model_name: Model name for accurate estimation
        
    Returns:
        List of message chunks
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    # Always include system message in first chunk
    system_msg = None
    if messages and hasattr(messages[0], '__class__') and messages[0].__class__.__name__ == 'SystemMessage':
        system_msg = messages[0]
        messages = messages[1:]
        current_tokens = estimate_tokens(str(system_msg.content), model_name)
        current_chunk.append(system_msg)
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        msg_tokens = estimate_tokens(str(getattr(msg, 'content', msg)), model_name)
        
        # Check if this is an AIMessage with tool_calls
        # If so, we need to include all corresponding ToolMessages in the same chunk
        tool_call_group = []
        tool_call_group_tokens = 0
        
        if hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # This is an AIMessage with tool calls - collect it and all following ToolMessages
                tool_call_group.append(msg)
                tool_call_group_tokens += msg_tokens
                
                # Collect all ToolMessages that follow (they respond to the tool calls)
                j = i + 1
                while j < len(messages):
                    next_msg = messages[j]
                    if hasattr(next_msg, '__class__') and next_msg.__class__.__name__ == 'ToolMessage':
                        tool_call_group.append(next_msg)
                        tool_call_group_tokens += estimate_tokens(
                            str(getattr(next_msg, 'content', next_msg)), model_name
                        )
                        j += 1
                    else:
                        # Not a ToolMessage, stop collecting
                        break
                
                # Check if the entire tool call group fits in current chunk
                if current_tokens + tool_call_group_tokens > max_tokens_per_chunk and current_chunk:
                    # Tool call group doesn't fit - start new chunk
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                    # Add system message to new chunk if we have one
                    if system_msg:
                        current_chunk.append(system_msg)
                        current_tokens = estimate_tokens(str(system_msg.content), model_name)
                
                # Add entire tool call group to current chunk
                current_chunk.extend(tool_call_group)
                current_tokens += tool_call_group_tokens
                i = j  # Skip to after the tool call group
                continue
        
        # Regular message (not part of tool call group)
        # If adding this message would exceed limit, start new chunk
        if current_tokens + msg_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
            # Add system message to new chunk if we have one
            if system_msg:
                current_chunk.append(system_msg)
                current_tokens = estimate_tokens(str(system_msg.content), model_name)
        
        current_chunk.append(msg)
        current_tokens += msg_tokens
        i += 1
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [messages]


def wait_for_rate_limit(seconds: int = 61) -> None:
    """
    Wait for rate limit window to reset.
    
    Args:
        seconds: Number of seconds to wait (default 61 to ensure window reset)
    """
    logger.info(f"⏳ Waiting {seconds} seconds for rate limit window to reset...")
    for remaining in range(seconds, 0, -10):
        if remaining > 10:
            logger.info(f"   {remaining} seconds remaining...")
            time.sleep(10)
        else:
            time.sleep(remaining)
            break
    logger.info("✓ Rate limit window reset - continuing...")


def format_token_warning(estimate: TokenEstimate, model_name: str) -> str:
    """
    Format a user-friendly warning message about token limits.
    
    Args:
        estimate: TokenEstimate object
        model_name: Model name
        
    Returns:
        Formatted warning message
    """
    return (
        f"⚠️ **Token Limit Warning**\n\n"
        f"Your request would use approximately **{estimate.total_tokens:,} tokens** "
        f"({estimate.input_tokens:,} input + ~{estimate.output_tokens:,} output), "
        f"which exceeds the **{estimate.tpm_limit:,} tokens/minute** limit for `{model_name}`.\n\n"
        f"**Estimated Cost:** ${estimate.estimated_cost:.4f}\n"
        f"**Chunks Required:** {estimate.chunks_needed} (with ~61s delays between chunks)\n"
        f"**Estimated Time:** ~{estimate.chunks_needed * 2} minutes\n\n"
        f"Would you like to proceed? The agent will automatically split the work into chunks "
        f"and wait between them to respect rate limits.\n\n"
        f"Reply: **yes** / **continue** / **proceed** to continue, or **no** / **cancel** to abort."
    )

