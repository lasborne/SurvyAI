"""
================================================================================
SurvyAI Command-Line Interface (CLI)
================================================================================

This module provides the command-line interface for interacting with SurvyAI.
It uses the Click library to create a user-friendly CLI experience.

WHAT IS CLICK?
--------------
Click is a Python package for creating command-line interfaces. It handles:
- Argument parsing
- Help text generation
- Error handling
- Input validation

AVAILABLE COMMANDS:
-------------------
1. query - Process a surveying query through the AI agent
   Usage: python -m cli query "Your question here"
   
2. list-models - List available Gemini models
   Usage: python -m cli list-models
   
3. test - Test the SurvyAI installation
   Usage: python -m cli test
   
4. version - Show version information
   Usage: python -m cli version

EXAMPLES:
---------
# Calculate area from a CAD file
python -m cli query "Calculate the area of red boundaries in survey.dwg"

# Extract owner names from a drawing
python -m cli query "Find the owner names in plan.dxf"

# Save results to a file
python -m cli query "Get survey data" --output results.json

# Run with verbose logging
python -m cli query "Convert coordinates" --verbose

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import sys
import json
from pathlib import Path
import threading
import time

# Click - The library for building command-line interfaces
# It provides decorators like @click.command() and @click.option()
import click

# -----------------------------------------------------------------------------
# Windows-friendly UTF-8 output
# -----------------------------------------------------------------------------
# Some Windows terminals default to legacy encodings (e.g., cp1252) which can
# crash on Unicode characters (like "⚠️"). We prefer UTF-8 and replace errors
# to avoid hard failures during long runs.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    # Never fail CLI startup due to output encoding configuration
    pass

# Local imports
from agent import SurvyAIAgent
from utils.logger import setup_logger

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

# Initialize the logger at module level
# This logger will capture all CLI-related messages
logger = setup_logger()

def _run_with_heartbeat(fn, heartbeat_seconds: int = 12, label: str = "Working"):
    """
    Run a blocking function while periodically printing a heartbeat message.
    This prevents the terminal from *appearing* frozen during long operations.
    
    Implementation detail: we run the work in a thread and print heartbeats
    from the main thread (Click output is not reliably thread-safe).
    """
    result_container = {"result": None, "error": None}

    def _runner():
        try:
            result_container["result"] = fn()
        except Exception as e:
            result_container["error"] = e

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()

    start = time.time()
    while worker.is_alive():
        worker.join(timeout=max(1, int(heartbeat_seconds)))
        if worker.is_alive():
            elapsed = int(time.time() - start)
            # Use print() for maximum compatibility; Click can buffer oddly here.
            print(f"… {label} (still running, {elapsed}s elapsed)", flush=True)

    if result_container["error"] is not None:
        raise result_container["error"]
    return result_container["result"]


# ==============================================================================
# INTERACTIVE PROCESSING FUNCTION
# ==============================================================================

def _process_query_interactive(agent, initial_query: str, verbose: bool):
    """
    Process a query in interactive mode with persistent context.
    
    This function:
    1. Maintains a persistent session ID across iterations
    2. Builds conversation history and injects it into each query
    3. Stores conversations immediately after each response
    4. Retrieves context from current session (priority) and past sessions
    5. Handles permission requests and continues conversation
    """
    # Get or create persistent session ID
    session_id = agent.get_session_id()
    agent.set_session_id(session_id)  # Ensure it's set
    
    # Store conversation history for context injection
    conversation_history = []
    
    # Start with the initial query
    base_query = initial_query
    current_query = initial_query
    click.echo("\n" + "=" * 60)
    click.echo("💬 INTERACTIVE MODE")
    click.echo("=" * 60)
    click.echo(f"🔗 Session: {session_id[:8]}...")
    click.echo("Type 'exit' or 'quit' to end the conversation\n")
    
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Build context-aware query with conversation history
        # This ensures the agent remembers what was discussed in this session
        if conversation_history:
            # Inject previous conversation context from this session
            # Format it clearly so the agent understands this is a continuation
            context_parts = []
            context_parts.append("=== CONTINUATION OF PREVIOUS WORK IN THIS SESSION ===")
            context_parts.append("IMPORTANT: The following conversation happened earlier in this same session.")
            context_parts.append("You should continue working with the same files, tools, and context.")
            context_parts.append("")
            
            for i, (q, r) in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
                context_parts.append(f"--- Exchange {i} ---")
                context_parts.append(f"USER: {q}")
                context_parts.append(f"ASSISTANT: {r[:500]}...")  # Show more context
                context_parts.append("")
            
            context_parts.append("=== END OF PREVIOUS CONVERSATION ===")
            context_parts.append("")
            context_parts.append("NOW, the user wants you to continue with this new request:")
            context_summary = "\n".join(context_parts)
            enhanced_query = f"{context_summary}\n\n{current_query}"
        else:
            enhanced_query = current_query
        
        # Process the current query with progress indication
        click.echo(f"\n⏳ Processing query (iteration {iteration}/{max_iterations})...")
        click.echo("   This may take a moment, especially for large documents...")
        
        try:
            result = _run_with_heartbeat(
                lambda: agent.process_query(
                    enhanced_query, 
                    session_id=session_id, 
                    interactive_mode=True
                ),
                heartbeat_seconds=12,
                label="Processing (LLM/tools)"
            )
        except TimeoutError as e:
            click.echo(f"\n❌ Query timed out: {e}", err=True)
            click.echo("\n💡 Suggestions:")
            click.echo("   - The document may be too large. Try using section extraction instead.")
            click.echo("   - Break the query into smaller, more specific requests.")
            click.echo("   - Check the document size first with document_get_resource_estimation()")
            return {
                "success": False,
                "error": str(e),
                "response": f"Query timed out. {e}"
            }
        except Exception as e:
            click.echo(f"\n❌ Error processing query: {e}", err=True)
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"Error: {e}"
            }
        
        if not result.get("success", False):
            error_type = result.get("error", "")
            response_text = result.get("response", "") or ""
            response_lower = response_text.lower()

            # Token limit approval flow
            if error_type in ["token_limit_exceeded", "tpm_rate_limit_hit"]:
                
                # Display the token limit warning
                click.echo("\n" + "=" * 60)
                click.echo("⚠️  TOKEN LIMIT WARNING")
                click.echo("=" * 60)
                click.echo(response_text)
                click.echo("=" * 60)
                
                # Get user approval
                while True:
                    user_response = click.prompt(
                        "\nYour response",
                        type=str,
                        default="yes"
                    ).strip().lower()
                    
                    if user_response in ['exit', 'quit', 'q']:
                        click.echo("\n👋 Ending conversation. Goodbye!")
                        return result
                    
                    if user_response in ['yes', 'y', 'ok', 'okay', 'sure', 'proceed', 'go ahead', 'continue', '']:
                        # User approved - retry with automatic chunking enabled
                        click.echo("✅ Proceeding with automatic chunking...\n")
                        # Modify query to indicate approval (agent will detect this and proceed)
                        current_query = f"{current_query} [USER APPROVED: Proceed with automatic chunking and rate limiting]"
                        break
                    elif user_response in ['no', 'n', 'deny', 'cancel', 'abort']:
                        click.echo("❌ Request cancelled by user.\n")
                        return result
                    else:
                        click.echo("Please respond with 'yes' to continue or 'no' to cancel.")
                        continue
                
                # Continue to next iteration with approved query
                continue

            # Internet permission approval flow (returned as success=False)
            if error_type in ["internet_permission_required"] or "[internet_permission_request]" in response_lower:
                click.echo("\n" + "=" * 60)
                click.echo("⚠️  INTERNET PERMISSION REQUIRED")
                click.echo("=" * 60)
                click.echo(response_text)
                click.echo("=" * 60)

                while True:
                    user_response = click.prompt(
                        "\nYour response",
                        type=str,
                        default="yes"
                    ).strip().lower()

                    if user_response in ['exit', 'quit', 'q']:
                        click.echo("\n👋 Ending conversation. Goodbye!")
                        return result

                    if user_response in ['yes', 'y', 'ok', 'okay', 'sure', 'proceed', 'go ahead', 'continue', '']:
                        click.echo("✅ Internet permission granted. Continuing...\n")
                        current_query = f"{base_query} [INTERNET_PERMISSION_GRANTED]"
                        break
                    elif user_response in ['no', 'n', 'deny', 'cancel', 'abort']:
                        click.echo("❌ Internet permission denied. Continuing without internet.\n")
                        current_query = f"{base_query} [INTERNET_PERMISSION_DENIED]"
                        break
                    else:
                        click.echo("Please respond with 'yes' to allow internet search or 'no' to continue offline.")
                        continue

                continue
            
            # Other errors - return as normal
            return result
        
        response_text = result.get("response", "")
        
        # Store conversation immediately for next iteration's context
        # This ensures the next query can access this conversation
        conversation_history.append((current_query, response_text))
        
        # Display the response
        click.echo("\n" + "=" * 60)
        click.echo("🤖 AGENT RESPONSE:")
        click.echo("=" * 60)
        click.echo(response_text)
        click.echo("=" * 60)
        
        # Show context status
        if result.get('context_retrieved'):
            click.echo("🧠 Context: Retrieved from previous sessions/documents")
        else:
            click.echo("🧠 Context: Using current session history")
        
        # Check if the agent is asking for permission
        # Look for common permission request patterns
        permission_indicators = [
            "may i",
            "can i",
            "permission",
            "proceed",
            "yes/no",
            "grant",
            "allow",
            "would you like",
            "should i"
        ]
        
        response_lower = response_text.lower()
        is_permission_request = any(
            indicator in response_lower 
            for indicator in permission_indicators
        ) and (
            "?" in response_text or 
            "(yes/no)" in response_lower or
            "yes or no" in response_lower
        )
        
        if is_permission_request:
            # Agent is asking for permission - get user input
            click.echo("\n" + "─" * 60)
            click.echo("⚠️  PERMISSION REQUEST DETECTED")
            click.echo("─" * 60)

            # Detect special permission types from agent markers
            # Be liberal: models often phrase this as "search the internet" not "internet search".
            wants_internet = (
                "[internet_permission_request]" in response_lower
                or "internet search" in response_lower
                or ("internet" in response_lower and any(k in response_lower for k in ["search", "browse", "sources", "cite", "citation", "web"]))
            )
            
            while True:
                user_response = click.prompt(
                    "\nYour response",
                    type=str,
                    default="yes"
                ).strip().lower()
                
                if user_response in ['exit', 'quit', 'q']:
                    click.echo("\n👋 Ending conversation. Goodbye!")
                    return result
                
                if user_response in ['yes', 'y', 'ok', 'okay', 'sure', 'proceed', 'go ahead', '']:
                    # User granted permission - continue with the operation
                    click.echo("✅ Permission granted. Continuing...\n")
                    if wants_internet:
                        # For internet permission, re-issue the ORIGINAL query with a permission marker
                        # so the agent proceeds with the intended task (instead of treating the user
                        # response as the new task).
                        current_query = f"{base_query} [INTERNET_PERMISSION_GRANTED]"
                    else:
                        # Continue the conversation by acknowledging permission
                        current_query = "User response: Yes, permission granted. Please proceed with the operation."
                    break
                elif user_response in ['no', 'n', 'deny', 'cancel', 'abort']:
                    # User denied permission
                    click.echo("❌ Permission denied. The agent will suggest alternatives.\n")
                    if wants_internet:
                        current_query = f"{base_query} [INTERNET_PERMISSION_DENIED]"
                    else:
                        current_query = "User response: No, permission denied. Please suggest alternative approaches."
                    break
                else:
                    # User provided a custom response
                    click.echo(f"📝 Your response: {user_response}\n")

                    # If this permission request is about internet usage, interpret common
                    # "option A / retry / proceed" responses as granting internet permission.
                    marker = ""
                    if wants_internet:
                        grantish = any(
                            kw in user_response
                            for kw in ["option a", "retry", "search", "proceed", "go ahead", "continue", "yes", "y", "allow", "granted"]
                        )
                        denyish = any(kw in user_response for kw in ["option b", "no", "n", "deny", "denied", "cancel", "abort", "offline"])
                        if grantish and not denyish:
                            marker = " [INTERNET_PERMISSION_GRANTED]"
                        elif denyish and not grantish:
                            marker = " [INTERNET_PERMISSION_DENIED]"

                    current_query = f"User response: {user_response}{marker}"
                    break
        else:
            # No permission request - check if user wants to continue
            click.echo("\n" + "─" * 60)
            user_input = click.prompt(
                "Continue conversation? (type your next question, or 'exit' to quit)",
                type=str,
                default="exit"
            ).strip()
            
            if user_input.lower() in ['exit', 'quit', 'q', '']:
                click.echo("\n👋 Ending conversation. Goodbye!")
                return result
            
            current_query = user_input
    
    # Max iterations reached
    click.echo("\n⚠️  Maximum iterations reached. Ending conversation.")
    return result


# ==============================================================================
# CLI GROUP
# ==============================================================================

@click.group()
def cli():
    """
    SurvyAI - AI-Powered Assistant for Land Surveyors.
    
    This is the main entry point for the CLI. All subcommands are grouped
    under this function. When you run 'python -m cli', this help text appears.
    
    \b
    QUICK START:
    ------------
    1. Set up your API keys in .env file
    2. Run: python -m cli test
    3. Try: python -m cli query "Your surveying question"
    
    \b
    GETTING HELP:
    -------------
    - python -m cli --help        (this text)
    - python -m cli query --help  (help for query command)
    
    Note: The \\b in docstrings tells Click not to reformat the text,
    preserving our custom formatting.
    """
    pass


# ==============================================================================
# QUERY COMMAND
# ==============================================================================

@cli.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument('query', nargs=-1, required=True)
@click.option(
    '--output', '-o', 
    type=click.Path(), 
    help='Save results to a JSON file at this path'
)
@click.option(
    '--verbose', '-v', 
    is_flag=True, 
    help='Enable detailed debug logging'
)
@click.option(
    '--interactive', '-i',
    is_flag=True,
    help='Enable interactive mode for permission requests and follow-up questions'
)
@click.pass_context
def query(ctx, query, output, verbose, interactive):
    """
    Process a natural language query using the SurvyAI agent.
    
    This command takes your question, sends it to the AI agent, which then
    uses its tools (AutoCAD, Excel processor, etc.) to find the answer.
    
    \b
    ARGUMENTS:
    ----------
    QUERY: Your question or request (can be multiple words, quotes optional)
    
    \b
    OPTIONS:
    --------
    -o, --output FILE    Save the response to a JSON file
    -v, --verbose        Show detailed processing information
    
    \b
    EXAMPLES:
    ---------
    # With quotes (recommended for complex queries)
    python -m cli query "Calculate the total area from survey.dwg"
    
    # Without quotes (works for simple queries)
    python -m cli query Calculate area from survey.dwg
    
    # Save output to file
    python -m cli query "Get owner names" --output results.json
    
    # Verbose mode for debugging
    python -m cli query "Convert coordinates" --verbose
    
    \b
    OPTIONS:
    --------
    -i, --interactive  Enable interactive mode for permission requests and follow-up questions
                       In interactive mode, you can respond to permission requests and continue
                       the conversation. Type 'exit' to end the conversation.
    
    \b
    HOW IT WORKS:
    -------------
    1. Your query is passed to the SurvyAI agent
    2. The agent uses an LLM (Gemini, DeepSeek, Claude, or OpenAI) to understand the query
    3. The agent calls appropriate tools (AutoCAD, Excel, etc.)
    4. Tool results are processed by the LLM
    5. A final answer is returned to you
    
    \b
    INTERACTIVE MODE:
    -----------------
    Use --interactive (-i) flag to enable interactive mode:
    
    python -m cli query "Read coordinates from survey.xlsx" --interactive
    
    In interactive mode:
    - The agent will pause when asking for permission
    - You can respond with: yes, no, or a custom response
    - You can continue the conversation with follow-up questions
    - Type 'exit' or 'quit' to end the conversation
    
    \b
    COMMON QUERIES:
    ---------------
    - "Calculate the area of [boundaries/red lines] in [file.dwg]"
    - "Extract coordinates from [file.xlsx]"
    - "Find owner names in [file.dxf]"
    - "Convert [coordinates] from [system] to [system]"
    - "What are the layers in [file.dwg]?"
    """
    # -------------------------------------------------------------------------
    # Step 1: Prepare the query string
    # -------------------------------------------------------------------------
    # The query comes as a tuple if multiple words without quotes
    # We join them back into a single string
    query_str = " ".join(query) if isinstance(query, tuple) else query
    
    # -------------------------------------------------------------------------
    # Step 2: Configure logging verbosity
    # -------------------------------------------------------------------------
    try:
        if verbose:
            # DEBUG level shows all internal processing details
            logger.setLevel("DEBUG")
            click.echo("Verbose mode enabled - showing detailed logs")
        
        # Show what we're processing
        click.echo(f"\n🔍 Processing query: {query_str}")
        click.echo("⏳ Initializing SurvyAI agent...")
        
        # -------------------------------------------------------------------------
        # Step 3: Initialize the agent
        # -------------------------------------------------------------------------
        # This creates the LangGraph, loads tools, and connects to LLMs
        # May take a few seconds on first run
        agent = SurvyAIAgent()
        
        click.echo("🤖 Agent ready. Processing your query...")
        
        # -------------------------------------------------------------------------
        # Step 4: Process the query (interactive or single-shot)
        # -------------------------------------------------------------------------
        if interactive:
            result = _process_query_interactive(agent, query_str, verbose)
        else:
            # Single-shot processing (original behavior)
            result = _run_with_heartbeat(
                lambda: agent.process_query(query_str),
                heartbeat_seconds=12,
                label="Processing (LLM/tools)"
            )
        
        # -------------------------------------------------------------------------
        # Step 5: Display results
        # -------------------------------------------------------------------------
        if result.get("success", False):
            # Success - show the response
            click.echo("\n" + "=" * 60)
            click.echo("✅ RESULT:")
            click.echo("=" * 60)
            
            response_text = result.get("response", "No response generated")
            click.echo(response_text)
            
            click.echo("=" * 60)
            
            # Show context retrieval status
            if result.get('context_retrieved'):
                click.echo("\n🧠 Context: Retrieved from previous sessions/documents")
            else:
                click.echo("\n🧠 Context: No relevant previous context found")
            
            # Show session ID (truncated for readability)
            session_id = result.get('session_id', 'unknown')
            if session_id and session_id != 'unknown':
                click.echo(f"🔗 Session: {session_id[:8]}...")
            
            # Display model information with complexity if available
            model_name = result.get('model_name', '')
            complexity = result.get('complexity', '')
            if model_name:
                model_display = f"{result.get('llm_used', 'unknown')} ({model_name})"
                if complexity:
                    complexity_labels = {
                        'simple': 'simple',
                        'average': 'average',
                        'complex': 'complex'
                    }
                    model_display += f" [{complexity_labels.get(complexity, complexity)} task]"
                click.echo(f"📊 Model: {model_display}")
            else:
                click.echo(f"📊 LLM used: {result.get('llm_used', 'unknown')}")
            
            # Optionally save to file
            if output:
                output_path = Path(output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Save full result as JSON
                    json.dump(result, f, indent=2, ensure_ascii=False)
                click.echo(f"💾 Results saved to: {output_path}")
        else:
            # Error occurred
            click.echo(f"\n❌ Error: {result.get('error', 'Unknown error')}", err=True)
            
            # Still show any partial response
            if result.get("response"):
                click.echo(f"Response: {result.get('response')}", err=True)
            
            raise click.Abort()
    
    except click.Abort:
        # Re-raise Click's Abort exception (for --help, etc.)
        raise
        
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error in CLI: {e}", exc_info=True)
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


# ==============================================================================
# LIST-MODELS COMMAND
# ==============================================================================

@cli.command("list-models")
def list_models():
    """
    List available Gemini models from Google's API.
    
    This command queries the Google Generative Language API to show which
    Gemini models are available for use. Useful for troubleshooting model
    availability issues.
    
    \b
    OUTPUT:
    -------
    Shows two categories of models:
    1. Chat-capable models - Can be used for conversation (SurvyAI uses these)
    2. Other models - Embedding, image, and other specialized models
    
    \b
    REQUIREMENTS:
    -------------
    - Valid GOOGLE_API_KEY in your .env file
    - Internet connection to reach Google's API
    
    \b
    EXAMPLE:
    --------
    $ python -m cli list-models
    
    === Chat-capable Gemini models (5) ===
      ✓ gemini-pro-latest
      ✓ gemini-2.0-flash
      ✓ gemini-2.5-flash
      ...
    """
    # -------------------------------------------------------------------------
    # Step 1: Load settings to get API key
    # -------------------------------------------------------------------------
    try:
        from config import get_settings
        settings = get_settings()
    except Exception as exc:
        click.echo(f"❌ Unable to load settings: {exc}", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Step 2: Validate API key
    # -------------------------------------------------------------------------
    api_key = getattr(settings, "google_api_key", "") or ""
    if not api_key or api_key == "your_google_api_key_here":
        click.echo("❌ Valid GOOGLE_API_KEY is required to list models.", err=True)
        click.echo("   Set it in your .env file or environment.", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Step 3: Make API request
    # -------------------------------------------------------------------------
    try:
        import requests
    except ImportError:
        click.echo("❌ requests library not installed.", err=True)
        click.echo("   Install with: pip install requests", err=True)
        raise click.Abort()
    
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key}
    
    click.echo("📡 Fetching Gemini v1beta models...")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response else "unknown"
        click.echo(f"❌ HTTP {status} while fetching models: {http_err}", err=True)
        raise click.Abort()
    except Exception as exc:
        click.echo(f"❌ Failed to fetch models: {exc}", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Step 4: Parse and categorize models
    # -------------------------------------------------------------------------
    models = response.json().get("models", [])
    if not models:
        click.echo("⚠️  Request succeeded but no models were returned.")
        return
    
    # Separate chat-capable models from others
    chat_models = []
    other_models = []
    
    for model in models:
        name = model.get("name", "")
        model_id = name.split("/")[-1] if name else ""
        methods = model.get("supportedGenerationMethods", [])
        
        # A model is "chat-capable" if:
        # - It starts with "gemini"
        # - It's not an embedding, image, or TTS model
        # - It supports generateContent method
        is_chat = (
            model_id.startswith("gemini") and
            not any(x in model_id.lower() for x in ["embedding", "image", "tts"]) and
            ("generateContent" in methods if methods else True)
        )
        
        if is_chat:
            chat_models.append(model)
        else:
            other_models.append(model)
    
    # -------------------------------------------------------------------------
    # Step 5: Display results
    # -------------------------------------------------------------------------
    click.echo(f"\n=== Chat-capable Gemini models ({len(chat_models)}) ===")
    for model in chat_models:
        name = model.get("name", "unknown").split("/")[-1]
        click.echo(f"  ✓ {name}")
    
    click.echo(f"\n=== Other models ({len(other_models)}) ===")
    # Only show first 10 to avoid cluttering output
    for model in other_models[:10]:
        name = model.get("name", "unknown").split("/")[-1]
        click.echo(f"    {name}")
    
    if len(other_models) > 10:
        click.echo(f"    ... and {len(other_models) - 10} more")


# ==============================================================================
# MEMORY COMMAND
# ==============================================================================

@cli.command()
@click.option(
    '--clear', '-c',
    is_flag=True,
    help='Clear all stored memories (conversations, documents)'
)
@click.option(
    '--clear-conversations',
    is_flag=True,
    help='Clear only conversation history'
)
def memory(clear, clear_conversations):
    """
    View and manage the agent's memory (vector store).
    
    This command shows what the AI agent remembers from previous sessions,
    including stored conversations, documents, drawings, and coordinates.
    
    \\b
    OPTIONS:
    --------
    --clear              Clear ALL stored memories
    --clear-conversations Clear only conversation history
    
    \\b
    EXAMPLES:
    ---------
    # View memory statistics
    python -m cli memory
    
    # Clear conversation history only
    python -m cli memory --clear-conversations
    
    # Clear everything
    python -m cli memory --clear
    """
    try:
        from config import get_settings
        from tools import VectorStore
        
        settings = get_settings()
        
        # Check if vector store is enabled
        if not getattr(settings, 'vector_store_enabled', True):
            click.echo("⚠️  Vector store is disabled in configuration.")
            click.echo("   Set VECTOR_STORE_ENABLED=True in .env to enable.")
            return
        
        click.echo("🧠 Loading vector store...")
        
        # Initialize vector store
        store = VectorStore(
            persist_directory=getattr(settings, 'vector_store_path', None),
            embedding_provider=getattr(settings, 'embedding_provider', 'local'),
        )
        
        # Handle clear operations
        if clear:
            if click.confirm("⚠️  This will delete ALL stored memories. Are you sure?"):
                store.clear_all_collections()
                click.echo("✅ All memories cleared successfully.")
            else:
                click.echo("❌ Operation cancelled.")
            return
        
        if clear_conversations:
            if click.confirm("⚠️  This will delete all conversation history. Are you sure?"):
                store.clear_collection("conversations")
                click.echo("✅ Conversation history cleared.")
            else:
                click.echo("❌ Operation cancelled.")
            return
        
        # Get and display statistics
        stats = store.get_stats()
        
        click.echo("\n" + "=" * 60)
        click.echo("🧠 SURVYAI MEMORY STATUS")
        click.echo("=" * 60)
        
        click.echo(f"\n📁 Storage Location: {stats['persist_directory']}")
        click.echo(f"🔧 Embedding Provider: {stats['embedding_provider']}")
        click.echo(f"📐 Embedding Model: {stats['embedding_model']}")
        click.echo(f"📊 Embedding Dimension: {stats['embedding_dimension']}")
        
        click.echo(f"\n📚 Collections:")
        total = 0
        for name, count in stats['collections'].items():
            icon = {
                'conversations': '💬',
                'documents': '📄',
                'drawings': '📐',
                'coordinates': '📍'
            }.get(name, '📁')
            click.echo(f"   {icon} {name}: {count} items")
            total += count
        
        click.echo(f"\n📈 Total stored items: {total}")
        
        # Show context retrieval settings
        click.echo(f"\n⚙️  Context Settings:")
        click.echo(f"   Auto-retrieve context: {getattr(settings, 'auto_context_retrieval', True)}")
        click.echo(f"   Auto-store conversations: {getattr(settings, 'auto_store_conversations', True)}")
        click.echo(f"   Context top-k: {getattr(settings, 'context_retrieval_top_k', 5)}")
        click.echo(f"   Score threshold: {getattr(settings, 'context_score_threshold', 0.3)}")
        
        click.echo("\n" + "=" * 60)
        
        if total == 0:
            click.echo("\n💡 Tip: Start asking questions to build up context memory!")
        else:
            click.echo(f"\n💡 Your agent has {total} memories to draw from for context.")
        
    except Exception as e:
        click.echo(f"❌ Error accessing vector store: {e}", err=True)
        raise click.Abort()


# ==============================================================================
# VERSION COMMAND
# ==============================================================================

@cli.command()
def version():
    """
    Display SurvyAI version information.
    
    Shows the current version of SurvyAI as defined in __init__.py.
    
    \b
    EXAMPLE:
    --------
    $ python -m cli version
    SurvyAI version 0.1.0
    """
    import importlib.util
    import os
    
    # Load version from __init__.py
    init_path = os.path.join(os.path.dirname(__file__), "__init__.py")
    spec = importlib.util.spec_from_file_location("survyai_init", init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    click.echo(f"SurvyAI version {module.__version__}")


# ==============================================================================
# TEST COMMAND
# ==============================================================================

@cli.command()
def test():
    """
    Test SurvyAI installation and configuration.
    
    This command performs a series of checks to verify that SurvyAI is
    properly installed and configured:
    
    \b
    CHECKS PERFORMED:
    -----------------
    1. Core module imports (agent, config)
    2. Configuration loading
    3. API key validation
    4. Tool module imports
    
    \b
    EXAMPLE:
    --------
    $ python -m cli test
    Testing SurvyAI installation...
    [OK] Core modules imported successfully
    [OK] Configuration loaded successfully
    [OK] Google Gemini API key configured
    ...
    
    \b
    TROUBLESHOOTING:
    ----------------
    - If you see import errors, run: pip install -r requirements.txt
    - If API keys are missing, create a .env file with your keys
    - For AutoCAD errors, ensure AutoCAD is installed and running
    """
    click.echo("🔧 Testing SurvyAI installation...")
    click.echo("")
    
    # -------------------------------------------------------------------------
    # Test 1: Core module imports
    # -------------------------------------------------------------------------
    click.echo("1. Testing core module imports...")
    try:
        from agent import SurvyAIAgent
        from config import get_settings
        click.echo("   [OK] Core modules imported successfully")
    except Exception as e:
        click.echo(f"   [ERROR] Import error: {e}", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Test 2: Configuration loading
    # -------------------------------------------------------------------------
    click.echo("\n2. Testing configuration...")
    try:
        settings = get_settings()
        click.echo("   [OK] Configuration loaded successfully")
        
        # Check API keys
        api_keys_configured = False
        
        if settings.deepseek_api_key and settings.deepseek_api_key.strip():
            click.echo("   [OK] DeepSeek API key configured")
            api_keys_configured = True
        else:
            click.echo("   [WARNING] DeepSeek API key not configured", err=True)
        
        if settings.google_api_key and settings.google_api_key.strip():
            click.echo("   [OK] Google Gemini API key configured")
            api_keys_configured = True
        else:
            click.echo("   [WARNING] Google Gemini API key not configured", err=True)
        
        if settings.anthropic_api_key and settings.anthropic_api_key.strip():
            click.echo("   [OK] Anthropic Claude API key configured")
            api_keys_configured = True
        else:
            click.echo("   [WARNING] Anthropic Claude API key not configured", err=True)
        
        if settings.openai_api_key and settings.openai_api_key.strip():
            click.echo("   [OK] OpenAI API key configured")
            api_keys_configured = True
        else:
            click.echo("   [WARNING] OpenAI API key not configured", err=True)
        
        if not api_keys_configured:
            click.echo("\n   [INFO] At least one API key is required to use SurvyAI", err=True)
            click.echo("          Configure API keys in your .env file", err=True)
    
    except Exception as e:
        click.echo(f"   [ERROR] Configuration error: {e}", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Test 3: Tool imports
    # -------------------------------------------------------------------------
    click.echo("\n3. Testing tool imports...")
    try:
        from tools import ExcelProcessor, DocumentProcessor, AutoCADProcessor
        click.echo("   [OK] Tool modules imported successfully")
        click.echo("        - ExcelProcessor ✓")
        click.echo("        - DocumentProcessor ✓")
        click.echo("        - AutoCADProcessor ✓")
    except ImportError as e:
        click.echo(f"   [WARNING] Some tools not available: {e}", err=True)
    except Exception as e:
        click.echo(f"   [ERROR] Tool import error: {e}", err=True)
        raise click.Abort()
    
    # -------------------------------------------------------------------------
    # Test 4: AutoCAD connection (optional)
    # -------------------------------------------------------------------------
    click.echo("\n4. Testing AutoCAD connection (optional)...")
    try:
        from tools import AutoCADProcessor
        acad = AutoCADProcessor(auto_connect=False)
        if acad.connect():
            click.echo("   [OK] AutoCAD connection successful")
        else:
            click.echo("   [INFO] AutoCAD not running (this is okay)")
    except Exception as e:
        click.echo(f"   [INFO] AutoCAD not available: {e}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    click.echo("\n" + "=" * 50)
    click.echo("✅ SurvyAI installation test completed!")
    click.echo("=" * 50)
    click.echo("\nNext steps:")
    click.echo("  1. Ensure your API keys are set in .env")
    click.echo("  2. Try: python -m cli query \"What is surveying?\"")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Main entry point for the CLI.
    
    This function handles a convenience feature: if the user runs
    'python -m cli "Some query"' without the 'query' subcommand,
    we automatically insert it for them.
    
    This makes usage more intuitive for the most common operation.
    """
    # List of known subcommands
    known_commands = ['query', 'test', 'version', 'list-models', 'memory', '--help', '-h']
    
    # If the first argument isn't a known command, assume it's a query
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        # But make sure it's not an option for a subcommand
        if sys.argv[1] not in ['-o', '--output', '-v', '--verbose']:
            # Insert 'query' as the subcommand
            sys.argv.insert(1, 'query')
    
    # Run the CLI
    cli()


# ==============================================================================
# SCRIPT ENTRY
# ==============================================================================

if __name__ == '__main__':
    main()
