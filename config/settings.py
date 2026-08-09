"""
================================================================================
SurvyAI Configuration Management
================================================================================

This module handles all configuration settings for SurvyAI using Pydantic
for validation and automatic environment variable loading.

CONFIGURATION SOURCES:
----------------------
Settings are loaded from (in order of priority):
1. Environment variables (highest priority)
2. .env file in the project root
3. Default values (lowest priority)

HOW TO CONFIGURE:
-----------------
Option 1: Create a .env file in the project root:
    ```
    GOOGLE_API_KEY=your_api_key_here
    DEEPSEEK_API_KEY=your_deepseek_key
    GEMINI_MODEL=gemini-pro-latest
    PRIMARY_LLM=gemini
    ```

Option 2: Set environment variables:
    - Windows: set GOOGLE_API_KEY=your_api_key_here
    - Linux/Mac: export GOOGLE_API_KEY=your_api_key_here

REQUIRED SETTINGS:
------------------
At minimum, you need ONE of these API keys:
- GOOGLE_API_KEY: For using Gemini models
- DEEPSEEK_API_KEY: For using DeepSeek models

SINGLETON PATTERN:
------------------
This module uses a singleton pattern for settings. The first call to
get_settings() creates the Settings instance, and all subsequent calls
return the same instance. This ensures consistent configuration across
the application.

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
from typing import Literal

# Pydantic for settings validation
# pydantic-settings provides BaseSettings for env var loading
from pydantic_settings import BaseSettings
from pydantic import Field

from runtime_paths import is_frozen_app, user_data_path


PRODUCTION_CLOUD_API_BASE_URL = "https://survyai-api.onrender.com"


def _default_log_file() -> str:
    if is_frozen_app():
        path = user_data_path("logs", "survyai.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return "survyai.log"


# ==============================================================================
# SETTINGS CLASS
# ==============================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    This class uses Pydantic's BaseSettings to automatically load values
    from environment variables and .env files. Each field can have:
    - A type annotation (str, int, float, etc.)
    - A default value
    - Field metadata (description, env var name, etc.)
    
    Type Validation:
    ----------------
    Pydantic automatically validates types. If AGENT_TEMPERATURE is set
    to "abc" instead of a number, it will raise a validation error.
    
    Environment Variable Mapping:
    -----------------------------
    By default, the env var name is the UPPERCASE version of the field name.
    For example:
    - google_api_key → GOOGLE_API_KEY
    - primary_llm → PRIMARY_LLM
    
    You can override this with the 'env' parameter in Field().
    
    Example Usage:
    --------------
    ```python
    from config import get_settings
    
    settings = get_settings()
    print(f"Using model: {settings.gemini_model}")
    print(f"Primary LLM: {settings.primary_llm}")
    ```
    """
    
    # ==========================================================================
    # DeepSeek API Configuration
    # ==========================================================================
    # DeepSeek is a cost-effective LLM provider with OpenAI-compatible API.
    # It serves as our fallback LLM if Gemini is unavailable.
    
    deepseek_api_key: str = Field(
        default="",  # Optional - user will add later
        env="DEEPSEEK_API_KEY",
        description="API key for DeepSeek. Get from https://platform.deepseek.com"
    )
    
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        env="DEEPSEEK_BASE_URL",
        description="Base URL for DeepSeek API. Usually no need to change."
    )
    
    # ==========================================================================
    # Google Gemini API Configuration
    # ==========================================================================
    # Gemini is Google's latest AI model family. We use it as the primary LLM.
    # It offers good performance for reasoning and tool use.
    
    google_api_key: str = Field(
        default="",  # Optional - user will add later
        env="GOOGLE_API_KEY",
        description="API key for Google Gemini. Get from https://makersuite.google.com"
    )
    
    gemini_model: str = Field(
        default="gemini-2.0-flash",  # Changed from gemini-pro-latest
        env="GEMINI_MODEL",
        description="Gemini model name. Use 'gemini-2.0-flash' for best free tier limits."
    )
    
    # ==========================================================================
    # Anthropic Claude API Configuration
    # ==========================================================================
    # Claude is Anthropic's advanced AI model family. Supports Opus, Sonnet, and Haiku.
    # Excellent for complex reasoning and tool use tasks.
    
    anthropic_api_key: str = Field(
        default="",  # Optional - user will add later
        env="ANTHROPIC_API_KEY",
        description="API key for Anthropic Claude. Get from https://console.anthropic.com"
    )
    
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        env="CLAUDE_MODEL",
        description=(
            "Claude model name. Options:\n"
            "  - claude-3-5-sonnet-20241022: Latest Sonnet (recommended, balanced)\n"
            "  - claude-3-opus-20240229: Opus model (most capable, best quality)\n"
            "  - claude-3-5-haiku-20241022: Haiku model (fastest, most cost-effective)\n"
            "  - claude-3-haiku-20240307: Original Haiku model (alternative)"
        )
    )
    
    # ==========================================================================
    # OpenAI API Configuration
    # ==========================================================================
    # OpenAI provides GPT-4, GPT-4o, GPT-4o-Turbo, and GPT-5 models.
    # Excellent performance for complex tasks and tool use.
    
    openai_api_key: str = Field(
        default="",  # Optional - user will add later
        env="OPENAI_API_KEY",
        description="API key for OpenAI. Get from https://platform.openai.com/api-keys"
    )
    
    openai_model: str = Field(
        default="gpt-5.6-terra",
        env="OPENAI_MODEL",
        description=(
            "OpenAI model name (legacy - used as fallback if tiered models not set).\n"
            "For tiered model selection, use OPENAI_MODEL_NANO, OPENAI_MODEL_MINI, OPENAI_MODEL_COMPLEX instead.\n"
            "Catalog includes gpt-5.6-sol/terra/luna, gpt-5.5, gpt-5.4(-mini/-nano), gpt-5*, gpt-4.1*, gpt-4o*."
        )
    )
    
    # Tiered OpenAI models for intelligent complexity-based selection
    openai_model_nano: str = Field(
        default="gpt-5.6-luna",
        env="OPENAI_MODEL_NANO",
        description=(
            "OpenAI model for simple tasks (lookups, short Q&A).\n"
            "Default: gpt-5.6-luna (cost-efficient high-volume). Fallbacks: gpt-5.4-nano, gpt-4o-mini."
        )
    )
    
    openai_model_mini: str = Field(
        default="gpt-5.6-terra",
        env="OPENAI_MODEL_MINI",
        description=(
            "OpenAI model for average tasks (CRS convert, GIS tool orchestration, drafting).\n"
            "Default: gpt-5.6-terra (balance of intelligence and cost). Fallbacks: gpt-5.4-mini, gpt-5-mini."
        )
    )
    
    openai_model_complex: str = Field(
        default="gpt-5.6-sol",
        env="OPENAI_MODEL_COMPLEX",
        description=(
            "OpenAI model for complex reasoning / hard agentic work.\n"
            "Default: gpt-5.6-sol (flagship). Fallbacks on quota: gpt-5.5, gpt-5.6-terra, gpt-5.4, gpt-5.4-mini."
        )
    )
    
    enable_tiered_models: bool = Field(
        default=True,
        env="ENABLE_TIERED_MODELS",
        description=(
            "Enable intelligent model selection based on task complexity for paid "
            "providers (OpenAI, Claude, Gemini, DeepSeek).\n"
            "If False, each provider uses its single legacy model setting for all tasks."
        )
    )
    
    # ==========================================================================
    # Agent Configuration
    # ==========================================================================
    # These settings control how the AI agent behaves.
    
    agent_temperature: float = Field(
        default=0.3,
        env="AGENT_TEMPERATURE",
        description=(
            "Controls randomness in LLM responses (0.0 to 1.0).\n"
            "  - 0.0: Very deterministic, same input → same output\n"
            "  - 0.3: Slight variation, good balance (default)\n"
            "  - 1.0: Very creative/random, less predictable\n"
            "For surveying tasks, lower values are usually better."
        )
    )
    
    agent_max_tokens: int = Field(
        default=16384,
        env="AGENT_MAX_TOKENS",
        description=(
            "Maximum tokens in LLM response (output token budget per completion).\n"
            "This value is automatically clamped to each model's real API limit before every call,\n"
            "so setting it higher than a model supports causes a harmless INFO-level log, not a warning.\n"
            "16384 is the practical ceiling for GPT-4o / GPT-5-mini; GPT-5.1 supports up to 128000.\n"
            "Typical useful range: 4000–16384 for most tasks; the 'complex' tier model (gpt-5.1) will\n"
            "automatically use its own higher limit without needing to raise this value."
        )
    )

    agent_config_path: str = Field(
        default="",
        env="AGENT_CONFIG_PATH",
        description=(
            "Optional local JSON config path for runtime agent behavior (prompt/model overrides).\n"
            "If empty, SurvyAI uses agent/agent_config.json when present."
        ),
    )

    agent_cloud_config_json: str = Field(
        default="",
        env="AGENT_CLOUD_CONFIG_JSON",
        description=(
            "Authenticated cloud-delivered JSON payload for runtime agent config.\n"
            "Desktop injects this from /v1/bootstrap; when blank, SurvyAI falls back to local config."
        ),
    )
    
    agent_query_timeout: int = Field(
        default=900,
        env="AGENT_QUERY_TIMEOUT",
        description=(
            "Maximum time (in seconds) for a single query to complete.\n"
            "Default: 900 seconds (15 minutes).\n"
            "If a query takes longer, it will timeout with an error message.\n"
            "Increase this for very large documents or complex multi-step tasks."
        )
    )

    llm_invoke_timeout_seconds: int = Field(
        default=180,
        env="LLM_INVOKE_TIMEOUT_SECONDS",
        description=(
            "Per-call timeout (seconds) for a single LLM invoke inside the agent graph.\n"
            "Default: 180. File-driven workflows with many tools need more than 60s for the first step."
        ),
    )
    
    agent_max_iterations: int = Field(
        default=20,
        env="AGENT_MAX_ITERATIONS",
        description=(
            "Maximum number of agent-tool iterations per query.\n"
            "Default: 20 iterations (agent → tools → agent → tools...).\n"
            "Prevents infinite loops. Increase for very complex multi-step tasks."
        )
    )
    
    primary_llm: Literal["deepseek", "gemini", "claude", "openai", "ollama"] = Field(
        default="ollama",
        env="PRIMARY_LLM",
        description=(
            "Which LLM to use as primary. Options:\n"
            "  - openai: Use OpenAI (GPT-4/4o/5)\n"
            "  - claude: Use Anthropic Claude (Opus/Sonnet/Haiku)\n"
            "  - gemini: Use Google Gemini\n"
            "  - deepseek: Use DeepSeek\n"
            "  - ollama: Use local Ollama models (offline-capable; default for installed desktop builds)"
        )
    )
    
    fallback_llm: Literal["deepseek", "gemini", "claude", "openai", "ollama"] = Field(
        default="ollama",
        env="FALLBACK_LLM",
        description="Which LLM to use if primary fails. Same options as PRIMARY_LLM."
    )

    # ==========================================================================
    # Ollama (local models) Configuration
    # ==========================================================================
    # Ollama runs models locally (no internet required once models are pulled).
    # Default daemon listens on http://localhost:11434

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL",
        description="Base URL for a running Ollama server (default: http://localhost:11434).",
    )

    ollama_model: str = Field(
        default="llama3.2:1b",
        env="OLLAMA_MODEL",
        description=(
            "Default local Ollama model to use (e.g. llama3.2:1b, qwen2.5:7b). "
            "You must pull the model in Ollama at least once."
        ),
    )

    # ==========================================================================
    # Performance controls (desktop UX toggles can inject these)
    # ==========================================================================

    fast_mode_non_file_prompts: bool = Field(
        default=True,
        env="FAST_MODE_NON_FILE_PROMPTS",
        description=(
            "If True, generic non-file prompts bypass the full agent graph/tool planning and run a single LLM call. "
            "File/tool workflows (CAD, documents, ArcGIS, etc.) are unaffected."
        ),
    )

    store_generic_conversations: bool = Field(
        default=False,
        env="STORE_GENERIC_CONVERSATIONS",
        description=(
            "If True, store generic Q&A into the local vector store. "
            "If False (recommended), only store file/tool workflows and explicitly memory-referencing prompts."
        ),
    )

    ollama_num_predict: int = Field(
        default=512,
        env="OLLAMA_NUM_PREDICT",
        description=(
            "For Ollama: cap the maximum predicted tokens per response. "
            "Lower values are faster and reduce runaway latency on CPU machines."
        ),
    )

    fast_mode_max_tokens: int = Field(
        default=700,
        env="FAST_MODE_MAX_TOKENS",
        description=(
            "Output token cap used in Fast mode across providers. "
            "Lower values respond faster; increase for longer answers."
        ),
    )
    
    disable_gemini_fallback: bool = Field(
        default=False,
        env="DISABLE_GEMINI_FALLBACK",
        description=(
            "If True, prevents fallback to Gemini when primary LLM fails. "
            "Useful when you only want to use GPT models and have Gemini quota issues. "
            "Default: False (allows Gemini fallback)."
        )
    )
    
    # ==========================================================================
    # ArcGIS Pro Configuration
    # ==========================================================================
    # ArcGIS Pro is professional GIS software for advanced spatial analysis.
    # SurvyAI can create projects, set coordinate systems, and perform analysis.
    
    arcgis_pro_path: str = Field(
        default="",
        env="ARCGIS_PRO_PATH",
        description=(
            "Path to ArcGIS Pro installation (optional).\n"
            "If empty, SurvyAI will auto-detect the installation.\n"
            "Example: C:\\Program Files\\ArcGIS\\Pro"
        )
    )
    
    arcgis_default_project_path: str = Field(
        default="",
        env="ARCGIS_DEFAULT_PROJECT_PATH",
        description=(
            "Default directory for saving ArcGIS Pro projects.\n"
            "If empty, uses Documents\\ArcGIS\\Projects"
        )
    )
    
    arcgis_default_coordinate_system: str = Field(
        default="WGS84",
        env="ARCGIS_DEFAULT_COORDINATE_SYSTEM",
        description=(
            "Default coordinate system for new projects.\n"
            "Examples: WGS84, UTM Zone 32N, EPSG:4326, 32632"
        )
    )

    arcgis_ui_execution_timeout: int = Field(
        default=1800,
        env="ARCGIS_UI_EXECUTION_TIMEOUT",
        description=(
            "Maximum time (in seconds) to wait for a live ArcGIS Pro UI script run to finish.\n"
            "Default: 1800 seconds (30 minutes).\n"
            "Used for visible ArcGIS workflows such as IDW, CutFill, raster generation, and exports."
        )
    )

    arcgis_ui_bootstrap_timeout_seconds: int = Field(
        default=55,
        env="ARCGIS_UI_BOOTSTRAP_TIMEOUT_SECONDS",
        description=(
            "Seconds allowed for ArcGIS Pro Python Window automation (SendKeys) to show a 'running' status.\n"
            "Increase on slow PCs if logs often show UI runner did not start (before propy fallback)."
        ),
    )

    arcgis_generated_execution_mode: Literal["auto", "live_ui_only", "propy_only"] = Field(
        default="auto",
        env="ARCGIS_GENERATED_EXECUTION_MODE",
        description=(
            "How SurvyAI runs dynamically generated arcpy when execute_automatically is True.\n"
            "- auto (default): run deterministically via propy.bat, then open/finalize ArcGIS Pro after outputs are ready.\n"
            "- live_ui_only: require ArcGIS Pro's live Python Window (uses UI automation; least reliable, only for explicit live execution needs).\n"
            "- propy_only: always use propy.bat (same execution style as auto, but explicitly forbids live UI execution)."
        ),
    )

    arcgis_generated_code_live_ui_only: bool = Field(
        default=False,
        env="ARCGIS_GENERATED_CODE_LIVE_UI_ONLY",
        description=(
            "Legacy flag: if True, forces execution mode to live_ui_only (same as ARCGIS_GENERATED_EXECUTION_MODE=live_ui_only).\n"
            "Prefer arcgis_generated_execution_mode instead."
        ),
    )

    arcgis_propy_timeout_seconds: int = Field(
        default=1800,
        env="ARCGIS_PROPY_TIMEOUT_SECONDS",
        description=(
            "Subprocess timeout (seconds) for propy.bat runs of generated scripts.\n"
            "Default matches arcgis_ui_execution_timeout (30 minutes) for IDW/CutFill-scale work."
        ),
    )

    arcgis_launch_verify_seconds: int = Field(
        default=8,
        env="ARCGIS_LAUNCH_VERIFY_SECONDS",
        description=(
            "Seconds to watch a freshly launched ArcGIS Pro process for immediate exit/crash.\n"
            "If ArcGIS Pro dies during this startup window, SurvyAI treats the launch as unstable and can retry more safely."
        ),
    )

    arcgis_launch_retry_without_project: bool = Field(
        default=True,
        env="ARCGIS_LAUNCH_RETRY_WITHOUT_PROJECT",
        description=(
            "If opening a specific ArcGIS Pro project appears to crash immediately, retry by launching ArcGIS Pro without that project.\n"
            "This keeps ArcGIS Pro usable for review while preserving the finished outputs on disk."
        ),
    )
    
    # ==========================================================================
    # Blue Marble Geographic Calculator Configuration
    # ==========================================================================
    # Blue Marble GeoCalc is a professional coordinate conversion tool.
    # If not installed, we fall back to pyproj for conversions.
    
    geographic_calculator_cmd_path: str = Field(
        default="",
        env="GEOGRAPHIC_CALCULATOR_CMD_PATH",
        description=(
            "Path to GeographicCalculatorCMD.exe.\n"
            "If not set, the system will auto-detect the installation."
        )
    )
    
    blue_marble_path: str = Field(
        default="",
        env="BLUE_MARBLE_PATH",
        description=(
            "Path to Blue Marble Geographic Calculator (optional).\n"
            "If empty, pyproj will be used for coordinate conversions."
        )
    )
    
    # ==========================================================================
    # Vector Database Configuration  (PostgreSQL + pgvector backend)
    # ==========================================================================
    # The desktop agent stores embeddings, conversation history, CAD entities,
    # and survey coordinates in PostgreSQL using pgvector (ANN) and PostGIS
    # (geospatial queries).  Set VECTOR_DB_URL to your PostgreSQL connection
    # string.  For local dev you can run the bundled docker-compose.yml which
    # starts a postgres+postgis+pgvector container on port 5432.

    vector_store_enabled: bool = Field(
        default=True,
        env="VECTOR_STORE_ENABLED",
        description="Enable/disable the vector database for semantic search.",
    )

    vector_db_url: str = Field(
        default="",
        env="VECTOR_DB_URL",
        description=(
            "PostgreSQL connection URL for the vector store.\n"
            "Format: postgresql://user:password@host:5432/dbname\n"
            "Falls back to DATABASE_URL if empty."
        ),
    )

    # Legacy field kept for back-compat; no longer used as a directory path
    vector_store_path: str = Field(
        default=".survyai_vectordb",
        env="VECTOR_STORE_PATH",
        description="Deprecated (ChromaDB era).  Kept for backward compatibility.",
    )

    embedding_provider: Literal["local", "openai"] = Field(
        default="local",
        env="EMBEDDING_PROVIDER",
        description=(
            "Which embedding provider to use:\n"
            "  - local: Sentence Transformers (free, offline capable) – Default\n"
            "  - openai: OpenAI embeddings (higher quality, requires API key)"
        ),
    )

    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="LOCAL_EMBEDDING_MODEL",
        description=(
            "Local embedding model from Sentence Transformers.\n"
            "Options: all-MiniLM-L6-v2 (384-dim, fast), "
            "all-mpnet-base-v2 (768-dim, better quality)"
        ),
    )

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
        description=(
            "OpenAI embedding model. Options:\n"
            "  - text-embedding-3-small: 1536-dim, cost-effective (default)\n"
            "  - text-embedding-3-large: 3072-dim, higher quality"
        ),
    )

    vector_search_mode: Literal["semantic", "hybrid", "keyword"] = Field(
        default="hybrid",
        env="VECTOR_SEARCH_MODE",
        description=(
            "Retrieval strategy for the RAG pipeline:\n"
            "  - semantic: pure cosine-similarity ANN (fast)\n"
            "  - hybrid: cosine ANN + BM25/ts_rank fused with RRF (best recall) – Default\n"
            "  - keyword: full-text search only (no embeddings required)"
        ),
    )

    vector_hybrid_alpha: float = Field(
        default=0.6,
        env="VECTOR_HYBRID_ALPHA",
        ge=0.0,
        le=1.0,
        description=(
            "Balance between semantic and keyword in hybrid search.\n"
            "0.0 = full keyword, 1.0 = full semantic, 0.6 = default."
        ),
    )
    
    # ==========================================================================
    # Web Research / Internet Search Configuration
    # ==========================================================================
    # SurvyAI's internet research uses a multi-stage pipeline (query rewriting →
    # multi-source retrieval → trust + relevance re-ranking → page reading →
    # evidence pack + confidence). It works KEY-FREE by default (DuckDuckGo HTML +
    # Wikipedia). For higher-quality ranked results, configure ONE search API key
    # below; the pipeline auto-detects and prefers it, falling back to key-free
    # providers if the call fails. These are read from the environment directly by
    # utils.internet, so they are optional here and primarily for documentation.

    tavily_api_key: str = Field(
        default="",
        env="TAVILY_API_KEY",
        description="Optional Tavily Search API key for high-quality ranked web results.",
    )

    brave_search_api_key: str = Field(
        default="",
        env="BRAVE_SEARCH_API_KEY",
        description="Optional Brave Search API key (alternative web search provider).",
    )

    serpapi_api_key: str = Field(
        default="",
        env="SERPAPI_API_KEY",
        description="Optional SerpAPI key (Google results) for web search.",
    )

    web_research_max_sources: int = Field(
        default=8,
        env="WEB_RESEARCH_MAX_SOURCES",
        description="Max ranked sources kept in the evidence pack after re-ranking.",
    )

    web_research_fetch_pages: int = Field(
        default=4,
        env="WEB_RESEARCH_FETCH_PAGES",
        description=(
            "How many top sources to actually open and read (content extraction).\n"
            "Higher = better evidence but slightly slower/more bandwidth. 0 = snippet-only."
        ),
    )

    # ==========================================================================
    # Context Retrieval Configuration
    # ==========================================================================
    # Controls automatic context retrieval and conversation storage.
    
    auto_context_retrieval: bool = Field(
        default=True,
        env="AUTO_CONTEXT_RETRIEVAL",
        description=(
            "Enable automatic retrieval of relevant context from vector store.\n"
            "When enabled, past conversations and documents are searched for\n"
            "relevant context before processing each query."
        )
    )
    
    context_retrieval_top_k: int = Field(
        default=5,
        env="CONTEXT_RETRIEVAL_TOP_K",
        description=(
            "Number of relevant context items to retrieve.\n"
            "Higher values provide more context but may slow down responses."
        )
    )
    
    auto_store_conversations: bool = Field(
        default=True,
        env="AUTO_STORE_CONVERSATIONS",
        description=(
            "Automatically store conversation history in vector store.\n"
            "Enables semantic search over past conversations for context."
        )
    )
    
    context_score_threshold: float = Field(
        default=0.3,
        env="CONTEXT_SCORE_THRESHOLD",
        description=(
            "Minimum similarity score (0-1) for context to be included.\n"
            "Higher values mean stricter relevance filtering."
        )
    )

    cadastral_intent_assessment_enabled: bool = Field(
        default=True,
        env="CADASTRAL_INTENT_ASSESSMENT_ENABLED",
        description=(
            "When enabled, cadastral CAD plotting uses vector-store retrieval plus a "
            "cheap LLM pass to interpret access roads, fences, and other plan extras "
            "from varied natural-language prompts. Regex parsing remains as fallback."
        ),
    )

    pdf_survey_replot_enabled: bool = Field(
        default=True,
        env="PDF_SURVEY_REPLOT_ENABLED",
        description=(
            "When enabled, survey plan PDF replot requests bypass the generic agent and "
            "use layout/vision extraction plus the cadastral CAD template pipeline "
            "(AutoCAD DWG output — not ArcGIS)."
        ),
    )
    
    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    # Controls how much information is logged and where.
    
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description=(
            "Logging verbosity level:\n"
            "  - DEBUG: Everything, including debug messages\n"
            "  - INFO: General information (default)\n"
            "  - WARNING: Warnings and errors only\n"
            "  - ERROR: Errors only"
        )
    )
    
    log_file: str = Field(
        default_factory=_default_log_file,
        env="LOG_FILE",
        description="File to write logs to. Frozen builds default to AppData\\Roaming\\SurvyAI\\logs."
    )
    
    # ==========================================================================
    # SurvyAI Desktop / Cloud
    # ==========================================================================
    # Used by the packaged Windows app for auth, billing, bootstrap config, and
    # the hosted LLM proxy. Fresh end-user installs do not have a project .env,
    # so the production backend URL must be a real default while tokens remain
    # empty until the user signs in.
    
    survyai_access_token: str = Field(
        default="",
        env="SURVYAI_ACCESS_TOKEN",
        description=(
            "Optional bearer token for SurvyAI cloud API (desktop commercial builds). "
            "When set, a future gateway can authenticate requests; Phase 1 only stores the value."
        ),
    )
    
    survyai_api_base_url: str = Field(
        default=PRODUCTION_CLOUD_API_BASE_URL,
        env="SURVYAI_API_BASE_URL",
        description=(
            "Base URL for SurvyAI cloud API. The installed desktop app defaults "
            "to the production backend; development can override via .env."
        ),
    )

    survyai_llm_proxy_enabled: bool = Field(
        default=False,
        env="SURVYAI_LLM_PROXY_ENABLED",
        description=(
            "If True, hosted LLM requests are routed through the SurvyAI cloud proxy "
            "instead of using provider API keys on the desktop."
        ),
    )

    survyai_llm_proxy_path: str = Field(
        default="/v1/llm/chat",
        env="SURVYAI_LLM_PROXY_PATH",
        description="Relative path for the hosted LLM proxy endpoint on the SurvyAI cloud API.",
    )

    survyai_device_id: str = Field(
        default="",
        env="SURVYAI_DEVICE_ID",
        description="Registered cloud device id sent with hosted LLM proxy calls for PC enforcement.",
    )

    # ==========================================================================
    # Plan policy scaffolding (NOT ENFORCED by default)
    # ==========================================================================
    # These fields define commercial rules for Free/Pro builds, but are intentionally
    # dormant while you continue development on the free build. When you are ready,
    # flip `ENFORCE_PLAN_POLICIES=true` in the packaged app environment.

    enforce_plan_policies: bool = Field(
        default=False,
        env="ENFORCE_PLAN_POLICIES",
        description=(
            "If True, apply plan policy constraints (Free=Ollama-only + CAD quotas; "
            "Pro=model switching + higher quotas). Default False (development mode)."
        ),
    )

    free_plan_cad_success_30d_cap: int = Field(
        default=10,
        env="FREE_PLAN_CAD_SUCCESS_30D_CAP",
        description="Free plan cap: successful CAD jobs allowed in a rolling 30 days (dormant unless enforced).",
        ge=0,
    )

    pro_plan_cad_success_30d_cap: int = Field(
        default=100,
        env="PRO_PLAN_CAD_SUCCESS_30D_CAP",
        description="Pro plan cap: successful CAD jobs allowed in a rolling 30 days (dormant unless enforced).",
        ge=0,
    )

    pro_plan_cad_success_365d_cap: int = Field(
        default=1300,
        env="PRO_PLAN_CAD_SUCCESS_365D_CAP",
        description="Pro plan cap: successful CAD jobs allowed in a rolling 365 days (dormant unless enforced).",
        ge=0,
    )
    
    # ==========================================================================
    # Pydantic Configuration
    # ==========================================================================
    
    class Config:
        """
        Pydantic model configuration.
        
        This inner class configures how Pydantic handles the settings:
        - env_file: Path to .env file for loading defaults
        - env_file_encoding: Character encoding for the .env file
        - case_sensitive: Whether env var names are case-sensitive
        - extra: Ignore unknown env vars / fields rather than raising ValidationError.
          This prevents crashes when .env contains experimental keys
          (e.g. CLOUD_ADMIN_API_KEY) that have not yet been promoted to named fields.
        """
        
        # Load settings from .env file if it exists.
        # Frozen desktop builds read %APPDATA%\\SurvyAI\\.env first, then CWD .env.
        env_file = (
            (str(user_data_path(".env")), ".env")
            if is_frozen_app()
            else ".env"
        )
        
        # Use UTF-8 encoding (supports special characters)
        env_file_encoding = "utf-8"
        
        # Environment variables are not case-sensitive
        # (GOOGLE_API_KEY = google_api_key = Google_Api_Key)
        case_sensitive = False

        # Silently ignore .env keys that don't map to a declared field.
        # Prevents ValidationError when new/experimental env vars are added to .env
        # before the corresponding Settings field is defined.
        extra = "ignore"


# ==============================================================================
# SINGLETON PATTERN
# ==============================================================================

# Global instance (starts as None)
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """
    Get the global Settings instance (creates it if needed).
    
    This function implements a singleton pattern - there's only ever one
    Settings instance in the application. This ensures:
    1. Settings are loaded once at startup
    2. All parts of the app see the same configuration
    3. No performance hit from repeatedly loading .env file
    
    Returns:
        Settings: The global settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.primary_llm)
        'openai'
        
        >>> # Same instance returned
        >>> settings2 = get_settings()
        >>> settings is settings2
        True
    
    Raises:
        ValidationError: If required settings are missing or invalid
    """
    global _settings_instance
    
    # Create instance on first call
    if _settings_instance is None:
        _settings_instance = Settings()
        # Log the primary LLM setting for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Settings loaded - Primary LLM: {_settings_instance.primary_llm}, Fallback LLM: {_settings_instance.fallback_llm}")
    
    return _settings_instance


def reset_settings():
    """
    Reset the global settings instance.
    
    Useful for testing or when .env file changes and you want to reload settings.
    """
    global _settings_instance
    _settings_instance = None


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = ["Settings", "get_settings", "reset_settings"]
