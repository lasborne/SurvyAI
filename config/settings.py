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
        default="gpt-4o-mini",
        env="OPENAI_MODEL",
        description=(
            "OpenAI model name (legacy - used as fallback if tiered models not set).\n"
            "For tiered model selection, use OPENAI_MODEL_NANO, OPENAI_MODEL_MINI, OPENAI_MODEL_COMPLEX instead."
        )
    )
    
    # Tiered OpenAI models for intelligent complexity-based selection
    openai_model_nano: str = Field(
        default="gpt-5-nano",
        env="OPENAI_MODEL_NANO",
        description=(
            "OpenAI model for simple tasks (e.g., basic questions, simple lookups).\n"
            "Default: gpt-5-nano (cost-effective for trivial tasks).\n"
            "If unavailable, falls back to OPENAI_MODEL or gpt-4o-mini."
        )
    )
    
    openai_model_mini: str = Field(
        default="gpt-5-mini",
        env="OPENAI_MODEL_MINI",
        description=(
            "OpenAI model for average complexity tasks (e.g., coordinate conversions, basic calculations).\n"
            "Default: gpt-5-mini (balanced cost and capability).\n"
            "If unavailable, falls back to OPENAI_MODEL or gpt-4o-mini."
        )
    )
    
    openai_model_complex: str = Field(
        default="gpt-5.1",
        env="OPENAI_MODEL_COMPLEX",
        description=(
            "OpenAI model for very complex tasks (e.g., multi-step analysis, complex reasoning).\n"
            "Default: gpt-5.1 (highest capability for complex tasks).\n"
            "If unavailable, falls back to OPENAI_MODEL or gpt-4o."
        )
    )
    
    enable_tiered_models: bool = Field(
        default=True,
        env="ENABLE_TIERED_MODELS",
        description=(
            "Enable intelligent model selection based on task complexity.\n"
            "If False, uses OPENAI_MODEL for all tasks (legacy behavior)."
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
        default=4000,
        env="AGENT_MAX_TOKENS",
        description=(
            "Maximum tokens in LLM response.\n"
            "1 token ≈ 4 characters or 0.75 words.\n"
            "4000 tokens ≈ 3000 words, enough for detailed responses.\n"
            "Note: This is capped per model (e.g., GPT-4o-mini max is 16384)."
        )
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
    
    agent_max_iterations: int = Field(
        default=20,
        env="AGENT_MAX_ITERATIONS",
        description=(
            "Maximum number of agent-tool iterations per query.\n"
            "Default: 20 iterations (agent → tools → agent → tools...).\n"
            "Prevents infinite loops. Increase for very complex multi-step tasks."
        )
    )
    
    primary_llm: Literal["deepseek", "gemini", "claude", "openai"] = Field(
        default="openai",
        env="PRIMARY_LLM",
        description=(
            "Which LLM to use as primary. Options:\n"
            "  - openai: Use OpenAI (GPT-4/4o/5) - Default\n"
            "  - claude: Use Anthropic Claude (Opus/Sonnet/Haiku)\n"
            "  - gemini: Use Google Gemini\n"
            "  - deepseek: Use DeepSeek"
        )
    )
    
    fallback_llm: Literal["deepseek", "gemini", "claude", "openai"] = Field(
        default="gemini",
        env="FALLBACK_LLM",
        description="Which LLM to use if primary fails. Same options as PRIMARY_LLM."
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
    # Vector Database Configuration
    # ==========================================================================
    # ChromaDB is used for local vector storage with semantic search.
    # Supports local embeddings (free) and OpenAI embeddings (higher quality).
    
    vector_store_enabled: bool = Field(
        default=True,
        env="VECTOR_STORE_ENABLED",
        description="Enable/disable the vector database for semantic search."
    )
    
    vector_store_path: str = Field(
        default=".survyai_vectordb",
        env="VECTOR_STORE_PATH",
        description=(
            "Directory for vector database persistence.\n"
            "Relative to project root or absolute path."
        )
    )
    
    embedding_provider: Literal["local", "openai"] = Field(
        default="local",
        env="EMBEDDING_PROVIDER",
        description=(
            "Which embedding provider to use:\n"
            "  - local: Sentence Transformers (free, offline capable) - Default\n"
            "  - openai: OpenAI embeddings (higher quality, requires API key)"
        )
    )
    
    local_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        env="LOCAL_EMBEDDING_MODEL",
        description=(
            "Local embedding model from Sentence Transformers.\n"
            "Options: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (better quality)"
        )
    )
    
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
        description=(
            "OpenAI embedding model. Options:\n"
            "  - text-embedding-3-small: Cost-effective (default)\n"
            "  - text-embedding-3-large: Higher quality, more expensive"
        )
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
        default="survyai.log",
        env="LOG_FILE",
        description="File to write logs to. Relative to project root."
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
        """
        
        # Load settings from .env file if it exists
        env_file = ".env"
        
        # Use UTF-8 encoding (supports special characters)
        env_file_encoding = "utf-8"
        
        # Environment variables are not case-sensitive
        # (GOOGLE_API_KEY = google_api_key = Google_Api_Key)
        case_sensitive = False


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
