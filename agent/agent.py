"""
================================================================================
SurvyAI Agent - LangGraph Implementation
================================================================================

This module implements the core AI agent for SurvyAI using LangGraph, a framework
for building stateful, multi-step AI applications.

ARCHITECTURE OVERVIEW:
----------------------
LangGraph uses a graph-based architecture where:
- NODES: Functions that process and transform state
- EDGES: Define the flow between nodes
- STATE: A shared data structure passed through the graph

For SurvyAI, the graph flow is:
    
    [START] 
       │
       ▼
    ┌─────────────────┐
    │   Agent Node    │  ◄── LLM reasons about the query
    │  (LLM Reasoning)│      and decides what to do
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐
    │  Should Use     │────►│   Tool Node     │
    │    Tools?       │ Yes │ (Execute Tools) │
    └────────┬────────┘     └────────┬────────┘
             │ No                    │
             ▼                       │
    ┌─────────────────┐              │
    │      END        │◄─────────────┘
    │  (Return Result)│
    └─────────────────┘

WHY LANGGRAPH FOR SURVYAI:
--------------------------
1. Complex workflows: Surveying tasks often require multiple steps
2. Tool orchestration: Need to coordinate AutoCAD, Excel, ArcGIS, etc.
3. State management: Track context across tool calls
4. Error recovery: Handle failures gracefully
5. Extensibility: Easy to add new capabilities

MODULES AND DEPENDENCIES:
-------------------------
- langgraph: Graph-based agent framework
- langchain_core: Base classes for messages, tools
- langchain_google_genai: Google Gemini LLM integration
- langchain_openai: OpenAI models (GPT-4/4o/5) and OpenAI-compatible API (for DeepSeek)
- langchain_anthropic: Anthropic Claude models (Opus/Sonnet/Haiku)

REFACTORED LAYOUT:
-----------------
- agent.prompts: SYSTEM_PROMPT and other prompt strings (editable without touching agent logic).
- agent.state: AgentState (LangGraph state), RAGRouteDecision, and looks_like_file_driven_task().

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import json
import operator
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

# Ensure Any is available globally (for Pydantic model evaluation)
# This prevents "name 'Any' is not defined" errors
# Import it explicitly and make it available in globals
import typing
# Make Any available in multiple ways for different evaluation contexts
Any = typing.Any
globals()['Any'] = typing.Any
# For eval/exec contexts, ensure it's available
if isinstance(__builtins__, dict):
    __builtins__['Any'] = typing.Any
elif hasattr(__builtins__, '__dict__'):
    __builtins__.__dict__['Any'] = typing.Any

# LangGraph imports for building the agent graph
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports for LLM and tool integration
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Pydantic for input validation
from pydantic import BaseModel, Field

# Local imports
from config import get_settings
from utils.logger import get_logger
from utils.token_limiter import (
    estimate_message_tokens,
    check_tpm_limit,
    chunk_messages,
    wait_for_rate_limit,
    format_token_warning,
    TokenEstimate,
    get_tpm_limit,
)
from tools import (
    ExcelProcessor,
    DocumentProcessor,
    AutoCADProcessor,
    BlueMarbleConverter,
    GeographicCalculatorCLI,
    ArcGISProcessor,
    VectorStore,
    COLLECTION_DOCUMENTS,
    COLLECTION_DRAWINGS,
    COLLECTION_COORDINATES,
    COLLECTION_CONVERSATIONS,
)
from tools.autocad_processor import DXFProcessor
from datetime import datetime
from utils.coordinate_parsing import extract_points, infer_crs_from_text
from utils.area import best_area
from utils.internet import internet_search as _internet_search

# Prompts and state live in separate modules for smaller, maintainable agent.py
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState, RAGRouteDecision, looks_like_file_driven_task

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

# Get a logger instance for this module
# All log messages will be prefixed with 'agent' for easy filtering
logger = get_logger(__name__)


# ==============================================================================
# MAIN AGENT CLASS
# ==============================================================================

class SurvyAIAgent:
    """
    The main AI agent for SurvyAI, built with LangGraph.
    
    This class orchestrates:
    1. LLM initialization (Gemini, DeepSeek, Claude, or OpenAI)
    2. Tool creation (AutoCAD, Excel, Document processors)
    3. Graph construction (nodes, edges, routing logic)
    4. Query processing (invoking the graph with user input)
    
    Architecture:
    ------------
    The agent uses a ReAct (Reasoning + Acting) pattern where:
    1. The LLM receives a query and decides what to do
    2. If tools are needed, they are executed
    3. Tool results are fed back to the LLM
    4. The LLM formulates a final response
    
    Usage:
    ------
    ```python
    agent = SurvyAIAgent()
    result = agent.process_query("Calculate area from survey.dwg")
    print(result["response"])
    ```
    """
    
    def __init__(self):
        """
        Initialize the SurvyAI agent.
        
        Initialization sequence:
        1. Load configuration settings
        2. Create application processors (AutoCAD, Excel, etc.)
        3. Initialize LLMs (primary and fallback)
        4. Create tool definitions
        5. Build the LangGraph
        
        Raises:
            Exception: If LLM initialization fails
        """
        # ------------------------------------------------------------------
        # Step 1: Load configuration
        # ------------------------------------------------------------------
        # Settings come from environment variables and .env file
        self.settings = get_settings()
        
        # Validate that primary LLM is set correctly
        logger.info(f"Configuration loaded - Primary LLM: {self.settings.primary_llm}, Fallback LLM: {self.settings.fallback_llm}")
        
        # Track which Gemini model we're using (for logging/debugging)
        self._current_gemini_model: Optional[str] = getattr(
            self.settings, "gemini_model", None
        )
        
        # Track which OpenAI model we're currently using (for tiered model selection)
        self._current_openai_model: Optional[str] = None

        # Lightweight caches to avoid expensive re-initialization/re-compilation
        # (No functional impact; improves latency and reduces mid-flight churn.)
        self._openai_llm_cache: Dict[tuple, BaseChatModel] = {}
        self._app_signature: Optional[tuple] = None  # (model_name, tool_names_tuple)
        
        # If primary is OpenAI, ensure API key is configured
        if self.settings.primary_llm == "openai":
            if not self.settings.openai_api_key or not self.settings.openai_api_key.strip():
                raise ValueError(
                    "Primary LLM is set to 'openai' but OPENAI_API_KEY is not configured. "
                    "Please set OPENAI_API_KEY in your .env file or environment variables."
                )
            model_name = getattr(self.settings, "openai_model", "gpt-4o-mini")
            logger.info(f"✓ OpenAI configured as primary LLM (model: {model_name})")
        
        # If primary is Claude, ensure API key is configured
        elif self.settings.primary_llm == "claude":
            if not self.settings.anthropic_api_key or not self.settings.anthropic_api_key.strip():
                raise ValueError(
                    "Primary LLM is set to 'claude' but ANTHROPIC_API_KEY is not configured. "
                    "Please set ANTHROPIC_API_KEY in your .env file or environment variables."
                )
            logger.info("✓ Claude configured as primary LLM")
        
        # ------------------------------------------------------------------
        # Step 2: Initialize application processors
        # ------------------------------------------------------------------
        # These are the "backends" that actually do the work.
        # They connect to AutoCAD, read Excel files, etc.
        
        # AutoCAD processor - connects via COM API
        # auto_connect=False means we connect on first use, not at startup
        self.autocad = AutoCADProcessor(auto_connect=False)
        
        # DXF fallback processor - works without AutoCAD installed
        # Uses ezdxf library to read DXF files directly
        self.dxf_fallback = DXFProcessor()
        if self.dxf_fallback.is_available:
            logger.info("✓ DXF fallback processor available (ezdxf)")
        
        # Excel processor - reads .xlsx and .xls files
        self.excel_processor = ExcelProcessor()
        
        # Document processor - extracts text from PDF and Word
        self.document_processor = DocumentProcessor()
        
        # Coordinate conversions:
        # IMPORTANT: pyproj is the default main method; Geographic Calculator COM is optional.
        # We lazy-connect COM only when explicitly requested to avoid noisy/irrelevant startup scans.
        self.blue_marble = BlueMarbleConverter(auto_connect=False)
        
        # Geographic Calculator CLI - lazy scan only when tool is used
        self.geocalc_cli = GeographicCalculatorCLI(auto_detect=False)
        
        # ArcGIS processor - advanced geospatial analysis
        self.arcgis_processor = ArcGISProcessor()
        
        # Vector store - semantic search and embeddings
        self.vector_store = None
        if getattr(self.settings, 'vector_store_enabled', True):
            try:
                self.vector_store = VectorStore(
                    persist_directory=getattr(self.settings, 'vector_store_path', None),
                    embedding_provider=getattr(self.settings, 'embedding_provider', 'local'),
                    openai_api_key=getattr(self.settings, 'openai_api_key', None),
                    local_model_name=getattr(self.settings, 'local_embedding_model', 'all-MiniLM-L6-v2'),
                    openai_model_name=getattr(self.settings, 'openai_embedding_model', 'text-embedding-3-small'),
                )
                logger.info("✓ Vector store initialized successfully")
            except Exception as e:
                logger.warning(f"⚠ Vector store initialization failed: {e}")
                logger.warning("Semantic search will be unavailable")
                self.vector_store = None
        
        # ------------------------------------------------------------------
        # Step 3: Initialize LLMs
        # ------------------------------------------------------------------
        # We have a primary LLM (default: OpenAI) and a fallback (default: Gemini)
        # If the primary fails, we automatically try the fallback
        
        logger.info(f"Initializing primary LLM: {self.settings.primary_llm}")
        logger.info(f"Initializing fallback LLM: {self.settings.fallback_llm}")
        
        try:
            self.llm_primary = self._initialize_llm(self.settings.primary_llm)
            logger.info(f"✓ Primary LLM ({self.settings.primary_llm}) initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize primary LLM ({self.settings.primary_llm}): {e}")
            raise ValueError(
                f"Failed to initialize primary LLM '{self.settings.primary_llm}'. "
                f"Please check your API key configuration. Error: {e}"
            )
        
        try:
            self.llm_fallback = self._initialize_llm(self.settings.fallback_llm)
            logger.info(f"✓ Fallback LLM ({self.settings.fallback_llm}) initialized successfully")
        except Exception as e:
            logger.warning(f"⚠ Fallback LLM ({self.settings.fallback_llm}) initialization failed: {e}")
            # Fallback initialization failure is not critical, but log it
        
        # ------------------------------------------------------------------
        # Step 4: Create tools
        # ------------------------------------------------------------------
        # Tools are functions the LLM can call to perform actions
        self.tools = self._create_tools()
        
        # Bind tools to the LLM so it knows what's available
        # This creates a new LLM instance that includes tool definitions
        logger.info(f"Binding tools to primary LLM: {self.settings.primary_llm}")
        self.llm_with_tools = self.llm_primary.bind_tools(self.tools)
        logger.info(f"✓ Tools bound to {self.settings.primary_llm}")
        
        # ------------------------------------------------------------------
        # Step 5: Build the LangGraph
        # ------------------------------------------------------------------
        # The graph defines how the agent processes queries
        self.graph = self._build_graph()
        
        # Compile the graph with a memory checkpointer
        # This enables conversation history and state persistence
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

        # Cache current app signature (model + tool names) for reuse in process_query
        try:
            model_sig = getattr(self.llm_primary, "model", None) or getattr(self.settings, "openai_model", None) or self.settings.primary_llm
        except Exception:
            model_sig = self.settings.primary_llm
        tool_sig = tuple(sorted([t.name for t in self.tools]))
        self._app_signature = (model_sig, tool_sig)
        
        logger.info("SurvyAI agent initialized successfully with LangGraph")
        
        # Session tracking for conversation continuity
        self._current_session_id: Optional[str] = None

        # Last cadastral plan output (for in-session modifications without re-prompting)
        # Template file remains read-only; modifications apply only to this output file.
        self._last_cadastral_output_dwg: Optional[str] = None
        self._last_cadastral_profile_path: Optional[str] = None

        # STRICT: Survey plan template paths must never be written (read-only to avoid corruption).
        # Populated from template_profiles/*.json and when learning a template.
        self._protected_template_paths: set = set()

        # Persistent cadastral CAD template memory (multi-template registry).
        # This lets users omit the template path after successful prior runs.
        self._cad_template_memory_file: Optional[str] = None

        # Internet permission (interactive, user-controlled)
        # Default: False (must ask user before searching the internet)
        self._internet_permission_granted: bool = False
    
    # ==========================================================================
    # CONTEXT RETRIEVAL AND STORAGE
    # ==========================================================================
    
    def _extract_document_paths(self, query: str) -> List[str]:
        """
        Extract document file paths from a query string.
        
        Looks for common document file patterns:
        - .docx, .doc (Word documents)
        - .pdf (PDF documents)
        - Paths in quotes or as-is
        
        Args:
            query: The user's query string
            
        Returns:
            List of detected document file paths
        """
        import re
        from pathlib import Path
        
        document_paths = []
        
        # Pattern to match file paths with document extensions.
        #
        # IMPORTANT: Do NOT rely on quoted-string extraction, because real Windows paths
        # may contain apostrophes (e.g. "MICHAEL's") which breaks naive single-quote parsing.
        #
        # Instead, match Windows/Unix-like paths up to a known extension.
        patterns = [
            # Windows absolute paths: allow spaces and apostrophes, but stop before illegal filename chars / quotes
            r'([A-Za-z]:\\[^\r\n"<>|]+?\.(?:docx?|pdf))',
            # Unix/relative paths (also allow backslashes for relative Windows-ish paths)
            r'((?:/|\\)[^\r\n"<>|]+?\.(?:docx?|pdf))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                path_str = (match or "").strip()
                # Trim common trailing punctuation / wrappers
                path_str = path_str.strip().strip('"').strip("'").rstrip(").,;")
                path = Path(path_str)
                # Verify the file exists and is a document
                if path.exists() and path.is_file():
                    if path.suffix.lower() in ['.docx', '.doc', '.pdf']:
                        if str(path.resolve()) not in document_paths:
                            document_paths.append(str(path.resolve()))
        
        return document_paths

    def _infer_output_path_from_input(
        self, 
        input_path: str, 
        output_filename: Optional[str] = None,
        output_type: str = "file"
    ) -> Optional[str]:
        """
        Infer output path from input file path when user doesn't specify location.
        
        CRITICAL RULE: If user doesn't explicitly specify where to create/locate a file or operation,
        default to the SAME FOLDER as the input file/folder/document.
        
        Args:
            input_path: Path to input file/folder
            output_filename: Optional output filename (if None, returns just the parent directory)
            output_type: "file" (returns file path) or "folder" (returns directory path)
            
        Returns:
            Inferred output path, or None if input_path is invalid
        """
        from pathlib import Path
        
        try:
            input_p = Path(input_path)
            # If input_path is a file, use its parent; if it's a directory, use it directly
            if input_p.is_file():
                parent_dir = input_p.parent
            elif input_p.is_dir():
                parent_dir = input_p
            else:
                # Path doesn't exist yet, but we can still extract parent from the path string
                parent_dir = input_p.parent if input_p.suffix else input_p
            
            if output_type == "folder":
                return str(parent_dir.resolve())
            elif output_filename:
                # If output_filename is already absolute, return as-is
                output_p = Path(output_filename)
                if output_p.is_absolute():
                    return str(output_p.resolve())
                # Otherwise, resolve relative to input's parent folder
                return str((parent_dir / output_filename).resolve())
            else:
                return str(parent_dir.resolve())
        except Exception as e:
            logger.debug(f"Failed to infer output path from {input_path}: {e}")
            return None
    
    def _extract_requested_output_docx(self, query: str, input_doc_path: str) -> Optional[str]:
        """
        Try to infer an output .docx filename/path from a query.
        
        Preference order:
        0) Explicit "Save ... as 'X.docx'" or "save the Summary file as 'X.docx'" (user intent)
        1) A quoted .docx path that does NOT exist (assumed intended output path)
        2) A quoted .docx filename that is NOT a substring of the input filename,
           resolved into the same folder as the input doc
        3) None (unknown)
        """
        import re
        from pathlib import Path

        input_path = Path(input_doc_path)
        q = query
        input_name_lower = input_path.name.lower()

        # 0) PRIORITY: Explicit "Save ... as 'X.docx'" - respect user intent
        explicit_patterns = [
            r"(?:save|saved|export)\s+(?:the\s+)?(?:summary\s+)?(?:file\s+)?as\s+['\"]([^'\"]+\.docx)['\"]",
            r"(?:save|saved|export)\s+(?:the\s+)?(?:summary\s+)?(?:file\s+)?as\s+([^\s,\.]+\.docx)",
            r"as\s+['\"]([^'\"]+\.docx)['\"]\s+in\s+(?:the\s+)?same\s+folder",
        ]
        for pat in explicit_patterns:
            m = re.search(pat, q, flags=re.IGNORECASE)
            if m:
                out = (m.group(1) or "").strip().strip("'").strip('"').rstrip(").,;")
                if out and out.lower().endswith(".docx"):
                    p = Path(out)
                    if p.is_absolute() and not p.exists():
                        return str(p)
                    if not p.is_absolute():
                        return str((input_path.parent / p.name).resolve())
                    return str(p)

        # Find any .docx-like candidates in the query.
        candidates: list[str] = []
        candidates.extend(re.findall(r'([A-Za-z]:\\[^\r\n"<>|]+?\.docx)', q, flags=re.IGNORECASE))
        candidates.extend(re.findall(r'([^\s"<>|]+?\.docx)', q, flags=re.IGNORECASE))

        # Normalize: keep order, strip whitespace, trim wrappers
        seen = set()
        normed: list[str] = []
        for c in candidates:
            if not c:
                continue
            s = c.strip().strip('"').strip("'").rstrip(").,;")
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            normed.append(s)
        candidates = normed
        if not candidates:
            return None

        # Filter: remove input path and any candidate that is a trailing substring of input filename
        # (e.g. "PROJECTS.docx" from "REPORT ON FIVE PROJECTS.docx" must be excluded)
        input_abs = str(input_path.resolve()).lower()

        def _is_input_or_substring(c: str) -> bool:
            if str(Path(c).resolve()).lower() == input_abs:
                return True
            c_name = Path(c).name.lower()
            if c_name == input_name_lower:
                return True
            if input_name_lower.endswith(c_name):
                return True
            return False

        candidates = [c for c in candidates if not _is_input_or_substring(c)]
        if not candidates:
            return None

        # 1) Full path candidates that don't exist -> likely output
        for c in candidates:
            p = Path(c)
            if p.is_absolute() and (p.suffix.lower() == ".docx") and (not p.exists()):
                return str(p)

        # 2) Relative candidates -> resolve relative to input folder
        for c in candidates:
            p = Path(c)
            if not p.is_absolute() and p.suffix.lower() == ".docx":
                return str((input_path.parent / p.name).resolve())

        return None

    def _extract_any_output_docx(self, query: str) -> Optional[str]:
        """
        Extract an intended output .docx path/filename from a query (no input doc context).

        - Supports quoted/unquoted strings
        - Supports Windows absolute paths and filename-only outputs
        """
        import re
        from pathlib import Path

        q = query or ""
        candidates: list[str] = []
        candidates.extend(re.findall(r'([A-Za-z]:\\[^\r\n"<>|]+?\.docx)', q, flags=re.IGNORECASE))
        candidates.extend(re.findall(r'([^\s"<>|]+?\.docx)', q, flags=re.IGNORECASE))

        # Normalize
        normed: list[str] = []
        seen = set()
        for c in candidates:
            if not c:
                continue
            s = c.strip().strip('"').strip("'").rstrip(").,;")
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            normed.append(s)

        if not normed:
            return None

        # Prefer absolute; else first filename
        for s in normed:
            p = Path(s)
            if p.is_absolute():
                return str(p)
        return normed[0]

    def _should_fastpath_large_doc_summary(self, query: str, doc_info: Dict[str, Any]) -> bool:
        """
        Decide if we should bypass LangGraph and run a deterministic pipeline.
        
        We do this for large documents + explicit summarize/save requests to avoid
        multi-iteration tool loops, TPM overflow, and timeouts. Uses multiple signals:
        - Metadata-based: pages, words, estimated_tokens (from get_resource_estimation)
        - File-size fallback: when metadata is missing/unreliable, use file_size_mb
        """
        q = (query or "").lower()
        if not doc_info:
            return False

        # Accept full doc_info (from preflight) or estimation sub-dict
        est = doc_info.get("estimation") or doc_info
        pages = int(doc_info.get("page_count") or est.get("page_count") or 0)
        words = int(doc_info.get("word_count") or est.get("word_count") or 0)
        tokens = int(doc_info.get("estimated_tokens") or est.get("estimated_tokens") or 0)
        file_size_mb = float(doc_info.get("file_size_mb") or est.get("file_size_mb") or 0)

        # Relaxed thresholds: trigger earlier to avoid TPM overflow (500K limit)
        is_large = (
            pages > 50 or words > 25000 or tokens > 50000 or
            # Fallback: large file + summarize request (metadata may be missing for table-heavy docs)
            (file_size_mb > 3 and (pages > 0 or words > 0 or tokens > 0))
        )
        # File-size-only fallback when metadata estimation failed or returned zeros
        if not is_large and file_size_mb > 5:
            is_large = True

        wants_summary = (
            ("summar" in q or "summary" in q) and
            (".docx" in q or "save" in q or "same folder" in q or "document" in q or
             "projects" in q or "key survey" in q or "key details" in q or "professionally" in q)
        )
        return bool(is_large and wants_summary)

    def _should_fastpath_docx_report(self, query: str) -> bool:
        """
        Fast-path: user asks for a generated report and to save it to a .docx,
        without providing an input document to summarize.
        """
        q = (query or "").lower()
        has_output_docx = ".docx" in q or "history.docx" in q or "save" in q
        wants_save = any(k in q for k in ["save", "saved", "export", "into the folder", "project folder", "same folder"])
        # Many users don't say "report" explicitly; treat any substantial write-up request as report-like.
        is_report_like = any(k in q for k in ["report", "trace", "history", "explain", "overview", "process", "licens", "licensing", "practice"])
        has_input_doc = bool(self._extract_document_paths(query))
        return bool(has_output_docx and wants_save and is_report_like and not has_input_doc)

    # ==========================================================================
    # FAST-PATH: CAD CADASTRAL PLAN (Template DWG -> Output DWG)
    # ==========================================================================

    def _cad_template_profiles_dir(self):
        from pathlib import Path
        # Anchor template memory/profiles to the project root, not the caller's cwd.
        # This prevents prompts from "losing" remembered templates when the CLI is launched
        # from a different working directory.
        d = (Path(__file__).resolve().parent.parent / "template_profiles").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cad_template_memory_path(self):
        from pathlib import Path
        if self._cad_template_memory_file:
            return Path(self._cad_template_memory_file).resolve()
        p = (self._cad_template_profiles_dir() / "template_memory.json").resolve()
        self._cad_template_memory_file = str(p)
        return p

    def _load_cad_template_memory(self) -> Dict[str, Any]:
        import json as _json
        mem_path = self._cad_template_memory_path()
        data: Dict[str, Any] = {"templates": []}
        if not mem_path.exists():
            return self._bootstrap_cad_template_memory_from_profiles(data)
        try:
            raw = _json.loads(mem_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("templates"), list):
                data = raw
        except Exception:
            data = {"templates": []}
        if not (data.get("templates") or []):
            data = self._bootstrap_cad_template_memory_from_profiles(data)
        return data

    def _save_cad_template_memory(self, data: Dict[str, Any]) -> None:
        import json as _json
        mem_path = self._cad_template_memory_path()
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        mem_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")

    def _bootstrap_cad_template_memory_from_profiles(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Seed template memory from existing learned template profiles.
        This supports older installations that already have template_profiles/*.json
        but were created before template_memory.json existed.
        """
        import json as _json
        from pathlib import Path

        data = dict(data or {"templates": []})
        entries = list(data.get("templates") or [])
        by_path: Dict[str, Dict[str, Any]] = {}

        for ent in entries:
            try:
                p = str(Path(str(ent.get("path") or "")).resolve())
                if p:
                    by_path[p] = ent
            except Exception:
                continue

        profile_dir = self._cad_template_profiles_dir()
        dirty = False
        for prof_path in profile_dir.glob("*.json"):
            try:
                if prof_path.name.lower() == "template_memory.json":
                    continue
                raw = _json.loads(prof_path.read_text(encoding="utf-8"))
                template_meta = raw.get("template") or {}
                tp_raw = str(template_meta.get("path") or "").strip()
                if not tp_raw:
                    continue
                tp = Path(tp_raw).resolve()
                tp_res = str(tp)
                learned_at = str(template_meta.get("learned_at") or "")
                sig = template_meta.get("signature") or {}
                stat = tp.stat() if tp.exists() else None
                entry = by_path.get(tp_res) or {
                    "id": tp.stem,
                    "path": tp_res,
                    "name": tp.name,
                    "aliases": self._candidate_template_aliases(tp_res),
                    "use_count": 0,
                }
                entry["profile_path"] = str(prof_path.resolve())
                entry["last_used_at"] = str(entry.get("last_used_at") or learned_at or "")
                entry["is_available"] = bool(tp.exists())
                entry["signature"] = {
                    "size_bytes": int(sig.get("size_bytes") or (stat.st_size if stat else -1)),
                    "mtime_ns": int(sig.get("mtime_ns") or (getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)) if stat else -1)),
                }
                by_path[tp_res] = entry
                dirty = True
            except Exception:
                continue

        if dirty:
            data["templates"] = list(by_path.values())
            try:
                self._save_cad_template_memory(data)
            except Exception:
                pass
        else:
            data["templates"] = list(by_path.values()) if by_path else list(entries)
        return data

    def _candidate_template_aliases(self, template_path: str) -> List[str]:
        import re
        from pathlib import Path
        tp = Path(template_path)
        stem = tp.stem.strip()
        name = tp.name.strip()
        aliases = [stem, name]
        # Split common separators so prompts like "template 3" or "1000 template" can still match.
        parts = re.split(r"[_\-\s]+", stem)
        aliases.extend([p for p in parts if p])
        out: List[str] = []
        seen: set[str] = set()
        for a in aliases:
            key = str(a).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(a))
        return out

    def _register_cad_template_memory(self, template_path: str, profile_path: str) -> None:
        import time
        from pathlib import Path

        try:
            tp = Path(template_path).resolve()
            prof = Path(profile_path).resolve()
            stat = tp.stat()
            data = self._load_cad_template_memory()
            entries = list(data.get("templates") or [])
            tp_res = str(tp)
            prof_res = str(prof)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            idx_hit = -1
            for idx, ent in enumerate(entries):
                try:
                    if str(Path(str(ent.get("path") or "")).resolve()) == tp_res:
                        idx_hit = idx
                        break
                except Exception:
                    continue
            new_entry = {
                "id": tp.stem,
                "path": tp_res,
                "name": tp.name,
                "aliases": self._candidate_template_aliases(tp_res),
                "last_used_at": now,
                "use_count": 1,
                "profile_path": prof_res,
                "signature": {
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
                },
                "is_available": True,
            }
            if idx_hit >= 0:
                cur = dict(entries[idx_hit] or {})
                new_entry["use_count"] = int(cur.get("use_count") or 0) + 1
                aliases = list(cur.get("aliases") or []) + new_entry["aliases"]
                seen: set[str] = set()
                dedup_aliases: List[str] = []
                for a in aliases:
                    key = str(a).strip().lower()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    dedup_aliases.append(str(a))
                new_entry["aliases"] = dedup_aliases
                entries[idx_hit] = new_entry
            else:
                entries.append(new_entry)
            data["templates"] = entries
            self._save_cad_template_memory(data)
        except Exception:
            pass

    def _resolve_cadastral_template_from_memory(self, query: str) -> Optional[Dict[str, str]]:
        import time
        from pathlib import Path

        q = (query or "").lower()
        data = self._load_cad_template_memory()
        entries = list(data.get("templates") or [])
        valid_entries: List[Dict[str, Any]] = []
        dirty = False

        for ent in entries:
            try:
                tp = Path(str(ent.get("path") or "")).resolve()
                exists = tp.exists()
                if bool(ent.get("is_available")) != bool(exists):
                    ent["is_available"] = bool(exists)
                    dirty = True
                if exists:
                    valid_entries.append(ent)
            except Exception:
                ent["is_available"] = False
                dirty = True

        if dirty:
            data["templates"] = entries
            try:
                self._save_cad_template_memory(data)
            except Exception:
                pass

        if not valid_entries:
            return None

        def _score(ent: Dict[str, Any]) -> Tuple[int, str]:
            score = 0
            aliases = [str(a).lower() for a in (ent.get("aliases") or []) if str(a).strip()]
            name = str(ent.get("name") or "").lower()
            stem = str(ent.get("id") or "").lower()
            for token in aliases + [name, stem]:
                if token and token in q:
                    score += max(5, len(token))
            # Most recent valid template wins when there is no clear semantic match.
            last_used = str(ent.get("last_used_at") or "")
            return (score, last_used)

        best = sorted(valid_entries, key=_score, reverse=True)[0]
        try:
            best["last_used_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            data["templates"] = entries
            self._save_cad_template_memory(data)
        except Exception:
            pass
        return {
            "template_path": str(Path(str(best.get("path") or "")).resolve()),
            "profile_path": str(Path(str(best.get("profile_path") or "")).resolve()) if best.get("profile_path") else "",
            "template_name": str(best.get("name") or ""),
        }

    def _should_fastpath_cadastral_cad(self, query: str) -> bool:
        q = (query or "").lower()
        if ".dwg" not in q:
            return False
        has_generate = any(k in q for k in ["generate", "create", "produce", "save"]) and ".dwg" in q
        has_coords = "coordinates" in q and ("mE" in q or "mn" in q or "(" in q)
        return bool(has_generate and has_coords)

    def _run_cadastral_cad_prompt_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Deterministic parser for your cadastral prompts.
        - Learns a template profile if missing
        - Applies the template and replots parcel + bearings/dist + tables
        - ABSOLUTE: The survey template DWG is never modified; all edits are to a copy (output drawing).
        - Access road: Extracts width, offset, and pillar ref from natural language (e.g. "7m (and an offset of 3m) on the side joining pillars X and Y") so the LLM/agent can reason through user input and assign tasks without hardcoded phrasing.
        """
        import re
        from pathlib import Path

        q = query or ""

        def _pick(pats: list[str]) -> Optional[str]:
            for pat in pats:
                m = re.search(pat, q, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    return (m.group(1) or "").strip()
            return None

        def _quoted_list(after_key_pat: str) -> list[str]:
            m = re.search(after_key_pat, q, flags=re.IGNORECASE | re.DOTALL)
            if not m:
                return []
            tail = m.group(1)
            toks = re.findall(r"'([^']+)'", tail)
            return [t.strip() for t in toks if t.strip()]

        template = _pick([r"template\s+'([^']+?\.dwg)'", r"template\s+\"([^\"]+?\.dwg)\""])
        output = _pick([r"generate\s+'([^']+?\.dwg)'", r"generate\s+\"([^\"]+?\.dwg)\""])

        buyer = _pick([r"buyer\s*name\s*=\s*'([^']+)'", r"buyer\s*name\s*=\s*\"([^\"]+)\""])
        location = _pick([r"location\s*=\s*'([^']+)'", r"location\s*=\s*\"([^\"]+)\""])
        lga = _pick([r"local\s+government\s+area\s*=\s*'([^']+)'", r"local\s+government\s+area\s*=\s*\"([^\"]+)\""])
        state = _pick([r"state\s*=\s*'([^']+)'", r"state\s*=\s*\"([^\"]+)\""])
        origin = _pick([r"origin(?:_crs|/crs|)\s*=\s*'([^']+)'", r"origin(?:_crs|/crs|)\s*=\s*\"([^\"]+)\""])
        plan_no = _pick([r"plan\s+number\s*=\s*'([^']+)'", r"plan\s+number\s*=\s*\"([^\"]+)\""])
        cert_date = _pick([r"date\s+on\s+the\s+certification\s*=\s*'([^']+)'", r"date\s+on\s+the\s+certification\s*=\s*\"([^\"]+)\""])
        surveyor = _pick([r"surveyor\s+name\s*=\s*'([^']+)'", r"surveyor\s+name\s*=\s*\"([^\"]+)\""])
        surveyor_addr = _pick([r"surveyor\s+company\s+and\s+address\s*=\s*'([^']+)'", r"surveyor\s+company\s+and\s+address\s*=\s*\"([^\"]+)\""])

        pillar_list = _quoted_list(r"pillar\s+numbers\s*=\s*(.*?)(?:coordinates\s+for\s+the\s+points|$)")
        pillars = ", ".join(pillar_list)
        coords_blob = _pick([r"coordinates\s+for\s+the\s+points\s*=\s*(.+)$"]) or q
        
        # Parse access road(s): support one or multiple roads (e.g. "6m on side of A and B; 4m on side of C and D")
        access_roads: List[str] = []
        # 1) Named form: access road = '...' or "..." (may contain multiple specs separated by ; or " and ")
        access_road_quoted = _pick([r"access\s+road\s*=\s*'([^']+)'", r"access\s+road\s*=\s*\"([^\"]+)\""])
        if access_road_quoted:
            for part in re.split(r"\s*;\s*|\s+and\s+an?\s+access\s+|\s+,\s*and\s+an?\s+access\s+", access_road_quoted, flags=re.IGNORECASE):
                part = (part or "").strip()
                if part and (re.search(r"\d+(?:\.\d+)?\s*m", part) or "width" in part.lower()):
                    access_roads.append(part)
        # 2) Free-form: collect all road specs from the prompt (split by sentence or "and add ..." then match each)
        if not access_roads:
            segments = re.split(r"(?<=[.;])\s+|\s+and\s+add\s+an?\s+access\s+|\s+also\s+add\s+an?\s+access\s+|\s+,?\s+and\s+an?\s+access\s+", q, flags=re.IGNORECASE)
            for seg in segments:
                seg = (seg or "").strip()
                if not seg or not re.search(r"access|road|width|side\s+of|connecting|joining", seg, re.IGNORECASE):
                    continue
                m_ar = re.search(
                    r"access\s+road\s+of\s+(\d+(?:\.\d+)?)\s*m\s+width\s+(?:should\s+be\s+)?(?:on|along)\s+(?:the\s+)?(?:boundary\s+line\s+)?connecting\s+(.+?)(?:\.|$)",
                    seg,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_ar:
                    w, ref = m_ar.group(1), m_ar.group(2).strip()
                    spec = f"{w}m width on the boundary line connecting {ref}"
                    m_o = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", seg, re.IGNORECASE)
                    if m_o:
                        spec += f" offset {(m_o.group(1) or m_o.group(2))}m"
                    access_roads.append(spec)
                    continue
                m_ar2 = re.search(
                    r"(?:add\s+)?an?\s+access\s+road\s+of\s+width\s+(\d+(?:\.\d+)?)\s*m\s+.*?joining\s+pillars\s+(.+?)(?:\.|$)",
                    seg,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_ar2:
                    w, ref = m_ar2.group(1), m_ar2.group(2).strip()
                    spec = f"{w}m width on the side joining pillars {ref}"
                    m_o = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", seg, re.IGNORECASE)
                    if m_o:
                        spec += f" offset {(m_o.group(1) or m_o.group(2))}m"
                    access_roads.append(spec)
                    continue
                m_ar3 = re.search(
                    r"(?:add\s+)?an?\s+access\s+(?:road\s+)?of\s+width\s+(\d+(?:\.\d+)?)\s*m\s+.*?on\s+the\s+side\s+of\s+(.+?)(?:\.|$)",
                    seg,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_ar3:
                    w, ref = m_ar3.group(1), m_ar3.group(2).strip()
                    spec = f"{w}m width on the side of {ref}"
                    m_o = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", seg, re.IGNORECASE)
                    if m_o:
                        spec += f" offset {(m_o.group(1) or m_o.group(2))}m"
                    access_roads.append(spec)
            # Backward compat: if we still have nothing, run original single-road logic on full q
            if not access_roads:
                m_ar = re.search(
                    r"access\s+road\s+of\s+(\d+(?:\.\d+)?)\s*m\s+width\s+(?:should\s+be\s+)?(?:on|along)\s+(?:the\s+)?(?:boundary\s+line\s+)?connecting\s+(.+?)(?:\.|$)",
                    q,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if m_ar:
                    w, ref = m_ar.group(1), m_ar.group(2).strip()
                    access_road = f"{w}m width on the boundary line connecting {ref}"
                    m_offset_q = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
                    if m_offset_q:
                        access_road += f" offset {(m_offset_q.group(1) or m_offset_q.group(2))}m"
                    access_roads.append(access_road)
                else:
                    m_ar2 = re.search(
                        r"(?:add\s+)?an?\s+access\s+road\s+of\s+width\s+(\d+(?:\.\d+)?)\s*m\s+.*?joining\s+pillars\s+(.+?)(?:\.|$)",
                        q,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    if m_ar2:
                        w, ref = m_ar2.group(1), m_ar2.group(2).strip()
                        access_road = f"{w}m width on the side joining pillars {ref}"
                        m_offset_q = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
                        if m_offset_q:
                            access_road += f" offset {(m_offset_q.group(1) or m_offset_q.group(2))}m"
                        access_roads.append(access_road)
                    else:
                        m_ar3 = re.search(
                            r"(?:add\s+)?an?\s+access\s+(?:road\s+)?of\s+width\s+(\d+(?:\.\d+)?)\s*m\s+.*?on\s+the\s+side\s+of\s+(.+?)(?:\.|$)",
                            q,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if m_ar3:
                            w, ref = m_ar3.group(1), m_ar3.group(2).strip()
                            access_road = f"{w}m width on the side of {ref}"
                            m_offset_q = re.search(r"offset\s+of\s+(\d+(?:\.\d+)?)\s*m|offset\s+(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
                            if m_offset_q:
                                access_road += f" offset {(m_offset_q.group(1) or m_offset_q.group(2))}m"
                            access_roads.append(access_road)

        # Parse Concrete Wall Fence / Dwarf Concrete Wall Fence requests (C.W.F / D.C.W.F)
        # Supports multiple fences across different traverse legs, but max 1 fence per leg.
        fences: List[Dict[str, str]] = []
        fence_segments = re.split(r"(?<=[.;])\s+|\s+and\s+add\s+|\s+also\s+add\s+|\s+,\s*and\s+", q, flags=re.IGNORECASE)
        for seg in fence_segments:
            seg = (seg or "").strip()
            if not seg:
                continue
            seg_l = seg.lower()
            if not re.search(r"\bfence\b|c\.w\.f|d\.c\.w\.f|concrete\s+wall\s+fence|dwarf\s+concrete\s+wall\s+fence|short\s+wall\s+fence|wall\s+fence", seg_l, re.IGNORECASE):
                continue
            kind = None
            if re.search(r"d\.c\.w\.f|dwarf\s+concrete\s+wall\s+fence|short\s+wall\s+fence|dwarf\s+wall\s+fence", seg_l, re.IGNORECASE):
                kind = "DCWF"
            elif re.search(r"c\.w\.f|concrete\s+wall\s+fence|wall\s+fence|\bfence\b", seg_l, re.IGNORECASE):
                kind = "CWF"
            if not kind:
                continue
            # Only keep segments that reference a traverse leg (side of / joining / connecting / along / between)
            if not re.search(r"side\s+of|joining|connecting|along|between|linking", seg_l, re.IGNORECASE):
                continue
            fences.append({"kind": kind, "spec": seg})

        # Parse user-requested plot scale (e.g. "Plot using scale 1:250", "scale 1:250")
        user_scale_denom = None
        scale_m = re.search(r"plot\s+using\s+scale\s+1\s*:\s*(\d+)|scale\s+1\s*:\s*(\d+)", q, re.IGNORECASE)
        if scale_m:
            user_scale_denom = int(scale_m.group(1) or scale_m.group(2) or 0)
        if not user_scale_denom:
            scale_m = re.search(r"1\s*:\s*(\d+)\s*(?:scale|plot)", q, re.IGNORECASE)
            if scale_m:
                user_scale_denom = int(scale_m.group(1))

        # Parse optional road title override for first road (e.g. "title as 'UMUAKURU-UMUALILI ROAD'")
        access_road_title = None
        if access_roads:
            access_road_title = _pick([
                r"(?:road\s+)?title\s+as\s+['\"]([^'\"]+)['\"]",
                r"(?:give\s+it\s+the\s+)?title\s+as\s+['\"]([^'\"]+)['\"]",
                r"title\s+['\"]([^'\"]+)['\"]",
                r"(?:labeled|named)\s+['\"]([^'\"]+)['\"]",
                r"road\s+title\s+['\"]([^'\"]+)['\"]",
            ])
            if access_road_title:
                access_road_title = access_road_title.strip()

        resolved_from_memory = None
        if not template:
            resolved_from_memory = self._resolve_cadastral_template_from_memory(q)
            if resolved_from_memory:
                template = resolved_from_memory.get("template_path")

        if not output:
            return {"success": False, "error": "Could not parse output DWG from prompt."}
        if not template:
            return {
                "success": False,
                "error": (
                    "No CAD template was provided and no valid remembered cadastral template is available on this system. "
                    "Please provide a template DWG path once so SurvyAI can learn and remember it."
                ),
            }

        template_p = Path(template)
        if not template_p.is_absolute():
            template_p = (Path.cwd() / template_p).resolve()
        if not template_p.exists():
            return {"success": False, "error": f"Template DWG not found: {str(template_p)}"}
        out_p = Path(output)
        if not out_p.is_absolute():
            out_p = (Path.cwd() / out_p.name).resolve()

        profile_dir = self._cad_template_profiles_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = None
        if resolved_from_memory:
            try:
                p_mem = Path(str(resolved_from_memory.get("profile_path") or "")).resolve()
                if p_mem.exists():
                    profile_path = p_mem
            except Exception:
                profile_path = None
        if profile_path is None:
            profile_path = (profile_dir / f"{template_p.stem}.json").resolve()
        # (Re)learn profile when missing OR when template file changed (mtime/size) OR when profile points to a different template path.
        need_learn = not profile_path.exists()
        if not need_learn:
            try:
                import json as _json

                data = _json.loads(profile_path.read_text(encoding="utf-8"))
                prof_tp = str((data.get("template") or {}).get("path") or "")
                try:
                    prof_tp_res = str(Path(prof_tp).resolve()) if prof_tp else ""
                except Exception:
                    prof_tp_res = prof_tp
                cur_tp_res = str(template_p.resolve())
                sig = (data.get("template") or {}).get("signature") or {}
                cur_stat = template_p.stat()
                if prof_tp_res and prof_tp_res != cur_tp_res:
                    need_learn = True
                elif int(sig.get("size_bytes") or -1) != int(cur_stat.st_size):
                    need_learn = True
                elif int(sig.get("mtime_ns") or -1) != int(getattr(cur_stat, "st_mtime_ns", int(cur_stat.st_mtime * 1e9))):
                    need_learn = True
            except Exception:
                # If profile is corrupt/unreadable, re-learn safely.
                need_learn = True

        if need_learn:
            learned = self._learn_cadastral_template_profile(str(template_p), profile_output=str(profile_path))
            if not learned.get("success"):
                return learned

        result = self._apply_cadastral_template(
            profile_path=str(profile_path),
            template_override_path=str(template_p),
            output_dwg_path=str(out_p),
            buyer_name=buyer or "",
            location=location or "",
            lga=lga or "",
            state=state or "",
            origin_crs=origin or "",
            plan_number=plan_no or "",
            surveyor_name=surveyor or "",
            surveyor_company_address=surveyor_addr or "",
            pillar_numbers=pillars or "",
            coordinates=coords_blob,
            certification_date=cert_date,
            access_roads=access_roads,
            fences=fences,
            access_road_title=access_road_title,
            user_scale_denom=user_scale_denom,
        )
        if isinstance(result, dict) and result.get("success"):
            self._register_cad_template_memory(str(template_p), str(profile_path))
        return result

    def _learn_cadastral_template_profile(
        self,
        template_path: str,
        profile_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Template learner: store key table handles + basic layer/layout metadata from the template.
        """
        from pathlib import Path
        import json as _json
        import time

        tp = Path(template_path).resolve()
        if not tp.exists():
            return {"success": False, "error": f"Template not found: {str(tp)}"}
        outp = Path(profile_output).resolve() if profile_output else (Path("template_profiles") / f"{tp.stem}.json").resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}
        # Template is always opened read-only; the survey template must never be tampered with.
        opened = self.autocad.open_drawing(str(tp), read_only=True)
        if not opened.get("success"):
            return {"success": False, "error": opened.get("error", "Failed to open template")}

        drawing_info = {}
        try:
            drawing_info = self.autocad.get_drawing_info() or {}
        except Exception:
            drawing_info = {}

        tables = self.autocad.list_tables().get("tables", [])
        by_layer = {}
        for t in tables:
            by_layer.setdefault(str(t.get("layer")), []).append(t)

        # Identify coordinate tables by cell suffix
        def _cell(handle: str) -> str:
            r = self.autocad.get_table_cell_text(handle, 0, 0)
            return str(r.get("text") or "") if r.get("success") else ""

        coord_handles = []
        for lyr in ["CADA_COORDINATES", "CADA_NORTHCOORDINATES", "CADA_EASTCOORDINATES"]:
            coord_handles.extend([str(t.get("handle")) for t in by_layer.get(lyr, []) if t.get("handle")])
        
        e_h = n_h = None
        for h in coord_handles:
            txt = _cell(h).upper()
            if ".E" in txt:
                e_h = h
            if ".N" in txt:
                n_h = h

        inserts = self.autocad.list_inserts(layer="CADA_PILLARS").get("inserts", [])
        block_name = None
        if inserts:
            block_name = inserts[0].get("block_name")

        # Sample text height from bearing/distance and road layers for template-matching
        bearing_road_height = 1.2
        try:
            hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
            if hr.get("success") and "height" in hr:
                bearing_road_height = float(hr["height"])
        except Exception:
            pass

        # "Sheet" layers that should move together as one unit when recentring the plan.
        # This must include the border/boxes and any sheet text so tables stay inside their boxes.
        # Geometry layers (boundary/bearing/pegs) are intentionally excluded.
        # CADA_SCALEBAR: scale bar (and any hatching/hashing in it, e.g. survey_plan_template2.dwg) is taken from the template and scaled with the sheet.
        layers_in_template = [str(x) for x in (drawing_info.get("layers") or []) if str(x)]
        sheet_layers_default = [
            "CADA_BORDER",
            "CADA_INTERIORBORDER",
            "CADA_SCALEBAR",
            "CADA_NORTHARROW",
            "CADA_EASTARROW",
            "CADA_TITLEBLOCK",
            "CADA_PLANNUMBER",
            "CADA_CERTIFICATION",
            "CADA_SURVEYOR",
            "CADA_COORDINATES",
            "CADA_NORTHCOORDINATES",
            "CADA_EASTCOORDINATES",
            # Some templates use generic text layers for headings/labels.
            "TITLE",
            "text",
        ]
        # Keep only ones that exist in the template (plus the core ones even if layer list was unavailable).
        layers_upper = {l.upper() for l in layers_in_template}
        sheet_layers = []
        for l in sheet_layers_default:
            if not layers_upper or l.upper() in layers_upper:
                sheet_layers.append(l)

        # Layers we expect a cadastral template to contain; used as a sanity check / metadata.
        # Include any CADA_* layers present plus key generic title/text layers.
        layers_expected = []
        for l in layers_in_template:
            lu = l.upper()
            if lu.startswith("CADA_") or lu in ("TITLE", "TEXT"):
                layers_expected.append(l)

        profile = {
            "success": True,
            "template": {
                "path": str(tp),
                "name": tp.name,
                "learned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "signature": {
                    "size_bytes": int(tp.stat().st_size),
                    "mtime_ns": int(getattr(tp.stat(), "st_mtime_ns", int(tp.stat().st_mtime * 1e9))),
                },
            },
            "drawing_info": drawing_info,
            "layers_expected": layers_expected,
            "sheet_layers": sheet_layers,
            "text_heights": {"bearing_dist_road": bearing_road_height},
            "tables": {
                "title_block": {"handle": (by_layer.get("CADA_TITLEBLOCK", [{}])[0] or {}).get("handle")},
                "plan_number": {"handle": (by_layer.get("CADA_PLANNUMBER", [{}])[0] or {}).get("handle")},
                "surveyor": {"handle": (by_layer.get("CADA_SURVEYOR", [{}])[0] or {}).get("handle")},
                "certification": {"handle": (by_layer.get("CADA_CERTIFICATION", [{}])[0] or {}).get("handle")},
                "coordinates": {"easting_table_handle": e_h, "northing_table_handle": n_h},
                "pillar_numbers": {"tables": by_layer.get("CADA_PILLARNUMBERS", [])},
            },
            "blocks": {"pillars": {"block_name": block_name or "PEG_SYMBOL"}},
        }
        outp.write_text(_json.dumps(profile, indent=2), encoding="utf-8")
        # STRICT: Register template so it is never written (read-only to avoid corruption).
        self._protected_template_paths.add(str(tp.resolve()))
        return {"success": True, "profile_path": str(outp), "profile": profile}

    def _apply_cadastral_template(
        self,
        profile_path: str,
        output_dwg_path: str,
        buyer_name: str,
        location: str,
        lga: str,
        state: str,
        origin_crs: str,
        plan_number: str,
        surveyor_name: str,
        surveyor_company_address: str,
        pillar_numbers: str,
        template_override_path: Optional[str] = None,
        coordinates: Optional[str] = None,
        certification_date: Optional[str] = None,
        access_road: Optional[str] = None,
        access_roads: Optional[List[str]] = None,
        fences: Optional[List[Dict[str, str]]] = None,
        access_road_title: Optional[str] = None,
        user_scale_denom: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Normalize to list: support legacy single access_road
        roads_to_draw = access_roads if access_roads is not None else ([access_road] if access_road else [])
        fences_to_draw = fences or []
        import json as _json
        import math
        import re
        import shutil
        import time
        from pathlib import Path

        prof = Path(profile_path).resolve()
        if not prof.exists():
            return {"success": False, "error": f"Profile not found: {str(prof)}"}
        profile = _json.loads(prof.read_text(encoding="utf-8"))
        template_src = template_override_path or profile.get("template", {}).get("path", "")
        template = Path(str(template_src or "")).resolve()
        if not template.exists():
            return {"success": False, "error": f"Template not found: {str(template)}"}

        # ABSOLUTE RULE: The survey template DWG is never modified, regardless of user prompt.
        # We only read from it (or copy from it). All edits are made to the output drawing copy.

        outp = Path(output_dwg_path)
        if not outp.is_absolute():
            outp = (Path.cwd() / outp).resolve()

        # Copy template to output (overwrite); template file itself is never written to.
        # The full template (including scale bar and any scale bar hashing/hatching, e.g. survey_plan_template2.dwg) is preserved and then scaled if plan scale changes.
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(template), str(outp))
        except PermissionError as e:
            # If the target DWG is open/locked (common during iterative edits), write to a unique filename
            # in the same folder rather than failing the entire workflow.
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                alt = outp.with_name(f"{outp.stem}_{ts}{outp.suffix}")
                shutil.copy2(str(template), str(alt))
                outp = alt
                output_dwg_path = str(outp)
            except Exception as e2:
                return {
                    "success": False,
                    "error": f"Failed to copy template to output (permission denied; also failed alternate name): {e2}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to copy template to output: {e}"}

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}
        opened = self.autocad.open_drawing(str(outp))
        if not opened.get("success"):
            return {"success": False, "error": opened.get("error", "Failed to open output drawing")}
        time.sleep(0.3)

        # --- helpers for table formatting preservation ---
        def _get_cell(h: str, r: int, c: int = 0) -> str:
            res = self.autocad.get_table_cell_text(h, r, c)
            return str(res.get("text") or "") if res.get("success") else ""

        def _set_cell(h: str, r: int, c: int, val: str):
            return self.autocad.set_table_cell_text(h, r, c, val)

        def _mtxt_replace(raw: str, new_content: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                return raw[: idx + 1] + new_content + "}"
            return new_content

        def _replace_after_label(raw: str, label: str, new_tail: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                wrapper = raw[: idx + 1]
                content = raw[idx + 1 : -1]
                pos = content.upper().find(label.upper())
                if pos == -1:
                    new_content = f"{label} {new_tail}".strip()
                else:
                    keep = content[: pos + len(label)]
                    new_content = (keep + " " + new_tail).strip()
                return wrapper + new_content + "}"
            pos = raw.upper().find(label.upper())
            if pos == -1:
                return f"{label} {new_tail}".strip()
            return (raw[: pos + len(label)] + " " + new_tail).strip()

        tables = profile.get("tables", {})
        text_heights = profile.get("text_heights") or {}
        bearing_road_height = float(text_heights.get("bearing_dist_road", 1.2))
        if bearing_road_height == 1.2:
            try:
                hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
                if hr.get("success") and hr.get("height"):
                    bearing_road_height = float(hr["height"])
            except Exception:
                pass
        # Clamp only to a minimum to avoid corrupted or COM-default tiny height; no max so scaled height (e.g. 24 for 1:10000) is allowed
        bearing_road_height = max(0.5, float(bearing_road_height))
        title_h = str((tables.get("title_block") or {}).get("handle") or "")
        plan_h = str((tables.get("plan_number") or {}).get("handle") or "")
        surv_h = str((tables.get("surveyor") or {}).get("handle") or "")
        cert_h = str((tables.get("certification") or {}).get("handle") or "")
        east_h = str(((tables.get("coordinates") or {}).get("easting_table_handle")) or "")
        north_h = str(((tables.get("coordinates") or {}).get("northing_table_handle")) or "")

        # Title block
        if title_h:
            _set_cell(title_h, 2, 0, _mtxt_replace(_get_cell(title_h, 2), buyer_name.strip().upper()))
            _set_cell(title_h, 4, 0, _mtxt_replace(_get_cell(title_h, 4), location.strip().upper()))
            lga_u = lga.strip().upper()
            lga_line = lga_u if "LOCAL GOVERNMENT AREA" in lga_u else f"{lga_u} LOCAL GOVERNMENT AREA"
            _set_cell(title_h, 5, 0, _mtxt_replace(_get_cell(title_h, 5), lga_line))
            _set_cell(title_h, 6, 0, _mtxt_replace(_get_cell(title_h, 6), state.strip().upper()))
            _set_cell(title_h, 11, 0, _replace_after_label(_get_cell(title_h, 11), "ORIGIN:-", origin_crs.strip().upper()))

        # Snapshot table dimensions (rows/cols) from the active drawing to support robust updates
        tables_now = {}
        try:
            for t in (self.autocad.list_tables().get("tables") or []):
                h = str(t.get("handle") or "")
                if h:
                    tables_now[h] = t
        except Exception:
            tables_now = {}

        # Plan number + surveyor
        if plan_h:
            _set_cell(plan_h, 1, 0, _mtxt_replace(_get_cell(plan_h, 1), plan_number.strip().upper()))
        if surv_h:
            # Surveyor name: if it includes bracket text, render that bracket part at ~2/3 height.
            def _format_surveyor_name(raw: str) -> str:
                s = (raw or "").strip().upper()
                if not s:
                    return s
                # Prefer bracket at end: "SURV. ... (MNIS)"
                m = re.match(r"^(.*?)(\s*\([^)]*\))\s*$", s)
                if m:
                    main = m.group(1).strip()
                    br = m.group(2).strip()
                    # Use grouped MTEXT height override limited to bracket portion
                    # (keeps rest of cell style intact).
                    return f"{main} {{\\H0.67x;{br}}}".strip()
                return s

            _set_cell(surv_h, 0, 0, _mtxt_replace(_get_cell(surv_h, 0), _format_surveyor_name(surveyor_name)))

            # Surveyor address: user may provide address-only (no company). In that case,
            # keep it as address only (no template company leakage) and format cleanly.
            def _looks_like_company(text: str) -> bool:
                t = (text or "").upper()
                return any(k in t for k in [
                    " LTD", " LIMITED", " NIG", " NIGERIA", " COMPANY", " CO.", " SERVICES", " ENTERPRISE",
                    " GLOBAL", " VENTURES", " CONSULT", " CONSULTS", " CONSULTANCY", " GEO", " SURVEY",
                ])

            def _format_address(raw: str) -> str:
                addr = (raw or "").strip().upper()
                if not addr:
                    return addr
                # If caller already passed MTEXT newline codes, respect them.
                if "\\P" in addr:
                    return addr
                # Heuristic: address-only → at most 2 lines (street; city/state tail).
                parts = [p.strip() for p in re.split(r",\s*", addr) if p.strip()]
                if parts and not _looks_like_company(addr):
                    if len(parts) >= 2:
                        return parts[0] + "\\P" + ", ".join(parts[1:])
                    # Try a readable split for common localities if no commas
                    if " PORT HARCOURT" in addr:
                        return addr.replace(" PORT HARCOURT", "\\PPORT HARCOURT")
                    return addr
                # Company+address → allow multi-line split on commas
                if len(parts) >= 2:
                    return "\\P".join(parts)
                return addr

            addr_u = _format_address(surveyor_company_address)

            def _mtxt_set_with_style(existing_cell: str, new_content: str) -> str:
                """
                Build a new MTEXT string that preserves ONLY formatting codes (color/font),
                and replaces ALL textual content (prevents template company/address leakage).
                """
                raw = existing_cell or ""
                # If cell uses MTEXT wrapper, try to keep color + font style.
                if raw.startswith("{") and raw.endswith("}"):
                    color = ""
                    font = ""
                    try:
                        m = re.search(r"(\\C\d+;)", raw)
                        if m:
                            color = m.group(1)
                    except Exception:
                        color = ""
                    try:
                        m = re.search(r"(\\f[^;]+;)", raw)
                        if m:
                            font = m.group(1)
                    except Exception:
                        font = ""
                    if color or font:
                        return "{" + (color or "") + (font or "") + (new_content or "") + "}"
                # Fallback to existing wrapper replacement (best-effort)
                return _mtxt_replace(raw, new_content or "")

            # Apply to all columns in the address row (some templates use multiple columns)
            cols = int((tables_now.get(surv_h) or {}).get("cols") or 1)
            for c in range(max(1, cols)):
                cur = _get_cell(surv_h, 1, c)
                _set_cell(surv_h, 1, c, _mtxt_set_with_style(cur, addr_u))

        # Certification date
        if cert_h and certification_date:
            date_in = certification_date.strip()
            m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$", date_in)
            if m:
                dd, mm, yy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
                if len(yy) == 2:
                    yy = "20" + yy
                date_norm = f"{dd}-{mm}-{yy}"
            else:
                date_norm = date_in.replace("/", "-")
            date_pat = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")

            def _replace_any_date(raw: str) -> str:
                raw = raw or ""
                # Preserve template MTEXT wrapper if present
                if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                    idx = raw.rfind(";")
                    wrapper = raw[: idx + 1]
                    content = raw[idx + 1 : -1]
                    if date_pat.search(content):
                        content2 = date_pat.sub(date_norm, content, count=1)
                        return wrapper + content2 + "}"
                    # Fallback: replace after "ON" if present, else append
                    if re.search(r"\bON\b", content, flags=re.IGNORECASE):
                        content2 = re.sub(r"(\bON\b\s*)(.*)$", lambda mm: (mm.group(1) + date_norm).strip(), content, flags=re.IGNORECASE)
                        return wrapper + content2 + "}"
                    return wrapper + (content.rstrip() + f" ON {date_norm}").strip() + "}"
                if date_pat.search(raw):
                    return date_pat.sub(date_norm, raw, count=1)
                if re.search(r"\bON\b", raw, flags=re.IGNORECASE):
                    return re.sub(r"(\bON\b\s*)(.*)$", lambda mm: (mm.group(1) + date_norm).strip(), raw, flags=re.IGNORECASE)
                return (raw.rstrip() + f" ON {date_norm}").strip()

            cols = int((tables_now.get(cert_h) or {}).get("cols") or 1)
            rows = int((tables_now.get(cert_h) or {}).get("rows") or 1)
            for r in range(max(1, rows)):
                for c in range(max(1, cols)):
                    cur = _get_cell(cert_h, r, c)
                    upd = _replace_any_date(cur)
                    if upd != cur:
                        _set_cell(cert_h, r, c, upd)

        # Parse coordinate pairs from prompt
        coord_pairs = []
        bowditch_info = None
        bearing_distance_legs: Optional[List[Dict[str, float]]] = None
        if coordinates:
            pairs = re.findall(
                r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[eE]\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[nN]\s*\)",
                coordinates,
            )
            for e_s, n_s in pairs:
                coord_pairs.append({"e": float(e_s), "n": float(n_s)})

            # Fallback: single coordinate + bearings/distances traverse legs
            # Example inputs:
            # - "plot using coordinate 286638.060mE, 544692.450mN then bearing 17deg 49' and distance 16.14m ..."
            # - "use coordinate 286638.060mE, 544692.450mN; bearing = 17 degrees 49 min, distance = 16.14m ..."
            if len(coord_pairs) < 3:
                try:
                    # 1) Start coordinate (E,N)
                    m0 = re.search(
                        r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[eE]\s*[,; ]+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\s*[nN]\b",
                        coordinates,
                        flags=re.IGNORECASE,
                    )
                    if m0:
                        e0 = float(m0.group(1))
                        n0 = float(m0.group(2))

                        # 2) Traverse legs: bearing + distance (flexible phrasing)
                        leg_re = re.compile(
                            r"\bbearing\b\s*(?:=|is|:)?\s*"
                            r"(\d{1,3})\s*(?:deg|degree|degrees|°|d)\s*"
                            r"([0-5]?\d)\s*(?:min|mins|minute|minutes|['’])"
                            r"(?:[^0-9]{0,80}?)"
                            r"(?:distance|dist\.?|measured\s+distance)\s*(?:=|is|:)?\s*"
                            r"([0-9]+(?:\.[0-9]+)?)\s*(?:m)?\b",
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        legs = []
                        for mm in leg_re.finditer(coordinates):
                            d = int(mm.group(1))
                            m = int(mm.group(2))
                            dist = float(mm.group(3))
                            # Normalize bearing to decimal degrees (from North, clockwise)
                            bdeg = (float(d) % 360.0) + (float(m) / 60.0)
                            legs.append((bdeg, dist))

                        if legs:
                            # Build unadjusted deltas + vertices
                            deltas = []
                            tmp_unadj = [{"e": float(e0), "n": float(n0)}]
                            ce, cn = float(e0), float(n0)
                            total_len = 0.0
                            for (bdeg, dist) in legs:
                                br = math.radians(float(bdeg))
                                de = float(dist) * math.sin(br)
                                dn = float(dist) * math.cos(br)
                                deltas.append((de, dn, float(dist)))
                                total_len += float(dist)
                                ce += de
                                cn += dn
                                tmp_unadj.append({"e": float(ce), "n": float(cn)})

                            mis_e = float(tmp_unadj[-1]["e"] - tmp_unadj[0]["e"])
                            mis_n = float(tmp_unadj[-1]["n"] - tmp_unadj[0]["n"])
                            mis = float(math.hypot(mis_e, mis_n))
                            threshold = 0.01  # 1 cm

                            tmp = tmp_unadj
                            applied = False
                            max_shift = 0.0
                            method_used = None
                            db_deg = None
                            bearing_distance_legs = [{"bearing_deg": float(bd), "distance": float(di)} for (bd, di) in legs]

                            # Default: bearing-adjustment (keep distances fixed).
                            # Bowditch is ONLY used when the user explicitly mentions it.
                            wants_bowditch = bool(re.search(r"\bbowditch\b", coordinates or "", flags=re.IGNORECASE))

                            if mis > threshold and total_len > 1e-9:
                                if wants_bowditch:
                                    method_used = "bowditch"
                                    applied = True
                                    tmp_adj = [{"e": float(e0), "n": float(n0)}]
                                    ce2, cn2 = float(e0), float(n0)
                                    for (de, dn, Li) in deltas:
                                        # distribute misclosure proportional to line length
                                        cde = (-mis_e) * (float(Li) / float(total_len))
                                        cdn = (-mis_n) * (float(Li) / float(total_len))
                                        ce2 += float(de + cde)
                                        cn2 += float(dn + cdn)
                                        tmp_adj.append({"e": float(ce2), "n": float(cn2)})
                                    tmp = tmp_adj

                                    # compute max point shift (excluding start)
                                    for k in range(1, min(len(tmp_unadj), len(tmp_adj))):
                                        sh = math.hypot(tmp_adj[k]["e"] - tmp_unadj[k]["e"], tmp_adj[k]["n"] - tmp_unadj[k]["n"])
                                        if sh > max_shift:
                                            max_shift = float(sh)
                                else:
                                    method_used = "bearing_adjustment"
                                    try:
                                        from tools import traverse_bearing_adjustment as _tba
                                        import numpy as _np

                                        b_list = [float(bd) for (bd, _di) in legs]
                                        d_list = [float(_di) for (_bd, _di) in legs]
                                        trv = _tba.traverse_from_agent(e0, n0, b_list, d_list, bearings_in_degrees=True)
                                        ba = _tba.form_matrices(trv)  # adjusted bearings in radians
                                        db = ba - _np.deg2rad(_np.asarray(b_list, dtype=float))
                                        db_deg = (_np.rad2deg(db)).tolist()
                                        bearing_distance_legs = [
                                            {"bearing_deg": float(_np.rad2deg(ba[i])), "distance": float(d_list[i])}
                                            for i in range(len(d_list))
                                        ]

                                        D_adj = _np.asarray(d_list, dtype=float) * _np.sin(ba)
                                        L_adj = _np.asarray(d_list, dtype=float) * _np.cos(ba)

                                        tmp_adj = [{"e": float(e0), "n": float(n0)}]
                                        ce2, cn2 = float(e0), float(n0)
                                        for i in range(len(d_list)):
                                            ce2 += float(D_adj[i])
                                            cn2 += float(L_adj[i])
                                            tmp_adj.append({"e": float(ce2), "n": float(cn2)})
                                        tmp = tmp_adj

                                        for k in range(1, min(len(tmp_unadj), len(tmp_adj))):
                                            sh = math.hypot(tmp_adj[k]["e"] - tmp_unadj[k]["e"], tmp_adj[k]["n"] - tmp_unadj[k]["n"])
                                            if sh > max_shift:
                                                max_shift = float(sh)
                                        applied = True
                                    except Exception:
                                        # If bearing-adjustment fails for any reason, keep unadjusted points
                                        method_used = "bearing_adjustment_failed"
                                        applied = False

                            # For start-coordinate + bearing/distance parcel traverses, the final computed point
                            # is the closure back to the start and must NOT be treated as an extra pillar.
                            # Keep only the unique parcel vertices; boundary polyline closure is handled separately.
                            if len(legs) >= 3 and len(tmp) == len(legs) + 1:
                                tmp = tmp[:-1]
                            # Safety fallback for any other near-duplicate closure point
                            elif len(tmp) >= 4:
                                dx0 = tmp[-1]["e"] - tmp[0]["e"]
                                dy0 = tmp[-1]["n"] - tmp[0]["n"]
                                if math.hypot(dx0, dy0) <= 0.25:
                                    tmp = tmp[:-1]

                            if len(tmp) >= 3:
                                coord_pairs = tmp
                                bowditch_info = {
                                    "mode": "bearing_distance",
                                    "method": method_used or ("bowditch" if wants_bowditch else "bearing_adjustment"),
                                    "applied": bool(applied),
                                    "threshold_m": float(threshold),
                                    "misclosure_m": float(mis),
                                    "misclosure_e_m": float(mis_e),
                                    "misclosure_n_m": float(mis_n),
                                    "perimeter_m": float(total_len),
                                    "max_point_shift_m": float(max_shift),
                                    "db_deg": db_deg,
                                    # Keep preview small (most jobs are 4-10 legs)
                                    "adjusted_points_preview": [
                                        {"idx": i, "e": float(p["e"]), "n": float(p["n"])}
                                        for i, p in enumerate(coord_pairs[: min(6, len(coord_pairs))])
                                    ],
                                }
                except Exception:
                    pass

        geometry = {"pillars_moved": 0, "boundary_redrawn": False, "bearing_mtext": 0, "access_road_title": None}
        if bowditch_info:
            geometry["bowditch"] = bowditch_info

        if coord_pairs and len(coord_pairs) >= 3:
            # Preserve user input order: pillar i is assigned to coordinate i (first pillar to first coord, etc.)
            # Use first coordinate as primary (e0, n0) so vertex order = coordinate list order.
            primary = coord_pairs[0]
            e0, n0 = primary["e"], primary["n"]
            pts = list(coord_pairs)

            # Existing pillar inserts in output (copied from template)
            ins = self.autocad.list_inserts(layer="CADA_PILLARS")
            pillar_ins = (ins.get("inserts") or []) if isinstance(ins, dict) else []
            # Template primary position in drawing coords (min X)
            def _ix(i):
                pt = i.get("insertion_point") or {}
                return float(pt.get("x", 0.0))
            primary_ins = sorted(pillar_ins, key=_ix)[0] if pillar_ins else None
            base_x = float((primary_ins.get("insertion_point") or {}).get("x", 0.0)) if primary_ins else 0.0
            base_y = float((primary_ins.get("insertion_point") or {}).get("y", 0.0)) if primary_ins else 0.0

            # Decide whether the template must be upscaled to the next allowed smaller plan scale
            # (e.g. template labeled 1:500 → treat output as 1:1000) BEFORE plotting/aligning,
            # so the result stays neat like the template.
            scale_k = 1.0
            chosen_denom = 500  # plan scale 1:chosen_denom; used for road min-length check and title
            try:
                allowed_denoms = [250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000]

                # Parse template denom from title block (best-effort)
                template_denom = 500
                if title_h:
                    try:
                        tbl = tables_now.get(title_h, {}) if isinstance(tables_now, dict) else {}
                        rows = int(tbl.get("rows", 25))
                        cols = int(tbl.get("cols", 2))
                        # Parse template scale from title block. We must use the MAIN scale (e.g. 1:500), never "SCALE: to 1:1000".
                        scale_pat = re.compile(r"1\s*:\s*(\d+)", re.IGNORECASE)
                        secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
                        main_scale_re = re.compile(r"\bSCALE\b\s*:-?\s*", re.IGNORECASE)  # "SCALE:-" or "SCALE : -"
                        candidates = []
                        for r in range(min(rows, 60)):
                            for c in range(min(cols, 10)):
                                cell = _get_cell(title_h, r, c) or ""
                                if secondary_re.search(cell):
                                    continue
                                m = scale_pat.search(cell)
                                if not m:
                                    continue
                                d = int(m.group(1))
                                if d not in allowed_denoms:
                                    continue
                                is_main = "SCALE" in cell.upper() and main_scale_re.search(cell) and not secondary_re.search(cell)
                                candidates.append((is_main, r, d))
                        if candidates:
                            main_candidates = [c for c in candidates if c[0]]
                            if main_candidates:
                                template_denom = max(c[2] for c in main_candidates)
                            else:
                                template_denom = min(c[2] for c in candidates)
                            found = True
                        else:
                            found = False
                    except Exception:
                        template_denom = 500
                if template_denom not in allowed_denoms:
                    template_denom = 500
                chosen_denom = template_denom  # plan scale denominator (1:chosen_denom); may be updated below

                # Interior border bbox (template coords)
                interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                if not interior_bb.get("success"):
                    interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBOUNDARY"])
                if not interior_bb.get("success"):
                    interior_bb = self.autocad.get_modelspace_bbox(block_name_contains="INTERIOR", prefer_largest=True)

                # Boundary extents from user coordinates (meters)
                es = [float(p.get("e", 0.0)) for p in pts]
                ns = [float(p.get("n", 0.0)) for p in pts]
                boundary_w = (max(es) - min(es)) if es else 0.0
                boundary_h = (max(ns) - min(ns)) if ns else 0.0

                # Debug always returned so we can see what happened
                geometry["scale_debug"] = {
                    "template_denom": int(template_denom),
                    "boundary_w": float(boundary_w),
                    "boundary_h": float(boundary_h),
                    "interior_found": bool(interior_bb.get("success")),
                }

                if interior_bb.get("success") and boundary_w > 1e-6 and boundary_h > 1e-6:
                    interior_w = float(interior_bb.get("maxx", 0.0)) - float(interior_bb.get("minx", 0.0))
                    interior_h = float(interior_bb.get("maxy", 0.0)) - float(interior_bb.get("miny", 0.0))
                    margin = 0.08
                    interior_usable_w = interior_w * (1.0 - 2.0 * margin)
                    interior_usable_h = interior_h * (1.0 - 2.0 * margin)
                    geometry["scale_debug"].update({
                        "interior_w": float(interior_w),
                        "interior_h": float(interior_h),
                        "interior_usable_w": float(interior_usable_w),
                        "interior_usable_h": float(interior_usable_h),
                        "margin": float(margin),
                    })
                    if interior_usable_w > 1e-6 and interior_usable_h > 1e-6:
                        required_k = max(boundary_w / interior_usable_w, boundary_h / interior_usable_h)
                        geometry["scale_debug"]["required_k"] = float(required_k)
                        chosen_denom = template_denom
                        # Prefer user-requested scale if it meets minimum standards and boundary fits
                        if user_scale_denom and user_scale_denom in allowed_denoms:
                            scale_k_user = float(user_scale_denom) / float(template_denom)
                            if scale_k_user >= required_k:
                                chosen_denom = user_scale_denom
                                scale_k = scale_k_user
                                geometry["scale_debug"].update({
                                    "user_scale_used": True,
                                    "chosen_denom": int(chosen_denom),
                                    "k": float(scale_k),
                                })
                        if scale_k == 1.0 and required_k > 1.0 + 1e-6:
                            target_denom = float(template_denom) * float(required_k) * 1.02
                            candidates = [s for s in allowed_denoms if s >= target_denom and s >= template_denom]
                            chosen_denom = min(candidates) if candidates else max(allowed_denoms)
                            scale_k = float(chosen_denom) / float(template_denom)
                            geometry["scale_debug"].update({
                                "target_denom": float(target_denom),
                                "chosen_denom": int(chosen_denom),
                                "k": float(scale_k),
                            })

                        if scale_k != 1.0:
                            # Scale the template/sheet about the TEMPLATE PRIMARY PILLAR (base_x/base_y).
                            # This preserves arrow/coordinate geometry emanating from the pillar.
                            layers_to_scale = list(profile.get("sheet_layers") or []) or [
                                "CADA_BORDER",
                                "CADA_INTERIORBORDER",
                                "CADA_SCALEBAR",
                                "CADA_NORTHARROW",
                                "CADA_EASTARROW",
                                "CADA_TITLEBLOCK",
                                "CADA_PLANNUMBER",
                                "CADA_CERTIFICATION",
                                "CADA_SURVEYOR",
                                "CADA_COORDINATES",
                                "CADA_NORTHCOORDINATES",
                                "CADA_EASTCOORDINATES",
                                "CADA_PRIMARYPILLAR_ARROWS",
                                "TITLE",
                                "text",
                            ]
                            # Ensure critical template layers are always included even if the learned profile omitted them
                            for req in [
                                "CADA_BORDER",
                                "CADA_INTERIORBORDER",
                                "CADA_NORTHARROW",
                                "CADA_EASTARROW",
                                "CADA_NORTHCOORDINATES",
                                "CADA_EASTCOORDINATES",
                                "CADA_COORDINATES",
                                "CADA_PRIMARYPILLAR_ARROWS",
                                "CADA_SCALEBAR",
                                "CADA_TITLEBLOCK",
                            ]:
                                if req not in layers_to_scale:
                                    layers_to_scale.append(req)
                            # Scale pillar-number TABLES too (they are part of the template look)
                            layers_to_scale += ["CADA_PILLARNUMBERS"]
                            sc = self.autocad.scale_modelspace_by_layers(base_x, base_y, scale_k, layers_to_scale)
                            geometry["scale_debug"]["scaled_entities"] = int(sc.get("scaled_entities", 0) or 0) if sc.get("success") else 0

                            # Keep scalebar labels consistent with the new plan scale.
                            # The label factor is the same ratio used for the plan scale change (e.g., 1:500 -> 1:250 => 0.5).
                            try:
                                sb_factor = float(chosen_denom) / float(template_denom)
                                if abs(sb_factor - 1.0) > 1e-9:
                                    self.autocad.scale_scalebar_text_values(sb_factor, layers=["scalebar", "CADA_SCALEBAR"])
                            except Exception:
                                pass

                            # Scale bar hashing: scaling is applied once. The scale bar is a block insert on
                            # CADA_SCALEBAR; scale_modelspace_by_layers above already scales that insert by scale_k,
                            # so the block (and the hatch inside it) is scaled once. We do NOT also call
                            # scale_hatch_pattern_scale_by_layers(scale_k), or the hatch would be scaled twice
                            # (insert scale_k × PatternScale scale_k = scale_k²).

                            # Force hatch regen so the scalebar hashing updates immediately
                            try:
                                self.autocad.execute_command("REGEN")
                            except Exception:
                                pass

                            # Ensure new bearing/road text is created at the correct size
                            try:
                                bearing_road_height = float(bearing_road_height) * float(scale_k)
                            except Exception:
                                pass

                            # Edit only the existing main scale text in CADA_TITLEBLOCK.
                            # IMPORTANT: Clear the extra "SCALE: to 1:xxxx" line (row 9 should be empty).
                            if title_h:
                                try:
                                    scale_pattern = re.compile(r"1\s*:\s*\d+", re.IGNORECASE)
                                    replacement = f"1:{chosen_denom}"
                                    # Locate the main "SCALE:- 1:xxx" cell and update only that one.
                                    # Also remove any secondary "SCALE: to 1:xxx" cell so it stays blank.
                                    tbl = tables_now.get(title_h, {}) if isinstance(tables_now, dict) else {}
                                    rows = int(tbl.get("rows", 25))
                                    cols = int(tbl.get("cols", 2))

                                    main_scale_cell = None  # (r, c)
                                    secondary_scale_cells = []  # [(r, c), ...]
                                    secondary_re = re.compile(r"\bSCALE\b\s*:.*\bto\b", re.IGNORECASE)
                                    main_hint_re = re.compile(r"\bSCALE\b\s*[:-]", re.IGNORECASE)  # SCALE:- / SCALE:

                                    for r in range(min(rows, 60)):
                                        for c in range(min(cols, 10)):
                                            cell = _get_cell(title_h, r, c) or ""
                                            if not cell.strip():
                                                continue
                                            if secondary_re.search(cell) and scale_pattern.search(cell):
                                                secondary_scale_cells.append((r, c))
                                                continue
                                            if main_scale_cell is None and main_hint_re.search(cell) and scale_pattern.search(cell):
                                                # Prefer the explicit "SCALE:- 1:xxx" style cell.
                                                main_scale_cell = (r, c)

                                    # Fallback to the previously-known position if we didn't locate it.
                                    if main_scale_cell is None:
                                        main_scale_cell = (8, 0)

                                    mr, mc = main_scale_cell
                                    cell_main = _get_cell(title_h, mr, mc) or ""
                                    new_cell_main = scale_pattern.sub(replacement, cell_main)
                                    if new_cell_main != cell_main:
                                        _set_cell(title_h, mr, mc, new_cell_main)
                                    elif not (cell_main or "").strip():
                                        _set_cell(title_h, mr, mc, f"SCALE:- {replacement}")

                                    # Clear secondary scale cells (row 9 should be empty / no duplicate scale line).
                                    for (sr, sc_) in secondary_scale_cells:
                                        try:
                                            _set_cell(title_h, sr, sc_, "")
                                        except Exception:
                                            pass

                                    # Extra safety: if row 9 contains a secondary "SCALE: to ..." line, blank it.
                                    # (We only clear it when it matches the secondary pattern to avoid wiping other content.)
                                    for rr in (9, 8):  # handle possible off-by-one table indexing variations
                                        for cc in range(min(cols, 10)):
                                            v = _get_cell(title_h, rr, cc) or ""
                                            if v.strip() and secondary_re.search(v) and scale_pattern.search(v):
                                                _set_cell(title_h, rr, cc, "")
                                except Exception:
                                    pass
            except Exception:
                scale_k = 1.0

            local_pts = [{"x": base_x + (p["e"] - e0), "y": base_y + (p["n"] - n0)} for p in pts]

            # Clear old parcel graphics (not tables/border)
            self.autocad.delete_entities("CADA_BEARING_DIST")
            self.autocad.delete_entities("CADA_BOUNDARY")
            self.autocad.delete_entities("CADA_PILLARS")
            self.autocad.delete_entities("CADA_ROAD")
            self.autocad.delete_entities("CADA_CWF")
            self.autocad.delete_entities("CADA_TEXT")
            time.sleep(0.2)
            # IMPORTANT: Do NOT delete generic sheet/title layers; they are part of the template
            # border/title presentation and must remain aligned with the border boxes.

            # Insert pillar blocks at EVERY boundary vertex (professional cadastral plan behavior).
            # Use a robust strategy: first try AutoCAD InsertBlock; if it fails (common COM quirk),
            # fall back to cloning one template peg block and moving it to each vertex.
            blk = profile.get("blocks", {}).get("pillars", {}).get("block_name") or "PEG_SYMBOL"
            inserted_ok = True
            for p in local_pts:
                r = self.autocad.insert_block(
                    str(blk),
                    p["x"],
                    p["y"],
                    layer="CADA_PILLARS",
                    xscale=float(scale_k),
                    yscale=float(scale_k),
                    zscale=float(scale_k),
                )
                if not r.get("success"):
                    inserted_ok = False
                    break

            if not inserted_ok:
                # Rebuild from template pegs: open template, copy one peg, then paste multiple copies.
                try:
                    # Clear any partial inserts
                    self.autocad.delete_entities("CADA_PILLARS")
                except Exception:
                    pass
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                try:
                    acad = self.autocad.acad
                    out_doc = self.autocad.doc
                    # open template invisibly/best-effort
                    tdoc = None
                    for d in acad.Documents:
                        try:
                            if str(getattr(d, "FullName", "")).lower() == str(template).lower():
                                tdoc = d
                                break
                        except Exception:
                            pass
                    if tdoc is None:
                        tdoc = acad.Documents.Open(str(template), True)
                    # find first peg in template
                    peg_ent = None
                    ms_t = tdoc.ModelSpace
                    for ii in range(ms_t.Count):
                        e = ms_t.Item(ii)
                        if str(getattr(e, "Layer", "")).upper() == "CADA_PILLARS" and "BlockReference" in str(getattr(e, "ObjectName", "")):
                            peg_ent = e
                            break
                    if peg_ent is not None:
                        peg_ent.Copy()
                        ms_out = out_doc.ModelSpace
                        for p in local_pts:
                            new_ent = ms_out.Paste()
                            try:
                                new_ent.Layer = "CADA_PILLARS"
                            except Exception:
                                pass
                            # Ensure pillar symbol scales to chosen plan scale
                            try:
                                for attr in ("XScaleFactor", "YScaleFactor", "ZScaleFactor"):
                                    setattr(new_ent, attr, float(scale_k))
                            except Exception:
                                pass
                            # move pasted peg so its insertion aligns to vertex
                            ip = getattr(new_ent, "InsertionPoint", None)
                            if ip is not None:
                                dxm = float(p["x"]) - float(ip[0])
                                dym = float(p["y"]) - float(ip[1])
                                try:
                                    new_ent.Move((0.0, 0.0, 0.0), (dxm, dym, 0.0))
                                except Exception:
                                    pass
                    try:
                        if tdoc is not None and tdoc != out_doc:
                            out_doc.Activate()
                    except Exception:
                        pass
                except Exception:
                    pass
            time.sleep(0.15)

            geometry["pillars_moved"] = 0

            # Boundary: red by layer, closed polyline
            self.autocad.set_layer_color("CADA_BOUNDARY", 1)
            pl = self.autocad.create_lwpolyline(local_pts, layer="CADA_BOUNDARY", closed=True)
            time.sleep(0.1)
            if pl.get("success"):
                geometry["boundary_redrawn"] = True
                # Area -> title block
                a = self.autocad.calculate_entity_area(str(pl.get("handle")))
                if a.get("success") and title_h:
                    sq_m = float(a.get("area_conversions", {}).get("sq_meters"))
                    _set_cell(title_h, 12, 0, _replace_after_label(_get_cell(title_h, 12), "AREA:-", f"{sq_m:.3f} SQ. MTRS."))

            # Update coordinate tables to primary
            def _replace_first_num(raw: str, val: float) -> str:
                if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                    idx = raw.rfind(";")
                    wrapper = raw[: idx + 1]
                    content = raw[idx + 1 : -1]
                    return wrapper + re.sub(r"[-+]?\d+(?:\.\d+)?", f"{val:.3f}", content, count=1) + "}"
                return re.sub(r"[-+]?\d+(?:\.\d+)?", f"{val:.3f}", raw, count=1)
            if east_h:
                _set_cell(east_h, 0, 0, _replace_first_num(_get_cell(east_h, 0), e0))
            if north_h:
                _set_cell(north_h, 0, 0, _replace_first_num(_get_cell(north_h, 0), n0))

            # Update pillar number tables (CADA_PILLARNUMBERS) to match user prompt and
            # position them near each corresponding boundary vertex (clockwise from primary).
            # Pillar-number entities in the template are TABLE objects on this layer.
            def _parse_pillar_numbers(raw: str) -> List[Dict[str, str]]:
                raw = (raw or "").strip()
                if not raw:
                    return []
                parts = [p.strip() for p in re.split(r"[,\n]+", raw) if p.strip()]
                out = []
                for p in parts:
                    m = re.search(r"([A-Za-z]+\s*/\s*[A-Za-z]+)\s*([0-9]+)\b", p)
                    if not m:
                        continue
                    prefix = re.sub(r"\s+", "", m.group(1)).upper()
                    num = m.group(2)
                    out.append({"prefix": prefix, "number": num})
                return out

            pn_list = _parse_pillar_numbers(pillar_numbers)
            # If the agent extracted fewer pillar numbers than vertices (often due to punctuation/quoting),
            # auto-extend sequentially using the last known prefix/number so we never drop pillar labels.
            try:
                need_n = len(local_pts) if isinstance(local_pts, list) else 0
                if need_n and pn_list and len(pn_list) < need_n:
                    last = pn_list[-1]
                    prefix = str(last.get("prefix") or "").strip() or "SP"
                    try:
                        start_num = int(str(last.get("number") or "0").strip())
                    except Exception:
                        start_num = 0
                    k = start_num
                    while len(pn_list) < need_n and k < start_num + 2000:
                        k += 1
                        pn_list.append({"prefix": prefix, "number": f"{k:04d}"})
            except Exception:
                pass
            pn_meta: List[Dict[str, Any]] = []
            if pn_list:
                # Compute a "template-typical" offset distance between a peg and its pillar-number table.
                # This makes the placement look like the template: close to the peg, but not on it.
                off = 4.0
                try:
                    acad = self.autocad.acad
                    tdoc = None
                    for d in acad.Documents:
                        try:
                            if str(getattr(d, "FullName", "")).lower() == str(template).lower():
                                tdoc = d
                                break
                        except Exception:
                            pass
                    if tdoc is None:
                        # CRITICAL: Always open template read-only. The survey template must never be tampered with.
                        tdoc = acad.Documents.Open(str(template), True)
                    ms_t = tdoc.ModelSpace
                    t_pegs = []
                    t_tabs = []
                    for ii in range(ms_t.Count):
                        e = ms_t.Item(ii)
                        lyr = str(getattr(e, "Layer", "")).upper()
                        on = str(getattr(e, "ObjectName", ""))
                        if lyr == "CADA_PILLARS" and "BlockReference" in on:
                            ip = getattr(e, "InsertionPoint", None)
                            if ip is not None:
                                t_pegs.append((float(ip[0]), float(ip[1])))
                        if lyr == "CADA_PILLARNUMBERS" and on == "AcDbTable":
                            ip = None
                            for attr in ("InsertionPoint", "Position"):
                                try:
                                    ip = getattr(e, attr, None)
                                    if ip is not None:
                                        break
                                except Exception:
                                    pass
                            if ip is not None:
                                t_tabs.append((float(ip[0]), float(ip[1])))
                    dists = []
                    if t_pegs and t_tabs:
                        for px, py in t_pegs:
                            tx, ty = min(t_tabs, key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2)
                            dists.append(math.hypot(tx - px, ty - py))
                    if dists:
                        dists.sort()
                        off = float(dists[len(dists) // 2])
                except Exception:
                    off = 4.0
                off = max(2.5, min(8.0, off))

                # Reuse pillar-number tables already in the copied output (from template),
                # move them close (but not on) each new pillar, and delete any extras from the template.
                t_res = self.autocad.list_tables(layer="CADA_PILLARNUMBERS")
                pn_tables = (t_res.get("tables") or []) if isinstance(t_res, dict) else []
                used_handles: set[str] = set()
                pn_meta: List[Dict[str, Any]] = []
                cxp = sum(p["x"] for p in local_pts) / len(local_pts)
                cyp = sum(p["y"] for p in local_pts) / len(local_pts)
                pillar_box_bbs: List[Dict[str, Any]] = []
                try:
                    pbb = self.autocad.list_entity_bboxes(
                        layers=["CADA_PILLARS"],
                        object_names=["AcDbBlockReference"],
                    )
                    if (pbb or {}).get("success"):
                        pillar_box_bbs = list(pbb.get("bboxes") or [])
                except Exception:
                    pillar_box_bbs = []

                def _dist2(t, vx, vy):
                    ip = (t.get("insertion_point") or {})
                    tx, ty = float(ip.get("x", 0.0)), float(ip.get("y", 0.0))
                    return (tx - vx) ** 2 + (ty - vy) ** 2

                def _nearest_pillar_bb(vx: float, vy: float) -> Optional[Dict[str, Any]]:
                    if not pillar_box_bbs:
                        return None
                    def _bb_d2(b: Dict[str, Any]) -> float:
                        mn = b.get("min") or {}
                        mx = b.get("max") or {}
                        cx = (float(mn.get("x", 0.0)) + float(mx.get("x", 0.0))) / 2.0
                        cy = (float(mn.get("y", 0.0)) + float(mx.get("y", 0.0))) / 2.0
                        return (cx - vx) ** 2 + (cy - vy) ** 2
                    return min(pillar_box_bbs, key=_bb_d2)

                for i_v, v in enumerate(local_pts[: len(pn_list)]):
                    vx, vy = float(v["x"]), float(v["y"])
                    cand = [t for t in pn_tables if t.get("handle") and str(t.get("handle")) not in used_handles]
                    if not cand:
                        # Template may contain fewer pillar-number tables than parcel vertices.
                        # Clone an existing table so all pillar numbers are preserved.
                        try:
                            seed = None
                            if pn_tables:
                                # Prefer the last used handle as it matches styling/scale in this drawing.
                                for hh in reversed(list(used_handles)):
                                    if hh:
                                        seed = hh
                                        break
                                if seed is None:
                                    seed = str((pn_tables[0] or {}).get("handle") or "")
                            if seed:
                                c = self.autocad.copy_entity_by_handle(seed, dx=0.0, dy=0.0, layer="CADA_PILLARNUMBERS")
                                if (c or {}).get("success") and c.get("handle"):
                                    new_h = str(c.get("handle"))
                                    pn_tables.append({"handle": new_h, "layer": "CADA_PILLARNUMBERS", "insertion_point": c.get("insertion_point") or {}})
                                    cand = [t for t in pn_tables if t.get("handle") and str(t.get("handle")) not in used_handles]
                        except Exception:
                            cand = cand
                        if not cand:
                            # Hard fallback: create an MTEXT-based pillar-number label so we never drop a pillar number.
                            try:
                                lbl = f"{pn_list[i_v]['prefix']}\\P{pn_list[i_v]['number']}"
                                self.autocad.add_mtext(
                                    f"{{\\fVerdana|b0|i0|c0|p34;{lbl}}}",
                                    vx + 2.0,
                                    vy + 2.0,
                                    layer="CADA_PILLARNUMBERS",
                                    rotation_rad=0.0,
                                    height=max(0.5, float(bearing_road_height) * 0.9),
                                    width=12.0,
                                    attachment_point=1,  # top-left
                                )
                            except Exception:
                                pass
                            continue
                    best = min(cand, key=lambda t: _dist2(t, vx, vy))
                    h = str(best.get("handle"))
                    used_handles.add(h)
                    try:
                        _set_cell(h, 0, 0, _mtxt_replace(_get_cell(h, 0), pn_list[i_v]["prefix"]))
                        _set_cell(h, 1, 0, _mtxt_replace(_get_cell(h, 1), pn_list[i_v]["number"]))
                    except Exception:
                        pass
                    # Place the TABLE close to the pillar but not on it.
                    # TABLE insertion points are often at a corner, so use its bounding box
                    # to position the *nearest table edge* a small gap away from the pillar.
                    dxv, dyv = vx - cxp, vy - cyp
                    Lvv = math.hypot(dxv, dyv) or 1.0
                    ux, uy = dxv / Lvv, dyv / Lvv  # outward from centroid
                    try:
                        pn_meta.append({"handle": h, "vx": vx, "vy": vy, "ux": ux, "uy": uy, "off": float(off)})
                    except Exception:
                        pass
                    # Gap target (survey drafting): ~1.0 unit from peg, except primary peg ~1.5 units.
                    gap = 1.5 if i_v == 0 else 1.0
                    try:
                        ms = self.autocad.doc.ModelSpace
                        ent = None
                        for ii in range(ms.Count):
                            e = ms.Item(ii)
                            if getattr(e, "Handle", None) == h:
                                ent = e
                                break
                        if ent is not None:
                            # insertion point
                            ip = None
                            for attr in ("InsertionPoint", "Position"):
                                try:
                                    ip = getattr(ent, attr, None)
                                    if ip is not None:
                                        break
                                except Exception:
                                    pass
                            ix, iy = (float(ip[0]), float(ip[1])) if ip is not None else (0.0, 0.0)
                            # bbox
                            bb = ent.GetBoundingBox()
                            pmin, pmax = bb[0], bb[1]
                            minx, miny = float(pmin[0]), float(pmin[1])
                            maxx, maxy = float(pmax[0]), float(pmax[1])
                            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
                            # center offset relative to insertion
                            dcx, dcy = cx - ix, cy - iy

                            def _aabb(minx0: float, miny0: float, maxx0: float, maxy0: float) -> Dict[str, Dict[str, float]]:
                                return {"min": {"x": minx0, "y": miny0}, "max": {"x": maxx0, "y": maxy0}}

                            def _shift_bbox(bb0: Dict[str, Dict[str, float]], dx0: float, dy0: float) -> Dict[str, Dict[str, float]]:
                                mn = bb0.get("min") or {}
                                mx = bb0.get("max") or {}
                                return _aabb(
                                    float(mn.get("x", 0.0)) + dx0,
                                    float(mn.get("y", 0.0)) + dy0,
                                    float(mx.get("x", 0.0)) + dx0,
                                    float(mx.get("y", 0.0)) + dy0,
                                )

                            def _aabb_overlaps_local(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], pad0: float = 0.0) -> bool:
                                amin, amax = a.get("min") or {}, a.get("max") or {}
                                bmin, bmax = b.get("min") or {}, b.get("max") or {}
                                return not (
                                    float(amax.get("x", 0.0)) + pad0 < float(bmin.get("x", 0.0)) - pad0
                                    or float(amin.get("x", 0.0)) - pad0 > float(bmax.get("x", 0.0)) + pad0
                                    or float(amax.get("y", 0.0)) + pad0 < float(bmin.get("y", 0.0)) - pad0
                                    or float(amin.get("y", 0.0)) - pad0 > float(bmax.get("y", 0.0)) + pad0
                                )

                            def _point_in_poly(px0: float, py0: float) -> bool:
                                inside0 = False
                                n0 = len(local_pts)
                                j0 = n0 - 1
                                for i0 in range(n0):
                                    xi0, yi0 = float(local_pts[i0]["x"]), float(local_pts[i0]["y"])
                                    xj0, yj0 = float(local_pts[j0]["x"]), float(local_pts[j0]["y"])
                                    hit0 = ((yi0 > py0) != (yj0 > py0)) and (
                                        px0 < (xj0 - xi0) * (py0 - yi0) / ((yj0 - yi0) or 1e-12) + xi0
                                    )
                                    if hit0:
                                        inside0 = not inside0
                                    j0 = i0
                                return inside0

                            def _seg_int(ax: float, ay: float, bx: float, by: float, cx0: float, cy0: float, dx0: float, dy0: float) -> bool:
                                def _orient(px1: float, py1: float, px2: float, py2: float, px3: float, py3: float) -> float:
                                    return (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)
                                o1 = _orient(ax, ay, bx, by, cx0, cy0)
                                o2 = _orient(ax, ay, bx, by, dx0, dy0)
                                o3 = _orient(cx0, cy0, dx0, dy0, ax, ay)
                                o4 = _orient(cx0, cy0, dx0, dy0, bx, by)
                                return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

                            def _crosses_traverse(bb0: Dict[str, Dict[str, float]]) -> bool:
                                mn = bb0.get("min") or {}
                                mx = bb0.get("max") or {}
                                minx0, miny0 = float(mn.get("x", 0.0)), float(mn.get("y", 0.0))
                                maxx0, maxy0 = float(mx.get("x", 0.0)), float(mx.get("y", 0.0))
                                corners0 = [(minx0, miny0), (minx0, maxy0), (maxx0, miny0), (maxx0, maxy0)]
                                if any(_point_in_poly(px0, py0) for (px0, py0) in corners0):
                                    return True
                                redges = [
                                    (minx0, miny0, minx0, maxy0),
                                    (minx0, maxy0, maxx0, maxy0),
                                    (maxx0, maxy0, maxx0, miny0),
                                    (maxx0, miny0, minx0, miny0),
                                ]
                                for kk in range(len(local_pts)):
                                    p_a = local_pts[kk]
                                    p_b = local_pts[(kk + 1) % len(local_pts)]
                                    ax0, ay0 = float(p_a["x"]), float(p_a["y"])
                                    bx0, by0 = float(p_b["x"]), float(p_b["y"])
                                    for (rx1, ry1, rx2, ry2) in redges:
                                        if _seg_int(ax0, ay0, bx0, by0, rx1, ry1, rx2, ry2):
                                            return True
                                return False

                            cur_bb = _aabb(minx, miny, maxx, maxy)
                            # extent along preferred direction for the current table box
                            corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
                            ext = 0.0
                            for (qx, qy) in corners:
                                proj = abs((qx - cx) * ux + (qy - cy) * uy)
                                if proj > ext:
                                    ext = proj

                            # desired center should clear the pillar box edge, not the pillar point.
                            pillar_bb = _nearest_pillar_bb(vx, vy)
                            p_cx, p_cy = vx, vy
                            p_ext = 0.0
                            own_pillar_bb = pillar_bb
                            if pillar_bb is not None:
                                pb_min = pillar_bb.get("min") or {}
                                pb_max = pillar_bb.get("max") or {}
                                pminx, pminy = float(pb_min.get("x", vx)), float(pb_min.get("y", vy))
                                pmaxx, pmaxy = float(pb_max.get("x", vx)), float(pb_max.get("y", vy))
                                p_cx = (pminx + pmaxx) / 2.0
                                p_cy = (pminy + pmaxy) / 2.0
                                for (qx, qy) in [(pminx, pminy), (pminx, pmaxy), (pmaxx, pminy), (pmaxx, pmaxy)]:
                                    proj = abs((qx - p_cx) * ux + (qy - p_cy) * uy)
                                    if proj > p_ext:
                                        p_ext = proj

                            # Candidate search: stay near the peg, prefer outside the parcel, and never cross the peg box.
                            placed_bbs: List[Dict[str, Any]] = []
                            for uu in used_handles:
                                if str(uu).upper() == str(h).upper():
                                    continue
                                for jj in range(ms.Count):
                                    ee = ms.Item(jj)
                                    if str(getattr(ee, "Handle", "")).upper() != str(uu).upper():
                                        continue
                                    try:
                                        bb_u = ee.GetBoundingBox()
                                        pmn, pmx = bb_u[0], bb_u[1]
                                        placed_bbs.append(_aabb(float(pmn[0]), float(pmn[1]), float(pmx[0]), float(pmx[1])))
                                    except Exception:
                                        pass
                                    break

                            base_r = p_ext + gap + ext
                            perp_x, perp_y = -uy, ux
                            best_pos = None
                            best_cost = 1e18
                            for r_mult in (1.0, 1.15, 1.35, 1.6):
                                rr = base_r * r_mult
                                for ang_deg in (0, 20, -20, 40, -40, 60, -60, 85, -85, 110, -110, 145, -145, 180):
                                    th = math.radians(float(ang_deg))
                                    dux = ux * math.cos(th) + perp_x * math.sin(th)
                                    duy = uy * math.cos(th) + perp_y * math.sin(th)
                                    tcx = p_cx + dux * rr
                                    tcy = p_cy + duy * rr
                                    cand_bb = _shift_bbox(cur_bb, tcx - cx, tcy - cy)
                                    cost = 0.0
                                    if own_pillar_bb is not None and _aabb_overlaps_local(cand_bb, own_pillar_bb, pad0=0.05):
                                        cost += 10000.0
                                    for other_pb in pillar_box_bbs:
                                        if own_pillar_bb is not None and other_pb is own_pillar_bb:
                                            continue
                                        if _aabb_overlaps_local(cand_bb, other_pb, pad0=0.05):
                                            cost += 5000.0
                                    if _crosses_traverse(cand_bb):
                                        cost += 10000.0
                                    for pb in placed_bbs:
                                        if _aabb_overlaps_local(cand_bb, pb, pad0=0.1):
                                            cost += 2500.0
                                    # prefer outward / shortest move
                                    cost += abs(float(ang_deg)) * 2.0
                                    cost += (r_mult - 1.0) * 200.0
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_pos = (tcx, tcy)
                                    if cost <= 1e-6:
                                        break
                                if best_cost <= 1e-6:
                                    break

                            if best_pos is None:
                                best_pos = (p_cx + ux * base_r, p_cy + uy * base_r)

                            tcx, tcy = best_pos
                            tx = tcx - dcx
                            ty = tcy - dcy
                            self.autocad.move_entity_to_xy(h, tx, ty)
                        else:
                            # fallback: simple outward offset
                            self.autocad.move_entity_to_xy(h, vx + ux * 4.0, vy + uy * 4.0)
                    except Exception:
                        # fallback: simple outward offset
                        self.autocad.move_entity_to_xy(h, vx + ux * 4.0, vy + uy * 4.0)

                for t in pn_tables:
                    h = str(t.get("handle") or "")
                    if h and h not in used_handles:
                        self.autocad.delete_entity_by_handle(h)

            # Bearings/distances (DD° MM' only) aligned to each edge, template-like MTEXT wrapper and height 1.2
            # Re-ensure active document before drawing text (avoids one bearing/distance drawn wrong if COM glitched)
            try:
                self.autocad._ensure_active_document()
                time.sleep(0.1)
            except Exception:
                pass
            # Use scaled bearing/road height (no max cap so e.g. 1:500→1:10000 gives height 24)
            _bd_height = max(0.5, float(bearing_road_height))
            def _bearing_ddmm(az_deg: float, hard_spaces: int = 1) -> str:
                az_deg = az_deg % 360.0
                d = int(az_deg)
                m = int(round((az_deg - d) * 60.0))
                if m == 60:
                    d = (d + 1) % 360
                    m = 0
                # Use AutoCAD MTEXT "hard space" (\\~) so the bearing never wraps mid-token.
                # Example: 143°~05'
                hs = max(1, int(hard_spaces or 1))
                return f"{d:03d}°" + ("\\~" * hs) + f"{m:02d}'"
            # Fetch interior border bbox once for clamping projecting-arrow text inside the drawing area.
            _interior_bb = None
            try:
                _ibb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                if not (_ibb or {}).get("success"):
                    _ibb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                if (_ibb or {}).get("success"):
                    _interior_bb = _ibb
            except Exception:
                pass

            # Minimum leg length (in drawing units) below which a projecting arrow is used.
            # Threshold: the bearing string "DDD° MM'" at 1 hard-space is roughly 8 chars wide;
            # stacked with distance, estimate ~9 char-widths at 0.6*height per char.
            _min_text_span = 9.0 * 0.6 * _bd_height

            # Smarter placement: avoid mixing with other plan contents by checking AABB collisions
            # against existing annotation (pillar-number tables, coordinate tables) and already-placed labels.
            _occupied_bboxes: List[Dict[str, Any]] = []
            try:
                bb = self.autocad.list_entity_bboxes(
                    layers=[
                        "CADA_PILLARNUMBERS",
                        "CADA_COORDINATES",
                        "CADA_NORTHCOORDINATES",
                        "CADA_EASTCOORDINATES",
                        "CADA_ROAD",
                        "CADA_TITLEBLOCK",
                    ],
                    object_names=["AcDbTable", "AcDbText", "AcDbMText"],
                )
                if (bb or {}).get("success"):
                    _occupied_bboxes = list(bb.get("bboxes") or [])
            except Exception:
                _occupied_bboxes = []
            _placed_label_bboxes: List[Dict[str, Dict[str, float]]] = []

            def _aabb_overlaps(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], pad: float = 0.0) -> bool:
                amin, amax = a.get("min") or {}, a.get("max") or {}
                bmin, bmax = b.get("min") or {}, b.get("max") or {}
                return not (
                    float(amax.get("x", 0.0)) + pad < float(bmin.get("x", 0.0)) - pad
                    or float(amin.get("x", 0.0)) - pad > float(bmax.get("x", 0.0)) + pad
                    or float(amax.get("y", 0.0)) + pad < float(bmin.get("y", 0.0)) - pad
                    or float(amin.get("y", 0.0)) - pad > float(bmax.get("y", 0.0)) + pad
                )

            def _aabb_for_centered_rect(cx: float, cy: float, w: float, h: float, rot_rad: float) -> Dict[str, Dict[str, float]]:
                hw, hh = 0.5 * float(w), 0.5 * float(h)
                c, s = math.cos(float(rot_rad)), math.sin(float(rot_rad))
                # Four corners around origin
                pts = [(-hw, -hh), (-hw, hh), (hw, -hh), (hw, hh)]
                xs = []
                ys = []
                for (px, py) in pts:
                    rx = cx + px * c - py * s
                    ry = cy + px * s + py * c
                    xs.append(rx)
                    ys.append(ry)
                return {"min": {"x": min(xs), "y": min(ys)}, "max": {"x": max(xs), "y": max(ys)}}

            def _aabb_inside_interior(a: Dict[str, Dict[str, float]], margin: float) -> bool:
                if _interior_bb is None:
                    return True
                ib_min = _interior_bb.get("min") or {}
                ib_max = _interior_bb.get("max") or {}
                xmin = float(ib_min.get("x", -1e18)) + margin
                ymin = float(ib_min.get("y", -1e18)) + margin
                xmax = float(ib_max.get("x", 1e18)) - margin
                ymax = float(ib_max.get("y", 1e18)) - margin
                amin, amax = a.get("min") or {}, a.get("max") or {}
                return (
                    float(amin.get("x", 0.0)) >= xmin
                    and float(amax.get("x", 0.0)) <= xmax
                    and float(amin.get("y", 0.0)) >= ymin
                    and float(amax.get("y", 0.0)) <= ymax
                )

            def _point_in_polygon(px: float, py: float, poly: List[Dict[str, float]]) -> bool:
                inside = False
                n = len(poly)
                if n < 3:
                    return False
                j = n - 1
                for i in range(n):
                    xi, yi = float(poly[i]["x"]), float(poly[i]["y"])
                    xj, yj = float(poly[j]["x"]), float(poly[j]["y"])
                    try:
                        hit = ((yi > py) != (yj > py)) and (
                            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi
                        )
                    except Exception:
                        hit = False
                    if hit:
                        inside = not inside
                    j = i
                return inside

            def _segments_intersect(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> bool:
                def _orient(px1: float, py1: float, px2: float, py2: float, px3: float, py3: float) -> float:
                    return (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1)

                def _on_segment(px1: float, py1: float, px2: float, py2: float, qx: float, qy: float) -> bool:
                    return (
                        min(px1, px2) - 1e-9 <= qx <= max(px1, px2) + 1e-9
                        and min(py1, py2) - 1e-9 <= qy <= max(py1, py2) + 1e-9
                    )

                o1 = _orient(ax, ay, bx, by, cx, cy)
                o2 = _orient(ax, ay, bx, by, dx, dy)
                o3 = _orient(cx, cy, dx, dy, ax, ay)
                o4 = _orient(cx, cy, dx, dy, bx, by)

                if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
                    return True
                if abs(o1) <= 1e-9 and _on_segment(ax, ay, bx, by, cx, cy):
                    return True
                if abs(o2) <= 1e-9 and _on_segment(ax, ay, bx, by, dx, dy):
                    return True
                if abs(o3) <= 1e-9 and _on_segment(cx, cy, dx, dy, ax, ay):
                    return True
                if abs(o4) <= 1e-9 and _on_segment(cx, cy, dx, dy, bx, by):
                    return True
                return False

            def _aabb_crosses_traverse(a: Dict[str, Dict[str, float]], poly: List[Dict[str, float]]) -> bool:
                amin, amax = a.get("min") or {}, a.get("max") or {}
                minx, miny = float(amin.get("x", 0.0)), float(amin.get("y", 0.0))
                maxx, maxy = float(amax.get("x", 0.0)), float(amax.get("y", 0.0))
                corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
                if any(_point_in_polygon(cx, cy, poly) for (cx, cy) in corners):
                    return True
                rect_edges = [
                    (minx, miny, minx, maxy),
                    (minx, maxy, maxx, maxy),
                    (maxx, maxy, maxx, miny),
                    (maxx, miny, minx, miny),
                ]
                n = len(poly)
                for i in range(n):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % n]
                    x1, y1 = float(p1["x"]), float(p1["y"])
                    x2, y2 = float(p2["x"]), float(p2["y"])
                    for (rx1, ry1, rx2, ry2) in rect_edges:
                        if _segments_intersect(x1, y1, x2, y2, rx1, ry1, rx2, ry2):
                            return True
                return False

            for i in range(len(local_pts)):
                p1 = local_pts[i]
                p2 = local_pts[(i + 1) % len(local_pts)]
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                L_geom = math.hypot(dx, dy)
                if L_geom <= 1e-6:
                    continue
                # For start-coordinate + bearing/distance traverses, preserve the adjusted leg
                # bearings/distances for annotation even after the duplicate closure vertex is removed.
                if bearing_distance_legs and len(bearing_distance_legs) == len(local_pts):
                    leg_meta = bearing_distance_legs[i]
                    az = float(leg_meta.get("bearing_deg", (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)) % 360.0
                    L_disp = float(leg_meta.get("distance", L_geom))
                else:
                    az = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                    L_disp = L_geom
                az_orient = az if az <= 180.0 else (az - 180.0)
                rot_deg = (90.0 - az_orient) % 360.0
                rot = math.radians(rot_deg)
                midx = (p1["x"] + p2["x"]) / 2.0
                midy = (p1["y"] + p2["y"]) / 2.0

                poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                vx = poly_cx - midx
                vy = poly_cy - midy
                n1x, n1y = dy / L_geom, -dx / L_geom
                n2x, n2y = -dy / L_geom, dx / L_geom
                if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                    inx, iny = n1x, n1y
                else:
                    inx, iny = n2x, n2y
                outx, outy = -inx, -iny

                use_projecting_arrow = L_geom < _min_text_span

                if not use_projecting_arrow:
                    # --- Normal on-leg placement (unchanged logic) ---
                    target_span = 0.6 * L_geom
                    base_span_est = 4.6 * _bd_height
                    space_span_est = 0.6 * _bd_height
                    hs = 1
                    if target_span > base_span_est and space_span_est > 1e-9:
                        hs = 1 + int(math.ceil((target_span - base_span_est) / space_span_est))
                        hs = max(1, min(hs, 60))
                    bearing_str = _bearing_ddmm(az, hard_spaces=hs)
                    dist_str = f"{L_disp:.2f}m"
                    first_line_out = -math.sin(rot) * outx + math.cos(rot) * outy
                    if first_line_out > 0:
                        stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{bearing_str}\\P{dist_str}}}"
                    else:
                        stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{dist_str}\\P{bearing_str}}}"
                    w = min(max(2.0, 0.75 * L_geom), 0.95 * L_geom)
                    self.autocad.add_mtext(
                        stacked_text,
                        midx,
                        midy,
                        layer="CADA_BEARING_DIST",
                        rotation_rad=rot,
                        height=_bd_height,
                        width=w,
                        attachment_point=5,
                    )
                    try:
                        _placed_label_bboxes.append(_aabb_for_centered_rect(midx, midy, w, 2.2 * _bd_height, rot))
                    except Exception:
                        pass
                else:
                    # --- Smart leader for short legs ---
                    # Prefer an L-shaped leader with a horizontal (90/270) branch; if that collides or
                    # approaches the border, progressively extend and try alternative orientations.

                    bearing_str = _bearing_ddmm(az, hard_spaces=1)
                    dist_str = f"{L_disp:.2f}m"
                    stacked_text = f"{{\\fVerdana|b0|i0|c0|p34;{bearing_str}\\P{dist_str}}}"

                    def _clamp_xy(x: float, y: float, margin: float) -> Tuple[float, float]:
                        if _interior_bb is None:
                            return x, y
                        ib_min = _interior_bb.get("min") or {}
                        ib_max = _interior_bb.get("max") or {}
                        xmin = float(ib_min.get("x", -1e18)) + margin
                        ymin = float(ib_min.get("y", -1e18)) + margin
                        xmax = float(ib_max.get("x", 1e18)) - margin
                        ymax = float(ib_max.get("y", 1e18)) - margin
                        return max(xmin, min(xmax, x)), max(ymin, min(ymax, y))

                    stem_base = max(_min_text_span * 0.9, 6.0 * _bd_height)
                    branch_base = max(_min_text_span * 1.35, 10.0 * _bd_height)
                    pad = 1.25 * _bd_height
                    margin = 2.0 * _bd_height

                    # Unit directions
                    t_ux, t_uy = dx / L, dy / L  # along leg in XY
                    stem_dirs = [
                        (outx, outy),            # outward normal
                        (-outx, -outy),          # inward normal
                        (t_ux, t_uy),            # along the leg
                        (-t_ux, -t_uy),          # reverse along the leg
                    ]
                    # Branch options in preference order: horizontal (E/W), vertical (N/S), then along-leg.
                    branch_opts = [
                        (1.0, 0.0, 0.0),     # East, rotation 0
                        (-1.0, 0.0, 0.0),    # West, rotation 0
                        (0.0, 1.0, math.pi / 2.0),   # North, rotation 90°
                        (0.0, -1.0, math.pi / 2.0),  # South, rotation 90°
                        (t_ux, t_uy, rot),   # along leg
                        (-t_ux, -t_uy, rot), # reverse along leg (rotation kept readable via rot)
                    ]

                    chosen = None
                    for (sdx, sdy) in stem_dirs:
                        for sm in (1.0, 1.6, 2.4, 3.4):
                            stem_len = stem_base * sm
                            ex = midx + sdx * stem_len
                            ey = midy + sdy * stem_len
                            ex, ey = _clamp_xy(ex, ey, margin)
                            for (bdx, bdy, text_rot) in branch_opts:
                                for bm in (1.0, 1.35, 1.8):
                                    bl = branch_base * bm
                                    bx = ex + bdx * bl
                                    by = ey + bdy * bl
                                    bx2, by2 = _clamp_xy(bx, by, margin)
                                    actual_bl = math.hypot(bx2 - ex, by2 - ey)
                                    if actual_bl < 0.55 * bl:
                                        continue

                                    tcx = (ex + bx2) / 2.0
                                    tcy = (ey + by2) / 2.0
                                    tw = max(2.0, actual_bl * 0.95)
                                    th = 2.2 * _bd_height
                                    aabb = _aabb_for_centered_rect(tcx, tcy, tw, th, text_rot)
                                    if not _aabb_inside_interior(aabb, margin):
                                        continue
                                    bad = False
                                    for ob in _occupied_bboxes:
                                        if _aabb_overlaps(aabb, ob, pad=pad):
                                            bad = True
                                            break
                                    if not bad:
                                        for pb in _placed_label_bboxes:
                                            if _aabb_overlaps(aabb, pb, pad=pad):
                                                bad = True
                                                break
                                    if bad:
                                        continue

                                    chosen = {
                                        "elbow": (ex, ey),
                                        "end": (bx2, by2),
                                        "text": (tcx, tcy),
                                        "text_rot": float(text_rot),
                                        "text_w": float(tw),
                                        "aabb": aabb,
                                    }
                                    break
                                if chosen:
                                    break
                            if chosen:
                                break
                        if chosen:
                            break
                    if not chosen:
                        # Fallback that should always succeed: long outward stem + horizontal branch with clamping.
                        ex = midx + outx * (stem_base * 3.4)
                        ey = midy + outy * (stem_base * 3.4)
                        ex, ey = _clamp_xy(ex, ey, margin)
                        bx = ex + (1.0 if outx >= 0 else -1.0) * (branch_base * 1.8)
                        by = ey
                        bx, by = _clamp_xy(bx, by, margin)
                        tcx = (ex + bx) / 2.0
                        tcy = (ey + by) / 2.0
                        tw = max(2.0, math.hypot(bx - ex, by - ey) * 0.95)
                        th = 2.2 * _bd_height
                        aabb = _aabb_for_centered_rect(tcx, tcy, tw, th, 0.0)
                        chosen = {"elbow": (ex, ey), "end": (bx, by), "text": (tcx, tcy), "text_rot": 0.0, "text_w": tw, "aabb": aabb}

                    elbow_x, elbow_y = chosen["elbow"]
                    branch_end_x, branch_end_y = chosen["end"]
                    text_cx, text_cy = chosen["text"]
                    text_rot = chosen["text_rot"]
                    text_w = chosen["text_w"]

                    # Draw leader polyline: midpoint → elbow → end.
                    try:
                        self.autocad.create_lwpolyline(
                            [
                                {"x": midx, "y": midy},
                                {"x": elbow_x, "y": elbow_y},
                                {"x": branch_end_x, "y": branch_end_y},
                            ],
                            layer="CADA_BEARING_DIST",
                            closed=False,
                        )
                    except Exception:
                        pass

                    # Arrowhead at the boundary midpoint, pointing along the first leader segment.
                    try:
                        arrow_size = _bd_height * 0.6
                        stem_dx = elbow_x - midx
                        stem_dy = elbow_y - midy
                        stem_L = math.hypot(stem_dx, stem_dy) or 1.0
                        su, sv = stem_dx / stem_L, stem_dy / stem_L
                        sp, sq = -sv, su
                        arrow_pts = [
                            {"x": midx, "y": midy},
                            {"x": midx + su * arrow_size + sp * arrow_size * 0.35, "y": midy + sv * arrow_size + sq * arrow_size * 0.35},
                            {"x": midx + su * arrow_size - sp * arrow_size * 0.35, "y": midy + sv * arrow_size - sq * arrow_size * 0.35},
                        ]
                        self.autocad.create_lwpolyline(arrow_pts, layer="CADA_BEARING_DIST", closed=True)
                    except Exception:
                        pass

                    # Place MTEXT on the second segment, oriented to that segment's bearing.
                    try:
                        self.autocad.add_mtext(
                            stacked_text,
                            text_cx,
                            text_cy,
                            layer="CADA_BEARING_DIST",
                            rotation_rad=float(text_rot),
                            height=_bd_height,
                            width=float(text_w),
                            attachment_point=5,
                        )
                    except Exception:
                        # Last-resort: try horizontal at elbow
                        try:
                            self.autocad.add_mtext(
                                stacked_text,
                                elbow_x,
                                elbow_y,
                                layer="CADA_BEARING_DIST",
                                rotation_rad=0.0,
                                height=_bd_height,
                                width=max(2.0, branch_base),
                                attachment_point=5,
                            )
                        except Exception:
                            pass
                    try:
                        _placed_label_bboxes.append(chosen["aabb"])
                    except Exception:
                        pass

                geometry["bearing_mtext"] += 1
                time.sleep(0.05)

            # Second pass (cartographic cleanup): nudge pillar-number labels away from overlaps.
            # This addresses very short legs where pegs and annotations cluster tightly.
            try:
                if pn_meta:
                    t_res2 = self.autocad.list_tables(layer="CADA_PILLARNUMBERS")
                    tbs = (t_res2.get("tables") or []) if isinstance(t_res2, dict) else []
                    ins_map = {}
                    for t in tbs:
                        hh = str(t.get("handle") or "").upper()
                        ip = t.get("insertion_point") or {}
                        if hh and ("x" in ip) and ("y" in ip):
                            ins_map[hh] = (float(ip.get("x", 0.0)), float(ip.get("y", 0.0)))

                    bb2 = self.autocad.list_entity_bboxes(
                        layers=["CADA_BEARING_DIST", "CADA_PILLARNUMBERS", "CADA_PILLARS"],
                        object_names=["AcDbTable", "AcDbText", "AcDbMText", "AcDbBlockReference"],
                    )
                    bbs = list((bb2 or {}).get("bboxes") or []) if (bb2 or {}).get("success") else []
                    bb_by_handle = {str((b.get("handle") or "")).upper(): b for b in bbs if b.get("handle")}

                    bearing_bbs = [b for b in bbs if str(b.get("layer") or "").upper() == "CADA_BEARING_DIST"]
                    pillar_bbs = {str((b.get("handle") or "")).upper(): b for b in bbs if str(b.get("layer") or "").upper() == "CADA_PILLARNUMBERS" and b.get("handle")}
                    pillar_obj_bbs = [b for b in bbs if str(b.get("layer") or "").upper() == "CADA_PILLARS"]

                    def _shift_aabb(a: Dict[str, Dict[str, float]], dx0: float, dy0: float) -> Dict[str, Dict[str, float]]:
                        amin, amax = a.get("min") or {}, a.get("max") or {}
                        return {
                            "min": {"x": float(amin.get("x", 0.0)) + dx0, "y": float(amin.get("y", 0.0)) + dy0},
                            "max": {"x": float(amax.get("x", 0.0)) + dx0, "y": float(amax.get("y", 0.0)) + dy0},
                        }

                    # Keep post-placement nudges local so pillar-number labels remain close to their pegs.
                    step = max(0.35, min(0.7, 0.18 * _bd_height))
                    pad = 1.15 * _bd_height
                    margin = 2.0 * _bd_height

                    # Iterate a few passes so clustered labels can settle without leaving residual overlaps.
                    for _pass in range(3):
                        moved = 0
                        for meta in pn_meta:
                            h_raw = str(meta.get("handle") or "")
                            hh = h_raw.upper()
                            if not hh or hh not in ins_map or hh not in pillar_bbs:
                                continue
                            cur_ip = ins_map[hh]
                            cur_bb = pillar_bbs[hh]

                            def _overlap_count(test_bb: Dict[str, Dict[str, float]]) -> int:
                                c = 0
                                # against bearing/distance texts
                                for ob in bearing_bbs:
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against already-placed bearing labels (local estimates)
                                for ob in (_placed_label_bboxes or []):
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against other sheet annotations we sampled earlier (exclude self layer to avoid double-counting)
                                for ob in (_occupied_bboxes or []):
                                    lyr = str(ob.get("layer") or "").upper()
                                    if lyr == "CADA_PILLARNUMBERS":
                                        continue
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # against other pillar numbers
                                for oh, ob in pillar_bbs.items():
                                    if oh == hh:
                                        continue
                                    if _aabb_overlaps(test_bb, ob, pad=pad):
                                        c += 1
                                # hard drafting penalties: don't cross peg symbols or intrude into traverse unless unavoidable
                                for ob in pillar_obj_bbs:
                                    if _aabb_overlaps(test_bb, ob, pad=0.05):
                                        c += 50
                                if _aabb_crosses_traverse(test_bb, local_pts):
                                    c += 50
                                return c

                            base_overlaps = _overlap_count(cur_bb)
                            if base_overlaps <= 0:
                                continue

                            ux0 = float(meta.get("ux", 0.0))
                            uy0 = float(meta.get("uy", 0.0))
                            L0 = math.hypot(ux0, uy0) or 1.0
                            ux0, uy0 = ux0 / L0, uy0 / L0
                            vx0, vy0 = -uy0, ux0

                            # Hard cap the search radius to preserve survey drafting style:
                            # labels should stay close to their corresponding pillar.
                            off0 = float(meta.get("off", 4.0) or 4.0)
                            max_r = max(1.0, min(2.2 * off0, 4.5))
                            angles_deg = [0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100, 120, -120, 150, -150, 180]
                            best = {"score": base_overlaps, "cost": float(base_overlaps) * 1000.0, "r": 0.0, "dx": 0.0, "dy": 0.0, "bb": cur_bb}

                            r = step
                            while r <= max_r + 1e-9:
                                for ad in angles_deg:
                                    th = math.radians(float(ad))
                                    dx0 = (ux0 * math.cos(th) + vx0 * math.sin(th)) * r
                                    dy0 = (uy0 * math.cos(th) + vy0 * math.sin(th)) * r
                                    test_bb = _shift_aabb(cur_bb, dx0, dy0)
                                    if not _aabb_inside_interior(test_bb, margin):
                                        continue
                                    score = _overlap_count(test_bb)
                                    # Prefer fewer overlaps, then shorter moves (cost function).
                                    cost = float(score) * 1000.0 + float(r)
                                    if (score < best["score"]) or (score == best["score"] and cost < best["cost"]):
                                        best = {"score": score, "cost": cost, "r": r, "dx": dx0, "dy": dy0, "bb": test_bb}
                                        if score == 0:
                                            break
                                if best["score"] == 0:
                                    break
                                r += step

                            # Only move if we strictly improve overlap score (prevents oscillation).
                            if best["r"] > 0.0 and best["score"] < base_overlaps and (best["dx"] != 0.0 or best["dy"] != 0.0):
                                try:
                                    self.autocad.move_entity_to_xy(h_raw, cur_ip[0] + float(best["dx"]), cur_ip[1] + float(best["dy"]))
                                    ins_map[hh] = (cur_ip[0] + float(best["dx"]), cur_ip[1] + float(best["dy"]))
                                    pillar_bbs[hh] = best["bb"]
                                    moved += 1
                                except Exception:
                                    pass
                        if moved == 0:
                            break

                    # Final proximity normalization: keep each pillar-number table close to its own pillar.
                    # This prevents any residual "too far" appearance after overlap reduction.
                    bb3 = self.autocad.list_entity_bboxes(
                        layers=["CADA_PILLARNUMBERS"],
                        object_names=["AcDbTable"],
                    )
                    bbs3 = list((bb3 or {}).get("bboxes") or []) if (bb3 or {}).get("success") else []
                    bb3_by_h = {str((b.get("handle") or "")).upper(): b for b in bbs3 if b.get("handle")}

                    def _dist_point_to_bbox(px: float, py: float, b: Dict[str, Any]) -> float:
                        mn = b.get("min") or {}
                        mx = b.get("max") or {}
                        minx, miny = float(mn.get("x", 0.0)), float(mn.get("y", 0.0))
                        maxx, maxy = float(mx.get("x", 0.0)), float(mx.get("y", 0.0))
                        dx0 = 0.0 if (minx <= px <= maxx) else (minx - px if px < minx else px - maxx)
                        dy0 = 0.0 if (miny <= py <= maxy) else (miny - py if py < miny else py - maxy)
                        return math.hypot(dx0, dy0)

                    target_min_gap = 0.8
                    target_max_gap = 2.0
                    for meta in pn_meta:
                        h_raw = str(meta.get("handle") or "")
                        hh = h_raw.upper()
                        if not hh or hh not in ins_map or hh not in bb3_by_h:
                            continue
                        vx = float(meta.get("vx", 0.0))
                        vy = float(meta.get("vy", 0.0))
                        cur_ip = ins_map[hh]
                        cur_bb = bb3_by_h[hh]
                        g = _dist_point_to_bbox(vx, vy, cur_bb)
                        # Move only when clearly outside desired proximity band.
                        if (g >= target_min_gap - 0.05) and (g <= target_max_gap + 0.05):
                            continue

                        vec_x = float(cur_ip[0]) - vx
                        vec_y = float(cur_ip[1]) - vy
                        vec_L = math.hypot(vec_x, vec_y)
                        if vec_L <= 1e-9:
                            vec_x = float(meta.get("ux", 1.0))
                            vec_y = float(meta.get("uy", 0.0))
                            vec_L = math.hypot(vec_x, vec_y) or 1.0
                        ux1, uy1 = vec_x / vec_L, vec_y / vec_L

                        if g < target_min_gap:
                            delta = target_min_gap - g
                        else:
                            delta = -(g - target_max_gap)

                        nx = float(cur_ip[0]) + ux1 * delta
                        ny = float(cur_ip[1]) + uy1 * delta
                        # Guardrail: do not worsen annotation collisions while enforcing proximity.
                        cur_score = 0
                        for ob in bearing_bbs:
                            if _aabb_overlaps(cur_bb, ob, pad=pad):
                                cur_score += 1
                        for oh, ob in bb3_by_h.items():
                            if oh == hh:
                                continue
                            if _aabb_overlaps(cur_bb, ob, pad=pad):
                                cur_score += 1
                        for ob in pillar_obj_bbs:
                            if _aabb_overlaps(cur_bb, ob, pad=0.05):
                                cur_score += 50
                        if _aabb_crosses_traverse(cur_bb, local_pts):
                            cur_score += 50
                        test_bb = _shift_aabb(cur_bb, nx - float(cur_ip[0]), ny - float(cur_ip[1]))
                        if not _aabb_inside_interior(test_bb, margin):
                            continue
                        new_score = 0
                        for ob in bearing_bbs:
                            if _aabb_overlaps(test_bb, ob, pad=pad):
                                new_score += 1
                        for oh, ob in bb3_by_h.items():
                            if oh == hh:
                                continue
                            if _aabb_overlaps(test_bb, ob, pad=pad):
                                new_score += 1
                        for ob in pillar_obj_bbs:
                            if _aabb_overlaps(test_bb, ob, pad=0.05):
                                new_score += 50
                        if _aabb_crosses_traverse(test_bb, local_pts):
                            new_score += 50
                        if new_score > cur_score:
                            continue
                        try:
                            self.autocad.move_entity_to_xy(h_raw, nx, ny)
                            ins_map[hh] = (nx, ny)
                            bb3_by_h[hh] = test_bb
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                self.autocad.execute_command("REGEN")
            except Exception:
                pass
            time.sleep(0.2)

            # Draw Concrete Wall Fence(s) if requested (C.W.F / D.C.W.F) on CADA_CWF
            # - Single line parallel to traverse leg(s)
            # - Sits outside the traverse (outward normal)
            # - Offset scales with plan scale: 0.3 @ 1:500, 0.15 @ 1:250, 0.6 @ 1:1000, etc.
            used_fence_edges: set[int] = set()
            try:
                fence_offset = 0.3 * (float(chosen_denom) / 500.0)
            except Exception:
                fence_offset = 0.3

            for f in fences_to_draw:
                try:
                    kind = str((f or {}).get("kind") or "").upper().strip()
                    spec = str((f or {}).get("spec") or "")
                    if kind not in ("CWF", "DCWF") or not spec:
                        continue

                    spec_lower = spec.lower()
                    ref_match = re.search(
                        r"(?:linking|between|connecting|on|along|joining)\s+(?:the\s+)?(?:side\s+)?(?:of\s+)?(?:boundary\s+)?(?:line\s+)?(?:pillars\s+)?(.*)$",
                        spec_lower,
                    )
                    target_indices: List[int] = []
                    if ref_match and pn_list:
                        ref_str = ref_match.group(1).strip()
                        ref_str_norm = re.sub(r"\s+", " ", ref_str)
                        matched_ordered: List[Tuple[int, int]] = []
                        for idx, p_info in enumerate(pn_list):
                            num = str(p_info.get("number", "")).strip()
                            prefix = str(p_info.get("prefix", "")).strip()
                            if not num:
                                continue
                            full_label = (prefix + " " + num).lower()
                            num_lower = num.lower()
                            ref_lower = ref_str_norm.lower()
                            pos = ref_lower.find(full_label)
                            if pos >= 0:
                                matched_ordered.append((pos, idx))
                                continue
                            m_num = re.search(r"\b" + re.escape(num_lower) + r"\b", ref_lower)
                            if m_num:
                                matched_ordered.append((m_num.start(), idx))
                        if len(matched_ordered) >= 2:
                            matched_ordered.sort(key=lambda x: x[0])
                            ordered_indices: List[int] = []
                            seen_idx: set[int] = set()
                            for _, idx in matched_ordered:
                                if idx not in seen_idx:
                                    ordered_indices.append(idx)
                                    seen_idx.add(idx)
                            n_pts = len(local_pts)
                            # Chain support: "A to B to C to D" means fences on AB, BC, and CD.
                            for kk in range(len(ordered_indices) - 1):
                                a_idx = ordered_indices[kk]
                                b_idx = ordered_indices[kk + 1]
                                for i in range(n_pts):
                                    j = (i + 1) % n_pts
                                    if (i == a_idx and j == b_idx) or (i == b_idx and j == a_idx):
                                        target_indices.append(i)
                                        break

                    if not target_indices and len(local_pts) >= 2:
                        target_indices = [0]
                    if not target_indices:
                        continue
                    for target_idx in target_indices:
                        if target_idx in used_fence_edges:
                            continue
                        used_fence_edges.add(target_idx)

                        p1 = local_pts[target_idx]
                        p2 = local_pts[(target_idx + 1) % len(local_pts)]
                        dx = p2["x"] - p1["x"]
                        dy = p2["y"] - p1["y"]
                        L_bound = math.hypot(dx, dy)
                        if L_bound <= 1e-6:
                            continue

                        ux, uy = dx / L_bound, dy / L_bound

                        # Outward normal using centroid
                        midx = (p1["x"] + p2["x"]) / 2.0
                        midy = (p1["y"] + p2["y"]) / 2.0
                        poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                        poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                        vx = poly_cx - midx
                        vy = poly_cy - midy
                        n1x, n1y = uy, -ux
                        n2x, n2y = -uy, ux
                        if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                            outx, outy = -n1x, -n1y
                        else:
                            outx, outy = -n2x, -n2y

                        f_s = {"x": p1["x"] + fence_offset * outx, "y": p1["y"] + fence_offset * outy}
                        f_e = {"x": p2["x"] + fence_offset * outx, "y": p2["y"] + fence_offset * outy}
                        self.autocad.create_lwpolyline([f_s, f_e], layer="CADA_CWF", closed=False, linetype_scale=3.0)

                        label = "C.W.F" if kind == "CWF" else "D.C.W.F"
                        fence_text_h = max(0.25, 0.5 * float(bearing_road_height))
                        tx = (f_s["x"] + f_e["x"]) / 2.0 + (fence_text_h * 1.2) * outx
                        ty = (f_s["y"] + f_e["y"]) / 2.0 + (fence_text_h * 1.2) * outy
                        rot_rad = math.atan2(uy, ux)
                        deg = math.degrees(rot_rad) % 360
                        if 90 < deg <= 270:
                            rot_rad += math.pi
                        fence_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{label}}}"
                        txt_width = max(10.0, L_bound)
                        self.autocad.add_mtext(
                            fence_fmt,
                            tx,
                            ty,
                            layer="CADA_CWF",
                            rotation_rad=rot_rad,
                            height=float(fence_text_h),
                            width=txt_width,
                            attachment_point=5,
                        )
                except Exception:
                    continue

            # Draw Access Road(s) if requested (supports multiple roads, each beside a traverse leg)
            for road_idx, road_spec in enumerate(roads_to_draw):
                try:
                    ar_lower = road_spec.lower()
                    # 1. Parse Width (support "7m width", "7m road", "width 7m")
                    width = 6.0  # fallback default
                    m_w = (
                        re.search(r"(\d+(?:\.\d+)?)\s*m\s+road", ar_lower)
                        or re.search(r"(\d+(?:\.\d+)?)\s*m\s+width", ar_lower)
                        or re.search(r"width\s+(\d+(?:\.\d+)?)\s*m", ar_lower)
                        or re.search(r"width.*?(\d+(?:\.\d+)?)\s*m", ar_lower)
                    )
                    if m_w:
                        width = float(m_w.group(1))
                    elif "5m road" in ar_lower:
                        width = 5.0
                    
                    # 2. Parse Offset
                    offset = 0.2 # default scale/(5*scale) -> 0.2m
                    m_o = re.search(r"offset.*?(\d+(?:\.\d+)?)\s*m", ar_lower)
                    if m_o:
                        offset = float(m_o.group(1))

                    # 3. Identify Reference Edge
                    # "connecting X - Y", "joining pillars X and Y", "on the side joining pillars X and Y", "on the side of X and Y"
                    ref_match = re.search(
                        r"(?:linking|between|connecting|on|along|joining)\s+(?:the\s+)?(?:side\s+)?(?:of\s+)?(?:boundary\s+)?(?:line\s+)?(?:pillars\s+)?(.*)$",
                        ar_lower,
                    )
                    target_idx = -1
                    if ref_match and pn_list:
                        ref_str = ref_match.group(1).strip()
                        # Normalize separators: "SC/CK 4324 - 4325" or "4324 and 4325"
                        ref_str_norm = re.sub(r"\s+", " ", ref_str)
                        matched_indices = []
                        for idx, p_info in enumerate(pn_list):
                            num = str(p_info.get("number", "")).strip()
                            prefix = str(p_info.get("prefix", "")).strip()
                            if not num:
                                continue
                            # Match full label (e.g. "sc/ck 4324") or number as whole word to avoid "432" matching "4324"
                            full_label = (prefix + " " + num).lower()
                            num_lower = num.lower()
                            if full_label in ref_str_norm.lower():
                                matched_indices.append(idx)
                            elif re.search(r"\b" + re.escape(num_lower) + r"\b", ref_str_norm.lower()):
                                matched_indices.append(idx)
                        
                        if len(matched_indices) >= 2:
                            n_pts = len(local_pts)
                            for i in range(n_pts):
                                j = (i + 1) % n_pts
                                if i in matched_indices and j in matched_indices:
                                    target_idx = i
                                    break
                    if target_idx == -1:
                        # Fallback: first edge (index 0) if we have at least two points
                        if len(local_pts) >= 2:
                            target_idx = 0

                    if target_idx != -1:
                        p1 = local_pts[target_idx]
                        p2 = local_pts[(target_idx + 1) % len(local_pts)]
                        dx = p2["x"] - p1["x"]
                        dy = p2["y"] - p1["y"]
                        L_bound = math.hypot(dx, dy)
                        
                        if L_bound > 1e-6:
                            # Road length = 1.4 × traverse leg length (e.g. 100m leg → 140m road)
                            L_road = 1.4 * L_bound
                            # Minimum plotted length: (L_road / scale_denom) must be >= 0.064 so the road is visible.
                            # If (L_road / chosen_denom) < 0.064, extend road so (new_L_road / chosen_denom) = 0.064
                            # => new_L_road = 0.064 * chosen_denom (e.g. 1:500 → 32m).
                            try:
                                scale_denom = float(chosen_denom) if chosen_denom and float(chosen_denom) > 1e-6 else 500.0
                            except Exception:
                                scale_denom = 500.0
                            min_plotted_ratio = 0.064
                            ratio = L_road / scale_denom
                            if ratio < min_plotted_ratio:
                                L_road = min_plotted_ratio * scale_denom
                                extension_total = max(0.0, L_road - L_bound)
                            else:
                                extension_total = 0.4 * L_bound
                            ext_side = extension_total / 2.0
                            
                            # Unit vector along edge
                            ux, uy = dx / L_bound, dy / L_bound
                            
                            # Normal vector (outward)
                            # Reuse centroid logic
                            midx = (p1["x"] + p2["x"]) / 2.0
                            midy = (p1["y"] + p2["y"]) / 2.0
                            poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                            poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                            vx = poly_cx - midx
                            vy = poly_cy - midy
                            
                            n1x, n1y = uy, -ux
                            n2x, n2y = -uy, ux
                            
                            # Dot product with centroid vector: interior normal points TOWARD centroid
                            # We want OUTWARD normal
                            if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                                # n1 is interior
                                outx, outy = -n1x, -n1y
                            else:
                                # n2 is interior
                                outx, outy = -n2x, -n2y
                                
                            # Define road line points
                            # Line 1: offset
                            # Start: p1 - ext_side*u + offset*out
                            # End: p2 + ext_side*u + offset*out
                            # Actually centered: Midpoint + (L_bound/2 + ext_side)*u ...
                            # Let's use p1 and p2 as basis
                            # Road Start projected on line: p1 - ext_side * u
                            # Road End projected on line: p2 + ext_side * u
                            
                            rsx = p1["x"] - ext_side * ux
                            rsy = p1["y"] - ext_side * uy
                            rex = p2["x"] + ext_side * ux
                            rey = p2["y"] + ext_side * uy
                            
                            # Draw Line 1 (Offset)
                            l1_s = {"x": rsx + offset * outx, "y": rsy + offset * outy}
                            l1_e = {"x": rex + offset * outx, "y": rey + offset * outy}
                            self.autocad.create_lwpolyline([l1_s, l1_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                            
                            # Draw Line 2 (Offset + Width)
                            l2_s = {"x": rsx + (offset + width) * outx, "y": rsy + (offset + width) * outy}
                            l2_e = {"x": rex + (offset + width) * outx, "y": rey + (offset + width) * outy}
                            self.autocad.create_lwpolyline([l2_s, l2_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)

                            # Road title: first road uses access_road_title if set, else "ACCESS    ROAD"; others use "ACCESS    ROAD"
                            road_title = (access_road_title if road_idx == 0 else None) or "ACCESS    ROAD"
                            road_title = road_title.strip() or "ACCESS    ROAD"
                            if road_idx == 0:
                                geometry["access_road_title"] = road_title

                            # Position text perfectly centered (vertically and horizontally) within the road
                            # cx, cy = geometric center of road (midpoint between l1 and l2, offset by width/2 outward)
                            cx = (l1_s["x"] + l1_e["x"]) / 2.0 + (width / 2.0) * outx
                            cy = (l1_s["y"] + l1_e["y"]) / 2.0 + (width / 2.0) * outy

                            rot_rad = math.atan2(uy, ux)
                            deg = math.degrees(rot_rad) % 360
                            if 90 < deg <= 270:
                                rot_rad += math.pi

                            # Same text height as bearing/distances (template-matched)
                            road_title_height = bearing_road_height
                            road_title_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{road_title}}}"
                            # Width = length of drawn road: single line when title fits, wrap only when it exceeds road length
                            L_road = math.hypot(l1_e["x"] - l1_s["x"], l1_e["y"] - l1_s["y"])
                            txt_width = max(10.0, L_road)
                            self.autocad.add_mtext(
                                road_title_fmt,
                                cx,
                                cy,
                                layer="CADA_ROAD",
                                rotation_rad=rot_rad,
                                height=road_title_height,
                                width=txt_width,
                                attachment_point=5,
                            )
                except Exception:
                    pass

            # Move ENTIRE plan so the primary pillar sits at the exact coordinate specified by user
            dx_all = e0 - base_x
            dy_all = n0 - base_y
            self.autocad.move_all_modelspace(dx_all, dy_all)
            time.sleep(0.15)

            # Re-center the SHEET (border + title block/tables/north arrow/etc) around the plotted
            # land boundary WITHOUT moving survey geometry layers.
            try:
                boundary_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BOUNDARY"])
                # Prefer the actual border geometry layer so we move tables AND their boxes together.
                frame_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                if not frame_bb.get("success"):
                    # Next best: a border block reference (often named BORDER_*).
                    frame_bb = self.autocad.get_modelspace_bbox(block_name_contains="BORDER", prefer_largest=True)
                if not frame_bb.get("success"):
                    # fallback: union bbox of likely sheet content
                    frame_bb = self.autocad.get_modelspace_bbox(
                        layers=["CADA_BORDER", "CADA_INTERIORBORDER", "CADA_SCALEBAR", "CADA_TITLEBLOCK", "CADA_PLANNUMBER", "CADA_CERTIFICATION", "CADA_SURVEYOR", "CADA_NORTHARROW","CADA_EASTARROW", "CADA_COORDINATES", "CADA_NORTHCOORDINATES", "CADA_EASTCOORDINATES", "TITLE", "text"],
                        prefer_largest=False,
                    )
                if boundary_bb.get("success") and frame_bb.get("success"):
                    bc = boundary_bb.get("center") or {}
                    fc = frame_bb.get("center") or {}
                    movable_layers = list(profile.get("sheet_layers") or []) or [
                        "CADA_BORDER",
                        "CADA_INTERIORBORDER",
                        "CADA_SCALEBAR",      # includes BORDER_BLOCK + scalebar graphics
                        "CADA_NORTHARROW",
                        "CADA_EASTARROW",
                        "CADA_TITLEBLOCK",
                        "CADA_PLANNUMBER",
                        "CADA_CERTIFICATION",
                        "CADA_SURVEYOR",
                        "CADA_COORDINATES",
                        "CADA_NORTHCOORDINATES",
                        "CADA_EASTCOORDINATES",
                        "TITLE",
                        "text",
                    ]

                    # Remove arrow layers AND coordinates from general sheet move so we can align them specifically
                    arrow_layers = ["CADA_NORTHARROW", "CADA_EASTARROW", "CADA_COORDINATES", "CADA_NORTHCOORDINATES", "CADA_EASTCOORDINATES"]
                    movable_layers = [L for L in movable_layers if L not in arrow_layers]
                    # Never move survey geometry: boundary, bearings/distances, pillars, road (prevents displacement)
                    survey_geometry_layers = ["CADA_BOUNDARY", "CADA_BEARING_DIST", "CADA_PILLARS", "CADA_PILLARNUMBERS", "CADA_ROAD", "CADA_CWF"]
                    movable_layers = [L for L in movable_layers if L not in survey_geometry_layers]

                    # Apply the move and then do a quick correction pass (AutoCAD bbox can shift slightly).
                    total_dx_sheet = 0.0
                    total_dy_sheet = 0.0

                    for _ in range(2):
                        dx_sheet = float(bc.get("x", 0.0)) - float(fc.get("x", 0.0))
                        dy_sheet = float(bc.get("y", 0.0)) - float(fc.get("y", 0.0))
                        if abs(dx_sheet) < 0.05 and abs(dy_sheet) < 0.05:
                            break
                        self.autocad.move_modelspace_by_layers(dx_sheet, dy_sheet, movable_layers)
                        total_dx_sheet += dx_sheet
                        total_dy_sheet += dy_sheet

                        # Recompute frame center after the move
                        frame_bb = self.autocad.get_modelspace_bbox(layers=["CADA_BORDER"], prefer_largest=True)
                        if not frame_bb.get("success"):
                            frame_bb = self.autocad.get_modelspace_bbox(block_name_contains="BORDER", prefer_largest=True)
                        if not frame_bb.get("success"):
                            frame_bb = self.autocad.get_modelspace_bbox(layers=movable_layers, prefer_largest=False)
                        fc = (frame_bb.get("center") or {}) if frame_bb.get("success") else fc

                    # Identify coordinate text handles
                    easting_handles = []
                    northing_handles = []
                    try:
                        all_ents_res = self.autocad.get_all_entities()
                        if all_ents_res.get("success"):
                            for ent in all_ents_res.get("entities", []):
                                lyr = str(ent.get("layer") or "").upper()
                                txt = str(ent.get("text_content") or "").upper()
                                h = ent.get("handle")
                                if not h: continue

                                # Check legacy layer
                                if lyr == "CADA_COORDINATES":
                                    if ".E" in txt:
                                        easting_handles.append(h)
                                    elif ".N" in txt:
                                        northing_handles.append(h)
                                # Check new split layers
                                elif lyr == "CADA_EASTCOORDINATES":
                                    # Assuming this layer contains Easting text (.E)
                                    easting_handles.append(h)
                                elif lyr == "CADA_NORTHCOORDINATES":
                                    # Assuming this layer contains Northing text (.N)
                                    northing_handles.append(h)
                    except Exception:
                        pass

                    # Align CADA_NORTHARROW and Easting text: Align X with primary pillar (e0), move Y with sheet
                    try:
                        na_bb = self.autocad.get_modelspace_bbox(layers=["CADA_NORTHARROW"])
                        if na_bb.get("success"):
                            na_c = na_bb.get("center") or {}
                            na_x = float(na_c.get("x", 0.0))
                            dx = e0 - na_x
                            dy = total_dy_sheet
                            self.autocad.move_modelspace_by_layers(dx, dy, ["CADA_NORTHARROW"])
                            if easting_handles:
                                self.autocad.move_entities_by_handles(dx, dy, easting_handles)
                                # Fine-tune block removed to prevent displacement issues
                                # The text moves rigidly with the arrow layer, preserving template relative positions.
                    except Exception:
                        pass

                    # Align CADA_EASTARROW and Northing text: Align Y with primary pillar (n0), move X with sheet
                    try:
                        ea_bb = self.autocad.get_modelspace_bbox(layers=["CADA_EASTARROW"])
                        if ea_bb.get("success"):
                            ea_c = ea_bb.get("center") or {}
                            ea_y = float(ea_c.get("y", 0.0))
                            dx = total_dx_sheet
                            dy = n0 - ea_y
                            self.autocad.move_modelspace_by_layers(dx, dy, ["CADA_EASTARROW"])
                            if northing_handles:
                                self.autocad.move_entities_by_handles(dx, dy, northing_handles)
                                # Fine-tune block removed to prevent displacement issues
                                # The text moves rigidly with the arrow layer, preserving template relative positions.
                    except Exception:
                        pass

                    # Snap arrows + coordinate texts onto the (scaled) interior border edges (template-neat behavior)
                    try:
                        interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBORDER"])
                        if not interior_bb.get("success"):
                            interior_bb = self.autocad.get_modelspace_bbox(layers=["CADA_INTERIORBOUNDARY"])
                        if interior_bb.get("success"):
                            imin_x = float(interior_bb.get("minx", 0.0))
                            imax_x = float(interior_bb.get("maxx", 0.0))
                            imin_y = float(interior_bb.get("miny", 0.0))
                            imax_y = float(interior_bb.get("maxy", 0.0))
                            bcx = float((bc or {}).get("x", 0.0))
                            bcy = float((bc or {}).get("y", 0.0))

                            # East arrow should sit on left/right interior border (keep its Y already aligned to n0)
                            ea_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_EASTARROW"])
                            if ea_bb2.get("success"):
                                eac = ea_bb2.get("center") or {}
                                ea_cx = float(eac.get("x", 0.0))
                                left_side = ea_cx < bcx
                                if left_side:
                                    dx2 = imin_x - float(ea_bb2.get("minx", 0.0))
                                else:
                                    dx2 = imax_x - float(ea_bb2.get("maxx", 0.0))
                                if abs(dx2) > 1e-6:
                                    self.autocad.move_modelspace_by_layers(dx2, 0.0, ["CADA_EASTARROW"])
                                    if northing_handles:
                                        self.autocad.move_entities_by_handles(dx2, 0.0, northing_handles)

                            # North arrow should sit on top/bottom interior border (keep its X already aligned to e0)
                            na_bb2 = self.autocad.get_modelspace_bbox(layers=["CADA_NORTHARROW"])
                            if na_bb2.get("success"):
                                nac = na_bb2.get("center") or {}
                                na_cy = float(nac.get("y", 0.0))
                                top_side = na_cy > bcy
                                if top_side:
                                    dy2 = imax_y - float(na_bb2.get("maxy", 0.0))
                                else:
                                    dy2 = imin_y - float(na_bb2.get("miny", 0.0))
                                if abs(dy2) > 1e-6:
                                    self.autocad.move_modelspace_by_layers(0.0, dy2, ["CADA_NORTHARROW"])
                                    if easting_handles:
                                        self.autocad.move_entities_by_handles(0.0, dy2, easting_handles)
                    except Exception:
                        pass

                # Align coordinate text and north arrow lines with CADA_PRIMARYPILLAR_ARROWS lines:
                # - Horizontal arrows: shift coordinate text horizontally to align with primary pillar X
                # - Vertical arrows: shift coordinate text vertically to align with primary pillar Y
                try:
                    prim_x, prim_y = float(e0), float(n0)
                    all_ents = self.autocad.get_all_entities()
                    ents = (all_ents.get("entities") or []) if all_ents.get("success") else []

                    def _bbox_from_ent(ent: Dict[str, Any]) -> Optional[Dict[str, float]]:
                        coords = ent.get("coordinates") or {}
                        pts = []
                        if ent.get("type") == "LINE":
                            s = coords.get("start") or {}
                            e = coords.get("end") or {}
                            pts = [(s.get("x"), s.get("y")), (e.get("x"), e.get("y"))]
                        elif ent.get("type") in ("LWPOLYLINE", "POLYLINE"):
                            for p in (coords.get("points") or []):
                                pts.append((p.get("x"), p.get("y")))
                        elif ent.get("type") in ("TEXT", "MTEXT", "INSERT", "TABLE"):
                            ip = ent.get("insertion_point") or {}
                            pts = [(ip.get("x"), ip.get("y"))]
                        else:
                            c = ent.get("center") or {}
                            if c:
                                pts = [(c.get("x"), c.get("y"))]
                        pts = [(float(x), float(y)) for (x, y) in pts if x is not None and y is not None]
                        if not pts:
                            return None
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        return {"minx": min(xs), "miny": min(ys), "maxx": max(xs), "maxy": max(ys)}

                    # Find entities on CADA_PRIMARYPILLAR_ARROWS layer
                    arrow_handles = []
                    for ent in ents:
                        lyr = str(ent.get("layer") or "").upper()
                        if lyr == "CADA_PRIMARYPILLAR_ARROWS":
                            h = str(ent.get("handle") or "").strip()
                            if h:
                                arrow_handles.append(h)

                    if arrow_handles:
                        # Compute bbox of all primary pillar arrow entities
                        arrow_bb = None
                        for ent in ents:
                            h = str(ent.get("handle") or "").upper()
                            if h not in [ah.upper() for ah in arrow_handles]:
                                continue
                            bb = _bbox_from_ent(ent)
                            if not bb:
                                continue
                            if arrow_bb is None:
                                arrow_bb = dict(bb)
                            else:
                                arrow_bb["minx"] = min(arrow_bb["minx"], bb["minx"])
                                arrow_bb["miny"] = min(arrow_bb["miny"], bb["miny"])
                                arrow_bb["maxx"] = max(arrow_bb["maxx"], bb["maxx"])
                                arrow_bb["maxy"] = max(arrow_bb["maxy"], bb["maxy"])

                        if arrow_bb:
                            # Horizontal alignment: align X center of arrows with primary pillar X
                            arrow_x_center = 0.5 * (arrow_bb["minx"] + arrow_bb["maxx"])
                            dx_horiz = prim_x - arrow_x_center
                            if abs(dx_horiz) > 1e-6:
                                move_h = [str(h) for h in arrow_handles]
                                if east_h:
                                    move_h.append(str(east_h))
                                # Also move coordinate text entities that should follow horizontally
                                for ent in ents:
                                    lyr = str(ent.get("layer") or "").upper()
                                    txt = str(ent.get("text_content") or "").upper()
                                    if (lyr == "CADA_COORDINATES" and ".E" in txt) or (lyr == "CADA_EASTCOORDINATES"):
                                        h = str(ent.get("handle") or "").strip()
                                        if h:
                                            move_h.append(h)
                                self.autocad.move_entities_by_handles(dx_horiz, 0.0, move_h)

                            # Vertical alignment: align Y center of arrows with primary pillar Y
                            arrow_y_center = 0.5 * (arrow_bb["miny"] + arrow_bb["maxy"])
                            dy_vert = prim_y - arrow_y_center
                            if abs(dy_vert) > 1e-6:
                                move_h = [str(h) for h in arrow_handles]
                                if north_h:
                                    move_h.append(str(north_h))
                                # Also move coordinate text entities that should follow vertically
                                for ent in ents:
                                    lyr = str(ent.get("layer") or "").upper()
                                    txt = str(ent.get("text_content") or "").upper()
                                    if (lyr == "CADA_COORDINATES" and ".N" in txt) or (lyr == "CADA_NORTHCOORDINATES"):
                                        h = str(ent.get("handle") or "").strip()
                                        if h:
                                            move_h.append(h)
                                self.autocad.move_entities_by_handles(0.0, dy_vert, move_h)
                except Exception:
                    pass
            except Exception:
                pass

            # Zoom to extents of generated plan
            try:
                self.autocad.execute_command("ZOOM E")
            except Exception:
                pass

        try:
            self.autocad.execute_command("REGEN")
        except Exception:
            pass
        time.sleep(0.15)

        # Save + zoom extents (best-effort). STRICT: Never save if active doc is template.
        self._safe_save_active_drawing()
        try:
            self.autocad.execute_command("ZOOM E")
        except Exception:
            pass

        # Post-verify counts from ModelSpace so we return truthful geometry even if some ops were skipped.
        try:
            ms = self.autocad.doc.ModelSpace
            cnt_p = cnt_b = cnt_bd = 0
            for i in range(ms.Count):
                e = ms.Item(i)
                layer = str(getattr(e, "Layer", "")).upper()
                on = str(getattr(e, "ObjectName", ""))
                if layer == "CADA_PILLARS" and "BlockReference" in on:
                    cnt_p += 1
                if layer == "CADA_BOUNDARY":
                    cnt_b += 1
                if layer == "CADA_BEARING_DIST" and (on == "AcDbMText" or on == "AcDbText"):
                    cnt_bd += 1
            geometry["pillar_inserts"] = cnt_p
            geometry["boundary_entities"] = cnt_b
            geometry["bearing_mtext"] = cnt_bd
        except Exception:
            pass

        out_result = {"success": True, "output_dwg": str(outp), "geometry": geometry, "profile_path": profile_path}
        if geometry.get("access_road_title"):
            out_result["access_road_title"] = geometry["access_road_title"]
        return out_result

    # ==========================================================================
    # IN-SESSION CAD MODIFICATIONS (same output file, template always read-only)
    # ==========================================================================

    def _extract_dwg_path_from_query(self, query: str) -> Optional[str]:
        """Extract a .dwg file path from the query (quoted or path-like). Returns resolved path or None."""
        import re
        from pathlib import Path
        q = query or ""
        # Quoted paths: 'path.dwg' or "path.dwg"
        for pat in [r"['\"]([^'\"]+?\.dwg)['\"]", r"(?:in|to|file|open|modify)\s+['\"]?([^\s'\"]+\.dwg)['\"]?", r"([A-Za-z]:\\[^\s]+\.dwg)", r"([^\s<>|]+\.dwg)"]:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                raw = (m.group(1) or "").strip().strip("'\"")
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if p.suffix.lower() == ".dwg":
                    return str(p)
        return None

    def _ensure_protected_templates_loaded(self) -> None:
        """Load all known survey plan template paths from template_profiles so they are never written."""
        from pathlib import Path
        import json as _json
        profile_dir = self._cad_template_profiles_dir()
        if not profile_dir.exists():
            return
        for prof_path in profile_dir.glob("*.json"):
            try:
                data = _json.loads(prof_path.read_text(encoding="utf-8"))
                if isinstance(data.get("templates"), list):
                    for ent in data.get("templates") or []:
                        tp = str((ent or {}).get("path") or "")
                        if tp:
                            self._protected_template_paths.add(str(Path(tp).resolve()))
                    continue
                tp = (data.get("template") or {}).get("path") or ""
                if tp:
                    self._protected_template_paths.add(str(Path(tp).resolve()))
            except Exception:
                continue

    def _is_protected_template_path(self, dwg_path: str) -> bool:
        """True if dwg_path is a protected survey plan template (must never be written)."""
        if not dwg_path:
            return False
        self._ensure_protected_templates_loaded()
        try:
            from pathlib import Path
            # Must compare str to str (protected set stores str(Path.resolve()))
            return str(Path(dwg_path).resolve()) in self._protected_template_paths
        except Exception:
            return False

    def _safe_save_active_drawing(self) -> None:
        """
        Save the active drawing only if it is NOT a protected survey plan template.
        STRICT: Never write to the template (read-only to avoid corruption).
        If the active document is a template, save is skipped and a warning is logged.
        """
        active_path = self.autocad.get_active_document_path() if getattr(self.autocad, "get_active_document_path", None) else None
        if active_path and self._is_protected_template_path(active_path):
            logger.warning("Survey plan template is read-only; save skipped to avoid corruption.")
            return
        try:
            self.autocad.save_active_drawing()
        except Exception as e:
            logger.warning("save_active_drawing failed: %s", e)

    def _is_template_path(self, dwg_path: str) -> bool:
        """True if dwg_path is the template path from the last-used profile (template is read-only)."""
        if not dwg_path or not self._last_cadastral_profile_path:
            return False
        try:
            import json as _json
            from pathlib import Path
            prof = Path(self._last_cadastral_profile_path).resolve()
            if not prof.exists():
                return False
            profile = _json.loads(prof.read_text(encoding="utf-8"))
            template_path = (profile.get("template") or {}).get("path") or ""
            if not template_path:
                return False
            return Path(dwg_path).resolve() == Path(template_path).resolve()
        except Exception:
            return False

    def _should_fastpath_cad_modification(self, query: str) -> bool:
        """True when the user asks to modify a cadastral plan in the same session (e.g. add road, change title)."""
        q = (query or "").lower()
        if ".dwg" not in q:
            if not self._last_cadastral_output_dwg:
                return False
        mod_keywords = [
            "add another road", "add a road", "add road", "add access road",
            "change the title", "change title", "set title", "update title",
            "modify the plan", "modify plan", "edit the plan", "edit plan",
            "add road on the other side", "add road on the other side of the boundary",
            "change the plan title", "update the plan title",
        ]
        if not any(k in q for k in mod_keywords):
            return False
        return bool(self._last_cadastral_output_dwg or ".dwg" in q)

    def _run_cad_modification_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Apply modifications to an existing cadastral plan (output DWG) in the same session.
        Template file is never written to; only the output/working file is modified.
        Works even if the CAD file is already open (activates existing document).
        """
        import json as _json
        import math
        import re
        from pathlib import Path

        # Resolve target file: explicit path in query or last-generated output
        target = self._extract_dwg_path_from_query(query) or (self._last_cadastral_output_dwg and str(Path(self._last_cadastral_output_dwg).resolve()))
        if not target:
            return {"success": False, "error": "No plan file specified and no plan was generated in this session. Generate a plan first or specify the output .dwg file."}
        target_p = Path(target).resolve()
        if not target_p.exists():
            return {"success": False, "error": f"Plan file not found: {target_p}"}
        if self._is_template_path(target):
            return {"success": False, "error": "The template file is read-only and cannot be modified. Use the output plan file or the plan we just generated."}

        if not self.autocad.is_connected and not self.autocad.connect():
            return {"success": False, "error": "Could not connect to AutoCAD via COM"}
        opened = self.autocad.open_drawing(str(target_p), read_only=False)
        if not opened.get("success"):
            return {"success": False, "error": opened.get("error", "Failed to open plan drawing")}

        # Load profile: from last session or build minimal from current drawing
        profile = None
        if self._last_cadastral_output_dwg and Path(self._last_cadastral_output_dwg).resolve() == target_p and self._last_cadastral_profile_path:
            try:
                prof_path = Path(self._last_cadastral_profile_path).resolve()
                if prof_path.exists():
                    profile = _json.loads(prof_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not profile:
            tables = self.autocad.list_tables().get("tables") or []
            by_layer = {}
            for t in tables:
                by_layer.setdefault(str(t.get("layer") or ""), []).append(t)
            profile = {
                "tables": {
                    "title_block": {"handle": (by_layer.get("CADA_TITLEBLOCK", [{}])[0] or {}).get("handle")},
                    "plan_number": {"handle": (by_layer.get("CADA_PLANNUMBER", [{}])[0] or {}).get("handle")},
                    "surveyor": {"handle": (by_layer.get("CADA_SURVEYOR", [{}])[0] or {}).get("handle")},
                    "certification": {"handle": (by_layer.get("CADA_CERTIFICATION", [{}])[0] or {}).get("handle")},
                }
            }

        def _get_cell(h: str, r: int, c: int = 0) -> str:
            if not h:
                return ""
            res = self.autocad.get_table_cell_text(h, r, c)
            return str(res.get("text") or "") if res.get("success") else ""

        def _set_cell(h: str, r: int, c: int, val: str):
            if h:
                self.autocad.set_table_cell_text(h, r, c, val)

        def _mtxt_replace(raw: str, new_content: str) -> str:
            raw = raw or ""
            if raw.startswith("{") and raw.endswith("}") and ";" in raw:
                idx = raw.rfind(";")
                return raw[: idx + 1] + new_content + "}"
            return new_content

        modifications_done = []
        q = query or ""

        # --- Change title (buyer/title block row 2) ---
        title_match = re.search(r"(?:change|set|update)\s+(?:the\s+)?title\s+(?:to\s+)?['\"]?([^'\"]+)['\"]?|title\s+(?:as|to)\s+['\"]?([^'\"]+)['\"]?", q, re.IGNORECASE)
        if title_match:
            new_title = (title_match.group(1) or title_match.group(2) or "").strip()
            if new_title:
                tables = profile.get("tables", {})
                title_h = str((tables.get("title_block") or {}).get("handle") or "")
                if title_h:
                    cur = _get_cell(title_h, 2, 0)
                    _set_cell(title_h, 2, 0, _mtxt_replace(cur, new_title.upper()))
                    modifications_done.append("title")
                else:
                    modifications_done.append("title_skip_no_handle")

        # --- Add (another) access road ---
        add_road = any(phrase in q.lower() for phrase in ["add another road", "add a road", "add road", "add access road", "add road on the other side"])
        if add_road:
            # Parse width, offset, pillar ref from query
            width = 7.0
            offset = 0.0
            m_w = re.search(r"(?:width|wide)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
            if m_w:
                width = float(m_w.group(1))
            m_off = re.search(r"offset\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m", q, re.IGNORECASE)
            if m_off:
                offset = float(m_off.group(1))
            # Boundary points from current drawing (get_all_entities returns "points" for LWPOLYLINE)
            all_ents = self.autocad.get_all_entities()
            entities = (all_ents.get("entities") or []) if all_ents.get("success") else []
            local_pts = []
            for ent in entities:
                if str(ent.get("layer") or "").upper() != "CADA_BOUNDARY":
                    continue
                if str(ent.get("type") or "").upper() not in ("LWPOLYLINE", "POLYLINE"):
                    continue
                points = ent.get("points") or []
                if isinstance(points, list) and len(points) >= 3:
                    for p in points:
                        if isinstance(p, dict) and "x" in p and "y" in p:
                            local_pts.append({"x": float(p["x"]), "y": float(p["y"])})
                    break
            if len(local_pts) >= 3:
                # Edge index: use 1 for "another" road, 0 otherwise
                target_idx = 1 if "another" in q.lower() or "other side" in q.lower() else 0
                if target_idx >= len(local_pts):
                    target_idx = 0
                p1 = local_pts[target_idx]
                p2 = local_pts[(target_idx + 1) % len(local_pts)]
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                L_bound = math.hypot(dx, dy)
                if L_bound > 1e-6:
                    # Road length = 1.4 × traverse leg length
                    extension_total = 0.4 * L_bound
                    ext_side = extension_total / 2.0
                    ux, uy = dx / L_bound, dy / L_bound
                    midx = (p1["x"] + p2["x"]) / 2.0
                    midy = (p1["y"] + p2["y"]) / 2.0
                    poly_cx = sum(p["x"] for p in local_pts) / len(local_pts)
                    poly_cy = sum(p["y"] for p in local_pts) / len(local_pts)
                    vx, vy = poly_cx - midx, poly_cy - midy
                    n1x, n1y = uy, -ux
                    n2x, n2y = -uy, ux
                    if (n1x * vx + n1y * vy) >= (n2x * vx + n2y * vy):
                        outx, outy = -n1x, -n1y
                    else:
                        outx, outy = -n2x, -n2y
                    rsx = p1["x"] - ext_side * ux
                    rsy = p1["y"] - ext_side * uy
                    rex = p2["x"] + ext_side * ux
                    rey = p2["y"] + ext_side * uy
                    l1_s = {"x": rsx + offset * outx, "y": rsy + offset * outy}
                    l1_e = {"x": rex + offset * outx, "y": rey + offset * outy}
                    l2_s = {"x": rsx + (offset + width) * outx, "y": rsy + (offset + width) * outy}
                    l2_e = {"x": rex + (offset + width) * outx, "y": rey + (offset + width) * outy}
                    self.autocad.create_lwpolyline([l1_s, l1_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                    self.autocad.create_lwpolyline([l2_s, l2_e], layer="CADA_ROAD", closed=False, linetype_scale=3.0)
                    road_title = "ACCESS    ROAD"
                    m_title = re.search(r"(?:title|labeled|named)\s+['\"]([^'\"]+)['\"]", q, re.IGNORECASE)
                    if m_title:
                        road_title = m_title.group(1).strip()
                    cx = (l1_s["x"] + l1_e["x"]) / 2.0 + (width / 2.0) * outx
                    cy = (l1_s["y"] + l1_e["y"]) / 2.0 + (width / 2.0) * outy
                    rot_rad = math.atan2(uy, ux)
                    deg = math.degrees(rot_rad) % 360
                    if 90 < deg <= 270:
                        rot_rad += math.pi
                    road_title_fmt = f"{{\\fVerdana|b0|i0|c0|p34;{road_title}}}"
                    L_road = math.hypot(l1_e["x"] - l1_s["x"], l1_e["y"] - l1_s["y"])
                    txt_width = max(10.0, L_road)
                    road_h = float((profile.get("text_heights") or {}).get("bearing_dist_road") or 1.2)
                    try:
                        hr = self.autocad.get_sample_text_height(layers=["CADA_BEARING_DIST", "CADA_ROAD"])
                        if hr.get("success") and hr.get("height"):
                            road_h = float(hr["height"])
                    except Exception:
                        pass
                    self.autocad.add_mtext(road_title_fmt, cx, cy, layer="CADA_ROAD", rotation_rad=rot_rad, height=road_h, width=txt_width, attachment_point=5)
                    modifications_done.append("access_road")
            elif add_road:
                return {"success": False, "error": "Could not find boundary (CADA_BOUNDARY) in the plan to add the road."}

        if not modifications_done:
            return {"success": False, "error": "Could not parse a modification from your request (e.g. 'change title to X' or 'add another road on the other side')."}

        # STRICT: Never save if active doc is template (read-only to avoid corruption).
        self._safe_save_active_drawing()
        try:
            self.autocad.execute_command("ZOOM E")
        except Exception:
            pass
        return {"success": True, "output_dwg": str(target_p), "modifications": modifications_done}

    def _run_docx_report_pipeline(
        self,
        query: str,
        output_doc_path: str,
        llm: BaseChatModel,
        model_name_used: str,
    ) -> Dict[str, Any]:
        """
        Deterministic report generation pipeline to avoid tool-loop recursion:
        - (Optional) fetch a small set of internet sources (permissioned)
        - single LLM call to draft report
        - save to output_doc_path
        """
        from pathlib import Path

        output_path = Path(output_doc_path)
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path.name).resolve()

        # Internet sourcing (permissioned)
        internet_block = ""
        if getattr(self, "_internet_permission_granted", False):
            try:
                # Build targeted searches from the user query (best-effort, no extra LLM call).
                ql = (query or "").lower()
                searches = []
                # Domain-specific boosts
                if "nigeria" in ql or "surcon" in ql or "surveying" in ql:
                    searches.extend(
                        [
                            "Surveyors Council of Nigeria SURCON licensing requirements",
                            "Surveyors Council of Nigeria Act",
                            "Nigerian Institution of Surveyors history",
                            "practice of surveying in Nigeria history",
                            "SURCON register of surveyors Nigeria licensing process",
                        ]
                    )
                # Also include the raw query (truncated)
                searches.append((query or "")[:180])
                # De-dup / sanitize
                searches = [s.strip() for s in searches if s and s.strip()]
                seen_s = set()
                searches = [s for s in searches if not (s.lower() in seen_s or seen_s.add(s.lower()))]
                searches = searches[:6]
                hits = []
                for s in searches:
                    res = _internet_search(s)
                    if res.get("success"):
                        for r in (res.get("results") or [])[:3]:
                            hits.append(
                                f"- {r.get('title','')}\n  - {r.get('url','')}\n  - {r.get('snippet','')}"
                            )
                if hits:
                    internet_block = (
                        "\n\nINTERNET-SOURCED (EXTERNAL) REFERENCES (permission granted):\n"
                        + "\n".join(hits[:12])
                        + "\n\nIMPORTANT: Treat as external; prefer official government / SURCON / NIS sources where possible.\n"
                    )
            except Exception as e:
                internet_block = f"\n\n[Internet lookup failed: {e}]\n"

        prompt = (
            "Write a well-structured report for a professional audience.\n"
            "REQUIREMENTS:\n"
            "- Follow the USER REQUEST exactly.\n"
            "- Include a section titled exactly: 'Internet-sourced (external) information' listing any external links used.\n"
            "- Use citations inline like [1], [2] where appropriate, matching the links you list.\n"
            "- If internet results are missing/empty, clearly state that live sources could not be retrieved and proceed with an offline explanation.\n"
            "- Keep the report structured with headings and bullet points.\n"
            "- Output should be suitable for saving into a Word document.\n\n"
            f"USER REQUEST:\n{query}\n"
            f"{internet_block}\n"
        )

        report_msg, err, timed_out = self._run_with_timeout(
            180, lambda: llm.invoke([HumanMessage(content=prompt)])
        )
        if timed_out:
            return {
                "success": False,
                "error": "LLM report call timed out after 180 seconds",
                "response": "LLM report call timed out. Try increasing AGENT_QUERY_TIMEOUT or using a smaller scope.",
                "output_path": str(output_path),
            }
        if err:
            return {
                "success": False,
                "error": str(err),
                "response": f"LLM report call failed: {err}",
                "output_path": str(output_path),
            }

        report_text = report_msg.content if hasattr(report_msg, "content") else str(report_msg)
        title = f"Report - {output_path.stem}"
        create_result = self.document_processor.create_word_document(str(output_path), report_text, title=title)
        if not create_result.get("success"):
            return {
                "success": False,
                "error": create_result.get("error", "Failed to create report document"),
                "response": str(create_result),
                "output_path": str(output_path),
            }

        return {
            "success": True,
            "response": (
                f"✅ Created report document:\n"
                f"- Output: {str(output_path)}\n"
                f"- Model: {model_name_used}\n"
                f"- Internet used: {'yes' if getattr(self, '_internet_permission_granted', False) else 'no'}\n"
            ),
            "output_path": str(output_path),
            "model_name": model_name_used,
        }

    def _run_large_doc_summary_pipeline(
        self,
        query: str,
        input_doc_path: str,
        output_doc_path: str,
        llm: BaseChatModel,
        model_name_used: str
    ) -> Dict[str, Any]:
        """
        Large-document summarization pipeline (fast path).
        
        Steps:
        - Preflight estimation (already done by caller)
        - Extract only relevant sections by keywords (streaming, no full text)
        - Pull relevant tables (coordinates/control points) and include them as context
        - Ask the LLM ONCE to write the final 3-page style summary
        - Save to output_doc_path
        """
        from pathlib import Path

        input_path = Path(input_doc_path)
        output_path = Path(output_doc_path)

        # Keyword set tuned for survey QA/QC extraction
        keywords = [
            "Location", "Personnel", "Personnels", "Contractor", "Client",
            "Purpose", "Scope", "Date", "Duration", "Equipment", "Equipments",
            "Quantities", "Quantity", "Achieved", "Surveyed",
            "Control", "Control Point", "Control Points",
            "Coordinate", "Coordinates", "Easting", "Northing", "UTM",
            "Check", "QC", "QA", "In-situ", "Insitu", "Verification", "Validation"
        ]

        section_result = self.document_processor.extract_sections_by_keywords(
            str(input_path), keywords=keywords, context_lines=12
        )

        # Avoid full table extraction on huge Word docs (python-docx can be very slow).
        # Instead, pull coordinate-like snippets from extracted text.
        relevant_tables: list[dict[str, Any]] = []

        extracted_text = ""
        if section_result.get("success"):
            extracted_text = section_result.get("extracted_text", "") or ""

        # Hard cap extracted text to avoid blowing context on pathological docs
        if len(extracted_text) > 200_000:
            extracted_text = extracted_text[:200_000] + "\n\n[TRUNCATED: extracted_text too long]\n"

        # Quick coordinate snippets
        coord_snippets = []
        try:
            import re
            patterns = [
                r"\bE(?:asting)?[:\s]*\d{5,}\.?\d*\s+N(?:orthing)?[:\s]*\d{5,}\.?\d*\b",
                r"\b\d{5,}\.?\d*\s*,\s*\d{5,}\.?\d*\b",
                r"\b(?:X|E)[:\s]*\d{5,}\.?\d*\s+(?:Y|N)[:\s]*\d{5,}\.?\d*\b",
            ]
            for pat in patterns:
                coord_snippets.extend(re.findall(pat, extracted_text, flags=re.IGNORECASE))
            coord_snippets = list(dict.fromkeys(coord_snippets))[:30]
        except Exception:
            coord_snippets = []

        prompt = (
            "You are a senior QA/QC Surveyor. Produce a professional 3-page-style summary (concise but complete) "
            "from the extracted sections and tables provided.\n\n"
            "REQUIREMENTS:\n"
            "- Extract and summarize: Location, Personnel involved, Contractor and Client, Purpose/scope, "
            "date and duration, equipment used, quantities achieved/surveyed, control points used, "
            "coordinates, and in-situ/verification checks.\n"
            "- If a field is not present in the extracted content, write 'Not stated in the provided extract' (do NOT guess).\n"
            "- Prefer structured headings and bullet points.\n"
            "- Keep it within ~1200-1800 words (approx 3 pages).\n\n"
            f"SOURCE DOCUMENT: {input_path.name}\n\n"
            "EXTRACTED TEXT (partial, keyword-based):\n"
            "----------------\n"
            f"{extracted_text}\n\n"
            "COORDINATE / CONTROL-POINT SNIPPETS (auto-detected):\n"
            "----------------\n"
            f"{coord_snippets}\n"
        )

        summary_msg, err, timed_out = self._run_with_timeout(
            120, lambda: llm.invoke([HumanMessage(content=prompt)])
        )
        if timed_out:
            return {
                "success": False,
                "error": "LLM summary call timed out after 120 seconds",
                "response": "LLM summary call timed out. Try increasing AGENT_QUERY_TIMEOUT or using a smaller keyword extract.",
                "output_path": str(output_path),
            }
        if err:
            return {
                "success": False,
                "error": str(err),
                "response": f"LLM summary call failed: {err}",
                "output_path": str(output_path),
            }
        summary_text = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)

        title = f"Summary - {input_path.stem}"
        create_result = self.document_processor.create_word_document(str(output_path), summary_text, title=title)

        if not create_result.get("success"):
            return {
                "success": False,
                "error": create_result.get("error", "Failed to create summary document"),
                "response": str(create_result),
                "output_path": str(output_path)
            }

        response_text = (
            f"✅ Created summary document:\n"
            f"- Source: {str(input_path)}\n"
            f"- Output: {str(output_path)}\n"
            f"- Model: {model_name_used}\n"
        )

        return {
            "success": True,
            "response": response_text,
            "output_path": str(output_path),
            "model_name": model_name_used
        }
    
    def _retrieve_relevant_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        collections: Optional[List[str]] = None,
    ) -> str:
        """
        Retrieve relevant context from the vector store for the given query.
        
        Prioritizes:
        1. Recent conversations from the current session (if session_id provided)
        2. Semantic search across all conversations
        3. Relevant documents and drawings
        
        Args:
            query: The user's query to find relevant context for
            session_id: Session ID to prioritize recent session context
            
        Returns:
            Formatted context string to inject into the conversation
        """
        if self.vector_store is None:
            return ""
        
        if not getattr(self.settings, 'auto_context_retrieval', True):
            return ""
        
        top_k = getattr(self.settings, 'context_retrieval_top_k', 5)
        threshold = getattr(self.settings, 'context_score_threshold', 0.3)

        allowed = set(collections or [])
        allow_all = not allowed
        
        context_parts = []
        recent_convs = []  # Initialize to avoid NameError
        
        try:
            # PRIORITY 1: Get recent conversations from current session
            if session_id and (allow_all or COLLECTION_CONVERSATIONS in allowed):
                recent_convs = self.vector_store.get_recent_conversations(
                    session_id=session_id,
                    limit=10  # Get last 10 messages from this session
                )
                
                if recent_convs:
                    # Format recent conversation history (most recent first)
                    context_parts.append("**RECENT CONVERSATION HISTORY (Current Session):**")
                    for conv in recent_convs[:5]:  # Show last 5 messages
                        role = conv.get('metadata', {}).get('role', 'unknown')
                        content = conv.get('content', '')
                        # Truncate long content
                        if len(content) > 400:
                            content = content[:400] + "..."
                        context_parts.append(f"  [{role.upper()}]: {content}")
                    context_parts.append("")  # Empty line separator
            
            relevant_convs = []
            other_session_convs = []
            if allow_all or COLLECTION_CONVERSATIONS in allowed:
                # PRIORITY 2: Semantic search for relevant conversations
                # Strategy: Get more results, then separate by session
                conv_results = self.vector_store.search(
                    query=query,
                    collection=COLLECTION_CONVERSATIONS,
                    top_k=top_k * 2  # Get more to filter by session
                )
                
                # Separate current session from other sessions
                if session_id:
                    current_session_convs = [
                        r for r in conv_results 
                        if r.get('metadata', {}).get('session_id') == session_id
                        and r.get('score', 0) >= threshold
                    ]
                    other_session_convs = [
                        r for r in conv_results 
                        if r.get('metadata', {}).get('session_id') != session_id
                        and r.get('score', 0) >= threshold
                    ]
                    # Prioritize current session, but also include highly relevant from other sessions
                    relevant_convs = current_session_convs[:3] + other_session_convs[:2]
                else:
                    relevant_convs = [r for r in conv_results if r.get('score', 0) >= threshold][:5]
                
                # Show semantic results if we have them and didn't already show recent session history
                if relevant_convs and not (session_id and recent_convs):
                    context_parts.append("**Relevant Past Conversations:**")
                    for i, result in enumerate(relevant_convs[:3], 1):
                        role = result.get('metadata', {}).get('role', 'unknown')
                        score = result.get('score', 0)
                        result_session = result.get('metadata', {}).get('session_id', 'unknown')
                        is_current_session = session_id and result_session == session_id
                        session_label = " (current session)" if is_current_session else " (past session)"
                        content_preview = result.get('content', '')[:300]
                        context_parts.append(
                            f"  {i}. [{role}]{session_label} (relevance: {score:.2f}): {content_preview}..."
                        )
                elif other_session_convs and session_id and recent_convs:
                    # We have recent session history, but also show highly relevant from other sessions
                    highly_relevant_other = [r for r in other_session_convs if r.get('score', 0) >= 0.7][:2]
                    if highly_relevant_other:
                        context_parts.append("\n**Highly Relevant from Past Sessions:**")
                        for i, result in enumerate(highly_relevant_other, 1):
                            role = result.get('metadata', {}).get('role', 'unknown')
                            score = result.get('score', 0)
                            content_preview = result.get('content', '')[:300]
                            context_parts.append(
                                f"  {i}. [{role}] (relevance: {score:.2f}): {content_preview}..."
                            )
            
            relevant_docs = []
            if allow_all or COLLECTION_DOCUMENTS in allowed:
                # PRIORITY 3: Search documents for relevant information
                doc_results = self.vector_store.search(
                    query=query,
                    collection=COLLECTION_DOCUMENTS,
                    top_k=top_k
                )
                relevant_docs = [r for r in doc_results if r.get('score', 0) >= threshold]
            
            if relevant_docs:
                context_parts.append("\n**Relevant Documents:**")
                for i, result in enumerate(relevant_docs[:3], 1):
                    source = result.get('metadata', {}).get('source', 'unknown')
                    score = result.get('score', 0)
                    content_preview = result.get('content', '')[:300]
                    context_parts.append(f"  {i}. [Source: {source}] (relevance: {score:.2f}): {content_preview}...")
            
            relevant_draws = []
            if allow_all or COLLECTION_DRAWINGS in allowed:
                # PRIORITY 4: Search drawings for relevant CAD data
                draw_results = self.vector_store.search(
                    query=query,
                    collection=COLLECTION_DRAWINGS,
                    top_k=3
                )
                relevant_draws = [r for r in draw_results if r.get('score', 0) >= threshold]

            relevant_coords = []
            if allow_all or COLLECTION_COORDINATES in allowed:
                coord_results = self.vector_store.search(
                    query=query,
                    collection=COLLECTION_COORDINATES,
                    top_k=top_k
                )
                relevant_coords = [r for r in coord_results if r.get('score', 0) >= threshold]
                if relevant_coords:
                    context_parts.append("\n**Relevant Coordinate Data:**")
                    for i, result in enumerate(relevant_coords[:3], 1):
                        src = result.get("metadata", {}).get("source", "unknown")
                        score = result.get("score", 0)
                        content_preview = result.get("content", "")[:300]
                        context_parts.append(f"  {i}. [Source: {src}] (relevance: {score:.2f}): {content_preview}...")
            
            if relevant_draws:
                context_parts.append("\n**Relevant Drawing Data:**")
                for i, result in enumerate(relevant_draws[:2], 1):
                    drawing = result.get('metadata', {}).get('drawing_name', 'unknown')
                    entity_type = result.get('metadata', {}).get('entity_type', 'unknown')
                    score = result.get('score', 0)
                    context_parts.append(f"  {i}. [Drawing: {drawing}, Type: {entity_type}] (relevance: {score:.2f})")
            
            if context_parts:
                total_convs = len(recent_convs) if session_id and recent_convs else len(relevant_convs)
                logger.info(
                    f"✓ Retrieved {total_convs} conversations, {len(relevant_docs)} documents, "
                    f"{len(relevant_draws)} drawings, {len(relevant_coords)} coordinates as context"
                )
                return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"⚠ Context retrieval failed: {e}")
        
        return ""
    
    def _store_conversation(
        self, 
        query: str, 
        response: str, 
        session_id: str,
        llm_used: str = "primary"
    ) -> None:
        """
        Store a conversation exchange in the vector store for future context.
        
        Args:
            query: The user's original query
            response: The agent's response
            session_id: Session identifier for grouping conversations
            llm_used: Which LLM was used for the response
        """
        if self.vector_store is None:
            return
        
        if not getattr(self.settings, 'auto_store_conversations', True):
            return
        
        try:
            timestamp = datetime.now().isoformat()
            
            # Store the user query
            self.vector_store.add_conversation(
                role="user",
                content=query,
                session_id=session_id,
                metadata={
                    "timestamp": timestamp,
                    "type": "query"
                }
            )
            
            # Store the assistant response
            self.vector_store.add_conversation(
                role="assistant",
                content=response,
                session_id=session_id,
                metadata={
                    "timestamp": timestamp,
                    "type": "response",
                    "llm_used": llm_used
                }
            )
            
            logger.debug(f"✓ Stored conversation in vector store (session: {session_id[:8]}...)")
            
        except Exception as e:
            logger.warning(f"⚠ Failed to store conversation: {e}")
    
    def set_session_id(self, session_id: str) -> None:
        """
        Set a persistent session ID for conversation continuity.
        
        Using the same session ID across queries allows the agent to maintain
        context and retrieve relevant past conversations.
        
        Args:
            session_id: Unique identifier for this conversation session
        """
        self._current_session_id = session_id
        logger.info(f"Session ID set: {session_id[:8]}...")
    
    def get_session_id(self) -> str:
        """
        Get the current session ID, generating one if not set.
        
        Returns:
            The current session ID
        """
        if self._current_session_id is None:
            self._current_session_id = str(uuid.uuid4())
        return self._current_session_id
    
    # ==========================================================================
    # COMPLEXITY DETECTION
    # ==========================================================================
    
    def _detect_task_complexity(self, query: str) -> Literal["simple", "average", "complex"]:
        """
        Analyze query to determine task complexity level.
        
        This heuristic-based approach categorizes queries into:
        - simple: Basic questions, lookups, single operations
        - average: Multi-step tasks, calculations, coordinate conversions
        - complex: Multi-tool workflows, complex reasoning, analysis tasks
        
        Args:
            query: The user's query string
            
        Returns:
            One of "simple", "average", or "complex"
        """
        query_lower = query.lower()
        
        # Simple task indicators (basic lookups, single operations)
        simple_indicators = [
            "what is", "what are", "tell me", "explain", "define",
            "show me", "list", "find", "search", "lookup",
            "is", "are", "can you", "does", "do you know"
        ]
        
        # Complex task indicators (multi-step, analysis, complex reasoning)
        complex_indicators = [
            "analyze", "compare", "calculate total", "compute volume",
            "multiple", "several", "all", "combine", "integrate",
            "correlate", "cross-reference", "relationship between",
            "create project", "generate report", "perform analysis",
            "extract and", "retrieve and", "process and",
            "coordinate system", "projection", "transformation",
            "calculate area", "measure distance", "boundary analysis"
        ]
        
        # Count keywords
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        
        # Check for multiple tool usage (indicates complexity)
        tool_indicators = ["autocad", "arcgis", "excel", "coordinate", "convert", "extract"]
        tool_count = sum(1 for indicator in tool_indicators if indicator in query_lower)
        
        # Check query length (longer queries often more complex)
        word_count = len(query.split())
        
        # Determine complexity
        if complex_count >= 2 or tool_count >= 3 or word_count > 30:
            return "complex"
        elif complex_count >= 1 or tool_count >= 2 or word_count > 15:
            return "average"
        elif simple_count >= 1 and complex_count == 0 and tool_count <= 1:
            return "simple"
        else:
            # Default to average for ambiguous cases
            return "average"

    # ==========================================================================
    # AGENTIC RAG ROUTING
    # ==========================================================================

    def _decide_rag_route(self, query: str, interactive_mode: bool = False) -> RAGRouteDecision:
        """
        Agentic RAG router: decide whether to:
        - call LLM directly (no augmentation),
        - retrieve local context from VectorStore,
        - run permissioned internet search,
        - or do both (hybrid).

        This is intentionally lightweight and defaults to HEURISTIC routing to avoid
        extra LLM calls. (An LLM-router can be added later behind a settings flag.)
        """
        q = (query or "").strip()
        ql = q.lower()

        # If query includes explicit input file paths, prefer tool/document pipelines over RAG.
        # (RAG is still allowed if user asks "based on our previous conversation" etc.)
        file_driven = looks_like_file_driven_task(q) or bool(self._extract_document_paths(q))

        wants_memory = any(k in ql for k in [
            "previous", "earlier", "last time", "as we discussed", "from our conversation",
            "continue", "resume", "what did you say", "you mentioned", "same as before",
        ])

        # Internet-needed signals (permissioned)
        # Expanded to catch more cases where internet search would be helpful
        # BUT: Exclude CAD/AutoCAD continuation tasks - these don't need internet
        cad_continuation_indicators = [
            "add another", "add a", "now add", "also add", "continue with",
            "draw another", "create another", "add to the", "modify the",
            "update the", "change the", "edit the", "in the same", "same drawing",
            "same file", "same dwg", "same dxf", "to the drawing", "to the plan"
        ]
        is_cad_continuation = any(indicator in ql for indicator in cad_continuation_indicators)
        
        # Also check if previous context mentions CAD/AutoCAD work
        has_cad_context = any(
            keyword in ql for keyword in [
                "autocad", "cad", "dwg", "dxf", "drawing", "plan", "survey plan",
                "pillar", "coordinates", "boundary", "road", "access road"
            ]
        )
        
        internet_signals = [
            "according to api", "api mpms", "api 653", "api standard", "astm", "iso",
            "latest", "current", "updated", "as of", "2023", "2024", "2025", "2026",
            "cite", "citations", "references", "journal", "paper", "studies",
            "who said", "source", "link", "search the internet", "search online",
            "find information", "look up", "what are the", "what is the", "constraints",
            "problems", "issues", "challenges", "standards", "regulations", "requirements",
            "nigerian", "nigeria", "country-specific", "national", "local standards",
        ]
        wants_internet = any(s in ql for s in internet_signals)
        
        # Override: If this looks like CAD continuation, don't ask for internet
        # Also check if the query mentions continuation context
        has_continuation_context = (
            "=== CONTINUATION OF PREVIOUS WORK" in q or
            "--- Exchange" in q or
            "PREVIOUS CONVERSATION" in q.upper()
        )
        
        if (is_cad_continuation and has_cad_context) or (has_continuation_context and has_cad_context):
            wants_internet = False
            logger.info("🔧 Detected CAD continuation task - skipping internet search request")
            # Also mark that this wants memory/context from previous conversation
            wants_memory = True

        # Local retrieval signals: user is asking about *their* stored materials,
        # prior runs, prior outputs, or asks to "search my documents".
        vector_signals = [
            "from my documents", "my document", "my drawing", "my dwg", "my dxf",
            "search the database", "search the vector store", "retrieve", "stored",
            "chroma", "vectordb", "what did we store", "our chat", "conversation history",
            "use the context", "based on stored",
        ]
        wants_vector = wants_memory or any(s in ql for s in vector_signals)

        # If it's a file-driven workflow (doc/dwg/xlsx), don't auto-retrieve unless user
        # explicitly wants memory/context. This prevents irrelevant injection.
        if file_driven and not wants_vector:
            return RAGRouteDecision(
                route="llm_only",
                use_vector=False,
                use_internet=False,
                reason="File-driven workflow detected; avoiding automatic RAG injection.",
            )

        # Choose route
        if wants_internet and wants_vector:
            return RAGRouteDecision(
                route="hybrid",
                use_vector=True,
                vector_collections=[COLLECTION_CONVERSATIONS, COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS, COLLECTION_COORDINATES],
                use_internet=True,
                internet_query=query,
                reason="User requested external standards/citations and also referenced prior/stored context.",
            )
        if wants_internet:
            return RAGRouteDecision(
                route="internet",
                use_vector=False,
                use_internet=True,
                internet_query=query,
                reason="Query asks for standards/citations/current information likely requiring web search.",
            )
        if wants_vector:
            return RAGRouteDecision(
                route="vector",
                use_vector=True,
                vector_collections=[COLLECTION_CONVERSATIONS, COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS, COLLECTION_COORDINATES],
                use_internet=False,
                reason="Query references prior/stored context; local retrieval likely helpful.",
            )

        # Default: LLM only (no augmentation)
        return RAGRouteDecision(
            route="llm_only",
            use_vector=False,
            use_internet=False,
            reason="No strong signal for retrieval or web search.",
        )

    def _format_internet_results_for_prompt(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for r in results[:8]:
            title = r.get("title", "").strip()
            url = r.get("url", "").strip()
            snippet = r.get("snippet", "").strip()
            if not (title or url or snippet):
                continue
            lines.append(f"- {title}\n  - {url}\n  - {snippet}")
        if not lines:
            return ""
        return (
            "\n\n---\n"
            "**INTERNET SEARCH RESULTS (EXTERNAL, PERMISSION GRANTED):**\n"
            "If you use any of this information, you MUST include a section titled exactly:\n"
            "\"Internet-sourced (external) information\" and include the URLs you relied on.\n\n"
            + "\n".join(lines)
            + "\n---\n"
        )
    
    def _get_model_tier(self, model_name: Optional[str]) -> str:
        """
        Determine the tier of a model (nano, mini, or complex).
        
        Args:
            model_name: Model name (e.g., "gpt-5-nano", "gpt-5-mini", "gpt-5.1")
            
        Returns:
            "nano", "mini", or "complex"
        """
        if not model_name:
            return "mini"  # Default
        
        model_lower = model_name.lower()
        if "nano" in model_lower:
            return "nano"
        elif "mini" in model_lower or "4o-mini" in model_lower:
            return "mini"
        elif "5.1" in model_lower or ("5" in model_lower and "nano" not in model_lower and "mini" not in model_lower) or "4o" in model_lower or "4-turbo" in model_lower:
            return "complex"
        else:
            return "mini"  # Default to mini for unknown models
    
    def _escalate_model_tier(self, current_tier: str) -> Optional[str]:
        """
        Get the next higher tier model for escalation.
        
        Args:
            current_tier: Current tier ("nano", "mini", or "complex")
            
        Returns:
            Model name for next tier, or None if already at highest tier
        """
        if current_tier == "nano":
            return getattr(self.settings, "openai_model_mini", "gpt-5-mini")
        elif current_tier == "mini":
            return getattr(self.settings, "openai_model_complex", "gpt-5.1")
        else:
            return None  # Already at highest tier
    
    def _switch_model_and_retry(
        self,
        query: str,
        original_query: str,
        current_model: Optional[str],
        current_llm: BaseChatModel,
        complexity: str,
        enhanced_system_prompt: str,
        initial_messages: List[BaseMessage],
        current_session_id: str,
        use_fallback: bool,
        interactive_mode: bool,
        context_retrieved: bool,
        switch_reason: str,
        tools_to_bind: List[BaseTool],
    ) -> Dict:
        """
        Switch to a higher-tier model and retry the query.
        
        This preserves state and seamlessly continues with a more capable model.
        """
        current_tier = self._get_model_tier(current_model)
        escalated_model = self._escalate_model_tier(current_tier)
        
        if not escalated_model:
            logger.warning("⚠ Already at highest model tier - cannot escalate further")
            # Return error instead of switching
            return {
                "query": original_query,
                "response": (
                    f"Query failed with {switch_reason}. "
                    "Already using the most capable model available. "
                    "The query may be too complex or require manual intervention."
                ),
                "success": False,
                "error": switch_reason,
                "llm_used": "fallback" if use_fallback else "primary",
                "model_name": current_model,
                "session_id": current_session_id,
            }
        
        logger.info(f"🔄 Switching from {current_model} (tier: {current_tier}) to {escalated_model} (tier: {self._get_model_tier(escalated_model)})")
        
        # Mark that we've switched to prevent infinite switching
        self._model_switched_this_query = True
        
        # Initialize new model
        try:
            new_llm = self._initialize_llm("openai", model_name=escalated_model)
            self._current_openai_model = escalated_model
            
            # Rebind tools with new LLM
            self.llm_with_tools = new_llm.bind_tools(tools_to_bind)
            self.graph = self._build_graph()
            self.app = self.graph.compile(checkpointer=self.memory)
            
            logger.info(f"✓ Switched to {escalated_model} - retrying query")
            
            # Retry with new model (preserve state via session_id)
            thread_id = current_session_id
            max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
            recursion_limit = getattr(self.settings, "agent_recursion_limit", max(50, (max_iterations * 3)))
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            }
            
            # Use the same initial messages (state is preserved via session_id)
            initial_state = {"messages": initial_messages}
            
            # Invoke with new model
            result = self.app.invoke(initial_state, config=config)
            
            # Extract response
            response_text = self._extract_response(result)
            
            # Store conversation
            llm_used = "fallback" if use_fallback else "primary"
            self._store_conversation(
                query=query,
                response=response_text,
                session_id=current_session_id,
                llm_used=llm_used
            )
            
            logger.info(f"✓ Model switch successful - query completed with {escalated_model}")
            
            return {
                "query": query,
                "response": response_text,
                "llm_used": llm_used,
                "model_name": escalated_model,
                "complexity": complexity,
                "success": True,
                "session_id": current_session_id,
                "context_retrieved": context_retrieved,
                "model_switched": True,
                "original_model": current_model,
                "switch_reason": switch_reason,
            }
            
        except Exception as e:
            logger.error(f"❌ Model switch failed: {e}")
            # Return error
            return {
                "query": original_query,
                "response": f"Query failed and model switch also failed: {e}",
                "success": False,
                "error": str(e),
                "llm_used": "fallback" if use_fallback else "primary",
                "model_name": current_model,
                "session_id": current_session_id,
            }
    
    def _get_openai_model_for_complexity(self, complexity: Literal["simple", "average", "complex"]) -> str:
        """
        Select appropriate OpenAI model based on task complexity.
        
        Args:
            complexity: Task complexity level ("simple", "average", or "complex")
            
        Returns:
            Model name string (e.g., "gpt-5-nano", "gpt-5-mini", "gpt-5.1")
        """
        if not getattr(self.settings, "enable_tiered_models", True):
            # Fallback to legacy single model configuration
            return getattr(self.settings, "openai_model", "gpt-4o-mini")
        
        # Map complexity to model tier
        model_mapping = {
            "simple": getattr(self.settings, "openai_model_nano", "gpt-5-nano"),
            "average": getattr(self.settings, "openai_model_mini", "gpt-5-mini"),
            "complex": getattr(self.settings, "openai_model_complex", "gpt-5.1"),
        }
        
        selected_model = model_mapping.get(complexity, getattr(self.settings, "openai_model_mini", "gpt-5-mini"))
        
        # Fallback to legacy model if tiered model is not set
        if not selected_model or selected_model.strip() == "":
            selected_model = getattr(self.settings, "openai_model", "gpt-4o-mini")
        
        return selected_model
    
    # ==========================================================================
    # LLM INITIALIZATION
    # ==========================================================================
    
    def _initialize_llm(self, llm_type: str, model_name: Optional[str] = None) -> BaseChatModel:
        """
        Initialize a Language Model based on the specified type.
        
        Supported LLM types:
        - "gemini": Google's Gemini models (gemini-2.0-flash, gemini-1.5-flash, gemini-pro-latest)
        - "deepseek": DeepSeek's models via OpenAI-compatible API
        - "claude": Anthropic's Claude models (Opus, Sonnet, Haiku)
        - "openai": OpenAI's models (GPT-4, GPT-4o, GPT-4o-Turbo, GPT-5, GPT-5-nano, GPT-5-mini, GPT-5.1)
        
        Args:
            llm_type: One of "gemini", "deepseek", "claude", or "openai"
            model_name: Optional specific model name (for OpenAI tiered models). If None, uses default from settings.
            
        Returns:
            BaseChatModel: An initialized LLM ready for use
            
        Raises:
            ValueError: If llm_type is not recognized or API key is missing
            Exception: If API connection fails
        """
        try:
            if llm_type == "deepseek":
                # DeepSeek uses an OpenAI-compatible API
                # We use ChatOpenAI with a custom base_url
                if not self.settings.deepseek_api_key:
                    raise ValueError("DEEPSEEK_API_KEY is required but not set")
                
                logger.info("Initializing DeepSeek LLM")
                return ChatOpenAI(
                    model="deepseek-chat",
                    api_key=self.settings.deepseek_api_key,
                    base_url=self.settings.deepseek_base_url,
                    temperature=self.settings.agent_temperature,
                    max_tokens=self.settings.agent_max_tokens
                )
                
            elif llm_type == "gemini":
                # Google Gemini - we check which models are available
                if not self.settings.google_api_key:
                    raise ValueError("GOOGLE_API_KEY is required but not set")
                
                model_name = getattr(self.settings, "gemini_model", "gemini-2.0-flash")
                
                # Query available models from Google's API
                available = self._list_gemini_models()
                
                # If configured model isn't available, pick a fallback
                if available and model_name not in available:
                    # Preference order for fallback models (flash models have better free tier limits)
                    preferred = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-pro-latest"]
                    selected = next((m for m in preferred if m in available), available[0])
                    logger.warning(f"Model '{model_name}' not available; using '{selected}'")
                    model_name = selected
                
                logger.info(f"Initializing Gemini LLM with model: {model_name}")
                
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.settings.google_api_key,
                    temperature=self.settings.agent_temperature,
                    max_output_tokens=self.settings.agent_max_tokens
                )
                self._current_gemini_model = model_name
                return llm
                
            elif llm_type == "claude":
                # Anthropic Claude models (Opus, Sonnet, Haiku)
                if not self.settings.anthropic_api_key or not self.settings.anthropic_api_key.strip():
                    raise ValueError(
                        "ANTHROPIC_API_KEY is required but not set. "
                        "Please set ANTHROPIC_API_KEY in your .env file or environment variables."
                    )
                
                model_name = getattr(self.settings, "claude_model", "claude-3-5-sonnet-20241022")
                
                # Model-specific max token limits for Claude
                # https://docs.anthropic.com/en/docs/about-claude/models
                claude_max_tokens_limits = {
                    "claude-3-5-sonnet-20241022": 8192,
                    "claude-3-opus-20240229": 4096,
                    "claude-3-5-haiku-20241022": 8192,
                    "claude-3-haiku-20240307": 4096,
                }
                
                # Cap max_tokens to model's limit
                model_max = claude_max_tokens_limits.get(model_name, 4096)
                requested_tokens = self.settings.agent_max_tokens
                actual_max_tokens = min(requested_tokens, model_max)
                
                if requested_tokens > model_max:
                    logger.warning(
                        f"⚠ Requested max_tokens ({requested_tokens}) exceeds model limit ({model_max}). "
                        f"Using {actual_max_tokens} instead."
                    )
                
                logger.info(f"Initializing Claude LLM with model: {model_name}")
                logger.info(f"Max tokens: {actual_max_tokens} (model limit: {model_max})")
                logger.info(f"Using Anthropic API key: {'*' * 10}{self.settings.anthropic_api_key[-4:] if len(self.settings.anthropic_api_key) > 4 else '****'}")
                
                llm = ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=self.settings.anthropic_api_key,
                    temperature=self.settings.agent_temperature,
                    max_tokens=actual_max_tokens
                )
                
                logger.info(f"✓ Claude LLM ({model_name}) initialized successfully")
                return llm
                
            elif llm_type == "openai":
                # OpenAI models (GPT-4, GPT-4o, GPT-4o-mini, GPT-5, GPT-5-mini, GPT-5-nano, GPT-5.1)
                if not self.settings.openai_api_key or not self.settings.openai_api_key.strip():
                    raise ValueError(
                        "OPENAI_API_KEY is required but not set. "
                        "Please set OPENAI_API_KEY in your .env file or environment variables."
                    )
                
                # Use provided model_name or fallback to settings
                if model_name is None:
                    model_name = getattr(self.settings, "openai_model", "gpt-4o-mini")
                
                # Model-specific max token limits for OpenAI (output tokens)
                # https://platform.openai.com/docs/models
                # These are the maximum output tokens each model can generate per request
                openai_max_tokens_limits = {
                    # GPT-4 series
                    "gpt-4": 8192,                    # GPT-4: 8,192 output tokens
                    "gpt-4-turbo": 4096,               # GPT-4 Turbo: 4,096 output tokens
                    "gpt-4o": 16384,                   # GPT-4o: 16,384 output tokens
                    "gpt-4o-2024-08-06": 16384,       # GPT-4o (specific version): 16,384 output tokens
                    "gpt-4o-mini": 16384,              # GPT-4o-mini: 16,384 output tokens
                    
                    # GPT-5 series (future models - limits based on expected capabilities)
                    "gpt-5-nano": 8192,                # GPT-5-nano: 8,192 output tokens (for simple tasks)
                    "gpt-5-mini": 16384,               # GPT-5-mini: 16,384 output tokens (for average complexity)
                    "gpt-5": 65536,                    # GPT-5: 65,536 output tokens (for complex tasks)
                    "gpt-5.1": 128000,                 # GPT-5.1: 128,000 output tokens (for very complex tasks)
                }
                
                # Cap max_tokens to model's limit
                model_max = openai_max_tokens_limits.get(model_name, 4096)
                requested_tokens = self.settings.agent_max_tokens
                actual_max_tokens = min(requested_tokens, model_max)

                # Cache key: model + token cap + temperature (sufficient for current usage)
                cache_key = (model_name, actual_max_tokens, float(self.settings.agent_temperature))
                cached = getattr(self, "_openai_llm_cache", {}).get(cache_key)
                if cached is not None:
                    logger.info(f"✓ Using cached OpenAI LLM ({model_name})")
                    return cached
                
                if requested_tokens > model_max:
                    logger.warning(
                        f"⚠ Requested max_tokens ({requested_tokens}) exceeds model limit ({model_max}). "
                        f"Using {actual_max_tokens} instead."
                    )
                
                logger.info(f"Initializing OpenAI LLM with model: {model_name}")
                logger.info(f"Max tokens: {actual_max_tokens} (model limit: {model_max})")
                logger.info(f"Using OpenAI API key: {'*' * 10}{self.settings.openai_api_key[-4:] if len(self.settings.openai_api_key) > 4 else '****'}")
                
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=self.settings.openai_api_key,
                    temperature=self.settings.agent_temperature,
                    max_tokens=actual_max_tokens
                )

                # Save to cache
                try:
                    self._openai_llm_cache[cache_key] = llm
                except Exception:
                    pass
                
                logger.info(f"✓ OpenAI LLM ({model_name}) initialized successfully")
                return llm
                
            else:
                raise ValueError(f"Unknown LLM type: {llm_type}. Supported types: gemini, deepseek, claude, openai")
                
        except Exception as e:
            logger.error(f"Error initializing {llm_type} LLM: {e}")
            raise
    
    def _list_gemini_models(self) -> List[str]:
        """
        Query Google's API to get a list of available Gemini models.
        
        This is useful because:
        1. Model availability varies by region and API version
        2. We can automatically fall back to available models
        3. Prevents errors from trying to use unavailable models
        
        Returns:
            List[str]: Names of available chat-capable Gemini models
            
        Note:
            Returns empty list if the API call fails (network error, etc.)
        """
        try:
            import requests
            
            api_key = getattr(self.settings, "google_api_key", None)
            if not api_key:
                return []
            
            # Google's model listing endpoint
            url = "https://generativelanguage.googleapis.com/v1beta/models"
            response = requests.get(url, params={"key": api_key}, timeout=10)
            response.raise_for_status()
            
            models = []
            for model in response.json().get("models", []):
                name = model.get("name", "").split("/")[-1]
                
                # Filter to chat-capable models only
                # Exclude embedding, image, TTS, and vision-only models
                if name.startswith("gemini") and not any(
                    x in name.lower() for x in ["embedding", "image", "tts", "vision"]
                ):
                    # Check if model supports text generation
                    methods = model.get("supportedGenerationMethods", [])
                    if not methods or "generateContent" in methods:
                        models.append(name)
                        
            return models
            
        except Exception as e:
            logger.debug(f"Could not list Gemini models: {e}")
            return []
    
    # ==========================================================================
    # TOOL CREATION
    # ==========================================================================
    
    def _create_tools(self) -> List[BaseTool]:
        """
        Create the tools that the agent can use.
        
        Tools are the "hands" of the agent - they allow it to interact with
        external systems like AutoCAD, Excel files, etc.
        
        Each tool has:
        - name: Unique identifier the LLM uses to call it
        - description: Tells the LLM what the tool does and when to use it
        - func: The Python function that executes the tool
        - args_schema: Pydantic model defining the expected arguments
        
        Returns:
            List[BaseTool]: List of configured tools
        """
        # ==================================================================
        # AUTOCAD TOOLS
        # ==================================================================
        # These tools interface with AutoCAD via COM API
        
        # --- Tool 1: Open Drawing ---
        class AutoCADOpenInput(BaseModel):
            """Input schema for opening AutoCAD drawings."""
            file_path: str = Field(
                description="Path to DWG or DXF file to open in AutoCAD"
            )
        
        def autocad_open(file_path: str) -> str:
            """
            Open a CAD drawing file in AutoCAD.
            
            This must be called before any other AutoCAD operations.
            Establishes connection to AutoCAD if not already connected.
            Falls back to ezdxf if AutoCAD is not available.
            STRICT: Survey plan templates are always opened read-only and never written.
            """
            from pathlib import Path
            # Try AutoCAD first
            if not self.autocad.is_connected:
                connected = self.autocad.connect()
                if not connected:
                    # Fall back to ezdxf
                    if self.dxf_fallback.is_available:
                        logger.warning("AutoCAD not available, using ezdxf fallback")
                        result = self.dxf_fallback.open_drawing(file_path)
                        if result.get("success"):
                            return str(result)
                        return f"Fallback failed: {result.get('error', 'Unknown error')}"
                    return "AutoCAD not available and ezdxf fallback not installed. Please open AutoCAD manually first."
            # STRICT: Never open survey plan template for writing (read-only to avoid corruption).
            resolved = str(Path(file_path).resolve()) if file_path else ""
            if self._is_protected_template_path(resolved):
                return str(self.autocad.open_drawing(file_path, read_only=True))
            return str(self.autocad.open_drawing(file_path))

        # ==================================================================
        # INTERNET SEARCH TOOL (Permissioned)
        # ==================================================================
        class InternetSearchInput(BaseModel):
            """Input schema for internet search (requires user permission)."""
            query: str = Field(description="What to search for on the internet")
            max_results: int = Field(5, description="Max results to return (1-10)")

        def internet_search(query: str, max_results: int = 5) -> str:
            """
            Search the internet for up-to-date information (permission required).

            Returns structured JSON with results (title, url, snippet).
            
            NOTE: If internet results were already injected into the system prompt by the router,
            this tool will return a note indicating that search was already performed.
            """
            # Check if internet was already searched by router (shouldn't happen, but safety check)
            if getattr(self, "_internet_already_searched_this_query", False):
                return json.dumps(
                    {
                        "success": True,
                        "source": "internet",
                        "note": "Internet search was already performed by the router. Results are in the system prompt context above.",
                        "message": "Please use the internet search results already provided in the conversation context.",
                    },
                    indent=2,
                )
            
            # Clamp
            try:
                max_results_i = int(max_results)
            except Exception:
                max_results_i = 5
            max_results_i = max(1, min(10, max_results_i))

            # CRITICAL: Permission should be checked BEFORE tool is called (by router).
            # If tool is called without permission, return a proper ToolMessage-compatible response
            # (not a permission request string, which breaks the tool call sequence).
            if not getattr(self, "_internet_permission_granted", False):
                # Return a proper error response (not a permission request string)
                return json.dumps(
                    {
                        "success": False,
                        "source": "internet",
                        "error": "Permission not granted. Internet search requires explicit user permission.",
                        "note": "Use the internet_search tool only after user has granted permission via interactive mode.",
                    },
                    indent=2,
                )

            result = _internet_search(query)
            if not result.get("success"):
                return json.dumps(
                    {
                        "success": False,
                        "source": "internet",
                        "provider": result.get("providers_attempted", ["duckduckgo_instant_answer", "wikipedia"]),
                        "query": query,
                        "error": result.get("error", "Unknown error"),
                        "note": "INTERNET_SOURCED",
                    },
                    indent=2,
                )

            # Trim results
            results = (result.get("results", []) or [])[:max_results_i]
            payload = {
                "success": True,
                "source": "internet",
                "providers": result.get("providers", ["duckduckgo_instant_answer", "wikipedia"]),
                "query": result.get("query", query),
                "results": results,
                "note": "INTERNET_SOURCED",
            }
            return json.dumps(payload, indent=2)
        
        # --- Tool 2: Calculate Area ---
        class AutoCADAreaInput(BaseModel):
            """Input schema for area calculation."""
            layer: Optional[str] = Field(
                None, 
                description="Filter by layer name (optional)"
            )
            color: Optional[str] = Field(
                None, 
                description="Filter by color name (e.g., 'red', 'blue')"
            )
        
        def autocad_calculate_area(
            layer: Optional[str] = None, 
            color: Optional[str] = None
        ) -> str:
            """
            Calculate the total area of closed shapes in the drawing.
            
            Uses AutoCAD's native area calculation for maximum precision.
            Can filter by layer or color to calculate specific regions.
            Falls back to ezdxf if AutoCAD is not available.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.calculate_area(layer=layer, color=color)
            if result.get("success") or "error" not in str(result) or "not connected" not in str(result).lower():
                return str(result)
            
            # Fallback to ezdxf
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                logger.info("Using ezdxf fallback for area calculation")
                return str(self.dxf_fallback.calculate_area(layer=layer, color=color))
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 3: Search Text ---
        class AutoCADTextSearchInput(BaseModel):
            """Input schema for text search."""
            pattern: str = Field(
                description="Text pattern to search for (supports regex)"
            )
            case_sensitive: bool = Field(
                False, 
                description="Whether search should be case-sensitive"
            )
        
        def autocad_search_text(
            pattern: str, 
            case_sensitive: bool = False
        ) -> str:
            """
            Search for text entities matching a pattern.
            
            Useful for finding owner names, survey titles, and annotations.
            Supports regular expressions for flexible matching.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.search_text(pattern, case_sensitive)
            if isinstance(result, dict) and result.get("success"):
                return str(result)
            
            # Fallback: get all text and filter
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                import re
                fallback_result = self.dxf_fallback.get_all_text()
                if fallback_result.get("success"):
                    flags = 0 if case_sensitive else re.IGNORECASE
                    matches = []
                    for text in fallback_result.get("texts", []):
                        content = text.get("content", "")
                        try:
                            if re.search(pattern, content, flags):
                                matches.append(text)
                        except re.error:
                            if pattern.lower() in content.lower():
                                matches.append(text)
                    return str({"success": True, "matches_found": len(matches), "matches": matches})
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 4: Get All Text ---
        class AutoCADGetTextInput(BaseModel):
            """Input schema for getting all text (no parameters needed)."""
            pass
        
        def autocad_get_all_text() -> str:
            """
            Extract all text content from the current drawing.
            
            Returns all TEXT and MTEXT entities with their content,
            layer, color, and position information.
            """
            # Try AutoCAD first - let the method handle connection checking
            result = self.autocad.get_all_text()
            if result.get("success") or (isinstance(result, dict) and "error" not in result):
                return str(result)
            
            # Fallback to ezdxf
            if self.dxf_fallback.is_available and self.dxf_fallback.doc:
                return str(self.dxf_fallback.get_all_text())
            
            return str(result)  # Return the error from AutoCAD method
        
        # --- Tool 5: Get All Entities (AI-Driven) ---
        class AutoCADGetAllEntitiesInput(BaseModel):
            """Input schema for getting all entities (no parameters needed)."""
            pass
        
        def autocad_get_all_entities() -> str:
            """
            Get ALL entities from the drawing with complete properties.
            
            [AI-DRIVEN TOOL] This returns raw data for agent reasoning.
            Returns all entities with their full properties (type, layer, color,
            coordinates, area, closed status, etc.) without any filtering.
            
            The agent should reason about which entities match criteria:
            - Identify red entities by checking color property (color_code=1 or color='red')
            - Identify closed shapes by checking 'closed' property and entity type
            - Filter by layer, coordinates, or other properties using reasoning
            
            Use this for AI-driven extraction where the agent reasons about
            what to extract rather than using hardcoded filters.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_all_entities())
        
        # --- Tool 6: Get Entities Summary (AI-Driven) ---
        class AutoCADEntitiesSummaryInput(BaseModel):
            """Input schema for getting entities summary (no parameters needed)."""
            pass
        
        def autocad_get_entities_summary() -> str:
            """
            Get a summary of all entities for quick analysis.
            
            [AI-DRIVEN TOOL] Returns lightweight summary (counts by type, color, layer)
            that the agent can use to reason about the drawing structure before
            calling get_all_entities() for detailed extraction.
            
            Use this first to understand what's in the drawing, then reason
            about which entities to extract in detail.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entities_summary())
        
        # --- Tool 7: Get Entity by Handle (AI-Driven) ---
        class AutoCADGetEntityInput(BaseModel):
            """Input schema for getting entity by handle."""
            handle: str = Field(
                description="Entity handle (unique identifier from get_all_entities)"
            )
        
        def autocad_get_entity(handle: str) -> str:
            """
            Get detailed information about a specific entity by its handle.
            
            [AI-DRIVEN TOOL] Use this to get detailed properties of an entity
            that was identified from get_all_entities(). Handles are unique
            identifiers for entities in AutoCAD.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entity_by_handle(handle))
        
        # --- Tool 8: Calculate Entity Area (AI-Driven) ---
        class AutoCADEntityAreaInput(BaseModel):
            """Input schema for calculating area of a specific entity."""
            handle: str = Field(
                description="Entity handle (unique identifier from get_all_entities)"
            )
        
        def autocad_calculate_entity_area(handle: str) -> str:
            """
            Calculate the area of a specific entity by handle.
            
            [AI-DRIVEN TOOL] Use this to calculate area of entities identified
            from get_all_entities(). The agent should first identify which
            entities are closed shapes (check 'closed' property and entity type),
            then call this method for each one.
            
            Workflow:
            1. Call get_all_entities() to get all entities
            2. Reason about which entities are closed shapes (LWPOLYLINE with closed=True, CIRCLE, HATCH)
            3. For each matching entity, call this method with its handle
            4. Sum the areas if needed
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.calculate_entity_area(handle))
        
        # --- Tool 9: Get Entities (Backward Compatibility) ---
        class AutoCADEntitiesInput(BaseModel):
            """Input schema for entity retrieval."""
            entity_type: Optional[str] = Field(
                None, 
                description="Entity type: LINE, POLYLINE, CIRCLE, TEXT, etc."
            )
            layer: Optional[str] = Field(
                None, 
                description="Filter by layer name"
            )
            color: Optional[str] = Field(
                None, 
                description="Filter by color name"
            )
        
        def autocad_get_entities(
            entity_type: Optional[str] = None,
            layer: Optional[str] = None,
            color: Optional[str] = None
        ) -> str:
            """
            Retrieve entities from the drawing with optional filters.
            
            [BACKWARD COMPATIBILITY] This uses get_all_entities() internally.
            For AI-driven extraction, prefer using get_all_entities() directly
            and let the agent reason about filtering.
            """
            # Let the method handle connection/document checking internally
            return str(self.autocad.get_entities_by_type(entity_type, layer, color))
        
        # --- Tool 6: Get Drawing Info ---
        class AutoCADInfoInput(BaseModel):
            """Input schema for drawing info (no parameters needed)."""
            pass
        
        def autocad_get_info() -> str:
            """
            Get metadata about the current drawing.
            
            Returns drawing name, path, units, layers, and entity counts.
            """
            return str(self.autocad.get_drawing_info())
        
        # --- Tool 7: Execute Command ---
        class AutoCADCommandInput(BaseModel):
            """Input schema for raw AutoCAD commands."""
            command: str = Field(
                description="AutoCAD command to execute (e.g., 'ZOOM E', 'REGEN')"
            )
        
        def autocad_execute_command(command: str) -> str:
            """
            Execute a raw AutoCAD command.
            
            Use for operations not covered by other specialized tools.
            Commands are sent directly to AutoCAD's command line.
            """
            return str(self.autocad.execute_command(command))
        
        # ==================================================================
        # EXCEL TOOLS
        # ==================================================================

        class ExcelInspectInput(BaseModel):
            """Input for inspecting Excel workbook structure."""
            file_path: str = Field(description="Path to the Excel file (.xlsx, .xls, .xlsm)")

        def excel_inspect_workbook(file_path: str) -> str:
            """
            Inspect an Excel workbook: list all sheet names and each sheet's column headers.
            MANDATORY FIRST STEP when the user refers to named data (e.g. 'Pre-fill', 'Post-fill',
            'coordinates', 'X/Y/Z'): call this to discover actual sheet and column names, then
            reason to map user terms to real names (e.g. 'Pre Fill' -> 'Pre_fill_2024', X/Y/Z -> EASTING, NORTHING, RL).
            Only after this deep research should you call ArcGIS/Excel tools or report that data was not found.
            """
            import json
            out = self.excel_processor.inspect_workbook(file_path)
            return json.dumps(out, indent=2)

        class ExcelInput(BaseModel):
            """Input schema for Excel processing."""
            file_path: str = Field(description="Path to Excel file")
            x_column: Optional[str] = Field(
                None,
                description="Column name containing X coordinates"
            )
            y_column: Optional[str] = Field(
                None,
                description="Column name containing Y coordinates"
            )

        def excel_processor_func(**kwargs) -> str:
            """
            Extract coordinate data from Excel spreadsheets.

            Supports .xlsx and .xls formats. Can automatically detect
            coordinate columns or use specified column names.
            """
            return str(self.excel_processor.process_file(**kwargs))

        class CsvToExcelInput(BaseModel):
            """Input schema for CSV to Excel conversion."""
            csv_path: str = Field(description="Path to the CSV file to convert")
            output_excel_path: Optional[str] = Field(
                None,
                description="Path for the output .xlsx file. If omitted, same folder as CSV, same name with .xlsx extension."
            )

        def csv_to_excel(csv_path: str, output_excel_path: Optional[str] = None) -> str:
            """
            Convert a CSV file to an Excel workbook (.xlsx).

            CRITICAL for workflows that start with CSV: ArcGIS ExcelToTable and many coordinate/import
            tools accept only .xlsx/.xls. If the user provides a .csv (e.g. Coords.csv), call this
            FIRST to create Coords.xlsx in the same folder, then use the Excel path for
            excel_coordinate_convert, arcgis_import_xy_points_from_excel, etc.
            """
            return str(self.excel_processor.csv_to_excel(csv_path, output_excel_path))
        
        # ==================================================================
        # DOCUMENT PROCESSING TOOLS (Atomic, AI-driven extraction)
        # ==================================================================
        
        class DocumentMetadataInput(BaseModel):
            """Input schema for document metadata."""
            file_path: str = Field(description="Path to PDF or Word document")
        
        class DocumentTextInput(BaseModel):
            """Input schema for text extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            preserve_structure: bool = Field(
                default=True,
                description="Preserve paragraph breaks and document structure"
            )
        
        class DocumentTablesInput(BaseModel):
            """Input schema for table extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            page_number: Optional[int] = Field(
                default=None,
                description="Specific page number (for PDF) or None for all pages"
            )
        
        class DocumentSectionInput(BaseModel):
            """Input schema for section extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            section_title: Optional[str] = Field(
                default=None,
                description="Title of section to extract (e.g., 'Signature', 'Summary')"
            )
            start_keyword: Optional[str] = Field(
                default=None,
                description="Keyword marking start of section"
            )
            end_keyword: Optional[str] = Field(
                default=None,
                description="Keyword marking end of section"
            )
        
        class DocumentSearchInput(BaseModel):
            """Input schema for text search."""
            file_path: str = Field(description="Path to PDF or Word document")
            pattern: str = Field(description="Text pattern to search for")
            case_sensitive: bool = Field(default=False, description="Case-sensitive search")
            use_regex: bool = Field(default=False, description="Use regex pattern")
            context_lines: int = Field(default=2, description="Lines of context around matches")
        
        class DocumentStructuredDataInput(BaseModel):
            """Input schema for structured data extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            data_types: Optional[List[str]] = Field(
                default=None,
                description="Data types to extract: dates, names, numbers, emails, coordinates, depths, or 'all'"
            )
        
        class DocumentCreateInput(BaseModel):
            """Input schema for document creation."""
            file_path: str = Field(description="Full path where the Word document should be saved (.docx)")
            content: str = Field(description="Text content to write to the document")
            title: Optional[str] = Field(
                default=None,
                description="Optional title for the document"
            )
        
        class DocumentCreateStructuredInput(BaseModel):
            """Input schema for structured document creation."""
            file_path: str = Field(description="Full path where the Word document should be saved (.docx)")
            title: str = Field(description="Document title")
            sections: List[Dict] = Field(
                description="List of sections, each with heading, level, content, and optional table"
            )
            metadata: Optional[Dict] = Field(
                default=None,
                description="Optional metadata (author, date, etc.)"
            )
        
        class DocumentReadInput(BaseModel):
            """Input schema for reading existing documents."""
            file_path: str = Field(description="Path to existing Word document to read")
        
        class DocumentUpdateInput(BaseModel):
            """Input schema for updating existing documents."""
            file_path: str = Field(description="Path to document to update (will be created if doesn't exist)")
            new_content: str = Field(description="New content to write")
            title: Optional[str] = Field(default=None, description="Optional title")
            overwrite: bool = Field(default=True, description="If True, replace entire document; if False, append")
        
        class DocumentStructureInput(BaseModel):
            """Input schema for document structure analysis."""
            file_path: str = Field(description="Path to PDF or Word document")
        
        class DocumentResourceEstimationInput(BaseModel):
            """Input schema for resource estimation."""
            file_path: str = Field(description="Path to PDF or Word document")
            model_name: Optional[str] = Field(
                default=None,
                description="LLM model name for cost estimation (defaults to current model)"
            )
        
        class DocumentExtractSectionsInput(BaseModel):
            """Input schema for keyword-based section extraction."""
            file_path: str = Field(description="Path to PDF or Word document")
            keywords: List[str] = Field(description="List of keywords to search for (e.g., ['Location', 'Personnel', 'Contractor'])")
            context_lines: int = Field(default=5, description="Number of lines of context around matches")
        
        def document_get_metadata(file_path: str) -> str:
            """
            Get document metadata (file info, structure, table presence).
            
            Use this first to understand document structure before extraction.
            Returns: file type, page count, table presence, etc.
            """
            result = self.document_processor.get_document_metadata(file_path)
            return str(result)
        
        def document_get_text(file_path: str, preserve_structure: bool = True) -> str:
            """
            Extract all text content from the document.
            
            Returns raw text that you can analyze. Use this for general text extraction
            or when you need to search through the entire document content.
            """
            # Always preflight cost/size first so the user is informed before expensive processing
            model_for_cost = None
            if self.settings.primary_llm == "openai":
                model_for_cost = self._current_openai_model or getattr(self.settings, "openai_model_mini", "gpt-5-mini")
            elif self.settings.primary_llm == "gemini":
                model_for_cost = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
            else:
                model_for_cost = getattr(self.settings, "openai_model_mini", "gpt-5-mini")

            est = self.document_processor.get_resource_estimation(file_path, model_for_cost)
            if est.get("success") and est.get("warnings"):
                return str({
                    "success": False,
                    "blocked": True,
                    "reason": "Document appears large; full text extraction is blocked by default to prevent timeouts/cost blowups.",
                    "resource_estimation": est,
                    "next_steps": [
                        "Use document_extract_sections_by_keywords(file_path, keywords=[...]) to pull only relevant parts",
                        "If you explicitly want full extraction, confirm and then use document_get_text_force(file_path)"
                    ],
                    "question": "Proceed with FULL text extraction via document_get_text_force? (yes/no)"
                })

            result = self.document_processor.get_full_text(file_path, preserve_structure)
            return str({
                "resource_estimation": est,
                "result": result
            })

        def document_get_text_force(file_path: str, preserve_structure: bool = True) -> str:
            """
            Force full text extraction from the document, even if large.
            
            Use ONLY after the user explicitly confirms they want full extraction.
            """
            result = self.document_processor.get_full_text_force(file_path, preserve_structure)
            return str(result)
        
        def document_get_tables(file_path: str, page_number: Optional[int] = None) -> str:
            """
            Extract all tables from the document.
            
            Returns structured table data with headers and rows. Use this when
            the document contains tabular data (e.g., feature lists, measurements).
            """
            result = self.document_processor.get_tables(file_path, page_number)
            return str(result)
        
        def document_get_section(
            file_path: str,
            section_title: Optional[str] = None,
            start_keyword: Optional[str] = None,
            end_keyword: Optional[str] = None
        ) -> str:
            """
            Extract text from a specific section of the document.
            
            Use this to extract specific sections like signatures, summaries,
            or findings. Provide either section_title or start_keyword/end_keyword.
            """
            result = self.document_processor.get_text_by_section(
                file_path, start_keyword, end_keyword, section_title
            )
            return str(result)
        
        def document_search_text(
            file_path: str,
            pattern: str,
            case_sensitive: bool = False,
            use_regex: bool = False,
            context_lines: int = 2
        ) -> str:
            """
            Search for specific text patterns in the document.
            
            Use this to find specific information like dates, names, or keywords.
            Supports regex patterns for flexible searching.
            """
            result = self.document_processor.search_text(
                file_path, pattern, case_sensitive, use_regex, context_lines
            )
            return str(result)
        
        def document_extract_structured_data(
            file_path: str,
            data_types: Optional[List[str]] = None
        ) -> str:
            """
            Extract structured data types (dates, names, numbers, emails, etc.).
            
            Use this to quickly extract common data types. Specify data_types as
            a list: ['dates', 'names', 'numbers', 'emails', 'coordinates', 'depths'] or ['all'].
            """
            result = self.document_processor.extract_structured_data(file_path, data_types)
            return str(result)
        
        def document_create_word(
            file_path: str,
            content: str,
            title: Optional[str] = None
        ) -> str:
            """
            Create a new Word document (.docx) with the specified content.
            
            CRITICAL CONTEXT RULES:
            - Use ONLY the data you JUST extracted and displayed in THIS conversation
            - When user says "save the summary", use the summary you JUST showed them above in YOUR CURRENT RESPONSE
            - NEVER use data from previous conversations or different documents
            - The 'content' parameter should be the text you displayed in your IMMEDIATELY PRECEDING response
            - CONTEXT ISOLATION: Each conversation is independent - do NOT mix data from different documents
            
            Use this when the user explicitly asks to save, export, or create a document file.
            When user says "save as [filename]" or "export as [filename]", use this tool immediately.
            
            Args:
                file_path: Full path where document should be saved (include .docx extension)
                content: Text content to write - MUST be from CURRENT conversation response, not previous ones
                title: Optional document title
                
            Returns: Success message with file path
            """
            result = self.document_processor.create_word_document(file_path, content, title)
            return str(result)
        
        def document_create_structured_word(
            file_path: str,
            title: str,
            sections: List[Dict],
            metadata: Optional[Dict] = None
        ) -> str:
            """
            Create a Word document from structured data with sections and tables.
            
            Use this for creating professional reports with multiple sections, headings, and tables.
            More advanced than document_create_word - use when you have structured data.
            
            Args:
                file_path: Full path where document should be saved
                title: Document title
                sections: List of section dicts with 'heading', 'level', 'content', optional 'table'
                metadata: Optional metadata dict
                
            Returns: Success message with file path
            """
            result = self.document_processor.create_word_document_from_structure(
                file_path, title, sections, metadata
            )
            return str(result)
        
        def document_read_word(file_path: str) -> str:
            """
            Read an existing Word document to get its content.
            
            CRITICAL: Use this PROACTIVELY when:
            - User asks to modify/update/shorten a document you JUST created in this conversation
            - User says "the same document" or "the same file" - use the file path from your previous response
            - You need to read a document before updating it
            
            REMEMBER: If you just created a file and mentioned its path in your response, use that path here.
            Don't ask the user for the path - you already know it from the conversation context.
            
            Args:
                file_path: Path to existing Word document (remember this from when you created it)
                
            Returns: Document content (text, paragraphs, tables)
            """
            result = self.document_processor.read_existing_word_document(file_path)
            return str(result)
        
        def document_update_word(
            file_path: str,
            new_content: str,
            title: Optional[str] = None,
            overwrite: bool = True
        ) -> str:
            """
            Update an existing Word document with new content.
            
            CRITICAL: Use this PROACTIVELY when:
            - User asks to modify/update/shorten a document you JUST created
            - User says "save in the same document" or "update the same file"
            - After reading a document with document_read_word, use this to save the modified content
            
            REMEMBER: Use the same file_path you used when creating the document - it's in your previous response.
            Don't ask the user for the path - you already know it from the conversation context.
            
            Args:
                file_path: Path to document (remember this from when you created it, or from document_read_word)
                new_content: New content to write (use actual extracted/condensed data, not placeholders)
                title: Optional title
                overwrite: If True, replace entire document; if False, append
                
            Returns: Success message with file path
            """
            result = self.document_processor.update_word_document(
                file_path, new_content, title, overwrite
            )
            return str(result)
        
        def document_get_structure(file_path: str) -> str:
            """
            Analyze document structure (headings, sections, organization).
            
            CRITICAL FOR LARGE DOCUMENTS (>100 pages or >50K words):
            - ALWAYS call this FIRST for large documents before extracting text
            - Use this to understand document organization
            - Then use document_extract_sections_by_keywords to extract only relevant sections
            - This prevents processing the entire document and saves tokens/costs
            
            Returns: Document structure with sections, headings, and outline.
            """
            result = self.document_processor.get_document_structure(file_path)
            return str(result)
        
        def document_get_resource_estimation(
            file_path: str,
            model_name: Optional[str] = None
        ) -> str:
            """
            Estimate resource requirements and costs for processing a document.
            
            CRITICAL FOR LARGE DOCUMENTS:
            - ALWAYS call this FIRST when user requests processing a document
            - Shows file size, estimated tokens, and cost
            - Provides warnings and recommendations for large documents
            - Use this to inform the user before processing
            
            Args:
                file_path: Path to document
                model_name: Optional model name (defaults to current model)
            
            Returns: Resource estimation with warnings, costs, and recommendations.
            """
            if not model_name:
                if self.settings.primary_llm == "openai":
                    model_name = self._current_openai_model or getattr(self.settings, "openai_model_mini", "gpt-5-mini")
                elif self.settings.primary_llm == "gemini":
                    model_name = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
                else:
                    model_name = getattr(self.settings, "openai_model_mini", "gpt-5-mini")
            
            result = self.document_processor.get_resource_estimation(file_path, model_name)
            return str(result)
        
        def document_extract_sections_by_keywords(
            file_path: str,
            keywords: List[str],
            context_lines: int = 5
        ) -> str:
            """
            Extract only document sections matching specific keywords.
            
            CRITICAL FOR LARGE DOCUMENTS:
            - Use this instead of document_get_text for documents >100 pages
            - Extracts only relevant sections, saving tokens and costs
            - Much faster than processing entire document
            
            Workflow for large documents:
            1. document_get_resource_estimation → Check if document is large
            2. document_get_structure → Understand document organization
            3. document_extract_sections_by_keywords → Extract only relevant sections
            4. Process extracted sections instead of full document
            
            Args:
                file_path: Path to document
                keywords: List of keywords to search for (e.g., ["Location", "Personnel", "Contractor"])
                context_lines: Number of lines of context around matches (default: 5)
            
            Returns: Extracted sections matching keywords with context.
            """
            result = self.document_processor.extract_sections_by_keywords(
                file_path, keywords, context_lines
            )
            return str(result)
        
        # ==================================================================
        # COORDINATE CONVERSION TOOL
        # ==================================================================
        
        class CoordConvertInput(BaseModel):
            """Input schema for coordinate conversion."""
            x: float = Field(description="X coordinate (Easting or Longitude)")
            y: float = Field(description="Y coordinate (Northing or Latitude)")
            source_crs: str = Field(
                "WGS84", 
                description="Source coordinate reference system"
            )
            target_crs: str = Field(
                "WGS84", 
                description="Target coordinate reference system"
            )
            use_geographic_calculator: bool = Field(
                False,
                description="If True, attempt to use Geographic Calculator COM interface. "
                           "If False (default), use pyproj. Only set to True if user explicitly "
                           "requests Geographic Calculator in their query."
            )
        
        def coordinate_convert(**kwargs) -> str:
            """
            Convert coordinates between different reference systems using pyproj (default).
            
            Uses pyproj by default for fast, reliable coordinate conversions. Supports:
            - WGS84, UTM zones, State Plane
            - Nigerian coordinate systems (Minna NTM, etc.)
            - Many other coordinate reference systems from EPSG database
            
            If user explicitly requests "Geographic Calculator" in their query, will attempt
            to use Geographic Calculator COM interface, but always falls back to pyproj if COM
            is unavailable or fails.
            
            The system automatically resolves informal CRS names (e.g., "MINNA_NTM_MIDBELT")
            to proper EPSG codes for accurate conversions.
            """
            # Detect if user explicitly requested Geographic Calculator
            # This is a simple heuristic - in practice, the LLM should detect this from context
            use_geocalc = kwargs.pop('use_geographic_calculator', False)
            return str(self.blue_marble.convert_coordinate(use_geographic_calculator=use_geocalc, **kwargs))

        class CoordConvertAutoInput(BaseModel):
            """
            Input schema for survey-aware coordinate conversion from free-form text.

            This is designed for real-world survey formats (DMS/DM/decimal degrees, hemisphere letters,
            and projected coordinates with E/N or X/Y labels).
            """
            text: str = Field(
                description=(
                    "Free-form text containing coordinates (and optionally CRS names/codes). "
                    "Examples: '6°12\\'30.5\"N 3°21\\'10\"E', 'E 512345.12 N 6789012.34', "
                    "'lat 6 12 30 N lon 3 21 10 E', '6.1234, 3.4567'."
                )
            )
            source_crs: Optional[str] = Field(
                default=None,
                description=(
                    "Optional source CRS name/code. If omitted, SurvyAI will try to infer from text "
                    "(e.g., 'WGS84', 'UTM Zone 32N', 'EPSG:4326')."
                ),
            )
            target_crs: Optional[str] = Field(
                default=None,
                description=(
                    "Optional target CRS name/code. If omitted, SurvyAI will try to infer from text. "
                    "If still unknown, defaults to WGS84."
                ),
            )
            use_geographic_calculator: bool = Field(
                default=False,
                description="If True, attempt to use Geographic Calculator COM interface; otherwise use pyproj.",
            )

        def coordinate_convert_auto(
            text: str,
            source_crs: Optional[str] = None,
            target_crs: Optional[str] = None,
            use_geographic_calculator: bool = False,
        ) -> str:
            """
            Survey-aware coordinate conversion from free-form text.

            What it does:
            - Auto-detect coordinates inside text (supports DMS like 6°12'30\"N, hemisphere letters, E/N labels)
            - Normalizes geodetic coordinates to decimal degrees
            - Attempts to infer CRS names/codes from the text ("from ... to ...", EPSG/WKID, UTM zone hints)
            - Converts each detected coordinate pair using the existing BlueMarbleConverter (pyproj by default)
            """
            try:
                parsed = infer_crs_from_text(text)
                src = (source_crs or parsed.get("source_crs") or "WGS84").strip()
                dst = (target_crs or parsed.get("target_crs") or "WGS84").strip()

                points = extract_points(text, max_points=20)
                if not points:
                    return (
                        "✗ No coordinates detected. Provide coordinates in one of these formats:\n"
                        "- Geodetic (DMS/DM): 6°12'30.5\"N 3°21'10\"E\n"
                        "- Geodetic (decimal): 6.1234N, 3.4567E or 6.1234, 3.4567\n"
                        "- Projected: E 512345.12 N 6789012.34 or X=512345.12 Y=6789012.34\n"
                        "Also include CRS hints like 'from WGS84 to UTM Zone 32N' or 'EPSG:4326 to EPSG:32632'."
                    )

                results = []
                for p in points:
                    r = self.blue_marble.convert_coordinate(
                        x=p.x,
                        y=p.y,
                        source_crs=src,
                        target_crs=dst,
                        use_geographic_calculator=use_geographic_calculator,
                    )
                    results.append(
                        {
                            "parsed": {
                                "x": p.x,
                                "y": p.y,
                                "kind": p.kind,
                                "source_text": p.source_text,
                                "notes": p.notes,
                            },
                            "conversion": r,
                        }
                    )

                payload = {
                    "success": True,
                    "source_crs": src,
                    "target_crs": dst,
                    "count": len(results),
                    "results": results,
                }
                return json.dumps(payload, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"✗ Auto coordinate conversion failed: {e}"
        
        # Excel batch coordinate conversion tool
        class ExcelCoordConvertInput(BaseModel):
            """Input schema for Excel coordinate conversion."""
            excel_path: str = Field(description="Path to Excel file containing coordinates")
            x_column: str = Field(
                default="X",
                description="Name of column containing X/Easting coordinates"
            )
            y_column: str = Field(
                default="Y",
                description="Name of column containing Y/Northing coordinates"
            )
            source_crs: str = Field(
                default="WGS84",
                description="Source coordinate reference system"
            )
            target_crs: str = Field(
                default="WGS84",
                description="Target coordinate reference system"
            )
            source_zone: Optional[int] = Field(
                default=None,
                description="Source UTM zone (if applicable)"
            )
            target_zone: Optional[int] = Field(
                default=None,
                description="Target UTM zone (if applicable)"
            )
            output_path: Optional[str] = Field(
                default=None,
                description=(
                    "Output file path. "
                    "CRITICAL: If not specified, automatically saves in same folder as excel_path "
                    "with '_converted' suffix. This ensures outputs are created alongside input files."
                )
            )
            sheet_name: Optional[str] = Field(
                default=None,
                description="Sheet name to process (default: first sheet)"
            )
            use_geographic_calculator: bool = Field(
                default=False,
                description="If True, attempt to use Geographic Calculator COM interface. "
                           "If False (default), use pyproj. Only set to True if user explicitly "
                           "requests Geographic Calculator in their query."
            )
        
        def excel_coordinate_convert(**kwargs) -> str:
            """
            Convert coordinates in an Excel file using pyproj (default).
            
            Reads coordinates from specified columns, converts them using pyproj (default),
            and saves results to a new Excel file with converted coordinates added as new columns.
            
            If user explicitly requests "Geographic Calculator" in their query, will attempt
            to use Geographic Calculator COM interface, but always falls back to pyproj if COM
            is unavailable or fails.
            
            The system automatically resolves informal CRS names (e.g., "MINNA_NTM_MIDBELT")
            to proper EPSG codes for accurate conversions.
            """
            try:
                # Extract use_geographic_calculator if provided, default to False
                use_geocalc = kwargs.pop('use_geographic_calculator', False)
                result = self.blue_marble.convert_excel_file(use_geographic_calculator=use_geocalc, **kwargs)
                return (
                    f"✓ Excel coordinate conversion completed successfully!\n\n"
                    f"Input file: {result['input_file']}\n"
                    f"Output file: {result['output_file']}\n"
                    f"Total coordinates: {result['total_coordinates']}\n"
                    f"Successful conversions: {result['successful_conversions']}\n"
                    f"Failed conversions: {result['failed_conversions']}\n"
                    f"Conversion method: {result['method']}\n"
                    f"Source CRS: {result['source_crs']}\n"
                    f"Target CRS: {result['target_crs']}\n"
                    f"Output columns: {', '.join(result['output_columns'])}"
                )
            except Exception as e:
                return f"✗ Excel coordinate conversion failed: {str(e)}"

        class ExcelConvertAndAreaInput(BaseModel):
            """Convert coordinates in an Excel file (including DMS) and compute area automatically."""
            excel_path: str = Field(description="Path to Excel file containing boundary coordinates")
            source_crs: str = Field(default="WGS84", description="Source CRS (e.g., WGS84, EPSG:4326)")
            target_crs: str = Field(default="WGS84", description="Target CRS (e.g., Minna Nigerian NTM MidBelt)")
            x_column: str = Field(default="Long.", description="Longitude/Easting column name (tabs/whitespace tolerated)")
            y_column: str = Field(default="Lat.", description="Latitude/Northing column name (tabs/whitespace tolerated)")
            output_filename: str = Field(
                default="converted1.xlsx",
                description="Output filename to save in the same folder as the input Excel file",
            )
            area_on: Literal["best", "source", "target"] = Field(
                default="best",
                description="Where to compute area: best (auto), source (original coords), or target (converted coords).",
            )

        def excel_convert_and_area(
            excel_path: str,
            source_crs: str = "WGS84",
            target_crs: str = "WGS84",
            x_column: str = "Long.",
            y_column: str = "Lat.",
            output_filename: str = "converted1.xlsx",
            area_on: str = "best",
        ) -> str:
            """
            One-shot workflow:
            - Reads Excel
            - Parses DMS/DM/decimal values in Lat/Long columns
            - Converts source->target CRS
            - Saves output to same folder as input
            - Computes area using best available method
            """
            try:
                from pathlib import Path
                import pandas as pd
                from utils.coordinate_parsing import parse_angle

                inp = Path(excel_path)
                out_path = (inp.parent / Path(output_filename).name).resolve()

                # Run conversion using the improved converter (handles DMS + messy headers)
                conv = self.blue_marble.convert_excel_file(
                    excel_path=str(inp),
                    x_column=x_column,
                    y_column=y_column,
                    source_crs=source_crs,
                    target_crs=target_crs,
                    output_path=str(out_path),
                    use_geographic_calculator=False,
                    # Default to a clean output: original columns + one converted X/Y pair
                    output_schema="clean",
                )

                # Compute area:
                # - for WGS84-like/geodetic source: geodesic (best)
                # - for projected: planar (best)
                # Read back the saved file so we can access converted columns reliably
                df_out = pd.read_excel(out_path, sheet_name=0)
                x_conv_col = conv.get("output_x_column") or "X"
                y_conv_col = conv.get("output_y_column") or "Y"

                # Build point lists (assume row order is vertex order; close polygon automatically)
                def _series_to_points(df, xcol, ycol, allow_dms: bool = False):
                    pts = []
                    for _, r in df.iterrows():
                        try:
                            rx = r.get(xcol)
                            ry = r.get(ycol)
                            if allow_dms:
                                xv = parse_angle(str(rx)) if rx is not None else None
                                yv = parse_angle(str(ry)) if ry is not None else None
                                if xv is None or yv is None:
                                    continue
                                xv = float(xv)
                                yv = float(yv)
                            else:
                                xv = float(rx)
                                yv = float(ry)
                            pts.append((xv, yv))
                        except Exception:
                            continue
                    return pts

                # Source columns may be DMS strings (lat/long), so allow parsing there.
                src_pts = _series_to_points(df_out, conv["x_column"], conv["y_column"], allow_dms=True)
                tgt_pts = _series_to_points(df_out, x_conv_col, y_conv_col, allow_dms=False)

                if not src_pts:
                    return f"✗ No usable source points found for area calculation in {out_path}"

                # Decide which set to use
                area_choice = (area_on or "best").lower()
                if area_choice == "target" and tgt_pts:
                    area_res = best_area(tgt_pts, crs_hint=target_crs)
                elif area_choice == "source":
                    area_res = best_area(src_pts, crs_hint=source_crs)
                else:
                    # best: prefer geodesic if source looks like lon/lat, else planar on target if present
                    try_src = best_area(src_pts, crs_hint=source_crs)
                    if try_src.method.startswith("geodesic"):
                        area_res = try_src
                    elif tgt_pts:
                        area_res = best_area(tgt_pts, crs_hint=target_crs)
                    else:
                        area_res = try_src

                summary = {
                    "success": True,
                    "input_file": str(inp),
                    "output_file": str(out_path),
                    "source_crs": source_crs,
                    "target_crs": target_crs,
                    "converted_points": conv.get("total_coordinates"),
                    "area": {
                        "method": area_res.method,
                        "m2": area_res.area_m2,
                        "hectares": area_res.hectares,
                        "ft2": area_res.ft2,
                        "acres": area_res.acres,
                        "perimeter_m": area_res.perimeter_m,
                        "computed_on": area_choice,
                    },
                }
                return json.dumps(summary, indent=2, ensure_ascii=False)
            except Exception as e:
                return f"✗ Excel convert+area failed: {e}"
        
        # ==================================================================
        # BUILD TOOL LIST
        # ==================================================================
        
        tools = [
            # AutoCAD tools
            StructuredTool(
                name="autocad_open_drawing",
                description=(
                    "Open a DWG or DXF file in AutoCAD. "
                    "MUST be called before any other AutoCAD operations. "
                    "STRICT: Survey plan templates (e.g. survey_plan_template2.dwg) are always opened read-only and must never be written to avoid corruption."
                ),
                func=autocad_open,
                args_schema=AutoCADOpenInput
            ),
            # Internet search (permissioned)
            StructuredTool(
                name="internet_search",
                description=(
                    "Search the internet for up-to-date information. "
                    "MUST ask user permission before using. "
                    "All results returned are internet-sourced and must be clearly highlighted in your response."
                ),
                func=internet_search,
                args_schema=InternetSearchInput
            ),
            StructuredTool(
                name="autocad_calculate_area",
                description=(
                    "Calculate area of closed shapes (polylines, circles, hatches). "
                    "Use color='red' for boundaries 'verged in red'. "
                    "Returns area in sq meters, sq feet, hectares, and acres."
                ),
                func=autocad_calculate_area,
                args_schema=AutoCADAreaInput
            ),
            StructuredTool(
                name="autocad_search_text",
                description=(
                    "Search for text matching a pattern. "
                    "Use patterns like 'property of' or 'plan shewing' "
                    "to find owner names and survey titles."
                ),
                func=autocad_search_text,
                args_schema=AutoCADTextSearchInput
            ),
            StructuredTool(
                name="autocad_get_all_text",
                description=(
                    "Get all text content from the drawing. "
                    "Use to find titles, names, annotations, and labels."
                ),
                func=autocad_get_all_text,
                args_schema=AutoCADGetTextInput
            ),
            # AI-Driven atomic tools (preferred for reasoning)
            StructuredTool(
                name="autocad_get_all_entities",
                description=(
                    "[AI-DRIVEN] Get ALL entities with complete properties for agent reasoning. "
                    "Returns raw data (type, layer, color, coordinates, area, closed status) "
                    "without filtering. Agent should reason about which entities match criteria. "
                    "Use this for AI-driven extraction where the agent reasons about what to extract."
                ),
                func=autocad_get_all_entities,
                args_schema=AutoCADGetAllEntitiesInput
            ),
            StructuredTool(
                name="autocad_get_entities_summary",
                description=(
                    "[AI-DRIVEN] Get lightweight summary (counts by type, color, layer) "
                    "to understand drawing structure before detailed extraction."
                ),
                func=autocad_get_entities_summary,
                args_schema=AutoCADEntitiesSummaryInput
            ),
            StructuredTool(
                name="autocad_get_entity",
                description=(
                    "[AI-DRIVEN] Get detailed properties of a specific entity by handle. "
                    "Use after identifying entities from get_all_entities()."
                ),
                func=autocad_get_entity,
                args_schema=AutoCADGetEntityInput
            ),
            StructuredTool(
                name="autocad_calculate_entity_area",
                description=(
                    "[AI-DRIVEN] Calculate area of a specific entity by handle. "
                    "Agent should first identify closed shapes from get_all_entities(), "
                    "then call this for each one. Workflow: get_all_entities() -> "
                    "reason about closed shapes -> calculate_entity_area(handle) for each."
                ),
                func=autocad_calculate_entity_area,
                args_schema=AutoCADEntityAreaInput
            ),
            # Backward compatibility tools
            StructuredTool(
                name="autocad_get_entities",
                description=(
                    "[BACKWARD COMPATIBILITY] Get entities with filters. "
                    "For AI-driven extraction, prefer autocad_get_all_entities() "
                    "and let the agent reason about filtering."
                ),
                func=autocad_get_entities,
                args_schema=AutoCADEntitiesInput
            ),
            StructuredTool(
                name="autocad_get_info",
                description="Get drawing metadata: units, layers, entity counts.",
                func=autocad_get_info,
                args_schema=AutoCADInfoInput
            ),
            StructuredTool(
                name="autocad_command",
                description=(
                    "Execute a raw AutoCAD command. "
                    "Use for operations not covered by other tools."
                ),
                func=autocad_execute_command,
                args_schema=AutoCADCommandInput
            ),
            
            # Other tools
            StructuredTool(
                name="excel_inspect_workbook",
                description=(
                    "Inspect Excel workbook structure: list all sheet names and each sheet's column headers. "
                    "MANDATORY FIRST when the user refers to named sheets or data (e.g. 'Pre-fill', 'Post-fill', "
                    "'coordinates', 'X/Y/Z'): discover actual names, then reason to map user intent to real sheet/column names. "
                    "Only report errors or ask for names after this deep research."
                ),
                func=excel_inspect_workbook,
                args_schema=ExcelInspectInput
            ),
            StructuredTool(
                name="excel_processor",
                description="Extract coordinate data from Excel files (.xlsx, .xls).",
                func=excel_processor_func,
                args_schema=ExcelInput
            ),
            StructuredTool(
                name="csv_to_excel",
                description=(
                    "Convert a CSV file to an Excel file (.xlsx). Use this FIRST when the user provides a .csv "
                    "but downstream steps need Excel (e.g. coordinate conversion, ArcGIS import). "
                    "Output defaults to same folder as CSV with .xlsx extension. "
                    "Parameters: csv_path (required), output_excel_path (optional)."
                ),
                func=csv_to_excel,
                args_schema=CsvToExcelInput,
            ),
            # Document processing tools (atomic, AI-driven)
            StructuredTool(
                name="document_get_metadata",
                description=(
                    "Get document metadata (file info, structure, table presence). "
                    "Use this first to understand document structure before extraction."
                ),
                func=document_get_metadata,
                args_schema=DocumentMetadataInput
            ),
            StructuredTool(
                name="document_get_text",
                description=(
                    "⚠️⚠️⚠️ DO NOT USE FOR LARGE DOCUMENTS! ⚠️⚠️⚠️\n"
                    "This tool AUTOMATICALLY FAILS for documents >50 pages or >25K words.\n"
                    "MANDATORY: Call document_get_resource_estimation(file_path) FIRST for ALL documents.\n"
                    "If document is large, use document_extract_sections_by_keywords() instead.\n"
                    "This tool ONLY works for small documents (<50 pages, <25K words).\n"
                    "If you truly need full extraction for a large document, use document_get_text_force after user confirmation."
                ),
                func=document_get_text,
                args_schema=DocumentTextInput
            ),
            StructuredTool(
                name="document_get_text_force",
                description=(
                    "FORCE full document text extraction, even for large documents. "
                    "Use ONLY after the user explicitly confirms they want full extraction "
                    "(it can be slow/expensive and may exceed limits in multi-step workflows)."
                ),
                func=document_get_text_force,
                args_schema=DocumentTextInput
            ),
            StructuredTool(
                name="document_get_tables",
                description=(
                    "Extract all tables from the document as structured data. "
                    "Use when document contains tabular data (feature lists, measurements, etc.)."
                ),
                func=document_get_tables,
                args_schema=DocumentTablesInput
            ),
            StructuredTool(
                name="document_get_section",
                description=(
                    "Extract text from a specific section (e.g., 'Signature', 'Summary', 'Findings'). "
                    "Use to extract specific document sections by title or keywords."
                ),
                func=document_get_section,
                args_schema=DocumentSectionInput
            ),
            StructuredTool(
                name="document_search_text",
                description=(
                    "Search for specific text patterns in the document. "
                    "Use to find dates, names, keywords, or any specific information. Supports regex."
                ),
                func=document_search_text,
                args_schema=DocumentSearchInput
            ),
            StructuredTool(
                name="document_extract_structured_data",
                description=(
                    "Extract structured data types: dates, names, numbers, emails, coordinates, depths. "
                    "Use to quickly extract common data types. Specify data_types as list or use 'all'."
                ),
                func=document_extract_structured_data,
                args_schema=DocumentStructuredDataInput
            ),
            StructuredTool(
                name="document_create_word",
                description=(
                    "Create a new Word document (.docx) with specified content. "
                    "CRITICAL: Use this IMMEDIATELY when user asks to 'save', 'export', or 'create' a document file. "
                    "When user says 'save as [filename]' or 'export as [filename]', use this tool right away - do not ask for confirmation again. "
                    "CONTEXT RULE: Use ONLY the data you JUST extracted and displayed in YOUR CURRENT RESPONSE - NEVER use data from previous conversations or different documents. "
                    "The 'content' parameter must be the summary/data you just showed the user above."
                ),
                func=document_create_word,
                args_schema=DocumentCreateInput
            ),
            StructuredTool(
                name="document_create_structured_word",
                description=(
                    "Create a Word document from structured data with sections, headings, and tables. "
                    "Use for professional reports. When user asks to save/export a structured document, use this."
                ),
                func=document_create_structured_word,
                args_schema=DocumentCreateStructuredInput
            ),
            StructuredTool(
                name="document_read_word",
                description=(
                    "Read an existing Word document to get its content. "
                    "CRITICAL: Use this PROACTIVELY when user asks to modify/update a document you JUST created. "
                    "Remember the file path from your previous response where you said 'saved as [path]'. "
                    "Don't ask user for the path - you already know it from conversation context. "
                    "Workflow: User says 'make it shorter' → Use this tool with the path you just used → Condense → document_update_word"
                ),
                func=document_read_word,
                args_schema=DocumentReadInput
            ),
            StructuredTool(
                name="document_update_word",
                description=(
                    "Update an existing Word document with new content. "
                    "CRITICAL: Use this PROACTIVELY when user asks to modify/update/shorten a document you JUST created. "
                    "Use the same file_path from when you created it (it's in your previous response). "
                    "Don't ask user for path or confirmation - use the path you already know. "
                    "Workflow: After document_read_word → Process content → Use this tool with same path and new content"
                ),
                func=document_update_word,
                args_schema=DocumentUpdateInput
            ),
            StructuredTool(
                name="document_get_structure",
                description=(
                    "Analyze document structure (headings, sections, organization). "
                    "CRITICAL FOR LARGE DOCUMENTS: ALWAYS call this FIRST for documents >100 pages or >50K words. "
                    "Use this to understand document organization before extracting text. "
                    "Then use document_extract_sections_by_keywords to extract only relevant sections."
                ),
                func=document_get_structure,
                args_schema=DocumentStructureInput
            ),
            StructuredTool(
                name="document_get_resource_estimation",
                description=(
                    "Estimate resource requirements and costs for processing a document. "
                    "CRITICAL FOR LARGE DOCUMENTS: ALWAYS call this FIRST when user requests processing a document. "
                    "Shows file size, estimated tokens, cost, warnings, and recommendations. "
                    "Use this to inform the user before processing large documents."
                ),
                func=document_get_resource_estimation,
                args_schema=DocumentResourceEstimationInput
            ),
            StructuredTool(
                name="document_extract_sections_by_keywords",
                description=(
                    "Extract only document sections matching specific keywords. "
                    "CRITICAL FOR LARGE DOCUMENTS: Use this instead of document_get_text for documents >50 pages. "
                    "Extracts only relevant sections, saving tokens and costs. Much faster than processing entire document. "
                    "Workflow: document_get_resource_estimation → document_get_structure → this tool → process extracted sections."
                ),
                func=document_extract_sections_by_keywords,
                args_schema=DocumentExtractSectionsInput
            ),
            StructuredTool(
                name="coordinate_converter",
                description=(
                    "Convert coordinates between reference systems using pyproj (default). "
                    "Supports WGS84, UTM zones, State Plane, Nigerian coordinate systems (Minna NTM, etc.), "
                    "and many other CRS from EPSG database. "
                    "Uses pyproj by default for fast, reliable conversions. "
                    "Only use Geographic Calculator COM if user explicitly requests it in their query "
                    "(set use_geographic_calculator=True). Always falls back to pyproj if COM is unavailable."
                ),
                func=coordinate_convert,
                args_schema=CoordConvertInput
            ),
            StructuredTool(
                name="coordinate_converter_auto",
                description=(
                    "Survey-aware coordinate conversion from FREE-FORM TEXT. "
                    "Use this when coordinates may contain degrees/minutes/seconds (DMS), hemisphere letters "
                    "(N/S/E/W), or projected labels (Easting/Northing, X/Y). "
                    "Automatically extracts coordinates, converts DMS/DM to decimal degrees, infers CRS hints "
                    "from text (e.g., 'from WGS84 to UTM Zone 32N', 'EPSG:4326 to EPSG:32632'), then converts "
                    "using pyproj by default (or Geographic Calculator COM if explicitly requested)."
                ),
                func=coordinate_convert_auto,
                args_schema=CoordConvertAutoInput,
            ),
            StructuredTool(
                name="excel_coordinate_converter",
                description=(
                    "Convert coordinates in an Excel file using pyproj (default). "
                    "Reads coordinates from specified columns, converts them using pyproj, "
                    "and saves results with converted coordinates as new columns. "
                    "Supports specialized coordinate systems like Nigerian NTM (Minna MidBelt, etc.). "
                    "Uses pyproj by default for fast, reliable conversions. "
                    "Only use Geographic Calculator COM if user explicitly requests it in their query "
                    "(set use_geographic_calculator=True). Always falls back to pyproj if COM is unavailable. "
                    "Parameters: excel_path (required), x_column (default: 'X'), y_column (default: 'Y'), "
                    "source_crs, target_crs, source_zone, target_zone (optional), output_path (optional), "
                    "sheet_name (optional), use_geographic_calculator (default: False)."
                ),
                func=excel_coordinate_convert,
                args_schema=ExcelCoordConvertInput
            ),
            StructuredTool(
                name="excel_convert_and_area",
                description=(
                    "One-shot Excel workflow for surveyors: reads an Excel file, auto-handles DMS Lat/Long (text), "
                    "converts coordinates from source CRS to target CRS, saves output in the same folder, and computes "
                    "area using the best available method (geodesic if geodetic/WGS84-like, otherwise planar). "
                    "Use this instead of asking the user to rename headers or pre-clean DMS."
                ),
                func=excel_convert_and_area,
                args_schema=ExcelConvertAndAreaInput,
            ),
        ]
        
        # ==================================================================
        # GEOGRAPHIC CALCULATOR CLI TOOLS
        # ==================================================================
        # Always register the check tool so the agent can verify availability
        # The execute tool is only registered if CLI is available
        
        # --- Tool: Check Geographic Calculator Availability ---
        # This tool should ALWAYS be available, even if CLI is not installed
        class GeoCalcCheckInput(BaseModel):
            """Input schema for checking Geographic Calculator availability (no parameters)."""
            pass
        
        def geocalc_check_availability() -> str:
            """
            Check if Geographic Calculator CLI is available on the system.
            
            Returns information about the installation status, version, and executable path.
            This is a read-only check that doesn't require user permission.
            """
            # Re-scan on every check so the agent can pick up installs/changes immediately.
            try:
                self.geocalc_cli.refresh()
            except Exception:
                pass

            # Check COM interface availability (optional method)
            # NOTE: We do NOT auto-connect here; we only report current state.
            com_available = self.blue_marble.is_available
            com_status = "✓ Available" if com_available else "✗ Not available"
            
            if not self.geocalc_cli.is_available:
                # Check if GUI is installed (even if CLI isn't found)
                from pathlib import Path
                gui_path = Path(r"C:\Program Files\Blue Marble Geo\Geographic Calculator\Geographic Calculator.exe")
                if gui_path.exists():
                    return (
                        f"Geographic Calculator Status:\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"GUI Installation: ✓ Found at {gui_path}\n"
                        f"COM Interface: {com_status}\n"
                        f"CLI Component: ✗ Not found\n\n"
                        f"RECOMMENDATION: Use pyproj by default; COM is optional\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"SurvyAI uses pyproj as the main conversion method (fast, reliable, no external COM).\n"
                        f"If your Geographic Calculator automation is installed/registered, COM can be used\n"
                        f"when explicitly requested for specialized workflows.\n\n"
                        f"Available Tools:\n"
                        f"  • coordinate_converter - Convert individual coordinates (pyproj default)\n"
                        f"  • coordinate_converter_auto - Parse DMS/text coords then convert (pyproj default)\n"
                        f"  • excel_coordinate_converter - Batch convert Excel (pyproj default, auto DMS)\n"
                        f"  • excel_convert_and_area - Convert Excel + compute area (best method)\n\n"
                        f"To use COM interface (optional):\n"
                        f"  - Re-run the Geographic Calculator installer and enable Automation/COM if available,\n"
                        f"    then request COM explicitly with use_geographic_calculator=True.\n\n"
                        f"Note: CLI component (GeographicCalculatorCMD.exe) is optional and\n"
                        f"not required for coordinate conversions. pyproj is sufficient."
                    )
                
                return (
                    "Geographic Calculator CLI (GeographicCalculatorCMD.exe) is not found on this system.\n\n"
                    "The system performed a thorough search but could not locate the executable.\n"
                    "Checks include: environment variable (GEOGRAPHIC_CALCULATOR_CMD_PATH), settings/.env, "
                    "Windows 'App Paths' registry, vendor registry keys, Uninstall registry entries, PATH/where.exe, "
                    "common install locations, and a targeted Program Files scan.\n\n"
                    "If Geographic Calculator is installed in a non-standard location, set the full path via "
                    "GEOGRAPHIC_CALCULATOR_CMD_PATH (or add its folder to PATH) and re-run this check."
                )
            
            version = self.geocalc_cli.get_version()
            cmd_path = self.geocalc_cli.cmd_path
            
            info = {
                "available": True,
                "status": "Geographic Calculator CLI is installed and ready to use",
                "executable_path": str(cmd_path) if cmd_path else "Unknown",
                "version": version if version else "Unknown (could not determine version)",
            }
            
            # Format the response nicely
            response = (
                "✓ Geographic Calculator CLI is available on your system!\n\n"
                f"Executable Path: {info['executable_path']}\n"
                f"Version: {info['version']}\n\n"
                "You can now execute Geographic Calculator jobs using the "
                "geographic_calculator_execute_job tool. Job files (.gpj, .gpp, .gpw) must be "
                "created in Geographic Calculator GUI first."
            )
            return response
        
        # Always add the check tool
        tools.append(
            StructuredTool(
                name="geographic_calculator_check",
                description=(
                    "MANDATORY TOOL FOR GEOGRAPHIC CALCULATOR QUERIES: "
                    "Check if Geographic Calculator CLI (GeographicCalculatorCMD.exe) is installed on the system. "
                    "Returns installation status, version information, and executable path if available. "
                    "CRITICAL INSTRUCTIONS: "
                    "1. When user asks about Geographic Calculator availability, installation status, or file path, "
                    "   you MUST call this tool IMMEDIATELY - do NOT ask for permission, do NOT provide menus, do NOT ask for more information. "
                    "2. This is a read-only check that does not access or modify files - no permission needed. "
                    "3. If user says 'yes' or grants permission after you ask, IMMEDIATELY call this tool - do NOT provide menus, do NOT ask for more info, do NOT give unrelated responses. "
                    "4. Just call this tool right away when asked about Geographic Calculator availability."
                ),
                func=geocalc_check_availability,
                args_schema=GeoCalcCheckInput
            )
        )
        
        if self.geocalc_cli.is_available:
            # --- Tool: Execute Geographic Calculator Job ---
            class GeoCalcJobInput(BaseModel):
                """Input schema for executing Geographic Calculator jobs."""
                job_path: str = Field(
                    description="Path to Geographic Calculator job/project/workspace file (.gpj, .gpp, or .gpw)"
                )
                close_after_done: bool = Field(
                    True,
                    description="Close Geographic Calculator after execution completes"
                )
                continue_after_error: bool = Field(
                    False,
                    description="Continue processing even if errors occur"
                )
            
            def geocalc_execute_job(
                job_path: str,
                close_after_done: bool = True,
                continue_after_error: bool = False
            ) -> str:
                """
                Execute a Geographic Calculator job, project, or workspace file.
                
                This tool runs pre-configured Geographic Calculator operations
                such as coordinate conversions, transformations, and batch processing.
                The job file must be created in Geographic Calculator GUI first.
                """
                result = self.geocalc_cli.execute_job(
                    job_path=job_path,
                    close_after_done=close_after_done,
                    continue_after_error=continue_after_error
                )
                return str(result)
            
            # Add Geographic Calculator execute job tool (only if CLI is available)
            tools.append(
                StructuredTool(
                    name="geographic_calculator_execute_job",
                    description=(
                        "Execute a Geographic Calculator job, project, or workspace file via command-line. "
                        "Use this when the user wants to run pre-configured Geographic Calculator "
                        "operations like coordinate conversions, transformations, or batch processing. "
                        "The job file (.gpj, .gpp, or .gpw) must be created in Geographic Calculator GUI first. "
                        "This tool executes GeographicCalculatorCMD.exe with the specified job file. "
                        "Parameters: job_path (required), close_after_done (default: true), continue_after_error (default: false)."
                    ),
                    func=geocalc_execute_job,
                    args_schema=GeoCalcJobInput
                )
            )
            
            logger.info("✓ Geographic Calculator execute job tool registered")
        else:
            logger.info("⚠ Geographic Calculator CLI not available - execute job tool not registered (check tool is always available)")
        
        logger.info("✓ Geographic Calculator check tool registered (always available)")
        
        # ==================================================================
        # VECTOR STORE TOOLS (if available)
        # ==================================================================
        
        if self.vector_store is not None:
            # --- Vector Search Tool ---
            class VectorSearchInput(BaseModel):
                """Input schema for vector search."""
                query: str = Field(
                    description="Search query text for semantic search"
                )
                collection: str = Field(
                    default="documents",
                    description="Collection to search: documents, drawings, or coordinates"
                )
                top_k: int = Field(
                    default=5,
                    description="Number of results to return (1-20)"
                )
            
            def vector_search(query: str, collection: str = "documents", top_k: int = 5) -> str:
                """
                Search for relevant documents using semantic similarity.
                
                Finds documents, drawings, or coordinates that are semantically
                similar to the query text.
                """
                try:
                    results = self.vector_store.search(
                        query=query,
                        collection=collection,
                        top_k=min(top_k, 20)
                    )
                    if not results:
                        return "No matching documents found."
                    
                    output = []
                    for i, r in enumerate(results, 1):
                        score = r.get('score', 0)
                        content = r.get('content', '')[:500]  # Truncate long content
                        metadata = r.get('metadata', {})
                        output.append(
                            f"{i}. [Score: {score:.2f}]\n"
                            f"   Content: {content}\n"
                            f"   Metadata: {metadata}"
                        )
                    return "\n\n".join(output)
                except Exception as e:
                    return f"Search error: {str(e)}"
            
            # --- Vector Store Document Tool ---
            class VectorStoreInput(BaseModel):
                """Input schema for storing documents."""
                content: str = Field(
                    description="Text content to store in the vector database"
                )
                collection: str = Field(
                    default="documents",
                    description="Collection to store in: documents, drawings, or coordinates"
                )
                source: str = Field(
                    default="",
                    description="Source of the content (file name, etc.)"
                )
            
            def vector_store_document(
                content: str, 
                collection: str = "documents",
                source: str = ""
            ) -> str:
                """
                Store a document in the vector database for later retrieval.
                
                Useful for saving extracted text, survey data, or any content
                that should be searchable later.
                """
                try:
                    metadata = {"source": source} if source else {}
                    doc_id = self.vector_store.add_text(
                        text=content,
                        metadata=metadata,
                        collection=collection
                    )
                    return f"✓ Document stored successfully (ID: {doc_id})"
                except Exception as e:
                    return f"Storage error: {str(e)}"
            
            # --- Vector Store Stats Tool ---
            class VectorStoreStatsInput(BaseModel):
                """Input schema for vector store stats (no parameters)."""
                pass
            
            def vector_store_stats() -> str:
                """Get statistics about the vector database."""
                try:
                    stats = self.vector_store.get_stats()
                    return (
                        f"Vector Store Statistics:\n"
                        f"  - Provider: {stats['embedding_provider']}\n"
                        f"  - Model: {stats['embedding_model']}\n"
                        f"  - Dimension: {stats['embedding_dimension']}\n"
                        f"  - Total Documents: {stats['total_documents']}\n"
                        f"  - Collections:\n" +
                        "\n".join(f"    - {k}: {v} docs" for k, v in stats['collections'].items())
                    )
                except Exception as e:
                    return f"Error getting stats: {str(e)}"
            
            # Add vector store tools
            tools.extend([
                StructuredTool(
                    name="semantic_search",
                    description=(
                        "Search for relevant documents using semantic similarity. "
                        "Use this to find previously stored information about surveys, "
                        "drawings, coordinates, or any text content. "
                        "Collections: documents, drawings, coordinates."
                    ),
                    func=vector_search,
                    args_schema=VectorSearchInput
                ),
                StructuredTool(
                    name="store_document",
                    description=(
                        "Store text content in the vector database for later retrieval. "
                        "Use this to save extracted text, survey data, or important information "
                        "that should be searchable in future queries."
                    ),
                    func=vector_store_document,
                    args_schema=VectorStoreInput
                ),
                StructuredTool(
                    name="vector_store_stats",
                    description="Get statistics about the vector database (document counts, etc.).",
                    func=vector_store_stats,
                    args_schema=VectorStoreStatsInput
                ),
            ])
            
            logger.info(f"✓ Added {3} vector store tools")
        
        # ==================================================================
        # ARCGIS PRO TOOLS
        # ==================================================================
        
        # Add ArcGIS tools if ArcGIS Pro is installed
        if self.arcgis_processor.is_installed or self.arcgis_processor.is_available:

            # --- Tool: filesystem stat/exists (verification gate) ---
            class FilesystemStatInput(BaseModel):
                """Input schema for checking file existence and size."""
                paths: List[str] = Field(description="List of file/folder paths to check")

            def filesystem_stat(paths: List[str]) -> str:
                from pathlib import Path
                out = []
                for p in paths or []:
                    try:
                        pp = Path(p)
                        exists = pp.exists()
                        info = {"path": str(pp), "exists": exists}
                        if exists:
                            try:
                                st = pp.stat()
                                info.update({"is_dir": pp.is_dir(), "size_bytes": st.st_size, "mtime": st.st_mtime})
                            except Exception:
                                pass
                        out.append(info)
                    except Exception as e:
                        out.append({"path": str(p), "exists": False, "error": str(e)})
                return json.dumps({"items": out}, indent=2, ensure_ascii=False)
            
            # --- Tool: Launch ArcGIS Pro ---
            class ArcGISLaunchInput(BaseModel):
                """Input schema for launching ArcGIS Pro."""
                pass
            
            def arcgis_launch() -> str:
                """
                Launch ArcGIS Pro application.
                Opens ArcGIS Pro on the user's computer.
                """
                result = self.arcgis_processor.launch_arcgis_pro()
                return str(result)
            
            # --- Tool: Create Project ---
            class ArcGISCreateProjectInput(BaseModel):
                """Input schema for creating an ArcGIS Pro project."""
                project_name: str = Field(
                    description="Name of the project (without .aprx extension)"
                )
                project_path: Optional[str] = Field(
                    None,
                    description="Directory to save the project (default: Documents/ArcGIS/Projects)"
                )
                coordinate_system: Optional[str] = Field(
                    None,
                    description=(
                        "Coordinate system for the project. "
                        "Examples: 'WGS84', 'UTM Zone 32N', 'EPSG:4326', '32632'"
                    )
                )
                template: str = Field(
                    default="MAP",
                    description="Project template: MAP, CATALOG, GLOBAL_SCENE, or LOCAL_SCENE"
                )
            
            def arcgis_create_project(
                project_name: str,
                project_path: Optional[str] = None,
                coordinate_system: Optional[str] = None,
                template: str = "MAP"
            ) -> str:
                """
                Create a new ArcGIS Pro project with specified settings.
                
                Can set the coordinate system for the map (e.g., UTM Zone 32N).
                """
                # Use default coordinate system if not specified
                if not coordinate_system:
                    coordinate_system = getattr(
                        self.settings, 
                        'arcgis_default_coordinate_system', 
                        None
                    )
                
                # Use default project path if not specified
                if not project_path:
                    project_path = getattr(
                        self.settings,
                        'arcgis_default_project_path',
                        None
                    ) or None
                
                result = self.arcgis_processor.create_project(
                    project_name=project_name,
                    project_path=project_path,
                    coordinate_system=coordinate_system,
                    template=template
                )
                return str(result)
            
            # --- Tool: Open Project ---
            class ArcGISOpenProjectInput(BaseModel):
                """Input schema for opening an ArcGIS Pro project."""
                project_path: str = Field(
                    description="Path to the .aprx project file"
                )
            
            def arcgis_open_project(project_path: str) -> str:
                """
                Open an existing ArcGIS Pro project.
                """
                result = self.arcgis_processor.open_project(project_path)
                return str(result)
            
            # --- Tool: Set Coordinate System ---
            class ArcGISSetCRSInput(BaseModel):
                """Input schema for setting coordinate system."""
                coordinate_system: str = Field(
                    description=(
                        "Coordinate system to set. "
                        "Examples: 'WGS84', 'UTM Zone 32N', 'EPSG:4326', '32632', 'British National Grid'"
                    )
                )
                map_name: Optional[str] = Field(
                    None,
                    description="Name of the map to modify (default: first map)"
                )
            
            def arcgis_set_coordinate_system(
                coordinate_system: str,
                map_name: Optional[str] = None
            ) -> str:
                """
                Set the coordinate system for a map in the current project.
                """
                result = self.arcgis_processor.set_map_coordinate_system(
                    coordinate_system=coordinate_system,
                    map_name=map_name
                )
                return str(result)
            
            # --- Tool: Get Project Info ---
            class ArcGISProjectInfoInput(BaseModel):
                """Input schema for getting project info."""
                pass
            
            def arcgis_get_project_info() -> str:
                """
                Get information about the current ArcGIS Pro project.
                Returns map names, coordinate systems, and other metadata.
                """
                result = self.arcgis_processor.get_project_info()
                return str(result)
            
            # --- Tool: List Coordinate Systems ---
            class ArcGISListCRSInput(BaseModel):
                """Input schema for listing coordinate systems."""
                filter_text: Optional[str] = Field(
                    None,
                    description="Optional text to filter coordinate systems (e.g., 'UTM', 'WGS')"
                )
            
            def arcgis_list_coordinate_systems(filter_text: Optional[str] = None) -> str:
                """
                List available coordinate systems with their WKID codes.
                Useful for finding the correct coordinate system name.
                """
                result = self.arcgis_processor.list_coordinate_systems(filter_text)
                return str(result)

            # --- Tool: Import XY Points from Excel ---
            class ArcGISImportXYPointsInput(BaseModel):
                """Input schema for importing XY points from an Excel file."""
                project_path: str = Field(
                    description="Path to the .aprx project file to import points into"
                )
                excel_path: str = Field(
                    description="Path to the Excel file containing coordinates"
                )
                x_field: str = Field(
                    description="Name of the X/Easting field in the Excel file (e.g., 'Long._converted' or 'Easting')"
                )
                y_field: str = Field(
                    description="Name of the Y/Northing field in the Excel file (e.g., 'Lat._converted' or 'Northing')"
                )
                coordinate_system: str = Field(
                    description=(
                        "Coordinate system of the X/Y fields. "
                        "Examples: 'Minna / Nigeria Mid Belt', 'UTM Zone 32N', 'EPSG:26392'"
                    )
                )
                sheet_name: Optional[str] = Field(
                    None,
                    description="Optional Excel sheet name (default: first sheet)"
                )
                layer_name: Optional[str] = Field(
                    None,
                    description="Optional output layer/feature class name (default: <project>_points)"
                )

            def arcgis_import_xy_points(
                project_path: str,
                excel_path: str,
                x_field: str,
                y_field: str,
                coordinate_system: str,
                sheet_name: Optional[str] = None,
                layer_name: Optional[str] = None,
            ) -> str:
                """
                Import XY points from an Excel file into an ArcGIS Pro project.

                Creates a point feature class in a file geodatabase within the project folder,
                adds it to the map, and saves the project.
                """
                result = self.arcgis_processor.import_xy_points_from_excel(
                    project_path=project_path,
                    excel_path=excel_path,
                    x_field=x_field,
                    y_field=y_field,
                    coordinate_system=coordinate_system,
                    sheet_name=sheet_name,
                    layer_name=layer_name,
                )
                return str(result)

            # --- Tool: Verified workflow from Excel -> points -> hull -> traverse -> CSV ---
            class ArcGISExcelHullTraverseInput(BaseModel):
                """Input schema for a verified Excel->ArcGIS hull+traverse workflow."""
                excel_path: str = Field(description="Path to the Excel file containing coordinates")
                project_name: str = Field(description="ArcGIS project name (folder + .aprx will be created)")
                project_folder: Optional[str] = Field(
                    None,
                    description=(
                        "Folder to create the project in. "
                        "CRITICAL: If not specified, automatically uses the same folder as excel_path. "
                        "This ensures projects are created alongside input files when user doesn't specify location."
                    )
                )
                coordinate_system: str = Field(
                    description="Coordinate system for the imported points (e.g., 'Minna / Nigeria Mid Belt', 'EPSG:26392')"
                )
                output_csv: str = Field(
                    description=(
                        "Output CSV path. Can be absolute path or just filename. "
                        "CRITICAL: If just a filename (not absolute), automatically saved in same folder as excel_path. "
                        "This ensures outputs are created alongside input files when user doesn't specify location."
                    )
                )
                sheet_name: Optional[str] = Field(None, description="Optional Excel sheet name (default: first)")
                close_traverse: bool = Field(True, description="If True, closes traverse last->first")
                clean_project_layers: bool = Field(
                    True,
                    description="If True, removes template/sample layers so only user data appears"
                )

            def arcgis_excel_hull_traverse(**kwargs) -> str:
                """
                VERIFIED ArcGIS workflow:
                - Creates/opens a clean project in the requested folder
                - Imports points from Excel (prefers X/Y)
                - Creates convex hull + computes area
                - Computes traverse distances/bearings
                - Exports CSV and verifies it exists
                """
                # Auto-infer missing path parameters from input file location
                excel_path = kwargs.get("excel_path")
                if excel_path:
                    # If project_folder not specified, use same folder as Excel file
                    if not kwargs.get("project_folder"):
                        inferred_folder = self._infer_output_path_from_input(excel_path, output_type="folder")
                        if inferred_folder:
                            kwargs["project_folder"] = inferred_folder
                            logger.info(f"Auto-inferred project_folder from excel_path: {inferred_folder}")
                    
                    # If output_csv is just a filename (not absolute path), resolve to same folder as Excel
                    output_csv = kwargs.get("output_csv")
                    if output_csv:
                        from pathlib import Path
                        csv_path = Path(output_csv)
                        if not csv_path.is_absolute():
                            inferred_csv = self._infer_output_path_from_input(excel_path, output_filename=output_csv)
                            if inferred_csv:
                                kwargs["output_csv"] = inferred_csv
                                logger.info(f"Auto-inferred output_csv location: {inferred_csv}")
                
                res = self.arcgis_processor.excel_points_convex_hull_traverse(**kwargs)
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: Fill volume (IDW + Cut Fill) - hardened workflow ---
            class ArcGISFillVolumeIDWCutfillInput(BaseModel):
                """Input for verified fill-volume workflow: Excel -> IDW rasters -> Cut Fill -> volume + results_fill.xlsx."""
                excel_path: str = Field(description="Path to the Excel file (one sheet with X, Y, pre and post elevation columns)")
                sheet_name: str = Field(description="Exact sheet name (from excel_inspect_workbook)")
                x_field: str = Field(description="Easting/X column name (e.g. Eastings, EASTING)")
                y_field: str = Field(description="Northing/Y column name (e.g. Northings, NORTHING)")
                post_z_field: str = Field(description="Post-fill elevation column (e.g. 'post fill', Post)")
                pre_z_field: str = Field(description="Pre-fill elevation column (e.g. 'pre fill', Pre)")
                coordinate_system: str = Field(
                    default="Minna / Nigeria Mid Belt",
                    description="Coordinate system (e.g. Nigerian Mid-Belt, EPSG:26392)",
                )
                output_excel_path: Optional[str] = Field(
                    None,
                    description="Output Excel path (default: same folder as input, results_fill.xlsx)",
                )

            def arcgis_fill_volume_idw_cutfill(
                excel_path: str,
                sheet_name: str,
                x_field: str,
                y_field: str,
                post_z_field: str,
                pre_z_field: str,
                coordinate_system: str = "Minna / Nigeria Mid Belt",
                output_excel_path: Optional[str] = None,
            ) -> str:
                """
                VERIFIED fill-volume workflow: no ArcGISProject('CURRENT'), ExcelToTable uses 3rd positional sheet.
                Use this when the user asks for fill volume from Pre-fill/Post-fill data, IDW rasters, Cut Fill, metric, results_fill.xlsx.
                Call excel_inspect_workbook first to get sheet and column names, then call this with the resolved names.
                """
                res = self.arcgis_processor.compute_fill_volume_idw_cutfill(
                    excel_path=excel_path,
                    sheet_name=sheet_name,
                    x_field=x_field,
                    y_field=y_field,
                    post_z_field=post_z_field,
                    pre_z_field=pre_z_field,
                    coordinate_system=coordinate_system,
                    output_excel_path=output_excel_path,
                )
                return json.dumps(res, indent=2, ensure_ascii=False)

            # --- Tool: Execute Python Code ---
            class ArcGISExecutePythonCodeInput(BaseModel):
                """Input schema for executing dynamically generated Python/arcpy code."""
                python_code: str = Field(
                    description=(
                        "Complete Python/arcpy code for FULL AUTOMATION. Generate code that:\n"
                        "1. Performs the complete workflow without manual steps\n"
                        "2. Calculates and prints results in structured format for parsing\n"
                        "3. Handles field discovery, imports, analysis, and result extraction\n\n"
                        "CRITICAL PATTERNS:\n"
                        "FIELD DISCOVERY (prefer value-safe X/Y):\n"
                        "  fields = [f.name for f in arcpy.ListFields(table)]\n"
                        "  # prefer canonical X/Y if present; else fall back to first numeric-looking pair\n"
                        "  x_field = 'X' if 'X' in fields else next((f for f in fields if 'east' in f.lower() or 'lon' in f.lower()), None)\n"
                        "  y_field = 'Y' if 'Y' in fields else next((f for f in fields if 'north' in f.lower() or 'lat' in f.lower()), None)\n"
                        "RESULT OUTPUT: Print results with clear labels: print('RESULT_AREA:', area); print('RESULT_BEARING_1_2:', bearing)\n"
                        "AREA CALCULATION: Use arcpy.da.SearchCursor with SHAPE@ token: area = row[0].area\n"
                        "BEARING/DISTANCE: Use geometry methods: bearing = math.degrees(math.atan2(dy, dx)); distance = math.sqrt(dx**2 + dy**2)\n"
                        "POLYGON CREATION (convex hull): use MinimumBoundingGeometry with a valid group option for points:\n"
                        "  arcpy.management.MinimumBoundingGeometry(points_fc, polygon_fc, 'CONVEX_HULL', group_option='ALL')\n"
                        "TRAVERSE ANALYSIS: Loop through points, calculate bearing/distance between consecutive points\n\n"
                        "IMPORTANT (ArcGIS Excel field types): Avoid XYTableToPoint directly on ExcelToTable outputs because ArcGIS often imports numeric-looking columns as TEXT, causing ERROR 000308.\n"
                        "IMPORTANT (ArcGISProject context): Do NOT set aprx.activeMap or rely on UI-only properties unless you are using arcpy.mp.ArcGISProject('CURRENT') inside the ArcGIS Pro Python Window.\n"
                        "When running headless (propy.bat) or opening an .aprx by path, use aprx.listMaps()[0] and DO NOT attempt to activate maps/views.\n"
                        "IMPORTANT (Headless automation): Do NOT use arcpy.mp.ArcGISProject('CURRENT') in scripts intended to be executed automatically via propy.bat. Always open a project by explicit .aprx path (or create a new project by path).\n"
                        "Preferred: Create the point feature class yourself with CreateFeatureclass + InsertCursor and cast coordinates with float().\n"
                        "Example:\n"
                        "  arcpy.management.CreateFeatureclass(gdb, name, 'POINT', spatial_reference=sr)\n"
                        "  with arcpy.da.SearchCursor(excel_table, ['OID@', x_field, y_field]) as sc:\n"
                        "      with arcpy.da.InsertCursor(fc, ['SrcOID','SHAPE@XY']) as ic:\n"
                        "          for oid, x_raw, y_raw in sc:\n"
                        "              x=float(str(x_raw).replace(',',''))\n"
                        "              y=float(str(y_raw).replace(',',''))\n"
                        "              ic.insertRow((oid, (x,y)))\n\n"
                        "CODE STRUCTURE:\n"
                        "1. Import libraries (arcpy, math, os)\n"
                        "2. Open project, discover fields, import data\n"
                        "3. Create analysis features (polygons, lines)\n"
                        "4. Perform calculations (area, bearings, distances)\n"
                        "5. Print results with RESULT_ prefix for parsing\n"
                        "6. Save project\n\n"
                        "ZOOM/EXTENT (headless-safe): DO NOT rely on Layer.getExtent(). Prefer:\n"
                        "  ext = arcpy.Describe(points_fc).extent\n"
                        "  try: mp.defaultCamera.setExtent(ext)\n"
                        "  except: pass\n\n"
                        "EXAMPLE RESULT FORMAT:\n"
                        "print('RESULT_AREA:', 12345.67, 'square_meters')\n"
                        "print('RESULT_BEARING_P1_P2:', 45.5, 'degrees')\n"
                        "print('RESULT_DISTANCE_P1_P2:', 123.45, 'meters')"
                    )
                )
                project_path: Optional[str] = Field(
                    None,
                    description="Optional path to .aprx project file (used to determine script save location)"
                )
                script_name: Optional[str] = Field(
                    None,
                    description="Optional script filename (default: auto-generated timestamp-based name)"
                )
                execute_automatically: bool = Field(
                    True,
                    description="If True, execute code automatically via propy.bat. If False, save script and provide instructions."
                )
            
            def arcgis_execute_python_code(
                python_code: str,
                project_path: Optional[str] = None,
                script_name: Optional[str] = None,
                execute_automatically: bool = True,
            ) -> str:
                """
                Execute dynamically generated Python/arcpy code.
                
                This tool allows you to generate arcpy code on-the-fly based on user requests and execute it.
                Use this for complex, multi-step operations that require custom arcpy code generation.
                The code is automatically executed and saved to the project's scripts folder for reference.
                
                For operations like "import points and zoom to extent", generate complete arcpy code that:
                - Imports arcpy and opens the project
                - Performs the import operation
                - Gets the layer extent
                - Zooms to the extent using arcpy techniques
                - Saves the project
                """
                result = self.arcgis_processor.execute_python_code(
                    python_code=python_code,
                    project_path=project_path,
                    script_name=script_name,
                    execute_automatically=execute_automatically,
                )
                return str(result)
            
            # --- Tool: Finalize Project Visualization ---
            class ArcGISFinalizeVisualizationInput(BaseModel):
                """Input schema for finalizing project visualization."""
                project_path: str = Field(
                    description="Path to the .aprx project file to finalize"
                )
                load_basemap: bool = Field(
                    default=True,
                    description="If True, add 'Imagery Hybrid' basemap to all maps"
                )
                basemap_name: str = Field(
                    default="Imagery Hybrid",
                    description="Name of basemap to add (default: 'Imagery Hybrid')"
                )
                load_geodatabase: bool = Field(
                    default=True,
                    description="If True, load native geodatabase and all feature classes"
                )
            
            def arcgis_finalize_visualization(
                project_path: str,
                load_basemap: bool = True,
                basemap_name: str = "Imagery Hybrid",
                load_geodatabase: bool = True,
            ) -> str:
                """
                Finalize ArcGIS Pro project visualization after user operations complete.
                
                This function is called AFTER all user-requested operations have been executed
                to ensure the project is visually ready for inspection:
                - Adds 'Imagery Hybrid' basemap to all maps
                - Loads the native geodatabase (project_dir/project_name.gdb) and all its feature classes
                
                IMPORTANT: This should be called AFTER user operations complete, not during project creation,
                so users can visually verify that their instructions were properly carried out.
                """
                result = self.arcgis_processor.finalize_project_visualization(
                    project_path=project_path,
                    load_basemap=load_basemap,
                    basemap_name=basemap_name,
                    load_geodatabase=load_geodatabase,
                )
                return str(result)
            
            # Add ArcGIS tools to the list
            tools.extend([
                StructuredTool(
                    name="filesystem_stat",
                    description=(
                        "Verify whether files/folders exist and their sizes. "
                        "Use this to confirm outputs were actually created before claiming success."
                    ),
                    func=filesystem_stat,
                    args_schema=FilesystemStatInput,
                ),
                StructuredTool(
                    name="arcgis_launch",
                    description=(
                        "Launch ArcGIS Pro application on the user's computer. "
                        "Use this to open ArcGIS Pro before other operations."
                    ),
                    func=arcgis_launch,
                    args_schema=ArcGISLaunchInput
                ),
                StructuredTool(
                    name="arcgis_create_project",
                    description=(
                        "Create a new ArcGIS Pro project. "
                        "Can specify project name, location, coordinate system (e.g., 'UTM Zone 32N', 'WGS84'), "
                        "and template (MAP, CATALOG, GLOBAL_SCENE, LOCAL_SCENE)."
                    ),
                    func=arcgis_create_project,
                    args_schema=ArcGISCreateProjectInput
                ),
                StructuredTool(
                    name="arcgis_open_project",
                    description=(
                        "Open an existing ArcGIS Pro project (.aprx file). "
                        "Provide the full path to the project file."
                    ),
                    func=arcgis_open_project,
                    args_schema=ArcGISOpenProjectInput
                ),
                StructuredTool(
                    name="arcgis_set_coordinate_system",
                    description=(
                        "DEPRECATED: Use arcgis_execute_python_code for complex operations. "
                        "This tool only sets coordinate system and requires manual steps. "
                        "For complete workflows, use arcgis_execute_python_code."
                    ),
                    func=arcgis_set_coordinate_system,
                    args_schema=ArcGISSetCRSInput
                ),
                StructuredTool(
                    name="arcgis_get_project_info",
                    description=(
                        "Get information about the current ArcGIS Pro project. "
                        "Returns maps, coordinate systems, and project details."
                    ),
                    func=arcgis_get_project_info,
                    args_schema=ArcGISProjectInfoInput
                ),
                StructuredTool(
                    name="arcgis_list_coordinate_systems",
                    description=(
                        "List available coordinate systems with their WKID/EPSG codes. "
                        "Filter by text (e.g., 'UTM', 'WGS') to find specific systems."
                    ),
                    func=arcgis_list_coordinate_systems,
                    args_schema=ArcGISListCRSInput
                ),
                StructuredTool(
                    name="arcgis_import_xy_points",
                    description=(
                        "Import XY points from an Excel file into an ArcGIS Pro project and add them to the map. "
                        "Generates a Python script, saves it to the project's scripts folder, launches ArcGIS Pro, "
                        "and provides instructions to run the script in ArcGIS Pro's Python window. "
                        "Use this after coordinate conversion (e.g., WGS84 -> Minna / Nigeria Mid Belt) to create a "
                        "proper point feature class with the correct coordinate system. "
                        "The script will create a file geodatabase, import points, set coordinate system, and add the layer to the map."
                    ),
                    func=arcgis_import_xy_points,
                    args_schema=ArcGISImportXYPointsInput
                ),
                StructuredTool(
                    name="arcgis_execute_python_code",
                    description=(
                        "*** PRIMARY TOOL for FULLY AUTOMATED ArcGIS workflows *** "
                        "Generate and execute complete ArcPy code that performs entire workflows WITHOUT user intervention. "
                        "Use for requests like: 'import points, create polygon, calculate area and bearings', "
                        "'analyze survey data and return traverse calculations', 'process coordinates and generate reports'. "
                        "The code executes automatically via propy.bat and returns computational results. "
                        "Generate code that: imports data, performs analysis, calculates results (areas, bearings, distances), "
                        "prints results with RESULT_ prefix for parsing, saves project. "
                        "NO MANUAL STEPS - user sees only the final results and analysis."
                    ),
                    func=arcgis_execute_python_code,
                    args_schema=ArcGISExecutePythonCodeInput
                ),
                StructuredTool(
                    name="arcgis_excel_hull_traverse",
                    description=(
                        "*** VERIFIED end-to-end workflow *** "
                        "Use this for tasks like: 'import Excel points, create convex hull, compute area, "
                        "compute traverse distances/bearings, export results'. "
                        "This tool verifies inserted point counts and output files on disk, and avoids "
                        "adding any non-user data to the project."
                    ),
                    func=arcgis_excel_hull_traverse,
                    args_schema=ArcGISExcelHullTraverseInput,
                ),
                StructuredTool(
                    name="arcgis_fill_volume_idw_cutfill",
                    description=(
                        "*** VERIFIED fill-volume workflow *** "
                        "Excel -> IDW rasters (pre + post) -> Cut Fill -> fill volume (m³) -> results_fill.xlsx. "
                        "Creates ArcGIS Pro project, adds all layers (pre_idw, post_idw, cutfill, points, post_hull) to map, and opens ArcGIS Pro—as if a GIS analyst did it manually. "
                        "Use excel_inspect_workbook first for sheet/column names. Report project path and layers to user before final volume."
                    ),
                    func=arcgis_fill_volume_idw_cutfill,
                    args_schema=ArcGISFillVolumeIDWCutfillInput,
                ),
                StructuredTool(
                    name="arcgis_finalize_visualization",
                    description=(
                        "Finalize ArcGIS Pro project visualization AFTER user operations complete. "
                        "Adds 'Imagery Hybrid' basemap and loads native geodatabase with all feature classes. "
                        "This should be called AFTER all user-requested operations are done, so users can "
                        "visually verify that their instructions were properly carried out. "
                        "NOTE: This is automatically called by arcgis_execute_python_code and arcgis_excel_hull_traverse, "
                        "but can be called manually if needed."
                    ),
                    func=arcgis_finalize_visualization,
                    args_schema=ArcGISFinalizeVisualizationInput,
                ),
            ])
            
            logger.info(f"✓ Added {10} ArcGIS Pro tools")
        else:
            logger.info("⚠ ArcGIS Pro not installed - ArcGIS tools not available")
        
        logger.info(f"Created {len(tools)} tools for the agent")
        return tools
    
    # ==========================================================================
    # LLM INVOCATION HELPERS
    # ==========================================================================

    def _run_with_timeout(
        self, timeout_seconds: int, fn: Callable[[], Any]
    ) -> Tuple[Optional[Any], Optional[Exception], bool]:
        """
        Run a callable in a daemon thread with a timeout.
        Returns (result, error, timed_out). Exactly one of result or error is set when not timed_out.
        """
        import threading
        result_container: List[Any] = [None]
        err_container: List[Optional[Exception]] = [None]

        def run():
            try:
                result_container[0] = fn()
            except Exception as e:
                err_container[0] = e

        t = threading.Thread(target=run)
        t.daemon = True
        t.start()
        t.join(timeout=timeout_seconds)
        return (result_container[0], err_container[0], t.is_alive())

    def _invoke_llm_with_retry(self, messages: List[Any]) -> Any:
        """Invoke LLM with timeout protection; raises TimeoutError or the LLM exception on failure."""
        timeout_seconds = 60
        result, error, timed_out = self._run_with_timeout(
            timeout_seconds, lambda: self.llm_with_tools.invoke(messages)
        )
        if timed_out:
            logger.error(f"LLM invocation timed out after {timeout_seconds} seconds")
            raise TimeoutError(
                f"LLM call timed out after {timeout_seconds} seconds. "
                "The query may be too complex or the document too large. "
                "Try breaking the task into smaller steps."
            )
        if error:
            if "429" in str(error) or "rate limit" in str(error).lower() or "tpm" in str(error).lower():
                logger.warning("Rate limit error detected, will be handled by caller")
            raise error
        return result

    def _ensure_app_bound(self, llm: BaseChatModel, model_name: Optional[str], tools_to_bind: List[BaseTool]) -> None:
        """
        Bind tools and (re)compile the graph only when necessary.
        This is a pure efficiency optimization: behavior is unchanged.
        """
        model_sig = model_name or getattr(llm, "model", None) or "unknown"
        tool_sig = tuple(sorted([t.name for t in tools_to_bind]))

        if self._app_signature == (model_sig, tool_sig) and getattr(self, "app", None) is not None:
            return

        self._current_tools = tools_to_bind
        self.llm_with_tools = llm.bind_tools(tools_to_bind)
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        self._app_signature = (model_sig, tool_sig)
    
    # ==========================================================================
    # LANGGRAPH CONSTRUCTION
    # ==========================================================================
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph that defines the agent's behavior.
        
        The graph has two main nodes:
        1. agent_node: Calls the LLM to reason about the query
        2. tools_node: Executes any tools the LLM requests
        
        The flow is:
        - Start → agent_node
        - agent_node → (if tool calls) → tools_node → agent_node
        - agent_node → (if no tool calls) → END
        
        Returns:
            StateGraph: The configured graph (not yet compiled)
        """
        
        # ==================================================================
        # Define the agent node
        # ==================================================================
        
        def agent_node(state: AgentState) -> Dict:
            """
            The agent node - where the LLM does its reasoning.
            
            This node:
            1. Takes the current conversation state
            2. Sends messages to the LLM (with system prompt)
            3. Returns the LLM's response (which may include tool calls)
            
            Args:
                state: Current conversation state with message history
                
            Returns:
                Dict with new messages to add to state
            """
            # Get current messages from state
            messages = list(state["messages"])
            
            # Ensure system prompt is at the start
            # This guides the LLM's behavior throughout the conversation
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
            # Get current model name for token estimation
            current_model = getattr(self, '_current_openai_model', None) or \
                          getattr(self, '_current_gemini_model', None) or \
                          getattr(self.settings, 'openai_model', 'gpt-4o-mini')
            
            # Estimate tokens and check if we need chunking
            input_tokens, output_tokens_estimate = estimate_message_tokens(messages, current_model)
            token_estimate = check_tpm_limit(input_tokens, output_tokens_estimate, current_model)
            
            # Check if there are any pending tool calls in the conversation
            # We cannot chunk messages that contain tool calls because tool calls must be
            # immediately followed by tool responses (OpenAI API requirement)
            has_pending_tool_calls = False
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    has_pending_tool_calls = True
                    break
                # Also check if we have ToolMessages that might be part of an active sequence
                if isinstance(msg, ToolMessage):
                    # If there's a ToolMessage, there was likely a tool call before it
                    # Check if the previous message was an AIMessage with tool_calls
                    msg_idx = messages.index(msg)
                    if msg_idx > 0 and isinstance(messages[msg_idx - 1], AIMessage):
                        if messages[msg_idx - 1].tool_calls:
                            has_pending_tool_calls = True
                            break
            
            # If tokens exceed limit AND there are no pending tool calls, chunk the messages
            # Otherwise, proceed normally and let rate limit errors be handled by retries
            if token_estimate.exceeds_tpm and not has_pending_tool_calls:
                logger.warning(
                    f"Token limit exceeded in agent_node: {token_estimate.total_tokens:,} tokens. "
                    f"Chunking into {token_estimate.chunks_needed} chunks."
                )
                # Chunk with 90% of TPM limit for safety
                safe_limit = int(token_estimate.tpm_limit * 0.9)
                message_chunks = chunk_messages(messages, safe_limit, current_model)
                
                # Process chunks with delays
                all_responses = []
                for chunk_idx, chunk in enumerate(message_chunks):
                    if chunk_idx > 0:
                        # Wait 61 seconds between chunks to reset rate limit window
                        wait_for_rate_limit(61)
                    
                    logger.info(f"Processing chunk {chunk_idx + 1}/{len(message_chunks)}")
                    try:
                        response = self._invoke_llm_with_retry(chunk)
                        all_responses.append(response)
                    except Exception as e:
                        # If rate limit error, wait and retry
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            logger.warning(f"Rate limit hit on chunk {chunk_idx + 1}, waiting and retrying...")
                            wait_for_rate_limit(61)
                            response = self._invoke_llm_with_retry(chunk)
                            all_responses.append(response)
                        else:
                            raise
                
                # Combine responses (for now, use the last response)
                # In a more sophisticated implementation, we could merge responses
                response = all_responses[-1] if all_responses else None
                if not response:
                    raise RuntimeError("No response received from chunked LLM calls")
            elif token_estimate.exceeds_tpm and has_pending_tool_calls:
                # Cannot chunk due to tool calls - proceed normally and handle rate limits via retries
                logger.warning(
                    f"Token limit exceeded ({token_estimate.total_tokens:,} tokens) but tool calls detected. "
                    f"Proceeding without chunking - rate limits will be handled by retries."
                )
                response = self._invoke_llm_with_retry(messages)
            else:
                # Normal path: no chunking needed
                response = self._invoke_llm_with_retry(messages)
            
            # Return the response to be added to state
            return {"messages": [response]}
        
        # ==================================================================
        # Define the routing function
        # ==================================================================
        
        def should_continue(state: AgentState) -> Literal["tools", "end"]:
            """
            Determine whether to execute tools or end the conversation.
            
            This function looks at the last message from the agent:
            - If it contains tool_calls → route to "tools" node
            - Otherwise → route to END (conversation complete)
            
            Also enforces max_iterations limit to prevent infinite loops.
            
            Args:
                state: Current conversation state
                
            Returns:
                "tools" if tools should be executed, "end" otherwise
            """
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check iteration count to prevent infinite loops
            max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
            tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
            iteration_count = len(tool_messages)

            if iteration_count >= max_iterations:
                logger.warning(
                    f"Max iterations ({max_iterations}) reached. "
                    "Stopping to prevent infinite loop. "
                    "The query may be too complex or require manual intervention."
                )
                return "end"

            # Same-error stop: if last two tool results look like the same failure, end to avoid runaway cost
            if iteration_count >= 2:
                last_two = tool_messages[-2:]
                contents = []
                for tm in last_two:
                    c = getattr(tm, "content", None) or ""
                    if isinstance(c, list):
                        c = " ".join(str(part.get("text", part)) for part in c if isinstance(part, dict))
                    else:
                        c = str(c)
                    contents.append(c[:300].lower().strip())
                if contents[0] and contents[1] and (
                    contents[0] == contents[1]
                    or (contents[0].split()[:20] == contents[1].split()[:20] and ("error" in contents[0] or "failed" in contents[0]))
                ):
                    logger.warning(
                        "Same or very similar tool error repeated; stopping loop to prevent runaway cost. "
                        "Report what was tried and suggest next step."
                    )
                    return "end"

            # Check if the AI wants to use tools
            # AIMessage has a tool_calls attribute when tools are requested
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                logger.info(f"Iteration {iteration_count + 1}/{max_iterations}: Agent requested tools: {', '.join(tool_names)}")
                # Warn if internet_search is being called (should be removed if already searched)
                if "internet_search" in tool_names:
                    logger.warning("⚠ Agent is trying to call internet_search - this may indicate a loop if search was already done")
                return "tools"
            
            # No tool calls - we're done
            logger.info(f"Agent completed after {iteration_count} tool iterations")
            return "end"
        
        # ==================================================================
        # Build the graph
        # ==================================================================
        
        # Create a new graph with our state schema
        graph = StateGraph(AgentState)
        
        # Add the agent node (LLM reasoning)
        graph.add_node("agent", agent_node)
        
        # Add the tools node (tool execution)
        # ToolNode is a pre-built node that handles tool execution.
        #
        # NOTE: We intentionally avoid wrapping ToolNode here because LangGraph/LangChain
        # passes runtime config through the graph, and wrappers can accidentally drop or
        # invalidate required config keys (causing errors like "Missing required config key ... for 'tools'").
        # Use filtered tools if available (e.g., when internet_search was removed), otherwise use all tools
        tools_for_node = getattr(self, "_current_tools", None) or self.tools
        tools_node = ToolNode(tools_for_node)
        graph.add_node("tools", tools_node)
        
        # Set the entry point - where the graph starts
        graph.set_entry_point("agent")
        
        # Add conditional routing from agent
        # After the agent runs, we check if tools should be called
        graph.add_conditional_edges(
            "agent",           # From node
            should_continue,   # Routing function
            {
                "tools": "tools",   # If should_continue returns "tools"
                "end": END          # If should_continue returns "end"
            }
        )
        
        # After tools run, go back to agent to process results
        graph.add_edge("tools", "agent")
        
        logger.info("LangGraph built successfully")
        return graph
    
    # ==========================================================================
    # QUERY PROCESSING
    # ==========================================================================
    
    def process_query(
        self, 
        query: str, 
        use_fallback: bool = False,
        session_id: Optional[str] = None,
        interactive_mode: bool = False
    ) -> Dict:
        """
        Process a user query through the agent.
        
        This is the main entry point for using the agent. It:
        1. Retrieves relevant context from vector store (if enabled)
        2. Creates a new conversation thread
        3. Invokes the LangGraph with the query and context
        4. Extracts and returns the final response
        5. Stores the conversation for future context (if enabled)
        6. Handles errors and fallback to secondary LLM
        7. Checks token limits and handles chunking with rate limiting
        
        Args:
            query: The user's question or request
            use_fallback: If True, use the fallback LLM instead of primary
            session_id: Optional session ID for conversation continuity.
                       If not provided, uses the current session or creates new.
            interactive_mode: If True, will ask user for approval when token limits are exceeded
            
        Returns:
            Dict containing:
            - query: The original query
            - response: The agent's response text
            - llm_used: Which LLM was used ("primary" or "fallback")
            - success: Whether the query was processed successfully
            - error: Error message if success is False
            - session_id: The session ID used for this query
            - context_retrieved: Whether context was retrieved from vector store
            
        Example:
            >>> result = agent.process_query("What is the area of this survey?")
            >>> print(result["response"])
            "The total area of the survey is 1,500 square meters..."
        """
        # Set interactive mode flag
        self._interactive_mode = interactive_mode
        
        # Reset model switch flag for new query
        self._model_switched_this_query = False

        # Handle internet permission markers from interactive CLI
        # IMPORTANT: Extract and set permission BEFORE routing/processing
        q_upper = (query or "").upper()
        original_query = query
        
        # FIRST: Extract actual query if this is a continuation (has context markers)
        # This must happen BEFORE permission tag handling so we route correctly
        actual_query_for_routing = query
        if "=== CONTINUATION OF PREVIOUS WORK" in query or "--- Exchange" in query:
            # Extract the actual current query from the context-enhanced query
            if "NOW, the user wants you to continue with this new request:" in query:
                parts = query.split("NOW, the user wants you to continue with this new request:")
                if len(parts) > 1:
                    actual_query_for_routing = parts[-1].strip()
                    logger.info(f"🔍 Detected continuation query - extracted actual request: {actual_query_for_routing[:100]}...")
            elif "\n\n" in query:
                # Fallback: get the last part after double newline
                parts = query.split("\n\n")
                actual_query_for_routing = parts[-1].strip()
                logger.info(f"🔍 Detected continuation query - using last part: {actual_query_for_routing[:100]}...")
        
        if "[INTERNET_PERMISSION_GRANTED]" in q_upper:
            self._internet_permission_granted = True
            # Clean the query to remove permission tags for cleaner processing
            query = query.replace("[INTERNET_PERMISSION_GRANTED]", "").replace("[internet_permission_granted]", "").strip()
            actual_query_for_routing = actual_query_for_routing.replace("[INTERNET_PERMISSION_GRANTED]", "").replace("[internet_permission_granted]", "").strip()
            logger.info("✓ Internet permission granted - permission tag removed from query")
        if "[INTERNET_PERMISSION_DENIED]" in q_upper:
            self._internet_permission_granted = False
            query = query.replace("[INTERNET_PERMISSION_DENIED]", "").replace("[internet_permission_denied]", "").strip()
            actual_query_for_routing = actual_query_for_routing.replace("[INTERNET_PERMISSION_DENIED]", "").replace("[internet_permission_denied]", "").strip()
            logger.info("✓ Internet permission denied - permission tag removed from query")
        
        try:
            logger.info(f"Processing query: {query[:200]}...")
            
            # ==================================================================
            # EARLY ROUTER CHECK: Ask for internet permission BEFORE any processing
            # This prevents loops and ensures proactive permission requests
            # Use the extracted actual query for routing (not the context-enhanced one)
            # ==================================================================
            early_rag_decision = self._decide_rag_route(actual_query_for_routing, interactive_mode=interactive_mode)
            if early_rag_decision.use_internet and not getattr(self, "_internet_permission_granted", False):
                if interactive_mode:
                    logger.info("🔍 Router detected internet need - requesting permission BEFORE processing")
                    return {
                        "query": original_query,
                        "response": (
                            "PERMISSION REQUIRED: INTERNET SEARCH\n\n"
                            "This query appears to require up-to-date external information (standards, citations, current data).\n"
                            "May I search the internet for up-to-date information? (yes/no)\n\n"
                            "[INTERNET_PERMISSION_REQUEST]"
                        ),
                        "success": False,
                        "error": "internet_permission_required",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": None,  # Not initialized yet
                        "session_id": self.get_session_id(),
                    }
                else:
                    # Non-interactive: proceed without internet (user can't grant permission)
                    logger.warning("⚠ Internet needed but non-interactive mode - proceeding without internet search")
            
            logger.info(f"Current primary LLM setting: {self.settings.primary_llm}")
            
            # Detect task complexity for tiered model selection
            complexity = self._detect_task_complexity(query)
            logger.info(f"Detected task complexity: {complexity}")
            
            # Determine which LLM and model to use
            llm_to_use = None
            model_name_used = None
            
            if use_fallback:
                logger.warning(f"⚠ Using fallback LLM ({self.settings.fallback_llm}) instead of primary ({self.settings.primary_llm})")
                llm_to_use = self.llm_fallback
                if self.settings.fallback_llm == "openai":
                    # For OpenAI fallback, still use complexity-based selection if enabled
                    if getattr(self.settings, "enable_tiered_models", True):
                        model_name = self._get_openai_model_for_complexity(complexity)
                        llm_to_use = self._initialize_llm("openai", model_name=model_name)
                        model_name_used = model_name
                        logger.info(f"Using OpenAI fallback model: {model_name} (complexity: {complexity})")
                    else:
                        model_name_used = getattr(self.settings, "openai_model", "gpt-4o-mini")
                elif self.settings.fallback_llm == "gemini":
                    model_name_used = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
                else:
                    model_name_used = self.settings.fallback_llm
            else:
                # Using primary LLM with complexity-based selection for OpenAI
                if self.settings.primary_llm == "openai" and getattr(self.settings, "enable_tiered_models", True):
                    model_name = self._get_openai_model_for_complexity(complexity)
                    llm_to_use = self._initialize_llm("openai", model_name=model_name)
                    self._current_openai_model = model_name
                    model_name_used = model_name
                    logger.info(f"✓ Using OpenAI model: {model_name} (complexity: {complexity})")
                else:
                    # Use standard primary LLM (either non-OpenAI or tiered models disabled)
                    llm_to_use = self.llm_primary
                    if self.settings.primary_llm == "openai":
                        model_name_used = getattr(self.settings, "openai_model", "gpt-4o-mini")
                        self._current_openai_model = model_name_used
                    elif self.settings.primary_llm == "gemini":
                        model_name_used = self._current_gemini_model or getattr(self.settings, "gemini_model", "gemini-2.0-flash")
                    else:
                        model_name_used = self.settings.primary_llm
                    logger.info(f"✓ Using primary LLM: {self.settings.primary_llm} (model: {model_name_used})")

            # FAST PATH: report generation to .docx (avoids LangGraph recursion/tool loops)
            if self._should_fastpath_docx_report(query):
                out_candidate = self._extract_any_output_docx(query) or "Report.docx"
                fast = self._run_docx_report_pipeline(
                    query=query,
                    output_doc_path=out_candidate,
                    llm=llm_to_use,
                    model_name_used=model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5-mini"),
                )
                llm_used = "fallback" if use_fallback else "primary"
                return {
                    "query": query,
                    "response": fast.get("response", ""),
                    "llm_used": llm_used,
                    "model_name": fast.get("model_name", model_name_used),
                    "complexity": complexity,
                    "success": bool(fast.get("success")),
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": fast.get("output_path"),
                    "error": fast.get("error") if not fast.get("success") else None,
                }

            # FAST PATH: cadastral CAD prompt (template DWG -> output DWG with parcel replot)
            if self._should_fastpath_cadastral_cad(query):
                fast = self._run_cadastral_cad_prompt_pipeline(query)
                llm_used = "fallback" if use_fallback else "primary"
                if fast.get("success"):
                    self._last_cadastral_output_dwg = fast.get("output_dwg")
                    self._last_cadastral_profile_path = fast.get("profile_path")
                    resp_lines = [
                        "✅ Cadastral plan generated from template.",
                        f"- Output: {fast.get('output_dwg')}",
                        f"- Geometry: {fast.get('geometry')}",
                    ]
                    if fast.get("access_road_title"):
                        resp_lines.append(f"- Access road title (as plotted): {fast.get('access_road_title')!r}")
                    try:
                        bow = (fast.get("geometry") or {}).get("bowditch") if isinstance(fast, dict) else None
                        if isinstance(bow, dict) and bow.get("mode") == "bearing_distance":
                            if bow.get("applied"):
                                resp_lines.append(
                                    "- Bowditch adjustment applied (misclosure > 1cm): "
                                    f"misclosure={bow.get('misclosure_m'):.3f}m "
                                    f"(E={bow.get('misclosure_e_m'):.3f}m, N={bow.get('misclosure_n_m'):.3f}m), "
                                    f"max point shift={bow.get('max_point_shift_m'):.3f}m."
                                )
                                prev = bow.get("adjusted_points_preview") or []
                                if prev:
                                    resp_lines.append(f"- Adjusted points preview (first {len(prev)}): {prev}")
                            else:
                                resp_lines.append(
                                    "- Bowditch adjustment not applied: "
                                    f"misclosure={bow.get('misclosure_m'):.3f}m (<= 0.010m threshold)."
                                )
                    except Exception:
                        pass
                    resp_lines.append("\nYou can request modifications in this session (e.g. add another road, change the title) without closing or re-prompting.")
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": fast.get("output_dwg"),
                    }
                return {
                    "query": query,
                    "response": str(fast),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": False,
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": None,
                    "error": fast.get("error") if isinstance(fast, dict) else "Fastpath cadastral pipeline failed",
                }

            # FAST PATH: in-session CAD plan modifications (add road, change title, etc.)
            # Template remains read-only; modifications apply to the output plan file (even if open).
            if self._should_fastpath_cad_modification(query):
                mod = self._run_cad_modification_pipeline(query)
                llm_used = "fallback" if use_fallback else "primary"
                if mod.get("success"):
                    resp_lines = [
                        "✅ Plan updated.",
                        f"- File: {mod.get('output_dwg')}",
                        f"- Modifications: {mod.get('modifications', [])}",
                    ]
                    if mod.get("save_warning"):
                        resp_lines.append(f"- Note: {mod.get('save_warning')}")
                    return {
                        "query": query,
                        "response": "\n".join(resp_lines) + "\n",
                        "llm_used": llm_used,
                        "model_name": model_name_used,
                        "complexity": complexity,
                        "success": True,
                        "session_id": self.get_session_id(),
                        "context_retrieved": False,
                        "output_path": mod.get("output_dwg"),
                    }
                return {
                    "query": query,
                    "response": str(mod.get("error", mod)),
                    "llm_used": llm_used,
                    "model_name": model_name_used,
                    "complexity": complexity,
                    "success": False,
                    "session_id": self.get_session_id(),
                    "context_retrieved": False,
                    "output_path": None,
                    "error": mod.get("error") if isinstance(mod, dict) else "CAD modification failed",
                }

            # Get or set session ID for conversation continuity
            if session_id:
                self.set_session_id(session_id)
            current_session_id = self.get_session_id()
            
            # AUTOMATIC DOCUMENT PRE-PROCESSING: Detect document paths and get resource estimation
            # This prevents the agent from trying to process large documents without knowing the cost/size
            document_paths = self._extract_document_paths(query)
            document_preflight_info = []
            
            if document_paths:
                logger.info(f"Detected {len(document_paths)} document path(s) in query - running pre-flight checks")
                for doc_path in document_paths:
                    try:
                        path_obj = Path(doc_path)
                        file_size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0
                        model_for_est = model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5-mini")
                        est = self.document_processor.get_resource_estimation(doc_path, model_for_est)
                        if est.get("success"):
                            doc_info = {
                                "path": doc_path,
                                "file_size_mb": est.get("file_size_mb", file_size_mb),
                                "page_count": est.get("page_count", 0),
                                "word_count": est.get("word_count", 0),
                                "estimated_tokens": est.get("estimated_tokens", 0),
                                "warnings": est.get("warnings", []),
                                "recommendations": est.get("recommendations", []),
                                "estimated_cost": est.get("estimated_cost", {}),
                                "estimation": est,
                            }
                            document_preflight_info.append(doc_info)
                            logger.info(f"Pre-flight check for {doc_path}: {doc_info['page_count']} pages, {doc_info['word_count']} words, ~{doc_info['estimated_tokens']} tokens")
                        else:
                            # Fallback: estimation failed (e.g., very large/table-heavy doc) - use file size for routing
                            doc_info = {
                                "path": doc_path,
                                "file_size_mb": file_size_mb,
                                "page_count": 0,
                                "word_count": 0,
                                "estimated_tokens": 0,
                                "estimation": {"success": False},
                            }
                            document_preflight_info.append(doc_info)
                            logger.info(f"Pre-flight fallback for {doc_path}: {file_size_mb:.2f} MB (estimation unavailable)")
                    except Exception as e:
                        logger.warning(f"Could not get resource estimation for {doc_path}: {e}")
                        # Still add minimal info so file-size-based fast-path can trigger
                        try:
                            path_obj = Path(doc_path)
                            file_size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0
                            document_preflight_info.append({
                                "path": doc_path,
                                "file_size_mb": file_size_mb,
                                "page_count": 0,
                                "word_count": 0,
                                "estimated_tokens": 0,
                                "estimation": {"success": False},
                            })
                        except Exception:
                            pass

                # FAST PATH: if this is a large-document summarize/save request, run a deterministic pipeline
                # Even if multiple .docx are mentioned (e.g., output filename), pick the *largest* doc as the input.
                if document_preflight_info:
                    # Pick largest doc by tokens, then by file size (for estimation failures)
                    primary_doc = max(
                        document_preflight_info,
                        key=lambda d: (int(d.get("estimated_tokens") or 0), float(d.get("file_size_mb") or 0))
                    )
                    if self._should_fastpath_large_doc_summary(query, primary_doc):
                        input_doc = primary_doc["path"]
                        output_doc = self._extract_requested_output_docx(query, input_doc) or str(
                            (Path(input_doc).parent / f"Summary_{Path(input_doc).stem}.docx").resolve()
                        )
                        logger.info("Using fast-path large document summary pipeline")
                        fast_result = self._run_large_doc_summary_pipeline(
                            query=query,
                            input_doc_path=input_doc,
                            output_doc_path=output_doc,
                            llm=llm_to_use,
                            model_name_used=model_name_used or getattr(self.settings, "openai_model_mini", "gpt-5-mini")
                        )
                        if fast_result.get("success"):
                            # Store conversation for future context
                            llm_used = "fallback" if use_fallback else "primary"
                            self._store_conversation(
                                query=query,
                                response=fast_result.get("response", ""),
                                session_id=current_session_id,
                                llm_used=llm_used
                            )
                            return {
                                "query": query,
                                "response": fast_result.get("response", ""),
                                "llm_used": llm_used,
                                "model_name": fast_result.get("model_name", model_name_used),
                                "complexity": complexity,
                                "success": True,
                                "session_id": current_session_id,
                                "context_retrieved": False
                            }
            
            # Agentic RAG routing: decide whether to use VectorStore retrieval and/or Internet search.
            # Use cleaned query (without permission tags) for routing decision
            # NOTE: We already checked permission above, so if we reach here and internet is needed, permission is granted
            rag_decision = early_rag_decision  # Reuse the early decision

            internet_block = ""
            internet_already_searched = False  # Track if we already did internet search
            if rag_decision.use_internet:
                # Permission should already be granted (checked above), but double-check
                if getattr(self, "_internet_permission_granted", False):
                    try:
                        logger.info(f"🔍 Executing internet search for: {rag_decision.internet_query or query}")
                        res = _internet_search(rag_decision.internet_query or query)
                        if res.get("success"):
                            internet_block = self._format_internet_results_for_prompt(res.get("results") or [])
                            internet_already_searched = True
                            # Mark that internet was already searched (for tool safety check)
                            self._internet_already_searched_this_query = True
                            logger.info(f"✓ Internet search completed: {len(res.get('results', []))} results")
                        else:
                            logger.warning(f"⚠ Internet search returned no results: {res.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"⚠ Internet search failed during routing: {e}")
                else:
                    # This shouldn't happen (we checked above), but log it
                    logger.warning("⚠ Internet needed but permission not granted - this should have been caught earlier")
            else:
                # Reset flag if internet is not needed
                self._internet_already_searched_this_query = False

            retrieved_context = ""
            if rag_decision.use_vector:
                retrieved_context = self._retrieve_relevant_context(
                    query, current_session_id, collections=rag_decision.vector_collections or None
                )
            context_retrieved = bool(retrieved_context)
            
            # Build enhanced system prompt with routed augmentation + document pre-flight info
            enhanced_system_prompt = SYSTEM_PROMPT
            if document_preflight_info:
                doc_context = "\n\n---\n**DOCUMENT PRE-FLIGHT ANALYSIS (AUTOMATIC):**\n"
                doc_context += "The following document(s) were detected in the user's query. Resource estimation has been performed:\n\n"
                for doc_info in document_preflight_info:
                    doc_context += f"**Document: {doc_info['path']}**\n"
                    doc_context += f"  • Size: {doc_info['file_size_mb']:.2f} MB\n"
                    doc_context += f"  • Pages: {doc_info['page_count']}\n"
                    doc_context += f"  • Words: {doc_info['word_count']:,}\n"
                    doc_context += f"  • Estimated tokens: {doc_info['estimated_tokens']:,}\n"
                    if doc_info.get('estimated_cost', {}).get('total_cost'):
                        doc_context += f"  • Estimated cost: ${doc_info['estimated_cost']['total_cost']:.4f}\n"
                    if doc_info.get('warnings'):
                        doc_context += f"  • Warnings: {'; '.join(doc_info['warnings'])}\n"
                    if doc_info.get('recommendations'):
                        doc_context += f"  • Recommendations: {'; '.join(doc_info['recommendations'])}\n"
                    doc_context += "\n"
                doc_context += "**CRITICAL INSTRUCTIONS (MANDATORY - TPM/RATE LIMIT PROTECTION):**\n"
                doc_context += "1. For ANY document with >50 pages, >25K words, or >50K estimated tokens: use document_extract_sections_by_keywords() ONLY. NEVER use document_get_text() or document_get_full_text().\n"
                doc_context += "2. For summarize/summary requests that save to .docx: extract sections by keywords (e.g. Location, Personnel, Purpose, Coordinates, Control Points, Projects) then write the summary from the extracted content only.\n"
                doc_context += "3. Full text extraction causes TPM overflow (500K limit) and will fail with 429 rate limit. ALWAYS prefer section extraction for large docs.\n"
                doc_context += "4. Process only the extracted sections - never attempt to load the entire document into context.\n"
                doc_context += "---\n"
                enhanced_system_prompt += doc_context
                logger.info("✓ Document pre-flight info injected into system prompt")
            
            if retrieved_context:
                enhanced_system_prompt += (
                    f"\n\n---\n"
                    f"**CONTEXT FROM PREVIOUS SESSIONS AND STORED DOCUMENTS:**\n"
                    f"The following context may be relevant to the user's query:\n\n"
                    f"{retrieved_context}\n"
                    f"---\n"
                    f"Use this context to provide more informed and consistent responses. "
                    f"If the context doesn't seem relevant to the current query, you may ignore it."
                )
                logger.info("✓ Context injected into system prompt")

            if internet_block:
                enhanced_system_prompt += internet_block
                # Add STRONG instruction to LLM: internet search already done, permission already granted, tool removed
                enhanced_system_prompt += (
                    "\n\n**CRITICAL - READ CAREFULLY:**\n"
                    "1. Internet search has ALREADY been performed with user permission.\n"
                    "2. The results are included above in the 'INTERNET SEARCH RESULTS' section.\n"
                    "3. The internet_search tool has been REMOVED from your available tools.\n"
                    "4. DO NOT ask for permission - permission was already granted.\n"
                    "5. DO NOT mention needing to search - the search is already done.\n"
                    "6. Use the information provided above to answer the user's query directly.\n"
                    "7. If you reference external information, cite it from the results above."
                )
                logger.info("✓ Internet results injected into system prompt (permission granted)")
            
            # Bind tools to the selected LLM and rebuild graph
            # CRITICAL: If internet was already searched, conditionally remove internet_search tool
            # to prevent the LLM from calling it again and causing loops
            tools_to_bind = self.tools
            if internet_already_searched:
                # Filter out internet_search tool to prevent redundant calls
                tools_to_bind = [t for t in self.tools if t.name != "internet_search"]
                logger.info("✓ Removed internet_search tool (already searched) to prevent loops")

            # Bind tools and compile graph only if (model, toolset) changed
            self._ensure_app_bound(llm_to_use, model_name_used, tools_to_bind)
            
            try:
                # Use session_id as thread_id to maintain conversation state across queries
                # This ensures LangGraph remembers the conversation history
                thread_id = current_session_id
                # Increase recursion_limit to avoid premature GRAPH_RECURSION_LIMIT on complex tool workflows.
                max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
                recursion_limit = getattr(self.settings, "agent_recursion_limit", max(50, (max_iterations * 3)))
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": recursion_limit,
                }
                
                # Prepare initial state with enhanced system prompt and user query
                initial_messages = [
                    SystemMessage(content=enhanced_system_prompt),
                    HumanMessage(content=query)
                ]
                
                # Pre-flight token check: Estimate tokens and check TPM limits
                input_tokens, output_tokens_estimate = estimate_message_tokens(
                    initial_messages, model_name_used
                )
                token_estimate = check_tpm_limit(
                    input_tokens, output_tokens_estimate, model_name_used
                )
                
                # Check if user has already approved chunking (from interactive mode)
                user_approved_chunking = "[USER APPROVED:" in query.upper()
                
                # If tokens exceed TPM limit, handle chunking or ask for approval
                if token_estimate.exceeds_tpm and not user_approved_chunking:
                    logger.warning(
                        f"Token limit exceeded: {token_estimate.total_tokens:,} tokens "
                        f"(limit: {token_estimate.tpm_limit:,}) for model {model_name_used}"
                    )
                    
                    # In interactive mode, ask for user approval
                    if interactive_mode:
                        warning_msg = format_token_warning(token_estimate, model_name_used)
                        # Return a special response that the CLI can detect and prompt for
                        return {
                            "query": query,
                            "response": warning_msg,
                            "success": False,
                            "error": "token_limit_exceeded",
                            "token_estimate": {
                                "total_tokens": token_estimate.total_tokens,
                                "tpm_limit": token_estimate.tpm_limit,
                                "chunks_needed": token_estimate.chunks_needed,
                                "estimated_cost": token_estimate.estimated_cost,
                            },
                            "llm_used": "fallback" if use_fallback else "primary",
                            "model_name": model_name_used,
                            "session_id": current_session_id,
                        }
                    else:
                        # Non-interactive mode: automatically proceed with chunking
                        logger.info(
                            f"Proceeding with automatic chunking: {token_estimate.chunks_needed} chunks "
                            f"with 61s delays"
                        )
                elif token_estimate.exceeds_tpm and user_approved_chunking:
                    # User approved, proceed with chunking
                    logger.info(
                        f"User approved chunking: {token_estimate.chunks_needed} chunks "
                        f"with 61s delays"
                    )
                
                initial_state = {"messages": initial_messages}
                
                # Invoke the graph with overall timeout protection
                # This runs the agent through its reasoning cycle
                # Add timeout wrapper to prevent indefinite hanging
                import threading
                import signal
                
                overall_timeout = getattr(self.settings, 'agent_query_timeout', 300)  # Default 5 minutes
                max_iterations = getattr(self.settings, 'agent_max_iterations', 20)
                result_container = [None]
                exception_container = [None]
                
                def invoke_graph():
                    try:
                        logger.info(f"Starting graph execution (timeout: {overall_timeout}s, max iterations: {max_iterations})")
                        logger.info("Processing query - this may take a moment for large documents...")
                        result_container[0] = self.app.invoke(initial_state, config=config)
                        logger.info("Graph execution completed successfully")
                    except Exception as e:
                        logger.error(f"Error during graph execution: {e}")
                        exception_container[0] = e
                
                graph_thread = threading.Thread(target=invoke_graph)
                graph_thread.daemon = True
                graph_thread.start()
                graph_thread.join(timeout=overall_timeout)
                
                if graph_thread.is_alive():
                    logger.error(f"Graph execution timed out after {overall_timeout} seconds")
                    raise TimeoutError(
                        f"Query processing timed out after {overall_timeout} seconds. "
                        "The document may be too large or the query too complex. "
                        "Try using document_get_resource_estimation() first, then document_extract_sections_by_keywords() "
                        "to process only relevant sections."
                    )
                
                if exception_container[0]:
                    error = exception_container[0]
                    error_str = str(error).lower()
                    
                    # Detect if model is struggling and should be switched
                    should_switch_model = False
                    switch_reason = None
                    
                    # Check for recursion limit (model can't handle complexity)
                    if "recursion limit" in error_str or "graph_recursion_limit" in error_str:
                        should_switch_model = True
                        switch_reason = "recursion_limit"
                        logger.warning("🔄 Model hit recursion limit - considering model switch")
                    
                    # Check for token/TPM limits (model too small for task)
                    if "tokens per min" in error_str or "tpm" in error_str or "token limit" in error_str:
                        # Only switch if we're not already on the highest tier
                        current_tier = self._get_model_tier(model_name_used)
                        if current_tier != "complex":
                            should_switch_model = True
                            switch_reason = "token_limit"
                            logger.warning(f"🔄 Model hit token limit (tier: {current_tier}) - considering model switch")
                    
                    # If model switch is needed and we haven't already switched
                    if should_switch_model and not getattr(self, "_model_switched_this_query", False):
                        logger.info(f"🔄 Attempting dynamic model switch (reason: {switch_reason})")
                        return self._switch_model_and_retry(
                            query=query,
                            original_query=original_query,
                            current_model=model_name_used,
                            current_llm=llm_to_use,
                            complexity=complexity,
                            enhanced_system_prompt=enhanced_system_prompt,
                            initial_messages=initial_messages,
                            current_session_id=current_session_id,
                            use_fallback=use_fallback,
                            interactive_mode=interactive_mode,
                            context_retrieved=context_retrieved,
                            switch_reason=switch_reason,
                            tools_to_bind=tools_to_bind if 'tools_to_bind' in locals() else self.tools,
                        )
                    
                    # If we can't switch or already switched, raise the error
                    raise error
                
                result = result_container[0]
                if result is None:
                    raise RuntimeError("Graph execution returned no result")
                
                # Extract the final response from messages
                response_text = self._extract_response(result)
                
                # Store conversation in vector store for future context
                llm_used = "fallback" if use_fallback else "primary"
                self._store_conversation(
                    query=query,
                    response=response_text,
                    session_id=current_session_id,
                    llm_used=llm_used
                )
                
                # Format model info for display
                model_display = f"{llm_used}"
                if model_name_used:
                    model_display = f"{model_display} ({model_name_used})"
                
                return {
                    "query": query,
                    "response": response_text,
                    "llm_used": llm_used,
                    "model_name": model_name_used,  # Include actual model name
                    "complexity": complexity,  # Include detected complexity
                    "success": True,
                    "session_id": current_session_id,
                    "context_retrieved": context_retrieved
                }
                
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"Error with {'fallback' if use_fallback else 'primary'} LLM: {e}")
                
                # Detect TPM / rate-limit errors (distinct from actual quota exhaustion)
                is_tpm_error = (
                    "tokens per min" in error_str or
                    "tokens per minute" in error_str or
                    "tpm" in error_str
                )

                if is_tpm_error:
                    current_model = model_name_used or self._current_openai_model or getattr(self.settings, "openai_model", "unknown")
                    logger.warning(f"TPM/rate-limit exceeded for model: {current_model}")
                    return {
                        "query": query,
                        "response": (
                            f"⚠️ **Rate Limit / TPM Exceeded**\n\n"
                            f"The request exceeded the tokens-per-minute (TPM) limit for `{current_model}`.\n\n"
                            f"**What to do:**\n"
                            f"- For large documents, the agent should use section extraction / the fast-path summarizer (not full text).\n"
                            f"- If you rerun the same request now, it should route into the large-document pipeline and avoid huge tool outputs.\n\n"
                            f"**Error:** {str(e)[:300]}..."
                        ),
                        "success": False,
                        "error": "tpm_rate_limit_exceeded",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": model_name_used if 'model_name_used' in locals() else current_model,
                        "complexity": complexity if 'complexity' in locals() else None
                    }

                # Detect quota exhaustion (429 errors / account quota)
                is_quota_error = (
                    "429" in str(e) or 
                    "quota" in error_str or 
                    "rate limit" in error_str or
                    "resourceexhausted" in error_str
                )
                
                if is_quota_error:
                    # Determine current model name based on LLM type
                    if model_name_used:
                        current_model = model_name_used
                    elif self.settings.primary_llm == "gemini" or (use_fallback and self.settings.fallback_llm == "gemini"):
                        current_model = self._current_gemini_model or "unknown"
                    elif self.settings.primary_llm == "openai" or (use_fallback and self.settings.fallback_llm == "openai"):
                        current_model = self._current_openai_model or getattr(self.settings, "openai_model", "unknown")
                    else:
                        current_model = "unknown"
                    
                    logger.warning(f"Quota exhausted for model: {current_model}")
                    
                    # Return helpful message instead of retrying
                    return {
                        "query": query,
                        "response": (
                            f"⚠️ **API Quota Exhausted**\n\n"
                            f"Your API quota for `{current_model}` has been exhausted.\n\n"
                            f"**Options:**\n"
                            f"1. **Wait for quota reset** - Quotas typically reset daily\n"
                            f"2. **Try a different model** - Adjust model settings in your .env file\n"
                            f"3. **Upgrade your plan** - Check your API provider's pricing page\n\n"
                            f"**Current model:** {current_model}\n"
                            f"**Error:** {str(e)[:200]}..."
                        ),
                        "success": False,
                        "error": "quota_exhausted",
                        "llm_used": "fallback" if use_fallback else "primary",
                        "model_name": model_name_used if 'model_name_used' in locals() else current_model,
                        "complexity": complexity if 'complexity' in locals() else None
                    }
                
                # Try fallback if primary failed (and it's not a quota error)
                # Check if Gemini fallback is disabled
                if not use_fallback:
                    if getattr(self.settings, 'disable_gemini_fallback', False) and self.settings.fallback_llm == "gemini":
                        logger.warning("⚠️ Fallback to Gemini is disabled. Using GPT models only.")
                        # Return error instead of falling back to Gemini
                        return {
                            "query": query,
                            "response": (
                                f"❌ **Error with primary LLM ({self.settings.primary_llm})**\n\n"
                                f"Error: {str(e)[:500]}\n\n"
                                f"**Note:** Fallback to Gemini is disabled (DISABLE_GEMINI_FALLBACK=True). "
                                f"Please ensure your OpenAI API key is valid and configured.\n"
                                f"Check your .env file for OPENAI_API_KEY."
                            ),
                            "success": False,
                            "error": str(e),
                            "llm_used": "primary",
                            "model_name": model_name_used if 'model_name_used' in locals() else "unknown"
                        }
                    else:
                        logger.info(f"Attempting with fallback LLM ({self.settings.fallback_llm})")
                        return self.process_query(query, use_fallback=True)
                raise
        
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _extract_response(self, result: Dict) -> str:
        """
        Extract the final text response from the graph result.
        
        The result contains a list of messages. We want the last AI message
        that contains the final response to the user (not tool calls).
        
        Args:
            result: The result from invoking the LangGraph
            
        Returns:
            str: The extracted response text
        """
        messages = result.get("messages", [])
        
        # Find the last AI message with actual content (not just tool calls)
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                # Check if this message has text content
                if message.content and isinstance(message.content, str):
                    return message.content
                # Handle structured content (some models return lists)
                elif message.content and isinstance(message.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for item in message.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    if text_parts:
                        return "\n".join(text_parts)
        
        return "No response generated."


# ==============================================================================
# MODULE-LEVEL EXPORTS
# ==============================================================================

# When someone does "from agent import ...", these are available
__all__ = ["SurvyAIAgent", "SYSTEM_PROMPT"]
