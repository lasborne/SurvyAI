"""
LangGraph state and RAG routing types for the SurvyAI agent.
"""

import operator
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class RAGRouteDecision(BaseModel):
    """
    Routing decision for Agentic RAG:
    - llm_only: no retrieval/search augmentation
    - vector: retrieve from local vector store
    - internet: web search (permissioned)
    - hybrid: both vector + internet
    """
    route: Literal["llm_only", "vector", "internet", "hybrid"] = Field(
        "llm_only", description="Chosen route"
    )
    use_vector: bool = Field(False, description="Whether to retrieve local context")
    vector_collections: List[str] = Field(
        default_factory=list,
        description="Collections to prioritize (documents/drawings/coordinates/conversations)",
    )
    use_internet: bool = Field(False, description="Whether to run an internet search (permissioned)")
    internet_query: Optional[str] = Field(None, description="Suggested web search query")
    reason: str = Field("", description="Short reason for routing choice")


def looks_like_file_driven_task(query: str) -> bool:
    """True if the query mentions file types or paths (documents, CAD, etc.)."""
    q = query or ""
    ql = q.lower()
    if any(ext in ql for ext in [".docx", ".pdf", ".xlsx", ".xls", ".dwg", ".dxf", ".csv", ".shp", ".aprx"]):
        return True
    if "\\Users\\" in q or ":\\" in q:
        return True
    return False


class AgentState(TypedDict):
    """
    State that flows through the LangGraph.
    messages: conversation history; Annotated with operator.add so new messages are appended.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
