"""
SurvyAI product layer: service façade, capabilities, flags, and CLI helpers.

Core agent logic remains under `agent/`; this package is the stable surface for
desktop GUI and packaging (Phase 1).
"""

from survyai.version import __version__
from survyai.agent_service import SurvyAIAgentService
from survyai.types import AgentRunResult
from survyai.feature_flags import (
    FeatureFlags,
    can_use_integration,
    categorize_tool_for_license,
)
from survyai.capabilities import (
    MachineCapabilities,
    scan_machine_capabilities,
    format_capabilities_summary,
)
from survyai.app_config import merge_settings

__all__ = [
    "__version__",
    "SurvyAIAgentService",
    "AgentRunResult",
    "FeatureFlags",
    "can_use_integration",
    "categorize_tool_for_license",
    "MachineCapabilities",
    "scan_machine_capabilities",
    "format_capabilities_summary",
    "merge_settings",
]
