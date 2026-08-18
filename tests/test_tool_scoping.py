"""Tool/prompt pack scoping — lite vs full (no live LLM)."""

from types import SimpleNamespace

from survyai.tool_scoping import select_tool_scope


def test_knowledge_without_cad_is_lite_even_if_average():
    assert (
        select_tool_scope(
            "What is the history of the theodolite?",
            complexity="average",
            intent="knowledge",
        )
        == "lite"
    )


def test_cad_markers_stay_full():
    assert (
        select_tool_scope(
            "Buffer the polygons in ArcGIS and export a DWG",
            complexity="simple",
            intent="task",
        )
        == "full"
    )


def test_file_driven_stays_full():
    assert (
        select_tool_scope(
            "Summarize this report",
            complexity="simple",
            intent="knowledge",
            file_driven=True,
        )
        == "full"
    )


def test_internet_lookup_stays_full():
    action = SimpleNamespace(kind="current_fact_lookup", needs_internet=True, needs_tools=False)
    assert (
        select_tool_scope(
            "Who is the current surveyor-general?",
            complexity="simple",
            intent="knowledge",
            prompt_action=action,
        )
        == "full"
    )
