"""Routing tests: knowledge questions must not trigger CAD/PDF fast-paths."""

from __future__ import annotations

HISTORY_WITH_PDF = """
=== CONVERSATION CONTEXT (REFERENCE ONLY) ===
--- Exchange 1 ---
User: Replot 49-Model.pdf to out.dwg
Assistant: Opening survey plan...
--- End of History (reference only) ---

NOW, the user wants you to continue with this new request:
"""

KNOWLEDGE_Q = (
    "What are the earliest Survey equipments in use; "
    "explain the principle of resection and intersection as a Surveyor"
)

ESSAY_ASSISTANT = (
    "Assistant: The earliest surveying equipment were simple instruments...\n"
    "Principle of resection...\nPrinciple of intersection...\n"
    "If you want, I can also turn this into a well-structured essay."
)

SAVE_ESSAY_Q = (
    "Turn this into a well-structured essay and save it into 'essay1.docx' "
    "in the same folder as this"
)


def _agent():
    from types import SimpleNamespace

    from agent.agent import SurvyAIAgent

    agent = SurvyAIAgent.__new__(SurvyAIAgent)
    agent.settings = SimpleNamespace(pdf_survey_replot_enabled=True)
    return agent


def test_knowledge_question_not_pdf_fastpath():
    agent = _agent()
    routing = KNOWLEDGE_Q
    full = HISTORY_WITH_PDF + KNOWLEDGE_Q
    assert agent._classify_query_intent(routing) == "knowledge"
    assert agent._should_fastpath_pdf_survey_replot(full, routing) is False


def test_save_session_docx_request():
    agent = _agent()
    routing = SAVE_ESSAY_Q
    full = (
        "=== CONVERSATION CONTEXT ===\n"
        f"{ESSAY_ASSISTANT}\n"
        "NOW, the user wants you to continue with this new request:\n"
        + routing
    )
    assert agent._should_fastpath_save_session_docx(routing, full) is True
    content = agent._extract_assistant_content_for_docx_save(full)
    assert "resection" in content.lower()


def test_go_ahead_after_essay_offer():
    agent = _agent()
    routing = "Go ahead"
    full = (
        "=== CONVERSATION CONTEXT ===\n"
        f"{ESSAY_ASSISTANT}\n"
        "NOW, the user wants you to continue with this new request:\n"
        + routing
    )
    assert agent._should_fastpath_save_session_docx(routing, full) is True


def test_session_text_truncation_detected():
    agent = _agent()
    assert agent._session_text_looks_truncated("Some GIS workflow…[truncated]") is True
    assert agent._session_text_looks_truncated("Complete answer with TIN and Cut Fill.") is False


def test_resolve_docx_save_prefers_non_truncated():
    agent = _agent()
    truncated = "ArcGIS volume method using TIN…[truncated]"
    full = "ArcGIS volume method using TIN and Cut Fill with full workflow steps."
    query = (
        "=== CONVERSATION CONTEXT ===\n"
        f"Assistant: {truncated}\n"
        "NOW, the user wants you to continue with this new request:\n"
        + SAVE_ESSAY_Q
    )
    agent._get_full_assistant_response_from_session = lambda session_id=None: full  # type: ignore[method-assign]
    resolved = agent._resolve_docx_save_source_text(query)
    assert resolved == full
    assert "[truncated]" not in resolved


ADIBAWA_VOLUME_Q = (
    'Go to the files "C:\\Users\\USER\\Documents\\SPDC\\ADIBAWA WELL 13\\'
    "Adibawa Well 13A BORROWPIT-Contractor\\ADIBAWA WELL 13 BORROW PIT VOLUME "
    "COMPUTATION_074938\\csv_adibawa__020416_0411510.csv\" and "
    '"C:\\Users\\USER\\Documents\\SPDC\\ADIBAWA WELL 13\\Adibawa Well 13A '
    "BORROWPIT-Contractor\\ADIBAWA WELL 13 BORROW PIT VOLUME COMPUTATION_074938\\"
    'csv_Adi post_031618_062229.csv", create a copy each of both (as .csv files in '
    "the SurvyAI folder), both are PRE and POST data respectively. Now use these "
    "point features with E, N, Z to generate point features in ArcGIS pro for PRE "
    "and for POST, use these point features to then create TIN for the PRE and POST "
    "surfaces (use Z as the input, while the extent and boundaries should be the "
    "Polygon from the file "
    '"C:\\Users\\USER\\Documents\\SPDC\\ADIBAWA WELL 13\\Adibawa Well 13A '
    "BORROWPIT-Contractor\\REVIEW\\POST SURVEY FOR ADIBAWA WELL 13 BORROW PIT 13A "
    'RESTORATION.dwg"). Then calculate the volume between the PRE and POST TIN '
    "using the CutFill tool (all on ArcGIS pro). Give the volume as an exported "
    "result to file 'Adibawa_VolumeResult.csv' in the workspace."
)


def test_gis_volume_prompt_not_save_session_docx():
    """PRE/POST CSV + DWG + TIN/CutFill must not route to essay-save fast path."""
    agent = _agent()
    routing = ADIBAWA_VOLUME_Q
    full = (
        "=== CONVERSATION CONTEXT ===\n"
        f"{ESSAY_ASSISTANT}\n"
        "NOW, the user wants you to continue with this new request:\n"
        + routing
    )
    assert agent._classify_query_intent(routing) == "task"
    assert agent._should_fastpath_save_session_docx(routing, full) is False


def test_unverified_completion_detected_without_tools():
    agent = _agent()
    fake_response = (
        "Saved essay to Word document.\n"
        "- Output: C:\\SurvyAI\\essay1.docx\n"
    )
    assert agent._response_looks_like_unverified_task_completion(
        ADIBAWA_VOLUME_Q, fake_response, tools_used=False
    ) is True
    assert agent._response_looks_like_unverified_task_completion(
        ADIBAWA_VOLUME_Q, fake_response, tools_used=True
    ) is False


CRS_ASSISTANT = (
    "Assistant: Done — I redid the conversion.\n"
    "Verified output: Converted_points.xlsx\n"
    "EPSG code used for the target CRS: EPSG:26392\n"
    "If you want, I can next try to retrieve the exact transformation name/details "
    "used by the CRS definition and include that in a note.\n"
)


def test_yes_resolves_to_last_crs_offer_not_volume():
    agent = _agent()
    routing = "yes"
    full = (
        "=== CONVERSATION CONTEXT ===\n"
        f"{CRS_ASSISTANT}\n"
        "NOW, the user wants you to continue with this new request:\n"
        + routing
    )
    resolved = agent._resolve_affirmative_to_last_offer(full, routing)
    assert resolved is not None
    assert "transformation" in resolved.lower() or "retrieve" in resolved.lower()
    assert "proceed with only that offer" in resolved.lower()


def test_bare_yes_suppresses_vector_rag_route():
    agent = _agent()
    decision = agent._decide_rag_route("yes")
    assert decision.use_vector is False
    assert decision.use_internet is False
    assert decision.route == "llm_only"
