"""Tests for DWG plan extract fast-path (no AutoCAD required)."""

from agent.pdf_survey_plan import (
    resolve_dwg_paths_from_query,
    resolve_dwg_extract_output_docx_path,
    should_fastpath_dwg_plan_extract_to_docx,
    parse_dwg_metadata_from_tables,
)


USER_QUERY = (
    'Go to this file "C:\\Users\\USER\\Documents\\NWUNNE_FORTUNE.dwg", '
    "JONATHAN_ODIGIE.dwg, MR.IKECHUKWU_OLEKA.dwg & KELECHI_SUSAN_AKAEZE.dwg "
    "(all in the same folder), extract all the important details in the plan "
    "and save it in arranged order in a new Microsoft word document "
    "'Plan_details_Extract' (in my current workspace)"
)


def test_should_fastpath_user_query():
    assert should_fastpath_dwg_plan_extract_to_docx(USER_QUERY) is True


def test_should_not_fastpath_replot():
    q = "Extract from plan.pdf and replot as 'out.dwg'"
    assert should_fastpath_dwg_plan_extract_to_docx(q) is False


def test_resolve_output_docx_workspace():
    out = resolve_dwg_extract_output_docx_path(USER_QUERY, workspace=__import__("pathlib").Path("C:/workspace"))
    assert out.name == "Plan_details_Extract.docx"
    assert str(out.parent).endswith("workspace")


def test_resolve_user_query_four_plans():
    paths = resolve_dwg_paths_from_query(USER_QUERY)
    names = {__import__("pathlib").Path(p).name for p in paths}
    assert names == {
        "NWUNNE_FORTUNE.dwg",
        "JONATHAN_ODIGIE.dwg",
        "MR.IKECHUKWU_OLEKA.dwg",
        "KELECHI_SUSAN_AKAEZE.dwg",
    }


def test_resolve_dwg_paths_ignores_ampersand_merge():
    q = "MR.IKECHUKWU_OLEKA.dwg & KELECHI_SUSAN_AKAEZE.dwg"
    names = [__import__("pathlib").Path(p).name for p in resolve_dwg_paths_from_query(q)]
    assert names == ["MR.IKECHUKWU_OLEKA.dwg", "KELECHI_SUSAN_AKAEZE.dwg"]


def test_resolve_dwg_paths_includes_bare_names(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for name in ("NWUNNE_FORTUNE.dwg", "JONATHAN_ODIGIE.dwg"):
        (folder / name).write_bytes(b"stub")

    q = f'Open "{folder / "NWUNNE_FORTUNE.dwg"}", JONATHAN_ODIGIE.dwg — extract details to Word'
    paths = resolve_dwg_paths_from_query(q)
    names = {__import__("pathlib").Path(p).name for p in paths}
    assert "NWUNNE_FORTUNE.dwg" in names
    assert "JONATHAN_ODIGIE.dwg" in names


def test_clean_autocad_mtext_jonathan_title_block():
    from agent.pdf_survey_plan import (
        clean_autocad_mtext,
        parse_dwg_title_block_fields,
        extract_survey_plan_from_dwg_layout,
    )

    raw = (
        r"{\fVerdana|b0|i0|c0|p34;\H0.9333x;PLAN SHEWING LANDED PROPERTY\POF\P"
        r"\fVerdana|b1|i0|c0|p34;\H1.286x;JONATHAN ODIGIE"
        r"\fVerdana|b0|i0|c0|p34;\H0.8333x;\P\H0.9334x;AT\POHIA IZOR MINI NKPUKPA"
        r"\P\fVerdana|b1|i0|c0|p34;MGBUCHI COMMUNITY, RUKPOKWU"
        r"\fVerdana|b0|i0|c0|p34;\POBIO/AKPOR LOCAL GOVERNMENT AREA"
        r"\PRIVERS  STATE, NIGERIA\PSCALE:- 1:500\PORIGIN:- UTM ZONE 32N\P\C7;AREA:- 472.82 SQ. MTRS}"
    )
    cleaned = clean_autocad_mtext(raw)
    assert "JONATHAN ODIGIE" in cleaned
    assert "472.82" in cleaned
    fields = parse_dwg_title_block_fields(cleaned)
    assert "JONATHAN ODIGIE" in fields.get("buyer_name", "")
    assert fields.get("area_sq_m") == "472.82"

    plan = extract_survey_plan_from_dwg_layout(cleaned, file_stem="JONATHAN_ODIGIE")
    assert "JONATHAN" in (plan.buyer_name or "").upper()
    assert plan.area_sq_m is not None and abs(plan.area_sq_m - 472.82) < 0.1


def test_parse_metadata_from_tables():
    tables = [
        {
            "grid": [
                ["Buyer Name", "NWUNNE FORTUNE"],
                ["Location", "Abuja"],
                ["Plan Number", "AB/1234"],
            ]
        }
    ]
    meta = parse_dwg_metadata_from_tables(tables)
    assert meta.get("buyer_name") == "NWUNNE FORTUNE"
    assert meta.get("location") == "Abuja"
    assert meta.get("plan_number") == "AB/1234"
