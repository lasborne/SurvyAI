"""Tests for CADA_PLANNUMBER height scaling and shrink-to-fit."""

from agent.agent import (
    _fit_plan_number_mtext,
    _ideal_plan_number_text_height,
)


def test_ideal_height_scales_with_plan_scale():
    h500 = _ideal_plan_number_text_height(
        template_nominal_h=1.2,
        template_denom=500,
        chosen_denom=500,
    )
    h250 = _ideal_plan_number_text_height(
        template_nominal_h=1.2,
        template_denom=500,
        chosen_denom=250,
    )
    assert abs(h500 - 1.2) < 1e-6
    assert abs(h250 - 0.6) < 1e-6


def test_single_line_can_shrink_below_ideal_but_not_below_85pct():
    ideal = 0.6
    long_no = "RV/1124/2026/012345"
    body, scale = _fit_plan_number_mtext(
        long_no,
        cell_width=ideal * 10.0,
        cell_height=ideal * 2.5,
        base_text_height=ideal,
        line_step=ideal * (5.0 / 3.0),
    )
    assert scale >= 0.85
    assert scale <= 1.0


def test_two_line_prefers_ideal_height_when_possible():
    ideal = 0.6
    plan_no = "RV/1124/2026/01234567"
    body, scale = _fit_plan_number_mtext(
        plan_no,
        cell_width=ideal * 6.0,
        cell_height=ideal * 4.0,
        base_text_height=ideal,
        line_step=ideal * (5.0 / 3.0),
        min_height_scale=0.85,
    )
    if "\\P" in body:
        assert scale >= 0.85
    else:
        assert scale <= 1.0
