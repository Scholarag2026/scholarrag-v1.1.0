"""Tests for qualitative coding schemas."""

import pytest
from pydantic import ValidationError


def test_theme_valid():
    from app.schemas.qualitative import Theme
    t = Theme(id="t1", name="Identity", description="Themes about identity", parent_id=None)
    assert t.id == "t1"
    assert t.parent_id is None


def test_theme_with_parent():
    from app.schemas.qualitative import Theme
    t = Theme(id="t2", name="Sub-identity", description="Sub-theme", parent_id="t1")
    assert t.parent_id == "t1"


def test_code_valid():
    from app.schemas.qualitative import Code
    c = Code(id="c1", name="Self-concept", description="References to self", theme_id="t1", color="#FF0000")
    assert c.theme_id == "t1"
    assert c.color == "#FF0000"


def test_codebook_schema():
    from app.schemas.qualitative import Code, CodebookSchema, Theme
    cb = CodebookSchema(
        codes=[Code(id="c1", name="Self-concept", description="d", theme_id="t1", color="#FF0000")],
        themes=[Theme(id="t1", name="Identity", description="d")],
    )
    assert len(cb.codes) == 1
    assert len(cb.themes) == 1


def test_coding_segment_valid():
    from app.schemas.qualitative import CodingSegment
    seg = CodingSegment(
        segment_id="s1", text="I feel like myself", codes=["c1", "c2"],
        confidence="high", reasoning="Strong self-reference",
    )
    assert seg.confidence == "high"
    assert len(seg.codes) == 2


def test_coding_segment_invalid_confidence():
    from app.schemas.qualitative import CodingSegment
    with pytest.raises(ValidationError):
        CodingSegment(
            segment_id="s1", text="text", codes=["c1"],
            confidence="very_high", reasoning="r",
        )


def test_uncertainty():
    from app.schemas.qualitative import Uncertainty
    u = Uncertainty(segment_id="s1", issue="Ambiguous reference", suggested_codes=["c1", "c2"])
    assert len(u.suggested_codes) == 2


def test_coding_result_complete():
    from app.schemas.qualitative import (
        Code, CodebookSchema, CodingResult, CodingSegment, Theme, Uncertainty,
    )
    result = CodingResult(
        codebook=CodebookSchema(
            codes=[Code(id="c1", name="n", description="d", theme_id="t1", color="#000")],
            themes=[Theme(id="t1", name="n", description="d")],
        ),
        coded_segments=[CodingSegment(
            segment_id="s1", text="txt", codes=["c1"], confidence="medium", reasoning="r",
        )],
        uncertainties=[Uncertainty(segment_id="s2", issue="unclear", suggested_codes=["c1"])],
        annotation_notes=["Note 1"],
    )
    assert len(result.coded_segments) == 1
    assert len(result.uncertainties) == 1
    assert result.annotation_notes == ["Note 1"]


def test_coding_session_update():
    from app.schemas.qualitative import CodingSegment, CodingSessionUpdate
    update = CodingSessionUpdate(
        coded_segments=[
            CodingSegment(segment_id="s1", text="t", codes=["c1"], confidence="low", reasoning="r"),
        ],
    )
    assert len(update.coded_segments) == 1


def test_inter_coder_report():
    from app.schemas.qualitative import InterCoderReport, KappaScore
    report = InterCoderReport(
        overall_kappa=0.85,
        per_code_kappa=[KappaScore(code_name="c1", kappa=0.9, agreement_pct=95.0)],
        agreement_pct=90.0,
        disagreement_segments=[{"segment_id": "s3", "ai_codes": ["c1"], "human_codes": ["c2"]}],
    )
    assert report.overall_kappa == 0.85
    assert len(report.per_code_kappa) == 1
    assert len(report.disagreement_segments) == 1
