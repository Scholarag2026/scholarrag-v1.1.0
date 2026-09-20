"""Tests for analysis schemas — PaperAnalysis, GapReport, API schemas."""

import pytest
from pydantic import ValidationError


def test_paper_analysis_valid():
    from app.schemas.analysis import PaperAnalysis

    pa = PaperAnalysis(
        paper_id="550e8400-e29b-41d4-a716-446655440000",
        relevance_score=0.8,
        quality_score=0.75,
        key_findings=["Finding 1", "Finding 2"],
        methodology="Mixed methods with survey and interviews",
        methodology_rigor="high",
        limitations=["Small sample size"],
        theories_used=["Social Constructivism"],
        sample_info="N=120 university students",
    )
    assert pa.quality_score == 0.75
    assert pa.methodology_rigor == "high"
    assert len(pa.key_findings) == 2


def test_paper_analysis_score_bounds():
    from app.schemas.analysis import PaperAnalysis

    with pytest.raises(ValidationError):
        PaperAnalysis(
            paper_id="550e8400-e29b-41d4-a716-446655440000",
            relevance_score=1.5,  # out of range
            quality_score=0.5,
            key_findings=["Finding"],
            methodology="Survey",
            methodology_rigor="high",
            limitations=[],
            theories_used=[],
        )


def test_paper_analysis_empty_findings_rejected():
    from app.schemas.analysis import PaperAnalysis

    with pytest.raises(ValidationError):
        PaperAnalysis(
            paper_id="550e8400-e29b-41d4-a716-446655440000",
            relevance_score=0.5,
            quality_score=0.5,
            key_findings=[],  # min_length=1
            methodology="Survey",
            methodology_rigor="high",
            limitations=[],
            theories_used=[],
        )


def test_paper_analysis_invalid_rigor():
    from app.schemas.analysis import PaperAnalysis

    with pytest.raises(ValidationError):
        PaperAnalysis(
            paper_id="550e8400-e29b-41d4-a716-446655440000",
            relevance_score=0.5,
            quality_score=0.5,
            key_findings=["Finding"],
            methodology="Survey",
            methodology_rigor="excellent",  # invalid literal
            limitations=[],
            theories_used=[],
        )


def test_batch_analysis_result():
    from app.schemas.analysis import BatchAnalysisResult, PaperAnalysis

    pa = PaperAnalysis(
        paper_id="550e8400-e29b-41d4-a716-446655440000",
        relevance_score=0.8,
        quality_score=0.7,
        key_findings=["F1"],
        methodology="Survey",
        methodology_rigor="medium",
        limitations=[],
        theories_used=[],
    )
    batch = BatchAnalysisResult(analyses=[pa])
    assert len(batch.analyses) == 1


def test_research_gap_valid():
    from app.schemas.analysis import ResearchGap

    gap = ResearchGap(
        gap_type="unexplored",
        description="No studies on X in context Y",
        evidence=["Paper A found...", "Paper B noted..."],
        severity="high",
    )
    assert gap.gap_type == "unexplored"
    assert gap.severity == "high"


def test_research_gap_invalid_type():
    from app.schemas.analysis import ResearchGap

    with pytest.raises(ValidationError):
        ResearchGap(
            gap_type="missing",  # invalid literal
            description="desc",
            evidence=[],
            severity="high",
        )


def test_position_and_controversy():
    from app.schemas.analysis import Controversy, Position

    pos = Position(view="Technology enhances learning", supporters=["Smith (2023)", "Jones (2022)"])
    c = Controversy(topic="EdTech effectiveness", positions=[pos])
    assert c.positions[0].view == "Technology enhances learning"
    assert len(c.positions[0].supporters) == 2


def test_suggested_question():
    from app.schemas.analysis import SuggestedQuestion

    sq = SuggestedQuestion(
        question="How does X affect Y in context Z?",
        rationale="No studies have examined this relationship",
        methodology_hint="Mixed methods with pre/post survey",
        related_gap_descriptions=["No studies on X in context Y"],
    )
    assert "How does" in sq.question


def test_gap_report_complete():
    from app.schemas.analysis import (
        Controversy,
        GapReport,
        Position,
        ResearchGap,
        SuggestedQuestion,
    )

    report = GapReport(
        gaps=[ResearchGap(
            gap_type="under_explored", description="d",
            evidence=["e"], severity="medium",
        )],
        controversies=[Controversy(topic="t", positions=[Position(view="v", supporters=["s"])])],
        suggested_questions=[
            SuggestedQuestion(
                question="q", rationale="r", methodology_hint="m", related_gap_descriptions=["d"]
            )
        ],
        theoretical_landscape=["Social Learning Theory"],
        summary="The field shows several gaps...",
    )
    assert len(report.gaps) == 1
    assert len(report.controversies) == 1
    assert report.theoretical_landscape[0] == "Social Learning Theory"


def test_quality_scoring_request_default():
    from app.schemas.analysis import QualityScoringRequest

    req = QualityScoringRequest()
    assert req.force is False


def test_quality_scoring_request_force():
    from app.schemas.analysis import QualityScoringRequest

    req = QualityScoringRequest(force=True)
    assert req.force is True


def test_paper_analysis_response():
    from app.schemas.analysis import PaperAnalysisResponse

    resp = PaperAnalysisResponse(
        id="550e8400-e29b-41d4-a716-446655440000",
        project_id="550e8400-e29b-41d4-a716-446655440001",
        paper_id="550e8400-e29b-41d4-a716-446655440002",
        quality_score=0.8,
        relevance_score=0.7,
        key_findings=["F1"],
        methodology="Survey",
        methodology_rigor="high",
        limitations=["L1"],
        theories_used=["T1"],
        sample_info="N=50",
        paper_title="Test Paper",
        paper_year=2024,
        paper_doi="10.1234/test",
    )
    assert resp.quality_score == 0.8
    assert resp.paper_title == "Test Paper"
