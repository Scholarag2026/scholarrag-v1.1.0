"""Tests for the Quality Scoring Agent and Gap Analysis Agent."""

import os

# Set dummy API key so Agent() constructors don't fail at import time
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")



def test_analysis_agent_exists():
    from app.agents.analysis_agent import AnalysisDependencies, get_analysis_agent
    assert get_analysis_agent is not None
    assert AnalysisDependencies is not None


def test_format_papers_for_prompt():
    from app.agents.analysis_agent import format_papers_for_prompt

    papers = [
        {
            "id": "paper-1",
            "title": "Impact of Technology on Learning",
            "authors": [{"name": "Smith, J."}],
            "year": 2023,
            "journal_name": "Journal of Education",
            "is_wos_indexed": True,
            "citation_count": 45,
            "abstract": "This study examines...",
        }
    ]
    result = format_papers_for_prompt(papers)
    assert "Paper 1:" in result
    assert "Impact of Technology on Learning" in result
    assert "Smith, J." in result
    assert "WoS indexed: yes" in result
    assert "Citation count: 45" in result


def test_format_papers_missing_abstract():
    from app.agents.analysis_agent import format_papers_for_prompt

    papers = [
        {
            "id": "paper-1",
            "title": "A Paper",
            "authors": [],
            "year": None,
            "journal_name": None,
            "is_wos_indexed": False,
            "citation_count": None,
            "abstract": None,
        }
    ]
    result = format_papers_for_prompt(papers)
    assert "Paper 1:" in result
    assert "Abstract: Not available" in result


def test_analysis_dependencies_no_db():
    from app.agents.analysis_agent import AnalysisDependencies

    deps = AnalysisDependencies(
        project_id="test-id",
        project_description="Test project",
        target_journal="Nature",
        citation_style="APA",
    )
    assert deps.project_id == "test-id"
    assert not hasattr(deps, "db")


def test_gap_agent_exists():
    from app.agents.gap_agent import get_gap_agent
    assert get_gap_agent is not None


def test_format_analyses_for_gap_prompt():
    from app.agents.gap_agent import format_analyses_for_gap_prompt

    analyses = [
        {
            "paper_title": "Paper A",
            "paper_year": 2023,
            "quality_score": 0.8,
            "key_findings": ["Finding 1"],
            "methodology": "Survey",
            "methodology_rigor": "high",
            "limitations": ["Small sample"],
            "theories_used": ["TAM"],
        }
    ]
    result = format_analyses_for_gap_prompt(
        analyses,
        project_description="EdTech in higher education",
        target_journal="BJET",
    )
    assert "Paper A" in result
    assert "Finding 1" in result
    assert "EdTech in higher education" in result
    assert "BJET" in result
