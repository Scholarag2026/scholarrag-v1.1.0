"""Tests for the deep search service — normalize_title, dedup, _add_papers, coverage metrics."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.schemas.paper import PaperData


def _make_paper(
    doi: str | None,
    title: str,
    source: str = "openalex",
    citations: int = 10,
    abstract: str | None = None,
) -> PaperData:
    return PaperData(
        doi=doi,
        title=title,
        authors=[{"name": "Test Author"}],
        year=2023,
        citation_count=citations,
        source_api=source,
        abstract=abstract,
    )


# ---------- normalize_title tests ----------


def test_normalize_title_lowercases():
    from app.services.deep_search import normalize_title

    assert normalize_title("Machine Learning In Education") == "machine learning in education"


def test_normalize_title_strips_punctuation():
    from app.services.deep_search import normalize_title

    assert normalize_title("A Study: Results & Analysis!") == "a study results  analysis"


def test_normalize_title_strips_whitespace():
    from app.services.deep_search import normalize_title

    assert normalize_title("  Some Title  ") == "some title"


# ---------- _is_new_paper tests ----------


def test_is_new_paper_doi_dedup():
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()
    p1 = _make_paper("10.1/abc", "Paper A")
    p2 = _make_paper("10.1/ABC", "Paper A Different Title")  # same DOI, different case

    assert pipeline._is_new_paper(p1) is True
    assert pipeline._is_new_paper(p2) is False


def test_is_new_paper_title_fallback():
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()
    p1 = _make_paper(None, "Machine Learning in Education")
    p2 = _make_paper(None, "machine learning in education")  # same title, no DOI

    assert pipeline._is_new_paper(p1) is True
    assert pipeline._is_new_paper(p2) is False


def test_is_new_paper_different_doi_same_normalized_title():
    """Different DOIs but same normalized title should still be deduped via title."""
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()
    p1 = _make_paper("10.1/first", "Same Title")
    p2 = _make_paper("10.2/second", "same title")

    assert pipeline._is_new_paper(p1) is True
    assert pipeline._is_new_paper(p2) is False


# ---------- _add_papers tests ----------


def test_add_papers_counts_new_and_tracks_sources():
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()
    papers = [
        _make_paper("10.1/a", "Paper A", "openalex"),
        _make_paper("10.1/b", "Paper B", "openalex"),
        _make_paper("10.1/a", "Paper A Dup", "openalex"),  # duplicate DOI
    ]

    new_count = pipeline._add_papers(papers)
    assert new_count == 2
    assert len(pipeline.all_papers) == 2
    assert pipeline.total_scanned == 3
    assert pipeline.sources["openalex"] == 2


def test_add_papers_across_multiple_rounds():
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()

    round1 = [
        _make_paper("10.1/a", "Paper A", "openalex"),
        _make_paper("10.1/b", "Paper B", "openalex"),
    ]
    round2 = [
        _make_paper("10.1/a", "Paper A again", "openalex"),  # dup from round 1
        _make_paper("10.1/c", "Paper C", "openalex"),  # new
    ]

    n1 = pipeline._add_papers(round1)
    n2 = pipeline._add_papers(round2)

    assert n1 == 2
    assert n2 == 1
    assert len(pipeline.all_papers) == 3
    assert pipeline.total_scanned == 4


# ---------- build_coverage_metrics tests ----------


def test_build_coverage_metrics():
    from app.schemas.deep_search import RoundMetrics
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()

    # Simulate round 1
    pipeline._add_papers([
        _make_paper("10.1/a", "Paper A", "openalex"),
        _make_paper("10.1/b", "Paper B", "openalex"),
    ])
    pipeline.round_metrics.append(
        RoundMetrics(round=1, strategy="initial_search", new_papers=2, queries=["test query"])
    )

    # Simulate round 2
    pipeline._add_papers([
        _make_paper("10.1/c", "Paper C", "openalex"),
    ])
    pipeline.round_metrics.append(
        RoundMetrics(round=2, strategy="query_expansion", new_papers=1, queries=["expanded query"])
    )

    metrics = pipeline.build_coverage_metrics()

    assert metrics.total_scanned == 3
    assert metrics.total_unique == 3
    assert len(metrics.rounds) == 2
    assert metrics.yield_curve == [2, 1]
    assert metrics.sources["openalex"] == 3
    assert metrics.confidence in ("high", "medium", "low")


def test_build_coverage_metrics_confidence_levels():
    """Verify confidence level computation based on total unique papers."""
    from app.services.deep_search import DeepSearchPipeline

    pipeline = DeepSearchPipeline()

    # With 0 papers -> low
    metrics = pipeline.build_coverage_metrics()
    assert metrics.confidence == "low"

    # Add enough papers for "medium" (>= 10)
    for i in range(15):
        pipeline._add_papers([_make_paper(f"10.1/{i}", f"Paper {i}", "openalex")])
    metrics = pipeline.build_coverage_metrics()
    assert metrics.confidence == "medium"

    # Add more for "high" (>= 30)
    for i in range(15, 35):
        pipeline._add_papers([_make_paper(f"10.1/{i}", f"Paper {i}", "openalex")])
    metrics = pipeline.build_coverage_metrics()
    assert metrics.confidence == "high"
