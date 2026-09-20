"""Tests for deep search schemas — DeepSearchRequest, RoundMetrics, CoverageMetrics, QueryExpansionResult."""

import pytest
from pydantic import ValidationError


def test_deep_search_request_valid():
    from app.schemas.deep_search import DeepSearchRequest

    req = DeepSearchRequest(query="machine learning in education")
    assert req.query == "machine learning in education"
    assert req.year_from is None
    assert req.year_to is None
    assert req.min_citations is None
    assert req.max_rounds == 6


def test_deep_search_request_all_fields():
    from app.schemas.deep_search import DeepSearchRequest

    req = DeepSearchRequest(
        query="EdTech higher education",
        year_from=2018,
        year_to=2024,
        min_citations=5,
        max_rounds=3,
    )
    assert req.year_from == 2018
    assert req.year_to == 2024
    assert req.min_citations == 5
    assert req.max_rounds == 3


def test_deep_search_request_empty_query_rejected():
    from app.schemas.deep_search import DeepSearchRequest

    with pytest.raises(ValidationError):
        DeepSearchRequest(query="")


def test_deep_search_request_too_long_query_rejected():
    from app.schemas.deep_search import DeepSearchRequest

    with pytest.raises(ValidationError):
        DeepSearchRequest(query="x" * 501)


def test_round_metrics_defaults():
    from app.schemas.deep_search import RoundMetrics

    rm = RoundMetrics(round=1, strategy="initial_search", new_papers=15)
    assert rm.round == 1
    assert rm.strategy == "initial_search"
    assert rm.new_papers == 15
    assert rm.queries == []
    assert rm.papers_expanded is None
    assert rm.stopped is False


def test_round_metrics_full():
    from app.schemas.deep_search import RoundMetrics

    rm = RoundMetrics(
        round=3,
        strategy="query_expansion",
        new_papers=8,
        queries=["query A", "query B"],
        papers_expanded=5,
        stopped=True,
    )
    assert rm.strategy == "query_expansion"
    assert len(rm.queries) == 2
    assert rm.papers_expanded == 5
    assert rm.stopped is True


def test_coverage_metrics_valid():
    from app.schemas.deep_search import CoverageMetrics, RoundMetrics

    rounds = [
        RoundMetrics(round=1, strategy="initial_search", new_papers=20),
        RoundMetrics(round=2, strategy="query_expansion", new_papers=10, queries=["alt query"]),
    ]
    cm = CoverageMetrics(
        total_scanned=50,
        total_unique=30,
        rounds=rounds,
        sources={"openalex": 25, "semantic_scholar": 20, "crossref": 5},
        yield_curve=[20, 10],
        confidence="high",
    )
    assert cm.total_scanned == 50
    assert cm.total_unique == 30
    assert len(cm.rounds) == 2
    assert cm.sources["openalex"] == 25
    assert cm.yield_curve == [20, 10]
    assert cm.confidence == "high"


def test_coverage_metrics_empty_rounds():
    from app.schemas.deep_search import CoverageMetrics

    cm = CoverageMetrics(
        total_scanned=0,
        total_unique=0,
        rounds=[],
        sources={},
        yield_curve=[],
        confidence="low",
    )
    assert cm.total_scanned == 0
    assert len(cm.rounds) == 0
    assert cm.confidence == "low"


def test_query_expansion_result_valid():
    from app.schemas.deep_search import QueryExpansionResult

    result = QueryExpansionResult(queries=["query 1", "query 2", "query 3"])
    assert len(result.queries) == 3
    assert result.queries[0] == "query 1"


def test_query_expansion_result_empty():
    from app.schemas.deep_search import QueryExpansionResult

    result = QueryExpansionResult(queries=[])
    assert result.queries == []
