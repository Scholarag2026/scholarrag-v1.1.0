"""Tests for seed papers schemas -- SeedExpandRequest, SeedExpandResult."""

import pytest
from pydantic import ValidationError


def test_seed_expand_request_dois_only():
    from app.schemas.seed_papers import SeedExpandRequest

    req = SeedExpandRequest(dois=["10.1000/abc", "10.2000/def"])
    assert len(req.dois) == 2
    assert req.titles == []


def test_seed_expand_request_titles_only():
    from app.schemas.seed_papers import SeedExpandRequest

    req = SeedExpandRequest(titles=["Machine Learning in Education"])
    assert len(req.titles) == 1
    assert req.dois == []


def test_seed_expand_request_both():
    from app.schemas.seed_papers import SeedExpandRequest

    req = SeedExpandRequest(
        dois=["10.1000/abc"],
        titles=["Some Paper Title"],
    )
    assert len(req.dois) == 1
    assert len(req.titles) == 1


def test_seed_expand_request_empty_raises():
    from app.schemas.seed_papers import SeedExpandRequest

    with pytest.raises(ValidationError):
        SeedExpandRequest(dois=[], titles=[])


def test_seed_expand_request_strips_whitespace():
    from app.schemas.seed_papers import SeedExpandRequest

    req = SeedExpandRequest(dois=["  10.1000/abc  ", "  ", "10.2000/def"])
    assert req.dois == ["10.1000/abc", "10.2000/def"]


def test_seed_expand_request_whitespace_only_titles_rejected():
    from app.schemas.seed_papers import SeedExpandRequest

    with pytest.raises(ValidationError):
        SeedExpandRequest(titles=["  ", "   "])


def test_seed_expand_result_valid():
    from app.schemas.paper import PaperData
    from app.schemas.seed_papers import SeedExpandResult

    papers = [
        PaperData(title="Related Paper A", source_api="semantic_scholar"),
        PaperData(title="Related Paper B", source_api="semantic_scholar"),
    ]
    result = SeedExpandResult(
        seeds_resolved=2,
        seeds_failed=["bad-doi-123"],
        expanded_papers=papers,
        total_expanded=2,
    )
    assert result.seeds_resolved == 2
    assert len(result.seeds_failed) == 1
    assert len(result.expanded_papers) == 2
    assert result.total_expanded == 2


def test_seed_expand_result_empty():
    from app.schemas.seed_papers import SeedExpandResult

    result = SeedExpandResult(
        seeds_resolved=0,
        seeds_failed=[],
        expanded_papers=[],
        total_expanded=0,
    )
    assert result.seeds_resolved == 0
    assert result.expanded_papers == []
