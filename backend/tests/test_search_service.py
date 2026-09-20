# backend/tests/test_search_service.py
import inspect
from unittest.mock import patch

import pytest

from app.schemas.paper import PaperData
from app.services.search import SearchService


def _make_paper(doi: str | None, title: str, source: str = "openalex", citations: int = 10):
    return PaperData(
        doi=doi,
        title=title,
        authors=[{"name": "Test Author"}],
        year=2023,
        citation_count=citations,
        source_api=source,
    )


@pytest.mark.asyncio
async def test_search_returns_openalex_results():
    oa_results = [
        _make_paper("10.1/a", "Paper A"),
        _make_paper("10.1/b", "Paper B"),
        _make_paper("10.1/c", "Paper C"),
    ]

    with patch.object(SearchService, "_search_openalex", return_value=oa_results):
        svc = SearchService()
        papers, sources = await svc.search("test query")

    assert len(papers) == 3
    assert sources == {"openalex": 3}


@pytest.mark.asyncio
async def test_search_deduplicates_by_doi():
    oa_results = [
        _make_paper("10.1/same", "Paper Same", citations=50),
        _make_paper("10.1/SAME", "Paper Same Other Casing", citations=45),
        _make_paper("10.1/unique", "Paper Unique", citations=20),
    ]

    with patch.object(SearchService, "_search_openalex", return_value=oa_results):
        svc = SearchService()
        papers, _ = await svc.search("test query")

    assert len(papers) == 2
    dois = {p.doi.lower() for p in papers}
    assert dois == {"10.1/same", "10.1/unique"}


@pytest.mark.asyncio
async def test_search_keeps_higher_citation_on_dedup():
    oa_results = [
        _make_paper("10.1/dup", "Low version", citations=50),
        _make_paper("10.1/dup", "High version", citations=100),
    ]

    with patch.object(SearchService, "_search_openalex", return_value=oa_results):
        svc = SearchService()
        papers, _ = await svc.search("test")

    assert len(papers) == 1
    assert papers[0].citation_count == 100


@pytest.mark.asyncio
async def test_search_sorts_by_citations():
    oa_results = [
        _make_paper("10.1/low", "Low", citations=5),
        _make_paper("10.1/high", "High", citations=500),
        _make_paper("10.1/mid", "Mid", citations=50),
    ]

    with patch.object(SearchService, "_search_openalex", return_value=oa_results):
        svc = SearchService()
        papers, _ = await svc.search("test")

    assert papers[0].citation_count == 500
    assert papers[1].citation_count == 50
    assert papers[2].citation_count == 5


@pytest.mark.asyncio
async def test_search_handles_api_failure_gracefully():
    with patch.object(
        SearchService, "_search_openalex", side_effect=Exception("OpenAlex down")
    ):
        svc = SearchService()
        papers, sources = await svc.search("test")

    assert papers == []
    assert sources == {"openalex": 0}


@pytest.mark.asyncio
async def test_search_raise_on_failure_propagates_the_provider_error():
    """Callers that must distinguish 'no results' from 'provider down' can opt out."""
    with patch.object(
        SearchService, "_search_openalex", side_effect=RuntimeError("OpenAlex down")
    ):
        svc = SearchService()
        with pytest.raises(RuntimeError, match="OpenAlex down"):
            await svc.search("test", raise_on_failure=True)


@pytest.mark.asyncio
async def test_search_passes_filters_through():
    with patch.object(SearchService, "_search_openalex", return_value=[]) as mock_oa:
        svc = SearchService()
        await svc.search("test", year_from=2020, year_to=2024, min_citations=10)

    assert mock_oa.await_args.kwargs == {
        "year_from": 2020,
        "year_to": 2024,
        "min_citations": 10,
        "to_publication_date": None,
    }


@pytest.mark.asyncio
async def test_search_passes_to_publication_date_through():
    """Forwarded to the OpenAlex client exactly like the other filters."""
    from datetime import date

    with patch.object(SearchService, "_search_openalex", return_value=[]) as mock_oa:
        svc = SearchService()
        await svc.search("test", to_publication_date=date(2026, 9, 8))

    assert mock_oa.await_args.kwargs["to_publication_date"] == date(2026, 9, 8)


def test_search_service_is_openalex_only():
    """S2 and CORE legs are gone from the search fan-out."""
    from app.services import search as search_module

    source = inspect.getsource(search_module)
    assert "semantic_scholar" not in source
    assert "SemanticScholar" not in source
    assert "core_api" not in source
    assert "CoreClient" not in source
    assert not hasattr(SearchService, "_search_semantic_scholar")
    assert not hasattr(SearchService, "_search_core")
