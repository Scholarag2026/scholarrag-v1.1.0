"""Tests for field foundations service."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.paper import PaperData

# ---------- _normalize / _titles_match tests ----------


def test_normalize_basic():
    from app.services.field_foundations import _normalize

    assert _normalize("Machine Learning: A Review!") == "machine learning a review"


def test_titles_match_exact():
    from app.services.field_foundations import _titles_match

    assert _titles_match(
        "Situated Cognition and the Culture of Learning",
        "Situated Cognition and the Culture of Learning",
    )


def test_titles_match_case_insensitive():
    from app.services.field_foundations import _titles_match

    assert _titles_match(
        "situated cognition and the culture of learning",
        "Situated Cognition and the Culture of Learning",
    )


def test_titles_match_no_match():
    from app.services.field_foundations import _titles_match

    assert not _titles_match(
        "Deep Learning Methods",
        "Situated Cognition and the Culture of Learning",
    )


# ---------- _verify_work tests ----------


@pytest.mark.asyncio
async def test_verify_work_found_via_openalex():
    from app.schemas.field_foundations import FoundationalWork
    from app.services.field_foundations import _verify_work

    work = FoundationalWork(
        suggested_title="Situated Cognition and the Culture of Learning",
        suggested_authors=["Brown, J. S."],
        suggested_year=1989,
        why_essential="Introduced situated learning theory",
    )

    mock_paper = PaperData(
        title="Situated Cognition and the Culture of Learning",
        doi="10.3102/0013189X018001032",
        source_api="openalex",
    )

    with patch(
        "app.services.field_foundations.OpenAlexClient.search",
        new_callable=AsyncMock,
        return_value=[mock_paper],
    ):
        result = await _verify_work(work)

    assert result.verified is True
    assert result.matched_paper is not None
    assert result.matched_paper.doi == "10.3102/0013189X018001032"


@pytest.mark.asyncio
async def test_verify_work_retries_openalex_with_the_first_author():
    """A bare-title miss is retried once, qualified with the first author."""
    from app.schemas.field_foundations import FoundationalWork
    from app.services.field_foundations import _verify_work

    work = FoundationalWork(
        suggested_title="Constructivism and Learning",
        suggested_authors=["Author A"],
        suggested_year=1995,
        why_essential="Key theoretical work",
    )

    mock_paper = PaperData(
        title="Constructivism and Learning",
        source_api="openalex",
        external_id="W789",
    )

    with patch(
        "app.services.field_foundations.OpenAlexClient.search",
        new_callable=AsyncMock,
        side_effect=[[], [mock_paper]],
    ) as search:
        result = await _verify_work(work)

    assert result.verified is True
    assert result.matched_paper is not None
    assert result.matched_paper.source_api == "openalex"
    assert search.await_count == 2
    assert search.await_args_list[0].args[0] == "Constructivism and Learning"
    assert search.await_args_list[1].args[0] == "Constructivism and Learning Author A"


@pytest.mark.asyncio
async def test_verify_work_not_found():
    from app.schemas.field_foundations import FoundationalWork
    from app.services.field_foundations import _verify_work

    work = FoundationalWork(
        suggested_title="A Completely Fabricated Paper Title",
        suggested_authors=["Nobody"],
        suggested_year=1900,
        why_essential="Does not exist",
    )

    with patch(
        "app.services.field_foundations.OpenAlexClient.search",
        new_callable=AsyncMock,
        return_value=[],
    ):
        result = await _verify_work(work)

    assert result.verified is False
    assert result.matched_paper is None


# ---------- run_field_foundations existence test ----------


def test_run_field_foundations_exists():
    from app.services.field_foundations import run_field_foundations

    assert run_field_foundations is not None


def test_field_foundations_has_no_semantic_scholar_references():
    """User decision #1: OpenAlex is the only scholarly data source."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "app" / "services" / "field_foundations.py"
    ).read_text(encoding="utf-8")

    assert "SemanticScholarClient" not in source
