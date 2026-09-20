# backend/tests/test_verification_service.py
"""Paper-record verification.

``overall_status`` is derived from *record checks* only (``doi_exists``,
``metadata_match``). Recency, citation count and WoS indexing are *indicators*: they
are reported with neutral statuses (``info`` / ``note`` / ``skipped``) and never change
the overall status of a record.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.services.verification import (
    INDICATOR_STATUSES,
    RECORD_CHECK_STATUSES,
    _check_citation_count,
    _check_metadata_match,
    _check_recency,
    _make_wos_check,
    verify_paper,
)

CROSSREF_RECORD = {
    "doi": "10.1038/test",
    "title": "Test Paper",
    "authors": [{"name": "John Smith"}],
    "year": 2023,
    "journal_name": "Nature",
    "issn": "0028-0836",
    "citation_count": 100,
}


def _crossref(record=CROSSREF_RECORD):
    mock = AsyncMock()
    mock.verify_doi = AsyncMock(return_value=record)
    return mock


# --- indicator helpers -------------------------------------------------------------


def test_check_recency_recent_paper():
    result = _check_recency(2024)
    assert result.status == "info"


def test_check_recency_old_paper():
    result = _check_recency(2010)
    assert result.status == "note"


def test_check_recency_very_old_paper():
    result = _check_recency(2000)
    assert result.status == "note"
    assert "older than 10 years" in result.message.lower() or "2000" in result.message


def test_check_recency_no_year():
    result = _check_recency(None)
    assert result.status == "skipped"


def test_check_citation_count_high():
    result = _check_citation_count(50)
    assert result.status == "info"


def test_check_citation_count_low():
    result = _check_citation_count(2)
    assert result.status == "note"


def test_check_citation_count_none():
    result = _check_citation_count(None)
    assert result.status == "skipped"


def test_wos_check_statuses_are_indicators():
    assert _make_wos_check(True).status == "info"
    assert _make_wos_check(False).status == "note"
    assert _make_wos_check(None).status == "skipped"


def test_indicator_helpers_never_emit_record_statuses():
    for check in (
        _check_recency(2024),
        _check_recency(1990),
        _check_recency(None),
        _check_citation_count(50),
        _check_citation_count(0),
        _check_citation_count(None),
        _make_wos_check(True),
        _make_wos_check(False),
        _make_wos_check(None),
    ):
        assert check.status in INDICATOR_STATUSES
        assert check.status not in {"pass", "fail", "warning"}


# --- record checks -----------------------------------------------------------------


def test_check_metadata_match_exact():
    crossref = {"title": "Deep Learning", "year": 2019}
    result = _check_metadata_match("Deep Learning", 2019, crossref)
    assert result.status == "pass"


def test_check_metadata_match_title_mismatch():
    crossref = {"title": "Machine Learning", "year": 2019}
    result = _check_metadata_match("Deep Learning", 2019, crossref)
    assert result.status == "warning"


def test_check_metadata_match_year_mismatch():
    crossref = {"title": "Deep Learning", "year": 2020}
    result = _check_metadata_match("Deep Learning", 2019, crossref)
    assert result.status == "warning"


# --- verify_paper ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_paper_with_valid_doi():
    mock_crossref = _crossref()

    paper_id = uuid4()
    result = await verify_paper(
        paper_id=paper_id,
        doi="10.1038/test",
        title="Test Paper",
        year=2023,
        citation_count=100,
        crossref_client=mock_crossref,
    )

    assert result.paper_id == paper_id
    assert result.overall_status == "pass"
    assert any(c.check_type == "doi_exists" and c.status == "pass" for c in result.checks)
    assert result.verified_at.tzinfo is not None
    assert result.verified_at.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.asyncio
async def test_verify_paper_separates_record_checks_from_indicators():
    """DOI found + metadata match + old year + 3 citations is still a verified record."""
    mock_crossref = _crossref({**CROSSREF_RECORD, "year": 2005, "citation_count": 3})

    result = await verify_paper(
        paper_id=uuid4(),
        doi="10.1038/test",
        title="Test Paper",
        year=2005,
        citation_count=3,
        crossref_client=mock_crossref,
        is_wos_indexed=False,
    )

    assert result.overall_status == "pass"
    assert [c.check_type for c in result.checks] == ["doi_exists", "metadata_match"]
    assert all(c.status in RECORD_CHECK_STATUSES for c in result.checks)

    indicators = {i.check_type: i for i in result.indicators}
    assert set(indicators) == {"wos_indexed", "recency", "citation_count"}
    assert indicators["recency"].status == "note"
    assert indicators["citation_count"].status == "note"
    assert indicators["wos_indexed"].status == "note"
    assert all(i.status in INDICATOR_STATUSES for i in result.indicators)


@pytest.mark.asyncio
async def test_verify_paper_positive_indicators_are_info():
    mock_crossref = _crossref()

    result = await verify_paper(
        paper_id=uuid4(),
        doi="10.1038/test",
        title="Test Paper",
        year=2023,
        citation_count=100,
        crossref_client=mock_crossref,
        is_wos_indexed=True,
    )

    indicators = {i.check_type: i.status for i in result.indicators}
    assert indicators == {"wos_indexed": "info", "recency": "info", "citation_count": "info"}
    assert result.overall_status == "pass"


@pytest.mark.asyncio
async def test_verify_paper_metadata_mismatch_is_warning():
    mock_crossref = _crossref({**CROSSREF_RECORD, "title": "A Completely Different Title"})

    result = await verify_paper(
        paper_id=uuid4(),
        doi="10.1038/test",
        title="Test Paper",
        year=2023,
        citation_count=100,
        crossref_client=mock_crossref,
    )

    assert result.overall_status == "warning"
    assert any(c.check_type == "metadata_match" and c.status == "warning" for c in result.checks)
    # Positive indicators do not rescue a metadata mismatch, negative ones do not worsen it.
    assert all(i.status in INDICATOR_STATUSES for i in result.indicators)


@pytest.mark.asyncio
async def test_verify_paper_without_doi():
    mock_crossref = AsyncMock()

    paper_id = uuid4()
    result = await verify_paper(
        paper_id=paper_id,
        doi=None,
        title="Test Paper",
        year=2023,
        citation_count=50,
        crossref_client=mock_crossref,
    )

    assert result.overall_status == "no_doi"
    assert any(c.check_type == "doi_exists" and c.status == "skipped" for c in result.checks)
    assert [c.check_type for c in result.checks] == ["doi_exists"]
    # Indicators are still computed from local metadata.
    indicators = {i.check_type: i.status for i in result.indicators}
    assert indicators == {"wos_indexed": "skipped", "recency": "info", "citation_count": "info"}
    mock_crossref.verify_doi.assert_not_called()


@pytest.mark.asyncio
async def test_verify_paper_doi_not_found():
    mock_crossref = AsyncMock()
    mock_crossref.verify_doi = AsyncMock(return_value=None)

    paper_id = uuid4()
    result = await verify_paper(
        paper_id=paper_id,
        doi="10.1234/fake",
        title="Fake Paper",
        year=2023,
        citation_count=10,
        crossref_client=mock_crossref,
    )

    assert result.overall_status == "fail"
    assert any(c.check_type == "doi_exists" and c.status == "fail" for c in result.checks)
    assert [c.check_type for c in result.checks] == ["doi_exists"]
    assert {i.check_type for i in result.indicators} == {
        "wos_indexed", "recency", "citation_count",
    }


@pytest.mark.asyncio
async def test_verify_paper_result_serialises_indicators_separately():
    """The JSON the job stores (and the frontend reads) keeps the two lists apart."""
    result = await verify_paper(
        paper_id=uuid4(),
        doi="10.1038/test",
        title="Test Paper",
        year=1999,
        citation_count=1,
        crossref_client=_crossref({**CROSSREF_RECORD, "year": 1999, "citation_count": 1}),
    )
    dumped = result.model_dump(mode="json")
    assert dumped["overall_status"] == "pass"
    assert {c["check_type"] for c in dumped["checks"]} == {"doi_exists", "metadata_match"}
    assert {i["check_type"] for i in dumped["indicators"]} == {
        "wos_indexed", "recency", "citation_count",
    }
    assert isinstance(dumped["verified_at"], str)


# --- structured indicator details (frontend contract: lib/indicators.ts) ------------


def test_indicator_details_are_structured():
    wos = _make_wos_check(True, collection="SCIE")
    assert wos.details == {"indexed": True, "collection": "SCIE"}
    assert _make_wos_check(False).details == {"indexed": False, "collection": None}
    assert _make_wos_check(None).details is None

    recency = _check_recency(2005)
    assert recency.details["year"] == 2005
    assert recency.details["window_years"] == 10
    assert recency.details["age_years"] == datetime.now(timezone.utc).year - 2005
    assert _check_recency(None).details is None

    assert _check_citation_count(3).details == {"count": 3, "threshold": 10}
    assert _check_citation_count(None).details is None


@pytest.mark.asyncio
async def test_verify_paper_passes_wos_collection_into_details():
    result = await verify_paper(
        paper_id=uuid4(),
        doi="10.1038/test",
        title="Test Paper",
        year=2005,
        citation_count=3,
        crossref_client=_crossref({**CROSSREF_RECORD, "year": 2005, "citation_count": 3}),
        is_wos_indexed=True,
        wos_collection="SSCI",
    )
    details = {i.check_type: i.details for i in result.indicators}
    assert details["wos_indexed"] == {"indexed": True, "collection": "SSCI"}
    assert details["recency"]["year"] == 2005
    assert details["citation_count"]["count"] == 3

    no_doi = await verify_paper(
        paper_id=uuid4(),
        doi=None,
        title="Test Paper",
        year=2005,
        citation_count=3,
        crossref_client=_crossref(),
        is_wos_indexed=False,
        wos_collection=None,
    )
    details = {i.check_type: i.details for i in no_doi.indicators}
    assert details["wos_indexed"] == {"indexed": False, "collection": None}
    assert details["citation_count"] == {"count": 3, "threshold": 10}
