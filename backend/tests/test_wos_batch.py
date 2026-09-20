import pytest
from unittest.mock import AsyncMock, MagicMock
from app.schemas.paper import PaperData
from app.services.wos_import import batch_enrich_wos


@pytest.mark.asyncio
async def test_batch_enrich_wos_sets_collection():
    """Papers with matching ISSN get wos_collection set."""
    papers = [
        PaperData(title="Paper A", journal_issn="1234-5678", source_api="openalex"),
        PaperData(title="Paper B", journal_issn=None, source_api="openalex"),
        PaperData(title="Paper C", journal_issn="9999-0000", source_api="openalex"),
    ]

    mock_journal = MagicMock()
    mock_journal.issn = "1234-5678"
    mock_journal.eissn = None
    mock_journal.collection = "SCIE"
    mock_journal.categories = "Computer Science"

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [mock_journal]

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    await batch_enrich_wos(mock_db, papers)

    assert papers[0].wos_collection == "SCIE"
    assert papers[1].wos_collection is None
    assert papers[2].wos_collection is None


@pytest.mark.asyncio
async def test_batch_enrich_wos_no_issns():
    """If no papers have ISSNs, no DB query is made."""
    papers = [
        PaperData(title="Paper A", source_api="openalex"),
    ]
    mock_db = AsyncMock()
    await batch_enrich_wos(mock_db, papers)
    mock_db.execute.assert_not_called()
