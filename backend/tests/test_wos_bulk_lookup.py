"""Tests for is_wos_indexed_bulk — the batched WoS indexing lookup (T3 / WOS-N-PLUS-1)."""

import pytest

from app.models.wos_journal import WosJournal
from app.services.wos_import import is_wos_indexed_bulk


async def _seed(db):
    db.add_all([
        WosJournal(
            journal_title="Nature",
            issn="0028-0836",
            eissn="1476-4687",
            collection="SCIE",
            categories="Multidisciplinary Sciences",
        ),
        WosJournal(
            journal_title="Journal of Social Policy",
            issn="0047-2794",
            eissn=None,
            collection="SSCI",
            categories="Social Sciences",
        ),
    ])
    await db.commit()


@pytest.mark.asyncio
async def test_bulk_lookup_returns_a_row_per_input_issn(db_session):
    await _seed(db_session)

    result = await is_wos_indexed_bulk(
        db_session, ["0028-0836", "0047-2794", "9999-9999"],
    )

    assert result["0028-0836"] == (True, "SCIE", "Multidisciplinary Sciences")
    assert result["0047-2794"] == (True, "SSCI", "Social Sciences")
    assert result["9999-9999"] == (False, None, None)


@pytest.mark.asyncio
async def test_bulk_lookup_matches_eissn_and_normalizes_case(db_session):
    await _seed(db_session)

    result = await is_wos_indexed_bulk(db_session, ["  1476-4687  "])

    assert result["1476-4687"] == (True, "SCIE", "Multidisciplinary Sciences")


@pytest.mark.asyncio
async def test_bulk_lookup_drops_blank_and_none_issns(db_session):
    await _seed(db_session)

    result = await is_wos_indexed_bulk(db_session, [None, "", "   ", "0028-0836"])

    assert set(result) == {"0028-0836"}


@pytest.mark.asyncio
async def test_bulk_lookup_with_no_issns_issues_no_query(db_session):
    result = await is_wos_indexed_bulk(db_session, [None, ""])
    assert result == {}


@pytest.mark.asyncio
async def test_bulk_lookup_uses_a_single_select(db_session):
    """The whole point: O(1) round trips, not O(papers)."""
    await _seed(db_session)

    executed: list[str] = []
    original_execute = db_session.execute

    async def _counting_execute(statement, *args, **kwargs):
        executed.append(str(statement))
        return await original_execute(statement, *args, **kwargs)

    db_session.execute = _counting_execute
    try:
        await is_wos_indexed_bulk(
            db_session, [f"0000-{i:04d}" for i in range(50)] + ["0028-0836"],
        )
    finally:
        db_session.execute = original_execute

    assert len(executed) == 1
