"""get_or_create_papers_bulk (issue PAPER-LOOKUP-N-PLUS-1).

Runs against the real test Postgres via the autouse db_session fixture in conftest.
"""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from sqlalchemy import func, select

from app.models.paper import Paper, SourceApi
from app.schemas.paper import PaperData
from app.services.paper import get_or_create_papers_bulk


def _data(
    title: str,
    *,
    doi: str | None = None,
    external_id: str | None = None,
    issn: str | None = None,
) -> PaperData:
    return PaperData(
        doi=doi,
        title=title,
        authors=[{"name": "Author One"}],
        year=2021,
        journal_issn=issn,
        citation_count=3,
        source_api="openalex",
        external_id=external_id,
    )


async def _paper_count(db) -> int:
    return (await db.execute(select(func.count()).select_from(Paper))).scalar() or 0


async def test_empty_input_returns_empty_list(db_session):
    assert await get_or_create_papers_bulk(db_session, []) == []


async def test_creates_rows_in_input_order(db_session):
    before = await _paper_count(db_session)

    rows = await get_or_create_papers_bulk(
        db_session,
        [
            _data("Alpha", external_id="W1"),
            _data("Beta", doi=f"10.1000/{uuid.uuid4().hex[:8]}"),
        ],
    )

    assert [r.title for r in rows] == ["Alpha", "Beta"]
    assert all(r.id is not None for r in rows)
    assert all(r.source_api is SourceApi.openalex for r in rows)
    assert await _paper_count(db_session) == before + 2


async def test_duplicates_inside_the_batch_collapse_to_one_row(db_session):
    doi = f"10.1000/{uuid.uuid4().hex[:8]}"
    before = await _paper_count(db_session)

    rows = await get_or_create_papers_bulk(
        db_session,
        [_data("Same", doi=doi), _data("Same again", doi=doi), _data("W dup", external_id="W7"),
         _data("W dup 2", external_id="W7")],
    )

    assert rows[0].id == rows[1].id
    assert rows[2].id == rows[3].id
    assert await _paper_count(db_session) == before + 2


async def test_existing_rows_are_reused_by_doi_and_by_external_id(db_session):
    doi = f"10.1000/{uuid.uuid4().hex[:8]}"
    existing_doi = Paper(
        doi=doi, title="Existing by DOI", authors=[], source_api=SourceApi.openalex
    )
    existing_ext = Paper(
        external_id="W42", title="Existing by W-id", authors=[], source_api=SourceApi.openalex
    )
    db_session.add_all([existing_doi, existing_ext])
    await db_session.flush()
    before = await _paper_count(db_session)

    rows = await get_or_create_papers_bulk(
        db_session, [_data("Incoming", doi=doi), _data("Incoming 2", external_id="W42")]
    )

    assert rows[0].id == existing_doi.id
    assert rows[1].id == existing_ext.id
    assert await _paper_count(db_session) == before


async def test_wos_flag_is_none_without_an_issn(db_session):
    rows = await get_or_create_papers_bulk(db_session, [_data("No ISSN", external_id="W99")])
    assert rows[0].is_wos_indexed is None


async def test_unindexed_issn_records_a_false_flag(db_session):
    rows = await get_or_create_papers_bulk(
        db_session, [_data("Unknown journal", external_id="W98", issn="9999-9999")]
    )
    assert rows[0].is_wos_indexed is False


async def test_bulk_lookup_survives_the_postgres_bind_parameter_limit(db_session):
    """Regression for PAPER-LOOKUP-N-PLUS-1: a large graph build hands every fetched
    work to get_or_create_papers_bulk in one call. Before the IN() chunking fix, a
    single query with dois.in_(...) OR external_ids.in_(...) blew Postgres's 32767
    bind-parameter limit once |unique dois| + |unique external_ids| exceeded it.
    17,000 papers, each with a distinct doi *and* a distinct external_id, produce
    34,000 unique lookup values — the smallest confirmed failing case.
    """
    count = 17_000
    papers = [
        _data(
            f"Bulk paper {i}",
            doi=f"10.9999/bulk-{uuid.uuid4().hex}",
            external_id=f"W-bulk-{uuid.uuid4().hex}",
        )
        for i in range(count)
    ]

    rows = await get_or_create_papers_bulk(db_session, papers)

    assert len(rows) == count
    assert [r.title for r in rows] == [p.title for p in papers]
    assert all(r.id is not None for r in rows)
