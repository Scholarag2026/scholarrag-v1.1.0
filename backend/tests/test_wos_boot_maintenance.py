"""Tests for boot-time WoS maintenance: advisory lock + bulk backfill (#51)."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.paper import Paper, SourceApi
from tests.conftest import TEST_DATABASE_URL

CSV_HEADER = "Journal title,ISSN,eISSN,Publisher name,Publisher address,Languages,Categories\n"


@pytest.fixture
async def session_factory():
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _write_scie_csv(tmp_path, body: str):
    (tmp_path / "Science Citation Index Expanded (SCIE).csv").write_text(
        CSV_HEADER + body, encoding="utf-8"
    )


async def test_ensure_wos_data_imports_when_table_is_empty(db_session, session_factory, tmp_path):
    from app.services.wos_import import ensure_wos_data, get_wos_journal_count

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\nJournal B,2222-2222,,,,,Cat B\n")

    result = await ensure_wos_data(session_factory, tmp_path)

    assert result["existing"] == 0
    assert result["imported"] == {"SCIE": 2}
    assert await get_wos_journal_count(db_session) == 2


async def test_ensure_wos_data_does_not_reimport_when_rows_exist(
    db_session, session_factory, tmp_path
):
    from app.services.wos_import import ensure_wos_data, get_wos_journal_count

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\nJournal B,2222-2222,,,,,Cat B\n")

    await ensure_wos_data(session_factory, tmp_path)
    second = await ensure_wos_data(session_factory, tmp_path)

    assert second["existing"] == 2
    assert second["imported"] == {}
    assert await get_wos_journal_count(db_session) == 2


async def test_ensure_wos_data_skips_entirely_when_a_sibling_holds_the_lock(
    session_factory, tmp_path
):
    """The check-then-act race is closed: the loser imports nothing at all."""
    from app.services.wos_import import WOS_MAINTENANCE_LOCK_KEY, ensure_wos_data

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\n")

    async with session_factory() as holder:
        acquired = (
            await holder.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"),
                {"key": WOS_MAINTENANCE_LOCK_KEY},
            )
        ).scalar()
        assert acquired is True

        result = await ensure_wos_data(session_factory, tmp_path)
        assert result == {"skipped": "lock-held-by-sibling"}

        await holder.rollback()


async def test_ensure_wos_data_backfills_papers_in_bulk(db_session, session_factory, tmp_path):
    from app.services.wos_import import ensure_wos_data

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\n")

    hit = Paper(
        title="Matching paper",
        journal_issn=" 1234-5678 ",
        source_api=SourceApi.manual,
        doi=f"10.1/{uuid.uuid4().hex[:8]}",
    )
    miss = Paper(
        title="Non-matching paper",
        journal_issn="9999-9999",
        source_api=SourceApi.manual,
        doi=f"10.1/{uuid.uuid4().hex[:8]}",
    )
    db_session.add_all([hit, miss])
    await db_session.commit()

    result = await ensure_wos_data(session_factory, tmp_path)

    assert result["backfilled_matched"] == 1
    assert result["backfilled_unmatched"] == 1

    await db_session.refresh(hit)
    await db_session.refresh(miss)
    assert hit.is_wos_indexed is True
    assert hit.wos_collection == "SCIE"
    assert hit.wos_categories == "Cat A"
    assert miss.is_wos_indexed is False
    assert miss.wos_collection is None


async def test_backfill_is_idempotent(db_session, session_factory, tmp_path):
    from app.services.wos_import import ensure_wos_data

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\n")
    db_session.add(
        Paper(
            title="Matching paper",
            journal_issn="1234-5678",
            source_api=SourceApi.manual,
            doi=f"10.1/{uuid.uuid4().hex[:8]}",
        )
    )
    await db_session.commit()

    await ensure_wos_data(session_factory, tmp_path)
    second = await ensure_wos_data(session_factory, tmp_path)

    assert second["backfilled_matched"] == 0
    assert second["backfilled_unmatched"] == 0


async def test_wos_import_endpoint_returns_202_and_imports_in_background(
    client, db_session, session_factory, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import app.api.wos as wos_api
    from app.dependencies import get_admin_user
    from app.main import app
    from app.services.wos_import import get_wos_journal_count

    _write_scie_csv(tmp_path, "Journal A,1234-5678,,,,,Cat A\nJournal B,2222-2222,,,,,Cat B\n")
    monkeypatch.setattr(wos_api, "_CSV_DIR", tmp_path)
    monkeypatch.setattr(wos_api, "_session_factory", session_factory)

    app.dependency_overrides[get_admin_user] = lambda: SimpleNamespace(id=uuid.uuid4())
    try:
        response = await client.post("/api/v1/wos/import")
    finally:
        app.dependency_overrides.pop(get_admin_user, None)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "started"
    assert body["files"] == ["Science Citation Index Expanded (SCIE).csv"]

    # Starlette runs BackgroundTasks before the ASGI call returns, so the rows are in by now.
    assert await get_wos_journal_count(db_session) == 2


async def test_wos_import_endpoint_404s_when_no_csv_files_present(
    client, session_factory, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import app.api.wos as wos_api
    from app.dependencies import get_admin_user
    from app.main import app

    monkeypatch.setattr(wos_api, "_CSV_DIR", tmp_path)
    monkeypatch.setattr(wos_api, "_session_factory", session_factory)

    app.dependency_overrides[get_admin_user] = lambda: SimpleNamespace(id=uuid.uuid4())
    try:
        response = await client.post("/api/v1/wos/import")
    finally:
        app.dependency_overrides.pop(get_admin_user, None)

    assert response.status_code == 404
