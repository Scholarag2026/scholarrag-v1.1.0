"""Tests for bulk WoS CSV import and off-event-loop parsing (#51 / D20)."""

import threading

from sqlalchemy import func, select

from app.models.wos_journal import WosJournal

CSV_HEADER = "Journal title,ISSN,eISSN,Publisher name,Publisher address,Languages,Categories\n"
CSV_ROWS = (
    "Journal of Testing, 1234-5678,2345-6789,Pub,Addr,English,Computer Science\n"
    "Journal of Nothing,,,Pub,Addr,English,\n"
)


def test_parse_wos_csv_rows_builds_insert_ready_dicts():
    from app.services.wos_import import parse_wos_csv_rows

    rows = parse_wos_csv_rows(CSV_HEADER + CSV_ROWS, "SCIE")

    assert len(rows) == 2
    assert rows[0]["journal_title"] == "Journal of Testing"
    assert rows[0]["issn"] == "1234-5678"
    assert rows[0]["eissn"] == "2345-6789"
    assert rows[0]["collection"] == "SCIE"
    assert rows[0]["categories"] == "Computer Science"
    assert rows[1]["issn"] is None
    assert rows[1]["categories"] is None
    assert rows[0]["id"] != rows[1]["id"]


def test_parse_wos_csv_rows_skips_blank_titles():
    from app.services.wos_import import parse_wos_csv_rows

    rows = parse_wos_csv_rows(CSV_HEADER + " ,1111-1111,,,,,\n", "SCIE")

    assert rows == []


async def test_import_wos_csv_parses_off_the_event_loop(db_session, monkeypatch):
    from app.services import wos_import

    seen: dict[str, str] = {}
    real_parse = wos_import.parse_wos_csv_rows

    def recording_parse(csv_content, collection):
        seen["thread"] = threading.current_thread().name
        return real_parse(csv_content, collection)

    monkeypatch.setattr(wos_import, "parse_wos_csv_rows", recording_parse)

    count = await wos_import.import_wos_csv(db_session, CSV_HEADER + CSV_ROWS, "SCIE")

    assert count == 2
    assert seen["thread"] != threading.main_thread().name


async def test_import_wos_csv_replaces_only_its_own_collection(db_session):
    from app.services.wos_import import import_wos_csv

    await import_wos_csv(db_session, CSV_HEADER + CSV_ROWS, "SCIE")
    await import_wos_csv(db_session, CSV_HEADER + "Solo Journal,9999-9999,,,,,\n", "SSCI")
    await import_wos_csv(db_session, CSV_HEADER + "Only One,1111-1111,,,,,\n", "SCIE")

    total = (await db_session.execute(select(func.count()).select_from(WosJournal))).scalar()
    scie = (
        await db_session.execute(
            select(func.count()).select_from(WosJournal).where(WosJournal.collection == "SCIE")
        )
    ).scalar()
    assert total == 2
    assert scie == 1


async def test_batch_enrich_wos_survives_the_postgres_bind_parameter_limit(db_session):
    """Regression for PAPER-LOOKUP-N-PLUS-1: batch_enrich_wos matches on
    ``or_(issn.in_(issns), eissn.in_(issns))``, so the same issn set is bound twice.
    Before chunking, a single query blew Postgres's 32767 bind-parameter limit once
    the unique issn count exceeded half that. 17,000 unique issns (34,000 bind
    params) reproduce the smallest confirmed failing case.
    """
    from app.services.wos_import import batch_enrich_wos

    papers = [{"journal_issn": f"{i:04d}-{i:04d}"} for i in range(17_000)]

    await batch_enrich_wos(db_session, papers)

    assert all(p.get("wos_collection") is None for p in papers)


async def test_import_all_csvs_reads_files_off_the_event_loop(db_session, tmp_path, monkeypatch):
    from app.services import wos_import

    (tmp_path / "Science Citation Index Expanded (SCIE).csv").write_text(
        CSV_HEADER + CSV_ROWS, encoding="utf-8"
    )

    seen: dict[str, str] = {}
    real_read = wos_import._read_csv_text

    def recording_read(filepath):
        seen["thread"] = threading.current_thread().name
        return real_read(filepath)

    monkeypatch.setattr(wos_import, "_read_csv_text", recording_read)

    results = await wos_import.import_all_csvs(db_session, tmp_path)

    assert results == {"SCIE": 2}
    assert seen["thread"] != threading.main_thread().name
