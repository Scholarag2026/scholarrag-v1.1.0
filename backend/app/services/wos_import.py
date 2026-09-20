"""WoS journal import service — handles all 4 Core Collection CSVs."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, insert, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.wos_journal import WosJournal

logger = logging.getLogger(__name__)

# Map CSV filenames (partial match) to short collection codes
_COLLECTION_MAP: dict[str, str] = {
    "SCIE": "SCIE",
    "SSCI": "SSCI",
    "AHCI": "AHCI",
    "ESCI": "ESCI",
}

_CSV_FILENAMES: list[str] = [
    "Science Citation Index Expanded (SCIE).csv",
    "Social Sciences Citation Index (SSCI).csv",
    "Arts & Humanities Citation Index (AHCI).csv",
    "Emerging Sources Citation Index (ESCI).csv",
]


def _detect_collection(filename: str) -> str:
    """Derive the collection code from a CSV filename.

    The parenthesized tag is matched first: a bare substring scan would classify
    "Social Sciences Citation Index (SSCI).csv" as SCIE, because "SCIE" occurs
    inside "SCIENCES".
    """
    name_upper = filename.upper()
    for key, code in _COLLECTION_MAP.items():
        if f"({key})" in name_upper:
            return code
    for key, code in _COLLECTION_MAP.items():
        if key in name_upper:
            return code
    return "WOS"


_INSERT_CHUNK = 1000


def parse_wos_csv_rows(csv_content: str, collection: str) -> list[dict[str, Any]]:
    """Parse a WoS collection CSV into insert-ready row dicts.

    Pure and synchronous — this is the ~4 MB / ~24k-row blocking hot spot, so callers run
    it via ``asyncio.to_thread`` (D20).

    CSV columns (with header row):
        Journal title, ISSN, eISSN, Publisher name, Publisher address,
        Languages, Web of Science Categories
    """
    reader = csv.reader(io.StringIO(csv_content))
    next(reader, None)  # skip header

    rows: list[dict[str, Any]] = []
    for row in reader:
        if len(row) < 1:
            continue
        title = row[0].strip()
        if not title:
            continue
        rows.append(
            {
                "id": uuid.uuid4(),
                "journal_title": title,
                "issn": row[1].strip().upper() if len(row) > 1 and row[1].strip() else None,
                "eissn": row[2].strip().upper() if len(row) > 2 and row[2].strip() else None,
                "collection": collection,
                "categories": row[6].strip() if len(row) > 6 and row[6].strip() else None,
            }
        )
    return rows


def _read_csv_text(filepath: Path) -> str:
    """Blocking file read (utf-8-sig handles the BOM). Run via asyncio.to_thread."""
    return filepath.read_text(encoding="utf-8-sig")


async def _replace_collection_rows(
    db: AsyncSession, collection: str, rows: list[dict[str, Any]]
) -> int:
    """Delete this collection's rows and bulk-insert *rows*. Does NOT commit.

    Callers control the transaction boundary because the boot path holds a
    transaction-scoped advisory lock that a mid-way commit would release.
    """
    await db.execute(delete(WosJournal).where(WosJournal.collection == collection))
    for start in range(0, len(rows), _INSERT_CHUNK):
        await db.execute(insert(WosJournal), rows[start : start + _INSERT_CHUNK])
    return len(rows)


async def _import_csv_dir(db: AsyncSession, csv_dir: Path) -> dict[str, int]:
    """Import every known WoS CSV found in *csv_dir*. Does NOT commit."""
    results: dict[str, int] = {}
    for filename in _CSV_FILENAMES:
        filepath = csv_dir / filename
        if not filepath.exists():
            continue
        collection = _detect_collection(filename)
        content = await asyncio.to_thread(_read_csv_text, filepath)
        rows = await asyncio.to_thread(parse_wos_csv_rows, content, collection)
        results[collection] = await _replace_collection_rows(db, collection, rows)
    return results


async def import_wos_csv(db: AsyncSession, csv_content: str, collection: str) -> int:
    """Import a single WoS collection CSV.

    Rows for this collection are deleted before re-importing so the function is
    idempotent. Returns the number of journals imported.
    """
    rows = await asyncio.to_thread(parse_wos_csv_rows, csv_content, collection)
    count = await _replace_collection_rows(db, collection, rows)
    await db.commit()
    return count


async def import_all_csvs(db: AsyncSession, csv_dir: str | Path) -> dict[str, int]:
    """Import all 4 WoS Core Collection CSV files from *csv_dir*.

    Returns a dict mapping collection code to import count.
    """
    results = await _import_csv_dir(db, Path(csv_dir))
    await db.commit()
    return results


async def get_wos_journal_count(db: AsyncSession) -> int:
    """Return total number of WoS journal rows in the database."""
    result = await db.execute(select(func.count()).select_from(WosJournal))
    return result.scalar() or 0


async def is_wos_indexed(
    db: AsyncSession,
    issn: str | None,
    eissn: str | None = None,
) -> tuple[bool, str | None, str | None]:
    """Check whether a journal is indexed in WoS Core Collection.

    Returns (indexed, collection, categories).
    - indexed is False (not None) when the ISSN is checked but not found.
    - collection and categories are None when not indexed.
    """
    if not issn and not eissn:
        return False, None, None

    conditions: list = []
    if issn:
        normalized = issn.strip().upper()
        conditions.append(WosJournal.issn == normalized)
        conditions.append(WosJournal.eissn == normalized)
    if eissn:
        normalized_e = eissn.strip().upper()
        conditions.append(WosJournal.issn == normalized_e)
        conditions.append(WosJournal.eissn == normalized_e)

    result = await db.execute(
        select(WosJournal).where(or_(*conditions)).limit(1)
    )
    journal = result.scalar_one_or_none()

    if journal is None:
        return False, None, None

    return True, journal.collection, journal.categories


async def is_wos_indexed_bulk(
    db: AsyncSession,
    issns: Iterable[str | None],
) -> dict[str, tuple[bool, str | None, str | None]]:
    """Bulk form of :func:`is_wos_indexed` — one SELECT for a whole ISSN list.

    Modelled on :func:`batch_enrich_wos`. Keys are normalized ISSNs
    (``value.strip().upper()``); values mirror :func:`is_wos_indexed`'s
    ``(indexed, collection, categories)`` tuple. Every non-empty input ISSN gets a key,
    including ones with no WoS match, which map to ``(False, None, None)``.
    Falsy or blank inputs are dropped. Matches on either ``issn`` or ``eissn``.
    """
    normalized: set[str] = set()
    for raw in issns:
        if not raw:
            continue
        value = raw.strip().upper()
        if value:
            normalized.add(value)

    if not normalized:
        return {}

    result = await db.execute(
        select(WosJournal).where(
            or_(WosJournal.issn.in_(normalized), WosJournal.eissn.in_(normalized))
        )
    )
    journals = result.scalars().all()

    found: dict[str, tuple[bool, str | None, str | None]] = {}
    for journal in journals:
        entry = (True, journal.collection, journal.categories)
        for candidate in (journal.issn, journal.eissn):
            if not candidate:
                continue
            key = candidate.strip().upper()
            if key in normalized:
                found.setdefault(key, entry)

    return {key: found.get(key, (False, None, None)) for key in normalized}


# Postgres caps bind parameters at 32767 per statement. Each chunk is matched against
# both issn and eissn, so keep chunks well under half that limit.
_BATCH_ENRICH_CHUNK_SIZE = 5000


async def batch_enrich_wos(
    db: AsyncSession, papers: list, *, field: str = "wos_collection"
) -> None:
    """Enrich a list of PaperData (or dicts) with wos_collection from wos_journals table.

    Modifies papers in-place. Papers without journal_issn are skipped. The lookup is
    chunked (see ``_BATCH_ENRICH_CHUNK_SIZE``) so a large paper list can't blow
    Postgres's per-statement bind-parameter limit.
    """
    issns: set[str] = set()
    for p in papers:
        issn = getattr(p, "journal_issn", None) or (
            p.get("journal_issn") if isinstance(p, dict) else None
        )
        if issn:
            issns.add(issn.strip().upper())
    if not issns:
        return

    issns_list = list(issns)
    journals: list[WosJournal] = []
    for start in range(0, len(issns_list), _BATCH_ENRICH_CHUNK_SIZE):
        chunk = issns_list[start : start + _BATCH_ENRICH_CHUNK_SIZE]
        result = await db.execute(
            select(WosJournal).where(
                or_(WosJournal.issn.in_(chunk), WosJournal.eissn.in_(chunk))
            )
        )
        journals.extend(result.scalars().all())

    lookup: dict[str, tuple[str | None, str | None]] = {}
    for j in journals:
        if j.issn:
            lookup[j.issn.strip().upper()] = (j.collection, j.categories)
        if j.eissn:
            lookup[j.eissn.strip().upper()] = (j.collection, j.categories)

    for p in papers:
        issn = getattr(p, "journal_issn", None) or (
            p.get("journal_issn") if isinstance(p, dict) else None
        )
        if issn:
            match = lookup.get(issn.strip().upper())
            if match:
                if isinstance(p, dict):
                    p["wos_collection"] = match[0]
                    p["wos_categories"] = match[1]
                else:
                    p.wos_collection = match[0]
                    p.wos_categories = match[1]


# Arbitrary but stable key for pg_try_advisory_xact_lock. Both uvicorn workers use it to
# serialize boot-time WoS maintenance.
WOS_MAINTENANCE_LOCK_KEY = 728301

# Set is_wos_indexed / collection / categories for every un-backfilled paper whose ISSN
# matches a WoS journal — one statement instead of one UPDATE per paper (#51).
_BACKFILL_MATCH_SQL = """
UPDATE papers AS p
SET is_wos_indexed = true,
    wos_collection = w.collection,
    wos_categories = w.categories
FROM wos_journals AS w
WHERE p.wos_collection IS NULL
  AND p.journal_issn IS NOT NULL
  AND (upper(btrim(p.journal_issn)) = w.issn OR upper(btrim(p.journal_issn)) = w.eissn)
"""

# Everything still un-backfilled after the join is genuinely not WoS-indexed. The
# `is_wos_indexed IS NULL` guard makes repeated boots a no-op instead of rewriting rows.
_BACKFILL_MISS_SQL = """
UPDATE papers
SET is_wos_indexed = false
WHERE wos_collection IS NULL
  AND journal_issn IS NOT NULL
  AND is_wos_indexed IS NULL
"""


async def backfill_paper_wos(db: AsyncSession) -> tuple[int, int]:
    """Bulk-backfill WoS indexing flags onto papers. Does NOT commit.

    Returns ``(matched_rowcount, unmatched_rowcount)``.
    """
    matched = await db.execute(text(_BACKFILL_MATCH_SQL))
    unmatched = await db.execute(text(_BACKFILL_MISS_SQL))
    return matched.rowcount, unmatched.rowcount


async def ensure_wos_data(session_factory, csv_dir: str | Path) -> dict[str, Any]:
    """Boot-time WoS maintenance, safe to run concurrently in every uvicorn worker (#51).

    ``pg_try_advisory_xact_lock`` makes "is the journal table empty?" and the import atomic,
    so two booting workers can no longer both pass the check and import the same ~24k rows.
    The lock is transaction-scoped: the single ``commit()`` at the end releases it, which is
    why this function inlines ``_import_csv_dir`` instead of calling the committing
    ``import_all_csvs``.

    A worker that loses the race returns ``{"skipped": "lock-held-by-sibling"}`` immediately
    rather than blocking boot.
    """
    async with session_factory() as db:
        acquired = (
            await db.execute(
                text("SELECT pg_try_advisory_xact_lock(:key)"),
                {"key": WOS_MAINTENANCE_LOCK_KEY},
            )
        ).scalar()
        if not acquired:
            logger.info("WoS boot maintenance skipped: sibling worker holds the advisory lock")
            return {"skipped": "lock-held-by-sibling"}

        existing = await get_wos_journal_count(db)
        imported: dict[str, int] = {}
        csv_path = Path(csv_dir)
        if existing == 0:
            if csv_path.exists():
                imported = await _import_csv_dir(db, csv_path)
            else:
                logger.warning("WoS CSV directory not found: %s", csv_path)

        matched, unmatched = await backfill_paper_wos(db)
        await db.commit()

        return {
            "existing": existing,
            "imported": imported,
            "backfilled_matched": matched,
            "backfilled_unmatched": unmatched,
        }
