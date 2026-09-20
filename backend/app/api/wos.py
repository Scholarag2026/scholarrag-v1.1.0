"""WoS journal import and status endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_admin_user
from app.models.user import User
from app.services.wos_import import (
    _CSV_FILENAMES,
    get_wos_journal_count,
    import_all_csvs,
    import_wos_csv,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wos", tags=["wos"])

# The CSV files are bundled inside backend/data/wos/.
# Path(__file__) = .../backend/app/api/wos.py  →  .parent.parent.parent = .../backend/
_CSV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "wos"

# Module-level session factory — overridden in tests to use the test DB.
_session_factory = async_session_factory


async def _run_wos_import(session_factory, csv_dir: Path) -> None:
    """Background job body: read + parse + bulk-insert all bundled WoS CSVs.

    Reading ~4.3 MB across 4 files and inserting ~24k rows used to run inline in the
    request handler, blocking one of two uvicorn workers for its whole duration (D20).
    """
    try:
        async with session_factory() as db:
            results = await import_all_csvs(db, csv_dir)
        logger.info("WoS import finished: %s (total %d)", results, sum(results.values()))
    except Exception:
        logger.exception("WoS import failed")


@router.post("/import", status_code=202)
async def import_wos_journals(
    background_tasks: BackgroundTasks,
    user: User = Depends(get_admin_user),
):
    """Start an import of all 4 WoS Core Collection CSV files bundled with the backend.

    Requires admin privileges. Returns 202 immediately; poll ``GET /wos/status`` for
    progress (``journal_count`` grows as collections land, ``ready`` flips to true).
    """
    if not _CSV_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CSV directory not found: {_CSV_DIR}",
        )

    found = [name for name in _CSV_FILENAMES if (_CSV_DIR / name).exists()]
    if not found:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No WoS CSV files found in {_CSV_DIR}. "
                "Expected files named after SCIE, SSCI, AHCI, or ESCI."
            ),
        )

    background_tasks.add_task(_run_wos_import, _session_factory, _CSV_DIR)

    return {
        "status": "started",
        "files": found,
        "csv_dir": str(_CSV_DIR),
    }


class ImportCsvRequest(BaseModel):
    csv_content: str
    collection: str


@router.post("/import-csv", status_code=200)
async def import_wos_csv_endpoint(
    req: ImportCsvRequest,
    user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Import a single WoS CSV via POST body (for remote servers without local files)."""
    if not req.csv_content or not req.collection:
        raise HTTPException(status_code=400, detail="csv_content and collection required")
    count = await import_wos_csv(db, req.csv_content, req.collection)
    return {"collection": req.collection, "imported": count}


@router.get("/status", status_code=200)
async def wos_status(
    db: AsyncSession = Depends(get_db),
):
    """Return the number of WoS journals loaded and whether the list is ready."""
    count = await get_wos_journal_count(db)
    return {
        "journal_count": count,
        "ready": count > 0,
    }
