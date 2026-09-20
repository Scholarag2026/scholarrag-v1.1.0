import asyncio
import logging
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings, validate_security_settings
from app.database import async_session_factory
from app.services import task as task_service
from app.services import wos_import

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

WOS_CSV_DIR = Path(__file__).resolve().parent.parent / "data" / "wos"


def configure_logging() -> None:
    """Install the process-wide stdlib logging configuration.

    Runs at import time, before the FastAPI app is created, so that every INFO
    job-lifecycle line emitted by services and background tasks reaches stdout
    (and therefore the Zeabur runtime log).

    ``force=True`` makes this idempotent and deterministic: without it,
    ``basicConfig`` silently no-ops whenever something else (pytest's logging
    plugin, an imported library) already attached a root handler, which is
    exactly how the current INFO lines get swallowed.

    uvicorn configures its own "uvicorn"/"uvicorn.error"/"uvicorn.access"
    loggers with ``propagate=False`` before it imports this module, so those
    handlers survive untouched and uvicorn access logging keeps working.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


configure_logging()
validate_security_settings(settings)

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # D12a: only jobs with no progress for settings.job_stale_after_minutes are failed, so a
    # booting worker can never kill a job its sibling worker is actively running.
    try:
        swept = await task_service.sweep_stale_jobs(async_session_factory)
        if swept:
            logger.warning("Boot sweep failed %d abandoned job(s)", len(swept))
    except Exception:
        logger.exception("Boot stale-job sweep failed")

    # #51: advisory-locked WoS import-if-empty plus a bulk paper backfill.
    try:
        wos_result = await wos_import.ensure_wos_data(async_session_factory, WOS_CSV_DIR)
        logger.info("WoS boot maintenance: %s", wos_result)
    except Exception:
        logger.exception("WoS boot maintenance failed")

    # D12b: periodic reaper so a job abandoned mid-flight cannot block its project forever.
    reaper_task = asyncio.create_task(
        task_service.stale_job_reaper(async_session_factory),
        name="stale-job-reaper",
    )
    logger.info(
        "Stale-job reaper started (every %ss, staleness %s min)",
        settings.job_reaper_interval_seconds,
        settings.job_stale_after_minutes,
    )

    try:
        yield
    finally:
        reaper_task.cancel()
        with suppress(asyncio.CancelledError):
            await reaper_task
        try:
            interrupted = await task_service.interrupt_in_flight_jobs(async_session_factory)
            if interrupted:
                logger.warning(
                    "Marked %d in-flight job(s) interrupted on shutdown", len(interrupted)
                )
        except Exception:
            logger.exception("Failed to mark in-flight jobs interrupted on shutdown")

app = FastAPI(title="ScholarRAG API", version="1.1.0", lifespan=lifespan)

origins = [o for o in [settings.frontend_url, settings.backend_url] if o]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "healthy"}
