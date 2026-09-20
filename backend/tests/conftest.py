import importlib
import os
import pkgutil
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app
from app.models.base import Base

# The suite creates and DROPS every table on this database, so it is read from
# TEST_DATABASE_URL only -- never from DATABASE_URL, which names the live application
# database. Override the guard below with SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS=1 only if
# you really mean it.
DEFAULT_TEST_DATABASE_URL = "postgresql+asyncpg://deepresearch:secret@localhost:5432/deepresearch_test"
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)

DESTRUCTIVE_TESTS_OVERRIDE = "SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS"


# Spellings of the local machine that all reach the same Postgres server.
LOOPBACK_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def _database_identity(url: str) -> tuple[str, int, str]:
    """(host, port, database) so the same server+database compares equal across drivers
    and across loopback spellings (``localhost`` / ``127.0.0.1`` / ``::1`` / no host)."""
    try:
        parsed = make_url(url)
    except ArgumentError as exc:
        raise pytest.UsageError(f"Not a valid SQLAlchemy database URL: {url!r} ({exc})") from exc
    host = (parsed.host or "").strip("[]").lower()
    if host in LOOPBACK_HOSTS:
        host = "localhost"
    return (host, parsed.port or 5432, parsed.database or "")


def assert_safe_test_database(url: str, live_urls: Iterable[str]) -> None:
    """Refuse to run the suite against anything that could be a live database.

    Raises ``pytest.UsageError`` unless the database named by ``url`` ends in ``_test``
    and differs from every non-empty entry of ``live_urls`` (the URLs the application
    itself would use). ``SCHOLARRAG_ALLOW_DESTRUCTIVE_TESTS=1`` bypasses both checks.
    """
    if os.environ.get(DESTRUCTIVE_TESTS_OVERRIDE) == "1":
        return
    host, port, database = _database_identity(url)
    for live in live_urls:
        if live and _database_identity(live) == (host, port, database):
            raise pytest.UsageError(
                f"Refusing to run tests: TEST_DATABASE_URL names the live application "
                f"database {database!r} on {host}:{port} (the same database as DATABASE_URL). "
                f"The suite drops every table it finds. Point TEST_DATABASE_URL at a separate "
                f"database whose name ends in '_test', or set {DESTRUCTIVE_TESTS_OVERRIDE}=1 to "
                f"proceed anyway."
            )
    if not database.endswith("_test"):
        raise pytest.UsageError(
            f"Refusing to run tests: the database name {database!r} in TEST_DATABASE_URL does "
            f"not end in '_test'. The suite drops every table it finds. Use a dedicated test "
            f"database (for example 'deepresearch_test'), or set {DESTRUCTIVE_TESTS_OVERRIDE}=1 "
            f"to proceed anyway."
        )


def pytest_configure(config):
    assert_safe_test_database(
        TEST_DATABASE_URL,
        [settings.database_url, os.environ.get("DATABASE_URL", "")],
    )


def skip_unless_run_dir(run_dir: Path) -> Path:
    """Return *run_dir* when the tracked ``demo/output/<run>`` directory it names is
    present in this checkout; otherwise skip the calling test with a plain reason.

    The published release export ships only the one run directory backing the
    promoted ``demo/expected/`` baseline (submission-checklist.md, release items);
    every other tracked ``demo/output/<run>`` directory a backend test reads directly
    is excluded from that export, so a test built against one of those must skip
    there rather than fail on a missing file.
    """
    if not run_dir.is_dir():
        pytest.skip(f"run directory {run_dir.name} is not shipped")
    return run_dir


@pytest.fixture(autouse=True)
async def db_session():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:

        async def override_get_db():
            yield session

        app.dependency_overrides[get_db] = override_get_db
        yield session
        app.dependency_overrides.clear()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def modules_with_session_factory() -> list[tuple[ModuleType, str]]:
    """Every module-level session factory the application holds, as (module, attribute).

    ``app.database.async_session_factory`` is built on the live engine at import time;
    ``app.main`` imports it by name for the lifespan services (job sweeper, reaper, WoS
    import) and each ``app.api`` module that runs background jobs copies it into a
    module-level ``_session_factory``. None of them goes through ``get_db``, so the
    dependency override alone would leave those code paths pointed at the live database.
    """
    import app.api
    import app.database
    import app.main

    targets: list[tuple[ModuleType, str]] = [
        (app.database, "async_session_factory"),
        (app.main, "async_session_factory"),
    ]
    for info in pkgutil.iter_modules(app.api.__path__):
        module = importlib.import_module(f"app.api.{info.name}")
        if hasattr(module, "_session_factory"):
            targets.append((module, "_session_factory"))
    return targets


@pytest.fixture(autouse=True)
async def override_session_factories(db_session):
    """Rebind every module-level session factory to the test database for the test's duration."""
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    originals = [
        (module, attr, getattr(module, attr)) for module, attr in modules_with_session_factory()
    ]
    for module, attr, _ in originals:
        setattr(module, attr, factory)
    yield
    for module, attr, original in originals:
        setattr(module, attr, original)
    await engine.dispose()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def _reset_openalex_filter_block():
    """The filter-query block memo is process-global; isolate it per test."""
    from app.clients.openalex import reset_filter_block

    reset_filter_block()
    yield
    reset_filter_block()
