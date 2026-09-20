"""Startup logging configuration."""

import logging


def test_configure_logging_sets_root_level_to_info():
    from app.main import configure_logging

    logging.getLogger().setLevel(logging.CRITICAL)
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_installs_a_root_handler_with_a_formatter():
    from app.main import LOG_FORMAT, configure_logging

    configure_logging()
    handlers = logging.getLogger().handlers
    assert handlers, "root logger must have at least one handler after configure_logging()"
    formatter = handlers[0].formatter
    assert formatter is not None
    assert formatter._fmt == LOG_FORMAT


def test_configure_logging_quiets_httpx_and_httpcore():
    from app.main import configure_logging

    configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_configure_logging_is_idempotent():
    from app.main import configure_logging

    configure_logging()
    first = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == first


def test_configure_logging_does_not_hijack_uvicorn_loggers():
    """uvicorn installs its own handlers with propagate=False; we must not remove them."""
    from app.main import configure_logging

    uvicorn_logger = logging.getLogger("uvicorn.access")
    sentinel = logging.NullHandler()
    uvicorn_logger.addHandler(sentinel)
    try:
        configure_logging()
        assert sentinel in uvicorn_logger.handlers
    finally:
        uvicorn_logger.removeHandler(sentinel)


def test_importing_app_main_configures_logging_before_app_creation():
    """Importing app.main must leave the root logger at INFO (config runs at import time)."""
    import app.main  # noqa: F401

    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING


def test_app_main_exposes_a_module_logger():
    import app.main

    assert isinstance(app.main.logger, logging.Logger)
    assert app.main.logger.name == "app.main"


def test_main_module_has_no_print_calls():
    """Startup diagnostics must go through the logger, not print()."""
    from pathlib import Path

    import app.main

    source = Path(app.main.__file__).read_text(encoding="utf-8")
    assert "print(" not in source


async def test_lifespan_logs_stale_job_cleanup(caplog):
    """The startup sweep must emit a visible log line, not a swallowed print()."""
    import logging
    import uuid
    from unittest.mock import patch

    import app.main
    import app.services.task as task_mod
    import app.services.wos_import as wos_mod

    swept_ids = [uuid.uuid4() for _ in range(3)]

    async def fake_sweep(session_factory):
        return swept_ids

    async def fake_ensure(session_factory, csv_dir):
        return {"existing": 1, "imported": {}, "backfilled_matched": 0, "backfilled_unmatched": 0}

    async def fake_reaper(session_factory, **kwargs):
        import asyncio

        await asyncio.sleep(3600)

    async def fake_interrupt(session_factory):
        return []

    with caplog.at_level(logging.INFO, logger="app.main"):
        with patch.object(task_mod, "sweep_stale_jobs", fake_sweep), patch.object(
            task_mod, "stale_job_reaper", fake_reaper
        ), patch.object(task_mod, "interrupt_in_flight_jobs", fake_interrupt), patch.object(
            wos_mod, "ensure_wos_data", fake_ensure
        ):
            async with app.main.lifespan(app.main.app):
                pass

    messages = [r.getMessage() for r in caplog.records if r.name == "app.main"]
    assert any("Boot sweep failed 3 abandoned job" in m for m in messages), messages
