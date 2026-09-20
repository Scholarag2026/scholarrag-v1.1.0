"""Tests for the FastAPI lifespan: boot sweep, WoS maintenance, reaper, shutdown hook."""

import asyncio
import inspect

import app.main as main_mod


async def test_lifespan_sweeps_starts_reaper_and_interrupts_on_shutdown(monkeypatch):
    import app.services.task as task_mod
    import app.services.wos_import as wos_mod

    calls: dict[str, int] = {"sweep": 0, "wos": 0, "interrupt": 0}
    reaper_started = asyncio.Event()
    reaper_cancelled = asyncio.Event()

    async def fake_sweep(session_factory):
        calls["sweep"] += 1
        return []

    async def fake_ensure(session_factory, csv_dir):
        calls["wos"] += 1
        return {"existing": 1, "imported": {}, "backfilled_matched": 0, "backfilled_unmatched": 0}

    async def fake_reaper(session_factory, **kwargs):
        reaper_started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            reaper_cancelled.set()
            raise

    async def fake_interrupt(session_factory):
        calls["interrupt"] += 1
        return []

    monkeypatch.setattr(task_mod, "sweep_stale_jobs", fake_sweep)
    monkeypatch.setattr(task_mod, "stale_job_reaper", fake_reaper)
    monkeypatch.setattr(task_mod, "interrupt_in_flight_jobs", fake_interrupt)
    monkeypatch.setattr(wos_mod, "ensure_wos_data", fake_ensure)

    async with main_mod.lifespan(main_mod.app):
        await asyncio.wait_for(reaper_started.wait(), timeout=2)
        assert calls["sweep"] == 1
        assert calls["wos"] == 1

    assert reaper_cancelled.is_set()
    assert calls["interrupt"] == 1


async def test_lifespan_boot_survives_wos_failure(monkeypatch):
    import app.services.task as task_mod
    import app.services.wos_import as wos_mod

    async def fake_sweep(session_factory):
        return []

    async def exploding_ensure(session_factory, csv_dir):
        raise RuntimeError("wos boom")

    async def fake_reaper(session_factory, **kwargs):
        await asyncio.sleep(3600)

    async def fake_interrupt(session_factory):
        return []

    monkeypatch.setattr(task_mod, "sweep_stale_jobs", fake_sweep)
    monkeypatch.setattr(task_mod, "stale_job_reaper", fake_reaper)
    monkeypatch.setattr(task_mod, "interrupt_in_flight_jobs", fake_interrupt)
    monkeypatch.setattr(wos_mod, "ensure_wos_data", exploding_ensure)

    async with main_mod.lifespan(main_mod.app):
        pass  # must not raise — a WoS failure cannot block boot


def test_lifespan_no_longer_contains_an_unscoped_job_update():
    """The unscoped `status.in_([pending, running])` UPDATE must be gone from main.py."""
    source = inspect.getsource(main_mod)

    assert "JobStatus.pending, JobStatus.running" not in source
    assert "print(" not in source
