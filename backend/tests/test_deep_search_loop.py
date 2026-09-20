"""Tests for the Deep Search round loop — fan-out, wall-clock cap, cancellation (T3)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.config import settings  # noqa: E402
from app.models.analysis_job import JobStatus  # noqa: E402
from app.schemas.paper import PaperData  # noqa: E402
from app.services import deep_search as deep_search_module  # noqa: E402
from app.services.search import SearchService  # noqa: E402


class _FakeSession:
    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


def _fake_session_factory() -> _FakeSession:
    return _FakeSession()


def _paper(idx: int) -> PaperData:
    return PaperData(
        doi=f"10.1/d{idx}",
        title=f"Deep Paper {idx}",
        authors=[{"name": "Test Author"}],
        year=2023,
        journal_issn="1234-5678",
        citation_count=5,
        source_api="openalex",
        abstract="An abstract.",
    )


class _FakeExpansionOutput:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries


class _FakeRunResult:
    def __init__(self, queries: list[str]) -> None:
        self.output = _FakeExpansionOutput(queries)


class _FakeExpansionAgent:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    async def run(self, prompt, deps=None):
        return _FakeRunResult(self._queries)


def _patches(search_impl, expanded: list[str], abort, update):
    """Common patch set for the deep-search loop."""
    return (
        patch.object(SearchService, "search", new=search_impl),
        patch.object(
            deep_search_module,
            "get_query_expansion_agent",
            return_value=_FakeExpansionAgent(expanded),
        ),
        patch.object(deep_search_module, "batch_enrich_wos", new=AsyncMock()),
        patch.object(deep_search_module.task_service, "should_abort", new=abort),
        patch.object(deep_search_module.task_service, "update_job_status", new=update),
    )


@pytest.mark.asyncio
async def test_expanded_queries_run_concurrently(monkeypatch):
    monkeypatch.setattr(settings, "search_fanout_concurrency", 5)
    state = {"active": 0, "peak": 0, "calls": 0}

    async def _fake_search(self, query, **kwargs):
        state["calls"] += 1
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return [_paper(state["calls"] * 100)], {"openalex": 1}

    update = AsyncMock()
    patches = _patches(
        _fake_search,
        ["e1", "e2", "e3", "e4", "e5"],
        AsyncMock(return_value=False),
        update,
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await deep_search_module.run_deep_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            filters={"max_rounds": 2},
            session_factory=_fake_session_factory,
        )

    # Round 1 is a single search; peak > 1 can only come from the round-2 fan-out.
    assert state["peak"] >= 2
    assert state["peak"] <= 5


@pytest.mark.asyncio
async def test_wall_clock_cap_stops_the_round_loop(monkeypatch):
    monkeypatch.setattr(settings, "deep_search_max_time_minutes", 0.0)
    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        return [_paper(len(calls))], {"openalex": 1}

    update = AsyncMock()
    patches = _patches(
        _fake_search, ["e1", "e2", "e3"], AsyncMock(return_value=False), update,
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await deep_search_module.run_deep_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            filters={"max_rounds": 6},
            session_factory=_fake_session_factory,
        )

    # Only the round-1 initial search runs; round 2 trips the budget at its boundary.
    assert calls == ["teacher burnout"]
    final = update.await_args_list[-1]
    assert final.kwargs["result"]["stop_reason"] == "time_limit"
    assert final.kwargs["result"]["coverage"]["rounds"][-1]["stopped"] is True


@pytest.mark.asyncio
async def test_wall_clock_cap_binds_between_expanded_queries(monkeypatch):
    monkeypatch.setattr(settings, "search_fanout_concurrency", 1)
    monkeypatch.setattr(settings, "deep_search_max_time_minutes", 0.005)  # 0.3 s
    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        await asyncio.sleep(0.2)
        return [_paper(len(calls))], {"openalex": 1}

    update = AsyncMock()
    patches = _patches(
        _fake_search,
        ["e1", "e2", "e3", "e4", "e5"],
        AsyncMock(return_value=False),
        update,
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await deep_search_module.run_deep_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            filters={"max_rounds": 6},
            session_factory=_fake_session_factory,
        )

    # 1 initial + at most a couple of expanded queries before the 0.3 s budget is gone.
    assert 1 <= len(calls) < 6


@pytest.mark.asyncio
async def test_cancellation_marks_the_job_cancelled():
    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        return [_paper(len(calls))], {"openalex": 1}

    update = AsyncMock()
    patches = _patches(
        _fake_search, ["e1", "e2"], AsyncMock(return_value=True), update,
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        await deep_search_module.run_deep_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            filters={"max_rounds": 6},
            session_factory=_fake_session_factory,
        )

    assert calls == ["teacher burnout"]
    final = update.await_args_list[-1]
    assert final.args[2] is JobStatus.cancelled
    assert final.kwargs["result"]["stop_reason"] == "cancelled"


def test_pipeline_sources_start_openalex_only():
    pipeline = deep_search_module.DeepSearchPipeline()
    assert pipeline.sources == {"openalex": 0}
