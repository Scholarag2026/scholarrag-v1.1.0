"""Tests for the Smart Search round loop.

Everything the pipeline touches outside its own logic is patched at the module boundary:
`generate_search_queries_with_provenance`, `screen_papers`, `SearchService.search`,
`is_wos_indexed_bulk`, `get_wos_journal_count`, `should_abort` and `update_job_status`.
No DB rows and no network are required.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.agents import relevance_screener_agent as screener_agent_module  # noqa: E402
from app.agents.query_generator_agent import GeneratedQueriesResult  # noqa: E402
from app.agents.relevance_screener_agent import (  # noqa: E402
    SCREENER_PROMPT_VERSION,
    ScreeningBatchResult,
    ScreeningError,
    SecondPassAnswerV2,
    SecondPassBatchResultV2,
    SecondPassSlotAnswer,
)
from app.models.analysis_job import JobStatus  # noqa: E402
from app.schemas.paper import PaperData  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402
from app.schemas.smart_search import SmartSearchRequest  # noqa: E402
from app.services import smart_search as smart_search_module  # noqa: E402

# ---------------------------------------------------------------- SmartSearchRequest


def test_smart_search_request_accepts_empty_stage_lists():
    req = SmartSearchRequest(query="q", inclusion_criteria=["a"], exclusion_criteria=["b"])
    assert req.inclusion_criteria_stages == []
    assert req.exclusion_criteria_stages == []


def test_smart_search_request_accepts_parallel_stage_lists():
    req = SmartSearchRequest(
        query="q",
        inclusion_criteria=["a", "b"],
        exclusion_criteria=["c"],
        inclusion_criteria_stages=["abstract", "full_text"],
        exclusion_criteria_stages=["full_text"],
    )
    assert req.inclusion_criteria_stages == ["abstract", "full_text"]
    assert req.exclusion_criteria_stages == ["full_text"]


def test_smart_search_request_rejects_a_mismatched_stage_list_length():
    with pytest.raises(ValueError, match="inclusion_criteria_stages"):
        SmartSearchRequest(
            query="q", inclusion_criteria=["a", "b"], inclusion_criteria_stages=["abstract"],
        )


def test_smart_search_request_rejects_an_unknown_stage_value():
    with pytest.raises(ValueError, match="exclusion_criteria_stages"):
        SmartSearchRequest(
            query="q", exclusion_criteria=["a"], exclusion_criteria_stages=["full-text"],
        )


# ---------------------------------------------------------------- replay support


def test_smart_search_request_defaults_replay_fields_to_none():
    req = SmartSearchRequest(query="q")
    assert req.queries_override is None
    assert req.publication_date_max is None


def test_smart_search_request_accepts_queries_override_and_publication_date_max():
    from datetime import date

    req = SmartSearchRequest(
        query="q",
        queries_override=[["round one query"], ["round two a", "round two b"]],
        publication_date_max=date(2026, 9, 8),
    )
    assert req.queries_override == [["round one query"], ["round two a", "round two b"]]
    assert req.publication_date_max == date(2026, 9, 8)


def test_smart_search_request_accepts_publication_date_max_as_an_iso_string():
    from datetime import date

    req = SmartSearchRequest(query="q", publication_date_max="2026-09-08")
    assert req.publication_date_max == date(2026, 9, 8)


def test_smart_search_request_rejects_an_empty_round_in_queries_override():
    with pytest.raises(ValueError, match="queries_override"):
        SmartSearchRequest(query="q", queries_override=[["a"], []])


_FAKE_QUERY_PROVENANCE = LLMCallProvenance(
    agent="query_generator",
    model_configured="deepseek-chat",
    model_reported="deepseek-v4-flash",
    prompt_version="sha256:qg_test",
    input_tokens=10,
    output_tokens=5,
)


def _query_result(
    queries: list[str], provenance: LLMCallProvenance = _FAKE_QUERY_PROVENANCE
) -> GeneratedQueriesResult:
    """Wrap a plain query list into the shape
    ``generate_search_queries_with_provenance`` returns, for tests that only care about
    the queries themselves, not the call's own provenance."""
    return GeneratedQueriesResult(queries=queries, provenance=provenance)


#: Two calls with distinct models, fingerprints and token counts, so a
#: test aggregating them over two rounds can tell the aggregation apart from a single
#: repeated value.
_QUERY_PROVENANCE_ROUND1 = LLMCallProvenance(
    agent="query_generator",
    model_configured="deepseek-chat",
    model_reported="deepseek-flash",
    system_fingerprint="fp_a",
    prompt_version="sha256:qg_test",
    input_tokens=40,
    output_tokens=8,
)
_QUERY_PROVENANCE_ROUND2 = LLMCallProvenance(
    agent="query_generator",
    model_configured="deepseek-chat",
    model_reported="deepseek-v4-flash",
    system_fingerprint="fp_b",
    prompt_version="sha256:qg_test",
    input_tokens=10,
    output_tokens=5,
)


def _batch(
    include: list[bool],
    reasons: list[str] | None = None,
    *,
    model_reported: str = "deepseek-v4-flash",
    fingerprint: str | None = "fp_a",
    input_tokens: int = 100,
    output_tokens: int = 10,
    reask_model_reported: str | None = None,
    reask_fingerprint: str | None = None,
    reask_input_tokens: int | None = None,
    reask_output_tokens: int | None = None,
) -> ScreeningBatchResult:
    """Build the value `screen_papers` returns for one batch.

    ``reask_*``, when any is given, builds a second
    ``LLMCallProvenance`` on ``reask_provenance`` -- the value ``screen_papers`` returns when
    a batch's response needed the one re-ask call -- with the given tokens/model/fingerprint,
    defaulting to the primary call's own values for whichever is omitted.
    """
    if reasons is None:
        reasons = ["relevant" if x else "off topic" for x in include]
    reask_given = (
        reask_model_reported is not None
        or reask_fingerprint is not None
        or reask_input_tokens is not None
        or reask_output_tokens is not None
    )
    reask_provenance = (
        LLMCallProvenance(
            agent="relevance_screener",
            model_configured="deepseek-chat",
            model_reported=reask_model_reported or model_reported,
            provider_response_id="resp-reask",
            system_fingerprint=reask_fingerprint if reask_fingerprint is not None else fingerprint,
            temperature=0.0,
            prompt_version=SCREENER_PROMPT_VERSION,
            input_tokens=reask_input_tokens if reask_input_tokens is not None else input_tokens,
            output_tokens=(
                reask_output_tokens if reask_output_tokens is not None else output_tokens
            ),
            called_at=datetime.now(timezone.utc),
        )
        if reask_given
        else None
    )
    return ScreeningBatchResult(
        include=include,
        reasons=reasons,
        reask_provenance=reask_provenance,
        provenance=LLMCallProvenance(
            agent="relevance_screener",
            model_configured="deepseek-chat",
            model_reported=model_reported,
            provider_response_id="resp",
            system_fingerprint=fingerprint,
            temperature=0.0,
            prompt_version=SCREENER_PROMPT_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            called_at=datetime.now(timezone.utc),
        ),
    )


@pytest.fixture(autouse=True)
def _journal_table_present():
    """Default every test to a populated WoS journal table (stage 1 active under "auto").

    Tests that need an empty table nest their own ``patch.object`` inside; the inner
    patch wins for its duration.
    """
    with patch.object(
        smart_search_module, "get_wos_journal_count", new=AsyncMock(return_value=1)
    ):
        yield


@pytest.fixture(autouse=True)
def _single_round_stop(monkeypatch):
    """Default every test to a stopping behaviour where a round with zero new
    included papers ends the job immediately (``smart_search_min_rounds=1``,
    ``smart_search_dry_round_patience=1``). The tests that exercise the min-rounds/patience
    stop rule itself override these two settings explicitly."""
    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 1)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 1)


_FAKE_SECOND_PASS_PROVENANCE = LLMCallProvenance(
    agent="relevance_screener_second_pass_v2",
    model_configured="deepseek-flash",
    model_reported="deepseek-flash",
    system_fingerprint="fp_second_pass",
    prompt_version="sha256:second_pass_test",
    input_tokens=50,
    output_tokens=20,
    called_at=datetime.now(timezone.utc),
)


async def _confirm_everything(papers, **kwargs) -> SecondPassBatchResultV2:
    """Default second-pass stub: every INCLUDE this batch reaches stays confirmed
    (quoting each paper's own title, always a verbatim substring of its own shown text), so
    every INCLUDE-producing fixture keeps its original outcome without each one having to
    mock the second pass individually. A test that wants to exercise the
    second pass's own demotion overrides ``screener_agent_module.confirm_inclusions``
    inline, the same way ``_journal_table_present`` is overridden. Patched on
    ``screener_agent_module``, not ``smart_search_module``: ``run_second_pass_stage``
    resolves ``confirm_inclusions`` from its own module's globals at call time when no
    ``judge`` is passed explicitly, so that is the only patch target that actually takes
    effect."""
    answers = [
        SecondPassAnswerV2(
            population=SecondPassSlotAnswer(
                established="established", quote=(paper.get("title") or "Paper")
            ),
            outcome=SecondPassSlotAnswer(
                established="established", quote=(paper.get("title") or "Paper")
            ),
            study_type="study",
        )
        for paper in papers
    ]
    return SecondPassBatchResultV2(answers=answers, provenance=_FAKE_SECOND_PASS_PROVENANCE)


@pytest.fixture(autouse=True)
def _second_pass_confirms_by_default():
    """``run_smart_search`` sends every surviving INCLUDE to the inclusion-only
    second pass. Default it to a no-op confirmation so the hundreds of pre-existing tests in
    this file, none of which mock the second pass, keep asserting on the batch pass's own
    decisions undisturbed."""
    with patch.object(
        screener_agent_module, "confirm_inclusions", new=AsyncMock(side_effect=_confirm_everything)
    ):
        yield


class _FakeSession:
    """Stand-in DB session — every DB helper the pipeline calls is patched out."""

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False


def _fake_session_factory() -> _FakeSession:
    return _FakeSession()


def _paper(idx: int, issn: str | None = "1234-5678") -> PaperData:
    return PaperData(
        doi=f"10.1/p{idx}",
        title=f"Paper {idx}",
        authors=[{"name": "Test Author"}],
        year=2023,
        journal_issn=issn,
        citation_count=10,
        source_api="openalex",
    )


def test_smart_search_no_longer_imports_the_per_paper_wos_lookup():
    assert not hasattr(smart_search_module, "is_wos_indexed")
    assert hasattr(smart_search_module, "is_wos_indexed_bulk")


@pytest.mark.asyncio
async def test_step_d_uses_one_bulk_wos_query_per_round():
    from uuid import uuid4

    from app.services.search import SearchService

    bulk = AsyncMock(return_value={})

    async def _fake_search(self, query, **kwargs):
        return [_paper(1, issn="1111-1111"), _paper(2, issn="2222-2222")], {"openalex": 2}

    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2", "q3"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=bulk),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=AsyncMock()),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert bulk.await_count == 1
    issns = list(bulk.await_args.args[1])
    assert issns == ["1111-1111", "2222-2222"]


@pytest.mark.asyncio
async def test_step_b_runs_the_rounds_queries_concurrently(monkeypatch):
    """SEARCH-SEQUENTIAL-QUERY-FANOUT: round time must be slowest-query, not sum-of-queries."""
    import asyncio
    from uuid import uuid4

    from app.config import settings
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "search_fanout_concurrency", 4)
    state = {"active": 0, "peak": 0}

    async def _fake_search(self, query, **kwargs):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return [_paper(int(query[1:]))], {"openalex": 1}

    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result([f"q{i}" for i in range(1, 7)])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=AsyncMock()),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert state["peak"] >= 2
    assert state["peak"] <= 4


@pytest.mark.asyncio
async def test_round_loop_stops_at_the_time_budget(monkeypatch):
    """SMART-SEARCH-CAP: with a zero budget the loop must not even start round 1's fetch."""
    from uuid import uuid4

    from app.config import settings
    from app.models.analysis_job import JobStatus
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "smart_search_max_time_minutes", 0.0)
    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        return [_paper(1)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2", "q3"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert calls == []
    final = update.await_args_list[-1]
    assert final.kwargs["status"] is JobStatus.completed
    assert final.kwargs["result"]["stop_reason"] == "time_limit"
    assert final.kwargs["result"]["rounds"] == 1


@pytest.mark.asyncio
async def test_time_budget_aborts_step_b_between_queries(monkeypatch):
    """The cap must bind INSIDE a round, not only at the round boundary."""
    import asyncio
    from uuid import uuid4

    from app.config import settings
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "search_fanout_concurrency", 1)
    monkeypatch.setattr(settings, "smart_search_max_time_minutes", 0.005)  # 0.3 s
    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        await asyncio.sleep(0.2)
        return [_paper(int(query[1:]))], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result([f"q{i}" for i in range(1, 7)])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert 1 <= len(calls) < 6
    assert update.await_args_list[-1].kwargs["result"]["stop_reason"] == "time_limit"


@pytest.mark.asyncio
async def test_cancellation_at_the_round_boundary_marks_the_job_cancelled():
    from uuid import uuid4

    from app.models.analysis_job import JobStatus
    from app.services.search import SearchService

    calls: list[str] = []

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        return [_paper(1)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2", "q3"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=True)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert calls == []
    final = update.await_args_list[-1]
    assert final.kwargs["status"] is JobStatus.cancelled
    assert final.kwargs["result"]["stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_cancellation_between_queries_marks_the_job_cancelled(monkeypatch):
    from uuid import uuid4

    from app.config import settings
    from app.models.analysis_job import JobStatus
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "search_fanout_concurrency", 1)
    calls: list[str] = []
    state = {"n": 0}

    async def _fake_search(self, query, **kwargs):
        calls.append(query)
        return [_paper(int(query[1:]))], {"openalex": 1}

    async def _abort(db, job_id) -> bool:
        state["n"] += 1
        # False for the initial boundary check, True from the first in-round check on.
        return state["n"] > 1

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result([f"q{i}" for i in range(1, 7)])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "should_abort", new=_abort),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    assert calls == []
    final = update.await_args_list[-1]
    assert final.kwargs["status"] is JobStatus.cancelled
    assert final.kwargs["result"]["stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_happy_path_completes_and_reports_included_papers():
    from uuid import uuid4

    from app.models.analysis_job import JobStatus
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx * 10), _paper(idx * 10 + 1)], {"openalex": 2}

    wos_hit = {
        "1234-5678": (True, "SSCI", "Education & Educational Research"),
    }

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(
            smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value=wos_hit)
        ),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(side_effect=[_batch([True] * 4), _batch([])]),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            project_id=uuid4(),
            job_id=uuid4(),
            query="teacher burnout",
            session_factory=_fake_session_factory,
        )

    final = update.await_args_list[-1]
    assert final.kwargs["status"] is JobStatus.completed
    assert final.kwargs["result"]["total_included"] == 4
    assert final.kwargs["result"]["total_scanned"] == 8
    assert final.kwargs["result"]["rounds"] == 2
    assert final.kwargs["result"]["stop_reason"] == "no_new_papers"
    assert final.kwargs["result"]["papers"][0]["wos_collection"] == "SSCI"
    assert final.kwargs["result"]["stage1_applied"] is True
    assert final.kwargs["result"]["papers"][0]["screening_reason"] == "relevant"


# --- Optional venue filter --------------------------------------------------------------


def _run_kwargs(**overrides):
    from uuid import uuid4

    kwargs = {
        "project_id": uuid4(),
        "job_id": uuid4(),
        "query": "teacher burnout",
        "session_factory": _fake_session_factory,
    }
    kwargs.update(overrides)
    return kwargs


def _reconciles(flow: dict) -> bool:
    return flow["identified"] == (
        flow["duplicates_removed"]
        + flow["stage1_excluded"]
        + flow["stage2_excluded"]
        + flow["needs_review"]
        + flow["unscreened"]
        + flow["included"]
    )


def _status_batch(
    statuses: list[str],
    criteria_ids: list[str] | None = None,
    quotes: list[str] | None = None,
    reasons: list[str] | None = None,
    *,
    guard_reasons: list[str] | None = None,
    to_confirm: list[list[str]] | None = None,
    model_reported: str = "deepseek-v4-flash",
    fingerprint: str | None = "fp_a",
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> ScreeningBatchResult:
    """Build the value ``screen_papers`` returns with explicit three-way statuses
    (INCLUDE/EXCLUDE/NEEDS_REVIEW), unlike ``_batch``'s legacy ``include=`` shim, which can
    only express the binary case."""
    n = len(statuses)
    if criteria_ids is None:
        criteria_ids = [""] * n
    if quotes is None:
        quotes = [""] * n
    if reasons is None:
        reasons = [f"reason {i}" for i in range(n)]
    kwargs: dict = {}
    if guard_reasons is not None:
        kwargs["guard_reasons"] = guard_reasons
    if to_confirm is not None:
        kwargs["to_confirm"] = to_confirm
    return ScreeningBatchResult(
        statuses=statuses,
        criteria_ids=criteria_ids,
        quotes=quotes,
        reasons=reasons,
        **kwargs,
        provenance=LLMCallProvenance(
            agent="relevance_screener",
            model_configured="deepseek-chat",
            model_reported=model_reported,
            provider_response_id="resp",
            system_fingerprint=fingerprint,
            temperature=0.0,
            prompt_version=SCREENER_PROMPT_VERSION,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            called_at=datetime.now(timezone.utc),
        ),
    )


@pytest.mark.asyncio
async def test_wos_filter_off_skips_stage1_and_never_reads_the_journal_table():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1, issn=None), _paper(2, issn="9999-9999")], {"openalex": 2}

    bulk = AsyncMock(return_value={})
    count = AsyncMock(return_value=0)
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=bulk),
        patch.object(smart_search_module, "get_wos_journal_count", new=count),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(side_effect=[_batch([True, True]), _batch([])]),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    bulk.assert_not_awaited()
    count.assert_not_awaited()
    result = update.await_args_list[-1].kwargs["result"]
    assert result["stage1_applied"] is False
    assert result["criteria"]["wos_filter"] == "off"
    assert result["total_included"] == 2
    assert all(p["wos_indexed"] is None for p in result["papers"])
    assert result["flow"]["stage1_screened"] == 0
    assert result["flow"]["stage1_excluded"] == 0
    assert result["stop_reason"] != "no_wos_papers"


@pytest.mark.asyncio
async def test_wos_filter_auto_with_empty_journal_table_skips_stage1():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2)], {"openalex": 2}

    bulk = AsyncMock(return_value={})
    count = AsyncMock(return_value=0)
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=bulk),
        patch.object(smart_search_module, "get_wos_journal_count", new=count),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(side_effect=[_batch([True, False]), _batch([])]),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs())  # wos_filter defaults to auto

    assert count.await_count == 1
    bulk.assert_not_awaited()
    result = update.await_args_list[-1].kwargs["result"]
    assert result["stage1_applied"] is False
    assert result["criteria"]["wos_filter"] == "auto"
    assert result["total_included"] == 1


@pytest.mark.asyncio
async def test_wos_filter_auto_reads_the_journal_count_once_per_job():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx)], {"openalex": 1}

    count = AsyncMock(return_value=5)
    lookup = {"1234-5678": (True, "SSCI", "Education")}
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                    _query_result(["q3"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(
            smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value=lookup)
        ),
        patch.object(smart_search_module, "get_wos_journal_count", new=count),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(side_effect=[_batch([True]), _batch([True]), _batch([False])]),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=AsyncMock()),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs())

    assert count.await_count == 1


# --- Three-way screening routing: INCLUDE / NEEDS_REVIEW / EXCLUDE ----


@pytest.mark.asyncio
async def test_needs_review_routes_to_needs_review_list_never_included_or_excluded():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2), _paper(3)], {"openalex": 3}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(
                return_value=_status_batch(
                    ["INCLUDE", "NEEDS_REVIEW", "EXCLUDE"],
                    criteria_ids=["", "", "E1"],
                    quotes=["", "", "off topic"],
                    reasons=[
                        "on topic",
                        "cannot decide from the shown text",
                        "different field",
                    ],
                )
            ),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]

    # INCLUDE is the only automatic route into the library.
    assert [p["doi"] for p in result["papers"]] == ["10.1/p1"]
    assert [e["doi"] for e in result["excluded"]] == ["10.1/p3"]
    assert [n["doi"] for n in result["needs_review"]] == ["10.1/p2"]

    flow = result["flow"]
    assert flow["needs_review"] == 1
    assert flow["stage2_excluded"] == 1
    assert flow["included"] == 1
    assert flow["stage2_screened"] == 3
    assert _reconciles(flow)

    # needs_review carries the full paper dict (PaperCard needs abstract/authors/etc.), not
    # the trimmed screening-record shape.
    nr = result["needs_review"][0]
    assert nr["screening_status"] == "NEEDS_REVIEW"
    assert nr["screening_reason"] == "cannot decide from the shown text"
    assert nr["screening_criterion"] == ""
    assert nr["screening_quote"] == ""
    assert "abstract" in nr

    excl = result["excluded"][0]
    assert excl["status"] == "EXCLUDE"
    assert excl["criterion"] == "E1"
    assert excl["quote"] == "off topic"


@pytest.mark.asyncio
async def test_needs_review_records_are_never_added_to_the_library_by_the_stop_condition():
    """A round where every screened record is NEEDS_REVIEW must stop like a round with no
    new INCLUDE, not be mistaken for progress."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2)], {"openalex": 2}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(
                return_value=_status_batch(["NEEDS_REVIEW", "NEEDS_REVIEW"])
            ),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert len(result["needs_review"]) == 2
    assert result["stop_reason"] == "no_new_included"
    assert _reconciles(result["flow"])


# --- min-rounds / dry-round-patience stop rule -------------------------------------


@pytest.mark.asyncio
async def test_loop_does_not_stop_for_no_new_included_before_min_rounds(monkeypatch):
    """Even though every round adds zero new inclusions, the loop must keep going until
    ``smart_search_min_rounds`` rounds have run."""
    from app.services.search import SearchService

    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 3)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 1)

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                    _query_result(["q3"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    # patience=1 would already justify stopping after round 1; min_rounds=3 overrides that.
    assert result["rounds"] == 3
    assert result["stop_reason"] == "no_new_included"
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_loop_stops_after_patience_consecutive_dry_rounds_past_min_rounds(monkeypatch):
    """Once ``smart_search_min_rounds`` rounds have run, the loop stops after
    ``smart_search_dry_round_patience`` consecutive rounds each add zero new inclusions."""
    from app.services.search import SearchService

    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 1)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 2)

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["rounds"] == 2
    assert result["stop_reason"] == "no_new_included"
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_a_round_with_a_new_inclusion_resets_the_dry_round_streak(monkeypatch):
    """A round that adds a new included paper resets the consecutive-dry-round count, so an
    isolated dry round never contributes to an unrelated later streak."""
    from app.services.search import SearchService

    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 1)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 2)

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                    _query_result(["q3"]),
                    _query_result(["q4"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(
                side_effect=[
                    _status_batch(["EXCLUDE"]),
                    _status_batch(["INCLUDE"]),
                    _status_batch(["EXCLUDE"]),
                    _status_batch(["EXCLUDE"]),
                ]
            ),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    # Round 1 dry (streak=1), round 2 includes (streak resets to 0), round 3 dry (streak=1,
    # below patience), round 4 dry (streak=2, patience reached) -- four rounds, not two.
    assert result["rounds"] == 4
    assert result["stop_reason"] == "no_new_included"
    assert result["total_included"] == 1


# --- round log -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_log_records_per_query_and_per_round_counts():
    """``provenance.rounds`` carries one entry per round: every query issued with how many
    papers it returned and how many survived dedup, plus the round's screening totals."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        if query == "q1":
            return [_paper(1), _paper(2)], {"openalex": 2}
        return [_paper(1)], {"openalex": 1}  # q2: the same paper q1 already returned

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE", "NEEDS_REVIEW"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    rounds = result["provenance"]["rounds"]

    # The second pass runs once
    # for the whole job, after the round loop ends, not once per round -- round_log carries
    # no second-pass timing of its own any more (see the job-level "second_pass_stage_wall_
    # time_s" on result["provenance"]["screener_second_pass"] instead).
    # Round 1: q1 returns 2 new papers, q2 returns 1 paper that is a duplicate of q1's.
    assert rounds[0] == {
        "round": 1,
        "queries": [
            {"query": "q1", "returned": 2, "new_unique": 2},
            {"query": "q2", "returned": 1, "new_unique": 0},
        ],
        "screened": 2,
        "included_new": 1,
        "needs_review_new": 1,
        "replayed": False,
    }

    # Round 2: the same two queries return the same (now fully duplicate) papers, so
    # nothing reaches screening -- the round is still logged, with zeroed screening counts.
    assert rounds[1] == {
        "round": 2,
        "queries": [
            {"query": "q1", "returned": 2, "new_unique": 0},
            {"query": "q2", "returned": 1, "new_unique": 0},
        ],
        "screened": 0,
        "included_new": 0,
        "needs_review_new": 0,
        "replayed": False,
    }
    assert result["stop_reason"] == "no_new_papers"
    assert result["rounds"] == 2


@pytest.mark.asyncio
async def test_round_log_provenance_states_the_stop_rule_settings(monkeypatch):
    """The exported provenance names the min-rounds/patience thresholds this run applied,
    beside the round-by-round log they explain."""
    from app.services.search import SearchService

    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 3)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 2)

    async def _fake_search(self, query, **kwargs):
        idx = int(query[1:])
        return [_paper(idx)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                    _query_result(["q3"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["provenance"]["settings"] == {"min_rounds": 3, "dry_round_patience": 2}
    assert [r["round"] for r in result["provenance"]["rounds"]] == [1, 2, 3]


# --- stage-aware criteria reach screen_papers; needs_review_by_reason -----------


@pytest.mark.asyncio
async def test_run_smart_search_forwards_criteria_stages_to_screen_papers():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    screen = AsyncMock(return_value=_status_batch(["INCLUDE"]))
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=AsyncMock()),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                wos_filter="off",
                inclusion_criteria=["uses LASSI"],
                exclusion_criteria=["did not use LASSI"],
                inclusion_criteria_stages=["full_text"],
                exclusion_criteria_stages=["full_text"],
            )
        )

    assert screen.await_args.kwargs["inclusion_stages"] == ["full_text"]
    assert screen.await_args.kwargs["exclusion_stages"] == ["full_text"]


@pytest.mark.asyncio
async def test_run_smart_search_defaults_criteria_stages_to_none():
    """No stages given (every pre-S8 caller): screen_papers still receives the keyword,
    forwarded as None, so its own "abstract" default applies."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    screen = AsyncMock(return_value=_status_batch(["INCLUDE"]))
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=AsyncMock()),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    assert screen.await_args.kwargs["inclusion_stages"] is None
    assert screen.await_args.kwargs["exclusion_stages"] is None


@pytest.mark.asyncio
async def test_criteria_block_carries_the_criteria_stages_when_given():
    """The job result's criteria block echoes the
    two stage arrays, the one input needed to reproduce the routing in the exported
    screening record."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    screen = AsyncMock(return_value=_status_batch(["INCLUDE"]))
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                wos_filter="off",
                inclusion_criteria=["uses LASSI"],
                exclusion_criteria=["did not use LASSI"],
                inclusion_criteria_stages=["full_text"],
                exclusion_criteria_stages=["full_text"],
            )
        )

    result = update.await_args_list[-1].kwargs["result"]
    assert result["criteria"]["inclusion_criteria_stages"] == ["full_text"]
    assert result["criteria"]["exclusion_criteria_stages"] == ["full_text"]


@pytest.mark.asyncio
async def test_criteria_block_defaults_criteria_stages_to_empty_lists_when_not_given():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    screen = AsyncMock(return_value=_status_batch(["INCLUDE"]))
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["criteria"]["inclusion_criteria_stages"] == []
    assert result["criteria"]["exclusion_criteria_stages"] == []


@pytest.mark.asyncio
async def test_needs_review_by_reason_buckets_by_guard_reason_and_no_abstract():
    from app.services.search import SearchService

    def _paper_with_abstract(idx: int, abstract: str) -> PaperData:
        dumped = _paper(idx).model_dump(mode="json")
        dumped["abstract"] = abstract
        return PaperData(**dumped)

    async def _fake_search(self, query, **kwargs):
        return [
            _paper_with_abstract(1, "some text"),
            _paper_with_abstract(2, "some text"),
            _paper_with_abstract(3, "some text"),
        ], {"openalex": 3}

    batch = _status_batch(
        ["NEEDS_REVIEW", "NEEDS_REVIEW", "NEEDS_REVIEW"],
        criteria_ids=["E1", "", ""],
        reasons=["routed", "no abstract shown", "cannot decide"],
        guard_reasons=["full_text_criterion", "", ""],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["needs_review_by_reason"] == {
        "full_text_criterion": 1,
        "undecidable": 2,
    }
    assert result["needs_review"][0]["screening_guard_reason"] == "full_text_criterion"


@pytest.mark.asyncio
async def test_needs_review_by_reason_buckets_a_missing_abstract_separately():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        paper = _paper(1)
        dumped = paper.model_dump(mode="json")
        dumped["abstract"] = None
        return [PaperData(**dumped)], {"openalex": 1}

    batch = _status_batch(["NEEDS_REVIEW"], reasons=["no abstract"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["needs_review_by_reason"] == {"no_abstract": 1}


@pytest.mark.asyncio
async def test_needs_review_by_reason_buckets_the_guards_own_no_abstract_reason():
    """The screener's guard can itself attribute ``guard_reason == "no_abstract"``
    (an EXCLUDE on a record with no abstract, demoted unconditionally) -- this lands in the
    same "no_abstract" bucket as the guard-silent fallback the previous test covers, since
    ``guard_reason`` alone is authoritative here and this value merges into the
    fallback's bucket name without any change to the bucketing code."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        paper = _paper(1)
        dumped = paper.model_dump(mode="json")
        dumped["abstract"] = None
        return [PaperData(**dumped)], {"openalex": 1}

    batch = _status_batch(
        ["NEEDS_REVIEW"],
        criteria_ids=["I2"],
        reasons=["no abstract shown"],
        guard_reasons=["no_abstract"],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["needs_review_by_reason"] == {"no_abstract": 1}
    assert result["needs_review"][0]["screening_guard_reason"] == "no_abstract"


@pytest.mark.asyncio
async def test_needs_review_by_reason_trusts_guard_reason_alone_not_criterion_membership():
    """The guard always attributes "full_text_criterion" or
    "unquoted_criterion" on a NEEDS_REVIEW naming a full-text exclusion criterion, whether or
    not it demoted anything, rather than staying silent on a compliant model routing.
    So ``run_smart_search`` no longer needs a criterion-membership fallback: a
    NEEDS_REVIEW naming a full-text id with an empty guard_reason (a shape the real guard can
    no longer produce) buckets as an ordinary undecided record, not "full_text_criterion"."""
    from app.services.search import SearchService

    def _paper_with_abstract(idx: int, abstract: str) -> PaperData:
        dumped = _paper(idx).model_dump(mode="json")
        dumped["abstract"] = abstract
        return PaperData(**dumped)

    async def _fake_search(self, query, **kwargs):
        return [_paper_with_abstract(1, "some text")], {"openalex": 1}

    batch = _status_batch(
        ["NEEDS_REVIEW"],
        criteria_ids=["E1"],
        reasons=["needs the full paper"],
        guard_reasons=[""],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                wos_filter="off",
                exclusion_criteria=["cannot be judged from the abstract"],
                exclusion_criteria_stages=["full_text"],
            )
        )

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["needs_review_by_reason"] == {"undecidable": 1}
    assert result["needs_review"][0]["screening_guard_reason"] == ""


@pytest.mark.asyncio
async def test_needs_review_by_reason_buckets_unquoted_criterion():
    """The fifth guard reason reaches ``flow.needs_review_by_reason``
    exactly like the other four."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(
        ["NEEDS_REVIEW"],
        criteria_ids=["E1"],
        reasons=["names a criterion, no quote"],
        guard_reasons=["unquoted_criterion"],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["needs_review_by_reason"] == {"unquoted_criterion": 1}
    assert result["needs_review"][0]["screening_guard_reason"] == "unquoted_criterion"


@pytest.mark.asyncio
async def test_included_paper_carries_screening_to_confirm():
    """An INCLUDE's ``to_confirm`` list reaches the paper dict as
    ``screening_to_confirm``, and a non-INCLUDE decision (which ``screen_papers`` always
    forces to ``[]``) carries an empty list too."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2)], {"openalex": 2}

    batch = _status_batch(
        ["INCLUDE", "EXCLUDE"],
        criteria_ids=["", "E1"],
        quotes=["", "off topic"],
        to_confirm=[["I1", "I3"], []],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"][0]["screening_to_confirm"] == ["I1", "I3"]
    assert result["excluded"][0]["criterion"] == "E1"


# --- Production wiring of the stage-aware screener design (deterministic post-model
# devices, the inclusion-only second pass, and their reach into the screening record) -------


def _paper_with_type(
    idx: int, *, openalex_type: str | None = None, is_paratext: bool = False
) -> PaperData:
    dumped = _paper(idx).model_dump(mode="json")
    dumped["openalex_type"] = openalex_type
    dumped["is_paratext"] = is_paratext
    return PaperData(**dumped)


@pytest.mark.asyncio
async def test_nonarticle_type_demotes_an_include_to_needs_review():
    """apply_type_demotion runs on every record screen_papers still returns as INCLUDE, using
    the OpenAlex type/is_paratext carried on the paper dict."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper_with_type(1, openalex_type="book")], {"openalex": 1}

    batch = _status_batch(["INCLUDE"], reasons=["on topic"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert len(result["needs_review"]) == 1
    assert result["needs_review"][0]["screening_status"] == "NEEDS_REVIEW"
    assert result["needs_review"][0]["screening_guard_reason"] == "nonarticle_type"
    assert result["flow"]["needs_review_by_reason"] == {"nonarticle_type": 1}
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_is_paratext_demotes_an_include_to_needs_review():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper_with_type(1, openalex_type="article", is_paratext=True)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["needs_review"][0]["screening_guard_reason"] == "nonarticle_type"


@pytest.mark.asyncio
async def test_ordinary_article_type_is_not_demoted():
    """The narrow type list deliberately excludes ``article`` (and ``conference-paper``,
    ``dissertation``, ``report``, ``other``): an ordinary article stays INCLUDE through this
    check."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper_with_type(1, openalex_type="article")], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert len(result["papers"]) == 1
    assert result["needs_review"] == []


@pytest.mark.asyncio
async def test_table_of_contents_abstract_demotes_an_include_to_needs_review():
    """A chapter-list abstract demotes an INCLUDE,
    independent of any OpenAlex type."""
    from app.services.search import SearchService

    def _paper_with_abstract(idx: int, abstract: str) -> PaperData:
        dumped = _paper(idx).model_dump(mode="json")
        dumped["abstract"] = abstract
        return PaperData(**dumped)

    toc_abstract = (
        "1. Introduction 2. Background 3. Method 4. Results 5. Discussion 6. Conclusion"
    )

    async def _fake_search(self, query, **kwargs):
        return [_paper_with_abstract(1, toc_abstract)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert result["needs_review"][0]["screening_guard_reason"] == "table_of_contents"
    assert result["flow"]["needs_review_by_reason"] == {"table_of_contents": 1}


@pytest.mark.asyncio
async def test_second_pass_demotes_an_include_on_not_established_outcome():
    """confirm_inclusions is called on every survivor, and apply_second_pass_guard's demotion
    reaches the job result and flow.needs_review_by_reason."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"], reasons=["on topic"])

    async def _not_established_outcome(papers, **kwargs):
        return SecondPassBatchResultV2(
            answers=[
                SecondPassAnswerV2(
                    population=SecondPassSlotAnswer(
                        established="established", quote=papers[0]["title"],
                    ),
                    outcome=SecondPassSlotAnswer(established="not_established", quote=""),
                    study_type="study",
                )
            ],
            provenance=_FAKE_SECOND_PASS_PROVENANCE,
        )

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(
            screener_agent_module,
            "confirm_inclusions",
            new=AsyncMock(side_effect=_not_established_outcome),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                wos_filter="off", inclusion_criteria=["a"], exclusion_criteria=["b"],
            ),
        )

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    review = result["needs_review"][0]
    assert review["screening_status"] == "NEEDS_REVIEW"
    assert review["screening_guard_reason"] == "not_established"
    assert review["screening_second_pass"] == {
        "population": "established", "outcome": "not_established", "study_type": "study",
    }
    assert result["flow"]["needs_review_by_reason"] == {"not_established": 1}
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_second_pass_records_its_answer_on_a_confirmed_include():
    """A record the second pass confirms stays INCLUDE, and its per-question answer is
    still recorded on screening_second_pass for the screening record export."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert len(result["papers"]) == 1
    assert result["papers"][0]["screening_second_pass"] == {
        "population": "established", "outcome": "established", "study_type": "study",
    }


@pytest.mark.asyncio
async def test_second_pass_call_failure_routes_to_needs_review_not_silently_included():
    """A second pass that fails outright must never silently ship its batch as INCLUDE
    (the same "never default to a lenient outcome silently" rule the batch pass's own
    failure follows, except these records already carry a real primary decision, so they
    are queued rather than marked unscreened)."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(
            screener_agent_module,
            "confirm_inclusions",
            new=AsyncMock(side_effect=ScreeningError("TimeoutError: model timed out")),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert len(result["unscreened"]) == 0
    assert result["needs_review"][0]["screening_guard_reason"] == "second_pass_unavailable"
    assert result["flow"]["needs_review_by_reason"] == {"second_pass_unavailable": 1}
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_a_failed_chunks_error_is_logged_once_per_chunk_not_once_per_record(caplog):
    """A failed chunk covers several records at once
    (SECOND_PASS_STAGE_BATCH_SIZE, 5 by default) and every one of them carries the exact
    same error text (run_second_pass_stage's own _route_unavailable assigns one f-string to
    every index in the chunk) -- the job's own "second pass could not reach a record" log
    line must appear once per chunk, not once per record the chunk happened to cover."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(i) for i in range(5)], {"openalex": 5}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"] * 5)),
        ),
        patch.object(
            screener_agent_module,
            "confirm_inclusions",
            new=AsyncMock(side_effect=ScreeningError("TimeoutError: model timed out")),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        with caplog.at_level(logging.WARNING, logger="app.services.smart_search"):
            await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert len(result["needs_review"]) == 5
    assert all(
        r["screening_guard_reason"] == "second_pass_unavailable" for r in result["needs_review"]
    )
    reach_lines = [
        r for r in caplog.records if "second pass could not reach a record" in r.message
    ]
    assert len(reach_lines) == 1


@pytest.mark.asyncio
async def test_second_pass_provenance_is_reported_separately_from_the_batch_pass():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    second_pass = update.await_args_list[-1].kwargs["result"]["provenance"]["screener_second_pass"]
    assert second_pass["calls"] == 1
    assert second_pass["model_reported"] == ["deepseek-flash"]
    assert second_pass["input_tokens"] == 50
    # The stage's own block -- one
    # candidate judged, confirmed (0 demotions, 0 unavailable), one call's own latency, and
    # a stage wall time that is a real elapsed duration, not a fixed value.
    assert second_pass["candidates"] == 1
    assert second_pass["demotions"] == 0
    assert second_pass["unavailable"] == 0
    assert len(second_pass["per_call_latencies_s"]) == 1
    assert isinstance(second_pass["stage_wall_time_s"], float) and second_pass["stage_wall_time_s"] >= 0.0


@pytest.mark.asyncio
async def test_second_pass_tokens_are_recorded_per_call_summed_in_the_stage_block_and_in_the_screening_record():
    """A provider can report a real call with a ``usage`` field of ``None``; traced to
    ``pydantic_ai.models.openai._map_usage`` returning an all-zero ``RequestUsage()`` in
    that case, not a wiring defect in this module (``_SecondPassProvenance.add``, already
    summing correctly whenever the judge reports real tokens -- see the two tests directly
    above). This mocked-model test proves the whole path end to end with two chunks
    reporting DIFFERENT token counts -- one with
    real usage, one with none (the exact shape a provider that drops usage produces) -- and
    asserts the stage block sums them, tells the two calls apart
    (``calls_missing_usage``), and that the per-record attribution this fix adds reaches
    both ``result["papers"]`` and the exported screening record (``screening_record.py``),
    not just the job-level total."""
    from app.services.screening_record import build_screening_record, screening_record_csv
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(i) for i in range(1, 7)], {"openalex": 6}

    async def _confirm_two_chunks_different_usage(papers, **kwargs):
        # SECOND_PASS_STAGE_BATCH_SIZE (5) splits six candidates into a chunk
        # of five and a chunk of one; chunks run concurrently (asyncio.gather), so which one
        # ``confirm_inclusions`` is actually invoked for first is not guaranteed -- keyed on
        # each chunk's own content (its size), never on call order, so this test cannot be
        # flaky under either scheduling.
        if len(papers) == 5:
            provenance = LLMCallProvenance(
                agent="relevance_screener_second_pass_v2",
                model_configured="deepseek-flash",
                model_reported="deepseek-flash",
                system_fingerprint="fp_second_pass",
                prompt_version="sha256:second_pass_test",
                input_tokens=800,
                output_tokens=150,
                called_at=datetime.now(timezone.utc),
            )
        else:
            # The provider-omitted-usage shape this fix makes visible: a real call (it
            # confirms every candidate below, same as the other chunk), reported with zero
            # tokens because the raw response carried no usage field at all.
            provenance = LLMCallProvenance(
                agent="relevance_screener_second_pass_v2",
                model_configured="deepseek-flash",
                model_reported="deepseek-flash",
                system_fingerprint="fp_second_pass",
                prompt_version="sha256:second_pass_test",
                input_tokens=0,
                output_tokens=0,
                called_at=datetime.now(timezone.utc),
            )
        answers = [
            SecondPassAnswerV2(
                population=SecondPassSlotAnswer(
                    established="established", quote=(paper.get("title") or "Paper")
                ),
                outcome=SecondPassSlotAnswer(
                    established="established", quote=(paper.get("title") or "Paper")
                ),
                study_type="study",
            )
            for paper in papers
        ]
        return SecondPassBatchResultV2(answers=answers, provenance=provenance)

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"] * 6)),
        ),
        patch.object(
            screener_agent_module, "confirm_inclusions",
            new=AsyncMock(side_effect=_confirm_two_chunks_different_usage),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    second_pass = result["provenance"]["screener_second_pass"]

    # Stage block: summed correctly, and the two calls' own tokens still distinguishable.
    assert second_pass["calls"] == 2
    assert second_pass["input_tokens"] == 800
    assert second_pass["output_tokens"] == 150
    assert sorted(second_pass["per_call_input_tokens"]) == [0, 800]
    assert sorted(second_pass["per_call_output_tokens"]) == [0, 150]
    assert len(second_pass["per_call_latencies_s"]) == 2
    assert second_pass["calls_missing_usage"] == 1

    # Per-record attribution: every one of the six candidates carries its own chunk's call
    # info, not just the job-level total.
    papers_by_doi = {p["doi"]: p for p in result["papers"]}
    assert set(papers_by_doi) == {f"10.1/p{i}" for i in range(1, 7)}
    five_papers = [p for p in papers_by_doi.values() if p["screening_second_pass_call"]["input_tokens"] == 800]
    one_paper = [p for p in papers_by_doi.values() if p["screening_second_pass_call"]["input_tokens"] == 0]
    assert len(five_papers) == 5
    assert len(one_paper) == 1
    for paper in five_papers:
        assert paper["screening_second_pass_call"]["output_tokens"] == 150
        assert paper["screening_second_pass_call"]["model_reported"] == "deepseek-flash"
    assert one_paper[0]["screening_second_pass_call"]["output_tokens"] == 0

    # The exported screening record (JSON and CSV) carries the same per-record figures --
    # "in the screening record", not only in the job's own raw result.
    record = build_screening_record(result)
    rows_by_doi = {r["doi"]: r for r in record["records"] if r["outcome"] == "included"}
    assert {rows_by_doi[doi]["second_pass_input_tokens"] for doi in rows_by_doi} == {800, 0}
    zero_doi = one_paper[0]["doi"]
    assert rows_by_doi[zero_doi]["second_pass_input_tokens"] == 0
    assert rows_by_doi[zero_doi]["second_pass_output_tokens"] == 0
    csv_header = screening_record_csv(result).splitlines()[0]
    assert "second_pass_input_tokens" in csv_header
    assert "second_pass_output_tokens" in csv_header
    assert "second_pass_latency_s" in csv_header


@pytest.mark.asyncio
async def test_second_pass_provenance_is_empty_when_nothing_reaches_it():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["EXCLUDE"])
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    second_pass = update.await_args_list[-1].kwargs["result"]["provenance"]["screener_second_pass"]
    assert second_pass["calls"] == 0
    assert second_pass["model_reported"] == []
    assert second_pass["candidates"] == 0
    assert second_pass["demotions"] == 0
    assert second_pass["unavailable"] == 0
    assert second_pass["per_call_latencies_s"] == []
    assert second_pass["stage_wall_time_s"] == 0.0


@pytest.mark.asyncio
async def test_second_pass_is_a_single_job_level_call_over_every_rounds_candidates():
    """The second pass runs once for
    the whole job, after the round loop ends, over every round's own provisional
    INCLUDE -- not once per round. Two rounds each contribute one candidate; the job-level
    stage must see both in a single call, not two separate ones."""
    from app.services.search import SearchService

    call_count = {"n": 0}

    async def _fake_search(self, query, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [_paper(1)], {"openalex": 1}
        if call_count["n"] == 2:
            return [_paper(2)], {"openalex": 1}
        return [], {"openalex": 0}

    seen_batches: list[int] = []

    async def _counting_confirm(papers, **kwargs):
        seen_batches.append(len(papers))
        return await _confirm_everything(papers, **kwargs)

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"])),
        ),
        patch.object(screener_agent_module, "confirm_inclusions", new=AsyncMock(side_effect=_counting_confirm)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    # One call, covering both rounds' own candidates together -- not one call per round.
    assert seen_batches == [2]
    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["included"] == 2
    assert result["provenance"]["screener_second_pass"]["calls"] == 1
    assert result["provenance"]["screener_second_pass"]["candidates"] == 2


@pytest.mark.asyncio
async def test_provisional_include_is_visible_to_a_later_rounds_query_generation():
    """During the round loop, a pass-1
    INCLUDE is a provisional include: a later round's own query generation sees it via
    existing_papers, and the earlier round's own stopping decision (Step F) is not gated on
    the second pass, which runs once only after the whole loop ends. Here the second pass
    later demotes the only candidate, confirming the provisional status was visible to round
    2 regardless of that later outcome."""
    from app.services.search import SearchService

    call_count = {"n": 0}

    async def _fake_search(self, query, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [_paper(1)], {"openalex": 1}
        return [], {"openalex": 0}

    generate_mock = AsyncMock(
        side_effect=[_query_result(["q1"]), _query_result(["q2"])]
    )

    async def _not_established(papers, **kwargs):
        return SecondPassBatchResultV2(
            answers=[
                SecondPassAnswerV2(
                    population=SecondPassSlotAnswer(
                        established="established", quote=papers[0]["title"],
                    ),
                    outcome=SecondPassSlotAnswer(established="not_established", quote=""),
                    study_type="study",
                )
            ],
            provenance=_FAKE_SECOND_PASS_PROVENANCE,
        )

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(smart_search_module, "generate_search_queries_with_provenance", new=generate_mock),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(
            screener_agent_module, "confirm_inclusions", new=AsyncMock(side_effect=_not_established),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    # Round 2's own query generation call saw round 1's provisional include's own title,
    # even though the second pass (run once, after the whole loop) later demotes it.
    round2_call = generate_mock.await_args_list[1]
    assert round2_call.kwargs["existing_papers"] == ["Paper 1"]

    result = update.await_args_list[-1].kwargs["result"]
    assert result["flow"]["included"] == 0
    assert result["flow"]["needs_review"] == 1
    assert result["needs_review"][0]["screening_guard_reason"] == "not_established"

    # The stage's own demotion must not be booked against
    # whichever round happened to run last. Round 1 produced the only candidate and keeps
    # its own provisional included_new; round 2 screened nothing and must not show a
    # needs_review_new the stage demoted well after round 2 itself had already finished.
    rounds = result["provenance"]["rounds"]
    assert rounds[0]["included_new"] == 1
    assert rounds[0]["needs_review_new"] == 0
    assert rounds[1]["screened"] == 0
    assert rounds[1]["needs_review_new"] == 0
    assert sum(r["included_new"] for r in rounds) == 1


@pytest.mark.asyncio
async def test_second_pass_stage_budget_is_independent_of_the_search_time_budget():
    """screener_second_pass_stage_budget_seconds is the stage's
    own ceiling, entirely separate from smart_search_max_time_minutes -- an already-expired
    stage budget demotes every candidate even though the search itself completed normally,
    well within its own (generous, default) time budget."""
    from app.config import settings
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    batch = _status_batch(["INCLUDE"])
    update = AsyncMock()
    with (
        patch.object(settings, "screener_second_pass_stage_budget_seconds", -1.0),
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["stop_reason"] != "time_limit"
    assert result["flow"]["included"] == 0
    assert result["needs_review"][0]["screening_guard_reason"] == "second_pass_unavailable"
    assert result["provenance"]["screener_second_pass"]["unavailable"] == 1


@pytest.mark.asyncio
async def test_cancelled_before_the_stage_never_calls_the_judge():
    """A job cancelled during the round loop must not wait on
    the judge stage at all -- every provisional include from the round(s) already run is
    queued for a human instead, with no judge call made and no further waiting."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    # Cancellation is only noticed once round 1's own screening has actually produced its
    # provisional include -- not tied to a raw call count, since Step B's own per-query
    # stop_check also calls should_abort and would otherwise make this fragile.
    cancel_flag = {"go": False}

    async def _dynamic_should_abort(db, job_id):
        return cancel_flag["go"]

    async def _screen_then_flip(query, batch, *args, **kwargs):
        cancel_flag["go"] = True  # round 1's own candidate is already provisional now
        return _status_batch(["INCLUDE"])

    judge_mock = AsyncMock(side_effect=_confirm_everything)
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers", new=AsyncMock(side_effect=_screen_then_flip),
        ),
        patch.object(screener_agent_module, "confirm_inclusions", new=judge_mock),
        patch.object(smart_search_module, "should_abort", new=_dynamic_should_abort),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    judge_mock.assert_not_awaited()
    result = update.await_args_list[-1].kwargs["result"]
    assert result["stop_reason"] == "cancelled"
    assert result["flow"]["included"] == 0
    assert result["flow"]["needs_review"] == 1
    assert result["needs_review"][0]["screening_guard_reason"] == "second_pass_unavailable"
    assert result["provenance"]["screener_second_pass"]["unavailable"] == 1
    assert result["provenance"]["screener_second_pass"]["calls"] == 0


@pytest.mark.asyncio
async def test_cancellation_mid_stage_stops_launching_further_chunks():
    """Cancellation noticed while the stage is already running
    must not wait out the rest of the stage's own budget -- a chunk already in flight still
    finishes (a call already paid for is not abandoned), but no further chunk is launched,
    and the job finishes as cancelled rather than completed."""
    from app.config import settings
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(i) for i in range(10)], {"openalex": 10}

    cancel_after_first_call = {"go": False}

    async def _dynamic_should_abort(db, job_id):
        return cancel_after_first_call["go"]

    async def _confirm_then_flip(papers, **kwargs):
        result = await _confirm_everything(papers, **kwargs)
        cancel_after_first_call["go"] = True  # the next chunk's own check now sees True
        return result

    update = AsyncMock()
    with (
        patch.object(settings, "screener_second_pass_concurrency", 1),  # forces serial chunks
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"] * 10)),
        ),
        patch.object(
            screener_agent_module, "confirm_inclusions",
            new=AsyncMock(side_effect=_confirm_then_flip),
        ),
        patch.object(smart_search_module, "should_abort", new=_dynamic_should_abort),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["stop_reason"] == "cancelled"
    # The first chunk (5 records) was already in flight and completed; the second chunk (5
    # records) was never launched once cancellation was noticed.
    assert result["flow"]["included"] == 5
    assert result["flow"]["needs_review"] == 5
    assert result["provenance"]["screener_second_pass"]["calls"] == 1
    assert result["provenance"]["screener_second_pass"]["unavailable"] == 5


@pytest.mark.asyncio
async def test_stage_emits_a_progress_update_before_starting_and_after_each_chunk():
    """Without a progress update, the interface shows nothing
    happening ("Round N: ...") for as long as the stage takes, up to its own budget. One
    update must land before the stage's first call, and one more after each chunk."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(i) for i in range(10)], {"openalex": 10}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"] * 10)),
        ),
        patch.object(
            screener_agent_module, "confirm_inclusions", new=AsyncMock(side_effect=_confirm_everything),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    messages = [
        call.kwargs.get("progress_message", "") for call in update.await_args_list
    ]
    confirming_messages = [m for m in messages if m and m.startswith("Confirming")]
    # One before the stage's first call ("0/2 judged"), one after each of the two chunks
    # (10 candidates / 5 per chunk) -- at least three "Confirming" updates in total.
    assert len(confirming_messages) >= 3
    assert any("0/2 judged" in m for m in confirming_messages)
    assert any("2/2 judged" in m for m in confirming_messages)


@pytest.mark.asyncio
async def test_a_raising_progress_update_does_not_fail_the_stage_or_discard_the_result():
    """The second-pass stage's own two progress-update
    callbacks (the pre-stage update and the per-chunk on_chunk_done) exist only to drive
    the interface's own "N/M judged" line -- a transient database error there must be
    logged and the stage must continue, and a search that otherwise completed must never
    be discarded because one of those writes failed. Only the "Confirming ..." progress
    writes raise here; the job's own final completion write must still land, carrying the
    full, correct result."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(i) for i in range(10)], {"openalex": 10}

    landed: list[dict] = []

    async def _flaky_update(db, job_id, **kwargs):
        if (kwargs.get("progress_message") or "").startswith("Confirming"):
            raise RuntimeError("transient database error")
        landed.append(kwargs)

    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers",
            new=AsyncMock(return_value=_status_batch(["INCLUDE"] * 10)),
        ),
        patch.object(
            screener_agent_module, "confirm_inclusions", new=AsyncMock(side_effect=_confirm_everything),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=_flaky_update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    # Every "Confirming ..." write raised and was swallowed; the job's own final
    # completion write is the only one that landed, and it carries the real result -- the
    # stage's own screening outcome is unaffected by its progress display failing.
    assert landed, "the final completion write must still have landed"
    result = landed[-1]["result"]
    assert result["stop_reason"] != "error"
    assert result["flow"]["included"] == 10
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_full_text_to_confirm_reaches_the_screening_record_end_to_end():
    """One end-to-end test at the smart_search and screening_record layer, pinning the
    contract that an INCLUDE the batch pass's own guard
    already demoted for an unconfirmed full-text inclusion criterion (guard reason
    "full_text_to_confirm") is never shipped as included, and a
    reviewer reading the exported screening record sees both why it was queued and which
    criteria it still needs."""
    from app.services.search import SearchService
    from app.services.screening_record import build_screening_record

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    # This is exactly what the real apply_decision_guard returns for an INCLUDE naming an
    # unconfirmed full-text inclusion criterion -- screen_papers is mocked (as it is
    # throughout this file), but the shape mocked here is the guard's own documented
    # contract, not an invented one.
    batch = _status_batch(
        ["NEEDS_REVIEW"],
        reasons=["on topic (guard: INCLUDE routed to NEEDS_REVIEW -- one or more full-text "
                 "inclusion criteria are not yet confirmed by the shown text)"],
        guard_reasons=["full_text_to_confirm"],
        to_confirm=[["I1"]],
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=AsyncMock(return_value=batch)),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                wos_filter="off",
                inclusion_criteria=["a full-text criterion"],
                inclusion_criteria_stages=["full_text"],
            ),
        )

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert result["flow"]["included"] == 0
    assert result["flow"]["needs_review"] == 1
    assert result["flow"]["needs_review_by_reason"] == {"full_text_to_confirm": 1}
    assert _reconciles(result["flow"])

    exported = build_screening_record(result)
    row = next(r for r in exported["records"] if r["outcome"] == "needs_review")
    assert row["needs_review_reason"] == "full_text_to_confirm"
    assert row["to_confirm"] == "I1"
    assert row["status"] == "NEEDS_REVIEW"


# --- Screening record: excluded / unscreened / flow / provenance --------------------------


@pytest.mark.asyncio
async def test_failed_screening_batch_lands_in_unscreened_not_in_papers():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2), _paper(3)], {"openalex": 3}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(side_effect=ScreeningError("TimeoutError: model timed out")),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert result["total_included"] == 0
    assert len(result["unscreened"]) == 3
    entry = result["unscreened"][0]
    assert entry["stage"] == "llm"
    assert entry["reason"] == "ScreeningError: TimeoutError: model timed out"
    assert entry["title"] == "Paper 1"
    assert entry["doi"] == "10.1/p1"
    assert result["flow"]["unscreened"] == 3
    assert result["flow"]["included"] == 0
    assert result["flow"]["stage2_screened"] == 0
    assert _reconciles(result["flow"])
    # The search stopped because screening failed, not because the data ran dry.
    assert result["stop_reason"] == "screening_failed"
    assert result["flow"]["stop_reason"] == "screening_failed"
    final = update.await_args_list[-1].kwargs
    assert final["status"] == JobStatus.completed
    assert "3 unscreened" in final["progress_message"]
    assert "screening failed" in final["progress_message"]
    # A batch that failed contributes no provenance.
    assert result["provenance"]["screener"]["calls"] == 0
    assert result["provenance"]["screener"]["model_reported"] == []


@pytest.mark.asyncio
async def test_retrieval_failure_on_every_query_is_reported_not_disguised_as_no_new_papers():
    """Observed 2026-09-03: OpenAlex answered every query with HTTP 429 (daily budget
    exhausted) and the job reported ``completed / no_new_papers`` with 0 identified, which
    reads as 'the literature ran dry'. The record must say the retrieval failed."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        raise RuntimeError("429 Too Many Requests: Insufficient budget, resets at midnight UTC")

    update = AsyncMock()
    screen = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2", "q3"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    final = update.await_args_list[-1].kwargs
    result = final["result"]
    assert final["status"] == JobStatus.completed
    assert result["stop_reason"] == "retrieval_failed"
    assert result["flow"]["stop_reason"] == "retrieval_failed"
    assert result["flow"]["identified"] == 0
    assert result["papers"] == []
    assert _reconciles(result["flow"])
    # The cause is carried to the UI / task message, not only to the server log.
    assert "retrieval failed" in final["progress_message"]
    assert "3 of 3 queries" in final["progress_message"]
    assert "Insufficient budget" in final["progress_message"]
    # And it is part of the stored record for the export.
    assert result["retrieval_failures"] == [
        ("q1", "RuntimeError: 429 Too Many Requests: Insufficient budget, resets at midnight UTC"),
        ("q2", "RuntimeError: 429 Too Many Requests: Insufficient budget, resets at midnight UTC"),
        ("q3", "RuntimeError: 429 Too Many Requests: Insufficient budget, resets at midnight UTC"),
    ]
    screen.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_retrieval_failure_with_results_still_screens_and_records_the_failures():
    """One query out of three failed but the others returned papers: the round proceeds
    and the failures are recorded, without changing the stop reason."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        if query == "q2":
            raise RuntimeError("timeout")
        return [_paper(1), _paper(2)], {"openalex": 2}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2", "q3"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_batch([True, False])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["stop_reason"] != "retrieval_failed"
    assert result["flow"]["included"] == 1
    # Recorded once per round (the same query set is re-issued each round).
    assert result["rounds"] >= 1
    assert result["retrieval_failures"] == [("q2", "RuntimeError: timeout")] * result["rounds"]


@pytest.mark.asyncio
async def test_screening_failure_in_a_later_round_keeps_earlier_results_and_names_the_cause():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        if query == "q1":
            return [_paper(1), _paper(2)], {"openalex": 2}
        return [_paper(3), _paper(4)], {"openalex": 2}

    screen = AsyncMock(
        side_effect=[
            _batch([True, True]),
            ScreeningError("TimeoutError: model timed out"),
        ]
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                    _query_result(["q3"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["rounds"] == 2  # round 3 never ran: the loop must not spin on a failing model
    assert result["stop_reason"] == "screening_failed"
    assert [p["doi"] for p in result["papers"]] == ["10.1/p1", "10.1/p2"]
    assert [e["doi"] for e in result["unscreened"]] == ["10.1/p3", "10.1/p4"]
    assert result["flow"]["included"] == 2
    assert result["flow"]["unscreened"] == 2
    assert _reconciles(result["flow"])
    assert result["provenance"]["screener"]["calls"] == 1


@pytest.mark.asyncio
async def test_partial_batch_failure_with_new_included_papers_does_not_stop_the_search(
    monkeypatch,
):
    """Only a round with no new included papers stops the loop; a failed batch alongside a
    successful one is recorded as unscreened and the search continues."""
    from app.config import settings
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "smart_search_batch_size", 2)

    async def _fake_search(self, query, **kwargs):
        if query == "q1":
            return [_paper(1), _paper(2), _paper(3), _paper(4)], {"openalex": 4}
        return [_paper(1)], {"openalex": 1}  # round 2: only duplicates

    screen = AsyncMock(
        side_effect=[
            _batch([True, False]),
            ScreeningError("TimeoutError: model timed out"),
        ]
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1"]),
                    _query_result(["q2"]),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["rounds"] == 2
    assert result["stop_reason"] == "no_new_papers"
    assert result["flow"]["included"] == 1
    assert result["flow"]["stage2_excluded"] == 1
    assert result["flow"]["unscreened"] == 2
    assert _reconciles(result["flow"])


@pytest.mark.asyncio
async def test_mismatched_batch_result_is_unscreened_but_its_model_call_is_still_counted():
    """A result whose lists are not parallel to the batch is unusable, yet the model call
    happened: tokens, model and fingerprint must still appear in the provenance."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2), _paper(3)], {"openalex": 3}

    mismatched = ScreeningBatchResult(
        include=[True, False],
        reasons=["r1", "r2"],
        provenance=_batch([True]).provenance,
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module, "screen_papers", new=AsyncMock(return_value=mismatched)
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["papers"] == []
    assert len(result["unscreened"]) == 3
    assert all(
        e["reason"] == "ScreeningError: decision count does not match the batch size"
        for e in result["unscreened"]
    )
    assert result["stop_reason"] == "screening_failed"
    assert _reconciles(result["flow"])
    screener = result["provenance"]["screener"]
    assert screener["calls"] == 1
    assert screener["input_tokens"] == 100
    assert screener["model_reported"] == ["deepseek-v4-flash"]


@pytest.mark.asyncio
async def test_flow_counts_reconcile_and_records_carry_stage_and_reason():
    from app.agents.query_generator_agent import QUERY_GENERATOR_PROMPT
    from app.config import settings
    from app.schemas.provenance import prompt_version
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        # q1 and q2 both return paper 1 -> one duplicate removed by DOI in round 1.
        if query == "q1":
            return [_paper(1), _paper(2, issn="9999-9999"), _paper(3)], {"openalex": 3}
        return [_paper(1), _paper(4)], {"openalex": 2}

    lookup = {
        "1234-5678": (True, "SSCI", "Education"),
        "9999-9999": (False, None, None),
    }
    screen = AsyncMock(
        side_effect=[
            _batch(
                [True, False, True],
                ["Directly on topic", "Different field", "Applies the framework"],
                fingerprint="fp_1",
                input_tokens=120,
                output_tokens=30,
            ),
            _batch([]),
        ]
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(
                side_effect=[
                    _query_result(["q1", "q2"], _QUERY_PROVENANCE_ROUND1),
                    _query_result(["q1"], _QUERY_PROVENANCE_ROUND2),
                ]
            ),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(
            smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value=lookup)
        ),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(
                inclusion_criteria=["empirical"],
                exclusion_criteria=["editorials"],
                wos_filter="on",
            )
        )

    result = update.await_args_list[-1].kwargs["result"]
    flow = result["flow"]
    # Round 1: q1 -> p1, p2, p3; q2 -> p1 (dup), p4.  Round 2: q1 -> p1, p2, p3 (all dups).
    assert flow["identified"] == result["total_scanned"] == 8
    assert flow["duplicates_removed"] == 4
    assert flow["stage1_screened"] == 4  # p1, p2, p3, p4
    assert flow["stage1_excluded"] == 1  # p2 (ISSN 9999-9999 not indexed)
    assert flow["stage2_screened"] == 3  # p1, p3, p4
    assert flow["stage2_excluded"] == 1  # p3
    assert flow["unscreened"] == 0
    assert flow["included"] == 2 == result["total_included"]
    assert flow["rounds"] == result["rounds"] == 2
    assert flow["stop_reason"] == result["stop_reason"] == "no_new_papers"
    assert _reconciles(flow)

    excluded = {e["doi"]: e for e in result["excluded"]}
    assert set(excluded) == {"10.1/p2", "10.1/p3"}
    assert excluded["10.1/p2"]["stage"] == "wos"
    assert "Web of Science" in excluded["10.1/p2"]["reason"]
    assert excluded["10.1/p2"]["journal_issn"] == "9999-9999"
    assert excluded["10.1/p3"]["stage"] == "llm"
    assert excluded["10.1/p3"]["reason"] == "Different field"
    assert excluded["10.1/p3"]["status"] == "EXCLUDE"
    assert set(excluded["10.1/p3"]) == {
        "title", "doi", "year", "journal", "journal_issn", "openalex_id", "abstract",
        "stage", "status", "criterion", "quote", "needs_review_reason", "to_confirm",
        "second_pass", "second_pass_input_tokens", "second_pass_output_tokens",
        "second_pass_latency_s", "reason",
    }

    reasons = {p["doi"]: p["screening_reason"] for p in result["papers"]}
    assert reasons == {"10.1/p1": "Directly on topic", "10.1/p4": "Applies the framework"}

    assert result["stage1_applied"] is True
    assert result["criteria"] == {
        "query": "teacher burnout",
        "inclusion_criteria": ["empirical"],
        "exclusion_criteria": ["editorials"],
        "wos_filter": "on",
        "inclusion_criteria_stages": [],
        "exclusion_criteria_stages": [],
        "publication_date_max": None,
    }

    screener = result["provenance"]["screener"]
    assert screener["agent"] == "relevance_screener"
    assert screener["model_configured"] == settings.deepseek_model
    assert screener["model_reported"] == ["deepseek-v4-flash"]
    assert screener["system_fingerprints"] == ["fp_1"]
    assert screener["temperature"] == 0.0
    assert screener["prompt_version"] == SCREENER_PROMPT_VERSION
    assert screener["calls"] == 1
    assert screener["input_tokens"] == 120
    assert screener["output_tokens"] == 30
    qg = result["provenance"]["query_generator"]
    assert qg["model_configured"] == settings.deepseek_model
    assert qg["prompt_version"] == prompt_version(QUERY_GENERATOR_PROMPT)
    # The aggregated fields the job result actually carries, not just
    # the two keys a two-key literal would already satisfy.
    assert qg["calls"] == 2
    assert qg["input_tokens"] == 50
    assert qg["output_tokens"] == 13
    assert qg["model_reported"] == ["deepseek-flash", "deepseek-v4-flash"]
    assert qg["system_fingerprints"] == ["fp_a", "fp_b"]


@pytest.mark.asyncio
async def test_provenance_aggregates_distinct_values_and_sums_tokens_across_batches(
    monkeypatch,
):
    from app.config import settings
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "smart_search_batch_size", 2)

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2), _paper(3), _paper(4)], {"openalex": 4}

    screen = AsyncMock(
        side_effect=[
            _batch([True, True], fingerprint="fp_b"),
            _batch([True, False], fingerprint="fp_a"),
            _batch([]),
        ]
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    screener = update.await_args_list[-1].kwargs["result"]["provenance"]["screener"]
    assert screener["calls"] == 2
    assert screener["model_reported"] == ["deepseek-v4-flash"]
    assert screener["system_fingerprints"] == ["fp_a", "fp_b"]
    assert screener["input_tokens"] == 200
    assert screener["output_tokens"] == 20


@pytest.mark.asyncio
async def test_provenance_counts_a_batchs_reask_as_a_second_call(monkeypatch):
    """A batch whose response needed the backend's own
    re-ask made two paid calls, not one -- the job summary must count both, sum both calls'
    tokens, and keep a model/fingerprint change on the re-ask visible."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    screen = AsyncMock(
        return_value=_batch(
            [True],
            fingerprint="fp_a",
            input_tokens=100,
            output_tokens=10,
            reask_model_reported="deepseek-v4-flash-2",
            reask_fingerprint="fp_reask",
            reask_input_tokens=70,
            reask_output_tokens=7,
        )
    )
    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    screener = update.await_args_list[-1].kwargs["result"]["provenance"]["screener"]
    assert screener["calls"] == 2
    assert screener["input_tokens"] == 170
    assert screener["output_tokens"] == 17
    assert screener["model_reported"] == ["deepseek-v4-flash", "deepseek-v4-flash-2"]
    assert screener["system_fingerprints"] == ["fp_a", "fp_reask"]


@pytest.mark.asyncio
async def test_time_limit_during_screening_records_pending_papers_as_unscreened(monkeypatch):
    import asyncio

    from app.config import settings
    from app.services.search import SearchService

    monkeypatch.setattr(settings, "smart_search_batch_size", 1)
    monkeypatch.setattr(settings, "smart_search_max_time_minutes", 0.005)  # 0.3 s

    async def _fake_search(self, query, **kwargs):
        return [_paper(1), _paper(2), _paper(3)], {"openalex": 3}

    async def _slow_screen(query, batch, *args, **kwargs):
        await asyncio.sleep(0.4)  # the first batch alone exhausts the budget
        return _batch([True] * len(batch))

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(smart_search_module, "screen_papers", new=_slow_screen),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    flow = result["flow"]
    assert result["stop_reason"] == "time_limit"
    # The search's own time budget
    # stops the round loop (paper 1's batch pass alone takes 0.4s against a 0.3s budget),
    # but the second pass runs once after the loop ends, on its own separate budget
    # (settings.screener_second_pass_stage_budget_seconds, unaffected by
    # smart_search_max_time_minutes) -- the judge is off the search's own critical path
    # entirely, so paper 1's provisional INCLUDE is still confirmed even though the search
    # itself stopped on time_limit.
    assert flow["included"] == 1
    assert flow["needs_review"] == 0
    assert flow["unscreened"] == 2
    assert all("time limit" in e["reason"] for e in result["unscreened"])
    assert _reconciles(flow)


# --- Replay (queries_override / publication_date_max) --------------------------


@pytest.mark.asyncio
async def test_queries_override_replaces_query_generation_for_its_rounds():
    """A round within ``queries_override``'s range uses that list verbatim and never
    calls the query generator agent."""
    from app.services.search import SearchService

    seen_queries: list[str] = []

    async def _fake_search(self, query, **kwargs):
        seen_queries.append(query)
        return [_paper(1)], {"openalex": 1}

    generate = AsyncMock(return_value=_query_result(["should not be used"]))
    update = AsyncMock()
    with (
        patch.object(smart_search_module, "generate_search_queries_with_provenance", new=generate),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(wos_filter="off", queries_override=[["replayed q1", "replayed q2"]])
        )

    generate.assert_not_awaited()
    assert seen_queries == ["replayed q1", "replayed q2"]
    result = update.await_args_list[-1].kwargs["result"]
    round1 = result["provenance"]["rounds"][0]
    assert round1["replayed"] is True
    assert round1["source"] == "queries_override"
    assert [q["query"] for q in round1["queries"]] == ["replayed q1", "replayed q2"]
    # A round that replays a prior run's queries never calls the query
    # generator agent, so the aggregated block must show no calls at all.
    assert result["provenance"]["query_generator"]["calls"] == 0
    assert result["provenance"]["query_generator"]["input_tokens"] == 0
    assert result["provenance"]["query_generator"]["output_tokens"] == 0


@pytest.mark.asyncio
async def test_a_round_with_no_override_left_is_not_marked_replayed():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    round1 = result["provenance"]["rounds"][0]
    assert round1["replayed"] is False
    assert "source" not in round1


@pytest.mark.asyncio
async def test_run_stops_after_the_last_override_list_when_no_later_round_would_run(
    monkeypatch,
):
    """With the default single-round stop settings, a dry round after the last
    override list ends the job -- the query generator agent is never called."""
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    generate = AsyncMock(return_value=_query_result(["should not be used"]))
    update = AsyncMock()
    with (
        patch.object(smart_search_module, "generate_search_queries_with_provenance", new=generate),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(wos_filter="off", queries_override=[["replayed q1"]])
        )

    generate.assert_not_awaited()
    result = update.await_args_list[-1].kwargs["result"]
    assert result["rounds"] == 1
    assert result["stop_reason"] == "no_new_included"


@pytest.mark.asyncio
async def test_round_after_the_override_is_exhausted_falls_back_to_generation(monkeypatch):
    """When the normal stop rule would still generate another round (min_rounds
    not yet reached), a round past the end of ``queries_override`` falls back to real
    query generation instead of the run stopping early."""
    from app.services.search import SearchService

    monkeypatch.setattr(smart_search_module.settings, "smart_search_min_rounds", 2)
    monkeypatch.setattr(smart_search_module.settings, "smart_search_dry_round_patience", 1)

    async def _fake_search(self, query, **kwargs):
        return [_paper(1 if query == "replayed q1" else 2)], {"openalex": 1}

    generate = AsyncMock(return_value=_query_result(["generated q2"]))
    update = AsyncMock()
    with (
        patch.object(smart_search_module, "generate_search_queries_with_provenance", new=generate),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(wos_filter="off", queries_override=[["replayed q1"]])
        )

    assert generate.await_count == 1
    result = update.await_args_list[-1].kwargs["result"]
    assert result["rounds"] == 2
    rounds = result["provenance"]["rounds"]
    assert rounds[0]["replayed"] is True
    assert rounds[1]["replayed"] is False
    assert "source" not in rounds[1]
    assert [q["query"] for q in rounds[1]["queries"]] == ["generated q2"]


@pytest.mark.asyncio
async def test_publication_date_max_is_forwarded_to_every_search_call_and_recorded():
    from datetime import date

    from app.services.search import SearchService

    seen: list[dict] = []

    async def _fake_search(self, query, **kwargs):
        seen.append(kwargs)
        return [_paper(1)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1", "q2"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(
            **_run_kwargs(wos_filter="off", publication_date_max=date(2026, 9, 8))
        )

    assert all(kwargs["to_publication_date"] == date(2026, 9, 8) for kwargs in seen)
    result = update.await_args_list[-1].kwargs["result"]
    assert result["criteria"]["publication_date_max"] == "2026-09-08"


@pytest.mark.asyncio
async def test_publication_date_max_defaults_to_none_in_criteria():
    from app.services.search import SearchService

    async def _fake_search(self, query, **kwargs):
        return [_paper(1)], {"openalex": 1}

    update = AsyncMock()
    with (
        patch.object(
            smart_search_module,
            "generate_search_queries_with_provenance",
            new=AsyncMock(return_value=_query_result(["q1"])),
        ),
        patch.object(SearchService, "search", new=_fake_search),
        patch.object(smart_search_module, "is_wos_indexed_bulk", new=AsyncMock(return_value={})),
        patch.object(
            smart_search_module,
            "screen_papers",
            new=AsyncMock(return_value=_status_batch(["EXCLUDE"])),
        ),
        patch.object(smart_search_module, "should_abort", new=AsyncMock(return_value=False)),
        patch.object(smart_search_module, "update_job_status", new=update),
    ):
        await smart_search_module.run_smart_search(**_run_kwargs(wos_filter="off"))

    result = update.await_args_list[-1].kwargs["result"]
    assert result["criteria"]["publication_date_max"] is None
