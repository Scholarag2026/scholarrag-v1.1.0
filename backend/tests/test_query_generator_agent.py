"""Tests for the query generator agent — OpenAlex-safe query output (D1)."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_prompt_forbids_wildcards_and_names_openalex():
    from app.agents.query_generator_agent import QUERY_GENERATOR_PROMPT

    text = QUERY_GENERATOR_PROMPT.lower()
    assert "openalex" in text
    assert "wildcard" in text
    assert "truncation" in text


def test_strip_wildcards_removes_truncation_stems():
    from app.agents.query_generator_agent import _strip_wildcards

    assert _strip_wildcards("compar*") == "compar"
    assert _strip_wildcards('"student*" AND process*') == '"student" AND process'
    assert _strip_wildcards("wom?n AND leadership") == "womn AND leadership"


def test_strip_wildcards_collapses_leftover_whitespace():
    from app.agents.query_generator_agent import _strip_wildcards

    assert _strip_wildcards("teacher * AND  * burnout") == "teacher AND burnout"


def test_strip_wildcards_leaves_clean_queries_untouched():
    from app.agents.query_generator_agent import _strip_wildcards

    query = '"social capital" AND (trust OR reciprocity)'
    assert _strip_wildcards(query) == query


class _FakeOutput:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries


class _FakeRunResult:
    def __init__(self, queries: list[str]) -> None:
        self.output = _FakeOutput(queries)


class _FakeAgent:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    async def run(self, prompt: str):
        return _FakeRunResult(self._queries)


@pytest.mark.asyncio
async def test_generate_search_queries_strips_wildcards_from_llm_output():
    from app.agents import query_generator_agent as mod

    fake = _FakeAgent(['compar* AND "student*"', "  *  ", "teacher burnout"])
    with patch.object(mod, "get_query_generator_agent", return_value=fake):
        queries = await mod.generate_search_queries("teacher burnout")

    assert queries == ['compar AND "student"', "teacher burnout"]


def _fake_query_run_result(queries: list[str], *, input_tokens=120, output_tokens=40):
    """A run result shaped like ``analyze_paper_text_with_provenance``'s own test
    fixture (``test_deep_analysis_agent.py``), so ``provenance_from_run`` reads every
    field the same way for both agents."""
    response = SimpleNamespace(
        model_name="deepseek-v4-flash",
        provider_response_id="resp-qg-1",
        provider_details={"system_fingerprint": "fp_qg"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    output = _FakeOutput(queries)
    return SimpleNamespace(output=output, response=response, usage=lambda: usage)


@pytest.mark.asyncio
async def test_generate_search_queries_with_provenance_returns_queries_and_call_provenance():
    """The query generator's own calls, tokens, model and
    fingerprint must be folded into a provenance record -- otherwise every Smart Search
    round's query-generation call is invisible to summary.json and the reported
    token/cost totals."""
    from app.agents import query_generator_agent as mod
    from app.config import settings
    from app.schemas.provenance import prompt_version

    fake_result = _fake_query_run_result(["teacher burnout"])
    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch.object(mod, "get_query_generator_agent", return_value=fake_agent):
        result = await mod.generate_search_queries_with_provenance("teacher burnout")

    assert isinstance(result, mod.GeneratedQueriesResult)
    assert result.queries == ["teacher burnout"]
    assert result.provenance.agent == "query_generator"
    assert result.provenance.model_configured == settings.deepseek_model
    assert result.provenance.model_reported == "deepseek-v4-flash"
    assert result.provenance.system_fingerprint == "fp_qg"
    assert result.provenance.prompt_version == prompt_version(mod.QUERY_GENERATOR_PROMPT)
    assert result.provenance.input_tokens == 120
    assert result.provenance.output_tokens == 40


@pytest.mark.asyncio
async def test_generate_search_queries_still_returns_only_the_queries():
    """Existing call sites (``app.services.smart_search.run_smart_search``) call
    ``generate_search_queries`` and expect just the list, not a tuple."""
    from app.agents import query_generator_agent as mod

    fake_result = _fake_query_run_result(["teacher burnout"])
    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch.object(mod, "get_query_generator_agent", return_value=fake_agent):
        queries = await mod.generate_search_queries("teacher burnout")

    assert queries == ["teacher burnout"]
