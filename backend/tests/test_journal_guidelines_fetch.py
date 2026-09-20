"""Candidate guideline pages must be fetched in parallel, preserving result order."""

import asyncio
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

SEARCH_HTML = """
<a class="result__a" href="https://example.org/one">One</a>
<a class="result__a" href="https://example.org/two">Two</a>
<a class="result__a" href="https://example.org/three">Three</a>
"""

GUIDELINES_HTML = (
    "<html><body>Instructions for author submission. Manuscript word limit 8000. "
    "References must be APA.</body></html>"
)

IRRELEVANT_HTML = "<html><body>Cookie policy.</body></html>"


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


@pytest.mark.asyncio
async def test_candidate_pages_are_fetched_concurrently():
    from app.agents.journal_guidelines_agent import _search_and_fetch_guidelines

    state = {"in_flight": 0, "max_in_flight": 0}
    bodies = {
        "https://example.org/one": IRRELEVANT_HTML,
        "https://example.org/two": GUIDELINES_HTML,
        "https://example.org/three": GUIDELINES_HTML,
    }

    async def _fake_get(self, url, *args, **kwargs):
        if "duckduckgo" in url:
            return _FakeResponse(SEARCH_HTML)
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        await asyncio.sleep(0.05)
        state["in_flight"] -= 1
        return _FakeResponse(bodies[url])

    with patch("httpx.AsyncClient.get", new=_fake_get):
        text = await _search_and_fetch_guidelines("Test Journal")

    assert state["max_in_flight"] > 1, "candidate URLs are still fetched serially"
    assert text is not None
    assert "Manuscript word limit 8000" in text


@pytest.mark.asyncio
async def test_first_relevant_candidate_in_search_order_wins():
    from app.agents.journal_guidelines_agent import _search_and_fetch_guidelines

    bodies = {
        "https://example.org/one": IRRELEVANT_HTML,
        "https://example.org/two": (
            "<html><body>Author guidelines: submission manuscript word limit 5000 "
            "references APA. MARKER-TWO</body></html>"
        ),
        "https://example.org/three": (
            "<html><body>Author guidelines: submission manuscript word limit 9000 "
            "references APA. MARKER-THREE</body></html>"
        ),
    }

    async def _fake_get(self, url, *args, **kwargs):
        if "duckduckgo" in url:
            return _FakeResponse(SEARCH_HTML)
        return _FakeResponse(bodies[url])

    with patch("httpx.AsyncClient.get", new=_fake_get):
        text = await _search_and_fetch_guidelines("Test Journal")

    assert "MARKER-TWO" in text
    assert "MARKER-THREE" not in text


@pytest.mark.asyncio
async def test_returns_none_when_no_candidate_is_relevant():
    from app.agents.journal_guidelines_agent import _search_and_fetch_guidelines

    async def _fake_get(self, url, *args, **kwargs):
        if "duckduckgo" in url:
            return _FakeResponse(SEARCH_HTML)
        return _FakeResponse(IRRELEVANT_HTML)

    with patch("httpx.AsyncClient.get", new=_fake_get):
        assert await _search_and_fetch_guidelines("Test Journal") is None
