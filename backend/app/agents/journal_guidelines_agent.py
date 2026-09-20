"""Journal Guidelines Agent — searches web for journal author guidelines and extracts structured rules."""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model for structured journal guidelines
# ---------------------------------------------------------------------------


class JournalGuidelines(BaseModel):
    journal_name: str
    word_limit_total: int | None = None
    word_limit_abstract: int | None = None
    abstract_structure: str | None = None  # "structured" or "unstructured"
    required_sections: list[str] = Field(default_factory=list)
    citation_style: str | None = None  # "APA 7th", "Harvard", "Vancouver", etc.
    reference_format: str | None = None
    heading_style: str | None = None
    line_spacing: str | None = None
    font_requirements: str | None = None
    figure_table_rules: str | None = None
    keyword_requirements: str | None = None
    special_requirements: str | None = None
    source_url: str | None = None
    raw_guidelines_text: str | None = None


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)",
}

_RELEVANCE_KEYWORDS = {"author", "manuscript", "submission", "word limit", "references"}


def _strip_html(html: str) -> str:
    """Remove HTML tags, scripts, and styles — return plain text."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_search_urls(html: str) -> list[str]:
    """Extract result URLs from DuckDuckGo HTML search results page."""
    # DuckDuckGo wraps results in <a class="result__a" href="...">
    urls: list[str] = []
    for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"', html):
        url = match.group(1)
        if url.startswith("http"):
            urls.append(url)
    # Fallback: broader pattern for result links
    if not urls:
        for match in re.finditer(
            r'<a[^>]+href="(https?://[^"]+)"[^>]*class="[^"]*result[^"]*"', html
        ):
            urls.append(match.group(1))
    # Another fallback: look for uddg redirect parameter
    if not urls:
        for match in re.finditer(r"uddg=([^&\"']+)", html):
            from urllib.parse import unquote

            decoded = unquote(match.group(1))
            if decoded.startswith("http"):
                urls.append(decoded)
    return urls[:3]


def _looks_like_guidelines(text: str) -> bool:
    """Check whether the text plausibly contains journal author guidelines."""
    text_lower = text.lower()
    matches = sum(1 for kw in _RELEVANCE_KEYWORDS if kw in text_lower)
    return matches >= 2


# ---------------------------------------------------------------------------
# Web search & fetch
# ---------------------------------------------------------------------------


async def _search_and_fetch_guidelines(journal_name: str) -> str | None:
    """Search web for journal author guidelines, fetch the page, return text content."""
    search_url = (
        f"https://html.duckduckgo.com/html/?q="
        f"{quote(journal_name)}+author+guidelines+instructions+for+authors"
    )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15.0,
        headers=_HEADERS,
    ) as client:
        # Step 1: Fetch DuckDuckGo search results page
        try:
            search_resp = await client.get(search_url)
            search_resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("DuckDuckGo search request failed: %s", exc)
            return None

        result_urls = _extract_search_urls(search_resp.text)
        if not result_urls:
            logger.warning("No search result URLs found for journal: %s", journal_name)
            return None

        # Step 2: fetch every candidate concurrently; the first relevant one, in search
        # result order, wins (issue #29 — this used to be up to 3 x 15s serially).
        async def _fetch_candidate(url: str) -> str | None:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.debug("Failed to fetch %s: %s", url, exc)
                return None
            return _strip_html(resp.text)

        pages = await asyncio.gather(
            *(_fetch_candidate(url) for url in result_urls), return_exceptions=True
        )

        for url, page in zip(result_urls, pages):
            if isinstance(page, BaseException) or not page:
                continue
            if not _looks_like_guidelines(page):
                logger.debug("Page does not look like guidelines: %s", url)
                continue
            truncated = page[:15_000]
            logger.info("Fetched guidelines from %s (%d chars)", url, len(truncated))
            return truncated

    logger.warning("Could not find guidelines page for journal: %s", journal_name)
    return None


# ---------------------------------------------------------------------------
# AI extraction agent (lazy singleton)
# ---------------------------------------------------------------------------

GUIDELINES_EXTRACTION_PROMPT = """\
You are an expert at reading academic journal "Instructions for Authors" pages and extracting structured formatting rules.

Given the text content of a journal's author guidelines page, extract:
- Total word/page limits
- Abstract requirements (structured/unstructured, word limit)
- Required sections and their order
- Citation and reference style
- Heading formatting rules
- Line spacing, font requirements
- Figure/table rules
- Keyword requirements
- Any special requirements

If a field is not mentioned in the guidelines, leave it as null.
Extract EXACTLY what the guidelines say — do not infer or assume."""

_guidelines_agent: Agent[None, JournalGuidelines] | None = None


def get_journal_guidelines_agent() -> Agent[None, JournalGuidelines]:
    """Lazily create the journal guidelines extraction agent."""
    global _guidelines_agent
    if _guidelines_agent is None:
        _guidelines_agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            output_type=JournalGuidelines,
            instructions=GUIDELINES_EXTRACTION_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _guidelines_agent


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


async def fetch_journal_guidelines(journal_name: str) -> JournalGuidelines:
    """Search web for journal guidelines and extract structured rules."""
    # 1. Search and fetch guidelines text
    guidelines_text = await _search_and_fetch_guidelines(journal_name)

    if not guidelines_text:
        # Return minimal guidelines with just the journal name
        return JournalGuidelines(journal_name=journal_name)

    # 2. AI extraction
    agent = get_journal_guidelines_agent()
    prompt = f"Journal: {journal_name}\n\nAuthor Guidelines Text:\n{guidelines_text}"
    result = await agent.run(prompt)
    output = result.output

    # Ensure journal name is set correctly and store raw text for reference
    output.journal_name = journal_name
    output.raw_guidelines_text = guidelines_text[:5000]
    return output
