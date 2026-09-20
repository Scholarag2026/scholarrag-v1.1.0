# backend/app/services/search.py
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import date

from app.clients.openalex import OpenAlexClient
from app.config import settings
from app.schemas.paper import PaperData

logger = logging.getLogger(__name__)

#: Query parameters that identify the account behind a request (polite-pool address, API
#: key). They must not reach a stored job result or an exported screening record.
_URL_IDENTITY_RE = re.compile(r"(mailto|api_key)=[^&\s']+")


def describe_search_failure(exc: BaseException) -> str:
    """Compact, shareable one-liner for a failed search query.

    ``"<ExceptionClass>: <message>"``; for an HTTP status error the message is the status
    plus the provider's own explanation (OpenAlex's ``message``/``error`` JSON field, e.g.
    "Insufficient budget ... Resets at midnight UTC") instead of the request URL, and any
    ``mailto=``/``api_key=`` value is redacted.
    """
    text = f"{type(exc).__name__}: {exc}"
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("message") or body.get("error") or "")[:200]
        except Exception:
            detail = (getattr(response, "text", "") or "")[:200]
        reason = getattr(response, "reason_phrase", "") or ""
        text = f"{type(exc).__name__}: HTTP {response.status_code}"
        if reason:
            text += f" {reason}"
        if detail:
            text += f" - {detail}"
    return _URL_IDENTITY_RE.sub(r"\1=***", text)


class SearchService:
    """Scholarly search fan-out. OpenAlex is the only scholarly data source."""

    def __init__(self) -> None:
        self._openalex = OpenAlexClient()

    async def search(
        self,
        query: str,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        to_publication_date: date | None = None,
        raise_on_failure: bool = False,
    ) -> tuple[list[PaperData], dict[str, int]]:
        """Search OpenAlex, deduplicate, and return merged results.

        Returns (papers, sources) where sources is a count per API. The tuple shape is
        preserved for callers even though there is now exactly one source.

        A provider failure is logged and yields an empty result by default (the one-shot
        ``/search`` endpoint). Callers that must tell "no results" from "provider down"
        (the Smart Search fan-out) pass ``raise_on_failure=True`` and get the exception.

        ``to_publication_date`` is forwarded to the OpenAlex client
        unchanged; see ``OpenAlexClient.search``.
        """
        try:
            oa_results = await self._search_openalex(
                query,
                year_from=year_from,
                year_to=year_to,
                min_citations=min_citations,
                to_publication_date=to_publication_date,
            )
        except Exception as exc:
            logger.warning("OpenAlex search failed for %r: %s", query, exc)
            if raise_on_failure:
                raise
            oa_results = []

        sources = {"openalex": len(oa_results)}

        # Deduplicate by DOI, keeping the version with higher citation count
        merged = self._deduplicate(list(oa_results))

        # Sort by citation count descending
        merged.sort(key=lambda p: p.citation_count or 0, reverse=True)

        return merged, sources

    async def _search_openalex(self, query: str, **kwargs) -> list[PaperData]:
        return await self._openalex.search(query, **kwargs)

    @staticmethod
    def _deduplicate(papers: list[PaperData]) -> list[PaperData]:
        """Deduplicate papers by DOI. Keep the one with higher citation count."""
        seen_dois: dict[str, PaperData] = {}
        no_doi: list[PaperData] = []

        for paper in papers:
            if not paper.doi:
                no_doi.append(paper)
                continue

            doi_lower = paper.doi.lower()
            if doi_lower in seen_dois:
                existing = seen_dois[doi_lower]
                if (paper.citation_count or 0) > (existing.citation_count or 0):
                    seen_dois[doi_lower] = paper
            else:
                seen_dois[doi_lower] = paper

        return list(seen_dois.values()) + no_doi


async def gather_search_results(
    search_service: SearchService,
    queries: Sequence[str],
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    min_citations: int | None = None,
    to_publication_date: date | None = None,
    concurrency: int | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
) -> tuple[list[tuple[str, list[PaperData]]], bool, list[tuple[str, str]]]:
    """Run *queries* against *search_service* with bounded concurrency.

    Returns ``(results, stopped, failures)``:

    * ``results`` holds one ``(query, papers)`` pair per query that actually executed and
      returned, in the input order of *queries*. Queries that raised are logged and omitted
      from ``results``.
    * ``stopped`` is True when *stop_check* returned True, meaning the remaining queries were
      deliberately skipped (time budget exhausted or job cancelled).
    * ``failures`` lists ``(query, "<ExceptionClass>: <message>")`` for every query that
      raised, in input order, so a caller can tell an empty round caused by a provider
      outage or quota block from one where the literature genuinely ran dry.

    *stop_check* is awaited immediately before each individual query is issued, which is what
    makes a wall-clock cap bind *inside* a round instead of only at round boundaries.
    """
    if not queries:
        return [], False, []

    limit = concurrency if concurrency is not None else settings.search_fanout_concurrency
    semaphore = asyncio.Semaphore(max(1, limit))
    stopped = False
    failures: dict[str, str] = {}

    async def _run_one(query: str) -> tuple[str, list[PaperData]] | None:
        nonlocal stopped
        async with semaphore:
            if stopped:
                return None
            if stop_check is not None and await stop_check():
                stopped = True
                return None
            try:
                papers, _sources = await search_service.search(
                    query,
                    year_from=year_from,
                    year_to=year_to,
                    min_citations=min_citations,
                    to_publication_date=to_publication_date,
                    raise_on_failure=True,
                )
            except Exception as exc:
                logger.warning("Search failed for query %r", query, exc_info=True)
                failures[query] = describe_search_failure(exc)
                return None
            return query, papers

    raw = await asyncio.gather(*(_run_one(q) for q in queries))
    ordered_failures = [(q, failures[q]) for q in queries if q in failures]
    return [item for item in raw if item is not None], stopped, ordered_failures
