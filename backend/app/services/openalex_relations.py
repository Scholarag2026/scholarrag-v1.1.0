"""Shared OpenAlex relation lookups for the citation graph and seed expansion.

One place decides how a stored paper maps onto an OpenAlex identifier and how its
references and citations are fetched, so the graph build, interactive expansion and
seed expansion cannot drift apart again (issue SEED-OPENALEX-ID-TO-S2).

Cost per paper: 1 `get_work` + ceil(refs/50) batched metadata reads + 1 `cites:` query,
with the last two issued concurrently — versus two sequential Semantic Scholar retry
chains plus 3s of deliberate sleeps before this rewrite.
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from app.clients.http_retry import retry_after_seconds
from app.clients.openalex import OpenAlexClient, filter_queries_blocked, note_filter_block

logger = logging.getLogger(__name__)

WORK_ID_RE = re.compile(r"^W\d+$")


def is_work_id(value: str | None) -> bool:
    """True when *value* is an OpenAlex work id such as ``W4381953137``."""
    return bool(value and WORK_ID_RE.match(value.strip()))


def resolve_work_id(external_id: str | None, doi: str | None) -> str | None:
    """Resolve a stored paper to an identifier OpenAlex accepts.

    ``papers.external_id`` is already an OpenAlex W-id for anything discovered through
    OpenAlex; older rows can hold a Semantic Scholar ``paperId``, which OpenAlex does not
    recognise, so those fall through to the DOI (``/works/doi:10.x/y``) and, failing that,
    to ``None``.
    """
    if is_work_id(external_id):
        return external_id.strip()
    if doi:
        return doi.strip()
    return None


def cap_works(works: list[dict], max_count: int) -> list[dict]:
    """Keep the ``max_count`` most-cited works, most cited first."""
    ordered = sorted(works, key=lambda w: w.get("citation_count") or 0, reverse=True)
    return ordered[:max_count]


async def fetch_relations(
    client: OpenAlexClient,
    work_id: str,
    *,
    max_refs: int,
    max_cites: int,
) -> tuple[list[dict], list[dict] | None] | None:
    """Fetch ``(references, citations)`` for one work.

    Args:
        client: An ``OpenAlexClient`` (shares the process-wide pool and 8 rps limiter).
        work_id: An OpenAlex W-id, a bare DOI or a DOI URL.
        max_refs: How many ``referenced_works`` ids to hydrate.
        max_cites: ``per_page`` for the ``cites:`` query.

    Returns:
        ``(references, citations)`` as ``parse_work_dict`` dicts, or ``None`` when OpenAlex
        has no record of *work_id* (HTTP 404 — common for very old CNKI DOIs).
        ``citations`` is ``None`` (distinct from ``[]``) when the ``cites:`` filter query
        is rate-limited: reference links are still trustworthy, citing-paper links are
        simply unavailable until the quota resets.

    Raises:
        ``httpx.HTTPError`` for any other failure, so the caller can trip a CircuitBreaker
        instead of silently recording "no citations".
    """
    work = await client.get_work(work_id)
    if work is None:
        return None

    ref_ids = (work.get("referenced_works") or [])[:max_refs]
    openalex_id = work.get("openalex_id")

    async def _refs() -> list[dict]:
        if not ref_ids:
            return []
        return await client.get_works_batch(ref_ids)

    async def _cites() -> list[dict] | None:
        if not openalex_id:
            return []
        if filter_queries_blocked("cites"):
            # Known quota block on cites: queries — skip the probe entirely.
            return None
        try:
            return await client.get_citations(openalex_id, per_page=max_cites)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                note_filter_block("cites", retry_after_seconds(exc))
                logger.warning(
                    "OpenAlex cites query rate-limited for %s; proceeding refs-only",
                    openalex_id,
                )
                return None
            raise

    refs, cites = await asyncio.gather(_refs(), _cites())
    return refs, cites
