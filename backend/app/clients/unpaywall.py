# backend/app/clients/unpaywall.py
"""Unpaywall API client -- finds legal OA copies of papers by DOI.

Uses one process-wide pooled httpx client and the shared transient-only retry
policy; every failure degrades to ``None`` rather than raising.
"""
from __future__ import annotations

import logging

import httpx

from app.clients.http_retry import request_with_retry
from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.unpaywall.org/v2"

_shared_http_client: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled Unpaywall client."""
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=settings.unpaywall_timeout,
            headers={"User-Agent": "DeepResearch/2.0"},
        )
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Close the shared client. Tests and graceful shutdown only."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
    _shared_http_client = None


#: F4 acquisition: priority rank for each OA location's Unpaywall ``version`` field --
#: the publisher's own typeset copy first, then the accepted-manuscript author copy,
#: then the pre-review submitted manuscript, with anything else (or no version at
#: all) last. Ties keep Unpaywall's own relative order (a stable sort).
_VERSION_PRIORITY = {
    "publishedVersion": 0,
    "acceptedVersion": 1,
    "submittedVersion": 2,
}
_OTHER_VERSION_PRIORITY = 3


class UnpaywallClient:
    """Look up open-access PDF URLs via the Unpaywall API."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client

    def _get_client(self) -> httpx.AsyncClient:
        return self._client if self._client is not None else get_shared_http_client()

    async def _fetch(self, doi: str) -> dict | None:
        """Return the raw Unpaywall record for *doi*, or ``None`` on any failure."""
        if not settings.unpaywall_email:
            logger.debug("unpaywall_email not configured; skipping lookup")
            return None

        client = self._get_client()

        async def _call() -> httpx.Response:
            resp = await client.get(
                f"/{doi}",
                params={"email": settings.unpaywall_email},
            )
            resp.raise_for_status()
            return resp

        try:
            resp = await request_with_retry(_call)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("DOI not found in Unpaywall: %s", doi)
            else:
                logger.warning(
                    "Unpaywall error for %s: %s", doi, exc.response.status_code
                )
            return None
        except Exception as exc:
            logger.warning("Unpaywall lookup failed for %s: %s", doi, exc)
            return None

        return resp.json()

    async def lookup(self, doi: str) -> str | None:
        """Return the best OA PDF URL for *doi*, or ``None`` if unavailable."""
        data = await self._fetch(doi)
        if not data or not data.get("is_oa"):
            return None

        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url")

    async def lookup_locations(self, doi: str) -> list[str]:
        """Return every OA download URL Unpaywall lists for *doi*, in priority order.

        Ordered publishedVersion, then acceptedVersion, then submittedVersion, then
        any other or unlabelled version, preserving Unpaywall's own relative order
        within each group. Each location contributes ``url_for_pdf`` when present,
        else its plain ``url``; a location with neither is skipped, and a URL
        already seen (e.g. the same file listed under two locations) is not
        repeated. Falls back to the single ``best_oa_location`` when the response
        carries no ``oa_locations`` list at all. Returns an empty list when
        Unpaywall is not configured, the DOI is unknown, or the work is not open
        access (F4 acquisition design).
        """
        data = await self._fetch(doi)
        if not data or not data.get("is_oa"):
            return []

        locations = data.get("oa_locations")
        if not locations:
            best = data.get("best_oa_location")
            locations = [best] if best else []

        ranked = sorted(
            enumerate(locations),
            key=lambda pair: (
                _VERSION_PRIORITY.get(pair[1].get("version"), _OTHER_VERSION_PRIORITY),
                pair[0],
            ),
        )
        urls: list[str] = []
        for _, loc in ranked:
            url = loc.get("url_for_pdf") or loc.get("url")
            if url and url not in urls:
                urls.append(url)
        return urls
