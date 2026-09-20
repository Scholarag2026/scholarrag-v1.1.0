# backend/app/clients/crossref.py
"""CrossRef client — DOI verification.

Uses one process-wide pooled httpx client (a fresh ``AsyncClient`` per DOI meant a
fresh TLS handshake for every paper in a verification run) and the shared
transient-only retry policy.
"""

from __future__ import annotations

import logging

import httpx

from app.clients.http_retry import request_with_retry
from app.config import settings

logger = logging.getLogger(__name__)

CROSSREF_BASE_URL = "https://api.crossref.org"

_shared_http_client: httpx.AsyncClient | None = None


def _user_agent() -> str:
    if settings.openalex_email:
        return f"DeepResearch/2.0 (mailto:{settings.openalex_email})"
    return "DeepResearch/2.0"


def get_shared_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled CrossRef client."""
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            timeout=settings.crossref_timeout,
            headers={"User-Agent": _user_agent()},
        )
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Close the shared client. Tests and graceful shutdown only."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
    _shared_http_client = None


class CrossRefClient:
    """Client for the CrossRef API — used for DOI verification."""

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client
        self.headers = {"User-Agent": _user_agent()}

    def _get_client(self) -> httpx.AsyncClient:
        return self._client if self._client is not None else get_shared_http_client()

    async def verify_doi(self, doi: str) -> dict | None:
        """Verify a DOI exists in CrossRef. Returns metadata dict, or None."""
        client = self._get_client()

        async def _call() -> httpx.Response:
            resp = await client.get(
                f"{CROSSREF_BASE_URL}/works/{doi}", headers=self.headers
            )
            resp.raise_for_status()
            return resp

        try:
            response = await request_with_retry(_call)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("DOI not found in CrossRef: %s", doi)
            else:
                logger.warning(
                    "CrossRef API error for %s: %s", doi, exc.response.status_code
                )
            return None
        except Exception as exc:
            logger.warning("CrossRef verification failed for %s: %s", doi, exc)
            return None

        message = response.json().get("message", {})

        title_list = message.get("title", [])
        title = title_list[0] if title_list else None

        authors = []
        for author in message.get("author", []):
            name_parts = []
            if author.get("given"):
                name_parts.append(author["given"])
            if author.get("family"):
                name_parts.append(author["family"])
            if name_parts:
                authors.append({"name": " ".join(name_parts)})

        date_parts = message.get("published-print", {}).get("date-parts", [[]])
        if not date_parts or not date_parts[0]:
            date_parts = message.get("published-online", {}).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        container = message.get("container-title", [])
        journal_name = container[0] if container else None

        issn_list = message.get("ISSN", [])
        issn = issn_list[0] if issn_list else None

        return {
            "doi": message.get("DOI", doi),
            "title": title,
            "authors": authors,
            "year": year,
            "journal_name": journal_name,
            "issn": issn,
            "citation_count": message.get("is-referenced-by-count"),
        }
