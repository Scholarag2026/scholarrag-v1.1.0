# backend/app/clients/openalex.py
"""OpenAlex API client — the single scholarly data source for DeepResearch.

Reliability contract:
  * one process-wide pooled httpx.AsyncClient and one process-wide rate limiter,
    never constructed per job (issue RATE-LIMITER-NOT-SHARED);
  * wildcards stripped from the stemmed ``search=`` parameter, which OpenAlex
    rejects with 400 (issue OPENALEX-WILDCARD-400, decision D1);
  * ``select=`` on every request to trim the payload;
  * transient-only retry (429/5xx/timeouts), fail fast on other 4xx;
  * polite pool: ``mailto`` query param plus a ``User-Agent`` header.

Error contract: 404 yields ``None``/``[]``; every other permanent 4xx and every
exhausted-retry transient failure raises ``httpx.HTTPError`` so callers can drive a
``CircuitBreaker`` instead of silently recording "no results".
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date

import httpx

from app.clients.http_retry import request_with_retry
from app.clients.rate_limiter import AsyncRateLimiter, get_rate_limiter
from app.config import settings
from app.schemas.paper import PaperData

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"

# Process-wide memo: monotonic deadlines until which OpenAlex *filter* queries are
# known to be quota-blocked, kept per endpoint family so a cites:-only throttle does
# not force batch hydration into a 50x singles amplification (and vice versa).
# Single-record GETs are free per OpenAlex pricing and keep working, so callers
# switch to them without re-paying a probe 429 on every call. Only QUOTA-LENGTH
# blocks (Retry-After above external_retry_after_cap) are memoized: a burst 429
# with no/short Retry-After affects one call only. Recovery is time-based —
# bounded at one hour, and real quota blocks expire at midnight UTC anyway.
_filter_block_until: dict[str, float] = {"batch": 0.0, "cites": 0.0}
_FILTER_BLOCK_MAX_SECONDS = 3600.0


def filter_queries_blocked(kind: str = "batch") -> bool:
    """True while *kind* ("batch" | "cites") filter queries are known rate-limited."""
    return time.monotonic() < _filter_block_until.get(kind, 0.0)


def note_filter_block(kind: str, retry_after: float | None) -> None:
    """Record a QUOTA-LENGTH 429 on a *kind* filter query; short/absent waits are not memoized."""
    if retry_after is None or retry_after <= settings.external_retry_after_cap:
        return
    _filter_block_until[kind] = time.monotonic() + min(
        retry_after, _FILTER_BLOCK_MAX_SECONDS
    )


def reset_filter_block() -> None:
    """Clear the filter-query block memos (tests only)."""
    for key in _filter_block_until:
        _filter_block_until[key] = 0.0


_API_KEY_RE = re.compile(r"(api_key=)[^&\s']+")


def _redact(text: str) -> str:
    """Strip the account API key from any URL-bearing string before it can be
    logged, persisted into job errors, or rendered to users."""
    return _API_KEY_RE.sub(r"\1***", text)

# Narrow field list — verified live against api.openalex.org on 2026-07-26.
# type/is_paratext are free (OpenAlex bills nothing for extra select
# fields) and let a caller apply the narrow non-article-type screening demotion
# (app.agents.relevance_screener_agent.apply_type_demotion) without a second request.
WORK_SELECT_FIELDS = (
    "id,doi,title,publication_year,authorships,primary_location,"
    "cited_by_count,abstract_inverted_index,open_access,type,is_paratext"
)
WORK_SELECT_FIELDS_WITH_REFS = f"{WORK_SELECT_FIELDS},referenced_works"

_WILDCARD_RE = re.compile(r"[*?]")
_WHITESPACE_RE = re.compile(r"\s+")

_shared_http_client: httpx.AsyncClient | None = None


def _polite_headers() -> dict[str, str]:
    """Polite-pool User-Agent. OpenAlex raises the rate ceiling for identified clients."""
    if settings.openalex_email:
        return {"User-Agent": f"DeepResearch/2.0 (mailto:{settings.openalex_email})"}
    return {"User-Agent": "DeepResearch/2.0"}


def get_shared_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled client (keep-alive; one TLS handshake reused)."""
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            timeout=settings.openalex_timeout,
            headers=_polite_headers(),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _shared_http_client


async def close_shared_http_client() -> None:
    """Close the shared client. Tests and graceful shutdown only."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
    _shared_http_client = None


def get_shared_rate_limiter() -> AsyncRateLimiter:
    """Return the single OpenAlex limiter shared by every client instance and job."""
    return get_rate_limiter("openalex", settings.openalex_rate_limit)


def sanitize_query(query: str) -> str:
    """Make an LLM-written query safe for OpenAlex's stemmed ``search=`` parameter.

    OpenAlex answers 400 for any ``*``/``?``: "Wildcards (* or ?) require exact
    (no-stem) search." Per decision D1 we strip them rather than switching to
    ``search.exact=`` (which would silently change recall for the whole query).
    """
    cleaned = _WILDCARD_RE.sub("", query or "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    limit = settings.openalex_max_query_length
    if len(cleaned) > limit:
        cleaned = cleaned[:limit].rsplit(" ", 1)[0].strip()
    return cleaned


def short_work_id(work_url_or_id: str | None) -> str | None:
    """``'https://openalex.org/W123' -> 'W123'``; passes a bare W-id through."""
    if not work_url_or_id:
        return None
    return work_url_or_id.rsplit("/", 1)[-1]


def normalize_work_id(work_id_or_doi: str) -> str:
    """Return the ``/works/<segment>`` path segment for a W-id, a DOI or a DOI URL."""
    value = (work_id_or_doi or "").strip()
    lowered = value.lower()
    if lowered.startswith("https://doi.org/"):
        return f"doi:{value[len('https://doi.org/'):]}"
    if lowered.startswith("http://doi.org/"):
        return f"doi:{value[len('http://doi.org/'):]}"
    if lowered.startswith("doi:"):
        return f"doi:{value[4:]}"
    if lowered.startswith("10."):
        return f"doi:{value}"
    if lowered.startswith("https://openalex.org/"):
        return value.rsplit("/", 1)[-1]
    return value


def parse_work_dict(work: dict) -> dict:
    """Parse a raw OpenAlex Work into a plain dict.

    Keys are exactly the :class:`PaperData` field names plus ``openalex_id`` (short
    W-id) and ``referenced_works`` (list of short W-ids, ``[]`` when not selected).
    """
    paper = _parse_work(work)
    data = paper.model_dump()
    data["openalex_id"] = paper.external_id
    data["referenced_works"] = [
        short_work_id(ref) for ref in (work.get("referenced_works") or []) if ref
    ]
    # openalex_type/is_paratext are PaperData fields set by _parse_work above,
    # so paper.model_dump() already carries them correctly -- no separate assignment needed.
    return data


def to_paper_data(work_dict: dict) -> PaperData:
    """Convert a :func:`parse_work_dict` result into a ``PaperData`` (extras ignored)."""
    return PaperData.model_validate(work_dict)


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for idx in indices:
            positions[idx] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def _parse_work(work: dict) -> PaperData:
    """Parse an OpenAlex Work object into PaperData."""
    doi_raw = work.get("doi")
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

    authors = [
        {"name": a["author"]["display_name"]}
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]

    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    journal_name = source.get("display_name")
    issn_list = source.get("issn") or []
    journal_issn = issn_list[0] if issn_list else None

    oa_id = work.get("id", "")
    external_id = oa_id.split("/")[-1] if oa_id else None

    return PaperData(
        doi=doi,
        title=work.get("title") or "Untitled",
        authors=authors,
        year=work.get("publication_year"),
        journal_name=journal_name,
        journal_issn=journal_issn,
        citation_count=work.get("cited_by_count"),
        abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
        source_api="openalex",
        external_id=external_id,
        full_text_url=work.get("open_access", {}).get("oa_url"),
        # Carried on every paper the app sees (search() included, not only
        # get_work()/get_works_batch() via parse_work_dict), so the Smart Search pipeline can
        # apply the narrow non-article-type screening demotion without a second request.
        openalex_type=work.get("type"),
        is_paratext=bool(work.get("is_paratext") or False),
    )


class OpenAlexClient:
    """OpenAlex reader.

    Pass *http_client* to inject a mock in tests; production code constructs
    ``OpenAlexClient()`` and transparently shares the pooled client and limiter.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self._client = http_client
        self._limiter = get_shared_rate_limiter()

    def _get_client(self) -> httpx.AsyncClient:
        return self._client if self._client is not None else get_shared_http_client()

    async def _get_json(self, path: str, params: dict) -> dict | None:
        """GET ``{OPENALEX_BASE_URL}{path}``; ``None`` on 404, raises on other errors."""
        client = self._get_client()
        request_params = dict(params)
        if settings.openalex_email:
            request_params["mailto"] = settings.openalex_email
        if settings.openalex_api_key:
            request_params["api_key"] = settings.openalex_api_key
        headers = _polite_headers()

        async def _call() -> httpx.Response:
            await self._limiter.acquire()
            resp = await client.get(
                f"{OPENALEX_BASE_URL}{path}", params=request_params, headers=headers
            )
            if resp.status_code == 404:
                return resp
            if not resp.is_success:
                # Same coverage as raise_for_status() (any non-2xx, incl. redirects —
                # the shared client has follow_redirects=False), but with the api_key
                # redacted: str(exc) flows into logged tracebacks and error=str(e)
                # job rows that authenticated users can read.
                kind = "Client" if resp.status_code < 500 else "Server"
                raise httpx.HTTPStatusError(
                    f"{kind} error '{resp.status_code} {resp.reason_phrase}' "
                    f"for url '{_redact(str(resp.url))}'",
                    request=resp.request,
                    response=resp,
                )
            return resp

        response = await request_with_retry(_call)
        if response.status_code == 404:
            logger.info("OpenAlex 404 for %s", path)
            return None
        return response.json()

    async def search(
        self,
        query: str,
        *,
        year_from: int | None = None,
        year_to: int | None = None,
        min_citations: int | None = None,
        to_publication_date: date | None = None,
        per_page: int = 25,
    ) -> list[PaperData]:
        """Keyword search. Wildcards are stripped and the query is length-capped.

        ``to_publication_date``: an OpenAlex ``to_publication_date``
        filter, so the search only sees works published on or before that date -- used to
        freeze the corpus a replayed Smart Search run sees, since OpenAlex's index keeps
        growing after any given date.
        """
        cleaned = sanitize_query(query)
        if not cleaned:
            logger.warning("OpenAlex search skipped: query empty after sanitization")
            return []

        params: dict = {
            "search": cleaned,
            "per_page": per_page,
            "sort": "cited_by_count:desc",
            "select": WORK_SELECT_FIELDS,
        }

        filters: list[str] = []
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:{year_from}-")
        elif year_to:
            filters.append(f"publication_year:-{year_to}")
        if min_citations:
            filters.append(f"cited_by_count:>{min_citations}")
        if to_publication_date:
            filters.append(f"to_publication_date:{to_publication_date.isoformat()}")
        if filters:
            params["filter"] = ",".join(filters)

        data = await self._get_json("/works", params)
        if not data:
            return []
        return [_parse_work(work) for work in data.get("results", [])]

    async def get_work(
        self, work_id_or_doi: str, *, with_references: bool = True
    ) -> dict | None:
        """Fetch one work by OpenAlex W-id, bare DOI or DOI URL.

        Returns a :func:`parse_work_dict` dict, or ``None`` when OpenAlex answers 404
        (common for very old CNKI DOIs — surface that honestly, never fabricate).
        """
        identifier = normalize_work_id(work_id_or_doi)
        if not identifier:
            return None
        select = WORK_SELECT_FIELDS_WITH_REFS if with_references else WORK_SELECT_FIELDS
        data = await self._get_json(f"/works/{identifier}", {"select": select})
        if not data:
            return None
        return parse_work_dict(data)

    async def get_works_batch(self, ids: list[str]) -> list[dict]:
        """Fetch metadata for many works in one request per chunk.

        OpenAlex allows at most 50 pipe-separated OR values in a single filter
        ("You can combine up to 50 values with pipes." — docs.openalex.org), so the
        input is chunked at ``settings.openalex_batch_size`` (capped at 50 here).
        """
        seen: set[str] = set()
        unique: list[str] = []
        for raw in ids:
            short = short_work_id(raw)
            if short and short not in seen:
                seen.add(short)
                unique.append(short)

        chunk_size = max(1, min(settings.openalex_batch_size, 50))
        results: list[dict] = []
        use_singles = filter_queries_blocked("batch")
        single_failures = 0
        last_single_error: httpx.HTTPStatusError | None = None
        for start in range(0, len(unique), chunk_size):
            chunk = unique[start : start + chunk_size]
            if not use_singles:
                try:
                    data = await self._get_json(
                        "/works",
                        {
                            "filter": f"openalex_id:{'|'.join(chunk)}",
                            "per_page": len(chunk),
                            "select": WORK_SELECT_FIELDS_WITH_REFS,
                        },
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 429:
                        raise
                    # Observed in production: filter queries 429 with a quota-length
                    # Retry-After (shared Zeabur egress IP) while single-work GETs
                    # still answer. Hydrate one id at a time instead of failing.
                    from app.clients.http_retry import retry_after_seconds

                    note_filter_block("batch", retry_after_seconds(exc))
                    logger.warning(
                        "OpenAlex batch filter rate-limited (Retry-After=%s); "
                        "falling back to single-work fetches for %d ids",
                        exc.response.headers.get("Retry-After"),
                        len(unique) - start,
                    )
                    use_singles = True
                else:
                    if data:
                        results.extend(
                            parse_work_dict(work) for work in data.get("results", [])
                        )
                    continue
            for wid in chunk:
                try:
                    work = await self.get_work(wid)
                except httpx.HTTPStatusError as exc:
                    single_failures += 1
                    last_single_error = exc
                    logger.warning(
                        "OpenAlex single-work fallback failed for %s: HTTP %s",
                        wid,
                        exc.response.status_code,
                    )
                    continue
                if work is not None:
                    results.append(work)
        if not results and single_failures and last_single_error is not None:
            # Every id failed over HTTP: this is an outage, not "no references" —
            # propagate so callers count a real failure (and trip their breaker)
            # instead of silently recording an empty result.
            raise last_single_error
        return results

    async def get_citations(self, work_id: str, *, per_page: int = 100) -> list[dict]:
        """Fetch works citing *work_id* (``filter=cites:Wxxx``), most-cited first.

        Requires an OpenAlex W-id; ``cites:`` does not accept DOIs. The returned dicts
        carry ``referenced_works == []`` because that field is not selected here.
        """
        short = short_work_id(work_id)
        if not short or not short.upper().startswith("W"):
            logger.warning("OpenAlex get_citations needs a W-id, got %r", work_id)
            return []

        data = await self._get_json(
            "/works",
            {
                "filter": f"cites:{short}",
                "per_page": min(per_page, 200),
                "select": WORK_SELECT_FIELDS,
                "sort": "cited_by_count:desc",
            },
        )
        if not data:
            return []
        results = []
        for work in data.get("results", []):
            parsed = parse_work_dict(work)
            parsed["referenced_works"] = []
            results.append(parsed)
        return results
