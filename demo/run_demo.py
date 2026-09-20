#!/usr/bin/env python
"""ScholarRAG v1.1.0 reviewer-executable demonstration.

Drives a locally running ScholarRAG instance (``docker compose up -d``) through the
complete workflow described in the manuscript, using only the public HTTP API:

1. register (or log in) a throw-away demo user
2. create a project
3. Smart Search with the fixed protocol question and criteria, venue filter off
4. export the screening record (JSON + CSV)
5. add the twelve fixed open-access seed papers (metadata from OpenAlex, Crossref
   as fallback)
6. check that both protocol claims carry a citation the backend verifier can parse
   and that it keys to the resolved seed paper (fails hard otherwise)
7. acquire their full texts (Unpaywall -> publisher PDF cascade, server side)
8. AI Write the protocol's literature-review section
9. append the two constructed protocol claims to the draft (done by this script,
   not by the model)
10. run claim verification and fetch the report
11. print a summary table, compare with ``demo/expected/`` when present, and save
    ``summary.json``

Exit codes: 0 success; 1 a step failed; 2 the run completed but a protocol claim did
not receive its expected status (or the report was empty), or the offline delivered-
text checker (``check_delivered.py``) found a violation; 3 ``--strict`` and the
comparison with ``demo/expected/`` drifted beyond tolerance.

Dependencies: Python 3.12+ standard library and ``httpx`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import check_delivered
import httpx

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_API_URL = "http://localhost:8000/api/v1"
DEFAULT_PROTOCOL = DEMO_DIR / "protocol.json"
DEFAULT_SEEDS = DEMO_DIR / "seed_dois.json"
#: Extra sections written and verified after the protocol section, on the same
#: project and the same 12 seed papers. Optional: a checkout without this file
#: runs exactly as before.
DEFAULT_SECTIONS = DEMO_DIR / "sections.json"
DEFAULT_EXPECTED = DEMO_DIR / "expected"
DEFAULT_OUTPUT_ROOT = DEMO_DIR / "output"
# Default wait for every background task except Smart Search: comfortably above the
# acquire-full-texts, AI Write and claim-verification steps' own observed running time.
DEFAULT_TIMEOUT_S = 40 * 60
# Smart Search's own inclusion judge runs once per job, after the search loop itself, as
# a concurrent stage with its own budget. Composition: search budget (1800 s) plus the
# judge stage budget (1800 s) plus one already-launched judge call (420 s) plus margin
# (480 s) equals 4500 s, so the wait covers the job's worst case with room to spare.
DEFAULT_SEARCH_TIMEOUT_S = 4500
OPENALEX_API = "https://api.openalex.org"
CROSSREF_API = "https://api.crossref.org"

POLL_INTERVAL_S = 5.0
POLL_MAX_RETRIES = 5  # consecutive transport errors / 5xx tolerated while polling a task
# How often, in seconds, to print an elapsed-time line while a poll's progress and
# progress_message are unchanged, so a long, silent stage still shows the wait is alive.
PROGRESS_HEARTBEAT_S = 60.0
COUNT_TOLERANCE = 0.30  # +-30 % on screening flow counts (retrieval drifts over time)
COUNT_ABS_SLACK = 2  # ... or +-2 records, whichever is larger (covers expected values of 0)
# A rate (0.0-1.0), unlike a count, can never use COUNT_ABS_SLACK as its floor: 2
# exceeds the whole possible range and would make every rate "pass".
RATE_ABS_SLACK = 0.15
SUMMARY_VERSION = "1.1.0"

EXIT_OK = 0
EXIT_FAILED = 1  # a step failed (DemoError)
EXIT_CLAIMS = 2  # the run completed but a protocol claim did not get its expected status
EXIT_DRIFT = 3  # --strict and the comparison with demo/expected/ is outside tolerance

# Same pattern the backend claim extractor uses (services/fulltext.py::_CITATION_RE):
# "(Author, 2020)", "[Author, 2020]", "(Author et al., 2020)", "(Author & Other, 2020)".
CITATION_RE = re.compile(
    r"[\[\(]"
    r"([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)"
    r",?\s*"
    r"(\d{4})"
    r"[\]\)]"
)

# Same pattern the backend's finalize step uses to strip a surviving marker
# (services/fulltext.py::_NEEDS_CITATION_RE): tolerant of extra internal whitespace and
# case, and of the leading space the marker itself leaves behind once removed. Widened
# from the exact-form ``\[\s*NEEDS\s+CITATION\s*\]`` to also catch whatever the
# writer put between "CITATION" and the closing bracket -- a colon and an explanatory
# clause, an em dash, or nothing -- the same widening as
# ``app.services.fulltext._NEEDS_CITATION_RE``, ``app.services.citation_audit.
# NEEDS_CITATION_FLAG`` and ``demo/check_delivered.py``'s own copy; a plain numbered
# citation ("[12]") or bracketed author-year aside ("[Smith, 2020]") is never matched,
# since neither carries "NEEDS" immediately before "CITATION".
NEEDS_CITATION_RE = re.compile(r"\s*\[\s*NEEDS\s+CITATION\b[^\]]*\]", re.IGNORECASE)

TERMINAL_FAILURE = ("failed", "cancelled")


# ---------------------------------------------------------------------------
# Errors and the API client
# ---------------------------------------------------------------------------


class DemoError(RuntimeError):
    """A demo step failed; the message is meant for the reviewer's terminal."""


class ApiError(DemoError):
    """An HTTP call returned an unexpected status or could not be made at all."""

    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _response_detail(response: httpx.Response) -> Any:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


class ApiClient:
    """Thin JSON client for the ScholarRAG API with Bearer authentication.

    ``transport`` is injectable so tests can use ``httpx.MockTransport``.

    The access token issued at registration lives for only
    ``JWT_ACCESS_TOKEN_EXPIRE_MINUTES`` (15 by default, ``backend/app/config.py``), and
    the per-paper full-text grounding plus the verify/revise/re-verify passes can make
    the AI Write step alone run past that budget. ``refresh_token`` (the response every
    ``/auth/register`` and ``/auth/login`` call carries) lets ``request`` recover from
    exactly one 401 by calling ``POST /auth/refresh`` and retrying the failed call once,
    instead of the whole run dying on the next poll.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.token: str | None = None
        self.refresh_token: str | None = None
        self._http = httpx.Client(base_url=self.base_url, transport=transport, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _refresh_access_token(self) -> bool:
        """POST /auth/refresh once and replace ``self.token`` in place. Returns False
        (never raises) when there is no refresh token to use or the refresh call itself
        fails, so the caller falls back to raising the original 401 rather than looping."""
        if not self.refresh_token:
            return False
        try:
            response = self._http.request(
                "POST",
                "auth/refresh",
                json={"refresh_token": self.refresh_token},
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError:
            return False
        if response.status_code != 200:
            return False
        try:
            new_token = response.json().get("access_token")
        except ValueError:
            return False
        if not new_token:
            return False
        self.token = new_token
        return True

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        ok: tuple[int, ...] = (200, 201, 202),
    ) -> httpx.Response:
        relative = path.lstrip("/")
        try:
            response = self._http.request(
                method, relative, json=json, params=params, headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise ApiError(
                f"{method} {path}: could not reach {self.base_url} ({exc.__class__.__name__}: "
                f"{exc}). Is the stack running (docker compose up -d)?"
            ) from exc
        # A 401 partway through a long run means the access token expired, not that
        # the credentials are wrong: refresh it once and retry the same call before
        # treating the failure as fatal.
        if response.status_code == 401 and self._refresh_access_token():
            try:
                response = self._http.request(
                    method, relative, json=json, params=params, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise ApiError(
                    f"{method} {path}: could not reach {self.base_url} "
                    f"({exc.__class__.__name__}: {exc}). Is the stack running "
                    "(docker compose up -d)?"
                ) from exc
        if response.status_code not in ok:
            detail = _response_detail(response)
            raise ApiError(
                f"{method} {path} returned HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
                detail=detail,
            )
        return response

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self.request(method, path, **kwargs).json()


# ---------------------------------------------------------------------------
# Workflow steps (each one is a plain function over the client)
# ---------------------------------------------------------------------------


def register_or_login(client: ApiClient, suffix: str, password: str) -> dict[str, str]:
    """POST /auth/register (201); on 409 (email exists) POST /auth/login instead."""
    email = f"scholarrag-demo-{suffix}@example.com"
    body = {
        "email": email,
        "password": password,
        "name": f"ScholarRAG demo {suffix}",
        "expertise_level": "researcher",
    }
    try:
        data = client.json("POST", "/auth/register", json=body, ok=(201,))
        mode = "registered"
    except ApiError as exc:
        if exc.status_code != 409:
            raise
        data = client.json(
            "POST", "/auth/login", json={"email": email, "password": password}, ok=(200,)
        )
        mode = "logged_in"
    client.token = data["access_token"]
    client.refresh_token = data.get("refresh_token")
    return {"email": email, "user_id": str(data["user"]["id"]), "mode": mode}


def create_project(client: ApiClient, protocol: dict[str, Any]) -> str:
    body = {
        "title": f"ScholarRAG demo: {protocol['topic']}"[:500],
        "description": protocol["research_question"],
        "citation_style": "APA",
    }
    return str(client.json("POST", "/projects", json=body, ok=(201,))["id"])


def start_smart_search(
    client: ApiClient,
    project_id: str,
    protocol: dict[str, Any],
    *,
    queries_override: list[list[str]] | None = None,
) -> str:
    """POST /projects/{id}/smart-search with the protocol's fixed question and criteria.

    ``publication_date_max`` is sent whenever the protocol carries it: an OpenAlex
    ``to_publication_date`` filter that freezes the corpus a run sees to whatever was
    indexed on or before that date. ``queries_override`` (``--replay-queries``), when
    given, replays a prior run's own per-round queries instead of asking the query
    generator agent for new ones.
    """
    body: dict[str, Any] = {
        "query": protocol["research_question"],
        "inclusion_criteria": list(protocol.get("inclusion_criteria", [])),
        "exclusion_criteria": list(protocol.get("exclusion_criteria", [])),
        "wos_filter": protocol.get("wos_filter", "off"),
    }
    if protocol.get("publication_date_max"):
        body["publication_date_max"] = protocol["publication_date_max"]
    if queries_override:
        body["queries_override"] = queries_override
    try:
        data = client.json("POST", f"/projects/{project_id}/smart-search", json=body, ok=(202,))
    except ApiError as exc:
        if exc.status_code == 409:
            raise DemoError(
                f"Smart Search refused (HTTP 409: {exc.detail}). With wos_filter='off' this "
                "means another Smart Search is still running for the project."
            ) from exc
        raise
    return str(data["task_id"])


def poll_task(
    client: ApiClient,
    task_id: str,
    *,
    label: str,
    timeout_s: float,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
    log: Callable[[str], None] = print,
    interval_s: float = POLL_INTERVAL_S,
    timeout_flag: str = "--timeout",
) -> dict[str, Any]:
    """GET /tasks/{id} until ``status`` is terminal; print progress as it changes.

    A transient failure of the poll itself (connection error or HTTP 5xx) is retried up
    to ``POLL_MAX_RETRIES`` consecutive times, because the backend job keeps running
    regardless of whether one status request got through.

    The job's own ``progress``/``progress_message`` are printed as soon as either
    changes, so a stage the job reports on its own (an inclusion-judge stage naming
    itself, for example) shows up the moment the job says so. When neither changes for
    ``PROGRESS_HEARTBEAT_S`` seconds, an elapsed-time line is printed instead, so a long
    stage that reports no progress of its own does not look hung.

    ``timeout_flag`` names the CLI option that controls ``timeout_s``, for the message
    raised on a timeout; callers polling a task under a different option than
    ``--timeout`` (Smart Search's own ``--search-timeout``) pass it explicitly.
    """
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    started = clock()
    last_seen: tuple[Any, Any] | None = None
    last_heartbeat = started
    consecutive_errors = 0
    while True:
        try:
            task = client.json("GET", f"/tasks/{task_id}")
        except ApiError as exc:
            transient = exc.status_code is None or exc.status_code >= 500
            consecutive_errors += 1
            if not transient or consecutive_errors > POLL_MAX_RETRIES:
                raise
            log(
                f"  [{label}] poll failed ({exc}); retry "
                f"{consecutive_errors}/{POLL_MAX_RETRIES} in {interval_s:.0f} s"
            )
            sleep(interval_s)
            continue
        consecutive_errors = 0
        status = task.get("status")
        seen = (task.get("progress"), task.get("progress_message"))
        now = clock()
        if seen != last_seen:
            progress = task.get("progress") or 0.0
            log(f"  [{label}] {progress:6.1%}  {task.get('progress_message') or status}")
            last_seen = seen
            last_heartbeat = now
        elif now - last_heartbeat >= PROGRESS_HEARTBEAT_S:
            log(f"  [{label}] still {status} ({now - started:.0f} s elapsed)")
            last_heartbeat = now
        if status == "completed":
            return task
        if status in TERMINAL_FAILURE:
            raise DemoError(f"{label}: task {task_id} {status}: {task.get('error') or 'no error'}")
        if clock() - started >= timeout_s:
            raise DemoError(
                f"{label}: task {task_id} still '{status}' after {timeout_s:.0f} s; "
                f"re-run with a larger {timeout_flag} or inspect the backend logs "
                "(docker compose logs backend)."
            )
        sleep(interval_s)


def fetch_screening_record(client: ApiClient, task_id: str) -> tuple[dict[str, Any], str]:
    """GET /tasks/{id}/screening-record in both formats."""
    record = client.json("GET", f"/tasks/{task_id}/screening-record", params={"format": "json"})
    csv_text = client.request(
        "GET", f"/tasks/{task_id}/screening-record", params={"format": "csv"}
    ).text
    return record, csv_text


def _reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted.items():
        positions.extend((i, word) for i in indexes)
    positions.sort()
    return " ".join(word for _, word in positions) or None


def _crossref_as_work(message: dict[str, Any]) -> dict[str, Any]:
    """Shape a Crossref ``message`` like the subset of an OpenAlex work the demo uses."""
    authors = []
    for a in message.get("author") or []:
        name = " ".join(part for part in (a.get("given"), a.get("family")) if part).strip()
        if name:
            authors.append({"author": {"display_name": name}})
    year = None
    for key in ("published-print", "published-online", "issued"):
        parts = ((message.get(key) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0]:
            year = parts[0]
            break
    titles = message.get("container-title") or []
    return {
        "id": None,
        "title": (message.get("title") or [None])[0],
        "publication_year": year,
        "authorships": authors,
        "primary_location": {
            "source": {"display_name": titles[0] if titles else None, "issn": message.get("ISSN")}
        },
        "abstract_inverted_index": None,
        "cited_by_count": message.get("is-referenced-by-count"),
        "metadata_source": "crossref",
    }


def openalex_resolver(
    http: httpx.Client | None = None,
    *,
    sleep: Callable[[float], None] | None = None,
    env: dict[str, str] | None = None,
) -> Callable[[str], dict[str, Any] | None]:
    """Return a function DOI -> work dict (or None) for the seed metadata.

    OpenAlex is asked first (``select`` keeps the response small). Nothing personal is
    sent unless the reviewer opts in through ``OPENALEX_EMAIL`` (polite pool) or
    ``OPENALEX_API_KEY`` (account quota), the same variables the backend uses. A 429 or
    5xx is retried a few times with backoff; when OpenAlex still fails, Crossref supplies
    the author list, year and journal so that citations can be matched.
    """
    http = http or httpx.Client(timeout=30.0)
    sleep = sleep or time.sleep
    env = os.environ if env is None else env
    params_extra: dict[str, str] = {}
    if env.get("OPENALEX_EMAIL"):
        params_extra["mailto"] = env["OPENALEX_EMAIL"]
    if env.get("OPENALEX_API_KEY"):
        params_extra["api_key"] = env["OPENALEX_API_KEY"]

    def _get_json(url: str, params: dict[str, str]) -> dict[str, Any] | None:
        for attempt in range(3):
            try:
                response = http.get(url, params=params)
            except httpx.HTTPError:
                response = None
            if response is not None and response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    return None
            if response is not None and response.status_code not in (429, 500, 502, 503, 504):
                return None
            sleep(2.0 * (attempt + 1))
        return None

    def resolve(doi: str) -> dict[str, Any] | None:
        select = (
            "id,doi,title,publication_year,authorships,primary_location,"
            "abstract_inverted_index,cited_by_count"
        )
        work = _get_json(
            f"{OPENALEX_API}/works/https://doi.org/{doi}", {"select": select, **params_extra}
        )
        if work and work.get("authorships"):
            return work
        crossref = _get_json(f"{CROSSREF_API}/works/{doi}", {})
        if crossref and isinstance(crossref.get("message"), dict):
            fallback = _crossref_as_work(crossref["message"])
            if work:  # OpenAlex answered without authors: keep its abstract/id, add authors
                work["authorships"] = fallback["authorships"]
                work["metadata_source"] = "openalex+crossref"
                return work
            return fallback
        return work

    return resolve


def build_paper_data(seed: dict[str, Any], work: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a seed_dois.json entry with its OpenAlex record into a ``PaperData`` body.

    The seed file is the source of truth for DOI, title, year, journal and OpenAlex id;
    OpenAlex adds authors (needed for APA citations and claim matching), abstract, ISSN
    and citation count. ``authors`` in the seed file, if present, take precedence.
    """
    work = work or {}
    authors = seed.get("authors") or [
        {"name": a["author"]["display_name"]}
        for a in work.get("authorships", []) or []
        if isinstance(a, dict) and (a.get("author") or {}).get("display_name")
    ]
    source = ((work.get("primary_location") or {}).get("source") or {}) if work else {}
    issns = source.get("issn") or []
    return {
        "doi": seed["doi"],
        "title": seed.get("title") or work.get("title") or seed["doi"],
        "authors": authors,
        "year": seed.get("year") or work.get("publication_year"),
        "journal_name": seed.get("journal") or source.get("display_name"),
        "journal_issn": issns[0] if issns else None,
        "citation_count": work.get("cited_by_count"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "source_api": "openalex",
        "external_id": seed.get("openalex_id") or (work.get("id") or "").rsplit("/", 1)[-1] or None,
        "full_text_url": seed.get("oa_pdf_url"),
    }


def add_seed_papers(
    client: ApiClient,
    project_id: str,
    seeds: dict[str, Any],
    resolver: Callable[[str], dict[str, Any] | None],
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """POST /projects/{id}/papers once per seed; 409 means the paper is already linked."""
    added: list[str] = []
    already: list[str] = []
    failed: list[str] = []
    without_authors: list[str] = []
    paper_data_by_doi: dict[str, dict[str, Any]] = {}
    for seed in seeds["papers"]:
        doi = seed["doi"]
        paper_data = build_paper_data(seed, resolver(doi))
        paper_data_by_doi[doi] = paper_data
        if not paper_data["authors"]:
            without_authors.append(doi)
        try:
            client.json(
                "POST", f"/projects/{project_id}/papers", json={"paper_data": paper_data}, ok=(201,)
            )
            added.append(doi)
        except ApiError as exc:
            if exc.status_code == 409:
                already.append(doi)
            else:
                failed.append(f"{doi}: {exc}")
        log(f"  [seeds] {len(added) + len(already) + len(failed)}/{len(seeds['papers'])} {doi}")
    if failed:
        raise DemoError("Adding seed papers failed:\n  " + "\n  ".join(failed))
    if without_authors:
        log(
            "  [seeds] WARNING: no authors resolved for "
            + ", ".join(without_authors)
            + " (OpenAlex unreachable?) - citations to these papers cannot be matched"
        )
    return {
        "requested": len(seeds["papers"]),
        "added": added,
        "already_present": already,
        "without_authors": without_authors,
        "paper_data": paper_data_by_doi,
    }


def start_fulltext_acquisition(client: ApiClient, project_id: str) -> str:
    data = client.json(
        "POST", f"/projects/{project_id}/acquire-full-texts", json={"paper_ids": []}, ok=(202,)
    )
    return str(data["task_id"])


def create_draft(client: ApiClient, project_id: str, protocol: dict[str, Any]) -> str:
    body = {"title": protocol["section"]["title"], "paper_type": "literature_review"}
    return str(client.json("POST", f"/projects/{project_id}/drafts", json=body, ok=(201,))["id"])


def start_generation(
    client: ApiClient, draft_id: str, protocol: dict[str, Any], *, section_title: str | None = None
) -> str:
    """*section_title*, when given, is sent as the request's own ``section_title`` so
    the backend saves this exact wording as the section's H2 (and compares the model's
    own opening heading line against it), instead of the ``section_type``-derived
    default ("literature_review" -> "Literature Review"). Used by
    `_write_one_extra_section` for each entry of `demo/sections.json`, whose own
    requested title otherwise never reached the saved draft's own heading; left unset
    for the protocol section's own call, so its long-standing "Literature Review"
    heading is unaffected."""
    body = {
        "section_type": "literature_review",
        "context": protocol["section"].get("instructions"),
        "language": "en",
    }
    if section_title is not None:
        body["section_title"] = section_title
    # Only sent when the protocol names a target, so a protocol without one (an older
    # fixture, a hand-run request) still generates with no length control.
    target_words = protocol["section"].get("target_words")
    if target_words is not None:
        body["target_words"] = target_words
    return str(client.json("POST", f"/drafts/{draft_id}/generate", json=body, ok=(200,))["task_id"])


def first_author_surname(paper_data: dict[str, Any]) -> str | None:
    """Surname key exactly as the backend builds it (``_build_paper_lookup``)."""
    authors = paper_data.get("authors") or []
    if not authors:
        return None
    first = authors[0]
    name = first if isinstance(first, str) else str(first.get("name", ""))
    tokens = name.split(",")[0].split()
    return tokens[-1] if tokens else None


ANY_CITATION_RE = re.compile(r"[\[(][^\[\]()]*\d{4}[a-z]?[\])]")


def ensure_citation(text: str, paper_data: dict[str, Any]) -> str:
    """Append ``(Surname, Year)`` when the claim text carries no citation at all.

    A citation the backend cannot parse (multi-word or accented surname) is left as
    written; ``prepare_claims`` flags it via ``citation_parseable`` instead.
    """
    if CITATION_RE.search(text) or ANY_CITATION_RE.search(text):
        return text
    surname = first_author_surname(paper_data)
    year = paper_data.get("year")
    if not surname or not year:
        return text
    body = text.rstrip().rstrip(".")
    return f"{body} ({surname}, {year})."


def prepare_claims(
    protocol: dict[str, Any], paper_data_by_doi: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for claim in protocol["claims"]:
        paper = paper_data_by_doi.get(claim["source_doi"], {})
        text = ensure_citation(claim["text"], paper)
        match = CITATION_RE.search(text)
        claims.append(
            {
                "id": claim["id"],
                "text": text,
                "source_doi": claim["source_doi"],
                "expected_status": claim["expected_status"],
                "construction": claim.get("construction", ""),
                "citation_parseable": match is not None,
                "citation_key": (
                    f"{match.group(1).split()[0].lower()}_{match.group(2)}" if match else None
                ),
                "paper_key": (
                    f"{first_author_surname(paper).lower()}_{paper.get('year')}"
                    if first_author_surname(paper) and paper.get("year")
                    else None
                ),
            }
        )
    return claims


def check_claim_citations(claims: list[dict[str, Any]]) -> None:
    """Raise DemoError unless every claim will reach the backend verifier.

    The backend only extracts sentences whose citation matches ``CITATION_RE`` and only
    matches them to a library paper whose ``surname_year`` key (first author's last name
    token + year) equals the citation's; anything else silently drops out of the report.
    """
    problems: list[str] = []
    for claim in claims:
        if not claim["citation_parseable"]:
            problems.append(
                f"claim {claim['id']}: no citation the backend can parse - the text must end "
                "with '(Surname, YEAR)' or '(Surname et al., YEAR)' where Surname is a single "
                f"ASCII capitalised token: {claim['text'][-80:]!r}"
            )
            continue
        if claim["paper_key"] is None:
            problems.append(
                f"claim {claim['id']}: no authors/year resolved for seed {claim['source_doi']} "
                "(OpenAlex and Crossref unreachable?), so the citation cannot be matched"
            )
        elif claim["citation_key"] != claim["paper_key"]:
            problems.append(
                f"claim {claim['id']}: citation key {claim['citation_key']!r} differs from the "
                f"key the backend derives for {claim['source_doi']} ({claim['paper_key']!r})"
            )
    if problems:
        raise DemoError("Protocol claims cannot be verified:\n  " + "\n  ".join(problems))


def _paragraph(
    text: str, citation_links: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "paragraph", "content": [{"type": "text", "text": text}]}
    if citation_links:
        node["attrs"] = {"citationLinks": citation_links}
    return node


SENTENCE_TERMINATORS = (".", "!", "?")


def terminate_sentence(text: str) -> str:
    """Return ``text`` ending in ``.``, ``!`` or ``?`` (a ``.`` is appended when needed).

    The backend claim extractor joins all paragraphs with spaces and splits sentences only
    at ``[.!?]`` followed by whitespace, so a generated block that ends without terminal
    punctuation (e.g. in ``[NEEDS CITATION]``) would otherwise be merged with the first
    appended claim into one sentence and the verifier would judge the merged text.
    """
    stripped = text.rstrip()
    if not stripped or stripped.endswith(SENTENCE_TERMINATORS):
        return stripped
    return stripped + "."


def build_draft_content(
    section_title: str,
    generated_text: str,
    claim_texts: list[str],
    citation_links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Tiptap document: H2 heading, one paragraph per blank-line block, then the claims.

    Mirrors what the web UI does when it accepts an AI Write result; the claims are
    appended by this script as a final paragraph each. The last generated block is
    terminated with ``.`` when it does not already end a sentence, so that the backend
    sentence splitter isolates the appended claims verbatim.

    ``citation_links`` is the writer's own sentence-to-reference map, grouped here by
    ``paragraph_index`` onto the matching generated-block paragraph node, exactly as
    ``drafts-tab.tsx`` does when it accepts an AI Write result. The blank-line split
    below (``if p.strip()``) uses the same predicate as the writer's own
    ``[b for b in content.split("\\n\\n") if b.strip()]``, so a link's
    ``paragraph_index`` lands on the same block here as it did there; stripping each
    kept block only changes its stored text, not which blocks are kept or their order.
    """
    nodes: list[dict[str, Any]] = [
        {
            "type": "heading",
            "attrs": {"level": 2},
            "content": [{"type": "text", "text": section_title}],
        }
    ]
    blocks = [p.strip() for p in generated_text.split("\n\n") if p.strip()]
    if blocks and claim_texts:
        blocks[-1] = terminate_sentence(blocks[-1])

    links_by_paragraph: dict[int, list[dict[str, Any]]] = {}
    for link in citation_links or []:
        index = link.get("paragraph_index")
        if not isinstance(index, int):
            continue
        links_by_paragraph.setdefault(index, []).append(
            {
                "sentence": link.get("sentence"),
                "keys": link.get("keys") or [],
                "citation_text": link.get("citation_text"),
            }
        )

    nodes.extend(_paragraph(b, links_by_paragraph.get(i)) for i, b in enumerate(blocks))
    nodes.extend(_paragraph(c) for c in claim_texts)
    return {"type": "doc", "content": nodes}


def save_draft(client: ApiClient, draft_id: str, content: dict[str, Any]) -> None:
    client.json("PUT", f"/drafts/{draft_id}", json={"content": content}, ok=(200,))


def fetch_draft(client: ApiClient, draft_id: str) -> dict[str, Any]:
    """The draft row itself (``DraftResponse``), notably its own current ``content``:
    the write job's own gated loop verifies, finalizes and saves the AI-written
    section itself, so the demo reads back what the backend already saved instead of
    building and PUTting that content itself."""
    return client.json("GET", f"/drafts/{draft_id}")["draft"]


def append_claim_paragraphs(
    existing_content: dict[str, Any] | None, claim_texts: list[str]
) -> dict[str, Any]:
    """Append one paragraph per demo-authored claim fixture onto *existing_content*:
    the write job's own gated loop already verified, finalized and saved the AI's own
    section; the two protocol claims (deliberately constructed, known-good and
    known-bad restatements of one seed paper's abstract) are appended afterwards,
    exactly as ``build_draft_content`` used to append them onto the document it built
    from scratch. The existing content's own last paragraph is terminated with '.'
    when it does not already end a sentence, so the backend's sentence splitter never
    merges it with the first appended claim."""
    nodes = list((existing_content or {}).get("content") or [])
    if claim_texts and nodes and nodes[-1].get("type") == "paragraph":
        last_text = "".join(
            c.get("text", "")
            for c in nodes[-1].get("content") or []
            if c.get("type") == "text"
        )
        terminated = terminate_sentence(last_text)
        if terminated != last_text:
            nodes[-1] = _paragraph(terminated)
    nodes.extend(_paragraph(c) for c in claim_texts)
    return {"type": "doc", "content": nodes}


def start_verification(client: ApiClient, project_id: str, draft_id: str) -> str:
    data = client.json(
        "POST", f"/projects/{project_id}/verify-and-heal", json={"draft_id": draft_id}, ok=(202,)
    )
    return str(data["task_id"])


def fetch_claim_report(client: ApiClient, draft_id: str) -> dict[str, Any]:
    return client.json("GET", f"/drafts/{draft_id}/claim-verification")


def _draft_paragraph_texts(content: dict[str, Any] | None) -> list[str]:
    """Every paragraph's own concatenated text from a Tiptap draft document, in document
    order. Reading the saved ``draft_content.json`` back, rather than trusting anything
    the writing or verification steps reported, is what lets
    `check_claim_text_in_draft` see exactly what the draft carries."""
    texts: list[str] = []
    for node in (content or {}).get("content") or []:
        if not isinstance(node, dict) or node.get("type") != "paragraph":
            continue
        texts.append(
            "".join(
                n.get("text") or ""
                for n in node.get("content") or []
                if isinstance(n, dict) and n.get("type") == "text"
            )
        )
    return texts


def _count_body_words(draft_content: dict[str, Any] | None) -> int:
    """The word count of every paragraph node's own text (`_draft_paragraph_texts`
    already excludes heading nodes, so nothing here needs to strip a markdown "#" line
    or a bold sub-heading the way the backend's own, text-based
    ``app.services.writing._count_body_words`` does for a document that has not been
    split into Tiptap nodes yet).

    The write job's own ``writing.length.final_words`` and
    ``writing.loop_stats.survival_rate`` are recorded right after the gated write
    loop's own finalize pass, before the standalone verify-and-heal action
    (`_verify_claims`, `_write_one_extra_section`) runs and can remove more sentences.
    This function recounts the SAVED, HEALED draft directly, so the run's own summary
    reports the length of the text a reader actually receives, not a pre-heal estimate
    of it."""
    return sum(len(text.split()) for text in _draft_paragraph_texts(draft_content))


def check_claim_text_in_draft(
    draft_content: dict[str, Any] | None, verifications: list[dict[str, Any]]
) -> tuple[int, int]:
    """``(n, m)``: of *m* verifications carrying a non-empty ``claim_text``, how many
    have that text occurring verbatim in one of the draft's own paragraphs. A fresh
    verification always stores the extracted sentence, so a value of ``n < m`` means
    either that storage regressed or the report was built against a different draft
    than the one saved. A verification with an empty or missing ``claim_text`` counts
    towards neither."""
    paragraphs = _draft_paragraph_texts(draft_content)
    n = m = 0
    for verification in verifications:
        claim_text = verification.get("claim_text")
        if not claim_text:
            continue
        m += 1
        if any(claim_text in paragraph for paragraph in paragraphs):
            n += 1
    return n, m


def assert_every_final_row_verified(final_verifications: list[dict[str, Any]]) -> None:
    """Part 1 of the standalone verify-and-heal action's own exit invariant:
    ``final_report.verifications`` is documented as the "ONE report of the final
    text", every row verified by construction
    (``app.services.fulltext.verify_and_heal_claims``). Raise the moment a regression
    lets a non-verified row survive into it, instead of trusting the docstring."""
    for row in final_verifications:
        if row.get("status") != "verified":
            raise DemoError(
                "final_report carries a non-verified row after healing: status "
                f"{row.get('status')!r}, claim_text {row.get('claim_text')!r}"
            )


def assert_no_needs_citation_marker(draft_content: dict[str, Any] | None) -> None:
    """Part 2 of the same exit invariant: ``finalize_draft_document`` strips every
    literal ``[NEEDS CITATION]`` marker from the text it heals
    (``app.services.fulltext.verify_and_heal_claims``). Raise if one survives into the
    saved draft this script reads back, in any paragraph."""
    for text in _draft_paragraph_texts(draft_content):
        if NEEDS_CITATION_RE.search(text):
            raise DemoError(
                f"a [NEEDS CITATION] marker survived healing in the saved draft: {text!r}"
            )


def assert_citation_coverage_fully_resolved(citation_coverage: dict[str, Any] | None) -> None:
    """Part 3 of the same exit invariant: every citation occurrence the draft's own
    ``extract_claims_from_document`` recognises must resolve to a paper in the project
    library (``CitationCoverage.unresolved``) -- a citation that resolves to a key
    naming a paper in the library is still sent to the verifier and reported
    ``no_full_text`` when that paper has no chunks, never silently dropped
    (``CitationCoverage.sent_to_verifier``'s own docstring); only a citation that never
    resolves to any project paper at all (a numbered citation with no matching
    reference-list entry, or a key nothing in the library answers to) is skipped
    entirely, with no row, verified or not, ever created for it. ``citation_coverage``
    is ``None`` on a report from a backend build before this field existed, or on
    ``verify_user_edits``'s own plain-string report, in which case there is nothing
    here to check."""
    if citation_coverage is None:
        return
    unresolved = citation_coverage.get("unresolved")
    if unresolved:
        raise DemoError(
            f"{unresolved} citation(s) in the draft never resolved to a library paper "
            "(citation_coverage.unresolved != 0), so they carry no verified row"
        )


def _draft_heading_texts(content: dict[str, Any] | None) -> list[str]:
    """Every top-level heading node's own concatenated text from a Tiptap draft
    document, in document order. Companion to `_draft_paragraph_texts`, used by
    `assert_no_duplicate_leading_heading`."""
    texts: list[str] = []
    for node in (content or {}).get("content") or []:
        if not isinstance(node, dict) or node.get("type") != "heading":
            continue
        texts.append(
            "".join(
                n.get("text") or ""
                for n in node.get("content") or []
                if isinstance(n, dict) and n.get("type") == "text"
            )
        )
    return texts


def assert_no_duplicate_leading_heading(draft_content: dict[str, Any] | None) -> None:
    """Part 4 of the same exit invariant: the writer strips its own leading markdown
    heading line when it repeats the section title
    (`app.services.writing._strip_duplicate_leading_heading`), so the saved draft
    should never carry two consecutive heading nodes with the same text -- the
    section's own H2 heading (`_build_section_tiptap_nodes`) immediately followed by
    the model's own restated title. Checked here, on the draft this script reads back,
    rather than only trusted from the write job's own report, in case a future change
    to the merge or save path reintroduces the duplicate a different way. Compares only
    ADJACENT heading nodes, never every pair, since a legitimate sub-heading later in
    the body may happen to repeat the section title in wording without being a
    duplicate of the opening heading."""
    headings = _draft_heading_texts(draft_content)
    for first_text, second_text in zip(headings, headings[1:]):
        if _norm(first_text) and _norm(first_text) == _norm(second_text):
            raise DemoError(
                "the saved draft opens with two consecutive headings carrying the "
                f"same text: {first_text!r}"
            )


def _parse_n_of_m(text: Any) -> tuple[int, int] | None:
    """Parse the ``"n of m"`` string `check_claim_text_in_draft`'s result is recorded as,
    or ``None`` when *text* is not that shape (an older summary, or the field absent)."""
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"(\d+) of (\d+)", text.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


#: The one guard-demotion shape that does not count as model drift for a fixture
#: expecting ``verified``: the verification model itself judged the claim ``verified``
#: (``model_status``) and the frozen ``quote_not_verbatim`` guard alone (no other
#: reason alongside it) demoted the reported status to ``needs_nuance`` because its
#: own copy of the quote drifted from the source by a word or two. The verifier prompt
#: and the guard set stay frozen; this is a report-reading rule, not a change to
#: either.
_GUARD_DEMOTION_REASONS = ["quote_not_verbatim"]


def is_guard_demotion(expected_status: str, hit: dict[str, Any] | None) -> bool:
    if not hit or expected_status != "verified" or hit.get("status") != "needs_nuance":
        return False
    return (
        hit.get("model_status") == "verified"
        and list(hit.get("machine_reasons") or []) == _GUARD_DEMOTION_REASONS
    )


def match_claim_results(
    claims: list[dict[str, Any]], report: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pair each protocol claim with the report entry whose ``claim_text`` contains it."""
    verifications = report.get("verifications") or []
    rows: list[dict[str, Any]] = []
    for claim in claims:
        wanted = _norm(claim["text"])
        hit = None
        for v in verifications:
            got = _norm(v.get("claim_text", ""))
            if wanted and (wanted in got or got in wanted):
                hit = v
                break
        expected_status = claim["expected_status"]
        guard_demotion = is_guard_demotion(expected_status, hit)
        rows.append(
            {
                "id": claim["id"],
                "text": claim["text"],
                "source_doi": claim["source_doi"],
                "construction": claim["construction"],
                "expected_status": expected_status,
                "status": hit.get("status") if hit else "not_in_report",
                "evidence_quote": hit.get("evidence_quote") if hit else None,
                "explanation": hit.get("explanation") if hit else None,
                "model_reported": hit.get("model_reported") if hit else None,
                # The raw verifier verdict and the guard reasons that changed it, so a
                # guard demotion can be told apart from real model drift here and in
                # `claim_problems`/the printed table below, without touching the frozen
                # guards or verifier prompt.
                "model_status": hit.get("model_status") if hit else None,
                "machine_reasons": list(hit.get("machine_reasons") or []) if hit else [],
                "guard_demotion": guard_demotion,
                "matches_expected": (
                    bool(hit) and hit.get("status") == expected_status
                ) or guard_demotion,
                # How many verification-model calls this claim actually made (1, or 2
                # when the bounded second pass fired), so the report shows this per
                # claim, not only in the aggregated provenance.
                "pass_count": len(hit.get("passes") or []) if hit else 0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Delivered-evidence record: one row per claim that survived into the final, healed
# report the user actually reads, each carrying the verbatim source passage a human
# judge checks it against -- built after the standalone verify-and-heal step, from
# `final_report.verifications` (every one of them "verified" by construction,
# `app.schemas.fulltext.FinalClaimReport`'s own docstring), not from the raw,
# unfiltered `verifications` list `match_claim_results` above reads.
# ---------------------------------------------------------------------------

_EVIDENCE_LOCATION_RE = re.compile(r"^chunk (\d+)(?: \((.*)\))?$")

#: Separator between two quotes' own excerpts inside one row's ``source_passage``:
#: each excerpt is a real, contiguous, verbatim substring of one chunk, so quotes
#: cannot be run together into one window without this marker.
_EXCERPT_SEPARATOR = "\n---\n"


def fetch_fulltext_chunk(client: ApiClient, paper_id: str, chunk_index: int) -> dict[str, Any]:
    """GET /papers/{paper_id}/fulltext-chunks/{chunk_index}: the one read-only
    endpoint added under the existing full-text routes so a chunk index from
    ``ClaimVerification.evidence_location`` resolves to real chunk text and
    section."""
    return client.json("GET", f"/papers/{paper_id}/fulltext-chunks/{chunk_index}")


def parse_evidence_location(location: str | None) -> tuple[int, str | None] | None:
    """``"chunk N (section)"``/``"chunk N"`` (``app.services.fulltext._find_evidence_
    location``'s own format) -> ``(chunk_index, section)``, 0-based -- ``None`` when
    *location* is missing or not that shape."""
    if not location:
        return None
    match = _EVIDENCE_LOCATION_RE.match(location.strip())
    if not match:
        return None
    return int(match.group(1)) - 1, match.group(2)


def _keep_letters_and_digits_with_offsets(text: str) -> tuple[str, list[int]]:
    """*text* folded exactly the way the verifier's own guard folds it for a verbatim
    check (``app.services.fulltext._normalise_for_match``/``_keep_letters_and_digits``:
    NFKC, then letters and digits only, case-folded -- every dash, hyphen, quote mark and
    stray control byte vanishes), plus the raw index of every character kept, in the same
    ``kept``/``offsets`` shape ``app.services.fulltext._normalised_with_offsets`` builds
    for its own, narrower whitespace-collapse fold. NFKC is applied one raw character at a
    time rather than to the whole string at once, so every kept, folded character can
    still be mapped back to the single raw index that produced it; the very rare
    cross-character NFKC compositions this misses are the same trade
    ``_normalised_with_offsets`` already accepts for its own fold. Neither NFKC nor
    ``casefold()`` is length preserving (``"ss".casefold()`` is unchanged but
    ``"ß".casefold()`` -- a German sharp s -- is ``"ss"``, two characters from one),
    so one offset is appended per character of the folded output, not per raw
    character: ``len("".join(kept)) == len(offsets)`` holds for every input. Needed so
    this module's own quote locator can find a span the guard already accepted as
    verbatim under this fold but that none of ``_find_quote_in_text``'s narrower
    matches can find: an en dash on the chunk side against a plain hyphen on the
    quote side, for instance, both fold to nothing under this rule."""
    kept: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        for folded_char in unicodedata.normalize("NFKC", character):
            if folded_char.isalpha() or folded_char.isdigit():
                for out_char in folded_char.casefold():
                    kept.append(out_char)
                    offsets.append(index)
    return "".join(kept), offsets


def _keep_letters_and_digits(text: str) -> str:
    """*text* folded the same way as :func:`_keep_letters_and_digits_with_offsets`,
    without the offset map -- used for the needle side of a guard-fold match, which only
    needs the folded string to search for, never a way back to raw indices."""
    folded, _offsets = _keep_letters_and_digits_with_offsets(text)
    return folded


def _lower_with_offsets(text: str) -> tuple[str, list[int]]:
    """*text* lowered character by character, plus the raw index of every character of
    the lowered output, in the same ``kept``/``offsets`` shape
    :func:`_keep_letters_and_digits_with_offsets` builds for its own fold. ``str.lower()``
    is not length preserving either (``"İ".lower()`` -- a Turkish dotted capital I --
    is two characters), so ``_find_quote_in_text``'s case-insensitive match cannot assume
    a match found in ``text.lower()`` starts and ends at the same offsets in *text*
    itself (the same defect as the guard fold's own offset map, one branch
    earlier)."""
    kept: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        for out_char in character.lower():
            kept.append(out_char)
            offsets.append(index)
    return "".join(kept), offsets


def _find_quote_in_text(text: str, quote: str | None) -> tuple[int, int] | None:
    """The exact span of *quote* in *text*: an exact match first, then a case-
    insensitive one, then a whitespace-tolerant one (a PDF-extraction line wrap or a
    typographic quote mark can separate the model's own copy of a span from the chunk's
    raw bytes), then, as a last resort, the same letters-and-digits fold the verifier's
    own guard uses to decide a quote is verbatim -- so a quote the guard accepted is
    not left unlocated here just because none of the three narrower matches above can
    bridge a dash, hyphen or typographic difference the guard's own fold already
    bridges. ``None`` when *quote* is missing or not found by any of the four (factored
    out of ``trim_source_passage`` so a quote can be checked against more than one
    chunk's text)."""
    if not quote or not text:
        return None
    idx = text.find(quote)
    if idx != -1:
        return idx, idx + len(quote)
    lowered_text, lower_offsets = _lower_with_offsets(text)
    lowered_quote = quote.lower()
    idx = lowered_text.find(lowered_quote)
    if idx != -1:
        end_pos = idx + len(lowered_quote) - 1
        return lower_offsets[idx], lower_offsets[end_pos] + 1
    tokens = [re.escape(t) for t in quote.split() if t]
    if tokens:
        match = re.search(r"\s+".join(tokens), text, re.IGNORECASE)
        if match:
            return match.start(), match.end()
    folded_quote = _keep_letters_and_digits(quote)
    if folded_quote:
        folded_text, offsets = _keep_letters_and_digits_with_offsets(text)
        pos = folded_text.find(folded_quote)
        if pos != -1:
            end_pos = pos + len(folded_quote) - 1
            return offsets[pos], offsets[end_pos] + 1
    return None


def trim_source_passage(chunk_text: str, quote: str | None, max_chars: int = 2000) -> str:
    """A verbatim excerpt of at most *max_chars* characters from *chunk_text*, the source
    text a human judge checks a delivered claim against.

    *chunk_text* is returned whole when it already fits. Otherwise the excerpt is
    centred on *quote* when it can be located (:func:`_find_quote_in_text`), or, when
    *quote* cannot be located at all, taken from the start of *chunk_text*: a reader
    always gets a real, verbatim excerpt of the source rather than nothing.
    """
    if len(chunk_text) <= max_chars:
        return chunk_text
    span = _find_quote_in_text(chunk_text, quote)
    start, end = span if span is not None else (0, min(max_chars, len(chunk_text)))
    pad = max(0, (max_chars - (end - start)) // 2)
    window_start = max(0, start - pad)
    window_end = min(len(chunk_text), window_start + max_chars)
    window_start = max(0, window_end - max_chars)
    return chunk_text[window_start:window_end]


#: `_locate_quote_excerpt`'s own sentinel for "found in the paper's abstract, not any
#: numbered chunk": distinct from every real ``chunk_index`` (always >= 0), so a
#: caller can tell the two apart without a second return value.
ABSTRACT_CHUNK_INDEX = -1


def _locate_quote_excerpt(
    quote: str,
    primary_text: str,
    primary_index: int,
    chunk_count: int,
    fetch_chunk_text: Callable[[int], str],
    *,
    abstract_text: str | None = None,
    max_chars: int = 2000,
) -> tuple[str, int] | None:
    """*quote*'s own verbatim excerpt and the 0-based chunk index it was found in:
    *primary_text* (the chunk ``evidence_location`` names) is tried first, with no
    extra fetch; only when *quote* is not there does this fetch every other chunk
    index up to *chunk_count*, in order, via *fetch_chunk_text*, stopping at the first
    one that contains it.

    When *quote* is in none of those chunks, *abstract_text* -- the paper's own
    abstract, given by the caller rather than fetched here, since every chunk of the
    same paper already carries it -- is tried last, returning
    ``ABSTRACT_CHUNK_INDEX`` in place of a real chunk index. The verifier's own prompt
    gives the model a paper's abstract as context alongside its numbered chunks, so a
    quote genuinely verbatim only there is evidence the guard already accepted, not a
    fabrication this function should refuse to locate just because
    ``evidence_location`` never names the abstract as a chunk.

    ``None`` when *quote* cannot be located in any chunk of the paper or its abstract --
    the caller marks the row instead of fabricating a passage from wherever the primary
    chunk happens to start."""
    if _find_quote_in_text(primary_text, quote) is not None:
        return trim_source_passage(primary_text, quote, max_chars=max_chars), primary_index
    for idx in range(chunk_count):
        if idx == primary_index:
            continue
        text = fetch_chunk_text(idx)
        if _find_quote_in_text(text, quote) is not None:
            return trim_source_passage(text, quote, max_chars=max_chars), idx
    if abstract_text and _find_quote_in_text(abstract_text, quote) is not None:
        return (
            trim_source_passage(abstract_text, quote, max_chars=max_chars),
            ABSTRACT_CHUNK_INDEX,
        )
    return None


def build_delivered_evidence_rows(
    verifications: list[dict[str, Any]],
    fetch_chunk: Callable[[str, int], dict[str, Any]],
    draft_content: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per entry of *verifications* (``final_report.verifications``), in
    order. ``fetch_chunk(paper_id, chunk_index)`` is injectable so this is testable
    against a fixture with no HTTP call; the production caller
    (``DemoRunner._write_delivered_evidence``) binds it to ``fetch_fulltext_chunk`` and
    a live ``ApiClient``. *draft_content* is the saved draft's own Tiptap document,
    when the caller already fetched one; when given, each row's ``sentence_in_draft``
    records whether that exact sentence occurs verbatim in it.

    Never raises on a data shape a legitimate, correctly-guarded verified row can
    carry: a verified row with no locatable evidence quote is not a data-integrity
    defect -- ``_quote_segments``/``QUOTE_SEGMENT_MIN_CHARS`` deliberately pass a real
    but short (under 30 raw characters) quote with "no evidence either way" rather than
    demote it, and the same floor makes ``evidence_location`` ``None`` for that row.
    Such a row, and one whose ``evidence_location`` is otherwise unparseable, is still
    emitted, with ``source_located`` false, ``source_locator`` and ``source_passage``
    both ``None`` and ``excerpt_locations`` an empty list. A row whose location parses
    and whose chunk fetch succeeds, but one of whose quotes cannot be found in any
    chunk of the paper, is also still emitted, with that quote named in
    ``unlocated_quotes`` instead of a fabricated passage. ``DemoError`` is still raised
    on the one shape that IS a real defect: a row whose own ``status`` is not
    ``"verified"`` (``assert_every_final_row_verified`` already checked this for the
    whole final report before this function is ever called, so reaching this loop with
    a non-verified row means that check itself regressed).

    ``source_located`` records only that the chunk ``evidence_location`` named was
    parsed and fetched -- not that any evidence passage was found in it. The separate
    ``passage_located`` flag is true only when ``evidence_quotes`` is non-empty, every
    one of them was located, and a passage is actually shown in ``source_passage``; it
    is false when nothing was located at all (``source_passage`` is ``None``), when
    only some quotes were (``source_passage`` is non-empty but ``unlocated_quotes``
    still names the rest), and when there was no evidence quote to locate in the first
    place (``source_passage`` then falls back to the start of the chunk, which is not a
    passage any quote was found in) -- a row a human should not be asked to judge as a
    full, source-verified reading either way.

    Every ``section`` label this function reports (``source_locator``,
    ``excerpt_locations``) names the section a chunk STARTS IN, not what the chunk
    actually contains -- a paper's own chunking can put thousands of characters of
    body text into a chunk labelled ``"abstract"`` simply because that chunk begins
    where the abstract does. ``location_section_mismatch`` therefore compares a stored
    label against itself and is always false; it is not a check that the fetched
    chunk's own content matches its label.

    ``source_locator`` names only the row's primary chunk (the one
    ``evidence_location`` itself points at); when a quote is located in a *different*
    chunk of the same paper (the fallback loop inside ``_locate_quote_excerpt``), the
    excerpt it contributes to ``source_passage`` would otherwise be indistinguishable
    from one found in the primary chunk, even though half the passage came from
    elsewhere. ``excerpt_locations`` closes that gap: one entry per excerpt actually
    joined into ``source_passage``, in the same order, each
    ``{"chunk_index", "section"}`` naming exactly where that excerpt was found -- so a
    human judge reading a multi-chunk passage can tell which half came from where,
    instead of trusting the single, row-level locator for all of it.
    """
    rows: list[dict[str, Any]] = []
    draft_paragraphs = _draft_paragraph_texts(draft_content) if draft_content is not None else None
    # Keyed by (paper_id, chunk_index) and shared across every row, not reset per row:
    # a run can have several rows over a handful of papers, and each chunk of each
    # paper only needs fetching once for the whole record.
    chunk_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _cached_chunk(pid: str, idx: int) -> dict[str, Any]:
        key = (pid, idx)
        if key not in chunk_cache:
            chunk_cache[key] = fetch_chunk(pid, idx)
        return chunk_cache[key]

    for i, verification in enumerate(verifications, 1):
        if verification.get("status") != "verified":
            raise DemoError(
                f"delivered_evidence row {i}: final_report verification has status "
                f"{verification.get('status')!r}, not \"verified\""
            )
        sentence = verification.get("claim_sentence") or verification.get("claim_text")
        evidence_quotes = list(verification.get("evidence_quotes") or [])
        if not evidence_quotes and verification.get("evidence_quote"):
            evidence_quotes = [verification["evidence_quote"]]
        row: dict[str, Any] = {
            "row": i,
            "sentence": sentence,
            "citation_text": verification.get("citation"),
            "paper_title": verification.get("paper_title"),
            "paper_doi": verification.get("paper_doi"),
            "claim_text": verification.get("claim_text"),
            "status": verification.get("status"),
            "evidence_quotes": evidence_quotes,
            "sentence_in_draft": (
                any(sentence in p for p in draft_paragraphs)
                if draft_paragraphs is not None and sentence
                else None
            ),
        }
        located = parse_evidence_location(verification.get("evidence_location"))
        if located is None:
            row.update(
                source_located=False,
                source_passage=None,
                source_locator=None,
                excerpt_locations=[],
                passage_located=False,
                unlocated_quotes=list(evidence_quotes),
                location_section_mismatch=False,
                fetch_error=None,
            )
            rows.append(row)
            continue
        chunk_index, location_section = located
        paper_id = str(verification.get("paper_id"))
        # This is the run's last step, after screening, writing and verification have
        # already spent their budget -- a 404, timeout or dropped connection here must
        # mark the row, not raise and lose the whole run.
        try:
            chunk = _cached_chunk(paper_id, chunk_index)
            primary_text = chunk.get("text") or ""
            chunk_count = chunk.get("chunk_count") or (chunk_index + 1)
            # Every chunk of the same paper carries the same `abstract` (the endpoint
            # returns it regardless of which index was requested), so the primary
            # chunk's own response already has it -- no extra fetch needed.
            abstract_text = chunk.get("abstract") or None

            def _fetch_chunk_text(idx: int, _paper_id: str = paper_id) -> str:
                return _cached_chunk(_paper_id, idx).get("text") or ""

            excerpts: list[str] = []
            excerpt_locations: list[dict[str, Any]] = []
            unlocated_quotes: list[str] = []
            for quote in evidence_quotes:
                located_excerpt = _locate_quote_excerpt(
                    quote, primary_text, chunk_index, chunk_count, _fetch_chunk_text,
                    abstract_text=abstract_text,
                )
                if located_excerpt is None:
                    unlocated_quotes.append(quote)
                    continue
                excerpt, found_index = located_excerpt
                if excerpt not in excerpts:
                    excerpts.append(excerpt)
                    # `source_locator` names only the row's primary chunk, so a
                    # passage built from a second quote's own, different chunk (the
                    # branch this loop can take, above) would otherwise be reported
                    # under the wrong chunk/section. One entry per excerpt in
                    # `excerpts`, in the same order, records where that excerpt was
                    # actually found. The abstract sentinel is never a fetchable chunk
                    # index, so it is named directly rather than passed to
                    # `_cached_chunk`.
                    if found_index == ABSTRACT_CHUNK_INDEX:
                        excerpt_locations.append({"chunk_index": None, "section": "abstract"})
                    else:
                        found_chunk = _cached_chunk(paper_id, found_index)
                        excerpt_locations.append(
                            {"chunk_index": found_index, "section": found_chunk.get("section")}
                        )
            if excerpts:
                source_passage: str | None = _EXCERPT_SEPARATOR.join(excerpts)
            elif not evidence_quotes:
                source_passage = trim_source_passage(primary_text, None)
            else:
                source_passage = None
            row.update(
                source_located=True,
                source_passage=source_passage,
                source_locator={"chunk_index": chunk_index, "section": chunk.get("section")},
                excerpt_locations=excerpt_locations,
                # True only when a passage is actually shown AND every one of
                # evidence_quotes was located in it -- not merely when the chunk fetch
                # (source_located) succeeded. Also false when evidence_quotes is empty
                # -- the "elif not evidence_quotes" branch above shows the start of the
                # chunk with no quote behind it at all, which "not unlocated_quotes"
                # alone (vacuously true on an empty list) does not catch.
                passage_located=(
                    bool(source_passage) and bool(evidence_quotes) and not unlocated_quotes
                ),
                unlocated_quotes=unlocated_quotes,
                # `chunk_index` here is the SAME index `evidence_location` itself
                # named, so `chunk.get("section")` is always the fetched chunk's own
                # stored label under that index -- the same value `location_section`
                # (parsed from that same string) already names. This compares a stored
                # label with itself and is always false; it is not a check that the
                # fetched chunk's own content matches its label (a chunk's `section`
                # names where it STARTS, not what it holds).
                location_section_mismatch=(
                    location_section is not None and location_section != chunk.get("section")
                ),
                fetch_error=None,
            )
        except ApiError as exc:
            row.update(
                source_located=False,
                source_passage=None,
                source_locator=None,
                excerpt_locations=[],
                passage_located=False,
                unlocated_quotes=list(evidence_quotes),
                location_section_mismatch=False,
                fetch_error=str(exc),
            )
        rows.append(row)
    return rows


def build_delivered_evidence(
    *,
    run_id: str,
    draft_id: str,
    section_title: str,
    verifications: list[dict[str, Any]],
    fetch_chunk: Callable[[str, int], dict[str, Any]],
    draft_content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``delivered_evidence.json``'s own top-level shape: the run and draft this
    record was built from, the section it belongs to, one row per claim delivered in
    the final, healed draft, and ``unlocated_rows`` (counted against
    ``passage_located``, since a row can carry ``source_located`` true and still show
    no passage at all) -- how many of those rows carry no fully located source
    passage, so a reader of the record itself can see the count without scanning
    every row."""
    rows = build_delivered_evidence_rows(verifications, fetch_chunk, draft_content)
    return {
        "run_id": run_id,
        "draft_id": draft_id,
        "section_title": section_title,
        "rows": rows,
        "unlocated_rows": sum(1 for r in rows if not r.get("passage_located")),
    }


def merge_delivered_evidence_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one ``build_delivered_evidence`` record per section into the ONE
    ``delivered_evidence.json`` a multi-section run writes: every row keeps its own
    position within its own section's own list, but is tagged with the
    ``section_title`` and ``draft_id`` of the record it came from -- a reader cannot
    assume every row belongs to the run's single top-level draft the way a
    one-section run's own record would let them. ``run_id`` is taken from the first
    record (every record given to this function is built from the very same run, so
    it is the same on every one of them); an empty *records* list still returns a
    well-shaped, empty record rather than raising."""
    rows: list[dict[str, Any]] = []
    for record in records:
        for row in record["rows"]:
            rows.append(
                {**row, "section_title": record["section_title"], "draft_id": record["draft_id"]}
            )
    return {
        "run_id": records[0]["run_id"] if records else None,
        "rows": rows,
        "unlocated_rows": sum(record["unlocated_rows"] for record in records),
    }


# ---------------------------------------------------------------------------
# Summary, comparison and printing
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    name: str
    status: str = "pending"
    elapsed_seconds: float = 0.0
    task_id: str | None = None
    detail: str | None = None


@dataclass
class RunState:
    api_url: str
    output_dir: Path
    started_at: datetime
    steps: list[StepRecord] = field(default_factory=list)
    user: dict[str, str] | None = None
    project_id: str | None = None
    draft_id: str | None = None
    screening: dict[str, Any] | None = None
    seeds: dict[str, Any] | None = None
    claims: list[dict[str, Any]] | None = None
    fulltext: dict[str, Any] | None = None
    writing: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    #: The delivered-evidence record built after the standalone verify-and-heal step
    #: (``delivered_evidence.json``'s own top-level shape, not stored inside
    #: ``summary.json``).
    delivered_evidence: dict[str, Any] | None = None
    #: One entry per extra section in ``sections.json``, each carrying that section's
    #: own ``draft_id``, loop statistics, verdict counts and delivered-evidence counts
    #: -- the protocol section's own equivalents stay under
    #: ``writing``/``verification``/``delivered_evidence`` above, unchanged, so an
    #: existing reader of those three keys is unaffected by a run that carries no
    #: extra sections at all.
    sections: list[dict[str, Any]] | None = None
    #: Every Violation the offline delivered-text checker (check_delivered.py) found,
    #: run right after each section is saved and again, comprehensively, as the run's
    #: own last step -- ``None`` when no section was ever written (``--search-only``,
    #: or a step failed before writing one), matching how ``writing`` and
    #: ``verification`` are also left ``None`` then.
    delivered_check: dict[str, Any] | None = None


def _prov(d: dict[str, Any] | None, key: str) -> Any:
    return (d or {}).get(key)


def build_summary(state: RunState, finished_at: datetime) -> dict[str, Any]:
    return {
        "summary_version": SUMMARY_VERSION,
        "api_url": state.api_url,
        "run_started_at": state.started_at.isoformat(timespec="seconds"),
        "run_finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round((finished_at - state.started_at).total_seconds(), 1),
        "project_id": state.project_id,
        "draft_id": state.draft_id,
        "demo_user": (state.user or {}).get("email"),
        "steps": [s.__dict__ for s in state.steps],
        "screening": state.screening,
        "seeds": {k: v for k, v in (state.seeds or {}).items() if k != "paper_data"} or None,
        "fulltext": state.fulltext,
        "writing": state.writing,
        "verification": state.verification,
        # Recorded here so the run's own summary carries whether the delivered-
        # evidence record was written and how many rows it has.
        "delivered_evidence": state.delivered_evidence,
        "sections": state.sections,
        # Every rule-1-through-7 violation the offline delivered-text checker
        # (check_delivered.py) found over this run's own artefacts.
        "delivered_check": state.delivered_check,
    }


def _within_tolerance(expected: float, actual: float, tolerance: float) -> bool:
    """Relative tolerance, but never tighter than COUNT_ABS_SLACK records (expected 0 -> +-2)."""
    return abs(actual - expected) <= max(tolerance * abs(expected), COUNT_ABS_SLACK)


def _rate_within_tolerance(
    expected: float, actual: float, tolerance: float = COUNT_TOLERANCE
) -> bool:
    """Like `_within_tolerance`, for a 0.0-1.0 rate: RATE_ABS_SLACK, not
    COUNT_ABS_SLACK, floors the tolerance band."""
    return abs(actual - expected) <= max(tolerance * abs(expected), RATE_ABS_SLACK)


def _query_lists_from_rounds(rounds: list[dict[str, Any]] | None) -> list[list[str]]:
    """``provenance.rounds`` -> one list of query strings per round, in round order."""
    return [[q.get("query") for q in (r.get("queries") or [])] for r in (rounds or [])]


def _describe_query_lists_diff(expected: list[list[str]], actual: list[list[str]]) -> str:
    """"identical", or a compact description of every round that differs
    (``--replay-queries``): a round present in one side only, or whose query list
    differs."""
    if expected == actual:
        return "identical"
    diffs: list[str] = []
    for i in range(max(len(expected), len(actual))):
        exp_round = expected[i] if i < len(expected) else None
        act_round = actual[i] if i < len(actual) else None
        if exp_round != act_round:
            diffs.append(f"round {i + 1}: expected {exp_round!r}, got {act_round!r}")
    return "; ".join(diffs)


def compare_with_expected(
    summary: dict[str, Any], expected: dict[str, Any], *, replay_queries: bool = False
) -> list[dict[str, Any]]:
    """Tolerances: flow counts +-30 %; citation_coverage.found/linked/by_source.mapping
    +-30 % (or +-2); citation_coverage.unresolved one-sided: never above the baseline,
    and a fall of any size is never drift; claim statuses exact; model names exact;
    writing.total_calls exact: a collapsed citation-link map leaves
    found/linked/unresolved unmoved on this draft (the audit-key fallback re-keys
    everything as author-year), so `by_source.mapping` collapsing to 0 and
    `total_calls` dropping to 1 are what actually catch it.

    ``replay_queries``: when the run was made with ``--replay-queries``, the
    round-by-round query lists themselves are expected to match the baseline exactly
    (the whole point of replaying them), so this adds one more check reporting them
    either "identical" or a per-round description of the differences. Off by default,
    since a normal run's queries are freshly generated and comparing them would only
    ever show drift."""
    checks: list[dict[str, Any]] = []

    def add(name: str, exp: Any, act: Any, ok: bool, kind: str) -> None:
        checks.append({"check": name, "expected": exp, "actual": act, "ok": ok, "kind": kind})

    exp_flow = ((expected.get("screening") or {}).get("flow")) or {}
    act_flow = ((summary.get("screening") or {}).get("flow")) or {}
    if summary.get("screening") is not None:
        for key, exp_value in exp_flow.items():
            if not isinstance(exp_value, (int, float)) or isinstance(exp_value, bool):
                continue
            act_value = act_flow.get(key)
            ok = isinstance(act_value, (int, float)) and _within_tolerance(
                exp_value, act_value, COUNT_TOLERANCE
            )
            add(
                f"screening.flow.{key}",
                exp_value,
                act_value,
                ok,
                f"+-{COUNT_TOLERANCE:.0%} or +-{COUNT_ABS_SLACK}",
            )
        exp_model = _prov(
            _prov((expected.get("screening") or {}).get("provenance"), "screener"), "model_reported"
        )
        act_model = _prov(
            _prov((summary.get("screening") or {}).get("provenance"), "screener"), "model_reported"
        )
        if exp_model is not None:
            add("screening.model_reported", exp_model, act_model, exp_model == act_model, "exact")
        if replay_queries:
            exp_rounds = _prov((expected.get("screening") or {}).get("provenance"), "rounds")
            act_rounds = _prov((summary.get("screening") or {}).get("provenance"), "rounds")
            exp_queries = _query_lists_from_rounds(exp_rounds)
            act_queries = _query_lists_from_rounds(act_rounds)
            add(
                "screening.replay_queries",
                "identical",
                _describe_query_lists_diff(exp_queries, act_queries),
                exp_queries == act_queries,
                "exact (--replay-queries)",
            )

    # Gated the same way the screening block above is: `--search-only` writes a
    # summary with `writing` left `None` (nothing here ever ran), so this block must
    # be skipped rather than comparing `actual: None` against a full baseline and
    # reporting every check under it as drift.
    if summary.get("writing") is not None:
        exp_writing = expected.get("writing") or {}
        act_writing = summary.get("writing") or {}
        if exp_writing.get("model_reported") is not None:
            add(
                "writing.model_reported",
                exp_writing["model_reported"],
                act_writing.get("model_reported"),
                exp_writing["model_reported"] == act_writing.get("model_reported"),
                "exact",
            )
        exp_total_calls = exp_writing.get("total_calls")
        if isinstance(exp_total_calls, (int, float)) and not isinstance(exp_total_calls, bool):
            act_total_calls = act_writing.get("total_calls")
            add(
                "writing.total_calls",
                exp_total_calls,
                act_total_calls,
                act_total_calls == exp_total_calls,
                "exact",
            )
        # The gated write loop's own rates, within the same band as every other
        # drift-prone number this run is compared against.
        exp_loop = exp_writing.get("loop_stats") or {}
        act_loop = act_writing.get("loop_stats") or {}
        for rate_key in ("first_pass_verified_rate", "survival_rate"):
            exp_rate = exp_loop.get(rate_key)
            if isinstance(exp_rate, (int, float)) and not isinstance(exp_rate, bool):
                act_rate = act_loop.get(rate_key)
                ok = isinstance(act_rate, (int, float)) and not isinstance(
                    act_rate, bool
                ) and _rate_within_tolerance(exp_rate, act_rate)
                add(
                    f"writing.loop_stats.{rate_key}",
                    exp_rate,
                    act_rate,
                    ok,
                    f"+-{COUNT_TOLERANCE:.0%} or +-{RATE_ABS_SLACK}",
                )

    # Same gating as the writing block above, for the same reason: `--search-only`
    # leaves `verification` `None` too.
    if summary.get("verification") is not None:
        exp_ver = expected.get("verification") or {}
        act_ver = summary.get("verification") or {}
        # The exit invariant `DemoRunner._verify_claims` asserts, over the saved draft
        # and the final report -- every row of `final_report.verifications` is
        # verified, no [NEEDS CITATION] marker survives in the saved draft, and no
        # citation occurrence is left with no verified row behind it
        # (`citation_coverage.unresolved == 0`) -- raises DemoError before this point
        # on any violation, so recording it here is an exact self-consistency check on
        # this run, not a comparison with *expected* (the same pattern
        # `verification.claim_text_in_draft`, below, already uses).
        # Only for a run whose summary carries the "exit_invariant" key at all -- an
        # older summary (before this check existed), or one from a run that never
        # reached the standalone healing action, never claims to have checked it, so
        # there is nothing here to compare.
        if act_ver.get("exit_invariant") is not None:
            inv = act_ver["exit_invariant"]
            add(
                "verification.exit_invariant",
                "every final_report row verified, no [NEEDS CITATION] marker in the "
                "saved draft, citation_coverage.unresolved == 0",
                f"final_report_rows_checked={inv.get('final_report_rows_checked')} "
                f"citation_coverage_unresolved={inv.get('citation_coverage_unresolved')}",
                True,
                "exact (enforced by the runner; a violation already raised DemoError "
                "before summary.json was written)",
            )
        exp_prov_model = _prov(exp_ver.get("provenance"), "model_reported")
        if exp_prov_model is not None:
            act_prov_model = _prov(act_ver.get("provenance"), "model_reported")
            add(
                "verification.model_reported",
                exp_prov_model,
                act_prov_model,
                exp_prov_model == act_prov_model,
                "exact",
            )
        act_claims = {c["id"]: c for c in act_ver.get("claims") or []}
        for exp_claim in exp_ver.get("claims") or []:
            act_status = (act_claims.get(exp_claim["id"]) or {}).get("status")
            add(
                f"verification.claims.{exp_claim['id']}.status",
                exp_claim.get("status"),
                act_status,
                exp_claim.get("status") == act_status,
                "exact",
            )

        # A self-consistency check on the run itself, not a comparison with *expected*
        # -- every verification's own claim_text must occur verbatim in the draft
        # this run saved, regardless of what an older baseline recorded.
        act_claim_text_in_draft = _parse_n_of_m(act_ver.get("claim_text_in_draft"))
        if act_claim_text_in_draft is not None:
            n, m = act_claim_text_in_draft
            add(
                "verification.claim_text_in_draft",
                f"{m} of {m}",
                act_ver.get("claim_text_in_draft"),
                n == m,
                "every claim_text occurs verbatim in the saved draft",
            )

        # Catches a degraded citation-link map (a provider outage or a linker
        # regression) that a fresh run's coverage numbers would otherwise hide from
        # every other check above. `found` and `linked` drift like any other count;
        # `unresolved` is one-sided: only a rise above the baseline is ever drift, at
        # any size; a fall, of any size, including all the way to zero, is an
        # improvement over the baseline and is never reported as drift, even when it
        # lands outside the relative tolerance band `found`/`linked` use.
        exp_cov = exp_ver.get("citation_coverage") or {}
        act_cov = act_ver.get("citation_coverage") or {}
        if exp_cov:
            for key in ("found", "linked"):
                exp_value = exp_cov.get(key)
                if not isinstance(exp_value, (int, float)) or isinstance(exp_value, bool):
                    continue
                act_value = act_cov.get(key)
                ok = isinstance(act_value, (int, float)) and not isinstance(
                    act_value, bool
                ) and _within_tolerance(exp_value, act_value, COUNT_TOLERANCE)
                add(
                    f"verification.citation_coverage.{key}",
                    exp_value,
                    act_value,
                    ok,
                    f"+-{COUNT_TOLERANCE:.0%} or +-{COUNT_ABS_SLACK}",
                )
            # `found`/`linked` alone cannot see a collapsed citation-link map on this
            # draft, since the audit-key fallback re-keys every occurrence the map
            # drops as `author-year` instead of `unresolved`. `by_source.mapping` is
            # the count that actually moves.
            exp_mapping = _prov(exp_cov.get("by_source"), "mapping")
            if isinstance(exp_mapping, (int, float)) and not isinstance(exp_mapping, bool):
                act_mapping = _prov(act_cov.get("by_source"), "mapping")
                ok = isinstance(act_mapping, (int, float)) and not isinstance(
                    act_mapping, bool
                ) and _within_tolerance(exp_mapping, act_mapping, COUNT_TOLERANCE)
                add(
                    "verification.citation_coverage.by_source.mapping",
                    exp_mapping,
                    act_mapping,
                    ok,
                    f"+-{COUNT_TOLERANCE:.0%} or +-{COUNT_ABS_SLACK}",
                )
            exp_unresolved = exp_cov.get("unresolved")
            if isinstance(exp_unresolved, (int, float)) and not isinstance(exp_unresolved, bool):
                act_unresolved = act_cov.get("unresolved")
                valid_unresolved = isinstance(act_unresolved, (int, float)) and not isinstance(
                    act_unresolved, bool
                )
                ok = valid_unresolved and act_unresolved <= exp_unresolved
                add(
                    "verification.citation_coverage.unresolved",
                    exp_unresolved,
                    act_unresolved,
                    ok,
                    "never above baseline; a fall, of any size, is never drift",
                )
    return checks


def claim_problems(summary: dict[str, Any]) -> list[str]:
    """Reasons the run did not demonstrate its claims (independent of demo/expected/).

    Two shapes are not model drift and never raise exit code 2, even though the raw
    status differs from ``expected_status``. First, a guard demotion
    (`is_guard_demotion`, already folded into ``matches_expected`` by
    `match_claim_results`): a claim expecting ``verified`` that the frozen
    ``quote_not_verbatim`` guard alone demoted to ``needs_nuance`` while the
    verification model itself said ``verified``. Second, a claim expecting
    ``unsupported`` whose raw ``status`` was itself something other than ``verified``
    and that is absent from the finalized (healed) report (``final_status``
    ``"not_in_report"``): verify-and-heal removing an unsupported claim is the intended
    outcome, not a failure to demonstrate it. A claim expecting ``verified`` is never
    let through as ``unsupported`` by either allowance: the first only ever widens
    ``needs_nuance``, and the second only ever applies when ``expected_status`` is
    ``unsupported``.

    The second allowance requires the raw ``status`` to be something other than
    ``verified``, not ``final_status`` alone: a fixture expecting ``unsupported`` that
    the raw verification pass itself called ``verified`` and that nonetheless ended up
    absent from the healed report (for instance because a co-cited citation on the
    same sentence was not verified, the case `surviving_verifications` exists to
    handle) must not pass silently, exiting 0 without ever demonstrating the
    unsupported case at all. This does not narrow the allowance for the ordinary case
    (a raw ``unsupported``/``needs_nuance``/``error`` status healed away entirely) at
    all.

    The guard-demotion allowance above (folded into ``matches_expected`` by
    `match_claim_results`/`is_guard_demotion`) could otherwise let a run in which the
    ``verified`` fixture is demoted AND THEN healed out of the final report exit 0
    with neither fixture actually demonstrated in ``final_report`` -- the demotion
    alone is not model drift, but the fixture then vanishing from the delivered report
    on top of it is a real failure to demonstrate the claim this run exists to
    demonstrate, and nothing above catches it (`matches_expected` is already `True`
    from the demotion allowance, so the loop never reaches this claim's own
    `final_status`). Checked once, after the per-claim loop, over every fixture whose
    ``expected_status`` is ``"verified"``: if none of them has
    ``final_status == "verified"`` AND none of them is itself a recorded guard
    demotion, this run demonstrates no verified fixture at all, and that is reported
    here rather than left for a reader to notice only by opening
    ``claim_report.json``.

    A recorded guard demotion (``guard_demotion``, already computed by
    `match_claim_results`/`is_guard_demotion`) is also accepted here, matching the
    same allowance `check_delivered.check_fixture_claims`'s own rule 7 grants
    (`_is_recorded_guard_demotion`): the frozen `quote_not_verbatim` guard demotes a
    `verified`-model-status claim to `needs_nuance`, and finalize then removes ANY
    non-`verified` cited sentence from the healed draft by construction
    (`_finalize_paragraph_text` rule 2), so a genuinely guard-demoted fixture can
    NEVER reach ``final_status == "verified"``. This keeps this gate, which is the one
    that actually sets the process exit code (`EXIT_CLAIMS`), in agreement with the
    allowance the first `matches_expected` check already grants, rather than silently
    re-imposing the stricter standard one line later."""
    verification = summary.get("verification") or {}
    problems: list[str] = []
    if not verification.get("claims_in_report"):
        problems.append("the claim-verification report contains no claims at all")
    claims = verification.get("claims") or []
    for claim in claims:
        if claim.get("matches_expected"):
            continue
        if (
            claim.get("expected_status") == "unsupported"
            and claim.get("status") != "verified"
            and claim.get("final_status") == "not_in_report"
        ):
            continue
        problems.append(
            f"claim {claim['id']}: status {claim.get('status')!r}, "
            f"expected {claim.get('expected_status')!r}"
        )
    verified_fixtures = [c for c in claims if c.get("expected_status") == "verified"]
    if verified_fixtures and not any(
        c.get("final_status") == "verified" or c.get("guard_demotion")
        for c in verified_fixtures
    ):
        problems.append(
            "no fixture expecting \"verified\" survived into the final report, and "
            "none is a recorded guard demotion "
            f"(final_status of {[c['id'] for c in verified_fixtures]}: "
            f"{[c.get('final_status') for c in verified_fixtures]})"
        )
    return problems


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "-"
    return str(value)


def format_summary_table(summary: dict[str, Any], checks: list[dict[str, Any]] | None) -> str:
    rows: list[tuple[str, str]] = []
    screening = summary.get("screening")
    if screening:
        flow = screening.get("flow") or {}
        for key in (
            "identified",
            "duplicates_removed",
            "stage1_screened",
            "stage1_excluded",
            "stage2_screened",
            "stage2_excluded",
            "unscreened",
            "included",
            "rounds",
            "stop_reason",
        ):
            rows.append((f"screening.flow.{key}", _fmt(flow.get(key))))
        screener = _prov(screening.get("provenance"), "screener") or {}
        for key in (
            "model_configured",
            "model_reported",
            "temperature",
            "prompt_version",
            "calls",
            "input_tokens",
            "output_tokens",
        ):
            rows.append((f"screening.screener.{key}", _fmt(screener.get(key))))
        rows.append(("screening.records_exported", _fmt(screening.get("records_exported"))))
    else:
        rows.append(("screening", "skipped (--skip-search)"))
    seeds = summary.get("seeds") or {}
    rows.append(("seeds.requested", _fmt(seeds.get("requested"))))
    rows.append(("seeds.added", _fmt(len(seeds.get("added") or []))))
    rows.append(("seeds.already_present", _fmt(len(seeds.get("already_present") or []))))
    fulltext = summary.get("fulltext") or {}
    for key in ("acquired", "abstract_only", "already_acquired"):
        rows.append((f"fulltext.{key}", _fmt(fulltext.get(key))))
    writing = summary.get("writing") or {}
    for key in (
        "model_configured",
        "model_reported",
        "temperature",
        "prompt_version",
        "input_tokens",
        "output_tokens",
        "papers_used",
    ):
        rows.append((f"writing.{key}", _fmt(writing.get(key))))
    # The generation call above plus the citation-link call, folded together: a
    # section's real DeepSeek total, not just the generation call's own tokens.
    for key in ("total_calls", "total_input_tokens", "total_output_tokens"):
        rows.append((f"writing.{key}", _fmt(writing.get(key))))
    link_call = writing.get("citation_link_call") or {}
    rows.append(
        ("writing.citation_link_call.model_reported", _fmt(link_call.get("model_reported")))
    )
    rows.append(("writing.citation_link_call.input_tokens", _fmt(link_call.get("input_tokens"))))
    rows.append(("writing.citation_link_call.output_tokens", _fmt(link_call.get("output_tokens"))))
    # A revision pass makes a second citation-link call, so ``citation_link_call``
    # above is only ever the most recent one; the full list (one entry per call, in
    # order, possibly with a ``None`` for a failed call) is here so a reader can tell
    # how many were actually made and paid for.
    link_calls = writing.get("citation_link_calls") or []
    rows.append(("writing.citation_link_calls.count", _fmt(len(link_calls))))
    # Absent (every value "-") when the request carried no target_words. ``words`` is
    # the pre-revision, pre-finalize body length (recorded right after generation);
    # ``final_words`` is the length of the text actually saved, after a revision (if
    # any) and finalize's own sentence removals -- the one to quote as "the finished
    # length".
    length = writing.get("length") or {}
    for key in (
        "target_words", "hard_maximum", "words", "final_words", "regenerated", "over_target",
    ):
        rows.append((f"writing.length.{key}", _fmt(length.get(key))))
    audit = writing.get("citation_audit") or {}
    rows.append(("writing.citation_audit.total", _fmt(audit.get("total"))))
    rows.append(("writing.citation_audit.matched", _fmt(len(audit.get("matched") or []))))
    rows.append(("writing.citation_audit.unmatched", _fmt(audit.get("unmatched"))))
    rows.append(
        ("writing.citation_audit.needs_citation_flags", _fmt(audit.get("needs_citation_flags")))
    )
    # The gated write loop's own internal statistics. ``length_regenerated`` (the
    # pre-generation body was over the hard maximum and was regenerated once) and
    # ``loop_revised`` (the verification gate found a problem and the whole section
    # was rewritten once) are two independent events, tracked under separate names
    # rather than one ambiguous ``regenerated`` flag. ``survival_rate`` is
    # ``final_words / target_words``: over 1.0 means the finished section ran long,
    # not that "more of it survived".
    loop_stats = writing.get("loop_stats") or {}
    for key in (
        "length_regenerated",
        "loop_revised",
        "first_pass_verified_rate",
        "survival_rate",
        "sentences_removed_unverified",
        "sentences_removed_uncited_finding",
        "sentences_removed_no_full_text",
        "citations_dropped_no_full_text",
        "needs_citation_markers_removed",
        # the deterministic coherence pass's own removal count, and how many
        # sentences the write loop kept with no classification at all -- both
        # already on this dict, neither previously printed.
        "sentences_removed_dangling",
        "sentences_unclassified_kept",
    ):
        rows.append((f"writing.loop_stats.{key}", _fmt(loop_stats.get(key))))
    # ``papers`` alone cannot say whether every selected paper's full-text grounding
    # call succeeded; ``papers_attempted``/``papers_failed`` make a silently reduced
    # corpus (a provider outage or token-limit failure on one paper's grounding call)
    # visible instead of indistinguishable from "nothing to analyse".
    analysis_metrics = (writing.get("metrics") or {}).get("analysis") or {}
    for key in (
        "papers",
        "papers_attempted",
        "papers_failed",
        "evidence_items",
        "rejected_items",
        "verbatim_rate",
    ):
        rows.append((f"writing.metrics.analysis.{key}", _fmt(analysis_metrics.get(key))))
    claim_report = writing.get("claim_report") or {}
    rows.append(("writing.claim_report.verified_count", _fmt(claim_report.get("verified_count"))))
    ver = summary.get("verification") or {}
    for key in (
        "verified_count",
        "unsupported_count",
        "nuance_count",
        "abstract_only_count",
        "error_count",
        "full_text_coverage",
    ):
        rows.append((f"verification.{key}", _fmt(ver.get(key))))
    rows.append(("verification.healed", _fmt(ver.get("healed"))))
    rows.append(
        ("verification.final_report_verified_count", _fmt(ver.get("final_report_verified_count")))
    )
    finalize_stats = ver.get("finalize_stats") or {}
    for key in (
        "sentences_removed_unverified",
        "sentences_removed_uncited_finding",
        "sentences_removed_no_full_text",
        "citations_dropped_no_full_text",
        "needs_citation_markers_removed",
        # the same two additions as `loop_stats` above, this standalone
        # verify-and-heal action's own copy.
        "sentences_removed_dangling",
        "sentences_unclassified_kept",
    ):
        rows.append((f"verification.finalize_stats.{key}", _fmt(finalize_stats.get(key))))
    prov = ver.get("provenance") or {}
    for key in (
        "model_configured",
        "model_reported",
        "temperature",
        "prompt_version",
        # Both keys already reach summary.json (the whole provenance dict is copied
        # there unchanged) but not this human-readable report; included here so a
        # reader can tell which guard set or which relocation rule produced its claim
        # statuses.
        "guard_digest",
        "quote_relocation_version",
        # Which version of the shared call-relocate-guard[-call-again] policy
        # produced this run's claim statuses.
        "verification_policy_version",
        "repair_prompt_version",
        "calls",
        "input_tokens",
        "output_tokens",
    ):
        rows.append((f"verification.provenance.{key}", _fmt(prov.get(key))))
    for claim in ver.get("claims") or []:
        # A guard demotion is printed as its own line, distinct from both a plain pass
        # and a real mismatch, so a reader of this table can tell the guard fired
        # correctly apart from model drift without opening claim_report.json. The
        # fixture's own `final_status` (whether it survived healing into the final,
        # delivered report, or is "not_in_report") is printed on the same line, so a
        # run that demoted the fixture AND then healed it away cannot read as an
        # undifferentiated "GUARD DEMOTION" pass -- a reader sees at a glance whether
        # the demoted claim actually made it into the report the demo delivers.
        if claim.get("guard_demotion"):
            flag = f"GUARD DEMOTION (final_status={_fmt(claim.get('final_status'))})"
        elif claim.get("matches_expected"):
            flag = "OK"
        else:
            flag = "MISMATCH"
        rows.append(
            (
                f"claim.{claim['id']}.status",
                f"{_fmt(claim.get('status'))} (expected {claim['expected_status']}) {flag}",
            )
        )
        quote = claim.get("evidence_quote") or "-"
        rows.append(
            (f"claim.{claim['id']}.quote", quote[:160] + ("..." if len(quote) > 160 else ""))
        )
        # How many verification-model calls this claim actually made (the bounded
        # second pass can fire once more).
        rows.append((f"claim.{claim['id']}.passes", _fmt(claim.get("pass_count"))))
    rows.append(("elapsed_seconds", _fmt(summary.get("elapsed_seconds"))))
    width = max(len(k) for k, _ in rows)
    lines = ["", "=" * (width + 40), "ScholarRAG demo summary", "=" * (width + 40)]
    lines.extend(f"{k.ljust(width)}  {v}" for k, v in rows)
    if checks is None:
        lines.append("")
        lines.append("Comparison with demo/expected/: no expected/summary.json found - skipped.")
    else:
        lines.append("")
        lines.append("Comparison with demo/expected/summary.json:")
        for c in checks:
            mark = "ok  " if c["ok"] else "DRIFT"
            lines.append(
                f"  {mark} {c['check']}: expected {_fmt(c['expected'])}, "
                f"got {_fmt(c['actual'])} [{c['kind']}]"
            )
        failed = sum(1 for c in checks if not c["ok"])
        lines.append(f"  {len(checks) - failed}/{len(checks)} checks within tolerance")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise DemoError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DemoError(f"{what} is not valid JSON ({path}): {exc}") from exc


def _load_json_array(path: Path, what: str) -> list[dict[str, Any]]:
    """Like `_load_json`, for a file whose own top level is a JSON array rather than
    an object (``sections.json``)."""
    if not path.exists():
        raise DemoError(f"{what} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DemoError(f"{what} is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, list):
        raise DemoError(f"{what} must be a JSON array of section objects ({path})")
    return data


def _validate_sections(sections: list[dict[str, Any]]) -> None:
    """Each entry of ``sections.json`` needs the same two fields ``protocol.json``'s
    own ``section`` needs (`create_draft`, `start_generation`); ``target_words`` is
    optional there too, so it is not required here either."""
    for index, section in enumerate(sections):
        for key in ("title", "instructions"):
            if not section.get(key):
                raise DemoError(f"sections.json entry {index} is missing {key!r}")


def _validate_inputs(protocol: dict[str, Any], seeds: dict[str, Any]) -> None:
    for key in (
        "topic",
        "research_question",
        "inclusion_criteria",
        "exclusion_criteria",
        "section",
        "claims",
    ):
        if key not in protocol:
            raise DemoError(f"protocol.json is missing '{key}'")
    if len(seeds.get("papers") or []) != 12:
        raise DemoError(
            f"seed_dois.json must list exactly 12 papers, found {len(seeds.get('papers') or [])}"
        )
    seed_dois = {p["doi"] for p in seeds["papers"]}
    for claim in protocol["claims"]:
        if claim["source_doi"] not in seed_dois:
            raise DemoError(
                f"claim {claim['id']} cites {claim['source_doi']}, which is not a seed DOI"
            )
        text = claim.get("text") or ""
        if ANY_CITATION_RE.search(text) and not CITATION_RE.search(text):
            raise DemoError(
                f"claim {claim['id']} carries a citation the backend cannot parse "
                f"({ANY_CITATION_RE.search(text).group(0)!r}); use '(Surname, YEAR)' or "
                "'(Surname et al., YEAR)' with a single ASCII capitalised surname"
            )


class DemoRunner:
    """Runs the steps in order and records timings; all I/O is injectable for tests."""

    def __init__(
        self,
        client: ApiClient,
        *,
        protocol: dict[str, Any],
        seeds: dict[str, Any],
        output_dir: Path,
        expected_dir: Path | None,
        timeout_s: float,
        search_timeout_s: float | None = None,
        skip_search: bool = False,
        search_only: bool = False,
        replay_queries: bool = False,
        sections: list[dict[str, Any]] | None = None,
        resolver: Callable[[str], dict[str, Any] | None] | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        log: Callable[[str], None] = print,
        suffix: str | None = None,
        password: str | None = None,
    ):
        self.client = client
        self.protocol = protocol
        self.seeds = seeds
        self.output_dir = output_dir
        self.expected_dir = expected_dir
        self.timeout_s = timeout_s
        # Smart Search waits under its own, larger budget (DEFAULT_SEARCH_TIMEOUT_S);
        # every other step keeps using timeout_s, unchanged.
        self.search_timeout_s = (
            search_timeout_s if search_timeout_s is not None else DEFAULT_SEARCH_TIMEOUT_S
        )
        self.skip_search = skip_search
        self.search_only = search_only
        self.replay_queries = replay_queries
        # The extra sections to write and verify after the protocol section, in
        # order; empty when the caller passes none (the default -- see
        # `build_parser`'s own ``--sections`` default).
        self.sections = list(sections or [])
        self.resolver = resolver or openalex_resolver()
        self.sleep = sleep or time.sleep
        self.clock = clock or time.monotonic
        self.now = now
        self.log = log
        self.suffix = suffix or secrets.token_hex(3)
        self.password = password or ("Demo-" + secrets.token_urlsafe(12))
        # Set by ``run()`` before the first HTTP call when ``replay_queries`` is True;
        # read by ``_smart_search`` rather than re-reading and re-validating the
        # baseline file a second time.
        self._replay_query_lists: list[list[str]] | None = None
        # Set by ``_verify_claims``; read by ``_write_delivered_evidence``. Kept off
        # ``self.state``/``summary.json`` -- see the comment where each is set.
        self._draft_content: dict[str, Any] | None = None
        self._final_report_verifications: list[dict[str, Any]] = []
        # One `build_delivered_evidence` record per extra section, collected as each
        # one's own verify-and-heal completes; merged with the protocol section's own
        # record by `_write_delivered_evidence` into the ONE `delivered_evidence.json`
        # this run writes.
        self._extra_section_delivered_records: list[dict[str, Any]] = []
        # Every Violation the offline delivered-text checker (check_delivered.py)
        # found so far, collected as each section's own files are saved and again,
        # comprehensively, at the end of the run (`_check_delivered_text`); folded
        # into `self.state.delivered_check` by `run()`, never raised from inside a
        # step -- see `_check_delivered_text_for_section`'s own docstring for why.
        self._delivered_check_violations: list[check_delivered.Violation] = []
        self.state = RunState(
            api_url=client.base_url.rstrip("/"), output_dir=output_dir, started_at=self.now()
        )

    # -- helpers -------------------------------------------------------------

    def _step(self, name: str, fn: Callable[[], Any]) -> Any:
        record = StepRecord(name=name, status="running")
        self.state.steps.append(record)
        self.log(f"\n== {name}")
        started = self.clock()
        try:
            result = fn()
        except Exception as exc:
            record.status = "failed"
            record.elapsed_seconds = round(self.clock() - started, 1)
            record.detail = str(exc)
            raise
        record.status = "completed"
        record.elapsed_seconds = round(self.clock() - started, 1)
        return result

    def _poll(
        self,
        task_id: str,
        label: str,
        *,
        timeout_s: float | None = None,
        timeout_flag: str = "--timeout",
    ) -> dict[str, Any]:
        return poll_task(
            self.client,
            task_id,
            label=label,
            timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
            sleep=self.sleep,
            clock=self.clock,
            log=self.log,
            timeout_flag=timeout_flag,
        )

    def _check_delivered_text_for_section(
        self, *, index: int | None, protocol_claims: list[dict[str, Any]] | None = None
    ) -> None:
        """Runs ``check_delivered.py``'s own rules 1-4, 6-9 against one
        section's own three files, immediately after they are saved to disk -- the
        same check a reviewer would get running ``python demo/check_delivered.py``
        against this run directory by hand, but caught here, inside the run itself,
        before it ever reports success. Rule 5 (``delivered_evidence.json``) is a
        whole-run artefact and is checked once, comprehensively, by
        `_check_delivered_text` after the run's last content-producing step.

        Every `check_delivered.Violation` found is appended to
        ``self._delivered_check_violations``, never raised here: a genuinely non-
        deterministic model output is not a step failure (``DemoError``, exit code 1)
        the way an unreachable API or a malformed response is, and raising here would
        turn every ``FakeServer``-driven test in ``test_run_demo.py`` that calls
        ``DemoRunner.run()`` directly and only inspects its return value into an
        unexpected exception. Folded into ``self.state.delivered_check`` by ``run()``
        instead, exactly the way `claim_problems` already reads ``summary`` after
        ``run()`` returns to decide the exit code -- see ``main()``."""
        files = check_delivered.load_section_files(self.output_dir, index=index)
        violations, _notices = check_delivered.check_section(files, protocol_claims=protocol_claims)
        self._delivered_check_violations.extend(violations)

    # -- steps ---------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        _validate_inputs(self.protocol, self.seeds)
        # Read and validate the replay baseline before the first HTTP call, not
        # inside the third step (`_smart_search`) -- a baseline with no
        # `provenance.rounds` must fail before a demo user is registered and a
        # project created on the live stack, not after.
        self._replay_query_lists = (
            self._load_replay_queries() if self.replay_queries else None
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"ScholarRAG demo -> {self.state.api_url}; outputs in {self.output_dir}")

        self.state.user = self._step(
            "register demo user", lambda: register_or_login(self.client, self.suffix, self.password)
        )
        _write_json(
            self.output_dir / "credentials.json", {**self.state.user, "password": self.password}
        )
        self.log(
            f"  {self.state.user['mode']}: {self.state.user['email']} "
            "(password in credentials.json)"
        )

        self.state.project_id = self._step(
            "create project", lambda: create_project(self.client, self.protocol)
        )
        self.log(f"  project {self.state.project_id}")

        if self.skip_search:
            self.state.steps.append(StepRecord(name="smart search", status="skipped"))
        else:
            self._step("smart search", self._smart_search)

        if self.search_only:
            # Stop right after the screening record export; the summary this writes
            # carries only the screening block (seeds/fulltext/writing/verification
            # all stay None, since those steps never ran).
            finished = self.now()
            summary = build_summary(self.state, finished)
            checks = self._compare(summary)
            summary["comparison"] = checks
            _write_json(self.output_dir / "summary.json", summary)
            self.log(format_summary_table(summary, checks))
            self.log(f"\nOutputs written to {self.output_dir}")
            return summary

        self.state.seeds = self._step(
            "add seed papers",
            lambda: add_seed_papers(
                self.client, self.state.project_id, self.seeds, self.resolver, self.log
            ),
        )
        self.state.claims = self._step("check claim citations", self._check_claims)
        self.state.fulltext = self._step("acquire full texts", self._acquire_full_texts)
        self._step("AI write section", self._write_section)
        self._step("append claim fixtures", self._append_fixture_claims)
        self._step("claim verification", self._verify_claims)
        # The extra sections run only after the protocol section's own standalone
        # verify-and-heal above -- on the same project and the same 12 seed papers,
        # through the identical AI Write job and standalone verify-and-heal, but with
        # no fixture claims appended (`self.sections` is empty by default; see
        # `build_parser`'s own ``--sections``).
        if self.sections:
            self._step("extra sections", self._write_extra_sections)
        self._step("delivered evidence", self._write_delivered_evidence)
        self._step("check delivered text", self._check_delivered_text)
        self.state.delivered_check = {
            "violations": [
                {
                    "rule": v.rule,
                    "section": v.section,
                    "detail": v.detail,
                    "sentence": v.sentence,
                }
                for v in self._delivered_check_violations
            ],
            "ok": not self._delivered_check_violations,
        }

        finished = self.now()
        summary = build_summary(self.state, finished)
        checks = self._compare(summary)
        summary["comparison"] = checks
        _write_json(self.output_dir / "summary.json", summary)
        self.log(format_summary_table(summary, checks))
        self.log(f"\nOutputs written to {self.output_dir}")
        return summary

    def _load_replay_queries(self) -> list[list[str]]:
        """One query list per round, read from the expected baseline's own
        ``screening_record.json`` (``--replay-queries``): the round-by-round log a
        prior run's Smart Search job recorded, reused here as this run's
        ``queries_override`` so it asks the same questions in the same order instead
        of the query generator agent inventing new ones."""
        if not self.expected_dir:
            raise DemoError(
                "--replay-queries needs --expected to point at a directory holding "
                "screening_record.json (the run to replay)"
            )
        path = self.expected_dir / "screening_record.json"
        record = _load_json(path, "expected screening_record.json (for --replay-queries)")
        rounds = ((record.get("provenance") or {}).get("rounds")) or []
        if not rounds:
            raise DemoError(f"{path} has no provenance.rounds to replay")
        return [[q["query"] for q in (r.get("queries") or [])] for r in rounds]

    def _smart_search(self) -> None:
        queries_override = self._replay_query_lists if self.replay_queries else None
        task_id = start_smart_search(
            self.client, self.state.project_id, self.protocol, queries_override=queries_override,
        )
        self.state.steps[-1].task_id = task_id
        task = self._poll(
            task_id, "smart search",
            timeout_s=self.search_timeout_s, timeout_flag="--search-timeout",
        )
        record, csv_text = fetch_screening_record(self.client, task_id)
        _write_json(self.output_dir / "screening_record.json", record)
        (self.output_dir / "screening_record.csv").write_text(csv_text, encoding="utf-8")
        result = task.get("result") or {}
        self.state.screening = {
            "task_id": task_id,
            "flow": record.get("flow") or {},
            "criteria": record.get("criteria") or {},
            "provenance": record.get("provenance") or {},
            "records_exported": len(record.get("records") or []),
            "elapsed_minutes": result.get("elapsed_minutes"),
            "stage1_applied": result.get("stage1_applied"),
        }

    def _check_claims(self) -> list[dict[str, Any]]:
        """Fail before any LLM step when a claim could not reach the verifier."""
        claims = prepare_claims(self.protocol, self.state.seeds["paper_data"])
        check_claim_citations(claims)
        for claim in claims:
            self.log(f"  {claim['id']}: {claim['citation_key']} -> {claim['source_doi']}")
        return claims

    def _acquire_full_texts(self) -> dict[str, Any]:
        """A seed that comes back ``abstract_only`` after this call gets exactly
        one more chance -- Unpaywall or the publisher host can be transiently
        unreachable rather than genuinely lacking an OA copy (the promoted
        2026-09-12 baseline hit exactly this on one host, see the acquisition
        note in ``demo/README.md``). When the first attempt's own result carries
        ``abstract_only > 0`` this waits 60 s (`self.sleep`, so a test can inject
        a fake clock) and calls the same endpoint again with no ``paper_ids``
        filter; the backend's own idempotency check
        (``fulltext_status == "acquired"``) means the second call only ever
        re-attempts the papers still abstract-only after the first, never the ones
        already acquired. Both attempts are kept, in order, under ``attempts``, so
        a reader can see exactly what each call returned; the top-level
        ``acquired``/``abstract_only`` keys are always the state AFTER the retry
        (unchanged, and with no second call made, when the first attempt already
        acquired every seed), so every other reader of this dict (the
        troubleshooting rows, `format_summary_table`, a drift comparison) keeps
        seeing the library's real, current state rather than a stale pre-retry
        count. ``already_acquired`` and ``reference_chunks_dropped``, by contrast,
        are left at the first attempt's own values on a retried run: they
        describe the first attempt only, not the whole run."""
        task_id = start_fulltext_acquisition(self.client, self.state.project_id)
        self.state.steps[-1].task_id = task_id
        task = self._poll(task_id, "full texts")
        first = dict(task.get("result") or {})
        first["task_id"] = task_id
        result = dict(first)
        attempts = [first]
        if first.get("abstract_only", 0) > 0:
            self.log(
                f"  [full texts] {first['abstract_only']} paper(s) still abstract-only "
                "after the first attempt; waiting 60s and requesting acquisition once more"
            )
            self.sleep(60)
            retry_task_id = start_fulltext_acquisition(self.client, self.state.project_id)
            self.state.steps[-1].task_id = retry_task_id
            retry_task = self._poll(retry_task_id, "full texts (retry)")
            retry = dict(retry_task.get("result") or {})
            retry["task_id"] = retry_task_id
            attempts.append(retry)
            result["acquired"] = first.get("acquired", 0) + retry.get("acquired", 0)
            result["abstract_only"] = retry.get("abstract_only", 0)
        result["retry_attempted"] = len(attempts) > 1
        result["attempts"] = attempts
        return result

    def _write_section(self) -> None:
        """The write job's own gated loop (compose -> link -> verify -> gate -> revise
        -> finalize) verifies, finalizes and saves the AI-written section itself, so
        this step reads the final text and the one report
        (`loop_stats`/`claim_report`/`metrics`) straight from the job result, and
        reads the saved draft back rather than building or PUTting it itself."""
        self.state.draft_id = create_draft(self.client, self.state.project_id, self.protocol)
        task_id = start_generation(self.client, self.state.draft_id, self.protocol)
        self.state.steps[-1].task_id = task_id
        task = self._poll(task_id, "AI write")
        result = task.get("result") or {}
        generated = result.get("content") or ""
        if not generated.strip():
            raise DemoError("AI Write returned empty content")
        _write_json(self.output_dir / "writing_result.json", result)
        draft = fetch_draft(self.client, self.state.draft_id)
        _write_json(self.output_dir / "draft_content.json", draft.get("content"))
        prov = result.get("provenance") or {}
        self.state.writing = {
            "task_id": task_id,
            "draft_id": self.state.draft_id,
            "section_type": result.get("section_type"),
            "papers_used": result.get("papers_used"),
            "characters": len(generated),
            "model_configured": prov.get("model_configured"),
            "model_reported": prov.get("model_reported"),
            "temperature": prov.get("temperature"),
            "prompt_version": prov.get("prompt_version"),
            "input_tokens": prov.get("input_tokens"),
            "output_tokens": prov.get("output_tokens"),
            # The citation-link call is a second, real DeepSeek request the writer's
            # own section generation makes: "input_tokens"/"output_tokens" above cover
            # the generation call alone, so an existing reader of those two keys keeps
            # seeing what it always saw; these four are additive.
            # ``citation_link_call`` is only ever the most recent call's own record (a
            # revision pass makes a second one); ``citation_link_calls`` is the full
            # list, one entry per call in order, so neither call's provenance is lost.
            "citation_link_call": prov.get("citation_link_call"),
            "citation_link_calls": prov.get("citation_link_calls"),
            "total_calls": prov.get("total_calls"),
            "total_input_tokens": prov.get("total_input_tokens"),
            "total_output_tokens": prov.get("total_output_tokens"),
            # Present only when the request carried a target_words (the protocol's
            # own section always does); {target_words, hard_maximum, words,
            # regenerated, over_target} as the writing job itself records it.
            "length": prov.get("length"),
            "citation_audit": result.get("citation_audit"),
            # The gated loop's own internal statistics (sentences removed by reason,
            # citations dropped, regenerated, first_pass_verified_rate,
            # survival_rate) and the "ONE report of the final text" (every row
            # verified).
            "loop_stats": result.get("loop_stats"),
            "claim_report": result.get("claim_report"),
            "metrics": result.get("metrics"),
        }

    def _append_fixture_claims(self) -> None:
        """Append the two protocol-authored claim fixtures (a known-good and a
        known-bad restatement of one seed paper's abstract) onto whatever the write
        job's own gated loop already saved, so the standalone verify-and-heal action
        has a controlled unsupported claim to heal away."""
        claims = self.state.claims or prepare_claims(self.protocol, self.state.seeds["paper_data"])
        check_claim_citations(claims)
        draft = fetch_draft(self.client, self.state.draft_id)
        content = append_claim_paragraphs(draft.get("content"), [c["text"] for c in claims])
        save_draft(self.client, self.state.draft_id, content)
        _write_json(self.output_dir / "draft_content.json", content)
        self.state.writing["claims_appended"] = claims

    def _verify_claims(self) -> None:
        """The standalone action (link -> verify -> finalize,
        `app.services.fulltext.verify_and_heal_claims`) actually heals the draft.
        Beyond the pre-existing checks, this checks the healing machinery
        itself did its job whenever the raw verification pass gave it something to
        heal: a fixture claim that the raw pass verified must survive into the final
        report, and a fixture claim the raw pass did not verify must be gone from the
        final report AND the saved draft, with the removal recorded in finalize_stats.
        A raw-pass verdict that itself does not match the protocol's own
        expected_status (model drift) is deliberately NOT raised here -- that is
        `claim_problems()`'s job, reported as exit code 2, not a second, redundant
        crash from this method for the same underlying disagreement."""
        task_id = start_verification(self.client, self.state.project_id, self.state.draft_id)
        self.state.steps[-1].task_id = task_id
        self._poll(task_id, "claim verification")
        report = fetch_claim_report(self.client, self.state.draft_id)
        _write_json(self.output_dir / "claim_report.json", report)
        rows = match_claim_results(self.state.writing["claims_appended"], report)
        draft = fetch_draft(self.client, self.state.draft_id)
        draft_content = draft.get("content")
        _write_json(self.output_dir / "draft_content.json", draft_content)
        # Kept as private instance attributes, not on `self.state`/`summary.json` --
        # `_draft_content` so `_write_delivered_evidence` can check each row's
        # sentence against the saved draft without a second fetch;
        # `_final_report_verifications`, the raw final-report rows (every one "verified"
        # by construction), so `_write_delivered_evidence` can build
        # `delivered_evidence.json` without this 53 KB-plus blob duplicating
        # `claim_report.json` inside `summary.json` (`state.verification["claims"]`
        # above only ever covers the two protocol-authored fixture claims
        # `match_claim_results` matches by text, not every claim the writer's own
        # citation links grounded, so it cannot serve the same purpose).
        self._draft_content = draft_content
        final_report = report.get("final_report") or {}
        self._final_report_verifications = final_report.get("verifications") or []
        # Recompute the healed length and survival rate from the draft actually
        # saved, not the write job's own pre-heal figures -- `_count_body_words`'s
        # own docstring.
        healed_words = _count_body_words(draft_content)
        length = self.state.writing.get("length")
        if isinstance(length, dict):
            length["final_words"] = healed_words
        loop_stats = self.state.writing.get("loop_stats")
        target_words = self.protocol["section"].get("target_words")
        if isinstance(loop_stats, dict) and target_words:
            loop_stats["survival_rate"] = healed_words / target_words
        # The exit invariant the standalone action's own result claims to hold for
        # the draft it just saved, checked here rather than trusted -- every
        # surviving row verified, no [NEEDS CITATION] marker, no citation occurrence
        # left with no verified row behind it, and no duplicated leading section
        # heading. Each part raises its own DemoError, so the reviewer's terminal
        # names exactly which guarantee broke.
        assert_every_final_row_verified(final_report.get("verifications") or [])
        assert_no_needs_citation_marker(draft_content)
        assert_citation_coverage_fully_resolved(report.get("citation_coverage"))
        assert_no_duplicate_leading_heading(draft_content)
        # An end-to-end check: every SURVIVING verification's own `claim_text` must
        # occur verbatim in the draft this script reads back, on every run. Checked
        # against `final_report`, not the raw, unfiltered `verifications` list: a
        # claim finalize legitimately healed away is not in the saved draft by
        # design, which is not the same thing as drift.
        claim_text_n, claim_text_m = check_claim_text_in_draft(
            draft_content, final_report.get("verifications") or []
        )
        if claim_text_n < claim_text_m:
            self.log(
                f"DRIFT: only {claim_text_n} of {claim_text_m} claim_text values occur "
                "verbatim in the saved draft"
            )

        finalize_stats = report.get("finalize_stats") or {}
        if report.get("verifications"):
            raw_by_id = {r["id"]: r for r in rows}
            final_rows = match_claim_results(
                self.state.writing["claims_appended"],
                {"verifications": final_report.get("verifications")},
            )
            final_by_id = {r["id"]: r for r in final_rows}
            # The healed report's own status for this same claim id, alongside (never
            # replacing) the raw `status` every other comparison uses, so
            # `claim_problems` can tell a claim expecting `unsupported` that
            # verify-and-heal correctly removed apart from one that still, wrongly,
            # survived.
            for row in rows:
                row["final_status"] = final_by_id.get(row["id"], {}).get("status")

            supported_status = raw_by_id.get("supported-1", {}).get("status")
            if supported_status == "verified" and final_by_id.get(
                "supported-1", {}
            ).get("status") != "verified":
                raise DemoError(
                    "supported-1 was verified but did not survive into the final "
                    f"(healed) report: {final_by_id.get('supported-1')}"
                )

            unsupported_status = raw_by_id.get("unsupported-1", {}).get("status")
            if unsupported_status not in ("verified", "no_full_text", "not_in_report"):
                if final_by_id.get("unsupported-1", {}).get("status") != "not_in_report":
                    raise DemoError(
                        "unsupported-1 was not verified but still appears in the "
                        f"final (healed) report: {final_by_id.get('unsupported-1')}"
                    )
                unsupported_claim = next(
                    c for c in self.state.writing["claims_appended"]
                    if c["id"] == "unsupported-1"
                )
                draft_texts = _draft_paragraph_texts(draft_content)
                if any(_norm(unsupported_claim["text"]) in _norm(t) for t in draft_texts):
                    raise DemoError(
                        "unsupported-1's text is still present in the saved (healed) draft"
                    )
                finalize_total = sum(
                    v for v in finalize_stats.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                )
                if finalize_total < 1:
                    raise DemoError(
                        "verify-and-heal recorded no removal in finalize_stats even "
                        "though unsupported-1 was healed away"
                    )

        self.state.verification = {
            "task_id": task_id,
            "verified_count": report.get("verified_count"),
            "unsupported_count": report.get("unsupported_count"),
            "nuance_count": report.get("nuance_count"),
            "abstract_only_count": report.get("abstract_only_count"),
            "error_count": report.get("error_count"),
            "full_text_coverage": report.get("full_text_coverage"),
            "claims_in_report": len(report.get("verifications") or []),
            "provenance": report.get("provenance"),
            "claims": rows,
            # The standalone action's own healing outcome.
            "healed": report.get("healed"),
            "finalize_stats": finalize_stats,
            "final_report_verified_count": final_report.get("verified_count"),
            # The three-part exit invariant asserted above (raises before this point
            # on any violation), recorded for `compare_with_expected`'s own
            # self-consistency check, not for a comparison with *expected*.
            "exit_invariant": {
                "final_report_rows_checked": len(final_report.get("verifications") or []),
                "citation_coverage_unresolved": (
                    (report.get("citation_coverage") or {}).get("unresolved")
                    if report.get("citation_coverage") is not None
                    else None
                ),
            },
            # How many citations were found, resolved and sent to the verifier,
            # independent of how they were rendered. Copied through verbatim; `None`
            # on a report from a backend build before this field existed.
            "citation_coverage": report.get("citation_coverage"),
            "claim_text_in_draft": f"{claim_text_n} of {claim_text_m}",
        }
        # The offline delivered-text checker's own rules, over the files this method
        # just saved -- the last step of the protocol section, as
        # `demo/check_delivered.py`'s own module docstring promises.
        self._check_delivered_text_for_section(
            index=None, protocol_claims=self.state.writing["claims_appended"]
        )

    def _write_extra_sections(self) -> None:
        """Write and verify every entry of ``self.sections``, in order, after the
        protocol section's own standalone verify-and-heal. Each one gets its
        own draft, on the same project and the same 12 seed papers, through the same AI
        Write job and the same standalone verify-and-heal action the protocol section
        used -- with no fixture claims appended (the protocol's two fixtures test one
        specific mechanism, the write job's own gate; they are not repeated per
        section) -- and the same four-part exit invariant `_verify_claims` already
        enforces on the protocol section's own draft."""
        self.state.sections = [
            self._write_one_extra_section(index, section)
            for index, section in enumerate(self.sections, start=2)
        ]

    def _write_one_extra_section(self, index: int, section: dict[str, Any]) -> dict[str, Any]:
        """One extra section's own write-then-heal, writing ``writing_result_<index>.
        json``, ``draft_content_<index>.json`` and ``claim_report_<index>.json`` (the
        protocol section's own three files, named ``<index>`` = 2, 3, 4, ... for "the
        Nth section of this run" rather than reusing the protocol section's own
        unindexed names). Returns the per-section summary `_write_extra_sections`
        collects under ``state.sections``, and appends this section's own
        `build_delivered_evidence` record to `self._extra_section_delivered_records`
        for `_write_delivered_evidence` to merge in afterwards.

        ``draft_content_<index>.json`` is written TWICE -- once right after the write
        job (so a failure between here and the heal still leaves something on disk),
        and again here, after the standalone verify-and-heal action, exactly as
        `_verify_claims` re-writes the protocol section's own unindexed
        ``draft_content.json`` after its own heal. Without the second write, the
        four-part invariant below and the offline checker
        (`_check_delivered_text_for_section`) would both read the PRE-heal draft
        against a POST-heal ``claim_report_<index>.json``: a sentence the heal removed
        would still be on disk and reported as an unclassified-sentence violation, and
        a sentence the heal rewrote in place would have its own ``claim_text`` report
        as not-found-in-draft -- and the saved per-section artefact would not be the
        text actually delivered."""
        section_as_protocol = {"section": section}
        title = section["title"]
        draft_id = create_draft(self.client, self.state.project_id, section_as_protocol)
        task_id = start_generation(self.client, draft_id, section_as_protocol, section_title=title)
        task = self._poll(task_id, f"AI write ({title})")
        result = task.get("result") or {}
        generated = result.get("content") or ""
        if not generated.strip():
            raise DemoError(f"AI Write returned empty content for section {title!r}")
        _write_json(self.output_dir / f"writing_result_{index}.json", result)
        draft = fetch_draft(self.client, draft_id)
        _write_json(self.output_dir / f"draft_content_{index}.json", draft.get("content"))

        verify_task_id = start_verification(self.client, self.state.project_id, draft_id)
        self._poll(verify_task_id, f"claim verification ({title})")
        report = fetch_claim_report(self.client, draft_id)
        _write_json(self.output_dir / f"claim_report_{index}.json", report)

        healed_draft = fetch_draft(self.client, draft_id)
        draft_content = healed_draft.get("content")
        # Re-write the per-section draft file from the HEALED draft before the
        # invariant and the offline check below read it back, exactly as
        # `_verify_claims` re-writes the protocol section's own `draft_content.json`.
        _write_json(self.output_dir / f"draft_content_{index}.json", draft_content)
        final_report = report.get("final_report") or {}
        final_verifications = final_report.get("verifications") or []

        # The same exit invariant `_verify_claims` enforces on the protocol section's
        # own draft.
        assert_every_final_row_verified(final_verifications)
        assert_no_needs_citation_marker(draft_content)
        assert_citation_coverage_fully_resolved(report.get("citation_coverage"))
        assert_no_duplicate_leading_heading(draft_content)
        # The offline delivered-text checker's own rules, over this section's own
        # files -- the last step of every extra section, same as the protocol
        # section's own call in `_verify_claims`.
        self._check_delivered_text_for_section(index=index)

        record = build_delivered_evidence(
            run_id=self.output_dir.name,
            draft_id=draft_id,
            section_title=title,
            verifications=final_verifications,
            fetch_chunk=lambda paper_id, chunk_index: fetch_fulltext_chunk(
                self.client, paper_id, chunk_index
            ),
            draft_content=draft_content,
        )
        self._extra_section_delivered_records.append(record)

        # Recompute the healed length and survival rate from the draft actually
        # saved above, not the write job's own pre-heal figures
        # (`_count_body_words`'s own docstring) -- copied, not mutated in place, since
        # `result["loop_stats"]` is not otherwise this method's own to change.
        healed_words = _count_body_words(draft_content)
        loop_stats = dict(result.get("loop_stats") or {})
        target_words = section.get("target_words")
        if target_words:
            loop_stats["survival_rate"] = healed_words / target_words
        return {
            "title": title,
            "draft_id": draft_id,
            "task_id": task_id,
            "verify_task_id": verify_task_id,
            "writing": {
                "section_type": result.get("section_type"),
                "loop_stats": loop_stats,
                "final_words": healed_words,
            },
            "verification": {
                "verified_count": report.get("verified_count"),
                "unsupported_count": report.get("unsupported_count"),
                "nuance_count": report.get("nuance_count"),
                "abstract_only_count": report.get("abstract_only_count"),
                "error_count": report.get("error_count"),
                "final_report_verified_count": final_report.get("verified_count"),
            },
            "delivered_evidence": {
                "rows": len(record["rows"]),
                "unlocated_rows": record["unlocated_rows"],
            },
        }

    def _write_delivered_evidence(self) -> None:
        """After the standalone verify-and-heal step, write ``delivered_evidence.json``
        -- one row per claim that survived into the final, healed report of the
        protocol section AND of every extra section
        (`merge_delivered_evidence_records`), each carrying the verbatim source
        passage a human judge checks it against, obtained through the read-only
        chunk endpoint added under the existing full-text routes for this record."""
        protocol_record = build_delivered_evidence(
            run_id=self.output_dir.name,
            draft_id=self.state.draft_id,
            section_title=self.protocol["section"]["title"],
            verifications=self._final_report_verifications,
            fetch_chunk=lambda paper_id, chunk_index: fetch_fulltext_chunk(
                self.client, paper_id, chunk_index
            ),
            draft_content=self._draft_content,
        )
        record = merge_delivered_evidence_records(
            [protocol_record, *self._extra_section_delivered_records]
        )
        _write_json(self.output_dir / "delivered_evidence.json", record)
        self.state.delivered_evidence = {
            "rows": len(record["rows"]),
            "unlocated_rows": record["unlocated_rows"],
        }

    def _check_delivered_text(self) -> None:
        """The last step of the whole run. Every section's own rules 1-4, 6-9 already
        ran, right after that section was saved (`_check_delivered_text_for_section`);
        this adds rule 5, checked once over the ``delivered_evidence.json``
        `_write_delivered_evidence` just wrote, since it is a whole-run artefact, not
        a per-section one."""
        violations, _notices = check_delivered.check_delivered_evidence_file(self.output_dir)
        self._delivered_check_violations.extend(violations)

    def _compare(self, summary: dict[str, Any]) -> list[dict[str, Any]] | None:
        if not self.expected_dir:
            return None
        path = self.expected_dir / "summary.json"
        if not path.exists():
            return None
        return compare_with_expected(
            summary, _load_json(path, "expected summary"), replay_queries=self.replay_queries,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-url", default=DEFAULT_API_URL, help=f"API base URL (default {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--protocol", type=Path, default=DEFAULT_PROTOCOL, help="protocol.json path"
    )
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS, help="seed_dois.json path")
    parser.add_argument(
        "--sections",
        type=Path,
        default=DEFAULT_SECTIONS if DEFAULT_SECTIONS.exists() else None,
        help=(
            "JSON array of extra sections (title, instructions, target_words), each "
            "written and verified on the same project after the protocol section "
            "(default demo/sections.json when that file exists; pass no flag and "
            "remove/rename the file to skip)"
        ),
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="directory holding expected/summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default demo/output/<timestamp>)",
    )
    search_mode = parser.add_mutually_exclusive_group()
    search_mode.add_argument(
        "--skip-search", action="store_true", help="skip Smart Search (seed papers only)"
    )
    search_mode.add_argument(
        "--search-only",
        action="store_true",
        help=(
            "stop after exporting the screening record; write a summary with only the "
            "screening block (exit codes unchanged)"
        ),
    )
    parser.add_argument(
        "--replay-queries",
        action="store_true",
        help=(
            "read provenance.rounds[].queries[].query from --expected's "
            "screening_record.json and send them as queries_override instead of "
            "generating queries (default off)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=(
            f"seconds to wait for each background task except Smart Search (default "
            f"{DEFAULT_TIMEOUT_S} = {DEFAULT_TIMEOUT_S // 60} min); see --search-timeout "
            "for the Smart Search task itself"
        ),
    )
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=DEFAULT_SEARCH_TIMEOUT_S,
        help=(
            f"seconds to wait for the Smart Search task specifically (default "
            f"{DEFAULT_SEARCH_TIMEOUT_S:.0f} = {DEFAULT_SEARCH_TIMEOUT_S / 60:.0f} min): "
            "the search loop's own budget plus the inclusion-judge stage's own budget, "
            "one more judge call, and margin"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 when the comparison with demo/expected/ drifts",
    )
    return parser


def main(
    argv: list[str] | None = None, *, client_factory: Callable[[str], ApiClient] | None = None
) -> int:
    args = build_parser().parse_args(argv)
    started = datetime.now(timezone.utc)
    output_dir = args.output or (DEFAULT_OUTPUT_ROOT / started.strftime("%Y%m%d-%H%M%S"))
    try:
        protocol = _load_json(args.protocol, "protocol.json")
        seeds = _load_json(args.seeds, "seed_dois.json")
        sections: list[dict[str, Any]] | None = None
        if args.sections is not None:
            sections = _load_json_array(args.sections, "sections.json")
            _validate_sections(sections)
        client = (client_factory or ApiClient)(args.api_url)
        try:
            runner = DemoRunner(
                client,
                protocol=protocol,
                seeds=seeds,
                output_dir=output_dir,
                expected_dir=args.expected,
                timeout_s=args.timeout,
                search_timeout_s=args.search_timeout,
                skip_search=args.skip_search,
                search_only=args.search_only,
                replay_queries=args.replay_queries,
                sections=sections,
            )
            summary = runner.run()
        finally:
            client.close()
    except DemoError as exc:
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\nDEMO INTERRUPTED", file=sys.stderr)
        return 130
    # --search-only never ran claim verification, so there is nothing for claim_problems
    # to check; the claim-mismatch exit code (2) does not apply to this run.
    problems = [] if args.search_only else claim_problems(summary)
    if problems:
        print("\nDEMO CLAIM MISMATCH:\n  " + "\n  ".join(problems), file=sys.stderr)
        return EXIT_CLAIMS
    # The offline delivered-text checker's own violations (rules 1-7), folded into
    # `summary["delivered_check"]` by `DemoRunner.run()` rather than raised from
    # inside a step -- see `DemoRunner._check_delivered_text_for_section`'s own
    # docstring. `None` for `--search-only` (no section was ever written).
    delivered_violations = (summary.get("delivered_check") or {}).get("violations") or []
    if delivered_violations:
        print(
            "\nDEMO DELIVERED-TEXT VIOLATION:\n  "
            + "\n  ".join(
                f"[{v['rule']}] {v['section']}: {v['detail']}" for v in delivered_violations
            ),
            file=sys.stderr,
        )
        return EXIT_CLAIMS
    checks = summary.get("comparison")
    if args.strict and checks and any(not c["ok"] for c in checks):
        print(
            "\nDEMO DRIFT: comparison with demo/expected/ is outside tolerance (--strict)",
            file=sys.stderr,
        )
        return EXIT_DRIFT
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
