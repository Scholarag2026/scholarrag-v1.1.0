"""Full-text acquisition, extraction, chunking, and claim verification."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from uuid import UUID

import fitz  # pymupdf
import httpx
from sqlalchemy import select

from app.clients.unpaywall import UnpaywallClient
from app.config import settings
from app.models.analysis_job import JobStatus
from app.models.draft import Draft
from app.models.paper import Paper
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.schemas.fulltext import (
    CitationCoverage,
    ClaimAssertion,
    ClaimVerification,
    ClaimVerificationReport,
)
from app.schemas.provenance import LLMCallProvenance, provenance_from_run
from app.services import sentence_coverage
from app.services import task as task_service
from app.services.citation_audit import (
    _NARRATIVE_CITE,
    _PAREN_GROUP,
    _SURNAME,
    _surname_key,
    _year_key,
    citation_spans,
)

logger = logging.getLogger(__name__)

MAX_TEXT_SIZE = 200 * 1024  # 200 KB
MAX_CHUNKS = 40
SECTION_PATTERNS = [
    r"(?i)^(abstract|introduction|literature\s+review|methodology|methods|"
    r"results|findings|discussion|conclusion|implications|references)\s*$"
]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf.

    Returns the full extracted text, truncated to MAX_TEXT_SIZE if necessary.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)
    if len(full_text.encode("utf-8")) > MAX_TEXT_SIZE:
        # Rough truncation -- keep the beginning of the paper
        full_text = full_text[: MAX_TEXT_SIZE // 2]
    return full_text


def detect_sections(text: str) -> list[dict]:
    """Split text into sections by detecting section headers.

    Returns a list of dicts with 'section' (header name) and 'text' keys.
    Sections with empty text are removed. At most MAX_CHUNKS sections returned.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_section = "preamble"
    current_text: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_header = False
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, stripped):
                # Flush the previous section
                if current_text:
                    sections.append({
                        "section": current_section,
                        "text": "\n".join(current_text).strip(),
                    })
                current_section = stripped.lower()
                current_text = []
                is_header = True
                break
        if not is_header:
            current_text.append(line)

    # Flush final section
    if current_text:
        sections.append({
            "section": current_section,
            "text": "\n".join(current_text).strip(),
        })

    return [s for s in sections if s["text"]][:MAX_CHUNKS]


def chunk_text(
    text: str, max_chunk_size: int = 2000, *, drop_reference_chunks: bool = True
) -> list[dict]:
    """Chunk text into pieces of *max_chunk_size* characters.

    First attempts to split by detected sections. If only one section is found,
    falls back to splitting by paragraphs.

    PDF upload and paste (`app/api/fulltext.py`) and bulk-upload confirm
    (`app/api/papers.py`, via `app/services/paper_upload.py`) call this function and
    store its raw output as `metadata["fulltext_chunks"]`. When *drop_reference_chunks*
    is true (the default), `drop_reference_and_backmatter_chunks` is applied to the
    chunk list, in the same order (truncate to `MAX_CHUNKS`, then drop)
    `acquire_full_texts` uses, so every caller of `chunk_text` gets the same
    reference-list and back-matter drop with no edit to those files.
    `acquire_full_texts` passes `drop_reference_chunks=False` and keeps its own
    explicit drop afterward, so it can still count and report the number dropped
    (`fulltext_dropped_reference_chunks`, `reference_chunks_dropped`) without dropping
    twice.

    Returns a list of dicts with 'section' and 'text' keys.
    """
    sections = detect_sections(text)
    if sections and len(sections) > 1:
        chunks = sections
    else:
        # Fallback: split by paragraphs
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > max_chunk_size:
                if current_chunk:
                    chunks.append({
                        "section": f"chunk_{len(chunks)}",
                        "text": current_chunk.strip(),
                    })
                current_chunk = para
            else:
                current_chunk += "\n\n" + para
        if current_chunk.strip():
            chunks.append({
                "section": f"chunk_{len(chunks)}",
                "text": current_chunk.strip(),
            })
        chunks = chunks[:MAX_CHUNKS]

    if drop_reference_chunks:
        chunks, _dropped = drop_reference_and_backmatter_chunks(chunks)
    return chunks


class PublisherInterstitialError(Exception):
    """Every attempt to download this URL came back looking like a publisher
    interstitial (an HTML/blocking page) rather than a PDF -- raised by
    `fetch_pdf_from_url` so `acquire_full_texts` can record `publisher_interstitial`
    instead of a bare, unexplained failure (F4 acquisition design)."""


#: Seconds to wait before the first and second retry of a blocked or
#: wrong-content-type PDF download.
PDF_FETCH_RETRY_DELAYS_S: tuple[float, float] = (3.0, 10.0)
#: HTTP statuses that are worth retrying on their own, regardless of content type:
#: blocking/rate-limiting (403, 429) and server-side failures (5xx).
PDF_FETCH_RETRY_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
#: A definitively missing document, not a publisher block: a
#: 404 or 410 is excluded from both the retry decision and `all_interstitial` below,
#: so a dead OA link is neither retried (wasting `PDF_FETCH_RETRY_DELAYS_S`) nor
#: reported as `publisher_interstitial`.
PDF_FETCH_MISSING_STATUSES = frozenset({404, 410})
#: The same Chrome-like User-Agent `demo/tools/verify_pdf.py` documents under
#: --browser-ua, sent only on the two retries: some publisher hosts serve an HTML
#: interstitial to httpx's default User-Agent and a real PDF to a browser-like one.
PDF_FETCH_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def _pdf_response_looks_like_interstitial(resp: httpx.Response) -> bool:
    """True when *resp* is not a real PDF body: HTML content-type, or a body that
    does not start with the ``%PDF`` magic bytes every genuine PDF opens with."""
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    return content_type == "text/html" or not resp.content.startswith(b"%PDF")


async def fetch_pdf_from_url(
    url: str, *, http_client: httpx.AsyncClient | None = None
) -> bytes | None:
    """Download a PDF from *url*. Returns raw bytes, or ``None`` on failure.

    A response that looks like a publisher interstitial rather than a PDF
    (`_pdf_response_looks_like_interstitial`), or an HTTP 403, 429 or 5xx, is
    retried after `PDF_FETCH_RETRY_DELAYS_S` (3 s, then 10 s); the two retries use
    `PDF_FETCH_BROWSER_HEADERS` in place of httpx's default headers. Every attempt
    is logged with the host, status and content-type it received.

    Raises `PublisherInterstitialError` when every attempt that reached the server
    (i.e. every attempt that is not a bare transport failure or a 404/410) looked like
    an interstitial, so the caller can tell "blocked by the publisher" apart from a
    download that simply never succeeded. A transport-level failure (DNS,
    connection refused, timeout) is not retried and returns ``None``. A 404 or 410 is
    a definitively missing document, not a publisher block: it
    is neither retried nor counted toward `PublisherInterstitialError`, and this
    function returns ``None`` for it exactly as it would for a transport failure.

    *http_client*: injection seam for tests (``httpx.AsyncClient(transport=...)``);
    production code always leaves this ``None`` and gets a fresh pooled client.
    """
    host = httpx.URL(url).host or url
    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    responses_seen = 0
    all_interstitial = True
    try:
        headers: dict[str, str] | None = None
        for attempt_index in range(1 + len(PDF_FETCH_RETRY_DELAYS_S)):
            try:
                resp = await client.get(url, headers=headers)
            except Exception as exc:
                logger.warning("Failed to fetch PDF from %s: %s", url, exc)
                return None
            responses_seen += 1
            content_type = resp.headers.get("content-type", "").split(";")[0].strip()
            logger.info(
                "PDF fetch attempt %d for host=%s: status=%s content-type=%s",
                attempt_index + 1, host, resp.status_code, content_type,
            )
            interstitial = _pdf_response_looks_like_interstitial(resp)
            if resp.status_code == 200 and not interstitial:
                return resp.content
            if resp.status_code in PDF_FETCH_MISSING_STATUSES:
                all_interstitial = False
                break
            all_interstitial = all_interstitial and interstitial
            should_retry = interstitial or resp.status_code in PDF_FETCH_RETRY_STATUSES
            if not should_retry:
                break
            if attempt_index < len(PDF_FETCH_RETRY_DELAYS_S):
                await asyncio.sleep(PDF_FETCH_RETRY_DELAYS_S[attempt_index])
                headers = PDF_FETCH_BROWSER_HEADERS
    finally:
        if own_client:
            await client.aclose()

    if responses_seen and all_interstitial:
        raise PublisherInterstitialError(url)
    return None


# --------------------------------------------------------------------------------------
# Full-text pipeline guards, run inside `acquire_full_texts` after extraction and before
# storage, so the claim verifier can never be shown the wrong paper's text or a reference
# list mislabelled as prose. This mirrors, but does not import,
# `evaluation/claims/build_hss_set.py`'s `check_fetched_text_identity` /
# `is_reference_or_frontmatter_chunk` -- the backend never depends on the evaluation
# package.
# --------------------------------------------------------------------------------------

#: How many leading characters of the extracted text are checked against the paper's own
#: title/author (a title page rarely repeats every title word verbatim, so this is a
#: threshold over a prefix, not an exact match over the whole document).
IDENTITY_TITLE_PREFIX_CHARS = 3000
#: Coverage below this fires the guard. Same value as the evaluation builder's
#: `IDENTITY_MIN_TOKEN_OVERLAP`, which this guard mirrors.
IDENTITY_MIN_TOKEN_OVERLAP = 0.3

_IDENTITY_TOKEN_RE = re.compile(r"[a-z0-9]+")
#: Function words stripped before comparing title tokens: present in nearly every
#: title/prefix regardless of subject matter, so they inflate overlap without
#: discriminating anything.
_IDENTITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
        "have", "in", "into", "is", "its", "not", "of", "on", "or", "our", "that", "the",
        "their", "this", "to", "was", "were", "with",
    }
)


def _fold_identity_diacritics(text: str) -> str:
    """NFKC-normalise, then NFKD-decompose and drop every combining mark.

    A plain `str.casefold()` maps some accented and dotted letters (e.g. the Turkish
    dotted capital I) to a base letter plus a *combining* mark rather than a plain ASCII
    letter, so an unfolded comparison can reject a correctly matching name. Decomposing
    and stripping combining marks before casefolding avoids that.
    """
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text))
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _identity_tokens(text: str) -> set[str]:
    folded = _fold_identity_diacritics(text).casefold()
    return {w for w in _IDENTITY_TOKEN_RE.findall(folded) if w not in _IDENTITY_STOPWORDS}


def check_full_text_identity(
    *,
    expected_title: str,
    extracted_text: str,
    expected_authors: Sequence[str] | None = None,
    prefix_chars: int = IDENTITY_TITLE_PREFIX_CHARS,
    min_token_overlap: float = IDENTITY_MIN_TOKEN_OVERLAP,
) -> tuple[bool, str | None]:
    """(ok, detected_fragment): does *extracted_text* actually describe *expected_title*?

    An open-access link can serve a wholly different article than the one its DOI, title
    and authors name (observed failure: `10.55593/ej.27108a7`'s Unpaywall link served
    Ambele (2022), a different paper in the same narrow subfield, instead of the named
    Thongwichit and Ulla TESL-EJ article: the two titles share enough vocabulary --
    "translanguaging", "Thailand", "English", "teachers", "classrooms" -- that title
    overlap alone reaches 0.45, comfortably over the threshold, so title overlap alone
    cannot be the deciding signal here). Checked by normalised token overlap between
    *expected_title* and the first *prefix_chars* characters of *extracted_text*, AND,
    whenever *expected_authors* is given and names a usable surname (longer than two
    characters), a literal, case- and diacritic-insensitive match of that surname
    somewhere in the same prefix -- both signals must agree for a match; a same-subfield
    decoy can satisfy the title signal alone, but is exceedingly unlikely to also carry
    the named author's own surname. When no usable author surname is available to check,
    the title signal alone decides.

    Returns `(True, None)` on a match, or when *expected_title* is empty, or folds to
    exactly "untitled" (nothing to check against: `app/clients/openalex.py`'s
    `_parse_work` stores the literal title "Untitled" when OpenAlex itself has no title
    for a work, and that single non-stopword token can never overlap a real PDF's own
    text). Returns `(False, fragment)` on a mismatch, where *fragment* is the
    whitespace-normalised first 200 characters of the extracted text, for the caller to
    record what was actually fetched.
    """
    prefix = extracted_text[:prefix_chars]
    title_tokens = _identity_tokens(expected_title)
    if not title_tokens or title_tokens == {"untitled"}:
        return True, None
    prefix_tokens = _identity_tokens(prefix)
    overlap = len(title_tokens & prefix_tokens) / len(title_tokens)
    author_ok = True
    if expected_authors:
        first = next((a for a in expected_authors if a and a.strip()), None)
        # Comma-aware surname extraction: `_build_author_year_lookup` in this same
        # module already handles both "Given Family" and "Family, Given" defensively
        # with this exact expression, since the bulk-upload path takes author names
        # from an LLM extraction agent and the format is not guaranteed. A comma-naive
        # `.split()[-1]` would instead return the given name for "Family, Given",
        # rejecting a correct full text whenever the PDF prints initials rather than
        # the full given name.
        surname = first.strip().split(",")[0].split()[-1] if first else None
        if surname and len(surname) > 2:
            folded_prefix = " ".join(_fold_identity_diacritics(prefix).casefold().split())
            author_ok = _fold_identity_diacritics(surname).casefold() in folded_prefix
    if overlap >= min_token_overlap and author_ok:
        return True, None
    fragment = " ".join(prefix.split())[:200]
    return False, fragment or None


# --------------------------------------------------------------------------------------
# The citation author string comes from the acquired full text's own byline, not only
# OpenAlex's author list. The Razmi and Ghane (2024) paper is a motivating case --
# OpenAlex records two authors in an order the article itself does not use, and the
# running head reads "Ghane, Razmi, Dehghanpoor and Nematollahi".
# --------------------------------------------------------------------------------------

_BYLINE_ABSTRACT_MARKER_RE = re.compile(r"\babstract\b", re.IGNORECASE)
#: A superscript affiliation footnote opening a new line ("\n1 Yazd University", "\n2
#: Shahid Bahonar..."): the second signal (besides "Abstract") that the byline itself
#: has ended, needed because nothing else punctuates the boundary between the last
#: author's own name and the affiliation list right after it.
_BYLINE_AFFILIATION_MARKER_RE = re.compile(r"\n\s*\d{1,2}\s*[A-ZÀ-Þ]")
#: A "Given Middle Family" token run: an initial-capital word, allowing an internal
#: apostrophe or hyphen ("O'Brien", "Bonilla-Lopez"), plus the trailing affiliation
#: superscript digits a byline commonly carries ("Razmi2").
_BYLINE_CAPITALISED_TOKEN_RE = re.compile(r"[A-ZÀ-Þ][\w’'\-]*\d*")
_BYLINE_TRAILING_DIGITS_RE = re.compile(r"\d+$")
#: A genuine author-name segment ("Mohammad Hasan Razmi2") is short; the title, a
#: copyright notice, or an affiliation line this split does not otherwise catch runs
#: much longer and must never be mistaken for one just because it happens to hold two
#: or more capitalised words of its own.
_BYLINE_MAX_SEGMENT_CHARS = 60


def _byline_prefix(extracted_text: str, prefix_chars: int = IDENTITY_TITLE_PREFIX_CHARS) -> str:
    """The identity prefix `check_full_text_identity` already reads, cut at the first
    "Abstract" marker or affiliation-footnote line within it, whichever comes first --
    the byline sits between the title and either of those on every layout this parser
    has been checked against, and cutting there keeps the abstract's own prose and the
    affiliation list out of the segment split below."""
    prefix = extracted_text[:prefix_chars]
    cut_points = [
        match.start()
        for match in (
            _BYLINE_ABSTRACT_MARKER_RE.search(prefix),
            _BYLINE_AFFILIATION_MARKER_RE.search(prefix),
        )
        if match is not None
    ]
    return prefix[: min(cut_points)] if cut_points else prefix


def _parse_byline_surnames(extracted_text: str) -> list[str]:
    """Every author surname `_byline_prefix`'s own text names, in order: the prefix is
    cut into segments on commas and on " and ", a segment
    survives only when it is short (`_BYLINE_MAX_SEGMENT_CHARS`) and holds at least two
    capitalised tokens (a plausible "Given Family" name, as opposed to a title
    fragment, a copyright notice, or a bare initial), and each surviving segment's own
    surname is its last capitalised token with any trailing affiliation-marker digits
    stripped ("Razmi2" -> "Razmi")."""
    prefix = _byline_prefix(extracted_text)
    segments = re.split(r",|\band\b", prefix)
    surnames: list[str] = []
    for segment in segments:
        stripped = segment.strip()
        if not stripped or len(stripped) > _BYLINE_MAX_SEGMENT_CHARS:
            continue
        tokens = _BYLINE_CAPITALISED_TOKEN_RE.findall(segment)
        if len(tokens) < 2:
            continue
        surname = _BYLINE_TRAILING_DIGITS_RE.sub("", tokens[-1])
        if surname:
            surnames.append(surname)
    return surnames


def _accept_byline(parsed_surnames: Sequence[str], openalex_author_names: Sequence[str]) -> bool:
    """True only when every OpenAlex author surname is found among *parsed_surnames*,
    case- and diacritic-insensitively, each exactly once, the positions they are found
    at form one contiguous run with no other token -- matching or not -- between them,
    and that run starts at index 0 of the parse. Membership alone accepted a live
    PDF's own running-header contact block ahead of the title, where "Razmi" occurs
    once from the header noise and once more, correctly, as the real byline's own
    second author, with "Languages" and "Kerman" (the header's own "Department of
    Foreign Languages ... University of Kerman") interleaved between the header's own
    "Razmi" and the real byline's "Ghane" -- exactly the shape that let a running
    header decide the first author rather than the paper's own byline. The contiguity
    check alone still let a DIFFERENT noise token -- one OpenAlex does not name at all,
    so it never collides with an OpenAlex surname -- sit ahead of an otherwise clean,
    contiguous OpenAlex run and still be handed to the writing prompt as the first
    surname; requiring the run to start the parse closes that gap without narrowing
    what a genuine byline (which always opens its own segment list) can still match.
    Otherwise the parse is not trusted and the caller falls back to OpenAlex's own
    author order, so the failure mode of a layout this parser does not handle -- or
    cannot tell apart from surrounding noise -- is no change, never a wrong author
    string."""
    if not parsed_surnames or not openalex_author_names:
        return False
    folded_parsed = [_fold_identity_diacritics(s).casefold() for s in parsed_surnames]
    folded_openalex: set[str] = set()
    for name in openalex_author_names:
        if not name or not name.strip():
            continue
        surname = name.strip().split(",")[0].split()[-1]
        if not surname:
            continue
        folded_openalex.add(_fold_identity_diacritics(surname).casefold())
    if not folded_openalex:
        return False
    matches = [index for index, surname in enumerate(folded_parsed) if surname in folded_openalex]
    # Every OpenAlex surname must be found, and none of them found more than once: a
    # match count that does not equal the number of distinct OpenAlex surnames means
    # either one is missing or one repeats in the parse.
    if len(matches) != len(folded_openalex):
        return False
    if {folded_parsed[index] for index in matches} != folded_openalex:
        return False
    # The matched positions must be one contiguous run: no token, name or not, sits
    # between the first OpenAlex surname the parse names and the last.
    if matches != list(range(matches[0], matches[-1] + 1)):
        return False
    # That run must also START the parse, so no surname
    # absent from the OpenAlex list -- a running header's own contact name, a journal
    # name, a department -- may precede the first OpenAlex author and be handed to the
    # writing prompt as the byline's own first surname.
    if matches[0] != 0:
        return False
    return True


#: Reference-list / bibliography signal patterns: "Retrieved from/on", a bare DOI, a
#: "pp. <n>" page range, a "(YYYY)" year in parentheses, and an initial-letter author name
#: ("Smith, J. A."). None of these is rare in prose on its own (a single parenthetical
#: citation year is normal); it is their *density* over a chunk that distinguishes a
#: bibliography from running text (observed failure: the Apples paper's chunk 2, 2,999
#: characters of pure reference list, carried the section label "discussion").
_REFERENCE_CHUNK_RETRIEVED_RE = re.compile(r"\bretrieved\s+(?:from|on)\b", re.IGNORECASE)
_REFERENCE_CHUNK_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
_REFERENCE_CHUNK_PP_RE = re.compile(r"\bpp?\.\s?\d")
_REFERENCE_CHUNK_YEAR_PAREN_RE = re.compile(r"\(\d{4}[a-z]?\)")
_REFERENCE_CHUNK_INITIAL_AUTHOR_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+,\s(?:[A-Z]\.\s?){1,3}")
#: Pattern hits per 1,000 characters above which a chunk reads as a reference list.
REFERENCE_CHUNK_DENSITY_THRESHOLD = 6.0
#: Below this length, one citation can look arbitrarily "dense"; too little text for a
#: density estimate to mean anything.
REFERENCE_CHUNK_MIN_LENGTH = 150
#: Minimum number of `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE` ("Smith, J. A."-style) matches
#: required, in addition to density, before a chunk is dropped as a reference list.
#: Density alone conflates two different things: a real
#: bibliography (dense AND structured as a list of "Author, Initials. (Year)." entries) and
#: a citation-dense narrative discussion (dense on years-in-parentheses and page cites
#: alone, but with no bibliography structure at all -- every citation is inline and
#: narrative, e.g. "Feryok (2012)", never a list entry). Reproduced on a real cached-corpus
#: paper (`evaluation/claims/data/hss_fulltext/10-55593_ej-26103a4.json`, re-chunked with
#: this module's own `chunk_text`): a 5,317-character "Data Analysis" narrative chunk scored
#: 7.44 hits/1,000 chars (over the 6.0 threshold) on 24 narrative-year citations and 12 page
#: cites, with zero author-initial matches, while the paper's true bibliography chunks in
#: the same document each carried 12-41 author-initial matches. Requiring at least three
#: such matches keeps that narrative prose (0 matches) while leaving every true reference
#: list flagged, since a bibliography entry that short-lived would not itself be worth
#: dropping. See `evaluation/tests` for the equivalent evaluation-side finding and
#: `backend/tests/test_fulltext_pipeline_guards.py` for the regression fixtures.
REFERENCE_CHUNK_MIN_AUTHOR_MATCHES = 3
#: Applying the author-match floor above
#: unconditionally re-opens the guard for the very front-matter/back-matter chunks it exists
#: to catch, whenever those chunks happen to be short. Re-chunking all 11 papers in
#: `evaluation/claims/data/hss_fulltext/` with this module's own `chunk_text` (245 chunks
#: total) reproduces two real regressions: `10-32601_ejal-710194.json` chunk 0 (779 chars,
#: density 7.70, 2 author matches) is journal front matter carrying the masthead, DOI,
#: submission dates and the paper's own APA self-citation, and
#: `10-55593_ej-27105a9.json` chunk 24 (243 chars, density 12.35, 2 author matches) is a
#: bibliography tail plus copyright notice -- both used to be correctly dropped on density
#: alone, and both are kept once two matches falls short of the three the floor requires.
#: Non-APA bibliographies lose the guard entirely at any length: a Vancouver-style block
#: ("1. Surname AB, Surname CD. Title...") or an IEEE-style block ("[1] A. Surname and B.
#: Surname, ...") carries zero `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE` matches regardless of
#: size, since that pattern is APA-specific ("Surname, A. B."). The floor is therefore gated
#: on length: only a chunk at or above this length needs the author-match requirement at
#: all, since the false positive the floor exists to fix is itself long (5,317 characters,
#: see `REFERENCE_CHUNK_MIN_AUTHOR_MATCHES` above); a chunk shorter than this goes back to
#: density alone.
REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH = 1500
#: A line opening with a bracketed or dotted/parenthesised reference number -- "[12] ",
#: "12. ", "12) " -- the structural signature of a numbered (Vancouver/IEEE-style)
#: bibliography entry. Checked in disjunction with `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE`
#: so a long numbered bibliography with zero APA-style
#: author-initial matches is still eligible for the density check: on a synthetic
#: 3,057-character, 25-entry Vancouver block with zero author-initial matches this scores
#: 25; on the narrative-discussion fixture described above it scores 0. This shape alone
#: is not specific to bibliographies -- it is also the
#: shape of an ordinary numbered list in running prose (enumerated research questions,
#: findings, or procedure steps), so a chunk carrying such a list, and nothing else that
#: looks like a reference, would be wrongly exempted from the author-match floor.
#: `_reference_chunk_numbered_entry_count` below gates each
#: match on carrying a reference signal on the *same line*; use that function, not this
#: regex's raw `findall`, wherever the numbered-entry signal decides anything.
_REFERENCE_CHUNK_NUMBERED_ENTRY_RE = re.compile(r"(?m)^\s*(?:\[\d{1,3}\]|\d{1,3}[.)])\s+\S")
#: A bare 4-digit year (1800-2099), unparenthesised -- the numeral half of a numbered
#: Vancouver/IEEE entry's own publication year (e.g. "... Lang Learn J. 2018;45(3):210-225
#: ..."), as distinct from `_REFERENCE_CHUNK_YEAR_PAREN_RE`'s "(YYYY)" narrative-citation
#: shape. Checked only on a line that already opens with a numbered-entry marker, so a
#: plain sentence elsewhere in the chunk that happens to contain a year is never counted
#: by this pattern.
_REFERENCE_CHUNK_BARE_YEAR_RE = re.compile(r"\b(?:1[89]\d{2}|20\d{2})\b")


def _numbered_entry_line_has_reference_signal(line: str) -> bool:
    """True when *line* carries a DOI, a "pp." page range, "Retrieved from/on", a
    parenthesised year, or a bare 4-digit year -- none of which a normal numbered list of
    research questions, findings or procedure steps carries, and at least one of which
    every Vancouver- or IEEE-style bibliography entry carries."""
    return bool(
        _REFERENCE_CHUNK_DOI_RE.search(line)
        or _REFERENCE_CHUNK_PP_RE.search(line)
        or _REFERENCE_CHUNK_RETRIEVED_RE.search(line)
        or _REFERENCE_CHUNK_YEAR_PAREN_RE.search(line)
        or _REFERENCE_CHUNK_BARE_YEAR_RE.search(line)
    )


def _reference_chunk_numbered_entry_count(text: str) -> int:
    """Count `_REFERENCE_CHUNK_NUMBERED_ENTRY_RE` lines that also carry a reference signal
    on the same line. Verified against the real
    corpus (`evaluation/claims/data/hss_fulltext/`, 245 chunks re-chunked with this
    module's own `chunk_text`): 45 chunks dropped before this filter, 45 dropped after, 0
    differences; every existing fixture in `test_fulltext_pipeline_guards.py` keeps its
    verdict; a normal numbered research-question list inserted into the
    dense-discussion fixture described above, which the unfiltered count wrongly exempted
    from the author-match floor, correctly scores 0 and the chunk stays kept."""
    return sum(
        1
        for line in text.splitlines()
        if _REFERENCE_CHUNK_NUMBERED_ENTRY_RE.match(line)
        and _numbered_entry_line_has_reference_signal(line)
    )


#: Minimum `_reference_chunk_numbered_entry_count` matches, checked in disjunction with
#: `REFERENCE_CHUNK_MIN_AUTHOR_MATCHES`, before a chunk at or above
#: `REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH` is exempted from the floor. Same value as
#: `REFERENCE_CHUNK_MIN_AUTHOR_MATCHES` for symmetry: a numbered list this short would not
#: itself be worth dropping.
REFERENCE_CHUNK_MIN_NUMBERED_ENTRIES = 3
#: A chunk that opens with a bare heading and nothing else on the same line: "References",
#: "Bibliography", "Works Cited" (with or without a trailing colon).
_REFERENCES_HEADING_START_RE = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited)\s*(?:[:\n]|$)", re.IGNORECASE,
)


def _reference_chunk_pattern_count(text: str) -> int:
    return sum(
        len(p.findall(text))
        for p in (
            _REFERENCE_CHUNK_RETRIEVED_RE,
            _REFERENCE_CHUNK_DOI_RE,
            _REFERENCE_CHUNK_PP_RE,
            _REFERENCE_CHUNK_YEAR_PAREN_RE,
            _REFERENCE_CHUNK_INITIAL_AUTHOR_RE,
        )
    )


def is_reference_or_frontmatter_chunk(
    text: str,
    *,
    density_threshold: float = REFERENCE_CHUNK_DENSITY_THRESHOLD,
    min_length: int = REFERENCE_CHUNK_MIN_LENGTH,
    min_author_matches: int = REFERENCE_CHUNK_MIN_AUTHOR_MATCHES,
) -> bool:
    """True when *text* reads as a reference list rather than prose.

    The density (per 1,000 characters) of the five signal patterns
    (`_reference_chunk_pattern_count`) must exceed *density_threshold*. For a chunk at
    or above `REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH`,
    density alone is not enough: *text* must also show at least one of two independent
    bibliography-structure signals -- *min_author_matches* `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE`
    ("Smith, J. A."-style, APA) matches, OR `REFERENCE_CHUNK_MIN_NUMBERED_ENTRIES`
    `_reference_chunk_numbered_entry_count` (numbered-list, Vancouver/IEEE-style,
    signal-gated per line) matches -- since a
    citation-dense narrative discussion (see `REFERENCE_CHUNK_MIN_AUTHOR_MATCHES`'s
    docstring) can clear the density bar on narrative years and page cites alone while
    having no bibliography-entry structure of either kind, and an ordinary numbered list in
    running prose (enumerated research questions, findings, or procedure steps) shares the
    numbered-entry signal's shape without being a bibliography at all. Below the length
    floor, density alone decides: an unconditional floor would wrongly exempt short
    front-matter and back-matter chunks, and a single APA-only structure signal would
    wrongly exempt long non-APA bibliographies -- see `REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH`
    and `_reference_chunk_numbered_entry_count`. Chunks shorter than *min_length* are never
    flagged.
    """
    stripped = str(text or "").strip()
    if len(stripped) < min_length:
        return False
    if (
        len(stripped) >= REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
        and len(_REFERENCE_CHUNK_INITIAL_AUTHOR_RE.findall(stripped)) < min_author_matches
        and _reference_chunk_numbered_entry_count(stripped) < REFERENCE_CHUNK_MIN_NUMBERED_ENTRIES
    ):
        return False
    density = _reference_chunk_pattern_count(stripped) / (len(stripped) / 1000.0)
    return density > density_threshold


def drop_reference_and_backmatter_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[list[dict], int]:
    """(kept_chunks, dropped_count): drop reference-list and back-matter chunks.

    Drops any chunk whose reference-pattern density exceeds the threshold, regardless of
    its own section label, plus every chunk from the first bare
    References/Bibliography/Works Cited heading onward (that heading's own chunk is
    dropped too, even if short). Kept chunks preserve their original relative order.
    """
    kept: list[dict] = []
    dropped = 0
    heading_seen = False
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        if not heading_seen and _REFERENCES_HEADING_START_RE.match(text.strip()):
            heading_seen = True
        if heading_seen or is_reference_or_frontmatter_chunk(text):
            dropped += 1
            continue
        kept.append(dict(chunk))
    return kept, dropped


def _extract_text_from_tiptap(content: dict) -> str:
    """Recursively extract plain text from a Tiptap JSON document."""
    parts: list[str] = []

    def _walk(node: dict | list) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        for child in node.get("content", []):
            _walk(child)

    _walk(content)
    return " ".join(parts)


# Citation pattern: [Author, Year] or (Author, Year) or (Author et al., Year)
_CITATION_RE = re.compile(
    r"[\[\(]"
    r"([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?)"
    r",?\s*"
    r"(\d{4})"
    r"[\]\)]"
)

#: Sentence splitter used only by `_extract_claims`. The naive `(?<=[.!?])\s+` split treats
#: the period in the "et al." abbreviation as a sentence end whenever it is directly
#: followed by whitespace, which a parenthetical citation such as "(Smith et al., 2020)"
#: never triggers (the period is followed by a comma, not whitespace) but a narrative
#: citation such as "Smith et al. (2020) found ..." always does -- the split lands between
#: "et al." and "(2020)", so the surname and the year end up in two different sentence
#: fragments and `_NARRATIVE_CITE` (searched per fragment, not over the whole text) can
#: never see both halves at once.
#:
#: An earlier version of this fix blocked the split whenever the text immediately before
#: it was a standalone "al." token, regardless of what followed. That over-blocks: "This
#: was demonstrated by Smith et al. The replication failed (Jones, 2021)." has an "al."
#: that genuinely ends a sentence, so the old rule merged both sentences into one claim
#: whose claim_text the verifier was never asked about. The rule now looks forward instead
#: of backward: `(?!\(\s*(?:\d{4}|n\.d\.|in\s+press))` is a negative lookahead that blocks
#: the split only when the text right after the whitespace is a parenthesised year or
#: year-placeholder, which is the one shape a narrative citation's "et al. (Year)" ever
#: takes. "Smith et al. (2020)" stays whole (blocked, since "(2020" follows); "Smith et
#: al. The replication ..." still splits (allowed, since "The" follows, not "("); ordinary
#: sentence endings such as "deal." or "medal." split as before, since nothing about them
#: is special to this rule at all now. Not a general-purpose abbreviation-aware sentence
#: splitter: a narrative citation whose "et al." is not immediately followed by a
#: parenthesised year still merges with the next sentence, same as any other period-based
#: splitter's known "et al." limitation.
_CLAIM_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?!\(\s*(?:\d{4}|n\.d\.|in\s+press))"
)

#: Lower-case abbreviation `_CLAIM_SENTENCE_SPLIT_RE` must not treat as a sentence end
#: even though nothing parenthesised follows it. "et al." is already handled by the regex's
#: own forward lookahead, because a narrative citation's year is always parenthesised
#: right after it; "vs." has no such marker -- it is followed by a plain number or word,
#: as in "Luo et al. (2025) found ChatGPT's precision exceeded Grammarly's (94-98% vs.
#: 85%) ...", where the lookahead never fires and the sentence was cut in two, so a
#: cited claim built from the first half was truncated mid-clause and the second half
#: was never linked to a citation at all. Checked against the single word immediately
#: before the split point (`_blocked_by_abbreviation`), not folded into the regex itself,
#: since a fixed-width look-behind cannot express "one of several words of different
#: lengths". Not a general abbreviation-aware splitter (the same limitation
#: `_CLAIM_SENTENCE_SPLIT_RE`'s own docstring already accepts for "al."): a sentence that
#: genuinely ends on one of these words is merged with the next one instead.
_ABBREVIATIONS_NOT_SENTENCE_FINAL = frozenset({"vs"})


def _blocked_by_abbreviation(text: str, split_start: int) -> bool:
    """True when the word right before *split_start* -- the point
    `_CLAIM_SENTENCE_SPLIT_RE` would otherwise split *text* at -- is one of
    `_ABBREVIATIONS_NOT_SENTENCE_FINAL` plus its own period, so the split is a false
    sentence boundary rather than a genuine one."""
    word = re.search(r"([A-Za-z]+)\.$", text[:split_start])
    return word is not None and word.group(1).lower() in _ABBREVIATIONS_NOT_SENTENCE_FINAL


def _sentence_split_points(text: str) -> list[re.Match[str]]:
    """Every `_CLAIM_SENTENCE_SPLIT_RE` match in *text* that is a real sentence
    boundary: the false ones `_blocked_by_abbreviation` recognises are filtered out
    here, once, so `_extract_claims` and `_sentence_fragments` cannot disagree about
    where a sentence ends."""
    return [
        match
        for match in _CLAIM_SENTENCE_SPLIT_RE.finditer(text)
        if not _blocked_by_abbreviation(text, match.start())
    ]


def _split_on_sentence_boundaries(text: str) -> list[str]:
    """`_CLAIM_SENTENCE_SPLIT_RE.split(text)`, with the false boundaries
    `_blocked_by_abbreviation` recognises removed first."""
    sentences: list[str] = []
    cursor = 0
    for match in _sentence_split_points(text):
        sentences.append(text[cursor : match.start()])
        cursor = match.end()
    sentences.append(text[cursor:])
    return sentences


#: One capitalised name-shaped token inside an already-matched `_NARRATIVE_CITE` span.
#: Compiled from `citation_audit._SURNAME`
#: itself, not a second, independently written pattern, for the same reason
#: `_NARRATIVE_CITE` is reused rather than redefined: what counts as "a name" cannot be
#: allowed to drift between the two modules.
_NARRATIVE_NAME_TOKEN_RE = re.compile(_SURNAME)


def _resolve_narrative_citation_key(
    match: re.Match[str], year: str, paper_lookup: dict[str, Paper] | None
) -> str:
    """Pick the surname to key one `_NARRATIVE_CITE` match on.

    `citation_audit._AUTHOR_TAIL` accepts a comma-separated co-author list, so the
    surname group it captures can start one token to the left of the real first author
    whenever a capitalised lead-in word ("However,", "In China,") is directly followed by
    a comma and then a two- or three-author narrative citation -- `match.group("surname")`
    resolves to the lead-in word ("However"), and the real first author ("Zhang") is
    never emitted as a key at all, so the cited paper is never looked up.

    When a paper library is available, every capitalised name-shaped token inside the
    whole matched span (`_NARRATIVE_NAME_TOKEN_RE`, in the order it appears -- the
    lead-in word if any, then each author) is tried against `paper_lookup`, and the first
    one whose `{token}_{year}` key is present wins: a citation the citation audit already
    calls "matched" against the library is then keyed to literally the paper it matches,
    not to whichever capitalised word happened to come first in the sentence. This keeps
    "Zhang, Li and Wang (2021)" on `zhang_2021` (the first token found is already the
    first author, so no lead-in word makes it past the check) and moves "However, Zhang
    and Li (2019)" from `however_2019` to `zhang_2019`.

    If no candidate is present in `paper_lookup` (no library was passed, or the citation
    is genuinely not in the project's library), the original rule applies unchanged: the
    *last* whitespace-separated token of `match.group("surname")`, lower-cased -- the same
    token `_build_paper_lookup` indexes a paper's first author under, so "Van Dijk" still
    resolves to "dijk" and a citation with no capitalised lead-in word is unaffected
    either way.
    """
    if paper_lookup:
        for candidate in _NARRATIVE_NAME_TOKEN_RE.finditer(match.group(0)):
            last_name = candidate.group(0).split()[-1].lower()
            if f"{last_name}_{year}" in paper_lookup:
                return last_name
    return match.group("surname").split()[-1].lower()


def _extract_claims(
    text: str, paper_lookup: dict[str, Paper] | None = None
) -> list[tuple[str, str, str]]:
    """Extract (sentence, citation_key, citation_text) triples from text.

    A claim is a sentence that contains a citation. Two citation shapes are recognised,
    searched independently per sentence and merged in left-to-right order so a sentence
    citing two papers, one of each shape, still yields two claims in the order they occur:

    * parenthetical -- ``[Author, Year]`` or ``(Author, Year)`` / ``(Author et al., Year)``
      / ``(Author & Other, Year)`` (`_CITATION_RE`);
    * narrative -- ``Author (Year)``, ``Author et al. (Year)``, ``Author and Other
      (Year)``, ``Author, Other and Third (Year)`` and the possessive ``Author's (Year)``,
      the forms the writing agent's own citation style produces alongside the parenthetical
      one. Recognised with `app.services.citation_audit`'s own `_NARRATIVE_CITE` pattern,
      not a second, independently written regex, so a citation the citation audit counts as
      "matched" against the library is resolved to the identical (surname, year) pair here
      -- the two modules' notion of "the same citation" cannot drift apart. A placeholder
      year (``n.d.``, ``in press``) is skipped: `_build_paper_lookup` never indexes a paper
      under anything but a real four-digit year, so no such claim could ever match a paper.

    ``paper_lookup``, when given, is used only to
    disambiguate a narrative citation's key: see `_resolve_narrative_citation_key`. It is
    the same dict `_build_paper_lookup` returns; both call sites already build it right
    next to their `_extract_claims` call. Passing ``None`` (the default) uses the *last*
    token of `_NARRATIVE_CITE`'s own ``surname`` group directly, so a narrative citation with
    no capitalised lead-in word is unaffected either way, and the parenthetical
    path never consults ``paper_lookup`` at all.

    A narrative claim whose resolved key is absent from ``paper_lookup`` (a capitalised
    prose word directly followed by a parenthesised year, with no real citation there at
    all) is still emitted, deliberately: the
    parenthetical path already emits a claim for any bracketed citation not in the
    library (`_verify_one` returns ``no_full_text``, "Could not match citation to a paper
    in the library"), so dropping only the narrative false positives here would make the
    two citation shapes disagree about what a claim is, for a case neither shape can
    actually tell apart from a real absent-from-library citation without the same
    library lookup this function already has to make anyway. The tradeoff -- a handful of
    extra `no_full_text` rows lowering `full_text_coverage`'s denominator slightly -- is
    an accepted cost of this design.

    The citation_key is 'lastname_year' for matching against paper_lookup: the *last*
    token of the matched surname, lower-cased, exactly the token `_build_paper_lookup`
    indexes a paper's first author under (`first_author.split(",")[0].split()[-1]`) -- for
    a single-token surname (the common case, and the only case `_CITATION_RE` itself can
    ever capture) this is the same token as the first, so the parenthetical path's key
    values are unchanged; a narrative citation's surname group can carry a leading
    particle ("van Dijk"), where the two conventions diverge and the last token is the one
    that agrees with `_build_paper_lookup`.

    citation_text is the raw matched text (e.g. "(Author, Year)" or "Author et al.
    (Year)"), kept for the claim-verification record export (app.services.claim_record).

    The returned list is de-duplicated on (sentence, citation_key): a source cited both
    narratively and parenthetically in the same sentence ("Zhang (2018) found X
    (Zhang, 2018).") would otherwise reach the verifier as two identical claims, costing
    two model calls and two report/workbook rows for the same input. The first
    citation_text encountered (left-to-right) is kept.
    """
    sentences = _split_on_sentence_boundaries(text)
    claims: list[tuple[str, str, str]] = []
    for sentence in sentences:
        found: list[tuple[int, str, str]] = []
        for match in _CITATION_RE.finditer(sentence):
            author_part = match.group(1)
            year = match.group(2)
            # Extract last name (first word of author part)
            last_name = author_part.split()[0].lower()
            found.append((match.start(), f"{last_name}_{year}", match.group(0)))
        for match in _NARRATIVE_CITE.finditer(sentence):
            year = _year_key(match.group("year"))
            if year is None:
                continue
            last_name = _resolve_narrative_citation_key(match, year, paper_lookup)
            found.append((match.start(), f"{last_name}_{year}", match.group(0)))
        for _start, citation_key, citation_text in sorted(found, key=lambda item: item[0]):
            claims.append((sentence.strip(), citation_key, citation_text))

    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for sentence_text, citation_key, citation_text in claims:
        dedup_key = (sentence_text, citation_key)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append((sentence_text, citation_key, citation_text))
    return deduped


# --------------------------------------------------------------------------------------
# Document-level extraction that consumes the writer's own citation-link map first, falls
# back to the same author-year regexes `_extract_claims` uses for a citation no link
# claimed, and resolves a numbered citation (`[12]`, `(12)`, a superscript digit run)
# against the draft's own reference list as a last resort. `_extract_claims` itself is
# unchanged and keeps serving `verify_user_edits`, which only ever receives plain strings
# (a changed section's text), never a Tiptap document; the document path does not call it
# directly, since it needs each regex match's position rather than the per-sentence claim
# list.
# --------------------------------------------------------------------------------------

#: A heading that starts a reference list: "References", "Bibliography", "Works Cited"
#: (any case, optional trailing colon) or the Chinese equivalent "参考文献". Matched
#: against a heading NODE's whole text (short, typically one to three words), unlike
#: `_REFERENCES_HEADING_START_RE` above, which matches the start of a much longer
#: full-text CHUNK.
_REFERENCE_LIST_HEADING_RE = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited|参考文献)\s*[:：]?\s*$",
    re.IGNORECASE,
)

#: Public delegation, no behaviour
#: change: `app.services.citation_render` drops an existing reference section with the
#: identical heading match this module uses, so a document never ends up with two.
REFERENCE_LIST_HEADING_RE = _REFERENCE_LIST_HEADING_RE

#: A flat paragraph opening a numbered reference entry: "12. Smith, J. ...",
#: "[12] Smith, J. ...", "12) Smith, J. ...".
_NUMBERED_REFERENCE_PARAGRAPH_RE = re.compile(r"^\s*\[?(\d{1,3})[.\)\]]\s+(.*)$", re.DOTALL)

#: Unicode superscript digits, as a citation-number run (e.g. "text shown¹²." -> "12").
_SUPERSCRIPT_DIGITS_RE = re.compile(r"[⁰¹²³⁴-⁹]+")
_SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
#: "[12]", "[3, 4]", "[5-7]" and its en-dash variant. Capped at 3 digits per number (a
#: reference list running into four digits of entries is not a case this resolver needs
#: to support).
_BRACKET_NUMBERED_CITE_RE = re.compile(
    r"\[\s*\d{1,3}(?:\s*(?:,|-|–)\s*\d{1,3})*\s*\]"
)
#: "(12)" -- the weakest numbered form: likeliest to catch a
#: plain parenthetical number in prose, e.g. a sample size. Gated at the call site on the
#: reference-list resolver being active AND the number being at or below the highest
#: entry number; capped at 3 digits so a bare 4-digit year such as "(2020)" can never
#: match this pattern at all.
_PAREN_NUMBERED_CITE_RE = re.compile(r"\(\s*(\d{1,3})\s*\)")
#: A DOI inside a reference entry's own text, for DOI-first resolution.
_REFERENCE_ENTRY_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")
#: The first plain 4-digit year in a reference entry's text (not a placeholder), used both
#: to cut the "first author" prefix off the entry and as the year half of its key.
_REFERENCE_ENTRY_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _doc_normalize_ws(text: str) -> str:
    """Collapse whitespace runs to one space, for a sentence-presence comparison that
    ignores formatting-only differences. Mirrors
    `app.agents.citation_link_agent._normalize_ws` exactly; kept
    as a separate, tiny function here rather than imported, since it is a one-line pure
    string operation and this module does not otherwise depend on that agent module."""
    return " ".join((text or "").split())


def _link_citation_covers_fragment(link: dict, fragment_norm: str) -> bool:
    """True when *link*'s own ``citation_text`` (whitespace-normalised) actually occurs
    inside *fragment_norm*.

    An earlier version compared *link*'s whole ``sentence`` against the fragment instead,
    which is true of nearly every fragment whenever a
    link's sentence spans more than one -- a sentence the claim splitter breaks at an
    abbreviation such as "U.S." is the common trigger -- and suppressed a numbered
    citation in a fragment the link's own citation never touches. Comparing the
    citation text itself restricts the gate to the one fragment it is meant for."""
    citation_norm = _doc_normalize_ws(link.get("citation_text") or "")
    return bool(citation_norm) and citation_norm in fragment_norm


def _normalize_doi(raw: str | None) -> str | None:
    """``"https://doi.org/10.1/X"`` / ``"DOI:10.1/x"`` -> ``"10.1/x"``; ``None`` if empty."""
    if not raw:
        return None
    doi = raw.strip().lower()
    doi = re.sub(r"^doi:\s*", "", doi)
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.rstrip(".,;)") or None


def _build_doi_key_lookup(papers: Sequence[Paper]) -> dict[str, str]:
    """normalised DOI -> the same ``surname_year`` key `_build_paper_lookup` indexes this
    paper under, for DOI-first numbered-citation resolution."""
    lookup: dict[str, str] = {}
    for p in papers:
        doi_norm = _normalize_doi(getattr(p, "doi", None))
        if not doi_norm or not p.authors or not p.year:
            continue
        first_author = (
            p.authors[0] if isinstance(p.authors[0], str) else p.authors[0].get("name", "")
        )
        last_name = first_author.split(",")[0].split()[-1] if first_author else ""
        if last_name:
            lookup[doi_norm] = f"{last_name.lower()}_{p.year}"
    return lookup


def _top_level_nodes(content: dict | None) -> list[dict]:
    if not isinstance(content, dict):
        return []
    return [n for n in (content.get("content") or []) if isinstance(n, dict)]


#: Node types whose own ``content`` holds further nodes to walk into for claim
#: extraction -- list containers, list items, and blockquotes -- so a citation inside a
#: bulleted list or a blockquote is extracted exactly like a top-level paragraph, with
#: its own ``citationLinks`` attribute intact.
#: `PUT /drafts/{id}` accepts user-pasted content, which routinely carries such nodes,
#: and `_parse_reference_entries` already descends into `orderedList`/`bulletList` for
#: the numbered-citation resolver; this closes the other half of the same feature.
#: Headings are deliberately excluded: a heading node is never treated as a paragraph.
_CONTAINER_NODE_TYPES = frozenset({"bulletList", "orderedList", "listItem", "blockquote"})


def _iter_paragraph_nodes(nodes: list[dict]) -> Iterator[dict]:
    """Yield every paragraph node in *nodes*, recursing into any nesting of
    `_CONTAINER_NODE_TYPES` so a citation inside a list item or a blockquote is
    extracted exactly like a top-level paragraph."""
    for node in nodes:
        node_type = node.get("type")
        if node_type == "paragraph":
            yield node
        elif node_type in _CONTAINER_NODE_TYPES:
            yield from _iter_paragraph_nodes(
                [n for n in (node.get("content") or []) if isinstance(n, dict)]
            )


def _split_body_and_reference_nodes(nodes: list[dict]) -> tuple[list[dict], list[dict]]:
    """(body_nodes, reference_section_nodes): nodes at and after a References /
    Bibliography / Works Cited / 参考文献 heading are excluded from claim extraction and
    used only by the numbered-citation resolver -- in the plain-text pipeline they are
    still flattened into the extracted text along with everything else, so a reference
    entry can itself become a claim there. The reference section runs from just after
    that heading to the next heading (of any level) or the end of the document."""
    for i, node in enumerate(nodes):
        if node.get("type") != "heading":
            continue
        heading_text = _extract_text_from_tiptap(node).strip()
        if not _REFERENCE_LIST_HEADING_RE.match(heading_text):
            continue
        end = len(nodes)
        for j in range(i + 1, len(nodes)):
            if nodes[j].get("type") == "heading":
                end = j
                break
        return nodes[:i], nodes[i + 1 : end]
    return nodes, []


def _parse_reference_entries(nodes: list[dict]) -> dict[int, str]:
    """entry number -> its own raw text, from either a structured Tiptap list
    (``orderedList``/``bulletList``) or flat numbered paragraphs."""
    entries: dict[int, str] = {}
    for node in nodes:
        node_type = node.get("type")
        if node_type in ("orderedList", "bulletList"):
            start = 1
            attrs = node.get("attrs") or {}
            if node_type == "orderedList" and isinstance(attrs.get("start"), int):
                start = attrs["start"]
            for offset, item in enumerate(node.get("content") or []):
                text = _extract_text_from_tiptap(item).strip()
                if text:
                    entries[start + offset] = text
        elif node_type == "paragraph":
            text = _extract_text_from_tiptap(node).strip()
            match = _NUMBERED_REFERENCE_PARAGRAPH_RE.match(text)
            if match:
                entries[int(match.group(1))] = match.group(2).strip()
    return entries


def _numbering_is_contiguous_from_one(entries: dict[int, str]) -> bool:
    if not entries:
        return False
    numbers = sorted(entries)
    return numbers == list(range(1, len(numbers) + 1))


def _unfolded_surname_key(name: str) -> str | None:
    """The same surname token `citation_audit._surname_key` selects, without its NFKD
    diacritic fold: `_build_paper_lookup` keys
    a paper on ``first_author.split(",")[0].split()[-1].lower()``, which keeps
    diacritics ("García, M." -> "garcía_2020"), while `_surname_key` folds them away
    ("garcia_2020") -- the two never meet for a numbered citation unless this unfolded
    form is also tried. Deliberately simpler than `_surname_key`'s no-comma branch (no
    trailing-initials stripping): every reference entry this resolver has ever been
    given is "Surname, Initials. (Year) ...", and this is only a secondary candidate
    tried after the folded key misses."""
    name = name.strip()
    if not name:
        return None
    surname = name.split(",", 1)[0].strip() if "," in name else name
    tokens = surname.split()
    if not tokens:
        return None
    return tokens[-1].lower()


def _resolve_reference_entry_key(
    entry_text: str, doi_key_lookup: dict[str, str], paper_lookup: dict[str, Paper]
) -> str | None:
    """The ``surname_year`` key for one reference-list entry: by its own DOI first
    (normalised, looked up against the project library's papers), else by the entry's
    first author surname (`citation_audit._surname_key`, applied to the text before the
    entry's own first year) and that year -- the identical key shape
    `_build_paper_lookup` indexes papers under, whether or not this particular key turns
    out to be present in the library (absence from the
    library is a coverage fact, not a reason to withhold a key). When the folded key is
    not itself present in ``paper_lookup``, the unfolded form (`_unfolded_surname_key`)
    is tried too, and used instead when only it matches --
    `_build_paper_lookup` does not fold diacritics, so a numbered citation
    to "García, M." must be able to find the library's "garcía_2020" key, not only the
    folded "garcia_2020" this function would otherwise always return. Returns ``None``
    only when the entry's own text carries neither a DOI this library recognises nor a
    recognisable author/year pair at all."""
    doi_match = _REFERENCE_ENTRY_DOI_RE.search(entry_text)
    if doi_match:
        key = doi_key_lookup.get(_normalize_doi(doi_match.group(0)))
        if key:
            return key
    year_match = _REFERENCE_ENTRY_YEAR_RE.search(entry_text)
    if year_match:
        year = year_match.group(0)
        author_text = entry_text[: year_match.start()]
        folded = _surname_key(author_text)
        if folded:
            folded_key = f"{folded}_{year}"
            if folded_key in paper_lookup:
                return folded_key
            unfolded = _unfolded_surname_key(author_text)
            if unfolded and unfolded != folded:
                unfolded_key = f"{unfolded}_{year}"
                if unfolded_key in paper_lookup:
                    return unfolded_key
            return folded_key
    return None


def _expand_bracket_numbers(token: str) -> list[int]:
    """``"[3, 4]"`` -> ``[3, 4]``; ``"[5-7]"`` / en-dash variant -> ``[5, 6, 7]``;
    ``"[12]"`` -> ``[12]``."""
    inner = token.strip()[1:-1]
    numbers: list[int] = []
    for part in inner.split(","):
        part = part.strip()
        range_match = re.match(r"^(\d{1,3})\s*(?:-|–)\s*(\d{1,3})$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end:
                numbers.extend(range(start, end + 1))
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


#: The alnum token immediately preceding a superscript-digit run, used to tell a
#: Vancouver-style citation superscript from an ordinary mathematical exponent.
_PRECEDING_ALNUM_TOKEN_RE = re.compile(r"[A-Za-z0-9]+$")
#: A superscript sign (U+207B minus, U+207A plus) directly before a superscript-digit
#: run is always the exponent's own sign, never part of a citation number:
#: `_SUPERSCRIPT_DIGITS_RE` matches digits only, so
#: "10⁻³" left the text right before the digit run ending in the superscript minus,
#: which `_PRECEDING_ALNUM_TOKEN_RE` (`[A-Za-z0-9]+$`) cannot match at all -- the guard
#: returned `False` and "p < 10⁻³" and "5 x 10⁻⁶" were each counted as a citation
#: (numbers 3 and 6), including once alongside a genuine "[1]" with a one-entry
#: reference list, where the bogus number exceeded `max_entry` and switched the whole
#: resolver off, reporting the genuine "[1]" unresolved too. Stripped here, from the
#: text being looked back over, before the base-token test runs at all.
_SUPERSCRIPT_SIGN_RE = re.compile(r"[⁻⁺]+$")
#: The base test is shape based, not length based: a single letter ("R²") or a run of
#: digits ("10⁻³") is an exponent's own base unambiguously, and so is a two-letter unit
#: abbreviation ("km²", "mm²", "Hz²", "dB²", "kb²"). An enumerated list of two-letter
#: units would miss others such as "mg", "Hz", "kb", "dB", "um", each of which would
#: otherwise collect a bogus cited number that could switch a working numbered resolver
#: off for a whole document; the shape rule instead covers any one- or two-letter token,
#: which costs only a genuine citation superscript written directly after a two-letter
#: word. "chi²" stays in the allowlist, being three letters. The micro-sign entry the
#: list used to carry was unreachable and is gone: `_PRECEDING_ALNUM_TOKEN_RE` is
#: `[A-Za-z0-9]+$`, which matches neither the micro sign nor a Greek mu, so the token a
#: "µm²" value ever hands this test is "m".
_EXPONENT_BASE_SHAPE_RE = re.compile(r"^(?:[A-Za-z]{1,2}|\d+)$")
_EXPONENT_BASE_ALLOWLIST = frozenset({"chi"})


def _is_exponent_base_shape(token: str) -> bool:
    """True when *token* is shaped like a mathematical exponent's own base: a one- or
    two-letter token, a run of digits, or one of the names in
    `_EXPONENT_BASE_ALLOWLIST`."""
    if _EXPONENT_BASE_SHAPE_RE.fullmatch(token):
        return True
    return token.lower() in _EXPONENT_BASE_ALLOWLIST


def _looks_like_a_mathematical_exponent(text: str, match_start: int) -> bool:
    """True when the superscript run starting at *match_start* in *text* is a
    mathematical exponent rather than a Vancouver citation superscript.

    Either the run is directly preceded by a superscript sign -- "10⁻³" is an exponent
    whatever precedes the "10" -- or it sits
    directly after a token shaped like an exponent's own base
    (`_is_exponent_base_shape`)."""
    preceding_text = text[:match_start]
    unsigned = _SUPERSCRIPT_SIGN_RE.sub("", preceding_text)
    if unsigned != preceding_text:
        return True
    preceding = _PRECEDING_ALNUM_TOKEN_RE.search(unsigned)
    return bool(preceding) and _is_exponent_base_shape(preceding.group(0))


def _find_numbered_citations(
    text: str, *, max_entry: int | None, allow_bare_paren: bool
) -> list[tuple[int, list[int], str]]:
    """``(start, numbers, raw_token)`` for every numbered-citation form recognised in
    *text*. The bracket form is unambiguous citation notation
    and always considered; a superscript-digit run is considered unless it reads as a
    mathematical exponent (`_looks_like_a_mathematical_exponent`); the bare ``"(12)"``
    form is gated by *allow_bare_paren* and,
    additionally, by *max_entry* (fires only at or below the highest reference-list
    entry number -- see the module-level note on `_PAREN_NUMBERED_CITE_RE`)."""
    found: list[tuple[int, list[int], str]] = []
    for match in _BRACKET_NUMBERED_CITE_RE.finditer(text):
        numbers = _expand_bracket_numbers(match.group(0))
        if numbers:
            found.append((match.start(), numbers, match.group(0)))
    for match in _SUPERSCRIPT_DIGITS_RE.finditer(text):
        if _looks_like_a_mathematical_exponent(text, match.start()):
            continue
        digits = match.group(0).translate(_SUPERSCRIPT_TRANSLATION)
        found.append((match.start(), [int(digits)], match.group(0)))
    if allow_bare_paren:
        for match in _PAREN_NUMBERED_CITE_RE.finditer(text):
            number = int(match.group(1))
            if max_entry is not None and number <= max_entry:
                found.append((match.start(), [number], match.group(0)))
    return sorted(found, key=lambda item: item[0])


def _numbering_covers_body(
    body_nodes: list[dict], max_entry: int
) -> bool:
    """True when every number cited anywhere in *body_nodes*, in any recognised numbered
    form, is at or below *max_entry* (the resolver switches on
    only when the reference list "covers every number cited in the body"). A document
    that cites `[15]` with only 10 reference entries never turns the resolver on at all --
    every numbered citation in it is reported unresolved instead of guessed. Recurses
    into list items and blockquotes (`_iter_paragraph_nodes`), same as the claim
    extraction below."""
    cited_numbers: set[int] = set()
    for node in _iter_paragraph_nodes(body_nodes):
        text = _extract_text_from_tiptap(node)
        for _pos, numbers, _token in _find_numbered_citations(
            text, max_entry=max_entry, allow_bare_paren=True
        ):
            cited_numbers.update(numbers)
    return not cited_numbers or max(cited_numbers) <= max_entry


# --------------------------------------------------------------------------------------
# This region counts ONE thing, the citation unit, rather than three different things at
# once (the ranges of a model-returned citation_text, the ranges of a regex match, and
# the positions the citation audit reports), and every other step is a question about a
# unit.
#
# A citation unit is one citation occurrence in one paragraph:
#
# * every span `citation_audit.citation_spans` reports (the authoritative answer to "how
#   many citations are in this text"), each given its own character extent: a member of a
#   parenthetical group gets the ";"-delimited segment it sits in, so
#   "(Lee, 2020; Storch, 2018)" is two units with two disjoint extents, and a narrative
#   citation gets the range of the citation audit's own match, from the first surname
#   token to the closing bracket of its year;
# * plus any `_CITATION_RE` citation the citation audit does not recognise at all -- the
#   square-bracket form "[Jones, 2021]" above all -- so a citation this module has always
#   claimed keeps being claimed and keeps being counted;
# * plus one unit per number inside a numbered citation ("[1, 2]" is two units), scanned
#   only in a sentence fragment that carries no author-year unit and no surviving link.
#
# Resolution happens once per paragraph, never once per sentence fragment (that
# per-fragment repetition is what counted a link twice when the claim splitter broke its
# sentence in two):
#
# 1. every surviving link with at least one key is matched, whitespace normalised on both
#    sides, to the occurrence of its own citation_text that overlaps an unconsumed unit;
#    that unit is consumed and carries the link's keys. A link that matches no unconsumed
#    unit is ignored and counted in `diagnostics`; a link with no keys consumes nothing,
#    so its unit stays available to the fallback;
# 2. the author-year regexes (`_CITATION_RE`, `_NARRATIVE_CITE`) run once over the
#    paragraph and each match is assigned, by overlap, to an unconsumed unit;
# 3. a unit still unconsumed is reported unresolved under its own rendered text.
#
# Claims are emitted per (unit, key): the claim sentence is the sentence fragment that
# holds the unit's extent, so it is always the document's own text. Two units with the
# same key in the same fragment (one source cited twice in one sentence) ask the verifier
# the identical question, so they collapse into one claim while still counting as two
# citations.
# --------------------------------------------------------------------------------------

#: Debug flag for the accounting invariants below.
#: The checks are a handful of integer comparisons over lists the function has already
#: built, so they are on by default; set to ``False`` to switch them off without touching
#: the algorithm.
CITATION_COVERAGE_INVARIANT_CHECKS = True

#: How many unresolved citations are listed for a human to spot check. The `unresolved`
#: count itself is never capped.
UNRESOLVED_CITATION_LIST_CAP = 50


@dataclass
class _CitationUnit:
    """One citation occurrence: the single unit this module's coverage accounts for.

    ``start``/``end`` are the extent in the paragraph's own text; ``rendered`` is the
    text a human is shown when the unit stays unresolved; ``kind`` is ``"author-year"``
    or ``"numbered"``; ``source`` is ``None`` until the unit is consumed, then
    ``"mapping"`` (a surviving link claimed it), ``"author-year"`` (a regex resolved it)
    or ``"numbered"`` (the reference-list resolver saw it, with or without a key);
    ``keys`` are the citation's author-year keys, ``citation_text`` the text carried into
    the claim record, and ``sentence`` the claim sentence. ``audit_key`` is the
    ``surname_year`` key `citation_audit.citation_spans` already computed for this exact
    unit, if any: the last resort when no link
    and no regex claims the unit at all. ``proposition`` is the citation-link's own
    narrowed span of ``sentence``, set only when a surviving link both named this unit and
    carried a non-empty ``proposition`` that occurs in the paragraph text; ``None`` for
    every unit a link did not resolve, and for a link-resolved unit whose link carried no
    usable proposition.

    ``extra_resolutions`` holds every
    ADDITIONAL ``(keys, proposition)`` pair a later link resolves to this same rendered
    occurrence once the unit's own primary slot (``keys``/``proposition`` above) is
    already taken. A citation rendered once but split by the citation-link step into two
    propositions against the same paper -- a sentence with two findings, one citation --
    sends two link entries whose own ``citation_text`` names the identical occurrence;
    without this, the second link matches no unconsumed unit and is silently dropped
    before the verifier ever sees it, which is the escape mechanism this function's own
    dedup keys must still guard against. One claim is still built per resolution, primary
    and extra alike, so the unit itself
    stays the thing `found`/`linked`/`unresolved`/`by_source` count -- one per rendered
    citation, never per proposition -- while every proposition it carries still reaches
    the verifier.
    """

    start: int
    end: int
    rendered: str
    kind: str
    source: str | None = None
    keys: list[str] = field(default_factory=list)
    audit_key: str | None = None
    citation_text: str = ""
    sentence: str = ""
    proposition: str | None = None
    extra_resolutions: list[tuple[list[str], str | None]] = field(default_factory=list)


def _find_all_spans(text: str, snippet: str) -> list[tuple[int, int]]:
    """Every ``(start, end)`` character range where *snippet* occurs verbatim inside
    *text*, left to right, none overlapping another. A citation rendered twice occurs
    twice here, so two links naming the identical text can each be matched to their own
    occurrence instead of both landing on the first one."""
    if not snippet:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(snippet, cursor)
        if start == -1:
            break
        spans.append((start, start + len(snippet)))
        cursor = start + len(snippet)
    return spans


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when character ranges *a* and *b* share at least one position."""
    return a[0] < b[1] and b[0] < a[1]


def _range_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """True when range *inner* lies entirely inside range *outer*."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _is_bracket_balanced(text: str) -> bool:
    """True when every ``(``/``)`` and ``[``/``]`` in *text* is matched, in order.

    `unit.rendered` is always balanced, since this module builds it from a citation's own
    extent; a model-returned `citation_text` is not. Checked both where `unit.citation_text`
    is set from a link (`_assign_links_to_units`) and again where a
    still-unresolved unit's text is displayed, so the unresolved list and the claim
    record built from the same unit always carry the same string. Never used to change
    whether the unit resolved."""
    depth_paren = depth_bracket = 0
    for character in text:
        if character == "(":
            depth_paren += 1
        elif character == ")":
            depth_paren -= 1
            if depth_paren < 0:
                return False
        elif character == "[":
            depth_bracket += 1
        elif character == "]":
            depth_bracket -= 1
            if depth_bracket < 0:
                return False
    return depth_paren == 0 and depth_bracket == 0


def _normalised_with_offsets(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed *text*, plus the raw index of every character kept.

    A link's ``citation_text`` is compared against the draft with whitespace normalised
    on both sides, exactly as `app.agents.citation_link_agent.validate_citation_link`
    compares it (V1/V2), so a link the validator accepted can always be located: a raw
    comparison missed "(Jones,  2021)" written with two spaces and left that link with no
    range at all, which is how one citation came to be counted twice. The offsets map a
    normalised range back onto the raw text a unit's extent lives in."""
    kept: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text):
        if character.isspace():
            if not kept or kept[-1] == " ":
                continue
            kept.append(" ")
            offsets.append(index)
            continue
        kept.append(character)
        offsets.append(index)
    while kept and kept[-1] == " ":
        kept.pop()
        offsets.pop()
    return "".join(kept), offsets


def _raw_range_to_normalised(offsets: list[int], raw: tuple[int, int]) -> tuple[int, int]:
    """The range in `_normalised_with_offsets`' output that covers raw range *raw*."""
    return bisect_left(offsets, raw[0]), bisect_left(offsets, raw[1])


def _semicolon_segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """``text[start:end]`` split on ";", as ranges in *text*'s own coordinates. A
    parenthetical group holding "(Lee, 2020; Storch, 2018)" yields two segments, so its
    two citations get two disjoint extents instead of both reconstructing to the whole
    group: one link on one member of a group would otherwise make every member of that
    group look covered."""
    segments: list[tuple[int, int]] = []
    cursor = start
    for index in range(start, end):
        if text[index] == ";":
            segments.append((cursor, index))
            cursor = index + 1
    segments.append((cursor, end))
    return segments


def _audit_span_units(text: str) -> list[_CitationUnit]:
    """One `_CitationUnit` per span `citation_audit.citation_spans` reports in *text*.

    `citation_spans` reports a position, a surname and a year, never an extent, so each
    span's extent is reconstructed here, once, from the two shapes the citation audit
    itself recognises:

    * inside a parenthetical group, the ";"-delimited segment the span sits in, widened
      to the group's own brackets at the ends of the group. When one segment holds more
      than one span ("(Smith, 2020 and Jones, 2021)"), each further span starts at its
      own position, so the extents inside a group never overlap each other;
    * otherwise the narrative match's own range, from the first surname token the audit
      matched (which may be a capitalised lead-in word, "However, Zhang and Li (2019)")
      to the closing bracket of its year.

    The rendered text of a group member is the member alone ("Storch, 2018"); a group
    with a single member keeps its brackets ("(Bonilla Lopez et al., 2018)"). Both are
    balanced, which a walk of a fixed number of characters to the left was not.

    Each unit also carries the ``surname_year`` key `citation_spans` already computed for
    its own position, as ``audit_key``: the last
    resort `_assign_author_year_fallback` reaches for when neither a link nor its own
    regexes ever claim the unit at all."""
    raw_spans = citation_spans(text)
    positions = sorted({position for position, _s, _y, _l in raw_spans})
    if not positions:
        return []
    audit_pairs = {position: (surname, year) for position, surname, year, _l in raw_spans}

    def _audit_key(position: int) -> str | None:
        surname, year = audit_pairs.get(position, (None, None))
        if not surname or not year:
            return None
        return f"{surname.split()[-1].lower()}_{year}"

    units: dict[int, _CitationUnit] = {}
    for group in _PAREN_GROUP.finditer(text):
        inner_start, inner_end = group.start(1), group.end(1)
        members = [p for p in positions if inner_start <= p < inner_end]
        if not members:
            continue
        for segment_start, segment_end in _semicolon_segments(text, inner_start, inner_end):
            segment_members = [p for p in members if segment_start <= p < segment_end]
            for index, position in enumerate(segment_members):
                start = segment_start if index == 0 else position
                if index + 1 < len(segment_members):
                    end = segment_members[index + 1]
                else:
                    end = segment_end
                if start == inner_start:
                    start = group.start()
                if end == inner_end:
                    end = group.end()
                rendered = text[start:end].strip()
                if len(members) > 1:
                    rendered = rendered.lstrip("([").rstrip(")]").strip()
                units[position] = _CitationUnit(
                    start=start,
                    end=end,
                    rendered=rendered,
                    kind="author-year",
                    audit_key=_audit_key(position),
                )
    narrative_ends = {match.start(): match.end() for match in _NARRATIVE_CITE.finditer(text)}
    for position in positions:
        if position in units:
            continue
        end = narrative_ends.get(position)
        if end is None:
            closing = text.find(")", position)
            end = closing + 1 if closing != -1 else len(text)
        units[position] = _CitationUnit(
            start=position,
            end=end,
            rendered=text[position:end].strip(),
            kind="author-year",
            audit_key=_audit_key(position),
        )
    return [units[position] for position in sorted(units)]


def citation_extents(text: str) -> list[tuple[int, int, str]]:
    """``(start, end, rendered)`` per `_audit_span_units` unit in *text*: a pure
    delegation, no behaviour change, exposing the same span reconstruction this module's
    coverage accounting already relies on so `app.services.citation_render` can resolve a
    chat answer's own citations to the identical spans a verdict would be keyed to."""
    return [(unit.start, unit.end, unit.rendered) for unit in _audit_span_units(text)]


def _regex_only_units(text: str, audit_units: list[_CitationUnit]) -> list[_CitationUnit]:
    """A unit for every `_CITATION_RE` citation the citation audit does not see.

    `citation_spans` looks only inside round parentheses and at the narrative form, so
    the square-bracket citation "[Jones, 2021]" (and a parenthetical written without its
    comma) is invisible to it while `_extract_claims` has always claimed it. Counting it
    as a unit is the one deliberate widening of "the unit is the audit span": without it
    a citation this module claims would carry no coverage entry at all. For a draft
    written in the author-year style the citation audit recognises -- everything the writer itself
    produces -- this function returns nothing and `found` is exactly the audit's own span
    count."""
    extras: list[_CitationUnit] = []
    for match in _CITATION_RE.finditer(text):
        span = (match.start(), match.end())
        if any(_ranges_overlap(span, (unit.start, unit.end)) for unit in audit_units):
            continue
        extras.append(
            _CitationUnit(
                start=span[0], end=span[1], rendered=match.group(0).strip(), kind="author-year"
            )
        )
    return extras


def _sentence_fragments(text: str) -> list[tuple[int, int]]:
    """The ranges `_CLAIM_SENTENCE_SPLIT_RE` splits *text* into (with
    `_blocked_by_abbreviation`'s false boundaries removed), keeping the offsets the
    plain `split()` throws away, so a unit can be told which fragment holds it."""
    fragments: list[tuple[int, int]] = []
    cursor = 0
    for separator in _sentence_split_points(text):
        fragments.append((cursor, separator.start()))
        cursor = separator.end()
    fragments.append((cursor, len(text)))
    return fragments


def _fragment_holding(text: str, fragments: list[tuple[int, int]], position: int) -> str:
    """The sentence fragment of *text* that holds *position*, stripped; the whole text
    when the position falls outside every fragment or its own fragment is blank."""
    for start, end in fragments:
        if start <= position < end:
            fragment = text[start:end].strip()
            if fragment:
                return fragment
            break
    return text.strip()


def _key_surname_year(key: str) -> tuple[str, str] | None:
    """``"lee_2020"`` -> ``("lee", "2020")``; ``None`` when *key* carries no year suffix.
    The surname is never itself hyphen/underscore-joined (every key this module builds is
    one lower-cased name token plus one four-digit year), so splitting on the last ``"_"``
    is exact, not a heuristic."""
    surname, separator, year = key.rpartition("_")
    return (surname, year) if separator and surname and year else None


def _key_matches_rendered(key: str, rendered: str) -> bool:
    """True when *key* plausibly names the citation *rendered* displays, for handing a
    group-wide link's keys to the right member of the group it covers: *key*'s surname
    as a case-insensitive substring, and its year verbatim, both present in *rendered*."""
    parts = _key_surname_year(key)
    if parts is None:
        return False
    surname, year = parts
    return surname.lower() in rendered.lower() and year in rendered


def _assign_links_to_units(
    text: str,
    units: list[_CitationUnit],
    links: list[dict],
    diagnostics: dict[str, int],
) -> None:
    """Consume one unit per surviving, keyed link.

    Each link is resolved exactly once against the whole paragraph, not once per sentence
    fragment: a link whose sentence the claim splitter breaks in two would otherwise be
    resolved twice and counted twice. Its ``citation_text`` is located with whitespace
    normalised on both sides; an occurrence inside the link's own sentence is an absolute
    preference over an occurrence of the same text elsewhere in the paragraph: whenever at
    least one occurrence of the link's own
    ``citation_text`` sits inside the link's own ``sentence``, the search never looks at
    an occurrence outside it, consumed or not. Without that preference, a citation
    rendered once per sentence but twice overall (two sentences of one paragraph each
    citing the same paper) could let a second link on the FIRST sentence, finding its own
    occurrence already consumed by an earlier link of that same sentence, fall through to
    the SECOND sentence's own occurrence and claim it there as a primary resolution --
    attaching one sentence's
    proposition to another sentence's unit, nulling it on re-validation and turning the
    claim into that other sentence's own text verbatim, citation included, while the
    original proposition never reached the verifier under its own name at all. Only when
    the link's own sentence carries no occurrence of its ``citation_text`` at all does the
    search fall through to an occurrence elsewhere in the paragraph. Within whichever set
    of occurrences applies, the first one whose range overlaps at least one unconsumed
    unit is used. A link that matches no unconsumed unit -- the same link returned twice,
    or a citation_text the paragraph no longer renders -- adds nothing and is counted in
    *diagnostics*.

    When that occurrence overlaps more than one unconsumed unit -- the model returned the
    whole parenthetical group as ``citation_text`` instead of one entry per citation --
    every unit it overlaps is consumed, and
    the link's keys are distributed across them by surname and year rather than all
    handed to whichever unit happens to come first: a two-key group resolves both
    members, and a one-key group attributes the key to the member it actually names
    instead of leaving that member reported unresolved while the other, unmentioned
    member is credited with it by accident.

    A link may also carry
    ``proposition``, a verbatim contiguous span of its own ``sentence``. Every unit the
    link consumes -- whether it is the single target or a member of a group -- is given
    that same proposition, once validated against *this paragraph's own text* (not just
    the link's sentence) with the same whitespace-insensitive comparison ``sentence``
    itself is checked with, since a proposition the paragraph no longer renders is exactly
    as stale as a sentence it no longer renders. This is only the first, coarse check: a
    unit does not yet know its own ``sentence`` (a fragment of the paragraph) at this
    point, so `extract_claims_from_document`'s caller re-checks the proposition against
    that fragment once it is set, and drops it back to `None` if the proposition does not
    actually live inside it -- an abbreviation such as "e.g." can split one link's own
    sentence into two fragments, and a proposition spanning the split is a real paragraph
    substring that is not a substring of the one fragment a given unit lands in."""
    normalised, offsets = _normalised_with_offsets(text)
    extents = [_raw_range_to_normalised(offsets, (unit.start, unit.end)) for unit in units]
    for link in links:
        keys = [key for key in (link.get("keys") or []) if key]
        if not keys:
            diagnostics["links_without_keys"] += 1
            continue
        proposition_stripped = (link.get("proposition") or "").strip()
        proposition = (
            proposition_stripped
            if proposition_stripped and _doc_normalize_ws(proposition_stripped) in normalised
            else None
        )
        citation_text = link.get("citation_text") or ""
        occurrences = _find_all_spans(normalised, _doc_normalize_ws(citation_text))
        sentence_ranges = _find_all_spans(
            normalised, _doc_normalize_ws(link.get("sentence") or "")
        )
        inside = [
            occurrence
            for occurrence in occurrences
            if any(_range_contains(sentence, occurrence) for sentence in sentence_ranges)
        ]
        outside = [occurrence for occurrence in occurrences if occurrence not in inside]
        # An occurrence inside the link's own sentence, once
        # any exists, is the ONLY place this link may resolve -- an occurrence outside
        # it is considered only when the link's own sentence carries no occurrence at
        # all. Without this restriction, searching `outside` whenever `inside` comes up
        # empty-handed (unconsumed AND consumed) could let a link fall through to a
        # different sentence's own occurrence and claim it as a primary resolution.
        group = inside if inside else outside
        targets: list[_CitationUnit] = []
        for occurrence in group:
            targets = [
                unit
                for index, unit in enumerate(units)
                if unit.source is None and _ranges_overlap(occurrence, extents[index])
            ]
            if targets:
                break
        reused = False
        if not targets:
            # No UNCONSUMED unit shares this
            # link's own rendered occurrence -- an earlier link already claimed the
            # physical citation, exactly the shape of a sentence the citation-link step
            # split into two propositions against the same paper. Reuse the unit(s) an
            # earlier link already claimed at
            # that same position as an EXTRA resolution rather than dropping this link:
            # the unit still counts once for `found`/`linked`/`by_source`, and this
            # link's own (keys, proposition) still reaches the claims-building loop
            # below as its own resolution. Still restricted to `group`: a consumed
            # occurrence inside the link's own sentence is preferred over an unconsumed
            # OR consumed occurrence outside it.
            for occurrence in group:
                targets = [
                    unit
                    for index, unit in enumerate(units)
                    if unit.source is not None and _ranges_overlap(occurrence, extents[index])
                ]
                if targets:
                    reused = True
                    break
        if not targets:
            diagnostics["links_ignored"] += 1
            continue
        if len(targets) == 1:
            target = targets[0]
            # A model-returned citation_text is not run
            # through the balance check that display and the workbook row both rely on
            # to see the same string. Falling back to `target.rendered` -- always
            # balanced, since it is built from the citation's own extent -- here too
            # means the unresolved list and the claim record never diverge.
            stripped = citation_text.strip()
            resolved_citation_text = (
                stripped if stripped and _is_bracket_balanced(stripped) else target.rendered
            )
            if reused:
                target.extra_resolutions.append((list(keys), proposition))
            else:
                target.source = "mapping"
                target.keys = list(keys)
                target.proposition = proposition
                target.citation_text = resolved_citation_text
            continue
        remaining_keys = list(keys)
        any_matched = False
        for target in targets:
            matched = [key for key in remaining_keys if _key_matches_rendered(key, target.rendered)]
            if not matched:
                continue
            any_matched = True
            if reused:
                target.extra_resolutions.append((matched, proposition))
            else:
                target.source = "mapping"
                target.keys = matched
                target.citation_text = target.rendered
                target.proposition = proposition
            for key in matched:
                remaining_keys.remove(key)
        if not any_matched:
            # The single-target branch above counts a
            # link that claims nothing in `links_ignored`; without this, every target
            # would stay unconsumed and the loop would end without incrementing anything
            # when none of the group-wide link's keys matched any member of the group it
            # covers.
            diagnostics["links_ignored"] += 1


def _assign_author_year_fallback(
    text: str, units: list[_CitationUnit], paper_lookup: dict[str, Paper] | None
) -> None:
    """Resolve the units no link consumed with the pre-existing author-year regexes.

    Both regexes run once over the paragraph and every match is assigned to the
    unconsumed unit its own range overlaps. Position is the only thing the two sides can
    be relied on to agree about: `citation_audit` and `_extract_claims` resolve the same
    citation to different (surname, year) pairs whenever a capitalised lead-in word is
    absorbed into the audit's surname group ("However, Zhang and Li (2019)"), which is
    what made a pair-based comparison drop real claims."""
    if not any(unit.source is None for unit in units):
        return
    matches: list[tuple[int, int, str, str]] = []
    for match in _CITATION_RE.finditer(text):
        last_name = match.group(1).split()[0].lower()
        matches.append(
            (match.start(), match.end(), f"{last_name}_{match.group(2)}", match.group(0))
        )
    for match in _NARRATIVE_CITE.finditer(text):
        year = _year_key(match.group("year"))
        if year is None:
            continue
        last_name = _resolve_narrative_citation_key(match, year, paper_lookup)
        matches.append((match.start(), match.end(), f"{last_name}_{year}", match.group(0)))
    for start, end, key, citation_text in sorted(matches):
        for unit in units:
            if unit.source is None and _ranges_overlap((start, end), (unit.start, unit.end)):
                unit.source = "author-year"
                unit.keys = [key]
                unit.citation_text = citation_text.strip()
                break


def _assign_audit_key_fallback(units: list[_CitationUnit]) -> None:
    """Key a unit from `citation_audit.citation_spans`'s own (surname, year) when neither
    a link nor `_assign_author_year_fallback`'s regexes ever claimed it.

    `_CITATION_RE` and `citation_audit._NARRATIVE_CITE` recognise fewer shapes than
    `citation_spans` does: a semicolon-group member, a two-token surname, an "as cited
    in" attribution and a "(see also X, Y)" lead-in are all citations the audit parses
    completely into a surname and a year, yet none of them can ever be matched by the
    fallback's own regexes, so without this fallback they would reach `unresolved` with
    no key and no claim even when the project library held the paper. The lead-in-word
    ambiguity that the
    unit's own position exists to avoid cannot arise here: a narrative citation carrying
    a capitalised lead-in word always has a `_NARRATIVE_CITE` match that overlaps its
    unit and claims it first, so this function only ever runs for a unit that regex
    genuinely cannot express, and `audit_key` is already the correct surname for that
    shape. A unit with no audit key (a placeholder year such as "n.d.") is left exactly
    as it was: still unresolved, with no key to send."""
    for unit in units:
        if unit.source is None and unit.audit_key:
            unit.source = "author-year"
            unit.keys = [unit.audit_key]
            unit.citation_text = unit.rendered


def _numbered_citation_units(
    text: str,
    fragments: list[tuple[int, int]],
    author_year_units: list[_CitationUnit],
    valid_links: list[dict],
    *,
    numbering_active: bool,
    max_entry: int | None,
    entry_key_lookup: dict[int, str | None],
) -> list[_CitationUnit]:
    """One unit per number cited in a numbered form ("[12]", "(12)", a superscript run),
    in the sentence fragments that carry no author-year citation and no surviving link.

    "[1, 2]" is two units, because the two numbers resolve against the reference list
    independently and an unresolvable one must stay visible in coverage. Unambiguous
    forms are scanned whether or not the resolver is active, so a numbered citation the
    reference list cannot resolve is reported unresolved rather than dropped; only the
    bare "(12)" form is withheld entirely when the resolver is inactive."""
    units: list[_CitationUnit] = []
    for fragment_start, fragment_end in fragments:
        fragment = text[fragment_start:fragment_end]
        fragment_norm = _doc_normalize_ws(fragment)
        if not fragment_norm:
            continue
        fragment_range = (fragment_start, fragment_end)
        if any(
            _ranges_overlap(fragment_range, (unit.start, unit.end)) for unit in author_year_units
        ):
            continue
        if any(_link_citation_covers_fragment(link, fragment_norm) for link in valid_links):
            continue
        for position, numbers, raw_token in _find_numbered_citations(
            fragment,
            max_entry=max_entry if numbering_active else None,
            allow_bare_paren=numbering_active,
        ):
            start = fragment_start + position
            end = start + len(raw_token)
            for number in numbers:
                key = entry_key_lookup.get(number) if numbering_active else None
                units.append(
                    _CitationUnit(
                        start=start,
                        end=end,
                        rendered=raw_token,
                        kind="numbered",
                        source="numbered",
                        keys=[key] if key is not None else [],
                        citation_text=raw_token,
                    )
                )
    return units


def _assert_citation_coverage_invariants(
    units: list[_CitationUnit], claims: list[tuple[str, str, str, str]], coverage: dict[str, Any]
) -> None:
    """The accounting invariants for this module's citation-unit bookkeeping.

    Every one of them is a statement about the unit list the function has just built, so
    a violation is a bug in this module rather than something a particular draft can
    provoke. The matching statement about the outside world -- that the unit count is
    what `citation_audit.citation_spans` itself reports -- is asserted by the tests, over
    generated documents, in `backend/tests/test_citation_coverage_properties.py`."""
    found = coverage["found"]
    diagnostics = coverage["diagnostics"]
    assert found == len(units), "found must count exactly the citation units"
    assert coverage["linked"] + coverage["unresolved"] == found, "linked + unresolved == found"
    assert sum(coverage["by_source"].values()) == found, "by_source must sum to found"
    assert (
        diagnostics["units_linked"]
        + diagnostics["units_fallback"]
        + diagnostics["units_numbered"]
        + diagnostics["units_without_keys"]
        == found
    ), "linked + fallback + unresolved == found"
    for unit in units:
        assert len(set(unit.keys)) == len(unit.keys), "a unit never carries one key twice"
    # One claim per (unit, proposition, key), not (unit, key)
    # alone -- two units sharing a sentence and a key but reporting DIFFERENT
    # propositions must both reach the verifier, so `pairs` is keyed on the
    # same triple `claims_from_citation_links`/this function's own dedup uses.
    # A unit's own EXTRA resolutions (a second
    # link that named the same rendered occurrence) each contribute their own pairs too,
    # so the count still matches the claims list one claim per proposition builds.
    pairs = {
        (unit.sentence, resolution_proposition or unit.sentence, key)
        for unit in units
        for resolution_keys, resolution_proposition in (
            [(unit.keys, unit.proposition)] + list(unit.extra_resolutions)
        )
        for key in resolution_keys
    }
    assert len(claims) == len(pairs), (
        "one claim per (unit, proposition, key) triple with a resolvable key"
    )
    assert coverage["sent_to_verifier"] == len(claims)
    # Uniqueness is checked on (claim_text,
    # claim_sentence, key), not (claim_sentence, key) alone -- two claims sharing a
    # sentence and a key are not forbidden, as long as their own claim_text
    # (proposition) differs; that is the dedup rule, not a coverage bug.
    assert (
        len({(text, claim_sentence, key) for text, key, _c, claim_sentence in claims})
        == len(claims)
    )
    for entry in coverage["unresolved_citations"]:
        assert entry, "an unresolved citation is never shown as an empty string"
        assert _is_bracket_balanced(entry), "an unresolved citation is always bracket-balanced"


def extract_claims_from_document(
    content: dict, paper_lookup: dict[str, Paper]
) -> tuple[list[tuple[str, str, str, str]], dict[str, Any]]:
    """Extract claims from a Tiptap draft document, plus a coverage summary.

    Walks the document paragraph by paragraph -- including a paragraph nested inside a
    list item or a blockquote -- excluding any node at or after a references/bibliography
    heading, which is used only to build the numbered-citation resolver. Each paragraph is
    turned into a list of citation units (see the region comment above); the paragraph's
    own ``citationLinks`` attribute claims one unit per surviving, keyed link; the
    author-year regexes claim the units left over; a unit still unclaimed is reported
    unresolved under its own rendered text. A sentence fragment with no author-year unit
    and no surviving link is scanned for numbered citations, which resolve against the
    reference list when its numbering is contiguous from one and covers every number
    cited in the body.

    Returns ``(claims, coverage)`` where ``claims`` is a
    ``(claim_text, citation_key, citation_text, claim_sentence)`` quadruple list --
    ``citation_key``, ``citation_text`` and ``claim_sentence`` in the same three
    positions and meaning as the ``(sentence, citation_key, citation_text)`` triple
    `_extract_claims` returns, plus ``claim_text`` prepended: the unit's own
    ``proposition`` -- the citation link's verbatim, contiguous span of its sentence --
    when the unit was resolved by a link that carried one and it still occurs in the
    paragraph text after whitespace normalisation, otherwise the sentence itself.
    ``claim_sentence`` is always the full sentence the claim was
    cut from, regardless of which text ``claim_text`` carries. ``coverage`` is a plain
    dict (wrapped as `app.schemas.fulltext.CitationCoverage` by the caller, which ignores
    the extra ``diagnostics`` key) with:

    * ``found`` -- the number of citation units, which for a draft in the author-year
      style is exactly the number of spans `citation_audit.citation_spans` reports;
    * ``linked`` -- units that resolved to at least one key in the project library;
    * ``unresolved`` -- units that did not, listed (up to
      `UNRESOLVED_CITATION_LIST_CAP`) in ``unresolved_citations``;
    * ``sent_to_verifier`` -- the claims returned, one per (unit, key) pair, with two
      units asking the identical (sentence, key) question collapsed into one claim;
    * ``by_source`` -- how each unit was resolved: ``"mapping"`` (the writer's own
      citation-link map), ``"numbered"`` (the reference-list resolver), or
      ``"author-year"`` (the pre-existing regexes, and any unit nothing resolved);
    * ``diagnostics`` -- unit and link counters for debugging, not part of the API
      schema.
    """
    body_nodes, reference_nodes = _split_body_and_reference_nodes(_top_level_nodes(content))
    entries = _parse_reference_entries(reference_nodes)
    numbering_active = _numbering_is_contiguous_from_one(entries)
    max_entry = len(entries) if numbering_active else None
    if numbering_active and not _numbering_covers_body(body_nodes, max_entry):
        numbering_active = False
        max_entry = None

    doi_key_lookup = _build_doi_key_lookup(list(paper_lookup.values()))
    entry_key_lookup: dict[int, str | None] = {
        number: _resolve_reference_entry_key(text, doi_key_lookup, paper_lookup)
        for number, text in entries.items()
    }

    diagnostics: dict[str, int] = {
        "links_seen": 0,
        "links_stale": 0,
        "links_without_keys": 0,
        "links_ignored": 0,
        "units_linked": 0,
        "units_fallback": 0,
        "units_numbered": 0,
        "units_without_keys": 0,
    }
    units: list[_CitationUnit] = []

    for node in _iter_paragraph_nodes(body_nodes):
        text = _extract_text_from_tiptap(node)
        if not text.strip():
            continue

        paragraph_units = _audit_span_units(text)
        paragraph_units.extend(_regex_only_units(text, paragraph_units))
        paragraph_units.sort(key=lambda unit: (unit.start, unit.end))

        paragraph_norm = _doc_normalize_ws(text)
        valid_links: list[dict] = []
        for link in (node.get("attrs") or {}).get("citationLinks") or []:
            diagnostics["links_seen"] += 1
            sentence_norm = _doc_normalize_ws(link.get("sentence") or "")
            # A link counts only while its own sentence is still
            # present in the paragraph's current text.
            if sentence_norm and sentence_norm in paragraph_norm:
                valid_links.append(link)
            else:
                diagnostics["links_stale"] += 1

        _assign_links_to_units(text, paragraph_units, valid_links, diagnostics)
        _assign_author_year_fallback(text, paragraph_units, paper_lookup)
        _assign_audit_key_fallback(paragraph_units)

        fragments = _sentence_fragments(text)
        paragraph_units.extend(
            _numbered_citation_units(
                text,
                fragments,
                paragraph_units,
                valid_links,
                numbering_active=numbering_active,
                max_entry=max_entry,
                entry_key_lookup=entry_key_lookup,
            )
        )
        paragraph_units.sort(key=lambda unit: (unit.start, unit.end))
        for unit in paragraph_units:
            unit.sentence = _fragment_holding(text, fragments, unit.start)
            if not unit.citation_text:
                unit.citation_text = unit.rendered
            # `_assign_links_to_units` only checked the proposition against
            # the whole paragraph, before any unit had a `sentence` to check against (an
            # abbreviation such as "e.g." can split one link's own sentence into two
            # fragments, so a proposition spanning the split is a real paragraph substring
            # that is not a substring of the one fragment this particular unit lands in).
            # Re-validate here, now that `sentence` is known, and drop it back to the
            # sentence fallback otherwise, so `claim_text` is always contained in
            # `claim_sentence` when a proposition is used.
            if unit.proposition and _doc_normalize_ws(unit.proposition) not in (
                _doc_normalize_ws(unit.sentence)
            ):
                unit.proposition = None
            # The same re-validation, now over
            # every EXTRA resolution too -- a second proposition can span the same
            # fragment split the primary one already had to be checked against.
            unit.extra_resolutions = [
                (
                    list(dict.fromkeys(extra_keys)),
                    extra_proposition
                    if extra_proposition
                    and _doc_normalize_ws(extra_proposition) in _doc_normalize_ws(unit.sentence)
                    else None,
                )
                for extra_keys, extra_proposition in unit.extra_resolutions
            ]
        units.extend(paragraph_units)

    claims: list[tuple[str, str, str, str]] = []
    seen_claims: set[tuple[str, str, str]] = set()
    coverage: dict[str, Any] = {
        "found": 0,
        "linked": 0,
        "sent_to_verifier": 0,
        "unresolved": 0,
        "unresolved_citations": [],
        "by_source": {"mapping": 0, "author-year": 0, "numbered": 0},
        "diagnostics": diagnostics,
    }
    for unit in units:
        deduped: list[str] = []
        for key in unit.keys:
            if key not in deduped:
                deduped.append(key)
        unit.keys = deduped
        coverage["found"] += 1
        coverage["by_source"][unit.source or "author-year"] += 1
        if not unit.keys:
            diagnostics["units_without_keys"] += 1
        elif unit.source == "mapping":
            diagnostics["units_linked"] += 1
        elif unit.source == "numbered":
            diagnostics["units_numbered"] += 1
        else:
            diagnostics["units_fallback"] += 1
        # Every resolution this unit carries --
        # its own primary (keys, proposition) plus any extra ones a later link attached
        # to the same rendered occurrence -- contributes its keys to the linked/
        # unresolved check, so a unit is never reported unresolved because its FIRST
        # link's own keys happened not to be in the library while a later link's did.
        all_keys: list[str] = list(unit.keys)
        for extra_keys, _extra_proposition in unit.extra_resolutions:
            for key in extra_keys:
                if key not in all_keys:
                    all_keys.append(key)
        if any(key in paper_lookup for key in all_keys):
            coverage["linked"] += 1
        else:
            coverage["unresolved"] += 1
            if len(coverage["unresolved_citations"]) < UNRESOLVED_CITATION_LIST_CAP:
                display_text = unit.citation_text or unit.rendered
                if not _is_bracket_balanced(display_text):
                    display_text = unit.rendered
                coverage["unresolved_citations"].append(display_text)
        # The text actually sent to the verifier is the unit's own
        # proposition when its link supplied a usable one, otherwise the sentence.
        # `claim_sentence` always stays the full sentence, so the
        # verifier's report can show both. The
        # unit's own primary resolution, plus every extra one a second (or later) link
        # attached to this same rendered occurrence, each build their own claim(s) below
        # -- a rendered citation unit carries every citation link that resolves to
        # it, one claim per proposition.
        resolutions: list[tuple[list[str], str | None]] = [(unit.keys, unit.proposition)]
        resolutions.extend(unit.extra_resolutions)
        for resolution_keys, resolution_proposition in resolutions:
            claim_text = resolution_proposition or unit.sentence
            for key in resolution_keys:
                # One claim per (unit, proposition, key). Two
                # units in one sentence fragment carrying the same key AND the same claim
                # text -- a source cited narratively and parenthetically in the same
                # sentence, or two link entries the writer's own dedup left identical --
                # still ask the verifier the identical question once, so they cost one
                # call and one workbook row while still counting as two citations in
                # `found`. The dedup key is (sentence, claim_text, key), not
                # (sentence, key): two units (or two resolutions of the same unit) naming
                # the same sentence and key but DIFFERENT propositions
                # must both reach the verifier, not collapse into one.
                dedup_key = (unit.sentence, claim_text, key)
                if dedup_key in seen_claims:
                    continue
                seen_claims.add(dedup_key)
                claims.append((claim_text, key, unit.citation_text, unit.sentence))
    coverage["sent_to_verifier"] = len(claims)

    if CITATION_COVERAGE_INVARIANT_CHECKS:
        _assert_citation_coverage_invariants(units, claims, coverage)
    return claims, coverage


#: A model-quoted fragment, e.g. the ``"..."`` in ``'"fragment one" ... "fragment two"'``.
#: Typography-aware: a straight or curly double
#: quote may open a fragment and a straight or curly double quote may close it, in any
#: combination, because this runs on the RAW model quote, before ``_normalise_for_match``
#: folds curly quotes to straight ones. Without this, a quote the model wrapped in
#: typographic double quotes (``“...”``) finds no fragment, falls back to the
#: whole raw string, and that fallback string still carries the curly quote characters,
#: which are absent from the chunk -- a false "not verbatim" demotion for exactly the
#: "curly quotes" edge case this guard exists to handle.
_QUOTED_FRAGMENT_RE = re.compile(r'["“”]([^"“”]+)["“”]')


# --------------------------------------------------------------------------------------
# Shared guard machinery.
#
# `_normalise_for_match` and `_quote_segments` implement Unicode-typography folding,
# which measurably outperforms a laxer verbatim rule: it recovers 20/23 and 20/25 HSS
# quote locations that whitespace-only normalisation found in only 11/23 and 10/25 --
# with zero of the ten correct HSS `verified` answers per run newly failing the check.
# They are used by every guard below and by `_find_evidence_location`.
# --------------------------------------------------------------------------------------

#: Every earlier fold here -- curly-to-straight quotes, en/em-dash-to-hyphen,
#: nbsp-to-space, and an earlier letter-hyphen-linebreak deletion -- was one more special
#: case of the same underlying problem: a PDF-extraction artefact (typography, a hard
#: line-wrap, an injected space) sits between two characters that are otherwise identical
#: on both sides of a comparison. Two further cases neither fold reached:
#: `real-test-13`'s cached "(r0.28; p0.00)" against the model's own "(r=0.28; p=0.00)" (an
#: "=" sign, not a dash, hyphen, quote or nbsp -- and, on inspection of the actual cached
#: bytes, not even a real "=" but a stray control byte pymupdf substituted for one), and
#: `hss-verbatim-03` (dev)'s "The \ufb01 ndings" -- pymupdf splits the "fi" ligature
#: (correctly NFKC-folded to the two letters "f", "i") from the rest of the word with an
#: injected space, which an earlier whitespace-collapse-to-one-space step could shrink but
#: never remove. Rather than add a fourth special-cased fold, the canonical form is the
#: general rule all of the specific ones were approximating: keep only Unicode letters and
#: digits, case-folded; every quote mark, dash, hyphen, "=" sign, stray control byte and
#: injected space vanishes identically on both the quote and the chunk side of every
#: comparison below, so whichever of these artefacts either side happens to carry, the two
#: still compare equal. `_claim_terms` applies the identical fold per term (see its
#: docstring), so the claim-term and chunk sides of `_guard_attribution`'s coverage check
#: stay symmetric.
#:
#: This is deliberately not gated on digits the way an earlier, narrower fold was: "12-15"
#: now folds to "1215" and "CTLA-4" to "ctla4", not left untouched. `_quote_segments` and
#: `_guard_attribution` only ever compare a canonicalised NEEDLE for containment inside a
#: canonicalised HAYSTACK built from a real (30+ raw character, for a quote segment) span of
#: prose, never a bare numeral or identifier standing alone, so the chance that removing a
#: numeral-internal hyphen manufactures an unrelated match is negligible; conversely it lets
#: "CTLA-4" and "CTLA4" (two real spellings of the same identifier) compare equal, which the
#: old digit-exclusion could not. `test_normalise_for_match_letter_hyphen_fold_never_reaches_
#: digits` (the fixture pinning the old, narrower behaviour) is updated to the new one,
#: not deleted, and says so in its own docstring. The same removal also erases a minus sign
#: and any comparison operator standing between two numerals: "r = -0.28; p < 0.05" and
#: "r = 0.28; p = 0.05" both fold to "r028p005", so a flipped correlation sign or a "<"
#: mis-quoted as "=" is no longer visible to a comparison built on this fold. That is a real,
#: accepted loss of precision, not "negligible" the way a numeral-internal hyphen removal is;
#: it only ever weakens a demotion (the guard can fail to flag a genuine sign/inequality
#: error, never invent one), and it is recorded as a limitation below, next to the
#: running-header/page-number one.
#:
#: What this fold still cannot bridge: an injected running header or page number *inside* a
#: quote span (`real-test-18`, `hss-verbatim-02`) survives, because a running header's own
#: text and a page number are themselves letters and digits -- there is nothing to delete.
#: Separately, a minus sign or a comparison operator ("=", "<", ">") between two numerals is
#: also erased by this fold (see immediately above), so a quote that flips a reported sign or
#: swaps "<" for "=" against the source is not caught either. All of these remain known,
#: accepted cache-noise / precision limitations.
def _keep_letters_and_digits(text: str) -> str:
    """NFKC-normalise, then keep only Unicode letters and digits, case-folded.

    The shared canonicalisation both `_normalise_for_match` (whole-string comparisons) and
    `_claim_terms` (per-token comparisons) apply, so the two sides of every guard comparison
    -- quote-fidelity's needle/haystack, attribution's claim-term/chunk-haystack -- fold
    identically. `str.isalpha()`/`str.isdigit()` are true for the corresponding Unicode
    categories generally, not only ASCII, and `str.casefold()` is Unicode case-folding
    (stronger than `.lower()` for a handful of scripts), matching this module's existing
    `_fold_identity_diacritics` policy of never assuming ASCII.

    This also drops a minus sign and any comparison operator ("=", "<", ">") standing between
    two numerals, not only the numeral-internal hyphen the module comment above discusses:
    "r = -0.28; p < 0.05" and "r = 0.28; p = 0.05" both fold to "r028p005". A flipped
    correlation sign or a "<" quoted as "=" is therefore invisible to any comparison built on
    this fold. Accepted as a known limitation, of the same kind
    as the injected-running-header/page-number one: the guards that consume this fold only
    ever demote, so the failure mode is a missed demotion, never a false one.
    """
    normalised = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalised if ch.isalpha() or ch.isdigit()).casefold()


def _normalise_for_match(text: str | None) -> str:
    """Canonical comparison form: NFKC, then letters and digits only, case-folded.

    See `_keep_letters_and_digits`'s docstring above for the acceptance evidence and the
    digit-exclusion removal. Folds away
    every Unicode-typography and PDF-extraction difference between a model's quote and the
    pymupdf-extracted chunk text -- curly and straight quotes, en/em dashes, hyphens, "="
    signs and other punctuation, non-breaking spaces, hard line-wrap whitespace, an
    injected space inside a ligature -- without weakening what counts as "verbatim": the
    letters and digits themselves are never reordered or altered.

    Not all of the discarded characters are noise, though: a minus sign and a comparison
    operator ("=", "<", ">") are semantic, not typographic, and this fold erases them exactly
    as it erases a curly quote or a stray control byte. See `_keep_letters_and_digits`'s
    docstring for the measured example and the accepted-limitation note.
    """
    if not text:
        return ""
    return _keep_letters_and_digits(text)


# --------------------------------------------------------------------------------------
# The verification model
# occasionally pre-escapes a quote or newline inside a JSON string field itself (e.g. it
# writes the two raw characters ``\`` + ``"`` where a single ``"`` belongs), so after the
# JSON response is parsed once, the literal backslash survives inside `explanation` /
# `evidence_quote` / `suggested_revision`. This is purely a display/matching artefact of
# the model's own escaping, never a status signal, so it is collapsed once, right after
# the model output is turned into a `ClaimVerification` and before any guard reads the
# text -- in particular before `_guard_quote_fidelity`, whose `_QUOTED_FRAGMENT_RE` scan
# would otherwise capture a trailing stray backslash from a doubly-escaped closing quote
# and fail to find that segment in the source chunks (a false "not verbatim" demotion).
# --------------------------------------------------------------------------------------

_DOUBLE_ESCAPE_MAP = {'\\"': '"', "\\'": "'", "\\n": "\n"}
_DOUBLE_ESCAPE_RE = re.compile("|".join(re.escape(k) for k in _DOUBLE_ESCAPE_MAP))
#: Straight/curly quote pairs recognised when stripping a fully-wrapped `explanation`.
_WRAPPING_QUOTE_PAIRS = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}


def _collapse_double_escapes(text: str | None) -> str | None:
    """Replace literal ``\\"``, ``\\'`` and ``\\n`` two-character sequences.

    Textual only: never inspects or changes ``status``. ``None`` passes through
    unchanged so optional fields (``evidence_quote``, ``suggested_revision``) stay
    ``None`` rather than becoming the string ``"None"``.
    """
    if text is None:
        return None
    return _DOUBLE_ESCAPE_RE.sub(lambda m: _DOUBLE_ESCAPE_MAP[m.group(0)], text)


def _strip_wrapping_quotes(text: str) -> str:
    """Strip one surrounding matched pair of quotation marks, if the whole string is wrapped.

    Only ever applied to ``explanation``: a model that answers with
    ``"The paper supports this."`` rather than ``The paper supports this.`` is quoting its
    whole answer, not quoting the source, so the wrapping marks carry no evidentiary
    meaning and just look like a formatting mistake to a reader.

    Stripping fired whenever
    the first and last characters merely happened to be a matched pair, so ``'"a" and "b"'``
    (two separate quoted spans joined by plain text) became ``'a" and "b'``, a text with a
    lone stray quote mark on each side rather than the two original, correctly-paired spans.
    It must strip only when the whole string is ONE quoted span, i.e. when the inner text
    (``text[1:-1]``) contains neither the opening nor the closing mark.
    """
    if len(text) < 2:
        return text
    close = _WRAPPING_QUOTE_PAIRS.get(text[0])
    if close is None or text[-1] != close:
        return text
    inner = text[1:-1]
    if text[0] in inner or close in inner:
        return text
    return inner


def normalise_verification_text(
    text: str | None, *, strip_wrapping_quotes: bool = False
) -> str | None:
    """Normalise one model-output text field before it is stored or guarded.

    Applied to ``explanation``, ``evidence_quote`` and ``suggested_revision`` right after
    the model's raw output is parsed (see ``_verify_one``), so every downstream reader --
    the guards, the stored ``job.result``, the claim-record CSV export and the frontend --
    sees the same, already-normalised text. ``strip_wrapping_quotes`` is only passed
    ``True`` for ``explanation``.
    """
    normalised = _collapse_double_escapes(text)
    if normalised is not None and strip_wrapping_quotes:
        normalised = _strip_wrapping_quotes(normalised)
    return normalised


_ELLIPSIS_LEAD_RE = re.compile(r"^(?:\.\.\.|\u2026)\s*")
_ELLIPSIS_TRAIL_RE = re.compile(r"\s*(?:\.\.\.|\u2026)$")
#: Segments shorter than this are dropped by `_quote_segments`: a priori, not tuned on
#: any dataset. Measured on the ORIGINAL candidate string (post ellipsis-strip, pre
#: `_normalise_for_match`), not on its normalised length -- an earlier, narrower
#: letter-hyphen-linebreak fold could already shrink the normalised form enough to drop a
#: 30-32 raw-character segment uncompared, and the much broader "letters and digits only"
#: canonical form (see `_normalise_for_match`) would shrink it far more (every space and
#: punctuation mark also vanishes), so measuring post-normalisation would silently narrow
#: the floor's real, on-the-page meaning to well under 30 characters of source text.
#: Measuring on the raw candidate keeps the floor's original a priori meaning ("a residual
#: too short to trust") independent of how aggressive the comparison-side folding becomes.
QUOTE_SEGMENT_MIN_CHARS = 30


def _quote_segments(quote: str | None) -> list[str]:
    """Split a model quote into verbatim-checkable segments.

    Double-quoted segments of 10 or more raw characters are used when present (the model
    routinely wraps a quote in literal double quotes and joins non-contiguous excerpts
    with ``' ... '``); otherwise the raw string is the sole candidate. Each candidate has
    a leading/trailing ``...``/``\u2026`` stripped, then candidates shorter than
    ``QUOTE_SEGMENT_MIN_CHARS`` **raw** characters are dropped (a short residual, e.g.
    a bare page number caught between quote marks, is not evidence of anything): measured
    on the candidate itself, not on ``_normalise_for_match(candidate)``; see
    the constant's own docstring.

    When every double-quoted fragment found above is
    dropped by that 30-character floor -- e.g. a long quote whose only quoted fragment is a
    short inner term such as ``'The authors found "significant" gains ...'`` -- the loop
    above would otherwise produce an empty list, silently exempting the whole (possibly
    fabricated) surrounding text from verification. In that case only, fall back to the raw
    quote itself as a single candidate, so the long span is still checked; a quote with no
    quoted fragment at all is unaffected (that already falls back to the raw string above).
    """
    if not quote:
        return []
    found = [f for f in _QUOTED_FRAGMENT_RE.findall(quote) if len(f) >= 10]
    candidates = found if found else [quote]
    segments: list[str] = []
    for candidate in candidates:
        stripped = candidate.strip()
        stripped = _ELLIPSIS_LEAD_RE.sub("", stripped)
        stripped = _ELLIPSIS_TRAIL_RE.sub("", stripped).strip()
        segments.append(stripped)
    filtered = [s for s in segments if len(s) >= QUOTE_SEGMENT_MIN_CHARS]
    if filtered or not found:
        return filtered
    raw = quote.strip()
    raw = _ELLIPSIS_LEAD_RE.sub("", raw)
    raw = _ELLIPSIS_TRAIL_RE.sub("", raw).strip()
    return [raw] if len(raw) >= QUOTE_SEGMENT_MIN_CHARS else []


#: Guard monotonicity: a guard may only move a status to a higher
#: rank here. `no_full_text` and `error` are deliberately absent -- no guard ever caps
#: against them; guard 0 rewrites `no_full_text` directly, outside this ranking.
STATUS_RANK: dict[str, int] = {"verified": 0, "needs_nuance": 1, "unsupported": 2}

#: Slugs that only ever land
#: in ``diagnostics``, never in ``machine_reasons`` -- a diagnostic never changes ``status``.
#: ``quote_relocated`` is
#: `relocate_evidence_quotes`'s own slug, appended by its callers after the guards return;
#: listed here so this module's own enumeration of diagnostic slugs stays accurate.
DIAGNOSTIC_SLUGS = frozenset(
    {"numeric_not_in_source", "centrality_unmarked", "quote_relocated"}
)


def _cap(status: str, floor: str) -> str:
    """Return the stricter (higher-ranked) of *status* and *floor*.

    Both arguments must be keys of :data:`STATUS_RANK`; a guard must check that before
    calling this (see the ``no_full_text``/``error`` short-circuit in
    :func:`apply_verification_guards`).
    """
    if STATUS_RANK.get(floor, -1) > STATUS_RANK.get(status, -1):
        return floor
    return status


def _locate_evidence_chunk_index(
    evidence_quote: str | None,
    chunks: list[dict],
    *,
    evidence_quotes: Sequence[str] | None = None,
) -> int | None:
    """0-based index of the first chunk containing a segment of the evidence quote(s).

    Takes ``evidence_quotes or
    [evidence_quote]`` and applies :func:`_quote_segments` to every quote in that list, so
    a caller that only ever knew the single-span ``evidence_quote`` keeps its exact
    behaviour (``evidence_quotes`` defaults to ``None``). Segments from every quote are
    pooled and tried longest first (least likely to collide with unrelated text).
    """
    quotes = evidence_quotes or [evidence_quote]
    segments = [seg for q in quotes for seg in _quote_segments(q)]
    for segment in sorted(segments, key=len, reverse=True):
        needle = _normalise_for_match(segment)
        if not needle:
            continue
        for idx, chunk in enumerate(chunks):
            haystack = _normalise_for_match(chunk.get("text", ""))
            if needle in haystack:
                return idx
    return None


def _locate_evidence_chunk_indices(
    evidence_quote: str | None,
    chunks: list[dict],
    *,
    evidence_quotes: Sequence[str] | None = None,
) -> list[int]:
    """Every distinct chunk index containing a segment of any evidence quote, in the order
    first located: tier B considers every located chunk, not just the
    single "first" one ``_locate_evidence_chunk_index`` returns. Each quote is located
    independently -- unlike ``_locate_evidence_chunk_index``'s own pooled-segment search --
    so a span answered in chunk 3 is not shadowed by a longer, earlier span found in chunk 1.
    """
    quotes = evidence_quotes or [evidence_quote]
    indices: list[int] = []
    for quote in quotes:
        idx = _locate_evidence_chunk_index(quote, chunks)
        if idx is not None and idx not in indices:
            indices.append(idx)
    return indices


def _find_evidence_location(
    evidence_quote: str | None,
    chunks: list[dict],
    *,
    evidence_quotes: Sequence[str] | None = None,
) -> str | None:
    """Locate which numbered chunk (as shown to the model) contains the evidence quote.

    Mirrors the 1-based "--- Chunk N ---" numbering in ``format_verification_prompt``, so
    the claim-verification record export can point a reader at the same chunk the model
    saw. Shares ``_normalise_for_match``/``_quote_segments``
    with the guards below, instead of the earlier whitespace-only helper pair. Returns
    ``None`` when there is no quote or none of its segments can be found.
    ``evidence_quotes``, when given, is searched alongside the single
    ``evidence_quote`` (see :func:`_locate_evidence_chunk_index`).
    """
    idx = _locate_evidence_chunk_index(evidence_quote, chunks, evidence_quotes=evidence_quotes)
    if idx is None:
        return None
    chunk = chunks[idx]
    section = chunk.get("section")
    return f"chunk {idx + 1} ({section})" if section else f"chunk {idx + 1}"


# --------------------------------------------------------------------------------------
# Guard 0: `no_full_text` returned despite non-empty chunks.
# `_verify_claims` already answers the zero-chunk case deterministically before any
# model call, so a model-returned `no_full_text` can only mean the model reinterpreted
# the label as "the chunks did not cover the claim" -- code, not the model, owns that
# distinction.
# --------------------------------------------------------------------------------------


def _guard_no_full_text_with_chunks(
    status: str, chunk_texts: Sequence[str]
) -> tuple[str, list[str]]:
    if status == "no_full_text" and chunk_texts:
        return "unsupported", ["no_full_text_with_chunks"]
    return status, []


# --------------------------------------------------------------------------------------
# Guard 1: attribution. Calibrated on `hss-dev-30`
# (`evaluation/claims/hss_dev_claims.jsonl`) and now status
# changing (`ATTRIBUTION_GUARD_REPORT_ONLY = False`).
# --------------------------------------------------------------------------------------

#: Fixed function-word list (not tuned on any dataset): only tokens of 4+ characters can
#: reach here (the tokenizer regex below), so short function words are already excluded.
ATTRIBUTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "that", "with", "from", "have", "this", "were", "their", "which", "also",
        "when", "where", "what", "these", "those", "been", "being", "into", "onto",
        "about", "after", "before", "between", "during", "under", "over", "because",
        "while", "although", "however", "therefore", "thus", "such", "some", "more",
        "most", "much", "many", "each", "both", "either", "neither", "only", "just",
        "very", "even", "still", "will", "shall", "would", "could", "should", "than",
        "them", "they", "than", "here", "there", "again", "further", "once", "does",
        "doing", "having", "against", "above", "below", "same", "other", "than",
    }
)

#: Coverage below this fires the guard (subject to the located-quote override). Calibrated
#: on `hss-dev-30` (`evaluation/claims/hss_dev_claims.jsonl`, seed 20260906, 10 papers
#: disjoint from the frozen test set), never on any test-set statistic: substring coverage
#: of each claim's content terms (`_claim_terms`/`_content_tokens`, citations stripped) in
#: the normalised, concatenated chunk text was computed for all 25 chunk-bearing dev items
#: (`no_full_text` has no chunks and is excluded). `_claim_terms` folds each term's own
#: letter-hyphen-linebreak the same way `_normalise_for_match` folds the chunk haystack, so
#: the two numbers below are measured under that symmetric fold. The classes separate
#: cleanly: `wrong_paper` coverage is {0.364, 0.474, 0.500, 0.545, 0.563} (max 0.5625); every
#: other rule (`verbatim`, `paraphrase`, `altered`) is {0.875 .. 1.0} (min 0.875). Since `hi`
#: (0.5625) < `lo` (0.875), the rule sets the constant to
#: `floor((hi + lo) / 2 * 100) / 100 = floor(71.875) / 100 = 0.71`. Both sides' fold covers
#: every non-alphanumeric character (`_keep_letters_and_digits`), not only a
#: letter-hyphen-linebreak; on the same `hss-dev-30` set under that fold, both bucket sets
#: are byte-for-byte identical to the numbers above (`_content_tokens` already extracted
#: alphabetic runs before either fold ever ran, so the wider fold changes nothing on this
#: particular dev set).
ATTRIBUTION_COVERAGE_MIN = 0.71

#: Section 5.3: the classes separated on `hss-dev-30` (see the constant's docstring above),
#: so the guard ships status changing, not report-only.
ATTRIBUTION_GUARD_REPORT_ONLY = False

_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


def _content_tokens(text: str) -> set[str]:
    """Lowercased alphabetic tokens of 4+ characters (brief section 4.2 / M3)."""
    return set(_CONTENT_TOKEN_RE.findall(text.lower()))


_CITATION_PAREN_RE = re.compile(r"\([^()]*(?:1[5-9]|20)\d{2}[a-z]?[^()]*\)")


def _strip_citations(text: str) -> str:
    """Remove parenthetical citations (any ``(...)`` group containing a four-digit year:
    ``(Smith, 2020)``, ``(Smith & Jones, 2019)``, ``(Smith et al., 2021)``, ``(2020)``)
    before content-term extraction.

    Guard 1 measures whether the *cited* paper's own text covers the claim's substantive
    assertion; a citation's author surname is essentially never repeated in that paper's own
    body text (the check is about the source's content, not a self-citation), so leaving it
    in would make the guard fire on citation metadata rather than content -- a false positive
    on every production claim, which always carries a trailing citation (production claim
    text is extracted with its in-text citation attached), not only the ``wrong_paper`` cases
    it is meant to catch. For example, the read-contract fixture claim
    "Scores improved (Smith, 2020)." against a chunk that supports "scores improved" but never
    says "smith" would otherwise be capped to ``unsupported`` regardless of how well the
    finding itself is supported.
    """
    return _CITATION_PAREN_RE.sub(" ", text)


def _claim_terms(claim_text: str) -> set[str]:
    """Content terms of a claim, folded to match Guard 1's chunk haystack exactly.

    `_guard_attribution` builds its haystack with `_normalise_for_match`, so a chunk's
    "self-evaluation", "self- evaluation" and "selfevaluation" all read identically there.
    `_content_tokens`'s tokenizer keeps an internal hyphen or apostrophe
    (`[A-Za-z][A-Za-z'-]{3,}`), so an un-folded claim term such as "self-evaluation" is a
    substring of none of those spellings once the chunk side is folded -- the fold must
    apply symmetrically, on the claim-term side too, or a hyphenated term can never be found
    in a source that plainly contains it.

    The per-term fold is `_keep_letters_and_digits`, the same general canonicalisation
    `_normalise_for_match` applies to the chunk haystack (see that function's docstring),
    not only a narrower letter-hyphen-linebreak deletion -- so an apostrophe or any other
    non-alphanumeric character inside a claim term is folded the same way an equivalent
    character inside the source text is.
    """
    terms = _content_tokens(_strip_citations(claim_text))
    terms = {_keep_letters_and_digits(t) for t in terms}
    return terms - ATTRIBUTION_STOPWORDS


def _quote_located(evidence_quote: str | None, chunk_texts: Sequence[str]) -> bool:
    """True when at least one quote segment is found in any chunk (guards 1 and 3)."""
    if not evidence_quote:
        return False
    haystacks = [_normalise_for_match(c) for c in chunk_texts]
    for segment in _quote_segments(evidence_quote):
        needle = _normalise_for_match(segment)
        if needle and any(needle in h for h in haystacks):
            return True
    return False


def _guard_attribution(
    status: str,
    *,
    claim_text: str,
    chunk_texts: Sequence[str],
    evidence_quote: str | None,
) -> tuple[str, list[str]]:
    """Guard 1: cap to ``unsupported`` when the claim's content terms are largely absent
    from the cited paper's chunks and the model located no quote (brief section 4.2).

    Reads the threshold from :data:`ATTRIBUTION_COVERAGE_MIN` (never a literal), so a
    calibration change is a one-line edit.

    Coverage is **substring containment** of each claim term in the normalised, lower-cased,
    concatenated chunk text -- matching the brief's formula literally ("t occurs in
    normalised(all chunks)") -- not whole-token set membership. WP-B7 review round 1, minor
    finding: an earlier version tokenised the chunks with :func:`_content_tokens` and tested
    set membership, which is a materially different statistic (measured on the HSS items:
    token-set gave `wrong_paper` min 0.500 / median 0.545 / max 0.800 versus substring's
    0.583 / 0.625 / 0.867), so the section 5.3 calibration must be run against *this*
    (substring) metric, not against the M3 numbers, which were never re-measured against it.
    """
    terms = _claim_terms(claim_text)
    if not terms:
        return status, []
    haystack = _normalise_for_match(" ".join(chunk_texts)).lower()
    covered = sum(1 for t in terms if t in haystack)
    coverage = covered / len(terms)
    if coverage >= ATTRIBUTION_COVERAGE_MIN:
        return status, []
    if _quote_located(evidence_quote, chunk_texts):
        return status, []
    return _cap(status, "unsupported"), ["attribution_mismatch"]


# --------------------------------------------------------------------------------------
# Assertion-status consistency guard: the v3 precedence
# over the model's own per-assertion verdicts (V3-8's first three clauses; the hedge-test
# clause is the model's own job under the v3 prompt, not a code guard).
# --------------------------------------------------------------------------------------


def _guard_assertion_status_consistency(
    status: str, *, assertions: Sequence[Mapping[str, Any]] | None
) -> tuple[str, list[str], list[str]]:
    """New guard: contradicted anywhere floors ``unsupported``; else CENTRAL absent floors
    ``unsupported``; else any peripheral absent floors ``needs_nuance``; else no fire.

    Never fires on empty ``assertions`` (a v1/v2-shaped answer, or a model that omitted the
    field, offers nothing to be inconsistent about). Zero or several ``central=True``
    assertions skip the centrality clause entirely -- there is no reliable "the" central
    assertion to test -- and append ``centrality_unmarked`` to the returned diagnostics,
    never to the returned machine reasons: a marking failure by the model is evidence about
    the model's own bookkeeping, not evidence that the claim itself is unsupported. The
    contradicted check is a strict first clause in an ``else`` chain: once it fires, the
    centrality clause (and its diagnostic) is never reached, whatever the central marking.

    Returns ``(status, machine_reasons, diagnostics)``, the same three-part split
    :func:`apply_verification_guards` returns as a whole.
    """
    assertions = list(assertions or [])
    if not assertions:
        return status, [], []
    if any(a.get("verdict") == "contradicted" for a in assertions):
        return _cap(status, "unsupported"), ["assertion_status_inconsistent"], []

    diagnostics: list[str] = []
    central_idxs = [i for i, a in enumerate(assertions) if a.get("central")]
    if len(central_idxs) == 1:
        central_idx = central_idxs[0]
        if assertions[central_idx].get("verdict") == "absent":
            return _cap(status, "unsupported"), ["assertion_status_inconsistent"], []
        peripheral = [a for i, a in enumerate(assertions) if i != central_idx]
    else:
        # Zero or several central=True: skip the centrality clause; every assertion is
        # treated as peripheral for the check below, so an absent one still floors
        # `needs_nuance` even though which assertion was "the" central one is unknown.
        diagnostics.append("centrality_unmarked")
        peripheral = assertions

    if any(a.get("verdict") == "absent" for a in peripheral):
        return _cap(status, "needs_nuance"), ["assertion_status_inconsistent"], diagnostics
    return status, [], diagnostics


# --------------------------------------------------------------------------------------
# Guard 2: numeric consistency. Tier B (floor `needs_nuance`) stays a
# status-changing guard; tier A is demoted to a diagnostic --
# see `_diagnose_numeric_not_in_source` below. A further status-changing guard,
# `_guard_numeric_value_absent`, applies the same underlying arithmetic as tier A's
# diagnostic, evidence-gated on the model's own per-assertion verdicts -- see that guard's
# own docstring for the evidence it was adopted on. Tolerances are set a priori
# (never tuned on any dataset): `NUMERIC_ABS_TOL` and `NUMERIC_REL_TOL`.
# --------------------------------------------------------------------------------------

NUMERIC_ABS_TOL = 0.005
NUMERIC_REL_TOL = 0.01

_FIGURE_TABLE_CHUNK_RE = re.compile(r"(?i)\b(?:table|figure|chunk)\s+\d+")
_NUMERAL_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(%)?")
_MIN_YEAR = 1500
_MAX_YEAR = 2100


def _numeral_is_identifier_fragment(text: str, start: int, end: int) -> bool:
    """True when the numeral at ``text[start:end]`` looks like part of an alphanumeric
    identifier rather than a quantity, not a real number to be checked against the source.

    Tier A's floor is ``unsupported``, the harshest
    verdict the product emits, and ``_extract_numerals`` had no notion of a quantity, so it
    fired on bare digits inside cell-line names ("NIH 3T3"), gene/protein codes ("Forkhead
    0"/"fox0") and hyphenated compound identifiers ("A-769662", "CTLA-4", "interleukin-2",
    "SHP-2"). Measured read-only on the SciFact **train** split (the brief's designated dev
    data, 919 claim-document pairs, no model call, via
    ``evaluation/claims/data/data/claims_train.jsonl`` + ``corpus.jsonl``): tier A fired on
    15 of 370 SUPPORT-gold pairs before this filter. This decision was recorded rather than
    silently tuned to the observed items: the filter below is purely structural (matches any
    digit run immediately fused to a letter, or immediately preceded by a hyphen whose other
    side is a run of letters), not a lexicon of the specific gene/compound names seen in this
    sample, so it generalises to the same syntactic pattern in any scientific text. With the
    filter, tier A fires on 9 of 370 SUPPORT-gold pairs (down from 15); the residual 9 are
    genuine limitations of a literal-numeral check (a paraphrased age, a bare digit used as a
    naming index with a space rather than a hyphen such as "Type 1"/"Forkhead 0", a
    fraction-style gene name "OCT3/4", a citation-style bare number, and a "30 million"
    magnitude word) that a syntactic filter cannot resolve without either a domain lexicon or
    per-assertion reasoning -- which is exactly what the v2 prompt's "quantity" assertion
    step already asks the model to do; tier A remains a coarse backstop, not primary
    detection. See ``evaluation/tests/test_claims_numeric_guard.py`` for the frozen dev-data
    regression pinning both the HSS M2 counts (unaffected: this filter matches none of the
    HSS numerals) and this SciFact-train SUPPORT/CONTRADICT/NEI firing count.
    """
    if end < len(text) and text[end].isalpha():
        return True
    if start > 0 and text[start - 1] == "-":
        j = start - 1
        while j > 0 and text[j - 1].isalpha():
            j -= 1
        if j < start - 1:
            return True
    return False


def _extract_numerals(text: str, *, skip_identifiers: bool = True) -> list[tuple[float, bool]]:
    """``(value, is_percent)`` pairs in *text* (brief section 4.3).

    Skips numerals that are a bare year in ``1500..2100`` (a citation year is
    syntactically just such a bare year) and bare figure/table/chunk indices ("Table 3",
    "Figure 2", "Chunk 4"), which are masked out before extraction, unconditionally.

    ``skip_identifiers`` (default ``True``) additionally drops numerals that look like part
    of an alphanumeric identifier rather than a quantity (see
    :func:`_numeral_is_identifier_fragment`).

    The identifier filter must be applied to the
    CLAIM's numerals only, never to the source-side candidate pool built from the shown
    chunks. A claim and its cited source routinely spell the same identifier or fused unit
    differently -- ``"CTLA - 4"`` (spaced) in a claim versus ``"CTLA-4"`` (hyphen-fused) in
    the abstract, or ``"50 mg"`` (spaced) versus ``"50mg"`` (unit-fused) -- so filtering both
    sides symmetrically strips the matching numeral out of one side and not the other, and
    tier A then reports a genuine numeral as "absent from every chunk" and floors the status
    to ``unsupported``, the harshest verdict the product emits, even when the model returned
    a verbatim supporting quote. Filtering only the claim side cannot create a spurious
    absence (it can only make the claim's own numeral list shorter, never the source's), so
    every call on ``chunk_texts`` below passes ``skip_identifiers=False`` and keeps the
    source pool maximal; only the two calls on ``claim_text`` use the default. Measured
    read-only on the SciFact-train dev set (the brief's designated development data, 919
    pairs, no model call): this fix leaves SUPPORT (9/370) and CONTRADICT (5/194) firing
    counts unchanged and removes 3 spurious NOT_ENOUGH_INFO fires that this asymmetry had
    manufactured -- see ``evaluation/tests/test_claims_numeric_guard.py``.
    """
    masked = _FIGURE_TABLE_CHUNK_RE.sub(lambda m: " " * len(m.group(0)), text)
    numerals: list[tuple[float, bool]] = []
    for match in _NUMERAL_RE.finditer(masked):
        numeral_str, percent_sign = match.group(1), match.group(2)
        is_percent = bool(percent_sign)
        value = float(numeral_str.replace(",", ""))
        is_bare_year = (
            not is_percent
            and "." not in numeral_str
            and "," not in numeral_str
            and _MIN_YEAR <= value <= _MAX_YEAR
        )
        if is_bare_year:
            continue
        if skip_identifiers and _numeral_is_identifier_fragment(
            masked, match.start(), match.end()
        ):
            continue
        numerals.append((value, is_percent))
    return numerals


def _numerals_match(a: float, b: float) -> bool:
    return abs(a - b) <= max(NUMERIC_ABS_TOL, NUMERIC_REL_TOL * max(abs(a), abs(b)))


def _numerals_equivalent(a: tuple[float, bool], b: tuple[float, bool]) -> bool:
    """True when *a* and *b* describe the same quantity within tolerance (brief 4.3).

    Two numerals with the **same** ``is_percent`` flag are compared directly, at their own
    scale. Two numerals with **different** flags (exactly one of them is a percentage) match
    when EITHER their raw values agree (a purely typographic difference -- the same number
    written once with a literal ``%`` and once spelled out or bare, e.g. claim "36 percent"
    against chunk "36%") OR their scaled values agree (a genuine percent/proportion
    equivalence, e.g. "36%" against "0.36") -- the proportion side multiplied by 100, so the
    relative tolerance is evaluated at the real magnitude of the percentage, not at an
    artificially 100x-scaled value.

    The previous implementation (``_numeral_candidates``)
    expanded *every* numeral to its times-100 form regardless of ``is_percent`` and then
    took the full cross product, so two numerals that were never percentages at all (e.g. a
    claim's ``1.62`` and a source's ``161``/``163``) could collide once one of them was
    multiplied by 100. That made tier A fire on 0 of 10 HSS ``altered`` items where brief
    M2 requires exactly 1 (``hss-altered-01``, claim numeral ``1.62`` vs source ``0.81``,
    spuriously "matched" by unrelated source numerals ``161``/``163`` through the scaled
    form). Restricting the percent/proportion equivalence to pairs where exactly one side
    is flagged ``is_percent`` reproduces the brief's M2 bucket counts exactly: 0/5
    (verbatim), 0/5 (paraphrase), 1/10 (altered, firing only on ``hss-altered-01``), 0/5
    (wrong_paper) -- see ``evaluation/tests/test_claims_numeric_guard.py``.

    An earlier fix dropped the same-scale comparison
    for mixed-flag pairs entirely, comparing ONLY the cross-scaled forms. Two numerals with
    the identical value but only one carrying a literal ``%`` therefore no longer matched at
    all (``_numerals_equivalent((36.0, False), (36.0, True))`` was ``False``), which forced
    tier A's ``unsupported`` floor onto claims that differ from their source by nothing but
    typography (claim "36 percent", source "36%"). Measured read-only on the SciFact
    **train** split (the brief's designated dev data, 919 pairs, no model call): this
    regression alone forced 3 SUPPORT-gold pairs to ``unsupported``, including a claim
    reading "76-85% of people with severe mental disorder receive no treatment ...". Adding
    the raw-value comparison back does not reopen the earlier false-positive collision: that
    collision came from expanding *every* numeral (including non-percent ones) to a times-100
    form and cross-multiplying, whereas the raw check here only ever runs between two numerals that
    already disagree on ``is_percent``, compared at their own unscaled values -- ``1.62``
    (not a percent) is still compared against ``161``/``163`` (not percents) exclusively
    through the same-flag branch above, never through this one. The brief's M2 bucket
    counts are unchanged (0/5, 0/5, 1/10 firing only on ``hss-altered-01``, 0/5) -- see
    ``evaluation/tests/test_claims_numeric_guard.py``.
    """
    value_a, percent_a = a
    value_b, percent_b = b
    if percent_a == percent_b:
        return _numerals_match(value_a, value_b)
    if _numerals_match(value_a, value_b):
        return True
    scaled_a = value_a if percent_a else value_a * 100.0
    scaled_b = value_b if percent_b else value_b * 100.0
    return _numerals_match(scaled_a, scaled_b)


def _numeral_matches_any(
    numeral: tuple[float, bool], candidates: Sequence[tuple[float, bool]]
) -> bool:
    return any(_numerals_equivalent(numeral, other) for other in candidates)


def _numeric_claim_values_absent_from_source(
    *, claim_text: str, chunk_texts: Sequence[str]
) -> list[tuple[float, bool]]:
    """Every claim numeral (``_extract_numerals``, identifier-fragment
    filtered) that matches nothing in the concatenated, unfiltered source-numeral pool.

    Shared by :func:`_diagnose_numeric_not_in_source` (the informational diagnostic)
    and :func:`_guard_numeric_value_absent` (the status-changing guard) so
    the two never disagree about which numerals are "absent": both call this, once each.
    """
    claim_numerals = _extract_numerals(claim_text)
    if not claim_numerals:
        return []
    source_numerals = [
        n for c in chunk_texts for n in _extract_numerals(c, skip_identifiers=False)
    ]
    return [n for n in claim_numerals if not _numeral_matches_any(n, source_numerals)]


def _diagnose_numeric_not_in_source(
    *, claim_text: str, chunk_texts: Sequence[str]
) -> list[str]:
    """Diagnostic only: a claim numeral absent from every shown
    chunk. Formerly a guard named ``_guard_numeric_tier_a``; no longer takes or returns a
    status, and never calls :func:`_cap` -- it cannot change ``status`` at all.

    Demoted from a (report-only) guard to a diagnostic for three reasons, recorded here
    rather than in a constant docstring since there is no longer a flag to attach them to:
    (1) it is a lexical, arithmetic-only check and cannot perform the unit, rate,
    denominator, fraction or percentage conversions the v3 prompt's ``supported``
    composition rule (V3-5) licenses, so a claim numeral it cannot find may still be a
    correct conversion of a source numeral; (2) its declared floor was ``unsupported``, the
    harshest verdict the product emits, reached on arithmetic alone -- which is why it
    shipped report-only (``NUMERIC_TIER_A_REPORT_ONLY``) and has never actually
    demoted a claim in the released product; (3) it fired identically, 8 of 120 times,
    under both the v1 and the v2 prompt, independent of the prompt's own reasoning, so it
    adds no discriminating signal beyond "this number could not be located verbatim".

    The slug stays ``numeric_not_in_source``; the frontend label "Number not found in
    source" stays byte identical.

    Reason (2) above is no
    longer the whole story -- :func:`_guard_numeric_value_absent` below now DOES change
    status on the same underlying condition this function reports, evidence-gated on the
    model's own per-assertion verdicts. This function's own contract is unchanged: it still
    takes no status, returns no status, and this slug never appears in ``machine_reasons``
    (:data:`DIAGNOSTIC_SLUGS`); the new guard reports through its own, different slug.
    """
    missing = _numeric_claim_values_absent_from_source(
        claim_text=claim_text, chunk_texts=chunk_texts
    )
    return ["numeric_not_in_source"] if missing else []


def _guard_numeric_tier_b(
    status: str,
    *,
    claim_text: str,
    chunk_texts: Sequence[str],
    evidence_index: int | None = None,
    evidence_indices: Sequence[int] | None = None,
) -> tuple[str, list[str]]:
    """Tier B (floor ``needs_nuance``, never ``unsupported``): a claim numeral matches
    somewhere in the paper but not in any chunk that contains a located quote span. Skipped
    when there is no located quote (guard 3 already covers that case).

    WP-B8 (brief section 1.3, multi-span quotes): ``evidence_indices``, when given, is the
    full pool of located chunks to check against (a claim's evidence may now span more than
    one chunk); ``evidence_index`` alone still works unchanged for a caller that only ever
    knew the single-span form.
    """
    if evidence_indices is not None:
        indices = list(evidence_indices)
    elif evidence_index is not None:
        indices = [evidence_index]
    else:
        indices = []
    indices = [i for i in indices if i is not None and 0 <= i < len(chunk_texts)]
    if not indices:
        return status, []
    claim_numerals = _extract_numerals(claim_text)
    if not claim_numerals:
        return status, []
    source_numerals = [
        n for c in chunk_texts for n in _extract_numerals(c, skip_identifiers=False)
    ]
    located_numerals = [
        n for i in indices for n in _extract_numerals(chunk_texts[i], skip_identifiers=False)
    ]
    for numeral in claim_numerals:
        if _numeral_matches_any(numeral, source_numerals) and not _numeral_matches_any(
            numeral, located_numerals
        ):
            return _cap(status, "needs_nuance"), ["numeric_outside_cited_passage"]
    return status, []


# --------------------------------------------------------------------------------------
# This guard promotes part of
# the numeric-not-in-source diagnostic back to status-changing, evidence-gated on the
# model's own per-assertion verdicts.
# --------------------------------------------------------------------------------------


def _assertion_carries_an_absent_missing_numeral(
    assertions: Sequence[Mapping[str, Any]], missing: Sequence[tuple[float, bool]]
) -> bool:
    """True when some assertion the model itself marked ``verdict == "absent"`` names a
    numeral (in its own ``text``/``claim_value``/``entity``/``measure`` fields) equivalent
    to one of *missing*.

    This is the model itself flagging the very number the diagnostic already found
    unsupported as something the source does not state -- as opposed to the model simply
    never quoting that particular figure while marking every assertion ``supported``.
    Numerals are read with :func:`_extract_numerals`'s default (``skip_identifiers=True``),
    the same setting every other claim-side call in this module uses: an assertion's own
    ``text`` is a model-composed natural-language restatement, not source prose, so it is
    treated exactly like claim text, never like the unfiltered source pool.
    """
    for assertion in assertions:
        if assertion.get("verdict") != "absent":
            continue
        text = " ".join(
            str(assertion.get(key) or "")
            for key in ("text", "claim_value", "entity", "measure")
        )
        assertion_numerals = _extract_numerals(text)
        if any(_numeral_matches_any(m, assertion_numerals) for m in missing):
            return True
    return False


def _guard_numeric_value_absent(
    status: str,
    *,
    claim_text: str,
    chunk_texts: Sequence[str],
    assertions: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Cap ``needs_nuance`` when a claim numeral is absent from every shown chunk (Rule A);
    floor further to ``unsupported`` when the model's own assertions mark the very number
    that is missing as ``absent`` (Rule B, :func:`_assertion_carries_an_absent_missing_numeral`).

    Measured against every committed v3 row (frozen `results/v3`, the superseded run, and
    the dev iterations, hss and real, A and B -- 472 STATUS_RANK-eligible rows in total)
    before being adopted: :func:`_numeric_claim_values_absent_from_source` fires on 16 of
    them, every one of them a row whose gold label (by construction for HSS/dev, adjudicated
    for real) is ``unsupported`` -- never `verified`, never `needs_nuance`. Rule A alone
    changes zero of those 472 rows' final status (every firing row was already
    `needs_nuance` or `unsupported` from another guard); Rule B changes exactly one --
    `hss-altered-15` run A, from a lenient `needs_nuance` miss (the model marked the altered
    normalisation-base assertion `absent` rather than `contradicted`) to the correct
    `unsupported`. See the guard report's table for the full per-row breakdown.

    Rule A is unconditional: it fires whenever any claim numeral is absent from every shown
    chunk, whether or not *assertions* is given at all (``None``/``[]`` simply means Rule B
    can never fire) -- a real behaviour change from the diagnostic's previous, purely
    informational role. This is deliberate, adopted on the committed-row evidence above, not
    an oversight; see `_diagnose_numeric_not_in_source`'s docstring and the guard report for
    the pre-existing test fixtures (built on a chunk with no numerals at all, a shape absent
    from every committed row) that this changes.

    Rule A's own measured effect on the 472 committed rows is exactly zero: every one of the
    16 firing rows was already ``needs_nuance`` or ``unsupported`` from another guard before
    Rule A's cap is applied, so the entire observed status change above (the one
    `hss-altered-15` row) comes from Rule B alone. Rule A ships unconditionally anyway because
    it reinstates, as a cap rather than a floor, the lexical check the WP-B8 brief demoted to
    a diagnostic partly because "it is a lexical check and cannot do the conversions V3-5
    licenses" -- for example a claim stating "a quarter (25%)" against a source that says
    "one in four" has a claim numeral matching nothing and would be capped to ``needs_nuance``
    on that ground alone. No row among the 472 committed rows exercises that class, so this
    particular effect of Rule A is unmeasured by the evidence base this guard was adopted on;
    it is a deliberate, documented exposure, not an oversight.
    """
    missing = _numeric_claim_values_absent_from_source(
        claim_text=claim_text, chunk_texts=chunk_texts
    )
    if not missing:
        return status, []
    capped = _cap(status, "needs_nuance")
    if _assertion_carries_an_absent_missing_numeral(assertions or [], missing):
        capped = _cap(capped, "unsupported")
    return capped, ["numeric_value_absent_from_source"]


# --------------------------------------------------------------------------------------
# Guard 3 (brief section 4.4): quote fidelity. Must ship together with
# `_normalise_for_match` / `_quote_segments` (M1) -- see the risk noted in their
# docstrings and in `evaluation/tests/test_claims_quote_fidelity_guard.py`.
# --------------------------------------------------------------------------------------


def _guard_quote_fidelity(
    status: str,
    *,
    evidence_quote: str | None,
    chunk_texts: Sequence[str],
    evidence_quotes: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Guard 3: cap ``verified`` to ``needs_nuance`` when the quote is not verbatim.

    Decided explicitly (WP-B7 review round 1, minor finding): when every candidate from
    :func:`_quote_segments` is dropped for being under ``QUOTE_SEGMENT_MIN_CHARS`` (e.g. the
    model's only double-quoted fragment is a handful of words), ``segments`` is ``[]`` and
    the loop below never runs, so the status passes through unchanged. This is a deliberate
    "no evidence either way" pass, not an oversight: a fragment too short to trust is not
    proof the quote is *wrong*, and demoting on it would reopen the false-demotion failure
    mode M1 exists to avoid (brief section 4.4). See
    ``test_guard_quote_fidelity_passes_when_every_segment_is_below_the_30_char_floor``.

    WP-B8 (brief section 1.3, multi-span quotes): takes ``evidence_quotes or
    [evidence_quote]`` and applies :func:`_quote_segments` to every quote in that list, so a
    caller offering only the single-span ``evidence_quote`` keeps its exact behaviour
    (``evidence_quotes`` defaults to ``None``). Every span of every quote must be verbatim;
    the first one that is not fires the guard, as before.
    """
    if status != "verified":
        return status, []
    quotes = evidence_quotes or [evidence_quote]
    if not any(quotes):
        return _cap(status, "needs_nuance"), ["quote_not_verbatim"]
    haystacks = [_normalise_for_match(c) for c in chunk_texts]
    segments = [seg for q in quotes for seg in _quote_segments(q)]
    for segment in segments:
        needle = _normalise_for_match(segment)
        if not needle or not any(needle in h for h in haystacks):
            return _cap(status, "needs_nuance"), ["quote_not_verbatim"]
    return status, []


# --------------------------------------------------------------------------------------
# Guard 7: scale-word fidelity. Every guard above
# compares presence -- is the quote in the source, is the numeral in the located chunk,
# does the model's own assertion list agree with its own status. This guard is the first to
# compare force: a delivered sentence whose own word sits at a different point on a closed
# ordinal scale than the source's word at the same place, even though every guard above
# already accepted its quotes as genuinely present. It never promotes and it never reaches
# `unsupported`; it only caps `verified` to `needs_nuance` (design section 5).
#
# Three closed, hand-written classes (design section 3), each a scale from a weak surface
# form to a strong one. A word belongs to exactly one class. "Significant"/"significantly",
# reporting verbs (find, found, report, observe, ...) and number words are deliberately not
# in any class -- the reasons are the design's, not repeated here.
# --------------------------------------------------------------------------------------

#: Curly quotes and dashes folded to ASCII before tokenising, exactly as guard 3's own
#: `_normalise_for_match` does for the same reason: a model or source that uses a typographic
#: apostrophe must not silently fail to match one that uses a straight one.
_SCALE_CURLY_FOLD: dict[str, str] = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
}

#: A dedicated tokenizer, not `_tokens_with_offsets`: this guard's own alignment (design
#: section 4) needs a hyphenated compound such as "self-reflection" to stay one token so it
#: reads as one content-stem anchor, which `_tokens_with_offsets`'s letters-and-digits-only
#: run would split in two. Measured, not assumed: re-running the design's own probe scripts
#: with a letters-and-digits-only tokenizer instead of
#: this one silently drops one of the eleven firings the design's section 9 records on the
#: full 852-row stored set (the one SciFact row in run A), so the two tokenizers are not
#: interchangeable on this data and this guard keeps its own.
_SCALE_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'.\-%]*")


def _scale_tokens(text: str | None) -> list[str]:
    """Lowercase, fold curly quotes/dashes to ASCII, then split into maximal runs of
    letters, digits, apostrophes, periods, hyphens and per cent signs (design section 4,
    "Tokens")."""
    folded = text or ""
    for source, target in _SCALE_CURLY_FOLD.items():
        folded = folded.replace(source, target)
    return _SCALE_TOKEN_RE.findall(folded.lower())


def _scale_stem(token: str) -> str:
    """Strip a trailing possessive, then one inflectional suffix, provided at least four
    characters remain (design section 4, "Stems, for anchoring only"). Stemming decides only
    where to anchor a comparison; lexicon membership and rank are always read from the exact
    surface form in :data:`_SCALE_LEXICON`, so no fold can turn one scale word into another."""
    token = token.rstrip(".,;:")
    if token.endswith("'s"):
        token = token[:-2]
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


#: Class `extent` (design section 3): frequency adverbs and set quantifiers merged onto one
#: scale, deliberately, because the observed escape ("none" in the source, "seldom" in the
#: proposition) crosses the two.
_SCALE_EXTENT_WORDS: dict[int, tuple[str, ...]] = {
    0: ("no", "none", "never", "neither", "nobody", "nothing", "zero"),
    1: (
        "rarely", "seldom", "hardly", "scarcely", "barely", "infrequently", "few",
        "minority", "handful",
    ),
    2: ("occasionally", "sometimes", "some", "several"),
    3: ("often", "frequently", "regularly", "commonly", "repeatedly", "many", "much", "numerous"),
    4: ("usually", "generally", "typically", "mostly", "predominantly", "most", "majority"),
    5: (
        "always", "invariably", "universally", "consistently", "all", "every", "entirely",
        "wholly",
    ),
}

#: Class `degree` (design section 3): intensity of a stated change or difference.
_SCALE_DEGREE_WORDS: dict[int, tuple[str, ...]] = {
    1: ("slight", "slightly", "marginal", "marginally", "minimally", "modest", "modestly"),
    2: ("somewhat", "moderate", "moderately", "partial", "partially"),
    3: (
        "considerable", "considerably", "substantial", "substantially", "notable",
        "notably", "appreciably",
    ),
    4: ("marked", "markedly", "great", "greatly", "strong", "strongly", "sharp", "sharply"),
    5: (
        "dramatic", "dramatically", "drastic", "drastically", "vast", "vastly", "enormous",
        "enormously", "radical", "radically", "overwhelming", "overwhelmingly", "huge",
        "hugely", "massive", "massively",
    ),
}

#: Class `hedge` (design section 3): epistemic force of the proposition's own verb or modal.
#: "Significant"/"significantly" are excluded (a statistical term of art, not commensurable
#: with a lay intensifier); reporting verbs (find, found, report, observe, ...) are excluded
#: (no epistemic force of their own, and the HSS paraphrase items swap exactly these words);
#: number words are excluded (the numeric guards already own them).
_SCALE_HEDGE_WORDS: dict[int, tuple[str, ...]] = {
    1: (
        "may", "might", "could", "possibly", "perhaps", "potentially", "seem", "seems",
        "appear", "appears", "suggest", "suggests", "suggesting", "tend", "tends",
    ),
    2: ("probably", "likely", "presumably"),
    3: (
        "show", "shows", "demonstrate", "demonstrates", "establish", "establishes",
        "prove", "proves", "confirm", "confirms",
    ),
}

#: Rank -> coarser band, per class (design section 3): near-synonym ranks are folded
#: together so the guard only ever fires on a step a reader would call a shift -- band 2
#: collapses "considerable" and "strong" so the guard stays quiet on "substantial affect"
#: against "strong impact" while still firing on "dramatically" against "considerably".
_SCALE_BAND_BY_RANK: dict[str, dict[int, int]] = {
    "extent": {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4},
    "degree": {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
    "hedge": {1: 1, 2: 2, 3: 3},
}

#: Surface form -> (class, rank), built once from the three word tables above. A surface
#: form belongs to exactly one class, so no token is ever compared on two scales at once.
_SCALE_LEXICON: dict[str, tuple[str, int]] = {
    word: (cls, rank)
    for cls, table in (
        ("extent", _SCALE_EXTENT_WORDS),
        ("degree", _SCALE_DEGREE_WORDS),
        ("hedge", _SCALE_HEDGE_WORDS),
    )
    for rank, words in table.items()
    for word in words
}

#: Closed function-word list for anchoring only (design section 4, "Content tokens"): a
#: content token is anything not in this list, not a lexicon word, and containing no digit.
_SCALE_FUNCTION_WORDS: frozenset[str] = frozenset(
    """a an the of to in on for with by from at as that this these those and or but yet so
    than then also into over under between within across about is are was were be been
    being am do does did has have had it its their they them he she his her we our you your
    i not nor if when while whereas because which who whom whose there here such more less
    other others one two both per via up out off again further s t re ve ll d m 's""".split()
)

#: Head span (design section 4): up to three content stems within six tokens to the right,
#: plus the nearest one within two tokens to the left. Six tokens to the right is the one
#: measured constant in the design (section 9): three tokens is not enough to reach across
#: the complementizer in "none reported that they critically engaged", the span the escape
#: (rows 3, 12, 17, 23) lives in.
_SCALE_HEAD_RIGHT_SPAN = 6
_SCALE_HEAD_RIGHT_MAX = 3
_SCALE_HEAD_LEFT_SPAN = 2
_SCALE_HEAD_LEFT_MAX = 1

#: Context span (design section 4): up to four content stems within eight tokens on each side.
_SCALE_CONTEXT_SPAN = 8
_SCALE_CONTEXT_MAX = 4

#: Admissibility minima (design section 4): the head condition means the same predicate, the
#: context condition means the same passage; both are needed (see the design's row-33 and
#: "no division of labour" examples).
_SCALE_MIN_HEAD_OVERLAP = 1
_SCALE_MIN_CONTEXT_OVERLAP = 2

#: Rule B's own closed word lists (design section 5, "negation scope"): the epistemic verb
#: stems it watches for, the absolute negators that raise over the claim's own verb, and the
#: ordinary negators that precede the quote's verb.
_SCALE_EPISTEMIC_VERB_STEMS: tuple[str, ...] = (
    "establish", "show", "demonstrat", "prove", "confirm",
)
_SCALE_ABSOLUTE_NEGATORS: frozenset[str] = frozenset({"no", "none", "never", "zero", "nothing"})
_SCALE_NEGATORS: frozenset[str] = frozenset(
    {"not", "never", "n't", "cannot", "failed", "nor", "neither", "without"}
)


class _ScaleOccurrence(NamedTuple):
    """One lexicon word found in a text, with the anchors :func:`_scale_align` compares it
    by. ``head``/``context`` are stems (:func:`_scale_stem`), never surface forms."""

    index: int
    word: str
    cls: str
    rank: int
    band: int
    head: frozenset[str]
    context: frozenset[str]


def _scale_is_content_token(token: str) -> bool:
    """Design section 4, "Content tokens": not a function word (by stem or by surface
    form), not itself a lexicon word, and contains no digit."""
    return (
        _scale_stem(token) not in _SCALE_FUNCTION_WORDS
        and token not in _SCALE_FUNCTION_WORDS
        and token not in _SCALE_LEXICON
        and not any(character.isdigit() for character in token)
    )


def _scale_content_indices(tokens: Sequence[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if _scale_is_content_token(token)]


def _scale_head_set(
    tokens: Sequence[str], index: int, content_indices: Sequence[int]
) -> frozenset[str]:
    right = [j for j in content_indices if index < j <= index + _SCALE_HEAD_RIGHT_SPAN]
    right = right[:_SCALE_HEAD_RIGHT_MAX]
    left = [j for j in content_indices if index - _SCALE_HEAD_LEFT_SPAN <= j < index]
    left = left[-_SCALE_HEAD_LEFT_MAX:] if left else []
    return frozenset(_scale_stem(tokens[j]) for j in right + left)


def _scale_context_set(
    tokens: Sequence[str], index: int, content_indices: Sequence[int]
) -> frozenset[str]:
    right = [j for j in content_indices if index < j <= index + _SCALE_CONTEXT_SPAN]
    right = right[:_SCALE_CONTEXT_MAX]
    left = [j for j in content_indices if index - _SCALE_CONTEXT_SPAN <= j < index]
    left = left[-_SCALE_CONTEXT_MAX:] if left else []
    return frozenset(_scale_stem(tokens[j]) for j in right + left)


def _scale_occurrences(text: str | None) -> list[_ScaleOccurrence]:
    """Every lexicon word in *text*, in order, each with its own head and context anchor
    sets computed against *text*'s own content tokens (design section 4)."""
    tokens = _scale_tokens(text)
    content_indices = _scale_content_indices(tokens)
    occurrences: list[_ScaleOccurrence] = []
    for index, token in enumerate(tokens):
        entry = _SCALE_LEXICON.get(token)
        if entry is None:
            continue
        cls, rank = entry
        occurrences.append(
            _ScaleOccurrence(
                index=index,
                word=token,
                cls=cls,
                rank=rank,
                band=_SCALE_BAND_BY_RANK[cls][rank],
                head=_scale_head_set(tokens, index, content_indices),
                context=_scale_context_set(tokens, index, content_indices),
            )
        )
    return occurrences


def _scale_admissible_pairs(
    claim_occurrences: Sequence[_ScaleOccurrence], quote_occurrences: Sequence[_ScaleOccurrence]
) -> list[tuple[int, int, int, int]]:
    """Every admissible ``(head_overlap, context_overlap, claim_index, quote_index)`` pair
    (design section 4, "Admissible pair"): same class, head sets share at least one stem,
    context sets share at least two."""
    pairs: list[tuple[int, int, int, int]] = []
    for ci, claim_occurrence in enumerate(claim_occurrences):
        for qi, quote_occurrence in enumerate(quote_occurrences):
            if claim_occurrence.cls != quote_occurrence.cls:
                continue
            head_overlap = len(claim_occurrence.head & quote_occurrence.head)
            context_overlap = len(claim_occurrence.context & quote_occurrence.context)
            if head_overlap >= _SCALE_MIN_HEAD_OVERLAP and context_overlap >= _SCALE_MIN_CONTEXT_OVERLAP:
                pairs.append((head_overlap, context_overlap, ci, qi))
    return pairs


def _scale_align(
    claim_occurrences: Sequence[_ScaleOccurrence], quote_occurrences: Sequence[_ScaleOccurrence]
) -> dict[int, int]:
    """One-to-one greedy alignment (design section 4, "Alignment"): every admissible pair is
    scored by head overlap first and context overlap second, sorted descending, and assigned
    so each claim and each quote occurrence is used at most once. Ties (equal head and
    context overlap) are broken towards the lenient reading: the candidate whose own band is
    closest to the claim occurrence's band sorts first, since a closer band is less likely to
    produce a mismatch finding."""
    pairs = _scale_admissible_pairs(claim_occurrences, quote_occurrences)

    def _sort_key(pair: tuple[int, int, int, int]) -> tuple[int, int, int]:
        head_overlap, context_overlap, ci, qi = pair
        band_gap = abs(claim_occurrences[ci].band - quote_occurrences[qi].band)
        return (-head_overlap, -context_overlap, band_gap)

    pairs.sort(key=_sort_key)
    used_claim: set[int] = set()
    used_quote: set[int] = set()
    chosen: dict[int, int] = {}
    for _head_overlap, _context_overlap, ci, qi in pairs:
        if ci in used_claim or qi in used_quote:
            continue
        used_claim.add(ci)
        used_quote.add(qi)
        chosen[ci] = qi
    return chosen


def scale_alignment_findings(
    claim_text: str, quotes: Sequence[str | None]
) -> list[dict[str, Any]]:
    """Every aligned claim/quote scale-word pair whose band differs (design section 4-5,
    Rule A): the alignment method's own output, exposed as a public helper so the delivered
    evidence exporter can record the pair next to a demoted row for the next human round
    (design section 5, point 6) -- it never changes ``status`` itself.

    Returns one dict per mismatch, in the claim's own token order:
    ``class``/``claim_word``/``claim_rank``/``claim_band``/``quote_word``/``quote_rank``/
    ``quote_band``. :func:`_guard_scale_fidelity` calls this directly, so the guard and any
    exporter always compute the identical pairing rather than two independent ones.
    """
    claim_occurrences = _scale_occurrences(claim_text)
    quote_occurrences = [
        occurrence for quote in quotes if quote for occurrence in _scale_occurrences(quote)
    ]
    findings: list[dict[str, Any]] = []
    for ci, qi in _scale_align(claim_occurrences, quote_occurrences).items():
        claim_occurrence, quote_occurrence = claim_occurrences[ci], quote_occurrences[qi]
        if claim_occurrence.band != quote_occurrence.band:
            findings.append(
                {
                    "class": claim_occurrence.cls,
                    "claim_word": claim_occurrence.word,
                    "claim_rank": claim_occurrence.rank,
                    "claim_band": claim_occurrence.band,
                    "quote_word": quote_occurrence.word,
                    "quote_rank": quote_occurrence.rank,
                    "quote_band": quote_occurrence.band,
                }
            )
    return findings


def _scale_negation_scope_shift(claim_text: str, quotes: Sequence[str | None]) -> bool:
    """Rule B (design section 5, point 4): negation raising. Fires when the claim contains
    an epistemic verb stem followed within three tokens by an absolute negator (e.g.
    "established no causal role", which reports a finding of absence), and some quote
    contains the same verb stem preceded within three tokens by a negator (e.g. "has not
    established a causal role", which reports the absence of a finding), with at least one
    shared content stem in the seven tokens after each verb."""
    claim_tokens = _scale_tokens(claim_text)
    for i, token in enumerate(claim_tokens):
        if not any(_scale_stem(token).startswith(verb) for verb in _SCALE_EPISTEMIC_VERB_STEMS):
            continue
        if not any(
            claim_tokens[j] in _SCALE_ABSOLUTE_NEGATORS
            for j in range(i + 1, min(len(claim_tokens), i + 4))
        ):
            continue
        claim_object = {
            _scale_stem(t)
            for t in claim_tokens[i + 1 : i + 8]
            if _scale_stem(t) not in _SCALE_FUNCTION_WORDS and t not in _SCALE_ABSOLUTE_NEGATORS
        }
        for quote in quotes:
            if not quote:
                continue
            quote_tokens = _scale_tokens(quote)
            for k, quote_token in enumerate(quote_tokens):
                if not any(
                    _scale_stem(quote_token).startswith(verb)
                    for verb in _SCALE_EPISTEMIC_VERB_STEMS
                ):
                    continue
                if not any(
                    quote_tokens[j] in _SCALE_NEGATORS for j in range(max(0, k - 3), k)
                ):
                    continue
                quote_object = {_scale_stem(t) for t in quote_tokens[k + 1 : k + 8]}
                if claim_object & quote_object:
                    return True
    return False


def _guard_scale_fidelity(
    status: str,
    *,
    claim_text: str,
    evidence_quote: str | None,
    evidence_quotes: Sequence[str] | None = None,
) -> tuple[str, list[str]]:
    """Guard 7: cap ``verified`` to ``needs_nuance`` when the claim restates the evidence at
    a different force on a closed ordinal scale (design section 5).

    Gated on ``status == "verified"`` (design section 5, point 1): nothing else in this
    guard runs otherwise, so it can never add a reason to a row a previous guard has already
    demoted, and -- because it runs immediately after guard 3 (quote fidelity) inside
    :func:`apply_verification_guards` -- every quote it reads has already been certified
    verbatim against the source, after relocation. Fires on either rule (design section 5,
    points 3-4): a scale-word mismatch (:func:`scale_alignment_findings`) or a negation-scope
    shift (:func:`_scale_negation_scope_shift`); either, both or neither can fire
    independently, and the reasons list carries whichever fired, in that order. Never
    promotes, never reaches ``unsupported``.
    """
    if status != "verified":
        return status, []
    quotes = evidence_quotes or [evidence_quote]
    reasons: list[str] = []
    if scale_alignment_findings(claim_text, quotes):
        reasons.append("scale_word_mismatch")
    if _scale_negation_scope_shift(claim_text, quotes):
        reasons.append("negation_scope_shift")
    if not reasons:
        return status, []
    return _cap(status, "needs_nuance"), reasons


# --------------------------------------------------------------------------------------
# Guard order: seven
# status-changing checks, in order -- guard 0, attribution, assertion-status consistency,
# quote fidelity, scale fidelity, numeric tier B, numeric value absent. Two diagnostics are
# computed alongside them and never change `status`: `centrality_unmarked` (from the
# assertion-status-consistency guard) and `numeric_not_in_source`
# (`_diagnose_numeric_not_in_source`, computed last and separately) -- the manuscript's
# "five status-changing checks plus two diagnostics" count is now seven plus two. Each
# status-changing guard applies `_cap`, so their order affects only the reason list, not the
# final status (every guard is monotonic on its own). All fired slugs are kept.
#
# `apply_verification_guards` is the module's public entry point: the evaluation harness --
# `evaluation/claims/verify_common.py`'s
# `ProductionVerifier` -- must call the *real* guards, not re-answer the model's raw
# status, so the numbers `summarize.py` scores are the numbers a user would actually see.
# `_verify_claims` below is its only other caller.
# --------------------------------------------------------------------------------------


def apply_verification_guards(
    status: str,
    *,
    claim_text: str,
    evidence_quote: str | None,
    evidence_quotes: Sequence[str] | None = None,
    assertions: Sequence[Mapping[str, Any]] | None = None,
    chunk_texts: Sequence[str],
    chunks: Sequence[dict],
) -> tuple[str, list[str], list[str]]:
    """Run every guard and diagnostic, in the order documented above.

    WP-B8 (brief section 1.3): ``evidence_quotes`` and ``assertions`` are new, both keyword
    only and both defaulting to ``None``, so a caller written against the old five-argument
    signature still runs unchanged. Returns ``(status, machine_reasons, diagnostics)`` --
    was a two-tuple before this change; ``diagnostics`` never changes ``status`` and never
    appears in ``machine_reasons``. Normalises ``evidence_quote``, every ``evidence_quotes``
    element and every assertion's ``quote``/``quotes`` as its first act (idempotent: a
    caller that already normalised these fields, such as ``_verify_one`` below, pays only a
    cheap no-op re-scan), so a caller that has not is still guarded against the same text a
    user will eventually see.
    """
    evidence_quote = normalise_verification_text(evidence_quote)
    evidence_quotes = [normalise_verification_text(q) for q in (evidence_quotes or [])] or None
    assertions = [
        {
            **a,
            "quote": normalise_verification_text(a.get("quote")),
            "quotes": [normalise_verification_text(q) for q in (a.get("quotes") or [])],
        }
        for a in (assertions or [])
    ]

    reasons: list[str] = []
    diagnostics: list[str] = []

    status, r = _guard_no_full_text_with_chunks(status, chunk_texts)
    reasons += r
    if status not in STATUS_RANK:
        # no_full_text (genuinely zero chunks) or error: the remaining checks never apply.
        return status, reasons, diagnostics

    capped, r = _guard_attribution(
        status, claim_text=claim_text, chunk_texts=chunk_texts, evidence_quote=evidence_quote
    )
    if r:
        reasons += r
        if not ATTRIBUTION_GUARD_REPORT_ONLY:
            status = capped

    status, r, d = _guard_assertion_status_consistency(status, assertions=assertions)
    reasons += r
    diagnostics += d

    status, r = _guard_quote_fidelity(
        status,
        evidence_quote=evidence_quote,
        evidence_quotes=evidence_quotes,
        chunk_texts=chunk_texts,
    )
    reasons += r

    status, r = _guard_scale_fidelity(
        status,
        claim_text=claim_text,
        evidence_quote=evidence_quote,
        evidence_quotes=evidence_quotes,
    )
    reasons += r

    located_indices = _locate_evidence_chunk_indices(
        evidence_quote, chunks, evidence_quotes=evidence_quotes
    )
    status, r = _guard_numeric_tier_b(
        status,
        claim_text=claim_text,
        chunk_texts=chunk_texts,
        evidence_indices=located_indices,
    )
    reasons += r

    status, r = _guard_numeric_value_absent(
        status, claim_text=claim_text, chunk_texts=chunk_texts, assertions=assertions
    )
    reasons += r

    diagnostics += _diagnose_numeric_not_in_source(claim_text=claim_text, chunk_texts=chunk_texts)

    return status, reasons, diagnostics


# --------------------------------------------------------------------------------------
# Quote relocation (authorised 2026-09-10): a separate, versioned step that runs before
# `apply_verification_guards`, never inside it -- the guard set above (`apply_verification_guards`
# and its seven guards, extended by guard 7 -- task authorisation 2026-09-14 -- from the
# frozen six) is frozen and stays byte-identical (`GUARD_DIGEST` below, and
# `test_verification_guard_functions_are_unchanged` / `test_verification_guards_are_still_frozen`,
# pin its current literal).
#
# The problem this closes: the quote-fidelity guard (`_guard_quote_fidelity`) demotes a
# claim the instant one candidate segment is not a verbatim substring of any chunk under
# `_normalise_for_match` -- exactly right for a fabricated quote, but it cannot tell that
# apart from a verification model that copied a real sentence with one word changed (the
# case that motivated this: "we" typed where the source said "and"). A quote is not
# rewritten to make the model look right; a quote that is *almost* the source's own words,
# by a small, source-anchored margin, is repointed at the source's own text so the guard
# then compares the same words it was always meant to compare.
#
# `relocate_evidence_quotes` is the only entry point; everything below it is a private
# implementation detail. It is deliberately not a guard itself and never touches `status`:
# it only ever rewrites the three quote-bearing fields the guards read next.
# --------------------------------------------------------------------------------------

#: A segment shorter than this many tokens is never relocated (module docstring above):
#: too little text to anchor a search on without risking a false, unrelated match. Measured
#: on nothing -- an a priori floor, in the same spirit as `QUOTE_SEGMENT_MIN_CHARS`.
#:
#: Word-level tolerance is abandoned entirely, after every
#: word-level allowlist, however small, turned out to contain a meaning-flipping pair.
#: `QUOTE_RELOCATION_MIN_SEGMENT_TOKENS_FOR_EDITS` and
#: `QUOTE_RELOCATION_MAX_EDIT_DISTANCE` (an edit-*count* budget) are deleted, not kept
#: dormant: there is no longer a numeric budget to cap, because the rule below never accepts
#: an edit by virtue of being small -- only by virtue of being a member of one of two
#: classes that cannot change what a sentence asserts, however many of them a quote needs.
#: The effective rule, stated once here and in `relocate_evidence_quotes`'s own docstring:
#: **at least 8 tokens, any number of neutral-class edits, nothing else.**
#:
#: The British/American
#: spelling class (the doubled-l
#: fold's `filled`/`filed`, the digraph fold's `shoe`/`she`) was found to merge distinct
#: English words and is deleted outright, along with
#: its fold function and its four fold-table constants -- not narrowed, the way a
#: content-word allowlist narrowed to articles alone still turned out to hide an
#: antonym pair. There are now exactly two neutral classes: articles, and one dropped
#: parenthetical citation. A spelling difference is left to the bounded second pass
#: (`verify_claim_with_policy`) like any other single-word difference.
QUOTE_RELOCATION_MIN_SEGMENT_TOKENS = 8

#: How many tokens longer than the model's own segment a candidate window may be, to leave
#: room for one dropped parenthetical citation (neutral class 2 below) -- a citation of more
#: than this many tokens is not considered droppable. Not tuned on any dataset: a generous,
#: documented ceiling on how large a single bracketed aside this rule will ever look inside,
#: not a policy cap on how many edits a quote may carry (there is none; see above).
QUOTE_RELOCATION_MAX_CITATION_TOKENS = 12

#: How many tokens shorter than the model's own segment a candidate window may be, to leave
#: room for one or more dropped articles (neutral class 1: the model's quote carried an
#: article -- "a"/"an"/"the" -- the source does not have at that position, so the correct
#: window is genuinely shorter by one token per dropped article). Not a general licence to
#: shrink: `_classify_relocation_alignment` only ever accepts a shorter window when every
#: token the shrink removes is, on its own, a single-token run that is an article; a window
#: missing any other, non-article token is refused regardless of how small the shrink is
#: (this is what keeps a shorter window from silently truncating content the model actually
#: wrote, the failure mode the window-length search itself used to allow).
QUOTE_RELOCATION_MAX_WINDOW_SHRINK = 6

#: Search-only slack added to the window-length delta when bounding the candidate search
#: (`_windows_within_distance`'s own `threshold` argument). This is a performance bound on
#: what the *search* looks at, never the acceptance rule: a window this pulls in still has
#: every one of its edits classified by `_classify_relocation_alignment` below, and is
#: rejected the instant one of them is not a member of a neutral class, regardless of how
#: small its raw token-edit distance is.
QUOTE_RELOCATION_SEARCH_SLACK = 6

#: Whole-word negations checked against a token's own raw (unfolded) text, case-folded --
#: never against `_normalise_for_match`'s letters-and-digits key, which would also fold
#: away the apostrophe a "n't" contraction is recognised by below.
#:
#: The original seven words missed most polarity markers a source sentence can use, so a
#: quote that reversed a source's own finding (a dropped or inserted negation) could still
#: relocate and pass. Extended to every lexical negation found to be a gap;
#: `lack`/`fail`/`absen` are handled as prefixes below (`lacked`, `failing`,
#: `absence`, `absent`, ... all start with one of them), not spelled out here.
_NEGATION_WORDS = frozenset(
    {
        "no", "not", "never", "none", "nor", "without", "cannot",
        "neither", "non", "nothing", "nobody", "nowhere",
        "unable", "rarely", "hardly", "barely", "unless", "except",
    }
)

#: Whole-word negation prefixes: any token that
#: *starts with* one of these is a negation by itself, covering every inflection
#: (`lack`/`lacks`/`lacked`/`lacking`, `fail`/`fails`/`failed`/`failing`,
#: `absent`/`absence`/`absently`) without enumerating each form.
_NEGATION_WORD_PREFIXES = ("lack", "fail", "absen")

#: Apostrophe characters `_tokens_with_offsets` lets continue a run of letters/digits, so a
#: contraction ("didn't") or a possessive ("authors'") is one token, not two -- the same
#: token a reader would call one word, and the only way a "n't" contraction's negation
#: survives to `_is_negation_token` below (letters-and-digits folding elsewhere in this
#: module drops the apostrophe entirely, which is right for substring matching but would
#: erase the very thing that makes "didn't" a negation and "did" not one).
_TOKEN_APOSTROPHES = ("'", "’")

#: Leading negating affixes: a token that is another
#: token plus one of these prefixes reverses that other token's meaning ("changed" ->
#: "unchanged", "significant" -> "insignificant", "effective" -> "ineffective", "possible"
#: -> "impossible") without either token being in `_NEGATION_WORDS` on its own -- checked
#: pairwise between a substitution's two sides by `_tokens_differ_only_by_negation_prefix`,
#: not by `_is_negation_token`, which only ever looks at one token in isolation.
_NEGATION_AFFIXES = ("non", "un", "in", "im")

#: Spelled-out cardinals and ordinals: a numeral written
#: in words ("three") carries no digit, so `_token_contains_digit` alone does not protect it
#: the way it protects "3". Checked case-folded, like `_NEGATION_WORDS`.
#:
#: The original list ran one to twenty, then
#: thirty/forty/fifty and stopped, so a decade-to-decade misquote ("sixty" for "ninety") with
#: both sides unlisted relocated and passed. Completed with the remaining decades, the large
#: magnitudes, the ordinals through "twentieth", and the plural magnitude/fraction forms.
_NUMBER_WORDS = frozenset(
    {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion",
        "dozen", "score", "half", "third", "quarter", "twice", "double",
        "first", "second", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
        "seventeenth", "eighteenth", "nineteenth", "twentieth",
        "hundreds", "thousands", "millions", "halves", "thirds", "quarters",
    }
)

#: Neutral class 1 (articles): the only word-membership list this rule still carries, and
#: closed to exactly these three tokens: an allowlist of any other part of speech -- even a
#: "tiny", carefully-curated one -- kept turning up an antonym pair on both sides of a
#: licensed substitution; measurement found fifteen such pairs surviving a closed class of
#: about a hundred words. A substitution *within* this set (`_is_relocation_article` true on
#: both sides), or an insertion or deletion of a single token that is a member, is accepted;
#: nothing else about an article -- its case, or which of the three it is -- matters.
_RELOCATION_ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})

#: Neutral class 2 (a dropped parenthetical citation): which opening bracket must be matched
#: by which closing one, checked by `_run_sits_in_brackets`.
_CITATION_BRACKET_PAIRS: dict[str, str] = {"(": ")", "[": "]"}

#: A four-digit year between 1900 and 2099, one of the two token-level markers `_run_
#: contains_citation_marker` recognises inside a bracketed run (the other is an "et"/"al"
#: pair) -- necessary but not sufficient on its own for `_run_has_citation_shape` to accept
#: the run as a citation.
_CITATION_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

#: A title-case token ("Smith", "Truscott", "Sheen") -- one capital followed by only lower
#: case letters, so an all-caps token ("SMITH") does not count -- the surname-like shape
#: `_run_has_citation_shape` requires beside a year, unless the run is instead an "et"/"al"
#: pair or a bracketed numeric-only reference.
_CITATION_SURNAME_RE = re.compile(r"^[A-Z][a-z]*$")


def _is_negation_token(raw_token: str) -> bool:
    """True for a whole negation word (or word carrying a negation prefix such as
    `lack-`/`fail-`/`absen-`, `_NEGATION_WORD_PREFIXES`), or a "n't" contraction (either
    apostrophe), by its own raw text -- checked before any letters-and-digits folding
    removes the apostrophe that makes "isn't" a negation and "is" not one."""
    folded = raw_token.casefold()
    if folded in _NEGATION_WORDS:
        return True
    if any(folded.endswith(f"n{mark}t") for mark in _TOKEN_APOSTROPHES):
        return True
    return any(folded.startswith(prefix) for prefix in _NEGATION_WORD_PREFIXES)


def _tokens_differ_only_by_negation_prefix(raw_a: str, raw_b: str) -> bool:
    """True when one of *raw_a*/*raw_b*, case-folded, equals the other with a leading
    negating affix removed (`_NEGATION_AFFIXES`) -- e.g. "changed"/"unchanged",
    "significant"/"insignificant", "effective"/"ineffective", "possible"/"impossible": a
    substitution edit-distance treats as one cheap edit but that reverses the source's own
    meaning (WP-B25 relocation review, blocker 2)."""
    a, b = raw_a.casefold(), raw_b.casefold()
    if a == b:
        return False
    for shorter, longer in ((a, b), (b, a)):
        if any(longer == prefix + shorter for prefix in _NEGATION_AFFIXES):
            return True
    return False


def _is_protected_number_word(raw_token: str) -> bool:
    """True for a spelled-out cardinal or ordinal (`_NUMBER_WORDS`, WP-B25 relocation
    review, major 3) -- the word form of a numeral, which `_token_contains_digit` cannot
    see."""
    return raw_token.casefold() in _NUMBER_WORDS


def _is_relocation_article(raw_token: str) -> bool:
    """True when *raw_token*, case-folded, is a member of neutral class 1
    (`_RELOCATION_ARTICLES`). Tested on the raw token, the same way `_is_negation_token` is:
    a homoglyph or mixed-script look-alike case-folds to a different string and so is simply
    not a member, with no separate confusable-folding step needed (WP-B25 relocation review,
    round 2 minor 4)."""
    return raw_token.casefold() in _RELOCATION_ARTICLES


def _token_is_citation_year(raw_token: str) -> bool:
    """True when *raw_token* is exactly four digits between 1900 and 2099 (`_CITATION_YEAR_
    RE`) -- one of the two token-level markers `_run_has_citation_shape` looks for inside a
    bracketed run (the other is an "et"/"al" pair); necessary but not sufficient on its own
    for the run to be recognised as neutral class 2, a dropped parenthetical citation."""
    return bool(_CITATION_YEAR_RE.match(raw_token))


def _looks_like_citation_surname(raw_token: str) -> bool:
    """True when *raw_token* looks like a capitalised surname token (`_CITATION_SURNAME_RE`)
    -- the shape `_run_has_citation_shape` requires beside a citation year, unless the run is
    instead an "et"/"al" pair or a bracketed numeric-only reference."""
    return bool(_CITATION_SURNAME_RE.match(raw_token))


def _run_contains_et_al_pair(raw_tokens: Sequence[str]) -> bool:
    """True when *raw_tokens* contains an "et" token immediately followed by "al"
    (case-folded)."""
    for index in range(len(raw_tokens) - 1):
        if raw_tokens[index].casefold() == "et" and raw_tokens[index + 1].casefold() == "al":
            return True
    return False


def _run_is_numeric_only(raw_tokens: Sequence[str]) -> bool:
    """True when *raw_tokens* is non-empty and every token is made up of digits only -- the
    shape of a bracketed numbered reference such as ``[12]`` or ``[3, 4]`` (the tokeniser
    drops the comma and brackets themselves, `_tokens_with_offsets`), where there is no
    surname to require and no separate "content" left to check for disqualifying digits,
    number words or negations: every token in a numeric-only run *is* the citation's own
    reference number."""
    return bool(raw_tokens) and all(token.isdigit() for token in raw_tokens)


def _run_has_citation_shape(raw_tokens: Sequence[str]) -> bool:
    """True when *raw_tokens* (a candidate dropped run) has the shape of a real citation,
    not merely the presence of a year (WP-B25 relocation review round 4 minor 2): an "et"/
    "al" pair (`_run_contains_et_al_pair`), a bracketed numeric-only reference (`_run_is_
    numeric_only`, e.g. ``[12]`` or ``[3, 4]``), or a citation year (`_token_is_citation_
    year`) together with a capitalised surname-like token (`_looks_like_citation_surname`)
    somewhere in the same run.

    Documented residual: a bracketed run that is a bare year and nothing else (``(2020)``)
    is indistinguishable, at this token level, from a bracketed numeric reference like
    ``[12]`` -- both satisfy `_run_is_numeric_only` -- so it is accepted here too, the same
    way a British/American spelling pair folding to a coincidentally identical string was an
    accepted residual of the neutral class this round deleted. `_is_dropped_parenthetical_
    citation`'s own disqualifying-content check (`_run_has_disqualifying_content`) still
    applies before either shape is exempted."""
    if _run_contains_et_al_pair(raw_tokens):
        return True
    if _run_is_numeric_only(raw_tokens):
        return True
    return any(_token_is_citation_year(token) for token in raw_tokens) and any(
        _looks_like_citation_surname(token) for token in raw_tokens
    )


def _run_has_disqualifying_content(raw_tokens: Sequence[str]) -> bool:
    """True when *raw_tokens* -- a run `_run_has_citation_shape` has already accepted --
    still carries a digit, a spelled-out number word or a negation somewhere other than its
    own recognised citation year (WP-B25 relocation review round 4 minor 2: defence in depth,
    applied to the run's tokens before the citation exemption is granted, the same
    digit/number-word/negation checks already applied to every other alignment step). A
    numeric-only run (`_run_is_numeric_only`) is exempt from this check entirely: every one
    of its tokens *is* the citation's own reference number, not disqualifying content beside
    one.

    `_is_dropped_parenthetical_citation` calls `_run_has_citation_shape` first, so this is
    only ever reached for a run that already looks like a citation; the review's four
    measured refusals (`"(no effect was seen after 2019)"`, `"[this reversed after 2021 in
    transport]"`, `"(from 42 percent in 2011 to 38 percent)"`, `"(1950 to 1990)"`) are in
    fact all refused earlier, by the shape check itself (none carries a surname beside its
    year, an "et"/"al" pair, or an all-digit run) -- this function is the second, independent
    layer the same review round required, guarding a shape-passing run that also happens to
    carry a genuine content digit, number word or negation beside its citation year (for
    example `"(Smith, 2020, but not confirmed)"`), which no measured case in the review
    exercises but which the digit/number-word/negation checks refuse the same way they
    refuse it for every other alignment step. `"(Truscott, 1996)"`, `"(Sheen et al.,
    2009)"`, `"(Mao, Lee and Li, 2024)"` and `"[12]"` all pass both checks."""
    if _run_is_numeric_only(raw_tokens):
        return False
    for token in raw_tokens:
        if _token_is_citation_year(token):
            continue
        if (
            _token_contains_digit(token)
            or _is_protected_number_word(token)
            or _is_negation_token(token)
        ):
            return True
    return False


def _run_sits_in_brackets(chunk_text: str, start_offset: int, end_offset: int) -> bool:
    """True when the raw span ``chunk_text[start_offset:end_offset]`` is immediately preceded
    (modulo whitespace) by an opening bracket and immediately followed (modulo whitespace) by
    its matching closing bracket (`_CITATION_BRACKET_PAIRS`) -- i.e. the span is the *whole*
    content of one bracketed aside in the raw source, not merely part of one."""
    before = start_offset - 1
    while before >= 0 and chunk_text[before].isspace():
        before -= 1
    if before < 0 or chunk_text[before] not in _CITATION_BRACKET_PAIRS:
        return False
    closing = _CITATION_BRACKET_PAIRS[chunk_text[before]]
    length = len(chunk_text)
    after = end_offset
    while after < length and chunk_text[after].isspace():
        after += 1
    return after < length and chunk_text[after] == closing


def _is_dropped_parenthetical_citation(
    window_raw: Sequence[str],
    window_tokens: Sequence[tuple[str, int, int]],
    chunk_text: str,
    run_start: int,
    run_end: int,
) -> bool:
    """True when ``window_raw[run_start:run_end + 1]`` -- tokens the source (the window)
    carries that the model's quote (the segment) does not -- is neutral class 2: a maximal
    token run that has the shape of a real citation (`_run_has_citation_shape`, WP-B25
    relocation review round 4 minor 2: a bracketed year on its own is no longer enough),
    carries no digit, spelled-out number word or negation beside that shape (`_run_has_
    disqualifying_content`, checked *before* the exemption below is granted, not after), and
    sits inside one whole bracketed aside in the raw source (`_run_sits_in_brackets`).
    Accepted as *one* edit however many tokens the run spans (module docstring): dropping a
    citation when quoting a sentence around it cannot change what the sentence asserts,
    regardless of how many author names or how long a year list it carries.

    Deliberately one-directional (WP-B25 relocation review round 4, Part A: "deletion, never
    insertion"): this is only ever consulted for a run the *window* carries and the *segment*
    lacks (the model's quote dropped a citation that is genuinely in the source). The reverse
    -- the model's own quote carrying a bracketed citation the source does not have -- is
    never given this exemption; `_classify_relocation_alignment` never calls this helper for
    that direction, so a fabricated citation is refused by the ordinary protected-token and
    closed-class checks like any other content the segment alone carries."""
    run_raw = window_raw[run_start : run_end + 1]
    if not _run_has_citation_shape(run_raw):
        return False
    if _run_has_disqualifying_content(run_raw):
        return False
    start_offset = window_tokens[run_start][1]
    end_offset = window_tokens[run_end][2]
    return _run_sits_in_brackets(chunk_text, start_offset, end_offset)


def _token_contains_digit(raw_token: str) -> bool:
    """True when *raw_token* carries at least one digit ("2021", "L2", "50%" -> "50")."""
    return any(ch.isdigit() for ch in raw_token)


def _tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Every maximal run of letters and digits in *text*, as ``(raw_token, start, end)``
    (``end`` exclusive, both raw indices into *text*): the "letters-and-digits fold" the
    relocation rule tokenises on, kept as whole words rather than `_keep_letters_and_digits`'s
    single fused string, and with the raw indices `_normalised_with_offsets` and
    `fold_with_offsets` already use elsewhere in this module for the same reason: to be
    able to recover the source's own raw text a token (or a run of them) came from.

    A single apostrophe continues a run when a letter follows it (`_TOKEN_APOSTROPHES`), so
    a contraction or a possessive is one token: "didn't" is the run ``"didn't"``, not the
    two runs ``"didn"`` and ``"t"`` a plain letters-and-digits split would produce, which
    would separate a negation from the very letters that mark it as one.
    """
    tokens: list[tuple[str, int, int]] = []
    length = len(text)
    index = 0
    start: int | None = None
    while index < length:
        character = text[index]
        if character.isalpha() or character.isdigit():
            if start is None:
                start = index
            index += 1
            continue
        if (
            start is not None
            and character in _TOKEN_APOSTROPHES
            and index + 1 < length
            and text[index + 1].isalpha()
        ):
            index += 1
            continue
        if start is not None:
            tokens.append((text[start:index], start, index))
            start = None
        index += 1
    if start is not None:
        tokens.append((text[start:length], start, length))
    return tokens


def _token_edit_distance_within(a: Sequence[str], b: Sequence[str], limit: int) -> int | None:
    """Levenshtein distance between token-key sequences *a* and *b*, or ``None`` as soon as
    it is certain to exceed *limit* -- a cheap accept/reject check for the window search
    below, which runs this (and the multiset bound in `_windows_within_distance`) on many
    candidate windows for every one it actually accepts. `_token_alignment` below computes
    the same distance again, once, only for a window this already accepted, to recover the
    edit-by-edit alignment the digit/negation check needs.
    """
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > limit:
        return None
    previous = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        current = [i] + [0] * len_b
        row_min = current[0]
        token_a = a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if token_a == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if current[j] < row_min:
                row_min = current[j]
        if row_min > limit:
            return None
        previous = current
    distance = previous[len_b]
    return distance if distance <= limit else None


def _token_alignment(a: Sequence[str], b: Sequence[str]) -> list[tuple[int | None, int | None]]:
    """The lowest-cost token-level edit path from *a* to *b*, as one ``(i, j)`` pair per
    step: both present is a match (``a[i] == b[j]``) or a substitution, only *i* present is
    a deletion, only *j* present is an insertion. Ties are broken match-over-substitution,
    then deletion, then insertion, so two sequences that need no edit at all are reported
    as nothing but matches, never as a substitution of a token by an equal one.

    Run once per window `_windows_within_distance` already accepted on distance alone,
    never during the search itself (`_token_edit_distance_within` is what that scans with);
    ``len(a)`` and ``len(b)`` are a segment's and one window's token counts, always small.
    """
    len_a, len_b = len(a), len(b)
    table = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(1, len_a + 1):
        table[i][0] = i
    for j in range(1, len_b + 1):
        table[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost
            )
    path: list[tuple[int | None, int | None]] = []
    i, j = len_a, len_b
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if table[i][j] == table[i - 1][j - 1] + cost:
                path.append((i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and table[i][j] == table[i - 1][j] + 1:
            path.append((i - 1, None))
            i -= 1
            continue
        path.append((None, j - 1))
        j -= 1
    path.reverse()
    return path


def _classify_relocation_alignment(
    segment_raw: Sequence[str],
    window_raw: Sequence[str],
    segment_keys: Sequence[str],
    window_keys: Sequence[str],
    window_tokens: Sequence[tuple[str, int, int]],
    chunk_text: str,
) -> int | None:
    """Classify every non-matching step of `_token_alignment`'s lowest-cost path between
    *segment* and one candidate *window*: the number of accepted neutral edits (a whole
    dropped citation run counts once, however many tokens it spans -- module docstring), or
    ``None`` the instant any step is not a member of one of the two neutral classes (WP-B25
    relocation review round 5: the spelling-variant class the round 4 review's major finding
    measured merging distinct English words -- ``filled``/``filed``, ``shoe``/``she`` -- is
    deleted outright, not narrowed; a British or American spelling difference now falls to
    the bounded second pass, `verify_claim_with_policy`, like any other word difference) or
    trips a protected-token check kept as defence in depth.

    A rejection anywhere refuses the *whole* window (no partial acceptance): "any other
    non-matching alignment step refuses relocation" (round 4 authorisation). Runs of
    consecutive insertion/deletion steps (a token present on only one side) are grouped and
    classified together, since a dropped citation is one edit spanning several tokens, not
    several one-token edits.

    Order per step:

    1. A match (``segment_keys[i] == window_keys[j]``) is free.
    2. A substitution (both present, keys differ): the digit/number-word/negation/negation-
       prefix protections fire first, as defence in depth (every one of those tokens is
       already excluded from every neutral class below, so this is redundant today, but it
       keeps working unchanged if a neutral class is ever edited). Otherwise accepted only
       when both sides are articles (neutral class 1, `_is_relocation_article`); anything
       else refuses.
    3. A run the segment has and the window lacks (the model's quote carries something the
       source does not -- an inserted citation is exactly this shape): accepted only when the
       run is a single article token; a longer run, or a single non-article token, refuses
       after the same defence-in-depth checks.
    4. A run the window has and the segment lacks (the model's quote dropped something the
       source has): accepted when the whole run is a dropped parenthetical citation (neutral
       class 2, `_is_dropped_parenthetical_citation`) or a single article token; otherwise
       refuses after the same defence-in-depth checks.
    """
    path = _token_alignment(segment_keys, window_keys)
    edit_count = 0
    index = 0
    total = len(path)
    while index < total:
        i, j = path[index]
        if i is not None and j is not None and segment_keys[i] == window_keys[j]:
            index += 1
            continue
        if i is not None and j is not None:
            seg_tok, win_tok = segment_raw[i], window_raw[j]
            if (
                _token_contains_digit(seg_tok)
                or _token_contains_digit(win_tok)
                or _is_protected_number_word(seg_tok)
                or _is_protected_number_word(win_tok)
                or _is_negation_token(seg_tok)
                or _is_negation_token(win_tok)
                or _tokens_differ_only_by_negation_prefix(seg_tok, win_tok)
            ):
                return None
            if _is_relocation_article(seg_tok) and _is_relocation_article(win_tok):
                edit_count += 1
                index += 1
                continue
            return None
        if j is None:
            run_start = index
            while index < total and path[index][1] is None and path[index][0] is not None:
                index += 1
            run = [path[k][0] for k in range(run_start, index)]
            if len(run) == 1 and _is_relocation_article(segment_raw[run[0]]):
                edit_count += 1
                continue
            for pos in run:
                if (
                    _token_contains_digit(segment_raw[pos])
                    or _is_protected_number_word(segment_raw[pos])
                    or _is_negation_token(segment_raw[pos])
                ):
                    return None
            return None
        run_start = index
        while index < total and path[index][0] is None and path[index][1] is not None:
            index += 1
        run = [path[k][1] for k in range(run_start, index)]
        if _is_dropped_parenthetical_citation(
            window_raw, window_tokens, chunk_text, run[0], run[-1]
        ):
            edit_count += 1
            continue
        if len(run) == 1 and _is_relocation_article(window_raw[run[0]]):
            edit_count += 1
            continue
        for pos in run:
            if (
                _token_contains_digit(window_raw[pos])
                or _is_protected_number_word(window_raw[pos])
                or _is_negation_token(window_raw[pos])
            ):
                return None
        return None
    return edit_count


def _windows_within_distance(
    segment_keys: Sequence[str], chunk_keys: Sequence[str], window_len: int, threshold: int
) -> Iterator[tuple[int, int]]:
    """``(start, distance)`` for every *window_len*-token window of *chunk_keys* whose
    token edit distance from *segment_keys* is at most *threshold*.

    A window is only handed to the exact (capped) check when a cheap multiset lower bound
    -- ``max(len(segment_keys), window_len) - |shared tokens|``, valid because every
    insertion, deletion or substitution can remove at most one token the two sides do not
    already share -- does not already rule it out. That bound is kept incrementally (one
    token added, one removed) as the window slides, so scanning a whole chunk of ``C``
    tokens for one window length costs ``O(C)`` before any exact edit-distance check runs,
    not ``O(C * window_len)``: the difference between checking a 40-chunk paper per
    relocated segment costing a few hundred cheap counter updates, and costing tens of
    millions of Levenshtein cells.
    """
    if window_len < 1 or window_len > len(chunk_keys):
        return
    segment_counts = Counter(segment_keys)
    window_counts: Counter[str] = Counter()
    common = 0

    def _add(token: str) -> None:
        nonlocal common
        if window_counts[token] < segment_counts[token]:
            common += 1
        window_counts[token] += 1

    def _drop(token: str) -> None:
        nonlocal common
        window_counts[token] -= 1
        if window_counts[token] < segment_counts[token]:
            common -= 1

    for token in chunk_keys[:window_len]:
        _add(token)
    start = 0
    while True:
        lower_bound = max(len(segment_keys), window_len) - common
        if lower_bound <= threshold:
            window = chunk_keys[start : start + window_len]
            distance = _token_edit_distance_within(segment_keys, window, threshold)
            if distance is not None:
                yield start, distance
        next_start = start + 1
        if next_start + window_len > len(chunk_keys):
            return
        _drop(chunk_keys[start])
        _add(chunk_keys[start + window_len])
        start = next_start


#: Straight double quote, left curly quote, right curly quote: the three characters
#: `_QUOTED_FRAGMENT_RE` pairs interchangeably. A splice that changes how
#: many of these a quote carries can silently re-pair every fragment after it.
_QUOTE_DELIMITER_CHARS: tuple[str, ...] = ('"', "“", "”")


def _delimiter_counts(text: str) -> Counter[str]:
    """How many of each `_QUOTE_DELIMITER_CHARS` character *text* carries."""
    return Counter(ch for ch in text if ch in _QUOTE_DELIMITER_CHARS)


def _best_relocation_window(
    segment: str, chunk_texts: Sequence[str]
) -> tuple[int, int, str] | None:
    """``(edit_count, chunk_index, replacement_text)`` for the window, across every chunk,
    with the fewest accepted neutral edits (`_classify_relocation_alignment`) among every
    window that has one accepted at all -- or ``None`` when no window does.

    *segment* is the raw text of the quote segment being relocated (tokenised internally,
    `_tokens_with_offsets`); the whole segment is replaced as one unit (`_relocate_one_quote`).
    Ties on edit count keep whichever window was found first, scanning chunks in their given
    order and, within a chunk, window lengths from shortest to longest and start positions
    left to right -- deterministic, so relocating the same input twice always relocates it
    the same way (idempotence and reproducibility both depend on this).

    Window lengths run from ``QUOTE_RELOCATION_MAX_WINDOW_SHRINK`` tokens shorter than the
    segment (round 4, Part A: the model's quote carried an article the source does not have
    at that position, so the correct window is genuinely shorter) up to
    ``QUOTE_RELOCATION_MAX_CITATION_TOKENS`` tokens longer (dropping a parenthetical citation
    makes the source's own window longer than the model's shortened quote). Neither bound is
    the acceptance rule, only a search-cost bound on what is *looked at*:
    `_windows_within_distance`'s numeric threshold here (``QUOTE_RELOCATION_SEARCH_SLACK``
    plus the window's own length delta) is likewise a search-cost bound only. Every candidate
    either bound admits is still classified by `_classify_relocation_alignment`, which is the
    sole authority on whether a window is accepted -- and rejects the instant one non-matching
    step is not a member of a neutral class, so a shorter window is only ever accepted when
    every token it drops is, on its own, a licensed article deletion (never a content word,
    which is what kept a shrink from silently truncating the stored quote before this window-
    length search existed, WP-B25 relocation review major 4): there is no cap on how many
    neutral edits one window may carry.

    Round 3 blocker 2: a candidate whose recovered raw slice would change how many
    `_QUOTE_DELIMITER_CHARS` the segment carries is refused outright (`_delimiter_counts`),
    since `_QUOTED_FRAGMENT_RE` re-pairs those characters across the whole quote, not just
    the spliced span. The two counts are compared unconditionally (WP-B25 relocation review
    round 4 minor 4: the earlier ``if replacement_delimiters and ...`` was one-sided against
    this same docstring, refusing a candidate that would *add* a delimiter the segment lacks
    but not one that would *remove* a delimiter the segment carries).

    The recovered raw slice absorbs any combining marks (`unicodedata.combining`)
    immediately following the window's last token (WP-B25 relocation review, minor 6): a
    combining mark is neither a letter nor a digit, so it ends a token run the same way
    whitespace does, and without this a base letter composed with a following mark under
    NFKC (`_normalise_for_match`) would make the raw slice one code point short of the
    chunk's own folded text, so the "verbatim by construction" replacement would not
    actually be verbatim.
    """
    segment_tokens = _tokens_with_offsets(segment)
    segment_raw = [token for token, _start, _end in segment_tokens]
    segment_keys = [_normalise_for_match(token) for token in segment_raw]
    token_count = len(segment_keys)
    if token_count < QUOTE_RELOCATION_MIN_SEGMENT_TOKENS:
        return None
    segment_delimiters = _delimiter_counts(segment)

    best: tuple[int, int, str] | None = None
    best_edit_count: int | None = None
    min_window_len = max(1, token_count - QUOTE_RELOCATION_MAX_WINDOW_SHRINK)
    max_window_len = token_count + QUOTE_RELOCATION_MAX_CITATION_TOKENS
    # Ordered so a tie on edit count prefers the window closest to the segment's own length,
    # and prefers growing over shrinking at the same distance: [n, n+1, n-1, n+2, n-2, ...].
    # Without this, a shorter window that ties on edit count with the correct same-length
    # window (e.g. dropping one article from the *front* of the segment ties with correctly
    # substituting it) could be "found first" under a plain ascending scan and silently
    # truncate content the model actually wrote -- the exact failure mode the window-length
    # search's shrink allowance (`QUOTE_RELOCATION_MAX_WINDOW_SHRINK`) must not reopen.
    window_lengths = [token_count]
    max_delta = max(
        max_window_len - token_count, token_count - min_window_len, 0
    )
    for delta in range(1, max_delta + 1):
        grown = token_count + delta
        if grown <= max_window_len:
            window_lengths.append(grown)
        shrunk = token_count - delta
        if shrunk >= min_window_len:
            window_lengths.append(shrunk)
    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk_tokens = _tokens_with_offsets(chunk_text)
        chunk_raw = [token for token, _start, _end in chunk_tokens]
        chunk_keys = [_normalise_for_match(token) for token in chunk_raw]
        for window_len in window_lengths:
            if window_len > len(chunk_keys):
                continue
            threshold = abs(window_len - token_count) + QUOTE_RELOCATION_SEARCH_SLACK
            for start, _distance in _windows_within_distance(
                segment_keys, chunk_keys, window_len, threshold
            ):
                window_raw = chunk_raw[start : start + window_len]
                window_keys = chunk_keys[start : start + window_len]
                window_tokens = chunk_tokens[start : start + window_len]
                edit_count = _classify_relocation_alignment(
                    segment_raw, window_raw, segment_keys, window_keys, window_tokens, chunk_text
                )
                if edit_count is None:
                    continue
                if best_edit_count is not None and edit_count >= best_edit_count:
                    continue
                raw_start = window_tokens[0][1]
                raw_end = window_tokens[-1][2]
                while raw_end < len(chunk_text) and unicodedata.combining(chunk_text[raw_end]):
                    raw_end += 1
                replacement = chunk_text[raw_start:raw_end]
                replacement_delimiters = _delimiter_counts(replacement)
                if replacement_delimiters != segment_delimiters:
                    continue
                best_edit_count = edit_count
                best = (edit_count, chunk_index, replacement)
    return best


def _segment_already_verbatim(segment: str, haystacks: Sequence[str]) -> bool:
    """True when *segment* is already a verbatim substring of some chunk, exactly the test
    `_guard_quote_fidelity` itself makes -- a segment that already passes needs no
    relocation, and relocating it anyway (onto the very text it already equals, after
    folding) would be a costly no-op at best and a way to accept a coincidental
    near-duplicate elsewhere in the source at worst."""
    needle = _normalise_for_match(segment)
    return bool(needle) and any(needle in haystack for haystack in haystacks)


def _relocate_one_quote(
    quote: str | None,
    haystacks: Sequence[str],
    chunk_texts: Sequence[str],
    diagnostics: list[dict[str, Any]],
) -> str | None:
    """*quote* with every one of `_quote_segments`' segments that is not already verbatim
    replaced by its best-accepted relocation window (`_best_relocation_window`), appending
    one ``quote_relocated`` diagnostic per segment actually relocated. A segment with no
    accepted window, or already verbatim, is left exactly as it was.

    Round 3 blocker 2 post-condition: a splice is discarded, and the segment left untouched,
    unless every OTHER segment `_quote_segments` found in the original *quote* -- one not yet
    relocated by an earlier iteration of this same loop -- still appears, verbatim, among
    `_quote_segments(candidate)` after the splice. A splice that would silently change how
    `_quote_segments` re-parses some other, untouched span of the same quote (e.g. by shifting
    the count of `_QUOTE_DELIMITER_CHARS` characters -- `_best_relocation_window`'s own check
    catches the common case, this catches any other way the same failure mode could occur) is
    never applied.
    """
    if not quote:
        return quote
    updated = quote
    original_segments = _quote_segments(quote)
    relocated_originals: set[str] = set()
    for segment in original_segments:
        if _segment_already_verbatim(segment, haystacks):
            continue
        result = _best_relocation_window(segment, chunk_texts)
        if result is None:
            continue
        edit_count, chunk_index, replacement = result
        position = updated.find(segment)
        if position == -1:
            # `_quote_segments`' segments are disjoint spans of the original quote, so this
            # should not happen; skip rather than guess at a different span to replace.
            continue
        candidate = updated[:position] + replacement + updated[position + len(segment) :]
        other_segments = [
            other
            for other in original_segments
            if other != segment and other not in relocated_originals
        ]
        candidate_segments = set(_quote_segments(candidate))
        if not all(other in candidate_segments for other in other_segments):
            continue
        updated = candidate
        relocated_originals.add(segment)
        diagnostics.append(
            {
                "type": "quote_relocated",
                "original": segment,
                "replacement": replacement,
                "edit_distance": edit_count,
                "chunk_index": chunk_index,
            }
        )
    return updated


def attribution_guard_fires_before_relocation(
    claim_text: str,
    chunk_texts: Sequence[str],
    evidence_quote: str | None,
) -> bool:
    """True when guard 1 (`_guard_attribution`) already caps the claim to ``unsupported``
    on the model's own, pre-relocation quote.

    WP-B25 relocation review (`.superpowers/sdd/wpB25-relocation-review.md`, blocker 1):
    `_guard_attribution` stands down (does not cap) when the quote is *located* in some
    chunk (`_quote_located`), and relocation's whole job is to turn a not-located quote into
    a located one -- so, unguarded, relocation would silently manufacture guard 1's own
    exemption for exactly the `wrong_paper` claims the guard exists to catch. A caller
    calls this on the quote *before* calling `relocate_evidence_quotes` and skips relocation
    entirely for the claim when it returns ``True``, so the exemption still has to be earned
    by the model's own quote, never by a relocation the guard's own coverage check never saw.

    The ``status`` passed to `_guard_attribution` here is a placeholder ("verified"): guard
    1 only ever reads it to decide what to cap *into*, never whether it fires, so any
    starting status yields the same answer to "did guard 1 fire".
    """
    _status, reasons = _guard_attribution(
        "verified",
        claim_text=claim_text,
        chunk_texts=chunk_texts,
        evidence_quote=evidence_quote,
    )
    return "attribution_mismatch" in reasons


def relocate_evidence_quotes(
    evidence_quote: str | None,
    evidence_quotes: Sequence[str] | None,
    assertions: Sequence[Mapping[str, Any]] | None,
    chunk_texts: Sequence[str],
) -> tuple[str | None, list[str] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Repoint an almost-verbatim quote at the source's own text, before any guard reads it.

    A separate, versioned step (module comment above; the rule itself rewritten for WP-B25
    relocation review round 4, Part A, task authorisation 2026-09-10, and narrowed again in
    round 5): for every segment `_quote_segments` yields, from ``evidence_quote``, every span
    of ``evidence_quotes`` and every assertion's ``quote``/``quotes``, that is not already a
    verbatim substring of a chunk, this looks for a run of chunk tokens that differs from the
    segment by nothing but neutral edits. **Effective rule: at least 8 tokens
    (``QUOTE_RELOCATION_MIN_SEGMENT_TOKENS``), any number of neutral-class edits, nothing
    else** (`_classify_relocation_alignment`'s own docstring states the same rule against
    the code): an article substituted, inserted or dropped (``a``/``an``/``the``,
    `_is_relocation_article`), or one whole parenthetical citation dropped from the quote
    (never inserted, `_is_dropped_parenthetical_citation`) -- counted as one edit however
    many tokens it spans. Word-level tolerance for anything else is deliberately absent:
    round 4's own three review rounds each found a meaning-flipping pair surviving whatever
    allowlist the previous round shipped, and round 5 deleted the British/American spelling
    class round 4 had introduced after its own review found it merging distinct English
    words (`filled`/`filed`, `shoe`/`she`) -- so this rule accepts no word-membership list
    beyond the three articles, and a spelling difference is now resolved the same way any
    other single-word difference is, by the bounded second pass below. Digit,
    spelled-out-number and negation protections are kept as defence in depth. When a
    window is found, the segment is replaced *inside the quote* by the source's own raw
    text for that window -- so the stored quote becomes verbatim by construction and
    `_guard_quote_fidelity` passes it the same way it would have passed a quote the model
    got exactly right. When none is found, the quote is returned completely unchanged, so
    the guard fires on it exactly as before this step existed. The authorised motivating
    case ("we" typed for "and", not an article) is not handled here:
    `verify_claim_with_policy`'s bounded second pass (Part B) is what resolves that case, and
    every other word-level difference (spelling included) besides.

    Returns ``(evidence_quote, evidence_quotes, assertions, diagnostics)``: the first three
    are *evidence_quote*/*evidence_quotes*/*assertions* with every accepted relocation
    applied (``assertions`` is a new list of new dicts; nothing is mutated in place).
    ``diagnostics`` is a list of ``{"type": "quote_relocated", "original", "replacement",
    "edit_distance", "chunk_index"}`` records, one per relocation actually made, in the
    order they were made -- never a plain slug, unlike `apply_verification_guards`'
    ``diagnostics``, because a relocation is an edit worth being able to audit, not a flag.
    A caller that wants a slug alongside the guards' own `diagnostics` list appends
    ``"quote_relocated"`` itself, after the guards have run, so a relocation is never lost
    to that list being overwritten by the guards' own return value.

    Idempotent: every field this returns, run through this function again against the same
    *chunk_texts*, comes back unchanged -- every segment it relocated is, by construction,
    now verbatim in some chunk, so `_segment_already_verbatim` skips it on the next pass.

    WP-B25 relocation review (`.superpowers/sdd/wpB25-relocation-review.md`, minor 7): both
    callers set ``evidence_quote`` to ``evidence_quotes[0]`` before calling this, so when
    ``evidence_quotes`` is given, its first element is relocated once and reused for the
    flat ``evidence_quote`` field, rather than relocating the same text twice under two
    names and double-counting the one relocation in ``diagnostics``.
    """
    haystacks = [_normalise_for_match(chunk) for chunk in chunk_texts]
    diagnostics: list[dict[str, Any]] = []

    new_evidence_quotes = evidence_quotes
    if evidence_quotes:
        new_evidence_quotes = [
            _relocate_one_quote(quote, haystacks, chunk_texts, diagnostics)
            for quote in evidence_quotes
        ]
        new_evidence_quote = new_evidence_quotes[0]
    else:
        new_evidence_quote = _relocate_one_quote(
            evidence_quote, haystacks, chunk_texts, diagnostics
        )

    new_assertions: list[dict[str, Any]] = []
    for assertion in assertions or []:
        updated_assertion = dict(assertion)
        if updated_assertion.get("quote"):
            updated_assertion["quote"] = _relocate_one_quote(
                updated_assertion["quote"], haystacks, chunk_texts, diagnostics
            )
        if updated_assertion.get("quotes"):
            updated_assertion["quotes"] = [
                _relocate_one_quote(quote, haystacks, chunk_texts, diagnostics)
                for quote in updated_assertion["quotes"]
            ]
        new_assertions.append(updated_assertion)

    return new_evidence_quote, new_evidence_quotes, new_assertions, diagnostics


def _source_digest(*items: Any) -> str:
    """First 16 hex characters of the sha256 over the concatenated text of *items*, in
    order -- the same value, computed the same way,
    `test_verification_guard_functions_are_unchanged` and
    `test_verification_guards_are_still_frozen` independently pin as a literal for the guard
    set ("570a5b663e6140da" for the original six guards; "8a6c833ffc329f89" for the current
    seven); used here so `GUARD_DIGEST` and `QUOTE_RELOCATION_VERSION`
    are read from the code (and the constants) that define them rather than retyped, which
    could drift.

    Each item is either a callable, hashed as its own source (`inspect.getsource`), or a
    plain string, hashed as itself: a
    constant's own *source line* (e.g. ``_NUMBER_WORDS = frozenset({...})``) is not on any
    function's source at all, so a caller that wants a value change (not just a function
    body change) to move the digest passes that value's own `repr` as a string, not the
    constant itself."""
    digest = hashlib.sha256()
    for item in items:
        text = item if isinstance(item, str) else inspect.getsource(item)
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def _compute_guard_digest() -> str | None:
    """`GUARD_DIGEST`'s value, computed defensively: `_source_digest` calls
    `inspect.getsource`, which raises `OSError` when the
    ``.py`` source is not on disk (for example a deployment that ships only compiled
    bytecode) -- computing this at import time, unguarded, would fail the whole module's
    import over one provenance field. Falls back to ``None`` (logged) instead, degrading
    the claim-report/evaluation-meta provenance rather than the API.

    Guard 7 adds its own function list and, by value
    (`repr`, not the constant itself -- the same reasoning `_source_digest`'s own docstring
    states), every
    lexicon, band table, closed word list, span and overlap minimum a demotion decision
    passes through: a word silently added to a lexicon named only inside a function body is
    not on any hashed function's own source, so leaving it out would let that change leave
    `GUARD_DIGEST` unchanged."""
    try:
        return _source_digest(
            apply_verification_guards,
            _guard_no_full_text_with_chunks,
            _guard_attribution,
            _guard_assertion_status_consistency,
            _guard_numeric_tier_b,
            _guard_numeric_value_absent,
            _guard_quote_fidelity,
            _guard_scale_fidelity,
            scale_alignment_findings,
            _scale_negation_scope_shift,
            _scale_tokens,
            _scale_stem,
            _scale_is_content_token,
            _scale_content_indices,
            _scale_head_set,
            _scale_context_set,
            _scale_occurrences,
            _scale_admissible_pairs,
            _scale_align,
            repr(_SCALE_CURLY_FOLD),
            repr(_SCALE_EXTENT_WORDS),
            repr(_SCALE_DEGREE_WORDS),
            repr(_SCALE_HEDGE_WORDS),
            repr(_SCALE_BAND_BY_RANK),
            repr(sorted(_SCALE_FUNCTION_WORDS)),
            repr(_SCALE_EPISTEMIC_VERB_STEMS),
            repr(sorted(_SCALE_ABSOLUTE_NEGATORS)),
            repr(sorted(_SCALE_NEGATORS)),
            repr(_SCALE_HEAD_RIGHT_SPAN),
            repr(_SCALE_HEAD_RIGHT_MAX),
            repr(_SCALE_HEAD_LEFT_SPAN),
            repr(_SCALE_HEAD_LEFT_MAX),
            repr(_SCALE_CONTEXT_SPAN),
            repr(_SCALE_CONTEXT_MAX),
            repr(_SCALE_MIN_HEAD_OVERLAP),
            repr(_SCALE_MIN_CONTEXT_OVERLAP),
        )
    except (OSError, TypeError):
        logger.warning("GUARD_DIGEST could not be computed; source unavailable", exc_info=True)
        return None


#: The frozen guard set's own digest (brief authorisation for this task; moved once,
#: deliberately, for guard 7, task authorisation 2026-09-14): identical to the literal
#: "8a6c833ffc329f89" `test_verification_guard_functions_are_unchanged` and
#: `test_verification_guards_are_still_frozen` each compute independently and pin inline --
#: this is not a third, competing source of truth, it is the same computation, run once at
#: import time so a caller that wants to record it (the claim report provenance, an
#: evaluation run's meta file) can read it off this module instead of retyping the literal.
GUARD_DIGEST = _compute_guard_digest()


def _compute_quote_relocation_version() -> str | None:
    """`QUOTE_RELOCATION_VERSION`'s value, computed defensively (same reasoning as
    `_compute_guard_digest`).

    Hashing only `relocate_evidence_quotes`'s own
    source (the entry-point wrapper) would leave every decision that actually determines whether a
    quote relocates -- the window search, the closed-class allowlist, the number/negation
    protections, the edit-distance caps, the attribution pre-check -- outside the digest, so
    none of them moving would move the recorded version. This hashes the whole relocation
    set the way `GUARD_DIGEST` hashes the guard set: every helper the acceptance decision
    passes through, plus every threshold and word list *by value* (`repr`, not the constant
    itself, exactly because a constant's own literal is not on any function's source --
    `_source_digest`'s own docstring)."""
    try:
        return _source_digest(
            relocate_evidence_quotes,
            _relocate_one_quote,
            _segment_already_verbatim,
            _best_relocation_window,
            _windows_within_distance,
            _classify_relocation_alignment,
            _is_relocation_article,
            _token_is_citation_year,
            _looks_like_citation_surname,
            _run_contains_et_al_pair,
            _run_is_numeric_only,
            _run_has_citation_shape,
            _run_has_disqualifying_content,
            _run_sits_in_brackets,
            _is_dropped_parenthetical_citation,
            _delimiter_counts,
            _token_alignment,
            _token_edit_distance_within,
            _tokens_with_offsets,
            _is_negation_token,
            _tokens_differ_only_by_negation_prefix,
            _is_protected_number_word,
            _token_contains_digit,
            attribution_guard_fires_before_relocation,
            repr(QUOTE_RELOCATION_MIN_SEGMENT_TOKENS),
            repr(QUOTE_RELOCATION_MAX_CITATION_TOKENS),
            repr(QUOTE_RELOCATION_MAX_WINDOW_SHRINK),
            repr(QUOTE_RELOCATION_SEARCH_SLACK),
            repr(sorted(_RELOCATION_ARTICLES)),
            repr(sorted(_CITATION_BRACKET_PAIRS.items())),
            # `_CITATION_YEAR_RE`'s own pattern is
            # named inside `_token_is_citation_year`'s body, so its *value* is not on that
            # function's own source line and is not reachable through any hashed function's
            # source at all: rebinding it to a wider or narrower pattern would leave both
            # digests unchanged without this. `_CITATION_SURNAME_RE` is hashed here the same
            # way.
            repr(_CITATION_YEAR_RE.pattern),
            repr(_CITATION_SURNAME_RE.pattern),
            repr(_QUOTE_DELIMITER_CHARS),
            repr(sorted(_NUMBER_WORDS)),
            repr(sorted(_NEGATION_WORDS)),
            repr(_NEGATION_WORD_PREFIXES),
            repr(_NEGATION_AFFIXES),
            # `_TOKEN_APOSTROPHES` decides where
            # `_tokens_with_offsets` breaks a token (so it decides both tokenisation and,
            # through `_is_negation_token`, which "n't" contractions are protected) but is
            # not reachable through any hashed function's own source line.
            repr(_TOKEN_APOSTROPHES),
        )
    except (OSError, TypeError):
        logger.warning(
            "QUOTE_RELOCATION_VERSION could not be computed; source unavailable", exc_info=True
        )
        return None


#: `relocate_evidence_quotes`'s own source digest (brief authorisation for this task):
#: recorded in the claim report provenance next to ``prompt_version`` and `GUARD_DIGEST`,
#: and in the evaluation run meta files, so a claim report or an evaluation run states
#: which version of the relocation rule (if any) it ran under.
QUOTE_RELOCATION_VERSION = _compute_quote_relocation_version()


# --------------------------------------------------------------------------------------
# Verification policy: one shared function, `verify_claim_with_policy`, that both the production
# path (`_verify_claims`'s `_verify_one`, below) and the evaluation harness
# (`evaluation/claims/verify_common.py`'s `ProductionVerifier`) call in place of their own
# separate relocate-then-guard calls, so there is exactly one verification policy, not two
# copies that could drift.
#
# Part A above (`relocate_evidence_quotes`) is restricted to provably neutral quote edits.
# Part B here is a second, independent line of defence for the one demotion shape Part A no
# longer reaches: a verification model that answered "verified" and then mis-copied its own
# quote by a real word (the "we" for "and" motivating case) is not a text-relocation problem
# at all -- it is asked to repair its own quote, in the same conversation (the frozen prompt
# and its own first reply as message history, plus one further user turn,
# `QUOTE_REPAIR_PROMPT`, naming the segments that failed), and the repaired answer (whatever
# it is) is trusted as final. This never fires more than once per claim, and never fires at
# all unless the first pass's own shape licenses it (`_SECOND_PASS_TRIGGER_REASONS` below).
# --------------------------------------------------------------------------------------


@dataclass
class ModelPassAnswer:
    """One verification model call's own answer, already normalised (`normalise_
    verification_text`) and already reduced to plain, JSON-serialisable data by the caller's
    own ``call_model`` closure -- `verify_claim_with_policy` never touches a pydantic-ai
    ``AgentRunResult`` or any provider-specific object directly, so it works identically for
    the production agent and for the evaluation harness's own agent call."""

    status: str
    evidence_quote: str | None
    evidence_quotes: list[str]
    assertions: list[dict[str, Any]]
    explanation: str | None = None
    suggested_revision: str | None = None
    unstated_details: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    system_fingerprint: str | None = None
    model_reported: str | None = None


@dataclass
class RepairRequest:
    """The second, optional model call's own caller-facing input: every
    evidence-quote segment the first pass's own guard pass found not verbatim in the
    source, after relocation (`_non_verbatim_quote_segments` below) -- named so the repair
    turn's own prompt (``QUOTE_REPAIR_PROMPT``,
    ``app.agents.claim_verification_agent``) can list them for the model. ``call_model``
    receives ``None`` for the first call and one of these for the second; a caller's own
    closure uses ``failed_segments`` to build the repair turn's user message and is
    responsible for continuing the same conversation (the frozen prompt and the first
    reply as message history) -- `verify_claim_with_policy` itself never touches a raw
    model message.

    ``failed_segments`` must be non-empty: guard 3
    (`_guard_quote_fidelity`) fires ``quote_not_verbatim`` on two different shapes -- a
    quote that is not verbatim, and no quote at all -- and only the first has a segment
    to repair. Constructing one of these with nothing to repair would send a degenerate
    prompt asking the model to correct "one of these segments" where there are none,
    which is not a quote repair at all; the caller below never constructs one for the
    second shape, and this assertion is the invariant that keeps it that way."""

    failed_segments: list[str]

    def __post_init__(self) -> None:
        assert self.failed_segments, (
            "RepairRequest.failed_segments must be non-empty: a no-quote first answer "
            "must keep its guard demotion instead of firing a repair turn with nothing "
            "to repair"
        )


@dataclass
class VerificationPolicyResult:
    """`verify_claim_with_policy`'s return value: the final, post-guard verdict, the quote
    fields as relocation left them, and a full record of every pass actually made."""

    status: str
    machine_reasons: list[str]
    diagnostics: list[str]
    evidence_quote: str | None
    evidence_quotes: list[str]
    assertions: list[dict[str, Any]]
    explanation: str | None
    suggested_revision: str | None
    unstated_details: list[str]
    model_status: str | None
    model_reported: str | None
    quote_relocations: list[dict[str, Any]]
    #: One entry per model call actually made (one, or two when the bounded repair turn
    #: below fired), each ``{"model_status", "quotes", "machine_reasons", "diagnostics",
    #: "tokens", "fingerprint", "repair_prompt_version"}`` (task authorisation
    #: 2026-09-10, Part B; ``repair_prompt_version`` added 2026-09-11, ``None`` on the
    #: first pass, ``QUOTE_REPAIR_PROMPT_VERSION`` on the second).
    passes: list[dict[str, Any]]


#: Part B's own trigger, stated once (task authorisation 2026-09-10): the repair turn fires
#: if, and only if, the first pass's post-guard status is ``needs_nuance`` with exactly this
#: one-element reasons list, after a model status of ``verified`` -- never on ``unsupported``,
#: never on ``attribution_mismatch``, never when the model's own first answer was already
#: ``needs_nuance``, and never more than this once.
_SECOND_PASS_TRIGGER_REASONS: tuple[str, ...] = ("quote_not_verbatim",)


def _non_verbatim_quote_segments(
    quotes: Sequence[str], chunk_texts: Sequence[str]
) -> list[str]:
    """Every `_quote_segments` segment, across *quotes*, that Guard 3
    (`_guard_quote_fidelity`) would find not verbatim in *chunk_texts* -- the repair
    turn's own input (task authorisation 2026-09-11). Unlike `_guard_quote_fidelity`,
    which returns at the first failing segment (it only ever needs to know *that* one
    fired, to cap the status), this collects every one, in call order, so the repair
    prompt can name all of them to the model at once, not just the first.

    Reuses `_quote_segments`/`_normalise_for_match` exactly as the guard does, rather than
    re-implementing the verbatim check a second way; both helpers are outside the frozen
    guard digest's own function list (`GUARD_DIGEST`), so calling them here does not move
    it.
    """
    haystacks = [_normalise_for_match(c) for c in chunk_texts]
    failed: list[str] = []
    for quote in quotes:
        for segment in _quote_segments(quote):
            needle = _normalise_for_match(segment)
            if not needle or not any(needle in h for h in haystacks):
                failed.append(segment)
    return failed


def _drop_non_verbatim_assertion_quotes(
    assertions: Sequence[Mapping[str, Any]], chunk_texts: Sequence[str]
) -> list[dict[str, Any]]:
    """Drop, from each assertion's own ``quote``/``quotes`` fields, any quote
    `_non_verbatim_quote_segments` would find not verbatim in *chunk_texts*.

    Guard 3 (`_guard_quote_fidelity`) only ever inspects the top-level
    ``evidence_quote``/``evidence_quotes``; it never reads ``assertions[].quote`` or
    ``assertions[].quotes``. A paraphrase the repair turn (or the model's own first pass)
    dropped from the guarded top-level list can still survive, unchecked, on an assertion
    the report stores and the claim-verification UI renders as "every verbatim span
    composed to reach this assertion's verdict" -- so a quote the cited paper does not
    contain could reach a ``verified`` row's own detail view even though the guard that
    exists to catch exactly that never saw it.

    Called by `verify_claim_with_policy`'s own caller, on the `assertions` field of the
    `VerificationPolicyResult` it already returned, after `apply_verification_guards` has
    long since decided that result's ``status``, ``machine_reasons`` and ``diagnostics``
    from the unfiltered assertions: it changes what is shown, never what was verified, and
    it touches none of the six frozen guard functions `GUARD_DIGEST` hashes. Deliberately
    not called from inside `verify_claim_with_policy` itself: that function and its own
    source line are what
    `VERIFICATION_POLICY_VERSION` hashes, and a display-only filter has no business moving
    the digest that identifies the call-relocate-guard-repair policy. See
    `_compute_verification_policy_version`'s docstring.
    """
    cleaned: list[dict[str, Any]] = []
    for assertion in assertions:
        assertion = dict(assertion)
        quote = assertion.get("quote")
        if quote and _non_verbatim_quote_segments([quote], chunk_texts):
            assertion["quote"] = None
        quotes = assertion.get("quotes") or []
        assertion["quotes"] = [
            q for q in quotes if not _non_verbatim_quote_segments([q], chunk_texts)
        ]
        cleaned.append(assertion)
    return cleaned


async def verify_claim_with_policy(
    claim_text: str,
    chunk_texts: Sequence[str],
    chunks: Sequence[Mapping[str, Any]],
    call_model: Callable[[RepairRequest | None], Awaitable[ModelPassAnswer]],
) -> VerificationPolicyResult:
    """Run the shared verification policy for one claim: call the model, relocate the quote
    (Part A), guard, and -- exactly once, exactly when the first pass's own shape licenses it
    (`_SECOND_PASS_TRIGGER_REASONS` above) -- ask the model to repair its own quote in the
    same conversation and take that second pass's result as final, whatever it is (task
    authorisation 2026-09-11, replacing the identical-prompt second pass authorised
    2026-09-10).

    ``call_model`` does everything caller-specific: building the prompt (the first user
    prompt for the first call; the repair turn's own follow-up message, continuing the same
    conversation via pydantic-ai ``message_history``, for the second), invoking the agent,
    normalising the raw output into a `ModelPassAnswer`, and recording that one call's own
    provenance (appending to whatever list the caller's own closure captures) -- calling it
    twice therefore counts both calls in the caller's own provenance bookkeeping without this
    function needing to know what a provenance record looks like, or what a raw model message
    looks like. It receives ``None`` for the first call and a `RepairRequest` (naming every
    segment `_non_verbatim_quote_segments` found not verbatim, after relocation) for the
    second. An exception raised by the *first* call to ``call_model`` is never caught here:
    it propagates to the caller, exactly as a single, unwrapped model call would have
    (unchanged since this function was introduced). An exception raised by the *second*,
    optional call is caught here (WP-B25 relocation review round 4 minor 3, unchanged by the
    repair-turn redesign): the first pass's own completed, guarded result is kept and
    returned as final, and the failed second attempt is recorded as one further ``passes``
    entry rather than lost.

    Every pass -- one, or two -- runs through `attribution_guard_fires_before_relocation`,
    `relocate_evidence_quotes` and `apply_verification_guards` identically; only the *second*
    call to the model is conditional, never the guard machinery around it. The repair turn's
    own answer is trusted as final exactly as the first pass would have been: the guards may
    still cap its status (they run on it unconditionally, so its status can only ever move
    towards ``unsupported``, never be promoted), but the model's own verdict, explanation and
    suggested_revision are taken as it returned them, a downgrade included.
    """
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION

    passes: list[dict[str, Any]] = []
    answers: list[ModelPassAnswer] = []
    last: dict[str, Any] = {}

    async def _run_one_pass(repair_request: RepairRequest | None) -> None:
        answer = await call_model(repair_request)
        answers.append(answer)
        evidence_quote = answer.evidence_quote
        evidence_quotes = list(answer.evidence_quotes)
        assertions = [dict(a) for a in answer.assertions]
        relocation_diagnostics: list[dict[str, Any]] = []
        if not attribution_guard_fires_before_relocation(claim_text, chunk_texts, evidence_quote):
            evidence_quote, evidence_quotes, assertions, relocation_diagnostics = (
                relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, chunk_texts)
            )
        status, machine_reasons, diagnostics = apply_verification_guards(
            answer.status,
            claim_text=claim_text,
            evidence_quote=evidence_quote,
            evidence_quotes=evidence_quotes,
            assertions=assertions,
            chunk_texts=chunk_texts,
            chunks=chunks,
        )
        if relocation_diagnostics:
            diagnostics = [*diagnostics, "quote_relocated"]
        passes.append(
            {
                "model_status": answer.status,
                "quotes": evidence_quotes or ([evidence_quote] if evidence_quote else []),
                "machine_reasons": machine_reasons,
                "diagnostics": diagnostics,
                "tokens": {
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                },
                "fingerprint": answer.system_fingerprint,
                "repair_prompt_version": (
                    QUOTE_REPAIR_PROMPT_VERSION if repair_request is not None else None
                ),
            }
        )
        last.update(
            status=status,
            machine_reasons=machine_reasons,
            diagnostics=diagnostics,
            evidence_quote=evidence_quote,
            evidence_quotes=evidence_quotes,
            assertions=assertions,
            quote_relocations=relocation_diagnostics,
        )

    await _run_one_pass(None)
    first_answer = answers[0]
    if (
        last["status"] == "needs_nuance"
        and list(last["machine_reasons"]) == list(_SECOND_PASS_TRIGGER_REASONS)
        and first_answer.status == "verified"
    ):
        repaired_quotes = last["evidence_quotes"] or (
            [last["evidence_quote"]] if last["evidence_quote"] else []
        )
        # WP-B25b review round M2: guard 3 fires this same reason on two shapes -- a
        # non-verbatim quote, and no quote at all (`not any(quotes)`). Only the first
        # has a segment to repair; firing the turn on the second would send a
        # degenerate prompt naming no segments, soliciting evidence the model never
        # produced rather than repairing a mis-copy. Skip the turn entirely when there
        # is nothing to name, so the guard's own demotion is kept as final.
        failed_segments = _non_verbatim_quote_segments(repaired_quotes, chunk_texts)
        if failed_segments:
            repair_request = RepairRequest(failed_segments=failed_segments)
            try:
                await _run_one_pass(repair_request)
            except Exception as exc:
                # WP-B25 relocation review round 4 minor 3: a transient failure on this
                # second, optional call must not destroy the first pass's own completed,
                # guarded verdict. `_run_one_pass` only ever mutates
                # `answers`/`passes`/`last` after its own `call_model()` call returns, so
                # an exception here leaves all three exactly as the first pass left them;
                # `last` below is still the first pass's own result, as if the repair
                # call had never been attempted. The failed attempt is still recorded, as
                # a `passes` entry that carries no verdict of its own, so a reader can see
                # that a repair call was tried and did not complete (`pass_count` becomes
                # 2 for this claim; `first_pass_machine_reasons`, read from
                # ``passes[0]``, is untouched). The *first* call's own exception
                # behaviour, just above, is unchanged: it still propagates uncaught.
                logger.warning(
                    "quote-repair turn failed; keeping the first pass's guarded result",
                    exc_info=True,
                )
                passes.append(
                    {
                        "model_status": None,
                        "quotes": [],
                        "machine_reasons": [],
                        "diagnostics": ["second_pass_error"],
                        "tokens": {"input_tokens": None, "output_tokens": None},
                        "fingerprint": None,
                        "repair_prompt_version": QUOTE_REPAIR_PROMPT_VERSION,
                        "error": str(exc),
                    }
                )

    final_answer = answers[-1]
    return VerificationPolicyResult(
        status=last["status"],
        machine_reasons=last["machine_reasons"],
        diagnostics=last["diagnostics"],
        evidence_quote=last["evidence_quote"],
        evidence_quotes=last["evidence_quotes"],
        assertions=last["assertions"],
        explanation=final_answer.explanation,
        suggested_revision=final_answer.suggested_revision,
        unstated_details=final_answer.unstated_details,
        model_status=final_answer.status,
        model_reported=final_answer.model_reported,
        quote_relocations=last["quote_relocations"],
        passes=passes,
    )


def _compute_verification_policy_version() -> str | None:
    """Computes `VERIFICATION_POLICY_VERSION` defensively (same reasoning as
    `_compute_guard_digest`): hashes the shared policy function itself, plus the whole
    relocation set `QUOTE_RELOCATION_VERSION` already hashes, the second-pass trigger, the
    repair turn's own segment-collection helper, and `QUOTE_REPAIR_PROMPT_VERSION`
    (`app.agents.claim_verification_agent`, imported here the same lazy way `_verify_claims`
    already imports that module, to avoid a module-level import cycle) -- so a change to
    the repair prompt's own text, not only to the code around it, moves this version too.
    The frozen guards are hashed separately (`GUARD_DIGEST`) and are not repeated here:
    this digest identifies the *policy* (call once, relocate, guard, repair-and-call-again
    under one exact condition), not the guard bodies it calls unchanged.

    `_drop_non_verbatim_assertion_quotes` is deliberately excluded: it runs after every
    guard has already decided `status`, `machine_reasons` and `diagnostics` from the
    unfiltered assertions, and only edits what is stored and displayed in the assertion
    detail view. It can change what a reader sees under an assertion but never which
    verdict was reached, so it does not identify the policy this digest names."""
    try:
        from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION

        return _source_digest(
            verify_claim_with_policy,
            _non_verbatim_quote_segments,
            relocate_evidence_quotes,
            _relocate_one_quote,
            _segment_already_verbatim,
            _best_relocation_window,
            _windows_within_distance,
            _classify_relocation_alignment,
            _is_relocation_article,
            _token_is_citation_year,
            _looks_like_citation_surname,
            _run_contains_et_al_pair,
            _run_is_numeric_only,
            _run_has_citation_shape,
            _run_has_disqualifying_content,
            _run_sits_in_brackets,
            _is_dropped_parenthetical_citation,
            _delimiter_counts,
            _token_alignment,
            _token_edit_distance_within,
            _tokens_with_offsets,
            _is_negation_token,
            _tokens_differ_only_by_negation_prefix,
            _is_protected_number_word,
            _token_contains_digit,
            attribution_guard_fires_before_relocation,
            repr(QUOTE_RELOCATION_MIN_SEGMENT_TOKENS),
            repr(QUOTE_RELOCATION_MAX_CITATION_TOKENS),
            repr(QUOTE_RELOCATION_MAX_WINDOW_SHRINK),
            repr(QUOTE_RELOCATION_SEARCH_SLACK),
            repr(sorted(_RELOCATION_ARTICLES)),
            repr(sorted(_CITATION_BRACKET_PAIRS.items())),
            repr(_CITATION_YEAR_RE.pattern),
            repr(_CITATION_SURNAME_RE.pattern),
            repr(_QUOTE_DELIMITER_CHARS),
            repr(sorted(_NUMBER_WORDS)),
            repr(sorted(_NEGATION_WORDS)),
            repr(_NEGATION_WORD_PREFIXES),
            repr(_NEGATION_AFFIXES),
            repr(_TOKEN_APOSTROPHES),
            repr(_SECOND_PASS_TRIGGER_REASONS),
            repr(QUOTE_REPAIR_PROMPT_VERSION),
        )
    except (OSError, TypeError, ImportError):
        logger.warning(
            "VERIFICATION_POLICY_VERSION could not be computed; source unavailable",
            exc_info=True,
        )
        return None


#: The shared verification policy's own digest: identifies the whole
#: call-relocate-guard[-call-again] policy `verify_claim_with_policy` implements, distinct
#: from `QUOTE_RELOCATION_VERSION` (the relocation rule alone) and `GUARD_DIGEST` (the
#: frozen guards, unaffected by this policy).
VERIFICATION_POLICY_VERSION = _compute_verification_policy_version()


def _build_analysis_deps(project: Project | None, project_id: UUID):
    """Build AnalysisDependencies from a real Project row.

    Mirrors the correct pattern already used in analysis.py:131-136 and
    research_design.py:112-119.
    """
    from app.agents.analysis_agent import AnalysisDependencies

    return AnalysisDependencies(
        project_id=str(project_id),
        project_description=project.description if project else None,
        target_journal=project.target_journal if project else None,
        citation_style=(
            project.citation_style.value if project and project.citation_style else "APA"
        ),
    )


async def _load_analysis_deps(project_id: UUID, session_factory):
    """Load the project row and build AnalysisDependencies from it."""
    async with session_factory() as session:
        project = await session.get(Project, project_id)
        return _build_analysis_deps(project, project_id)


def _build_paper_lookup(papers: list[Paper]) -> dict[str, Paper]:
    """Index papers by 'lastname_year' so [Author, Year] citations can be matched.

    When a paper's acquired full text carries an accepted byline
    (``metadata["fulltext_byline"]["source"] == "fulltext"``, `_accept_byline`), its own
    first surname is registered as an ADDITIONAL alias key for the same paper -- the
    canonical ``surname_year`` key stays exactly what it always was, so a citation the
    writer renders from the byline ("Ghane et al. (2024)") still resolves to the same
    paper OpenAlex's own author order names, and ``citation_coverage.unresolved`` does
    not rise just because the rendered author string changed."""
    lookup: dict[str, Paper] = {}
    for p in papers:
        if not (p.authors and p.year):
            continue
        first_author = (
            p.authors[0] if isinstance(p.authors[0], str) else p.authors[0].get("name", "")
        )
        last_name = first_author.split(",")[0].split()[-1] if first_author else ""
        if last_name:
            lookup[f"{last_name.lower()}_{p.year}"] = p
        byline = (getattr(p, "metadata_", None) or {}).get("fulltext_byline") or {}
        if byline.get("source") == "fulltext" and byline.get("surnames"):
            alias_surname = byline["surnames"][0]
            if alias_surname:
                lookup.setdefault(f"{alias_surname.lower()}_{p.year}", p)
    return lookup


#: Public delegation: `app.services.citation_render` needs the identical lookup the
#: verifier itself builds, so a rendered citation number and a verification verdict
#: always name the same paper.
build_paper_lookup = _build_paper_lookup


def _author_names(paper: Paper) -> list[str]:
    """Flatten a paper's authors (str or {'name': ...}) into display names."""
    names: list[str] = []
    for a in (paper.authors or []):
        if isinstance(a, str):
            names.append(a)
        elif isinstance(a, dict):
            names.append(a.get("name", ""))
    return names


def _aggregate_provenance(
    records: list[LLMCallProvenance],
    *,
    agent: str,
    model_configured: str,
    temperature: float | None,
    prompt_version: str,
    guard_digest: str | None = None,
    quote_relocation_version: str | None = None,
    verification_policy_version: str | None = None,
    repair_prompt_version: str | None = None,
) -> dict:
    """Collapse per-call provenance into the report-level record.

    ``guard_digest``/``quote_relocation_version`` (keyword-only, both defaulting to
    ``None`` so this stays additive) sit next to ``prompt_version``: which frozen guard
    set, and which version (if any) of the quote-relocation step
    (``GUARD_DIGEST``/``QUOTE_RELOCATION_VERSION`` above), produced this report's claims.
    ``verification_policy_version`` (same additive keyword-only shape) is which version of
    the whole shared policy (relocation and the bounded repair turn together,
    ``VERIFICATION_POLICY_VERSION``) produced them. ``repair_prompt_version`` (same
    additive keyword-only shape) is which version of the repair turn's own prompt template
    (``QUOTE_REPAIR_PROMPT_VERSION``) this report's claims were repaired under, if any of
    them were.
    """

    def _distinct(values) -> list:
        seen: list = []
        for value in values:
            if value is not None and value not in seen:
                seen.append(value)
        return seen

    def _total(values) -> int | None:
        present = [v for v in values if v is not None]
        return sum(present) if present else None

    return {
        "agent": agent,
        "model_configured": model_configured,
        "model_reported": _distinct(r.model_reported for r in records),
        "system_fingerprints": _distinct(r.system_fingerprint for r in records),
        "temperature": temperature,
        "prompt_version": prompt_version,
        "guard_digest": guard_digest,
        "quote_relocation_version": quote_relocation_version,
        "verification_policy_version": verification_policy_version,
        "repair_prompt_version": repair_prompt_version,
        "calls": len(records),
        "input_tokens": _total(r.input_tokens for r in records),
        "output_tokens": _total(r.output_tokens for r in records),
    }


def _evidence_first_chunk_order(
    filtered_chunks: list[dict],
    chunk_texts: list[str],
    evidence_quotes: Sequence[str] | None,
) -> tuple[list[dict], list[str]]:
    """Reorder *filtered_chunks*/*chunk_texts* (the same permutation applied to both, in
    lockstep) so a chunk holding a segment of one of *evidence_quotes* comes first, in
    the order its quote is located, followed by every other chunk in its original
    stored order -- the set of chunks and their own contents are never changed, only
    their position in the list shown to the verifier.

    Consistency matters here: `format_verification_prompt` numbers "--- Chunk N ---" by
    a chunk's position in whichever list it is given, so the caller must pass this same
    reordered pair to `_find_evidence_location` and `apply_verification_guards` too, or
    the chunk number the model saw and the chunk number a guard reports would disagree.
    The claim-verification prompt and every guard function are untouched by this: only
    the order of what is handed to them changes (`test_evidence_chunk_ordering.py`
    pins both frozen digests unchanged).

    Returns the inputs completely unchanged (the same list objects, not copies) when
    *evidence_quotes* is empty or none of them can be located in *filtered_chunks* --
    exactly the pre-existing code path for every claim with no evidence ids, which is
    every claim before this feature existed and every claim the frozen internal
    evaluation harness sends, so that evaluated path stays byte-identical."""
    if not evidence_quotes:
        return filtered_chunks, chunk_texts
    priority = _locate_evidence_chunk_indices(
        None, filtered_chunks, evidence_quotes=evidence_quotes
    )
    if not priority:
        return filtered_chunks, chunk_texts
    priority_set = set(priority)
    order = priority + [i for i in range(len(filtered_chunks)) if i not in priority_set]
    return [filtered_chunks[i] for i in order], [chunk_texts[i] for i in order]


def claims_from_citation_links(
    citation_links: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """(claim_text, citation_key, citation_text, claim_sentence) per (sentence, key)
    pair a validated citation link names, for the write job's own freshly generated
    text: no Tiptap document exists yet at this point, unlike
    `extract_claims_from_document`, so this reads the linker's own validated dicts
    (``app.agents.citation_link_agent.link_citations``) directly instead of
    re-deriving citation units from scratch.

    One claim per (sentence, proposition, key) a validated citation link names: deduping
    on ``(sentence, key)`` alone would collapse a sentence's SECOND proposition against
    the same paper into the first, so the second one would never reach the verifier even
    though the citation-link step had reported it. ``proposition`` (already guaranteed
    non-empty by `validate_citation_link` for a real, validated link) is ``claim_text``,
    falling back to ``sentence`` when a dict happens to omit it. Two links naming the
    same (sentence, proposition, key) triple collapse to one claim, the first
    ``citation_text`` encountered kept -- the identical dedup rule
    `extract_claims_from_document` applies to its own units."""
    claims: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for link in citation_links:
        sentence = link.get("sentence") or ""
        claim_text = link.get("proposition") or sentence
        citation_text = link.get("citation_text") or ""
        for key in link.get("keys") or []:
            dedup_key = (sentence, claim_text, key)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            claims.append((claim_text, key, citation_text, sentence))
    return claims


def evidence_quotes_by_claim_map(
    citation_links: Sequence[Mapping[str, Any]],
    evidence_catalog: Sequence[Mapping[str, Any]] | None,
) -> dict[tuple[str, str], list[str]]:
    """(sentence, key) -> the evidence-catalog quotes a link's own ``evidence_ids`` name,
    for `_verify_claims`'s ``evidence_quotes_by_claim`` parameter. Built from the same
    catalog `app.services.writing._build_evidence_context` returned when the section was
    composed; ``evidence_catalog`` is ``None`` or ``[]`` whenever no evidence was
    available, in which case every claim maps to nothing and `_verify_claims` takes its
    pre-existing, unordered code path."""
    catalog_by_id = {item["id"]: item.get("quote") for item in (evidence_catalog or [])}
    result: dict[tuple[str, str], list[str]] = {}
    for link in citation_links:
        quotes = [
            catalog_by_id[eid]
            for eid in (link.get("evidence_ids") or [])
            if eid in catalog_by_id and catalog_by_id[eid]
        ]
        if not quotes:
            continue
        sentence = link.get("sentence") or ""
        for key in link.get("keys") or []:
            result.setdefault((sentence, key), []).extend(quotes)
    return result


async def _verify_claims(
    claims: list[tuple[str, str, str, str]],
    paper_lookup: dict[str, Paper],
    deps,
    job_id: UUID,
    session_factory,
    progress_label: str = "Verified",
    evidence_quotes_by_claim: dict[tuple[str, str], list[str]] | None = None,
) -> dict:
    """Verify (claim_text, citation_key) pairs against the matched papers' full text.

    ``claims`` is a ``(claim_text, citation_key, citation_text, claim_sentence)``
    quadruple list: ``claim_text`` is what is actually sent to the verification model and
    stored as ``ClaimVerification.claim_text``; ``claim_sentence`` is always the full
    sentence it was cut from and is stored separately, unchanged by anything the model
    returns.

    ``evidence_quotes_by_claim``, when given, maps ``(claim_sentence, citation_key)`` to
    the evidence quotes that claim's own citation link named; the chunks holding those
    quotes are moved to the front of the list sent to the verifier for that one claim
    only (`_evidence_first_chunk_order`), the rest kept in their stored order. ``None``
    verifies with the chunks in their stored order throughout.

    Returns a dict with keys: verifications, verified_count, unsupported_count,
    nuance_count, abstract_only_count, error_count, contradicted_count, guarded_count,
    cancelled, full_text_coverage (claims whose source had full-text chunks / all claims)
    and provenance (see ``_aggregate_provenance``).
    """
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        QUOTE_REPAIR_PROMPT_VERSION,
        VERIFICATION_PROMPT,
        format_quote_repair_prompt,
        format_verification_prompt,
        get_claim_verification_agent,
    )
    from app.agents.model_config import DETERMINISTIC_LONG_MODEL_SETTINGS

    agent = get_claim_verification_agent()
    temperature = DETERMINISTIC_LONG_MODEL_SETTINGS.get("temperature")
    total_claims = len(claims)
    semaphore = asyncio.Semaphore(max(1, settings.analysis_concurrency))
    progress_lock = asyncio.Lock()
    state = {"completed": 0, "cancelled": False, "with_chunks": 0}
    provenances: list[LLMCallProvenance] = []
    # `drop_reference_and_backmatter_chunks` runs once per paper here rather than inside
    # `_verify_one`, so a draft citing the same paper from N claims does not pay for the
    # same CPU-bound recomputation N times (measured at 2.67 ms per call for a 30-chunk
    # paper -- roughly half a second of blocking for a 200-claim draft citing few sources).
    # Memoized here, in the scope enclosing every `_verify_one` call, keyed on paper id.
    _filtered_chunks_cache: dict[UUID, tuple[list[dict], list[str]]] = {}

    async def _record_progress() -> None:
        """Count one item; every N items write progress and check for cancellation."""
        async with progress_lock:
            state["completed"] += 1
            every = max(1, settings.job_progress_update_every)
            done = state["completed"]
            if done % every and done != total_claims:
                return
            async with session_factory() as session:
                if await task_service.should_abort(session, job_id):
                    state["cancelled"] = True
                    return
                await task_service.update_job_status(
                    session,
                    job_id,
                    JobStatus.running,
                    progress=done / max(total_claims, 1),
                    progress_message=f"{progress_label} {done}/{total_claims} claims...",
                )

    async def _verify_one(
        claim_text: str, citation_key: str, citation_text: str, claim_sentence: str
    ) -> ClaimVerification:
        matched_paper = paper_lookup.get(citation_key.lower())

        if matched_paper is None:
            await _record_progress()
            return ClaimVerification(
                claim_text=claim_text,
                claim_sentence=claim_sentence,
                paper_id=UUID("00000000-0000-0000-0000-000000000000"),
                status="no_full_text",
                explanation="Could not match citation to a paper in the library.",
                citation=citation_text,
            )

        metadata = matched_paper.metadata_ or {}
        acquisition_route = metadata.get("fulltext_source")
        # `acquire_full_texts` is only one of four chunk-storage paths, and a paper
        # acquired before this guard existed keeps its unfiltered chunks forever. Re-run
        # the drop here, on every read, so a reference list can never reach the verifier
        # as prose regardless of how the chunks were stored. This is deliberately
        # `drop_reference_and_backmatter_chunks`, not a re-run of `check_full_text_identity`
        # (the read side re-filters reference chunks; it never re-validates identity).
        # The other three chunk-storage paths (`backend/app/api/fulltext.py`'s PDF upload
        # and paste, and `backend/app/api/papers.py`'s bulk-upload confirm via
        # `backend/app/services/paper_upload.py`) inherit the same guard from
        # `chunk_text`'s own default (`drop_reference_chunks=True`), so this read-side
        # re-filter is a second line of defence for freshly-stored chunks, and the only
        # guard at all for any row stored before that default existed.
        # Memoized per paper id in `_filtered_chunks_cache` (see `_verify_claims`), since
        # this recomputes the same result for every claim that cites the same paper.
        cached = _filtered_chunks_cache.get(matched_paper.id)
        if cached is None:
            stored_chunks = [c for c in metadata.get("fulltext_chunks", []) if "text" in c]
            filtered_chunks, _dropped_at_read = drop_reference_and_backmatter_chunks(
                stored_chunks
            )
            cached = (filtered_chunks, [c["text"] for c in filtered_chunks])
            _filtered_chunks_cache[matched_paper.id] = cached
        filtered_chunks, chunk_texts = cached
        if not chunk_texts:
            await _record_progress()
            # Nothing else in the report surfaces `fulltext_reason` to the user, so the
            # explanation below is extended with the reason whenever the acquisition guard
            # recorded one: `wrong_work` and `publisher_interstitial` each get their own
            # specific message instead of the generic "no full-text chunks" one.
            explanation = "No full-text chunks available for this paper."
            fulltext_reason = metadata.get("fulltext_reason")
            if fulltext_reason == "wrong_work":
                explanation = (
                    "No full-text chunks available: the fetched full text did not match "
                    "this paper's title or author."
                )
            elif fulltext_reason == "publisher_interstitial":
                explanation = (
                    "No full-text chunks available: the publisher blocked automatic "
                    "access to the full text (a login wall or interstitial page, not "
                    "the paper itself)."
                )
            return ClaimVerification(
                claim_text=claim_text,
                claim_sentence=claim_sentence,
                paper_id=matched_paper.id,
                status="no_full_text",
                explanation=explanation,
                citation=citation_text,
                paper_doi=matched_paper.doi,
                paper_title=matched_paper.title,
                acquisition_route=acquisition_route,
            )
        state["with_chunks"] += 1

        async with semaphore:
            if state["cancelled"]:
                return ClaimVerification(
                    claim_text=claim_text,
                    claim_sentence=claim_sentence,
                    paper_id=matched_paper.id,
                    status="error",
                    explanation="Verification cancelled before this claim was checked.",
                    citation=citation_text,
                    paper_doi=matched_paper.doi,
                    paper_title=matched_paper.title,
                    acquisition_route=acquisition_route,
                )
            call_chunks, call_chunk_texts = _evidence_first_chunk_order(
                filtered_chunks,
                chunk_texts,
                (evidence_quotes_by_claim or {}).get((claim_sentence, citation_key)),
            )
            prompt = format_verification_prompt(
                claim_text=claim_text,
                chunks=call_chunk_texts,
                paper_title=matched_paper.title,
                paper_authors=_author_names(matched_paper),
            )

            # The first call's own raw result, kept here (not returned to
            # `verify_claim_with_policy`, which never touches a raw model message) so a
            # repair turn, if one fires, can continue the SAME conversation via
            # pydantic-ai's own `message_history`.
            latest_call_result: dict[str, Any] = {}

            async def _call_model(repair_request: RepairRequest | None = None) -> ModelPassAnswer:
                """One verification model call, normalised into a `ModelPassAnswer`
                (`verify_claim_with_policy`'s shared call-relocate-guard[-repair-and-call-
                again] policy). ``prompt`` is built once, above, outside this closure, so the
                first call always sends it unchanged; a licensed repair turn instead
                continues that same conversation (``message_history=latest_call_result[
                "result"].all_messages()``, the *first* call's own result, since a repair
                turn only ever fires once) with one further user turn,
                `format_quote_repair_prompt`, naming the segments that failed."""
                if repair_request is None:
                    result = await agent.run(prompt, deps=deps)
                else:
                    repair_prompt = format_quote_repair_prompt(repair_request.failed_segments)
                    result = await agent.run(
                        repair_prompt,
                        deps=deps,
                        message_history=latest_call_result["result"].all_messages(),
                    )
                latest_call_result["result"] = result
                output = result.output.model_dump()
                # Normalise before anything else touches these fields, in particular
                # before relocation/guarding reads `evidence_quote` -- both must see the
                # same text a user later sees.
                explanation = normalise_verification_text(
                    output.get("explanation"), strip_wrapping_quotes=True
                )
                # `evidence_quotes` (every span the model used) is normalised
                # element-wise, and `evidence_quote` keeps its name, type and position by
                # being set to that list's first span. A v1/v2-shaped model answer with no
                # `evidence_quotes` falls back to its own (still normalised)
                # `evidence_quote`.
                evidence_quotes = [
                    normalise_verification_text(q) for q in output.get("evidence_quotes") or []
                ]
                evidence_quote = (
                    evidence_quotes[0] if evidence_quotes
                    else normalise_verification_text(output.get("evidence_quote"))
                )
                suggested_revision = normalise_verification_text(output.get("suggested_revision"))
                # Every composed span an assertion cites (brief 1.2's `quotes`) and its
                # single `quote`: normalised so the claim-record export (P4) reads back only
                # normalised quotes, matching `evidence_quote`/`evidence_quotes`.
                assertions: list[dict[str, Any]] = []
                for assertion in output.get("assertions") or []:
                    assertion = dict(assertion)
                    assertion["quote"] = normalise_verification_text(assertion.get("quote"))
                    assertion["quotes"] = [
                        normalise_verification_text(q) for q in assertion.get("quotes") or []
                    ]
                    assertions.append(assertion)
                # Provenance bookkeeping must never change a verdict the model produced.
                input_tokens = output_tokens = system_fingerprint = model_reported = None
                try:
                    record = provenance_from_run(
                        "claim_verification",
                        result,
                        model_configured=settings.deepseek_model,
                        temperature=temperature,
                        prompt=VERIFICATION_PROMPT,
                    )
                except Exception:
                    logger.warning(
                        "Could not record provenance for claim verification", exc_info=True
                    )
                else:
                    provenances.append(record)
                    input_tokens = record.input_tokens
                    output_tokens = record.output_tokens
                    system_fingerprint = record.system_fingerprint
                    model_reported = record.model_reported
                return ModelPassAnswer(
                    status=output.get("status"),
                    evidence_quote=evidence_quote,
                    evidence_quotes=evidence_quotes,
                    assertions=assertions,
                    explanation=explanation,
                    suggested_revision=suggested_revision,
                    unstated_details=output.get("unstated_details") or [],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    system_fingerprint=system_fingerprint,
                    model_reported=model_reported,
                )

            try:
                # The shared verification policy: call the model, relocate the quote
                # (Part A), guard, and -- exactly once, exactly when licensed -- ask the
                # model to repair its own quote in the same conversation and take that
                # repair turn's result as final (Part B). One policy, shared with the
                # evaluation harness (`evaluation/claims/verify_common.py`'s
                # `ProductionVerifier`).
                policy_result = await verify_claim_with_policy(
                    claim_text, call_chunk_texts, call_chunks, _call_model
                )
            except Exception as exc:
                logger.warning("Verification agent failed for claim: %s", exc)
                verification = ClaimVerification(
                    claim_text=claim_text,
                    claim_sentence=claim_sentence,
                    paper_id=matched_paper.id,
                    status="error",
                    explanation=f"Verification agent error: {exc}",
                )
            else:
                # This runs outside `verify_claim_with_policy` itself, so a display-only
                # filter cannot move `VERIFICATION_POLICY_VERSION`: the guards above have
                # already decided `status`/`machine_reasons`/`diagnostics` from the
                # unfiltered assertions, so dropping a non-verbatim assertion quote here
                # changes only what is stored and shown, never what was verified.
                stored_assertions = _drop_non_verbatim_assertion_quotes(
                    policy_result.assertions, call_chunk_texts
                )
                verification = ClaimVerification(
                    claim_text=claim_text,
                    paper_id=matched_paper.id,
                    status=policy_result.status,
                    evidence_quote=policy_result.evidence_quote,
                    evidence_quotes=policy_result.evidence_quotes,
                    explanation=policy_result.explanation or "",
                    suggested_revision=policy_result.suggested_revision,
                    assertions=[ClaimAssertion(**a) for a in stored_assertions],
                    unstated_details=policy_result.unstated_details,
                )
                verification.model_status = policy_result.model_status
                verification.model_reported = policy_result.model_reported
                verification.machine_reasons = policy_result.machine_reasons
                verification.diagnostics = policy_result.diagnostics
                verification.quote_relocations = policy_result.quote_relocations
                verification.passes = policy_result.passes
            # Claim-verification record export fields: set once here so
            # both the success and error branches above carry the same trace metadata.
            # `claim_text` is reset to the extracted draft sentence here too: the success
            # branch above rebuilds `verification` from `result.output.model_dump()`, so
            # `claim_text` otherwise stays whatever the model transcribed the sentence as,
            # which need not be verbatim. The prompt, the guards above and the verdict all
            # already use the extracted `claim_text`, so this only corrects what is stored
            # and exported.
            # `claim_sentence` is set here for the identical reason: the model is never
            # asked for it, so it must be filled from the extractor's own value on every
            # path, success or error alike.
            verification.claim_text = claim_text
            verification.claim_sentence = claim_sentence
            verification.citation = citation_text
            verification.paper_doi = matched_paper.doi
            verification.paper_title = matched_paper.title
            verification.acquisition_route = acquisition_route
            # Computed after relocation (and skipped only when the model itself returned
            # no status to guard, `no_full_text`/`error`, in which case the quote was
            # never relocated either), so the stored location always describes the stored
            # quote -- computing it on the raw, pre-relocation quote would leave it `None`
            # for exactly the rows relocation rescues, even though the relocated quote is
            # verbatim in a named chunk.
            verification.evidence_location = _find_evidence_location(
                verification.evidence_quote,
                call_chunks,
                evidence_quotes=verification.evidence_quotes,
            )
        await _record_progress()
        return verification

    verifications: list[ClaimVerification] = list(
        await asyncio.gather(
            *(
                _verify_one(text, key, citation, sentence)
                for text, key, citation, sentence in claims
            )
        )
    )

    return {
        "verifications": verifications,
        "verified_count": sum(1 for v in verifications if v.status == "verified"),
        "unsupported_count": sum(1 for v in verifications if v.status == "unsupported"),
        "nuance_count": sum(1 for v in verifications if v.status == "needs_nuance"),
        "abstract_only_count": sum(1 for v in verifications if v.status == "no_full_text"),
        "error_count": sum(1 for v in verifications if v.status == "error"),
        # Informational counts. "Contradicted" is judged from the model's own
        # per-assertion verdicts; "guarded" is judged from whether a code guard changed
        # the status the model returned.
        "contradicted_count": sum(
            1 for v in verifications if any(a.verdict == "contradicted" for a in v.assertions)
        ),
        "guarded_count": sum(
            1 for v in verifications if v.model_status is not None and v.status != v.model_status
        ),
        "cancelled": state["cancelled"],
        "full_text_coverage": round(state["with_chunks"] / max(total_claims, 1), 3),
        "provenance": _aggregate_provenance(
            provenances,
            agent="claim_verification",
            model_configured=settings.deepseek_model,
            temperature=temperature,
            prompt_version=CLAIM_VERIFICATION_PROMPT_VERSION,
            guard_digest=GUARD_DIGEST,
            quote_relocation_version=QUOTE_RELOCATION_VERSION,
            verification_policy_version=VERIFICATION_POLICY_VERSION,
            repair_prompt_version=QUOTE_REPAIR_PROMPT_VERSION,
        ),
    }


async def acquire_full_texts(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    paper_ids: list[UUID] | None = None,
) -> None:
    """Background task: Unpaywall-first cascade for all library papers.

    Papers that already have full text (upload, paste, or an earlier run) are skipped
    before any network call is made.

    For each remaining paper with a DOI, try in order:
    1. Unpaywall (DOI -> OA PDF URL)
    2. Stored ``full_text_url`` on the paper row
    3. ``oa_url`` / ``open_access_url`` in the paper metadata (OpenAlex)

    Downloads PDF, extracts text, chunks, stores in Paper.metadata_.
    Papers not found are marked 'abstract_only' for user upload.
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session,
            job_id,
            JobStatus.running,
            progress=0.0,
            progress_message="Starting full-text acquisition...",
        )

    try:
        # 1. Load all papers for this project
        async with session_factory() as session:
            query = (
                select(Paper)
                .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                .where(ProjectPaper.project_id == project_id)
            )
            if paper_ids:
                query = query.where(Paper.id.in_(paper_ids))
            result = await session.execute(query)
            papers = list(result.scalars().all())

        total = len(papers)
        acquired = 0
        abstract_only = 0
        already_acquired = 0
        wrong_work = 0
        interstitial = 0
        reference_chunks_dropped = 0
        unpaywall = UnpaywallClient()

        for idx, paper in enumerate(papers):
            # Idempotency FIRST (issue #30): never pay for the ~250s network cascade on a
            # paper that already has full text. Mirrors analysis.py / deep_analysis.py.
            if (paper.metadata_ or {}).get("fulltext_status") == "acquired":
                already_acquired += 1
                # The byline parse sits inside the SAME branch that sets
                # `fulltext_status = "acquired"` below, so a paper acquired before that
                # parse existed takes this early return and never gets a
                # `fulltext_byline` entry at all, keeping its citation stuck on
                # OpenAlex's own author order forever. Backfill it here, once, from the
                # chunks already stored: no network call, and a paper that already
                # carries the key (backfilled or acquired since) is never re-parsed.
                stored_metadata = paper.metadata_ or {}
                if "fulltext_byline" not in stored_metadata:
                    prefix_parts: list[str] = []
                    prefix_len = 0
                    for chunk in stored_metadata.get("fulltext_chunks") or []:
                        prefix_parts.append((chunk or {}).get("text", ""))
                        prefix_len += len(prefix_parts[-1])
                        if prefix_len >= IDENTITY_TITLE_PREFIX_CHARS:
                            break
                    parsed_surnames = _parse_byline_surnames("".join(prefix_parts))
                    backfilled_byline = (
                        {"surnames": parsed_surnames, "source": "fulltext"}
                        if _accept_byline(parsed_surnames, _author_names(paper))
                        else {"source": "openalex"}
                    )
                    async with session_factory() as session:
                        db_paper = await session.get(Paper, paper.id)
                        if db_paper is not None:
                            backfilled_metadata = dict(db_paper.metadata_ or {})
                            backfilled_metadata["fulltext_byline"] = backfilled_byline
                            db_paper.metadata_ = backfilled_metadata
                            await session.commit()
                async with session_factory() as session:
                    await task_service.update_job_status(
                        session,
                        job_id,
                        JobStatus.running,
                        progress=(idx + 1) / max(total, 1),
                        progress_message=f"Processed {idx + 1}/{total} papers...",
                    )
                continue

            # --- Build the ordered candidate list (F4 acquisition design) ---
            # 1. every Unpaywall OA location, already ordered publishedVersion,
            #    acceptedVersion, submittedVersion, then anything else
            #    (`UnpaywallClient.lookup_locations`);
            # 2. the stored `full_text_url` on the paper row;
            # 3. OpenAlex/other `oa_url` in the paper metadata.
            candidates: list[tuple[str, str]] = []
            if paper.doi:
                try:
                    for pdf_url in await unpaywall.lookup_locations(paper.doi):
                        candidates.append((pdf_url, "unpaywall"))
                except Exception as exc:
                    logger.warning("Unpaywall failed for DOI %s: %s", paper.doi, exc)
            if paper.full_text_url:
                candidates.append((paper.full_text_url, "stored_url"))
            if paper.metadata_:
                oa_url = paper.metadata_.get("oa_url") or paper.metadata_.get("open_access_url")
                if oa_url:
                    candidates.append((oa_url, "metadata_oa_url"))

            # --- Try each candidate in order, stopping at the first download that
            # passes the identity guard. A real PDF describing a DIFFERENT paper does
            # not stop the search early -- the next candidate might still be right --
            # but it is remembered so a paper where NO candidate ever matches is still
            # reported as `wrong_work` rather than a bare, unexplained `abstract_only`.
            pdf_bytes: bytes | None = None
            source = "none"
            text = ""
            chunks: list[dict] = []
            dropped_count = 0
            extraction_error: str | None = None
            saw_real_pdf = False
            saw_any_candidate = False
            # Stays True only if every candidate that was actually attempted raised
            # `PublisherInterstitialError` -- a plain download failure (`None`) or a
            # genuine PDF (whether or not its identity matched) both clear it, since
            # neither means "every attempt looked like a publisher block".
            all_interstitial = True
            last_wrong_work_fragment: str | None = None

            for candidate_url, candidate_source in candidates:
                saw_any_candidate = True
                try:
                    candidate_bytes = await fetch_pdf_from_url(candidate_url)
                except PublisherInterstitialError:
                    continue
                except Exception as exc:
                    logger.warning(
                        "PDF fetch raised for paper %s (%s): %s", paper.id, candidate_url, exc
                    )
                    all_interstitial = False
                    continue
                all_interstitial = False
                if not candidate_bytes:
                    continue
                try:
                    # pymupdf extraction and chunking are CPU-bound blocking calls; run
                    # them off the event loop (D20 / issue #27).
                    candidate_text = await asyncio.to_thread(
                        extract_text_from_pdf, candidate_bytes
                    )
                except Exception as exc:
                    logger.warning(
                        "PDF extraction failed for paper %s (%s): %s",
                        paper.id, candidate_url, exc,
                    )
                    extraction_error = str(exc)
                    continue
                # A candidate that reaches this point is a genuine, decodable PDF --
                # remembered so a paper where every candidate is real but describes
                # the wrong work is still reported as `wrong_work`, not a bare,
                # unexplained `abstract_only`.
                saw_real_pdf = True
                # Reject a full text that does not describe this paper's own
                # title/author BEFORE it is ever chunked or stored, so the claim
                # verifier can never be shown the wrong paper's text.
                identity_ok, wrong_work_fragment = check_full_text_identity(
                    expected_title=paper.title or "",
                    extracted_text=candidate_text,
                    expected_authors=_author_names(paper),
                )
                if not identity_ok:
                    logger.warning(
                        "Full text for paper %s does not match its title/author "
                        "(wrong_work): %s",
                        paper.id,
                        wrong_work_fragment,
                    )
                    last_wrong_work_fragment = wrong_work_fragment
                    continue
                pdf_bytes = candidate_bytes
                source = candidate_source
                text = candidate_text
                break

            # --- Process result ---
            async with session_factory() as session:
                db_paper = await session.get(Paper, paper.id)
                if db_paper is None:
                    continue

                metadata = dict(db_paper.metadata_ or {})

                if pdf_bytes:
                    # `drop_reference_chunks=False`: this function does its own explicit
                    # drop just below (so it can count and report the number dropped),
                    # rather than relying on chunk_text's own default drop.
                    chunks = await asyncio.to_thread(
                        chunk_text, text, drop_reference_chunks=False
                    )
                    # Drop reference-list/back-matter chunks before storage, so the
                    # verifier is never shown bibliography text as if it were prose.
                    chunks, dropped_count = drop_reference_and_backmatter_chunks(chunks)
                    metadata["fulltext_status"] = "acquired"
                    metadata["fulltext_source"] = source
                    metadata["fulltext_chunks"] = chunks
                    metadata["fulltext_char_count"] = len(text)
                    # Parse the acquired text's own byline and accept it only when it
                    # accounts for every OpenAlex author surname -- otherwise keep
                    # today's OpenAlex-derived author string, with the provenance
                    # recorded either way, so the failure mode of a layout this parser
                    # does not handle is no change, never a wrong string.
                    parsed_surnames = _parse_byline_surnames(text)
                    if _accept_byline(parsed_surnames, _author_names(paper)):
                        metadata["fulltext_byline"] = {
                            "surnames": parsed_surnames,
                            "source": "fulltext",
                        }
                    else:
                        metadata["fulltext_byline"] = {"source": "openalex"}
                    # A paper retried after an earlier wrong_work rejection must not
                    # keep that rejection's markers once it succeeds.
                    metadata.pop("fulltext_reason", None)
                    metadata.pop("fulltext_wrong_work_fragment", None)
                    if dropped_count:
                        metadata["fulltext_dropped_reference_chunks"] = dropped_count
                        reference_chunks_dropped += dropped_count
                    acquired += 1
                elif saw_real_pdf:
                    # At least one candidate produced a genuine PDF, but every one of
                    # them described the wrong paper -- report the last mismatch found.
                    metadata["fulltext_status"] = "abstract_only"
                    metadata["fulltext_reason"] = "wrong_work"
                    metadata["fulltext_wrong_work_fragment"] = last_wrong_work_fragment
                    wrong_work += 1
                    abstract_only += 1
                elif saw_any_candidate and all_interstitial:
                    # Every attempt across the whole cascade looked like a publisher
                    # interstitial -- never a real PDF, and never a plain download
                    # failure either. Distinct from `wrong_work` (a real PDF, just
                    # about the wrong paper): here no real PDF was ever seen at all.
                    metadata["fulltext_status"] = "abstract_only"
                    metadata["fulltext_reason"] = "publisher_interstitial"
                    interstitial += 1
                    abstract_only += 1
                elif extraction_error is not None:
                    metadata["fulltext_status"] = "abstract_only"
                    metadata["fulltext_error"] = extraction_error
                    abstract_only += 1
                else:
                    metadata["fulltext_status"] = "abstract_only"
                    abstract_only += 1

                db_paper.metadata_ = metadata
                await session.commit()

            # Update progress
            progress = (idx + 1) / max(total, 1)
            async with session_factory() as session:
                await task_service.update_job_status(
                    session,
                    job_id,
                    JobStatus.running,
                    progress=progress,
                    progress_message=f"Processed {idx + 1}/{total} papers...",
                )

        # Final status: `wrong_work` and `reference_chunks_dropped` are added only when
        # a guard actually fired, so a run that never triggers either guard keeps the
        # original three-key result shape byte identical. `interstitial` (F4
        # acquisition) follows the same rule.
        result = {
            "acquired": acquired,
            "abstract_only": abstract_only,
            "already_acquired": already_acquired,
        }
        if wrong_work:
            result["wrong_work"] = wrong_work
        if interstitial:
            result["interstitial"] = interstitial
        if reference_chunks_dropped:
            result["reference_chunks_dropped"] = reference_chunks_dropped
        async with session_factory() as session:
            await task_service.update_job_status(
                session,
                job_id,
                JobStatus.completed,
                progress=1.0,
                progress_message="Full-text acquisition complete.",
                result=result,
            )
    except Exception as e:
        logger.exception("Full-text acquisition failed for project %s", project_id)
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e)
            )


#: A markdown heading line (a whole finalize "paragraph" block that is nothing but a
#: heading), mirroring `app.services.writing._HEADING_LINE_RE` but kept as this
#: module's own constant rather than imported, since `app.services.writing` imports
#: from `app.services.fulltext`, not the other way around. Finalize (below) never
#: applies sentence-removal rules inside a heading block: a heading is never split into
#: fragments, never tagged an "uncited finding" sentence, and is always kept verbatim.
_FINALIZE_HEADING_BLOCK_RE = re.compile(r"^#{1,6}[ \t]+")

#: A whole block that is nothing but one bold run, start to end, mirroring
#: `app.services.writing._MARKDOWN_BOLD_HEADING_RE` exactly -- the writer's other way of
#: marking a sub-heading, promoted by `_build_section_tiptap_nodes` to a real level-3
#: heading node instead of a paragraph. `_FINALIZE_HEADING_BLOCK_RE` alone does not match
#: this shape, so without this pattern a bold sub-heading block would be split into
#: "sentences" by `unclassified_body_sentences` and reported as a coverage gap for
#: something that was never body prose at all; kept as this module's own constant, not
#: imported, for the same reason `_FINALIZE_HEADING_BLOCK_RE` is.
_FINALIZE_BOLD_HEADING_BLOCK_RE = re.compile(r"^\*\*([^*]+)\*\*$")

#: Sentence-terminal punctuation: a whole-line bold run ending this way reads as a claim
#: sentence the writer happened to bold, not a sub-heading -- a genuine sub-heading is a
#: short label, never a full sentence.
_SENTENCE_TERMINAL_PUNCTUATION = (".", "!", "?")


def _is_bold_heading_candidate(inner_text: str) -> bool:
    """True when *inner_text* -- the captured wording of a whole-line bold run
    (`_FINALIZE_BOLD_HEADING_BLOCK_RE.match(stripped).group(1)`) -- still reads as a
    sub-heading rather than a claim sentence the writer happened to bold.

    Without this guard, `_is_heading_block` would treat ANY whole-line bold run as a
    heading, which is also what `app.services.writing._build_section_tiptap_nodes`
    promotes to a real level-3 heading node. A bolded claim -- ``**Direct corrections
    outperformed metalinguistic codes by 32% (Smith, 2020).**`` -- matches that shape
    too, so it would be "always kept verbatim, never split into sentences and never
    removed": an unsupported claim would survive the delivered text with its link
    dropped from ``surviving_links`` (no row in the final report), invisible to
    `unclassified_body_sentences`'s own coverage scan and to
    `extract_claims_from_document` (paragraph nodes only) on every later heal.

    False when *inner_text* ends in `_SENTENCE_TERMINAL_PUNCTUATION` (a genuine
    sub-heading is a short label, not a full sentence) OR carries a citation
    (`_CITATION_RE`, mirrored in `app.services.writing._is_bold_heading_candidate`'s own
    caller by importing this exact function -- writing.py imports from this module, not
    the other way around, so there is one predicate, not two copies to keep in sync)."""
    stripped = inner_text.strip()
    if not stripped:
        return False
    if stripped.endswith(_SENTENCE_TERMINAL_PUNCTUATION):
        return False
    return not _CITATION_RE.search(stripped)


#: The literal marker `writing.py`'s own _BASE_RULES allows the model to write on a
#: statement no supplied material could support. Finalize strips it (and any leading run
#: of whitespace immediately before it) from the surviving text of any sentence it
#: appears in. The whitespace INSIDE the marker is matched with ``\s+`` rather than one
#: literal space, so a marker a formatting mark split across two text nodes, or that the
#: model itself typed with a doubled space or a tab, is the same marker to every rule
#: below instead of one that only a prior whitespace collapse could reach. Widened
#: from the exact-form ``\[\s*NEEDS\s+CITATION\s*\]`` to also swallow whatever
#: the writer put between "CITATION" and the closing bracket -- a colon and an
#: explanatory clause, an em dash, or nothing at all -- since the model sometimes
#: writes the marker as its own editorial note ("[NEEDS CITATION: evidence on peer
#: feedback relative to teacher feedback ...]") rather than the bare form the writing
#: rules ask for. ``[^\]]*`` cannot cross into a second bracketed run, so an ordinary
#: numbered citation ("[12]") or a bracketed author-year aside ("[Smith, 2020]") is
#: never matched: neither contains the literal word "CITATION" preceded by "NEEDS".
_NEEDS_CITATION_RE = re.compile(r"\s*\[\s*NEEDS\s+CITATION\b[^\]]*\]", re.IGNORECASE)

#: Statuses treated as "this citation can never be checked, but the sentence itself is
#: not thereby wrong": only the citation is dropped, not the whole sentence, unless
#: every citation on the sentence turns out to be one of these.
_FINALIZE_UNCHECKABLE_STATUSES = frozenset({"no_full_text"})
#: Any other non-"verified" status (unsupported, needs_nuance, error, or a (sentence,
#: key) pair `claim_status` has no entry for at all) removes the whole sentence.


def _is_heading_block(text: str) -> bool:
    """True when *text* (a whole finalize paragraph block) is a markdown heading -- a
    ``#``-style line, or a whole-line bold run that still reads as a sub-heading rather
    than a claim sentence (`_is_bold_heading_candidate`): the second form is the
    writer's other way of marking a sub-heading, and `_build_section_tiptap_nodes`
    already renders it as a real heading node, not a paragraph, under the identical
    condition."""
    stripped = text.strip()
    if _FINALIZE_HEADING_BLOCK_RE.match(stripped):
        return True
    bold_match = _FINALIZE_BOLD_HEADING_BLOCK_RE.match(stripped)
    return bool(bold_match and _is_bold_heading_candidate(bold_match.group(1)))


#: A short, general stopword list for `_title_echo_words` -- articles, prepositions
#: and a handful of common function words that routinely differ between a heading's
#: own short wording and a body paragraph that otherwise echoes it (e.g. a heading
#: reading "Learner engagement with written corrective feedback" against a demoted
#: sub-heading reading "Engagement with Written Corrective Feedback in Second-
#: Language Writing"): dropping them keeps the overlap check reading the words that
#: actually carry the topic, not the function words every English sentence shares.
_TITLE_ECHO_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "with", "for", "and", "or", "to", "at",
    "by", "from", "into", "onto", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "their", "its", "it",
})

_TITLE_ECHO_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ɏ]+")

#: A minimal, deliberately conservative suffix strip -- not a real stemmer, just
#: enough to fold a title's own noun/verb form onto a restatement's ("codes" /
#: "coded", "writing" / "written" is NOT folded, since "en" is not one of these
#: suffixes -- an accepted, narrow miss, not a false positive). Checked longest
#: suffix first so "-ing" is never mistaken for a shorter match, and only ever
#: applied once, keeping at least 3 characters, so a genuinely short, unrelated word
#: ("as", "is") is never mangled into another one by coincidence.
_TITLE_ECHO_SUFFIXES = ("ing", "ed", "es", "s")


def _title_echo_stem(word: str) -> str:
    for suffix in _TITLE_ECHO_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _title_echo_words(text: str) -> set[str]:
    """The lower-cased, stopword-stripped, lightly stemmed (`_title_echo_stem`) words
    of *text* -- shared by both the app's own drop (below) and the checker's own copy
    (`demo/check_delivered.py`'s ``_title_echo_words``), so a heading and a paragraph
    are compared on the same vocabulary on both sides."""
    return {
        _title_echo_stem(word)
        for word in _TITLE_ECHO_WORD_RE.findall((text or "").lower())
        if word not in _TITLE_ECHO_STOPWORDS
    }


def _is_title_echo(paragraph_text: str, reference_texts: Sequence[str]) -> bool:
    """True when *paragraph_text* -- a whole, non-heading body paragraph -- echoes one
    of *reference_texts* (a section's own title, or a heading still standing in its
    own draft): at least half of the SMALLER of the two texts' own significant words
    (`_title_echo_words`) also occur in the other (the containment coefficient,
    ``|A intersect B| / min(|A|, |B|)`` -- more forgiving than a plain overlap ratio
    of the combined vocabulary when one side adds its own extra words the other
    lacks, which is exactly the shape a restated title takes).

    Not exact string equality: a heading demoted to plain text by the very next step
    in `finalize_generated_section` keeps the heading's own wording verbatim (an exact
    match, the common case -- a short section whose only line was itself a restated
    title), but a model's OWN sub-heading, demoted the same way after its real body
    was removed by an earlier rule, or a restatement of the section's own externally
    injected title, is often reworded with an article, a suffix ("... in Second-
    Language Writing"), a different inflection ("codes" against "coded"), or a
    dropped qualifier rather than word for word -- still unmistakably the same short
    label once function words are set aside and the two texts' remaining words are
    lightly stemmed, never a genuine finding or framing sentence with something new
    to say.

    Guarded by *paragraph_text* itself being short (at most 12 significant words --
    generously more than any real heading in this product's writing rules, and far
    short of an ordinary multi-clause body sentence): a long, substantive paragraph
    that merely shares a heading's own topic vocabulary is never caught by this
    accident of overlap, no matter how high the overlap happens to be.

    Also guarded by at least TWO shared words, not the ratio alone: a heading as
    short and generic as the section type's own default title ("Literature Review",
    two words) shares one word with an unrelated sentence often enough (any sentence
    that happens to use "review" as a verb, for instance) to clear a 50% containment
    ratio on one word alone. Every real occurrence this predicate is built from
    (and the three further ones found by later diagnosis) shares at least three
    words once stemmed; two is already a conservative floor above the one-word
    false positive a short generic title produces."""
    paragraph_words = _title_echo_words(paragraph_text)
    if not paragraph_words or len(paragraph_words) > 12:
        return False
    for reference in reference_texts:
        reference_words = _title_echo_words(reference)
        if not reference_words:
            continue
        overlap = paragraph_words & reference_words
        smaller = min(len(paragraph_words), len(reference_words))
        if smaller and len(overlap) >= 2 and len(overlap) / smaller >= 0.5:
            return True
    return False


def _strip_needs_citation(text: str) -> tuple[str, int]:
    """(text with every literal "[NEEDS CITATION]" marker removed, how many were
    removed)."""
    count = len(_NEEDS_CITATION_RE.findall(text))
    if not count:
        return text, 0
    return _NEEDS_CITATION_RE.sub("", text), count


def _finalize_sentence_norm_counted(text: str) -> tuple[str, int]:
    """(*text* in finalize's one normalised sentence form, how many
    ``[NEEDS CITATION]`` markers that form removed from it).

    The form is: every marker stripped (`_NEEDS_CITATION_RE`, which tolerates any
    whitespace inside the marker itself), then every whitespace run collapsed to one
    space, then trimmed. It is idempotent -- normalising an already normalised sentence
    returns it unchanged, with a count of zero -- which is what lets finalize write the
    normalised text straight back into the document as the surviving sentence
    (`_finalized_sentence_text`) and still recognise it on the next heal.

    ONE form, used for every comparison finalize makes about a sentence: the fragment
    the splitter hands `_finalize_paragraph_text`, that function's per-sentence link
    map, its uncited "finding" set and its claim-status lookup are all built with this
    function, so no rule can see a marker, or a whitespace difference, that another
    rule does not. Keeping the marker out of all four is what makes
    the three rules agree about which sentence a fragment is. The splitter puts a
    marker written at the end of one sentence at the HEAD of the fragment for the
    sentence that follows it, so that fragment carries a marker its linker never
    recorded; and the linker records a sentence the writer itself flagged WITH the
    marker, since the citation-link prompt asks for each uncited sentence exactly as it
    appears in the text. Stripping it on one side only moved the mismatch from one rule
    to another instead of closing it."""
    stripped, count = _strip_needs_citation(text or "")
    return _doc_normalize_ws(stripped), count


def _finalize_sentence_norm(text: str) -> str:
    """*text* in finalize's one normalised sentence form, for building a key to compare
    a fragment against; see `_finalize_sentence_norm_counted`, whose marker count this
    discards."""
    return _finalize_sentence_norm_counted(text)[0]


def _finalize_status_severity(status: str | None) -> int:
    """How far one claim status goes towards removing text, for choosing between two
    verdicts recorded against sentences finalize cannot tell apart: 0 keeps the
    sentence and its citation ("verified"), 1 drops that one citation
    (`_FINALIZE_UNCHECKABLE_STATUSES`), 2 removes the whole sentence (unsupported,
    needs_nuance, error, or no verdict at all).

    Two claims collide when their sentences differ only in whitespace or in a
    ``[NEEDS CITATION]`` marker, because `_finalize_paragraph_text` looks a verdict up
    by the normalised sentence. The more severe verdict wins, so a collision can only
    ever remove text the exit invariant is unsure about, never ship it."""
    if status == "verified":
        return 0
    if status in _FINALIZE_UNCHECKABLE_STATUSES:
        return 1
    return 2


#: The number words `_enumeration_announced_count` recognises opening an uncited
#: framing sentence ("Three gaps emerge."), alongside a plain numeral ("4 themes
#: follow."), which needs no lookup.
_ENUMERATION_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
#: A number word or numeral, a plural noun, then a verb that actually ANNOUNCES an
#: enumeration -- "Three gaps emerge.", "4 themes follow.", "Two limitations stand
#: out.". Deliberately narrowed to the three verbs that only ever introduce a list,
#: excluding generic existential verbs such as "are", "were", "exist(s/ed)" and
#: "remain(s/ed)" that a plain factual sentence uses just as often as a real
#: announcement ("Three studies were conducted in naturalistic classrooms.", "Four
#: papers are included in this review.", "Two limitations remain unaddressed in the
#: literature." are none of them announcing anything to be counted). Leading markdown
#: emphasis characters (the writer's own ``**``/``*``/``_``) are skipped, not matched,
#: so a bolded opener is still recognised.
_ENUMERATION_OPENER_RE = re.compile(
    r"^[\s*_]*(?P<num>[A-Za-z]+|\d+)\s+[a-z]+s\s+"
    r"(?:emerge[sd]?|stands?\s+out|follow(?:s|ed)?)\b",
    re.IGNORECASE,
)
#: A discourse connective or ordinal transition, sentence-initial.
_DANGLING_CONNECTIVE_RE = re.compile(
    r"^[\s*_]*(however|moreover|similarly|crucially|notably|complementing|furthermore|"
    r"additionally|conversely|nevertheless|nonetheless|consequently|meanwhile|"
    r"likewise|correspondingly|first|second|third|fourth|fifth|sixth|finally)\b",
    re.IGNORECASE,
)
#: The ordinal words `_next_fragment_resolves_enumeration_as_a_list` looks for, in
#: order, to recognise a colon-free, self-resolving list an enumeration opener's own
#: next sentence can also take ("Three gaps emerge. First, designs are short; second,
#: corpora are small; third, engagement is untracked.") -- a shape
#: `_enumeration_announced_count`'s own colon check cannot see, because the colon (if
#: any) would be in a DIFFERENT sentence than the opener's own.
_ORDINAL_SEQUENCE_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth"]
#: An unresolved deictic reference, sentence-initial -- "the study", "the authors",
#: "the same authors" verbatim, and the generalised "such X", "this X", "these X"
#: templates.
_DANGLING_DEICTIC_RE = re.compile(
    r"^[\s*_]*(the\s+stud(?:y|ies)'?s?\b|the\s+same\s+authors?\b|the\s+authors?\b|"
    r"such\b|this\s+\w+|these\s+\w+)",
    re.IGNORECASE,
)


def _enumeration_announced_count(sentence: str) -> int | None:
    """The number an uncited framing sentence's own opening announces -- ``None`` when
    *sentence* does not open with a number word or numeral, a
    plural noun and an enumeration verb (`_ENUMERATION_OPENER_RE`), or when it carries
    its own colon: "Three gaps remain: short designs; small corpora; and no tracking
    of engagement over time." announces and resolves its own count in one sentence, so
    counting the SENTENCES that follow it (a property test found this shape) would
    fault a self-contained list for a shortfall it does not have -- the announcement's
    own colon is this function's signal that nothing external needs to satisfy it."""
    if ":" in sentence:
        return None
    match = _ENUMERATION_OPENER_RE.match(sentence)
    if not match:
        return None
    raw = match.group("num").lower()
    if raw.isdigit():
        return int(raw)
    return _ENUMERATION_NUMBER_WORDS.get(raw)


def _next_fragment_resolves_enumeration_as_a_list(text: str, announced: int) -> bool:
    """True when *text* -- the very next surviving fragment
    after an enumeration opener with no colon of its own -- itself lists at least
    *announced* items via sentence-internal ordinal markers, in increasing order
    ("First, designs are short; second, corpora are small; third, engagement is
    untracked (Smith, 2020)."). This is a colon-free list the opener's own colon check
    (`_enumeration_announced_count`) cannot see: the colon-free list splits on
    semicolons, not periods, so it is one sentence, not *announced* separate ones, and
    counting SENTENCES that follow the opener (the ordinary shortfall check) would
    fault it for a shortfall it does not have -- the same false positive the colon
    check exists to avoid, one sentence later."""
    if announced > len(_ORDINAL_SEQUENCE_WORDS):
        return False
    lowered = text.lower()
    positions: list[int] = []
    for word in _ORDINAL_SEQUENCE_WORDS[:announced]:
        match = re.search(rf"\b{word}\b", lowered)
        if match is None:
            return False
        positions.append(match.start())
    return positions == sorted(positions)


def _opens_with_dangling_marker(sentence: str) -> bool:
    """True when *sentence* opens with a discourse connective/ordinal or an unresolved
    deictic: `_DANGLING_CONNECTIVE_RE` or `_DANGLING_DEICTIC_RE`."""
    return bool(_DANGLING_CONNECTIVE_RE.match(sentence) or _DANGLING_DEICTIC_RE.match(sentence))


def _drop_dangling_framing_sentences(
    records: Sequence[Mapping[str, Any]], stats: dict[str, int]
) -> list[dict[str, Any]]:
    """The deterministic coherence pass over one paragraph's own surviving fragments
    (``_finalize_paragraph_text``'s ordinary per-sentence removal rules already ran),
    run together with rule (a)'s own heading-with-no-body cleanup. A cited sentence, a
    fragment with no earlier removal anywhere in its own paragraph (a section's own
    opening framing paragraph, or a mid-sentence adverb such as "similarly"), and a
    sentence that is not its own paragraph's first delivered fragment are all, by
    design, outside what this deterministic pass can safely remove or rewrite without
    either deleting cited content or adding an LLM turn. A dangling-or-misattributed
    sentence outside that scope is residue this pass leaves behind, and is never gated
    on this module's or the checker's exit code (see `check_dangling_framing`'s own
    docstring for the matching, narrower checker rule).

    Each record carries ``text``, ``has_citation`` (a cited sentence is never dropped
    here -- both rules are scoped to an *uncited* framing sentence),
    ``preceded_by_removal`` (whether one of the three ordinary rules had already
    removed a fragment BEFORE the first fragment this paragraph delivers at all -- see
    the seeding note below) and ``uncited_entries`` (the raw uncited-sentence entries
    this fragment matched; a fragment carrying one with ``"unclassified": True`` is
    exempt from both rules below, the same way a cited fragment already is: the whole
    point of ``"unclassified"`` is that no call has yet said this sentence is dangling
    filler rather than framing, and `finalize_generated_section`'s own rule (a) already
    grants a whole such paragraph the identical exemption).

    (b): an uncited fragment whose own opening announces an enumeration
    (`_enumeration_announced_count`) is dropped when fewer fragments than announced
    still follow it AND the very next surviving fragment does not itself resolve the
    count as a colon-free list of its own
    (`_next_fragment_resolves_enumeration_as_a_list`) -- independent of
    ``preceded_by_removal``, since a bare stub with no matching enumeration is
    self-evidently broken whether or not anything was removed to make it so.

    (c): an uncited fragment that opens with a discourse connective/ordinal or an
    unresolved deictic (`_opens_with_dangling_marker`) is dropped once trouble is
    "in the air" for it -- either (b) removed an earlier fragment of the same
    paragraph (this propagates forward without limit: a genuine unfulfilled count is
    a real, permanent absence, whatever else survives after it), or this fragment
    would otherwise be the very FIRST one this paragraph delivers at all and an
    ordinary rule had already removed something before it (``preceded_by_removal``).
    An ordinary rule's removal only ever poisons that first
    delivered fragment -- once even one fragment has survived, it is itself a real,
    visible antecedent for whatever discourse marker follows it, however many more
    ordinary removals happen afterward ("Tutoring boosts accuracy (Smith, 2020). [an
    unrelated sentence removed] However, this finding rests on a single instructional
    setting." keeps the "However" sentence, because the Smith sentence is still a real
    antecedent for it); only an enumeration's own genuine shortfall is severe enough to
    keep poisoning every fragment after it regardless of what else survives in
    between -- `check_dangling_framing`'s own docstring already argues exactly this.

    Returns the surviving records, in order; a caller left with none, or with a
    paragraph now missing every cited sentence, decides what that means for the whole
    paragraph -- this function only ever drops one fragment at a time.

    Both phases repeat to a fixed point (bounded by ``len(records)`` passes, since each
    pass that changes anything removes at least one more fragment): a single forward
    pass can under-count, because (b)'s own ``following`` tally for an earlier opener
    is taken before a LATER opener's own (b) violation -- discovered further on in the
    very same pass -- has removed anything yet. Two enumeration openers in one
    paragraph ("One gap emerges. Three gaps emerge. These gaps ...", an edge case a
    property test over random fragment orders found, not one the report's own examples
    needed) is the shape that shows it: the second opener's shortfall cascades and
    removes the sentence that was propping up the first opener's own count, which a
    single pass never revisits."""
    removed = [False] * len(records)
    # The only fragment an ORDINARY rule's own removal can ever poison is the first one
    # this paragraph delivers at all -- see the docstring above.
    any_removed = [False] * len(records)
    if records:
        any_removed[0] = bool(records[0]["preceded_by_removal"])

    # A fragment carrying an "unclassified" entry is exempt from both rules below, the
    # way a cited fragment already is -- `_finalize_paragraph_text`'s own docstring and
    # `writing.py`'s revision comment both say such a sentence is never removed, only
    # `sentences_unclassified_kept`, and rule (a) already grants its whole paragraph
    # the same exemption (see `finalize_generated_section`'s `has_unclassified` check).
    # Without this exemption, a genuinely uncertain sentence that happens to also open
    # with an enumeration or a discourse marker would be dropped here, silently, by a
    # pass that has no way to tell "uncertain" from "dangling".
    is_unclassified = [
        any(entry.get("unclassified") for entry in record.get("uncited_entries", ()))
        for record in records
    ]

    changed = True
    while changed:
        changed = False
        for i, record in enumerate(records):
            if removed[i] or record["has_citation"] or is_unclassified[i]:
                continue
            announced = _enumeration_announced_count(record["text"])
            if announced is None:
                continue
            following_indexes = [j for j in range(i + 1, len(records)) if not removed[j]]
            if len(following_indexes) < announced:
                # A colon-free list resolves the SAME count one sentence later ("Three
                # gaps emerge. First, ...; second, ...; third, ... (Smith, 2020)."),
                # which is one surviving fragment, not *announced* of them -- checked
                # against the very next surviving fragment only.
                resolved_as_list = following_indexes and (
                    _next_fragment_resolves_enumeration_as_a_list(
                        records[following_indexes[0]]["text"], announced
                    )
                )
                if not resolved_as_list:
                    removed[i] = True
                    stats["sentences_removed_dangling"] += 1
                    for j in range(i + 1, len(records)):
                        any_removed[j] = True
                    changed = True

        for i, record in enumerate(records):
            if (
                removed[i]
                or record["has_citation"]
                or not any_removed[i]
                or is_unclassified[i]
            ):
                continue
            if _opens_with_dangling_marker(record["text"]):
                removed[i] = True
                stats["sentences_removed_dangling"] += 1
                for j in range(i + 1, len(records)):
                    any_removed[j] = True
                changed = True

    return [dict(record) for i, record in enumerate(records) if not removed[i]]


def unclassified_body_sentences(
    text: str,
    links: Sequence[Mapping[str, Any]],
    uncited_sentences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every body sentence of *text* that a citation-link call's own ``links`` and
    ``uncited_sentences`` together leave unclassified.

    Headings (`_is_heading_block`) are excluded, and every other block is split into
    sentences with `_sentence_fragments` -- the same splitter `extract_claims_from_document`
    and `_finalize_paragraph_text` use to cut a claim's own sentence out of a paragraph,
    so a sentence found here is cut on exactly the same boundary finalize itself will
    later look it up by. A sentence counts as classified when its own normalised text
    (`_finalize_sentence_norm`) equals a link's ``sentence`` or an uncited-sentence
    entry's ``sentence``, OR when one of those two normalised texts CONTAINS it, or it
    contains one of them: `_CLAIM_SENTENCE_SPLIT_RE` only tolerates
    the "vs." abbreviation, so any other abbreviation the linker's own sentence
    boundary disagrees with (e.g. "U.S.") splits a sentence the linker reported whole
    into two fragments that would otherwise match neither list -- a splitter/linker
    boundary disagreement, not a real gap, and it must never make correct, cited text
    removable. ``paragraph_index`` is ignored for this membership test, since the two
    lists are not always indexed against the same paragraph numbering the caller is
    checking (`resolve_unclassified_sentences`'s own caller may hold a citation-link
    map computed from a different pass's own text).

    This is the gap this function exists to close: a sentence the citation-link call
    returns in NEITHER list is invisible both to the verification gate (nothing to
    verify) and to finalize's own rule 1 (which only removes a sentence *listed* in
    ``uncited_sentences``
    tagged "finding" -- an unlisted sentence is kept, by design, so a hand-written or
    locked paragraph is never silently deleted). Naming every such gap here, once, is
    what lets `resolve_unclassified_sentences` close it instead of leaving a specific,
    unverified empirical claim to survive in the delivered text with no citation, no
    verification row and no record anywhere that it was never checked.

    A duplicate sentence within *text* (the same normalised text more than once) is
    reported once, at its first occurrence's ``paragraph_index`` -- the same accepted
    limitation `_finalize_paragraph_text` already documents for a paragraph that repeats
    a sentence: nothing a linker returns says which occurrence it meant, so there is no
    position to key a second, distinct gap on.

    An ``uncited_sentences`` entry still carrying ``"unclassified": True`` does NOT
    count as covered: that flag records that no model call ever
    positively classified the sentence -- `resolve_unclassified_sentences`'s own retry
    call either failed outright or succeeded without ever mentioning it -- and treating
    it as coverage would freeze the sentence as unclassified forever, since every later
    caller of this function (a revision's own second citation-link pass, the standalone
    verify-and-heal action) would then see no gap at all and never try again. Reporting
    it as a gap again gives every such sentence one fresh, bounded retry attempt on each
    heal, exactly as a sentence with no entry at all gets on its first one.

    A sentence that already has a link is ALSO reported as a gap when that link's own
    ``coverage_incomplete`` flag is set (``app.agents.
    citation_link_agent._flag_sentence_coverage``) -- the citation-link step's own
    propositions and citations leave at least one non-frame residue. Sent back through
    this same bounded retry, the model gets one more chance to report the missing
    residue as its own proposition (its own prompt asks for exactly that); a sentence
    with no coverage problem at all is never re-sent just because it already has a
    link.

    Returns one ``{"paragraph_index", "sentence"}`` dict per gap, in document order."""
    from app.agents.citation_link_agent import split_paragraphs

    covered_norms = [
        norm
        for norm in (
            _finalize_sentence_norm(item.get("sentence") or "")
            for item in (
                *links,
                *(entry for entry in uncited_sentences if not entry.get("unclassified")),
            )
        )
        if norm
    ]
    covered = set(covered_norms)
    coverage_incomplete_norms = {
        norm
        for norm in (
            _finalize_sentence_norm(link.get("sentence") or "")
            for link in links
            if link.get("coverage_incomplete")
        )
        if norm
    }
    gaps: dict[str, dict[str, Any]] = {}
    for index, block in enumerate(split_paragraphs(text)):
        if _is_heading_block(block):
            continue
        for start, end in _sentence_fragments(block):
            fragment = block[start:end].strip()
            if not fragment:
                continue
            norm = _finalize_sentence_norm(fragment)
            if not norm or norm in gaps:
                continue
            is_coverage_gap = norm in coverage_incomplete_norms or any(
                norm in c or c in norm for c in coverage_incomplete_norms
            )
            if norm in covered and not is_coverage_gap:
                continue
            # A splitter/linker boundary disagreement -- the fragment is a substring of
            # a covered sentence, or a covered sentence is a substring of it --
            # classifies the fragment too, exact equality or not.
            if not is_coverage_gap and any(
                norm in covered_norm or covered_norm in norm for covered_norm in covered_norms
            ):
                continue
            gaps[norm] = {"paragraph_index": index, "sentence": fragment}
    return list(gaps.values())


async def resolve_unclassified_sentences(
    text: str,
    links: Sequence[Mapping[str, Any]],
    uncited_sentences: Sequence[Mapping[str, Any]],
    linker: Callable[[str], Awaitable[Any]],
) -> tuple[list[dict], list[dict], LLMCallProvenance | None, int, bool]:
    """Bounded, one-call repair of `unclassified_body_sentences`'s own gaps: every gap
    is sent back to *linker* together, as its own miniature text (one sentence per synthetic
    paragraph, in gap order), exactly ONCE more -- never a retry loop, and never one
    call per gap -- and whatever it resolves is merged back into *links* and
    *uncited_sentences*, remapped from the retry call's own paragraph numbering to
    each gap's real ``paragraph_index`` in *text* (matched by the returned entry's own
    ``sentence``, not by the retry call's paragraph number, since a model is not
    guaranteed to report the same paragraph index back for a text it did not choose
    the shape of).

    A gap the retry still cannot classify is appended to the returned uncited-sentence
    list as a synthetic entry: ``tag: "finding"``. Whether that entry also carries
    ``"unclassified": True`` depends on WHY the retry did not classify it, since the two
    causes call for different treatment. When the retry call itself failed (a provider
    outage, a malformed structured response, caught and logged here exactly like any
    other citation-link call), nothing about the sentence was ever actually looked at,
    so the entry is marked ``"unclassified": True`` -- the flag that stops
    `_finalize_paragraph_text` from removing it, keeping it exactly like a
    framing-tagged sentence and counting it under `_finalize_paragraph_text`'s own
    ``sentences_unclassified_kept`` statistic instead of destroying it on a transient
    failure. When the retry call answered but simply never named this particular
    sentence, the model did see it and chose not to report it as a link, a framing
    sentence, or a finding; the entry is then an ordinary, removable ``"finding"`` with
    no ``"unclassified"`` flag, so rule 1 of `_finalize_paragraph_text` removes it and
    counts it under ``sentences_removed_uncited_finding``, where a reader can see it,
    instead of it surviving indefinitely under a flag meant for a transient failure.

    A gap `unclassified_body_sentences` reports because it still carries a STALE
    ``"unclassified": True`` entry from an earlier pass's own failed retry has that
    stale entry REPLACED, never left in place beside whatever
    this call resolves: the old placeholder is dropped from *uncited_sentences* before
    anything new is added, so each heal gets exactly one fresh entry per gap, matching
    whatever this pass actually learns, not an ever-growing pile of superseded guesses.

    Returns ``(links, uncited_sentences, retry_provenance, calls_made, retry_failed)``:
    the first two are always fresh lists (copies of the ones given, merged with
    anything resolved), even when there was nothing to resolve; ``retry_provenance`` is
    the retry call's own provenance, or ``None`` when no call was made at all (no gap)
    or the call itself failed; ``calls_made`` is 0 or 1, for the caller to fold into the
    section's running ``total_calls`` exactly as every other citation-link call already
    is; ``retry_failed`` is ``True`` only when a call was made and it raised, so the
    caller can record it separately (``loop_stats["link_coverage_retry_failed"]``) --
    ``False`` both when no call was needed at all and when the call succeeded,
    whatever it did or did not classify."""
    gaps = unclassified_body_sentences(text, links, uncited_sentences)
    links_out = [dict(link) for link in links]
    uncited_out = [dict(entry) for entry in uncited_sentences]
    if not gaps:
        return links_out, uncited_out, None, 0, False

    gap_index_by_sentence = {
        _finalize_sentence_norm(gap["sentence"]): gap["paragraph_index"] for gap in gaps
    }
    # Drop any stale entry (typically a prior pass's own ``"unclassified": True``
    # placeholder) this pass is about to resolve fresh, so the replacement below never
    # lands beside it.
    uncited_out = [
        entry
        for entry in uncited_out
        if _finalize_sentence_norm(entry.get("sentence") or "") not in gap_index_by_sentence
    ]
    retry_text = "\n\n".join(gap["sentence"] for gap in gaps)
    retry_failed = False
    try:
        retry_result = await linker(retry_text)
    except Exception:
        logger.warning("Citation-link coverage retry call failed", exc_info=True)
        retry_result = None
        retry_failed = True

    retry_provenance = getattr(retry_result, "provenance", None) if retry_result else None
    resolved_norms: set[str] = set()
    for link in (getattr(retry_result, "links", None) or []):
        norm = _finalize_sentence_norm(link.get("sentence") or "")
        original_index = gap_index_by_sentence.get(norm)
        if original_index is None:
            continue  # V1 already guarantees this sentence is one we asked about.
        links_out.append({**link, "paragraph_index": original_index})
        resolved_norms.add(norm)
    for entry in (getattr(retry_result, "uncited_sentences", None) or []):
        norm = _finalize_sentence_norm(entry.get("sentence") or "")
        original_index = gap_index_by_sentence.get(norm)
        if original_index is None:
            continue
        uncited_out.append({**entry, "paragraph_index": original_index})
        resolved_norms.add(norm)

    for gap in gaps:
        norm = _finalize_sentence_norm(gap["sentence"])
        if norm not in resolved_norms:
            entry = {
                "paragraph_index": gap["paragraph_index"],
                "sentence": gap["sentence"],
                "tag": "finding",
            }
            if retry_failed:
                entry["unclassified"] = True
            uncited_out.append(entry)

    return links_out, uncited_out, retry_provenance, 1, retry_failed


def _strip_citation_substring(text: str, citation_text: str) -> str:
    """Remove one dropped citation's own rendered text from a sentence that otherwise
    survives (a sentence with two citations, one no_full_text and one verified, keeps
    the sentence but not the uncheckable citation's own text).

    A best-effort cleanup, not a grammar engine: the exact substring is removed, then a
    few of the most common leftover artefacts (a doubled space, a space before
    punctuation, empty parentheses, a dangling "and"/"&" the removal exposed) are tidied.
    An occasional slightly awkward remainder (e.g. a stray conjunction) is an accepted,
    documented limitation -- reconstructing perfect prose after a citation is surgically
    removed is out of this feature's scope."""
    if not citation_text:
        return text
    stripped = text.replace(citation_text, "")
    if stripped == text:
        return text
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"\(\s*\)", "", stripped)
    stripped = re.sub(r"\[\s*\]", "", stripped)
    stripped = re.sub(r"\s+(and|&)\s*([,.;:!?])", r"\2", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"(^|\s)(and|&)\s+(and|&)\s", r"\1\2 ", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _enclosing_mention(text: str, start: int, end: int) -> tuple[int, int] | None:
    """The bracketed citation mention of *text* that wholly contains ``text[start:end]``,
    brackets included, or None when that span is not inside one. "Zhang, 2022" inside
    "(Zhang, 2022; Lee, 2021)" is; "(Zhang, 2022)" on its own, and the year bracket of a
    narrative citation such as "Smith (2020)", are not."""
    for open_char, close_char in (("(", ")"), ("[", "]")):
        open_index = text.rfind(open_char, 0, start)
        if open_index == -1 or close_char in text[open_index:start]:
            continue
        close_index = text.find(close_char, end)
        if close_index == -1 or open_char in text[end:close_index]:
            continue
        return open_index, close_index + 1
    return None


def _drop_would_split_a_kept_citation(
    text: str, citation_text: str, kept_citation_texts: Sequence[str]
) -> bool:
    """True when cutting *citation_text* out of *text* would cut into a citation this
    same sentence keeps: either it is rendered inside one of *kept_citation_texts*, or it
    shares a bracketed mention with one of them.

    Both shapes are the same co-cited group, seen from two sides. "(Jones, 2019; Smith,
    2020)" with Jones uncheckable and Smith verified reaches `_finalize_paragraph_text`
    as one stored link naming both keys on the pass that narrows it, and as a kept link
    carrying the whole group's rendered text plus a separately re-resolved "Jones, 2019"
    on the pass after that; a group nothing ever linked reaches it as two re-resolved
    halves, "Jones, 2019" and "Smith, 2020", on every pass. Cutting the uncheckable half
    out of any of those leaves "(; Smith, 2020)" -- the mangled remainder of a mention
    the sentence still needs -- and does it again, a little further, on the next heal.
    Splitting one multi-key mention into a kept part and a dropped part is out of this
    feature's scope on every path (the limitation `verify_and_heal_claims` documents),
    so the mention is left rendered as written and only the link's own keys are
    narrowed."""
    normalised = _doc_normalize_ws(citation_text)
    if not normalised:
        return False
    kept_normalised = [_doc_normalize_ws(kept) for kept in kept_citation_texts if kept]
    if any(normalised in kept for kept in kept_normalised):
        return True
    start = text.find(normalised)
    while start != -1:
        mention = _enclosing_mention(text, start, start + len(normalised))
        if mention is not None:
            low, high = mention
            for kept in kept_normalised:
                kept_start = text.find(kept, low)
                if kept_start != -1 and kept_start + len(kept) <= high:
                    return True
        start = text.find(normalised, start + 1)
    return False


def _finalized_sentence_text(
    normalised_text: str, marker_count: int, stats: dict[str, int]
) -> str:
    """One surviving sentence exactly as `_finalize_paragraph_text` writes it into the
    document: *normalised_text* -- the fragment its caller already put through
    `_finalize_sentence_norm_counted`, possibly with a dropped citation's own text cut
    out of it since -- put back into that same normalised form, with the *marker_count*
    markers that one normalisation removed counted into *stats*.

    Applied to each sentence rather than to the finished paragraph so that a link
    refreshed against this sentence names the text the document actually ends up
    carrying. A link left naming the pre-strip or
    pre-collapse wording is stale the moment the paragraph is rebuilt, and the heal
    after it has to rewrite the paragraph's attrs to fix that, which is not the no-op a
    second heal has to be.

    The count is passed in rather than recomputed here because the fragment is
    normalised exactly once, before any removal rule is checked, and a marker in a
    sentence those rules then removed must not be counted at all: it never reaches this
    function, exactly as it never reached the joined paragraph before."""
    stats["needs_citation_markers_removed"] += marker_count
    return _doc_normalize_ws(normalised_text)


#: Mirrors ``demo/check_delivered.py``'s own ``_NUMERAL_PREFIX_CHARS`` (duplicated,
#: not imported -- see this function's own docstring below): every character a
#: prefix that is nothing but a bare number or percentage before a sentence's own
#: first word can carry ("88% ", "3.5%, ", "139 ").
_NUMERAL_PREFIX_CHARS = frozenset("0123456789.,% \t")

#: Mirrors ``demo/check_delivered.py``'s own ``_OPENING_QUOTE_CHARS``.
_OPENING_QUOTE_CHARS = "\"'‘“"

#: Mirrors ``demo/check_delivered.py``'s own ``_OPENING_BRACKET_CHARS``.
_OPENING_BRACKET_CHARS = "([‘“"


def _prefix_has_unmatched_closer(prefix: str) -> bool:
    """Mirrors ``demo/check_delivered.py``'s own function of the same name: True
    when *prefix* carries a ``)`` or ``]`` with no ``(`` or ``[`` earlier in the
    same prefix to balance it -- the tell of a fragment cut from the middle of a
    parenthetical, not a genuine number or percentage opening a fresh sentence."""
    paren_depth = 0
    bracket_depth = 0
    for char in prefix:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            if paren_depth > 0:
                paren_depth -= 1
            else:
                return True
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            if bracket_depth > 0:
                bracket_depth -= 1
            else:
                return True
    return False


def _prefix_is_legitimate_opening(prefix: str) -> bool:
    """Mirrors ``demo/check_delivered.py``'s own function of the same name: True
    when non-empty *prefix* -- already cleared of a stray unmatched closing
    bracket -- is a bare number or percentage, an opening parenthesis or bracket,
    or a quotation mark."""
    if all(char in _NUMERAL_PREFIX_CHARS for char in prefix):
        return True
    first_visible = prefix.lstrip()[:1]
    return first_visible in _OPENING_BRACKET_CHARS or first_visible in _OPENING_QUOTE_CHARS


#: A finite verb that reads as complete on its own everywhere else in a sentence, but
#: not sentence-final with nothing after it: English licenses a bare "remains" only
#: with a nominal subject ("the question remains"), not a clausal one ("whether X
#: remains" needs a complement, e.g. "remains an open question" or "remains unclear").
#: The same holds for the rest of this list. Checked only where the writer's own
#: ``[NEEDS CITATION]`` marker was stripped from exactly this position (see
#: ``marker_stripped`` on `_fragment_truncation_reason` below): the marker sat where
#: the complement belonged, so its removal is what turned a sentence still being
#: drafted into a fragment that reads as finished. Not mirrored in
#: `demo/check_delivered.py` rule 6 -- that checker has no access to which sentences
#: were marker-stripped and stays frozen -- so this signal is backend-only, never
#: compared against the checker's own `truncation_reasons`.
_MARKER_STRIPPED_BARE_VERBS = frozenset({
    "remains", "remain", "remained",
    "is", "are", "was", "were",
    "seems", "seem", "seemed",
    "appears", "appear", "appeared",
    "becomes", "become", "became",
    "suggests", "suggest",
    "indicates", "indicate",
    "shows", "show", "showed",
    "found", "finds",
    "reported", "reports",
    "revealed", "reveals",
    "demonstrated", "demonstrates",
    "requires", "require",
    "warrants", "warrant",
    "deserves", "deserve",
    "merits", "merit",
    "lacks", "lack",
    "involves", "involve",
    "includes", "include",
})


#: The first two structural signals `demo/check_delivered.py` rule 6
#: (`truncation_reasons`) checks a delivered sentence for, duplicated here rather
#: than imported -- `demo/` runs under the system Python, without this backend's own
#: dependencies, exactly as every other constant that script mirrors from this module
#: is duplicated, not imported (see that script's own module docstring). A backend
#: test (``backend/tests/test_finalize_truncated_fragment_gate.py``) loads that script
#: by file path and asserts this function agrees with it, sentence for sentence, over
#: one shared fixture list, so the two copies cannot silently drift apart. The third
#: signal below (``marker_stripped``) is not part of that agreement: it has no
#: counterpart in the checker, by design (see `_MARKER_STRIPPED_BARE_VERBS` above).
def _fragment_truncation_reason(sentence: str, *, marker_stripped: bool = False) -> str | None:
    """Why *sentence* reads like a fragment cut from a longer sentence -- an ending
    that is really an abbreviation (`_ABBREVIATIONS_NOT_SENTENCE_FINAL`), an opening
    letter that is lower-case, as the second half of a wrongly split sentence would
    be, or, when *marker_stripped* is true, a bare finite verb
    (`_MARKER_STRIPPED_BARE_VERBS`) sitting immediately before the final punctuation
    with nothing after it -- or ``None`` when no signal fires. Used only on a
    sentence about to be kept with no citation of its own
    (`_finalize_paragraph_text`'s ``not sentence_links`` branch): the shapes this
    catches are both the writer's own ``[NEEDS CITATION]`` marker stripped from in
    front of a dependent clause it introduced, leaving that clause to stand as if it
    were its own sentence (the head case, the first two signals), and the marker
    stripped from the position of a missing predicate complement, leaving a
    capital-initial, period-terminated sentence with no complement (the tail case,
    the third signal).

    *marker_stripped* must be true only when the caller's own strip of this exact
    sentence actually removed a marker (`_finalize_sentence_norm_counted`'s own
    ``marker_count`` for this fragment being greater than zero) -- never merely
    because the sentence appears in an uncited list under some other tag. A sentence
    never marker-stripped is never touched by the third signal, whatever it ends
    with.

    The lower-case-start signal is judged on the sentence's own first LETTER, not
    its first character, and is suppressed when what precedes that letter is
    itself a legitimate opening a real sentence can have and nothing else -- a bare
    digit or percentage, a parenthesised citation, or a quotation mark
    (`_prefix_is_legitimate_opening`), UNLESS that same prefix also carries a
    stray, unmatched closing bracket (`_prefix_has_unmatched_closer`), which is
    exactly the tell of a fragment cut from the middle of a parenthetical rather
    than a genuine number or percentage opening a fresh sentence."""
    stripped = sentence.strip()
    if not stripped:
        return None
    tail = re.search(r"([A-Za-z]+)\.\s*$", stripped)
    if tail is not None:
        tail_word = tail.group(1).lower()
        if tail_word in _ABBREVIATIONS_NOT_SENTENCE_FINAL:
            return f"ends with {tail.group(0).strip()!r}, an abbreviation, not a real sentence end"
        if marker_stripped and tail_word in _MARKER_STRIPPED_BARE_VERBS:
            return (
                f"a stripped [NEEDS CITATION] marker leaves {tail.group(0).strip()!r} "
                "as a bare verb with no complement"
            )
    first_letter = re.search(r"[A-Za-z]", stripped)
    if first_letter is None or not first_letter.group(0).islower():
        return None
    prefix = stripped[: first_letter.start()]
    if (
        prefix
        and not _prefix_has_unmatched_closer(prefix)
        and _prefix_is_legitimate_opening(prefix)
    ):
        return None
    return "begins with a lower-case letter, as a fragment cut from a previous sentence would"


def _merge_continuation_fragment(previous_text: str, fragment_text: str) -> str:
    """*fragment_text* -- a sentence `_fragment_truncation_reason` has already
    flagged as truncated, kept because it carries no citation of its own -- joined onto
    the end of *previous_text* as the continuation clause it grammatically is, rather
    than delivered as a second, broken sentence. *previous_text*'s own terminal
    punctuation is swapped for a comma so the two read as one sentence; a terminal
    period is added back only when *fragment_text* did not already supply its own
    sentence-ending punctuation."""
    previous = previous_text.rstrip()
    if previous and previous[-1] in _SENTENCE_TERMINAL_PUNCTUATION:
        previous = previous[:-1]
    fragment = fragment_text.strip()
    merged = f"{previous}, {fragment}" if previous else fragment
    if not merged.endswith(_SENTENCE_TERMINAL_PUNCTUATION):
        merged += "."
    return merged


def _coverage_incomplete_reason(
    sentence: str,
    bad_links: Sequence[Mapping[str, Any]],
    non_frame_residues: Sequence[str],
    has_meta_evaluation: bool,
) -> dict[str, Any]:
    """Why one ``sentences_removed_coverage_incomplete`` fired, for
    ``stats["coverage_incomplete_reasons"]`` -- every value here is read off
    data `_finalize_paragraph_text` already computed for *sentence*; nothing new is
    looked up. A sentence can carry more than one reason at once (an unverified
    proposition alongside a residue), so ``reasons`` is always a list, never a bare
    string."""
    reasons: list[str] = []
    for link in bad_links:
        proposition = link.get("proposition") or ""
        keys = ", ".join(link.get("keys") or []) or "(no key)"
        reasons.append(f"proposition {proposition!r} against {keys} carried no verified verdict")
    for residue in non_frame_residues:
        reasons.append(f"unresolved residue {residue!r}")
    if has_meta_evaluation:
        reasons.append("a comparative meta-evaluation of the literature")
    return {"sentence": sentence, "reasons": reasons}


def _finalize_paragraph_text(
    paragraph_text: str,
    links: Sequence[Mapping[str, Any]],
    uncited: Sequence[Mapping[str, Any]],
    claim_status: Mapping[tuple[str, str, str], str],
    claim_evidence_quotes: Mapping[tuple[str, str, str], Sequence[str]] | None = None,
) -> tuple[str, list[dict], dict[str, int], dict[str, str], list[dict]]:
    """Apply the finalize removal rules (design section 6 amendments A3/A4/A5) to one
    paragraph's own text.

    Every fragment is put into finalize's one normalised sentence form
    (`_finalize_sentence_norm_counted`: every ``[NEEDS CITATION]`` marker stripped,
    every whitespace run collapsed to one space, trimmed) ONCE, before any of the rules
    below is checked, and every key those rules compare it against is built with that
    same normaliser: the per-sentence link map from *links*, the uncited "finding" set
    from *uncited*, and the status lookup from *claim_status*. No rule can therefore
    see a marker, or a whitespace difference, that another rule does not -- which
    otherwise could make a sentence match rule 1's set and none of the links rule 2
    needs, or the other way round, depending only on which side of which comparison had
    been stripped. The surviving text is that same normalised form, so the next heal
    normalises it to itself.

    Rules, in the order they are checked for each sentence fragment
    (`_sentence_fragments`, the same splitter `extract_claims_from_document` uses):

    1. A sentence carrying no citation link at all is removed only when it is listed in
       *uncited* tagged ``"finding"`` AND the retry that classified it actually
       committed to that verdict -- not carrying its own ``"unclassified": True`` flag
       (design amendment A4); a
       ``"framing"``-tagged, altogether unlisted, or still-``"unclassified"`` uncited
       sentence is kept. A gap `resolve_unclassified_sentences` or
       `_close_citation_link_coverage_gap` could not classify -- because its own bounded
       retry call itself failed (a provider outage), or because the retry succeeded but
       never mentioned that particular sentence at all -- is recorded, exactly as
       `_build_section_tiptap_nodes` already writes an unresolved link-map failure into
       ``link_map_failed`` rather than treating an empty result as "nothing to verify,
       nothing to remove": without this distinction, every such gap would be removed
       outright, so a transient linker failure during the user-facing "verify and heal"
       action would delete most of a real user's saved section. Removal is reserved for
       a gap the retry actually, positively classified as an uncited finding.
    2. A link whose every key is ``"no_full_text"`` is dropped -- its own citation_text
       is cut from the surviving sentence text (`_strip_citation_substring`) below,
       once every link has been classified. A link with no verified key at all (its
       proposition's own verdict is unsupported, needs_nuance, error, or a
       (sentence, proposition, key) triple with no verdict at all) is set aside as
       ``bad_links`` rather than immediately removing the whole
       sentence: outcome depends on whether what remains still covers the sentence
       (edit 6, below).
    3. Otherwise, each of the sentence's own citation links is kept when at least one of
       its keys is ``"verified"``, narrowed to only its verified keys for a mixed
       co-citation group.

    3a. Once rules 2 and 3 have decided which links survive:
        the surviving (kept) propositions and citations must still cover the sentence
        (``app.services.sentence_coverage.sentence_residue_violations``) and carry no
        comparative meta-evaluation of the literature (``has_meta_evaluation``, edit 8);
        a `bad_links` entry's own, unverified proposition is never treated as covering
        text either way. When they do not, a single, deterministic trailing excision
        (``trailing_excision_cut``) is tried -- cutting the sentence at the clause
        boundary in front of the one residue that runs to its own end, keeping every
        verified proposition intact and closing the sentence with its own terminal
        punctuation -- and the whole sentence is removed when no such safe cut exists
        (a leading or mid-sentence residue, more than one residue span, or an
        unverified proposition). A sentence with an otherwise-verified finding whose
        only problem is a leading attribution lead-in (blocked by a numeral or an
        out-of-lexicon noun) is removed this way too, not rewritten: this module's
        own guarantee is to deliver only what is verified, or omit the sentence, and
        a deterministic rewrite that manufactures a subject is not part of that
        guarantee. The revision pass in ``app.services.writing.generate_section`` is
        the first line of defence for this same invariant, run once before finalize
        ever sees the text; this is the deterministic backstop for whatever it does
        not fix.

    4. Once every rule above has decided what survives, one more, unconditional
       check runs over the exact text about to be delivered
       (``sentence_coverage.opens_with_reporting_verb``): a sentence whose own
       leading residue carried zero content tokens already passes rule 3a's own
       coverage check with no issue at all, regardless of whether a grammatical
       subject actually precedes the verb -- delivered with no subject at all when
       the sentence's own citation trails at the end instead. Caught and removed
       whole here, independent of which rule produced the text.

    ``[NEEDS CITATION]`` markers are stripped from every surviving sentence
    unconditionally, whether or not that sentence carried a citation, and a run of
    spaces inside it is collapsed to one -- the text a paragraph's own marks leave
    behind (`_finalized_sentence_text`). A marker in a sentence one of the rules
    removed is not counted, since it is not in the text this function returns.

    Accepted limitation, one paragraph repeating a sentence: the per-sentence link map
    is keyed on the normalised sentence, so two fragments of the same paragraph that
    normalise to the same string -- two identical sentences, or two that become
    identical once their ``[NEEDS CITATION]`` markers are stripped -- share one link
    list, and the links, the verdicts and the uncited entries recorded for either
    occurrence apply to both. Nothing a linker returns says which occurrence it meant:
    an uncited entry and a citation link each name a sentence by its text alone, and a
    claim's verdict is keyed on that text too, so there is no position to key on
    instead. Sharing is also what keeps the exit invariant on such a paragraph -- an
    unsupported citation removes every repeat of its sentence, not only the first --
    and it happens on the first pass rather than the pass after the marker between the
    two occurrences is stripped. The shared links
    are RETURNED once for the repeated sentence rather than once per occurrence, so
    what this function returns for it is what it returns again on the next pass instead
    of doubling on every heal.

    What that limitation still costs, on the saved-draft path only: a paragraph that
    stored a SEPARATE link for each occurrence (two links naming the same sentence once
    the marker is gone) keeps both on the pass that strips the marker and one of them
    on the pass after, because `_claims_for_paragraph_text` then re-derives one claim
    where the marker used to make two, and one stored link owns it. The text and the
    final report are stable from the first pass either way; what moves is one
    paragraph's attrs, one pass later.

    Rule 3 never cuts a citation rendered INSIDE a citation the same sentence keeps:
    a co-cited group such as "(Jones, 2019; Smith, 2020)" whose two keys reach this
    function as two separate links -- which is exactly what the pass AFTER a heal sees,
    since the surviving group link is narrowed to its verified keys and extraction then
    resolves the dropped key on its own -- would otherwise have "Jones, 2019" cut out of
    the kept link's own rendered text, leaving "(; Smith, 2020)". That is the same split
    the mixed-status branch below already refuses to make on the pass that narrows the
    group, so refusing it here too is what makes a second heal a no-op instead of a
    slow mangling of the same sentence, one heal at a time.

    Returns ``(rebuilt_text, surviving_links, stats, healed_sentences,
    surviving_uncited)``. ``surviving_links`` is a list of the same link dicts *links*
    was given (their own ``keys`` narrowed when a no_full_text key was dropped from a
    mixed-status link, and their own ``sentence`` always set to the sentence as this
    function writes it -- with ``proposition`` dropped if a rewrite left it no longer
    occurring in that text), for the caller to
    fold back into the document. ``surviving_uncited`` is the subset of *uncited* whose
    own sentence this pass kept (a ``"framing"``-tagged, still-``"unclassified"``, or
    altogether unlisted-by-*uncited* sentence never removes anything, so every *uncited*
    entry naming it survives too), with ``sentence`` refreshed to the text this function
    actually wrote for it, exactly as a surviving link's own ``sentence`` is refreshed:
    the caller persists this back onto the document
    (`_build_section_tiptap_nodes`, `_rebuild_finalized_node_list`) so a sentence this
    pass already classified is never treated as an unclassified coverage gap again on a
    later pass. ``healed_sentences`` maps each sentence that changed FROM the wording
    its own claims were verified against -- the wording *claim_status* is keyed on -- TO
    the text that replaced it, so a caller pairing a verdict back to a surviving link
    (`surviving_verifications`) can still find a link whose ``sentence`` no longer
    equals what the claim was verified against. The map is returned rather than stored
    on the link itself precisely so that it cannot outlive this pass: a pairing key
    written into a link is written into the document and into the job result with it,
    where the next heal reads it as if it still described that pass. ``stats`` counts
    ``sentences_removed_unverified``,
    ``sentences_removed_uncited_finding`` (an uncited "finding" entry the retry actually,
    positively classified), ``sentences_unclassified_kept``
    (an uncited-"finding" entry the caller marked ``"unclassified": True`` --
    `resolve_unclassified_sentences`'s own synthetic entry for a gap not even a
    bounded retry call could classify, or one left over because the retry call itself
    failed -- kept, not removed, and counted separately so a reader can tell "this
    sentence was never actually classified by any model call" apart from an ordinary,
    linker-verified sentence), ``sentences_removed_no_full_text``,
    ``citations_dropped_no_full_text`` (a citation whose own rendered text this pass cut
    out of a surviving sentence, or that went with a sentence removed whole because
    every citation on it was uncheckable), ``needs_citation_markers_removed``,
    ``sentences_residue_excised`` (a trailing residue cut away, the sentence otherwise
    kept), ``sentences_removed_coverage_incomplete`` (no safe cut existed, so the whole
    sentence was removed for a residue or a comparative meta-evaluation rather than a
    plain unverified verdict), ``frame_spans_accepted`` (the number of residue spans
    the frame classifier accepted on a sentence that survived, so what it permits is
    reported rather than left implicit) and ``sentences_removed_heading_shaped`` (an
    uncited fragment that reads as a heading rather than a sentence,
    ``sentence_coverage.is_heading_shaped_paragraph`` -- the second, independent
    guard against the same defect the leading-heading handling above targets)."""
    healed_sentences: dict[str, str] = {}
    stats = {
        "sentences_removed_unverified": 0,
        "sentences_removed_uncited_finding": 0,
        "sentences_unclassified_kept": 0,
        "sentences_removed_no_full_text": 0,
        "citations_dropped_no_full_text": 0,
        "needs_citation_markers_removed": 0,
        # An uncited framing sentence dropped by the deterministic coherence pass
        # below, over and above the three ordinary removal rules -- see
        # `_drop_dangling_framing_sentences`.
        "sentences_removed_dangling": 0,
        # A sentence whose fully-verified propositions leave a trailing residue (or a
        # comparative meta-evaluation) safely excised rather than removed whole; a
        # sentence removed whole for that same reason (leading or mid-sentence residue,
        # or no safe excision boundary); and the number of residue spans the frame
        # classifier accepted as one of F1-F3 rather than flagging, across every
        # sentence this pass actually finalized (so what the classifier permits is
        # reported, not left implicit).
        "sentences_residue_excised": 0,
        "sentences_removed_coverage_incomplete": 0,
        "frame_spans_accepted": 0,
        # A sentence kept with no citation of its own that
        # `_fragment_truncation_reason` flagged as truncated. A marker-headed
        # fragment (marker stripped from its own front, leaving a lower-case-opening
        # clause with nothing of its own to merge onto) is dropped outright and named
        # in `truncated_fragment_reasons` below -- it is not a continuation of the
        # sentence before it, and merging it there produced a broken run-on.
        # Every other shape (an abbreviation-ending or a
        # marker-stripped bare-verb-ending fragment) is still merged into the
        # preceding sentence as its own continuation clause when one exists in this
        # paragraph, or dropped outright when there is none to merge into. Either way
        # counted here, never delivered as its own broken sentence.
        "sentences_truncated_fragment_repaired": 0,
        # One entry per marker-headed fragment dropped above, naming the sentence
        # that was dropped and the reason `_fragment_truncation_reason` gave for it.
        # Popped the same way as `coverage_incomplete_reasons`, for the same reason.
        "truncated_fragment_reasons": [],
        # An uncited sentence (whatever tag it carried) about to be kept that
        # `sentence_coverage.has_meta_evaluation` flags as a comparative
        # meta-evaluation of the literature -- dropped instead of delivered, since
        # it has no proposition of its own for the cited-sentence gate below to
        # have already tested. Also counts a CITED sentence dropped by the same
        # guard run over its own final, whole delivered text after the coverage
        # decision: the proposition-only check that gate already runs cannot see a
        # comparative folded into an accepted frame span (a colon lead-in, for
        # instance), so the whole text is checked again, once, right before
        # delivery. Kept as one counter for both shapes -- the guard and its harm
        # are the same regardless of which gate found it -- with
        # `meta_evaluation_reasons` below recording which shape and which matched
        # span for each occurrence.
        "sentences_removed_meta_evaluation_uncited": 0,
        # A sentence kept this far whose own delivered text still opens with a
        # past-tense or third-person reporting verb followed by "that"
        # (`sentence_coverage.opens_with_reporting_verb`) -- a subject-less
        # sentence the frame classifier's own "zero content residue is always a
        # permitted frame" branch does not, by itself, rule out. Removed whole,
        # never rewritten: the pipeline delivers only what is verified, or omits
        # the sentence, and manufacturing a subject is not part of that guarantee.
        "sentences_removed_verb_initial": 0,
        # One entry per `sentences_removed_coverage_incomplete` above, naming which
        # proposition lacked a verdict or which residue caused it. Never summed as
        # an int like every other key here: both of this function's callers
        # (`finalize_generated_section`, `_rebuild_finalized_node_list`) pop it out
        # of `stats` before their own ordinary per-key integer aggregation, so it is
        # carried (write stage) or dropped (the standalone verify-and-heal action,
        # which has no use for it) by the caller, never miscounted as though it
        # were a number.
        "coverage_incomplete_reasons": [],
        # One entry per `sentences_removed_meta_evaluation_uncited` above, naming the
        # sentence that was dropped and the exact span `sentence_coverage.meta_
        # evaluation_match` matched on it. Popped the same way as `coverage_
        # incomplete_reasons`, for the same reason.
        "meta_evaluation_reasons": [],
        # One entry per `sentences_removed_verb_initial` above, naming the sentence
        # before finalize touched it and the exact text that was about to be
        # delivered when the guard caught it. Popped the same way as `coverage_
        # incomplete_reasons`, for the same reason.
        "verb_initial_reasons": [],
        # An uncited fragment `sentence_coverage.is_heading_shaped_paragraph` flags
        # as reading like a heading rather than a sentence -- the second,
        # independent guard against a heading-shaped fragment reaching the delivered
        # draft as a body paragraph, whatever path left it here.
        "sentences_removed_heading_shaped": 0,
        # One entry per `sentences_removed_heading_shaped` above, naming the
        # sentence. Popped the same way as `coverage_incomplete_reasons`, for the
        # same reason.
        "heading_shaped_reasons": [],
    }
    links_by_sentence: dict[str, list[dict]] = {}
    for link in links:
        links_by_sentence.setdefault(
            _finalize_sentence_norm(link.get("sentence") or ""), []
        ).append(dict(link))
    uncited_by_sentence: dict[str, list[dict]] = {}
    for entry in uncited:
        uncited_by_sentence.setdefault(
            _finalize_sentence_norm(entry.get("sentence") or ""), []
        ).append(dict(entry))
    # Only an uncited "finding" entry the retry actually, positively classified --
    # carrying no ``"unclassified": True`` flag -- is removable. An entry
    # `resolve_unclassified_sentences` or `_close_citation_link_coverage_gap` marked
    # ``unclassified`` (because its own bounded retry call itself failed, or succeeded
    # without ever mentioning this sentence) is never removed: it is kept, exactly
    # like a framing-tagged sentence, and counted under `sentences_unclassified_kept`
    # instead.
    uncited_finding_sentences = {
        _finalize_sentence_norm(u.get("sentence") or "")
        for u in uncited
        if u.get("tag") == "finding" and not u.get("unclassified")
    }
    unclassified_kept_sentences = {
        _finalize_sentence_norm(u.get("sentence") or "")
        for u in uncited
        if u.get("unclassified")
    }
    # The third key built with the same normaliser, now including the claim's own
    # proposition: the verdict is keyed on (sentence, proposition, key), not
    # (sentence, key) alone, so two propositions against the same key on the same
    # sentence -- the escape mechanism edit 4 closes upstream --
    # each keep their own verdict here too, rather than one silently standing in for
    # the other. Two claims whose sentences/propositions differ only in whitespace or
    # in a marker collide here; the more severe verdict wins
    # (`_finalize_status_severity`), so a collision can only remove text, never ship it.
    status_by_sentence: dict[tuple[str, str, str], str] = {}
    for (status_sentence, status_proposition, status_key), status in claim_status.items():
        pair = (
            _finalize_sentence_norm(status_sentence),
            _finalize_sentence_norm(status_proposition),
            status_key,
        )
        held = status_by_sentence.get(pair)
        if held is None or _finalize_status_severity(status) > _finalize_status_severity(held):
            status_by_sentence[pair] = status

    # The same normalised (sentence, proposition, key) key as `status_by_sentence`,
    # but for the claim's own verified evidence quote(s) rather than its status --
    # see `_finalize_paragraph_text`'s own docstring, item 3a, and
    # `sentence_coverage.covered_numeral_keys`.
    evidence_by_sentence: dict[tuple[str, str, str], list[str]] = {}
    for (evidence_sentence, evidence_proposition, evidence_key), quotes in (
        claim_evidence_quotes or {}
    ).items():
        pair = (
            _finalize_sentence_norm(evidence_sentence),
            _finalize_sentence_norm(evidence_proposition),
            evidence_key,
        )
        evidence_by_sentence.setdefault(pair, []).extend(q for q in quotes if q)

    # One record per surviving fragment, in paragraph order, for the deterministic
    # coherence pass to run over once the ordinary per-sentence rules below are done
    # with it. ``preceded_by_removal`` is this flag's own value at
    # the moment the fragment survived -- true the instant one of the three ordinary
    # removal rules has fired anywhere earlier in this same paragraph's original
    # fragment order, and never reset, so a fragment kept after such a removal always
    # carries the flag even when several more fragments have survived since.
    fragment_records: list[dict[str, Any]] = []
    any_removed_so_far = False
    surviving_links: list[dict] = []
    # Normalised sentences whose surviving links have already been returned, so that a
    # paragraph repeating a sentence returns their one shared link list once rather
    # than once per occurrence (the duplicate-sentence limitation documented above).
    linked_sentences_returned: set[str] = set()
    for start, end in _sentence_fragments(paragraph_text):
        fragment = paragraph_text[start:end]
        fragment_stripped = fragment.strip()
        if not fragment_stripped:
            continue
        # The one normalisation of this fragment. Every rule below compares against
        # `fragment_norm`; nothing below derives a second form of the same sentence.
        # A fragment that is nothing but a marker normalises to the empty string: it
        # matches no key (the lookups below are guarded), its marker is counted, and
        # the empty text it leaves is dropped when the paragraph is joined.
        fragment_norm, marker_count = _finalize_sentence_norm_counted(fragment_stripped)
        sentence_links = links_by_sentence.get(fragment_norm, []) if fragment_norm else []

        if not sentence_links:
            if fragment_norm and fragment_norm in uncited_finding_sentences:
                stats["sentences_removed_uncited_finding"] += 1
                any_removed_so_far = True
                continue
            kept_text = _finalized_sentence_text(fragment_norm, marker_count, stats)
            # A sentence about to be kept with no citation of its own -- whether the
            # retry marked it "unclassified", a later heal re-tagged it "framing", or
            # it was never listed in *uncited* at
            # all -- must never be delivered as a fragment. Checked here, on every path
            # into this branch, so the fix holds regardless of which upstream tag
            # produced the sentence (item (b): the fallback's own kept sentence; item
            # (c): a later heal's "framing" re-tag can no longer hide one from this
            # check the way it hid one from rule 1).
            # Narrowed to a sentence the citation-link/coverage-gap machinery actually
            # touched -- one whose own ``[NEEDS CITATION]`` marker this pass just
            # stripped (``marker_count``), or one *uncited* names at all (whatever tag
            # it carries: "unclassified", "framing", or an ordinary "finding" this
            # branch is about to keep because it is not in ``uncited_finding_
            # sentences``). Plain body prose with no citation that was never listed
            # anywhere -- text this pipeline's own citation-link call was never even
            # asked about, the shape a citation-link failure leaves behind -- is a
            # pre-existing, accepted case `_finalize_paragraph_text` has always kept
            # verbatim; this gate must not start rewriting it just because it happens
            # to open lower-case.
            gate_applies = marker_count > 0 or (
                fragment_norm and fragment_norm in uncited_by_sentence
            )
            truncation_reason = (
                _fragment_truncation_reason(kept_text, marker_stripped=marker_count > 0)
                if kept_text and gate_applies
                else None
            )
            if truncation_reason:
                stats["sentences_truncated_fragment_repaired"] += 1
                # A marker-headed fragment -- the writer's own [NEEDS CITATION]
                # marker sat in front of this fragment's own dependent clause (this
                # pass's own strip, ``marker_count > 0``, or an earlier pass's,
                # which is why this fragment is in *uncited* at all -- ``gate_
                # applies``, already required above for `truncation_reason` to be
                # anything but ``None``, covers both), and stripping it left that
                # clause opening in lower case -- is never merged into the sentence
                # before it: it is not that sentence's own continuation, only a
                # caveat the writer tacked on as its own (uncited) thought, and
                # appending it produces a broken run-on ("... outperformed Grammarly
                # in spotting direct translation, word choice, sentence structure,
                # and pronoun errors, for whether these detection advantages
                # translate into improved writing accuracy."). Dropped outright
                # instead, whether or not a preceding sentence exists to merge into,
                # and named in `truncated_fragment_reasons`. The tail case (an
                # abbreviation ending, or a marker-stripped bare verb) is not this
                # shape -- it never begins lower case -- so it is unaffected and
                # still merges, below, exactly as before.
                marker_headed = truncation_reason.startswith(
                    "begins with a lower-case letter"
                )
                if marker_headed:
                    stats["truncated_fragment_reasons"].append(
                        {"sentence": kept_text, "reason": truncation_reason}
                    )
                elif fragment_records:
                    previous = fragment_records[-1]
                    old_text = previous["text"]
                    merged_text = _merge_continuation_fragment(old_text, kept_text)
                    previous["text"] = merged_text
                    if previous["has_citation"] and old_text != merged_text:
                        healed_sentences[old_text] = merged_text
                        for link in surviving_links:
                            if link.get("sentence") == old_text:
                                link["sentence"] = merged_text
                # else: no preceding sentence in this paragraph to merge into -- item
                # (a)'s "otherwise removed", already achieved by not appending anything
                # for this fragment below.
                any_removed_so_far = True
                continue
            # `has_meta_evaluation` below is
            # the same predicate the cited-sentence path already runs over
            # `kept_propositions`; an UNCITED sentence has no proposition of
            # its own to test, so a comparative meta-evaluation written as a
            # paragraph-opening framing sentence -- "The strongest comparative
            # evidence concerns feedback explicitness." -- passed the write-time gate
            # untouched, whatever tag it carried ("framing", "unclassified", or no
            # entry at all -- the same three shapes item (c) above already refuses to
            # let hide a truncated fragment). Checked here, on every path that is
            # about to keep an uncited sentence, so the fix holds regardless of tag.
            meta_match = sentence_coverage.meta_evaluation_match(kept_text) if kept_text else None
            if meta_match:
                stats["sentences_removed_meta_evaluation_uncited"] += 1
                stats["meta_evaluation_reasons"].append({
                    "sentence": kept_text,
                    "matched": meta_match,
                })
                any_removed_so_far = True
                continue
            # A second, independent guard against the same defect the leading-
            # heading handling above targets: an uncited fragment that reads as a
            # heading rather than a sentence (`sentence_coverage.is_heading_shaped_
            # paragraph`) -- no sentence-final punctuation, at most ten words, no
            # finite verb, no citation -- whatever path left it here, not only the
            # one the leading-heading fix already closes. That fix keeps a genuine
            # markdown heading marked as a heading node instead of ever demoting it
            # to plain text, so this check is what still catches a heading-shaped
            # fragment the model wrote with no markdown marker at all, or one some
            # future change reintroduces. Removed whole, never rewritten, the same
            # as every other guard in this function.
            if kept_text and sentence_coverage.is_heading_shaped_paragraph(kept_text):
                stats["sentences_removed_heading_shaped"] += 1
                stats["heading_shaped_reasons"].append({"sentence": kept_text})
                any_removed_so_far = True
                continue
            uncited_entries = []
            if fragment_norm:
                if fragment_norm in unclassified_kept_sentences:
                    stats["sentences_unclassified_kept"] += 1
                uncited_entries = uncited_by_sentence.get(fragment_norm, [])
            fragment_records.append({
                "text": kept_text,
                "has_citation": False,
                "preceded_by_removal": any_removed_so_far,
                "uncited_entries": uncited_entries,
            })
            continue

        kept_links_for_sentence: list[dict] = []
        # A link whose every key comes back a genuinely bad verdict (neither verified
        # nor uncheckable) does not abort this sentence on sight -- every link is
        # classified first, so a sentence with one verified proposition and one that
        # failed can still be rescued by excision below, instead of the whole sentence
        # being thrown away for the sake of one clause.
        bad_links: list[dict] = []
        dropped_citation_texts: list[str] = []
        # Every verified evidence quote behind a link this sentence keeps, keyed by
        # nothing (a flat, deduplicated-by-membership list): the coverage check
        # below narrows a residue's own numeral disqualification against this list
        # (`sentence_coverage.covered_numeral_keys`), never against a bad or
        # dropped link's own quote.
        kept_evidence_quotes: list[str] = []
        for link in sentence_links:
            keys = link.get("keys") or []
            proposition_norm = _finalize_sentence_norm(link.get("proposition") or fragment_stripped)
            # `fragment_norm` and the link's own proposition: this link was found under
            # that key, and the status map is built with the same normaliser, so a
            # verdict recorded against any wording of this sentence AND this exact
            # proposition is looked up under exactly the triple that matched the link
            # (the proposition is part of the key so a second proposition against the
            # same key never inherits the first one's verdict).
            statuses = [
                status_by_sentence.get((fragment_norm, proposition_norm, key)) for key in keys
            ]
            if all(s in _FINALIZE_UNCHECKABLE_STATUSES for s in statuses):
                # Classified now and cut below: which citations this sentence keeps is
                # not known until every link has been classified, and a dropped
                # citation rendered inside a kept one must not be cut out of it.
                dropped_citation_texts.append(link.get("citation_text") or "")
                continue
            verified_keys = [k for k, s in zip(keys, statuses) if s == "verified"]
            non_verified_statuses = [s for s in statuses if s != "verified"]
            if not verified_keys:
                # This whole link -- every key -- is neither verified nor uncheckable:
                # its own proposition never earned a place in the delivered sentence.
                bad_links.append(link)
                continue
            if len(verified_keys) < len(keys) and all(
                s in _FINALIZE_UNCHECKABLE_STATUSES for s in non_verified_statuses
            ):
                # A mixed co-citation group (e.g. "(Jones, 2019; Lee, 2021)" with one
                # key verified and one no_full_text): the link survives, narrowed to
                # only its verified keys, so a downstream reader of `keys` never sees a
                # key this sentence's own status disagrees with. `citation_text` itself
                # is left rendered as written -- splitting one multi-key citation's own
                # text is out of this feature's scope, a documented limitation.
                kept_links_for_sentence.append({**link, "keys": verified_keys})
                for key in verified_keys:
                    kept_evidence_quotes.extend(
                        evidence_by_sentence.get((fragment_norm, proposition_norm, key), [])
                    )
            elif len(verified_keys) < len(keys):
                # A mixed co-citation group with a genuinely bad key (unsupported,
                # needs_nuance or error, not merely uncheckable) mixed in: narrowing
                # here would still leave that key's own paper name rendered inside a
                # group whose text is never split (the same limitation the branch
                # above accepts for a no_full_text mix), so the whole link is treated
                # as unsafe rather than narrowed -- the sentence is removed whole
                # unless another, wholly clean citation on it can carry it (the
                # `bad_links` handling below).
                bad_links.append(link)
            else:
                kept_links_for_sentence.append(link)
                for key in keys:
                    kept_evidence_quotes.extend(
                        evidence_by_sentence.get((fragment_norm, proposition_norm, key), [])
                    )

        if not kept_links_for_sentence:
            if dropped_citation_texts and not bad_links:
                stats["citations_dropped_no_full_text"] += len(dropped_citation_texts)
                stats["sentences_removed_no_full_text"] += 1
            else:
                stats["sentences_removed_unverified"] += 1
            any_removed_so_far = True
            continue

        kept_citation_texts = [
            kept.get("citation_text") or "" for kept in kept_links_for_sentence
        ]

        # A `bad_link` whose own mention shares a bracket with a kept citation (a
        # co-citation group, e.g. "(Park, 2018; Mao, 2024)" with Park unsupported and
        # Mao verified) can never be safely excised: the group's own text is never
        # split (the same documented limitation the mixed-status branch above already
        # accepts), so the kept link's own `citation_text` still renders the
        # unsupported paper's name, and the coverage check below -- which treats that
        # same group text as a covering span -- would otherwise judge the sentence
        # clean while it still visibly names an unsupported source. The whole sentence
        # is removed instead, exactly as a `no_full_text` drop already refuses to
        # split such a group's own rendered text.
        if any(
            _drop_would_split_a_kept_citation(
                fragment_norm, bad.get("citation_text") or "", kept_citation_texts
            )
            for bad in bad_links
        ):
            stats["sentences_removed_unverified"] += 1
            any_removed_so_far = True
            continue

        # Cut each dropped (no_full_text) citation's own rendered text out of the
        # sentence FIRST, unless that cut would split a mention this sentence keeps
        # (`_drop_would_split_a_kept_citation`) -- run before the coverage check below
        # so that a dropped citation's own conjunction/punctuation footprint (the
        # dangling "and (Jones, 2019)" a no_full_text drop leaves) is never mistaken
        # for unverified residue: it is an expected, already-handled drop, not a
        # proposition that failed verification.
        working_text = fragment_norm
        for citation_text in dropped_citation_texts:
            if _drop_would_split_a_kept_citation(
                working_text, citation_text, kept_citation_texts
            ):
                continue
            # Matched on the whitespace-normalised citation text, against the
            # already-collapsed sentence: a citation split by a formatting mark reaches
            # this function as "(Jones,  2019)" while the sentence being rewritten
            # carries "(Jones, 2019)", and an exact-substring cut would silently do
            # nothing while the counter said a citation had been dropped.
            cut = _strip_citation_substring(working_text, _doc_normalize_ws(citation_text))
            if cut == working_text:
                # Nothing was rendered to cut (a numbered citation, or a mention the
                # linker recorded in wording the sentence does not use). The link is
                # dropped either way, but the counter describes text this function
                # actually removed, so a pass that removes nothing reports nothing.
                continue
            stats["citations_dropped_no_full_text"] += 1
            working_text = cut

        # The gate's own, final coverage check -- independent of whatever
        # `coverage_incomplete`/`meta_evaluation` the citation-link step flagged before
        # verification ran, since a proposition it reported can since have failed
        # verification and dropped out of `kept_links_for_sentence` entirely. Run over
        # `working_text` (after the no_full_text drops above), not the raw fragment,
        # and covered only by the propositions and citations that actually survived; a
        # `bad_links` entry's own proposition is therefore never treated as covering
        # text, so its own content -- if it has any -- shows up as residue exactly
        # like an editorial aside no citation ever claimed.
        kept_propositions = [
            kept.get("proposition") or fragment_stripped for kept in kept_links_for_sentence
        ]
        non_frame_residues, frame_residues = sentence_coverage.sentence_residue_violations(
            working_text, kept_propositions, kept_citation_texts, kept_evidence_quotes
        )
        has_meta_evaluation = any(
            sentence_coverage.has_meta_evaluation(p) for p in kept_propositions
        )
        sentence_has_issue = bool(bad_links) or bool(non_frame_residues) or has_meta_evaluation

        # A sentence whose only problem is a leading attribution lead-in blocked by
        # a numeral or an out-of-lexicon noun (a fully verified finding otherwise)
        # is never rewritten here: the pipeline's own guarantee is to deliver only
        # what is verified, or omit the sentence entirely, and a deterministic
        # rewrite that manufactures or relocates a subject is not part of that
        # guarantee. It falls through to the ordinary trailing-excision/removal
        # path below, unchanged, like every other coverage-incomplete sentence.

        stats["frame_spans_accepted"] += len(frame_residues)

        if sentence_has_issue:
            cut = sentence_coverage.trailing_excision_cut(
                working_text, kept_propositions, kept_citation_texts
            )
            if cut is None:
                stats["sentences_removed_coverage_incomplete"] += 1
                stats["coverage_incomplete_reasons"].append(
                    _coverage_incomplete_reason(
                        fragment_stripped, bad_links, non_frame_residues, has_meta_evaluation
                    )
                )
                any_removed_so_far = True
                continue
            working_text = sentence_coverage.apply_trailing_excision(working_text, cut)
            stats["sentences_residue_excised"] += 1

        working_text = _finalized_sentence_text(working_text, marker_count, stats)

        # The coverage decision above leaves one gap open: `has_meta_evaluation`
        # there only ever runs over `kept_propositions`, so a comparative
        # meta-evaluation folded into an ACCEPTED frame span -- a colon lead-in the
        # residue classifier already let through ("Li and Hebert (2023) provide
        # stronger evidence: ...", the F2 frame is "provide stronger evidence"
        # itself) -- is invisible to it: the frame span is never part of any
        # proposition, so the proposition-only check never sees the words that
        # carry the comparison. Checked here instead, once, over `working_text`
        # itself -- the exact string about to be delivered, after whatever the
        # coverage decision already did to it -- so the same guard
        # `demo.check_delivered` rule 11 runs over the delivered draft directly
        # also runs over the app's own output before it ships. Conservative on a
        # match: the sentence is dropped whole, never rewritten or excised further
        # -- unlike the coverage gate above, this one does not know which part of
        # the text is the comparison and which is a verified finding sharing the
        # same sentence, so it does not attempt to separate them.
        meta_match = sentence_coverage.meta_evaluation_match(working_text)
        if meta_match:
            stats["sentences_removed_meta_evaluation_uncited"] += 1
            stats["meta_evaluation_reasons"].append({
                "sentence": fragment_stripped,
                "matched": meta_match,
            })
            any_removed_so_far = True
            continue

        # A final, unconditional guard over every sentence still kept at this
        # point, run for the same reason as the meta-evaluation guard just above:
        # a sentence whose own leading residue carries zero content tokens (a bare
        # "reports that"/"shows that" attribution) already passes the coverage
        # decision with no issue at all, regardless of whether anything actually
        # precedes that verb as its subject -- the residue rule's own "zero
        # content, always a permitted frame" branch does not check for a subject.
        # A sentence delivered exactly in this shape, with its own citation
        # trailing at the end rather than sitting in front of the verb, reads with
        # no subject at all ("Showed that one student resubmitted her essay 13
        # times while another made one resubmission (Zhang & Hyland, 2018)."). This
        # check is independent of how the sentence reached this point -- the
        # writer's own first draft, a revision pass, or any future change to the
        # coverage decision above -- so it is the pipeline's own last word on the
        # shape, not a patch on one particular path that produces it.
        if sentence_coverage.opens_with_reporting_verb(working_text):
            stats["sentences_removed_verb_initial"] += 1
            stats["verb_initial_reasons"].append({
                "sentence": fragment_stripped,
                "delivered": working_text,
            })
            any_removed_so_far = True
            continue

        # A surviving link whose stored ``sentence`` is not the sentence this
        # paragraph now carries is stale -- a later pass over this paragraph
        # (`extract_claims_from_document`'s own staleness check) will not find it, and
        # falls back to the author-year regex, losing `evidence_ids` and the narrowed
        # `proposition` the same way it does for a grouped citation. Every kept link is
        # therefore refreshed to ``working_text`` itself: not only when rule 3 rewrote
        # the sentence, but whenever the two differ at all. They also differ when the
        # claim this link was rebuilt from was cut from another paragraph carrying the
        # same sentence in different whitespace (a mark makes
        # `_extract_text_from_tiptap` join a paragraph's text nodes with an extra
        # space), which left a link naming a sentence no paragraph contained and the
        # heal after it rewriting that paragraph's attrs, which is not a no-op. A
        # ``proposition`` the rewritten sentence no longer holds is dropped with the
        # same occurrence check `extract_claims_from_document` itself applies, rather
        # than carried forward as a claim clause the new text cannot support.
        #
        # The refresh leaves nothing on the link itself to say what its ``sentence``
        # was BEFORE it, so `surviving_verifications` -- which pairs the raw verdict
        # list back to this link on ``(claim own sentence, key)`` -- would otherwise
        # not find a link it had, in fact, just kept: the verdict was computed against
        # the pre-heal sentence, not the rewritten one. The pre-to-post wording goes
        # into ``healed_sentences``, returned to that caller for this pass only, rather
        # than onto the link, which would carry it into the saved document and into
        # the next heal.
        rewritten = working_text != fragment_stripped
        refreshed_links = []
        for link in kept_links_for_sentence:
            claim_sentence = link.get("sentence") or fragment_stripped
            if claim_sentence == working_text:
                refreshed_links.append(link)
                continue
            healed_sentences[claim_sentence] = working_text
            updated = {**link, "sentence": working_text}
            proposition = updated.get("proposition")
            # `working_text` is already in the normalised form, so a normalised
            # proposition is compared against it directly.
            if rewritten and proposition and _doc_normalize_ws(proposition) not in working_text:
                updated.pop("proposition", None)
            refreshed_links.append(updated)

        fragment_records.append({
            "text": working_text,
            "has_citation": True,
            "preceded_by_removal": any_removed_so_far,
            "uncited_entries": [],
        })
        if fragment_norm not in linked_sentences_returned:
            linked_sentences_returned.add(fragment_norm)
            surviving_links.extend(refreshed_links)

    # The deterministic coherence pass, over this paragraph's own surviving fragments
    # only -- it can only ever drop one of them (`has_citation` is always kept), never
    # add or rewrite one.
    surviving_records = _drop_dangling_framing_sentences(fragment_records, stats)
    kept_fragments = [record["text"] for record in surviving_records]
    surviving_uncited: list[dict] = []
    for record in surviving_records:
        for entry in record["uncited_entries"]:
            surviving_uncited.append({**entry, "sentence": record["text"]})

    rebuilt = " ".join(f for f in kept_fragments if f).strip()
    return rebuilt, surviving_links, stats, healed_sentences, surviving_uncited


class FinalizeResult(NamedTuple):
    """What finalizing one freshly generated section produces: the cleaned text, its
    repaired citation-link list (paragraph_index recomputed
    against the returned text's own blocks), the removal/drop statistics,
    ``healed_sentences``, the pre-heal-to-healed sentence map (see
    `_finalize_paragraph_text`) that `surviving_verifications` needs to pair this
    pass's own verdicts back to a link rule 3 rewrote, and ``uncited_sentences``, the
    surviving (framing, or still-unclassified-but-kept) uncited entries, likewise
    recomputed against the returned text's own blocks so the
    caller can persist them onto the saved document (`_build_section_tiptap_nodes`) and
    a later heal never re-treats an already-classified sentence as a fresh coverage gap.
    That map belongs to the pass that produced it and to nothing else: it is
    deliberately not stored on the links, which are saved into the document and
    reported in the job result. ``coverage_
    incomplete_reasons`` is every ``sentences_removed_coverage_incomplete``
    sentence's own cause, popped out of each paragraph's own ``stats`` before that
    dict's ordinary per-key integer aggregation (so it is never miscounted as
    though it were a number) and concatenated here instead, in paragraph order,
    for the write job's own job result to carry into ``loop_stats``.
    ``meta_evaluation_reasons`` is the same treatment for every ``sentences_
    removed_meta_evaluation_uncited`` sentence (an uncited sentence dropped for its
    own full text, or a cited one dropped for its own full delivered text after
    the coverage decision), naming the sentence and the matched span.
    ``verb_initial_reasons`` is the same treatment again for every ``sentences_
    removed_verb_initial`` sentence, naming the sentence before finalize touched
    it and the text that was about to be delivered when the guard caught it.
    ``heading_shaped_reasons`` is the same treatment again for every ``sentences_
    removed_heading_shaped`` sentence, naming the sentence the second, independent
    heading-shape guard removed. ``headings_removed_reasons`` names every heading
    block *text* itself carried -- the section title is the only heading a saved
    section ever delivers, so every markdown heading line or whole-line bold heading
    run the model wrote inside its own body, at any position, is dropped whole
    (``stats["headings_removed"]``, the plain count); this list carries each dropped
    heading's own verbatim text, in the order finalize met them, for the write job's
    own job result to carry into ``loop_stats`` alongside the other reason lists.
    ``truncated_fragment_reasons`` is the same treatment again for every
    marker-headed fragment `_finalize_paragraph_text` dropped rather than merged
    into its predecessor (see that function's own ``sentences_truncated_fragment_
    repaired`` comment), naming the sentence that was dropped and
    `_fragment_truncation_reason`'s own text for why."""

    text: str
    citation_links: list[dict]
    stats: dict[str, int]
    healed_sentences: dict[str, str]
    uncited_sentences: list[dict]
    coverage_incomplete_reasons: list[dict[str, Any]] = []
    meta_evaluation_reasons: list[dict[str, Any]] = []
    verb_initial_reasons: list[dict[str, Any]] = []
    heading_shaped_reasons: list[dict[str, Any]] = []
    headings_removed_reasons: list[dict[str, Any]] = []
    truncated_fragment_reasons: list[dict[str, Any]] = []


def _paragraph_index_for_sentence(
    blocks: Sequence[str], item: Mapping[str, Any]
) -> int | None:
    """The index, into *blocks*, of the block that actually contains *item*'s own
    ``sentence``: a citation-link call's own reported
    ``paragraph_index`` is untrusted model output, and `unclassified_body_sentences`
    already ignores it entirely for exactly this reason. `finalize_generated_section`,
    unlike that coverage check, DOES key its per-paragraph removal rules on
    ``paragraph_index`` -- a link naming the wrong paragraph is invisible to the
    paragraph that actually holds its sentence (finalize sees no citation on it there)
    while the coverage check still marks the sentence covered (it matches the
    misindexed link's own ``sentence`` text, index-agnostically), so an unsupported
    cited sentence could survive, unverified, with zero gap ever reported. Matched with
    the same containment rule `unclassified_body_sentences` uses (M2): a fragment/
    sentence pair counts as a match when either contains the other, normalised. Falls
    back to the item's own reported ``paragraph_index`` (or ``None``) when no block
    matches at all -- a sentence a rewrite already changed out from under it, for
    instance -- so a caller's existing "unknown index" handling is unchanged."""
    sentence_norm = _finalize_sentence_norm(item.get("sentence") or "")
    if sentence_norm:
        for i, block in enumerate(blocks):
            if _is_heading_block(block):
                continue
            for start, end in _sentence_fragments(block):
                fragment_norm = _finalize_sentence_norm(block[start:end])
                if not fragment_norm:
                    continue
                if sentence_norm in fragment_norm or fragment_norm in sentence_norm:
                    return i
    index = item.get("paragraph_index")
    return index if isinstance(index, int) else None


_HeadingBodyItem = Any


def _heading_no_body_keep_mask(
    items: Sequence[_HeadingBodyItem], is_heading: Callable[[_HeadingBodyItem], bool]
) -> list[bool]:
    """One ``True``/``False`` per item of *items* -- ``False`` for a
    heading with nothing after it, because it is the last item at all or because the
    very next item is itself another heading. Rule (a) (`finalize_generated_section`,
    `_rebuild_finalized_node_list`) can remove the only body a heading had, whether
    because every sentence under it failed the ordinary per-sentence rules, because it
    announced an enumeration it never delivered, or because it carried no citation at
    all and was not the document's own opening paragraph; whichever rule emptied it, a
    heading left standing over nothing reads, in a rendered document, exactly like the
    dangling sentence rule (a)/(b)/(c) exist to remove.

    Processed in one right-to-left pass: *body_seen* is whether a non-heading item has
    been seen since the last heading kept (going backward), so a heading is dropped
    when nothing but headings (or nothing at all) follows it, and kept -- resetting
    *body_seen* for whatever precedes it -- the moment a real body item is found before
    the next heading. This cascades correctly when two headings end up adjacent (each
    with no body of its own): both are dropped, in one pass, because *body_seen* is
    never set by a heading itself, only by a genuine body item.

    The very first item is never dropped for lacking a body, even when it is a heading:
    a document (or section) every other rule has hollowed out down to nothing still
    keeps its own title, exactly as the opening PARAGRAPH exemption above never drops
    the document's own first paragraph either -- an empty document with no title at all
    is a worse outcome than one whose only content left is its own heading."""
    keep_reversed: list[bool] = []
    body_seen = False
    for item in reversed(items):
        if is_heading(item):
            keep_reversed.append(body_seen)
            if body_seen:
                body_seen = False
        else:
            body_seen = True
            keep_reversed.append(True)
    keep_reversed.reverse()
    if keep_reversed:
        keep_reversed[0] = True
    return keep_reversed


def _drop_headings_with_no_body(
    items: Sequence[_HeadingBodyItem], is_heading: Callable[[_HeadingBodyItem], bool]
) -> list[_HeadingBodyItem]:
    """`items` with every heading `_heading_no_body_keep_mask` marks as bodiless
    dropped -- for a caller (`finalize_draft_document`) that has no positional index to
    remap on the items it drops."""
    mask = _heading_no_body_keep_mask(items, is_heading)
    return [item for item, keep in zip(items, mask) if keep]


def finalize_generated_section(
    text: str,
    citation_links: Sequence[Mapping[str, Any]],
    uncited_sentences: Sequence[Mapping[str, Any]],
    claim_status: Mapping[tuple[str, str, str], str],
    section_title: str | None = None,
    claim_evidence_quotes: Mapping[tuple[str, str, str], Sequence[str]] | None = None,
) -> FinalizeResult:
    """Finalize one section's freshly generated markdown text: remove every cited
    sentence whose final status is not verified, remove every uncited
    sentence the retry actually classified as a "finding", drop a citation of a paper
    without full text (and the sentence too, when no verified citation remains on it),
    and strip every ``[NEEDS CITATION]`` marker.

    The section's own title, injected separately by the caller
    (`app.services.writing._build_section_tiptap_nodes`) and never part of *text*
    itself, is the only heading a saved section ever carries. Every markdown heading
    block *text* itself contains (`_is_heading_block`, ``#``-style or a whole-line
    bold run), at any position, is dropped whole -- never split into sentences,
    never demoted to a plain-text paragraph, and never kept as a heading node either,
    whatever it says: a genuine repeat of *section_title*, a differently-worded
    sub-heading, or a structural label such as "References" or "Synthesis and Gaps".
    Counted under ``stats["headings_removed"]``, with each dropped heading's own
    verbatim text recorded in ``headings_removed_reasons``, in the order they were
    met. A body paragraph that followed a dropped heading is otherwise treated
    exactly like any other paragraph -- removing a heading never removes prose.
    `app.services.writing._strip_duplicate_leading_heading` makes the identical,
    unconditional drop for a heading leading the model's raw text, before
    generation's own removals ever run; this is the same rule applied here, after
    those removals, to every heading anywhere in the body, including one that only
    became finalize's own first surviving block because the paragraph that used to
    precede it was itself removed.

    ``citation_links`` and ``uncited_sentences`` are grouped by the paragraph that
    actually contains each entry's own ``sentence`` (`_paragraph_index_for_sentence`),
    not blindly by the entry's own reported
    ``paragraph_index`` -- untrusted model output a mistaken index would otherwise let
    disagree with `unclassified_body_sentences`'s own, index-agnostic coverage check.
    ``claim_status`` maps ``(sentence, key)`` to the claim's final verification status,
    built by the caller from whichever ``(claims, verifications)`` pair it actually sent
    to the verifier.

    A body paragraph left with no surviving citation link at all is dropped whole --
    exactly like an empty ``rebuilt`` already was -- unless it is the
    section's own opening paragraph (the first non-heading block, tracked once and
    never reset: this function finalizes one section's own text, and a heading is
    never body of its own). A framing-only paragraph beyond the first is orphaned
    prose the removals above already hollowed out; the section's own opening paragraph
    is kept regardless, since dropping it would leave the section's own heading
    followed by nothing at all.

    A body paragraph left standing after every rule above whose own text echoes
    *section_title* (`_is_title_echo`) is dropped too, counted under
    ``sentences_removed_title_echo`` -- a section reading only as a restatement of
    its own title delivers no finding and no framing either, and reads to a reader
    as the section repeating itself. ``section_title`` defaults to ``None`` for a
    caller with no title to check against (every existing test of this function that
    does not pass one), in which case this check never fires. Every heading this
    function might otherwise have compared against is already gone by this point --
    dropped whole above, not merely demoted -- so there is no "heading still
    standing in the text" case left to check here at all, unlike before this fix.

    ``claim_evidence_quotes`` -- keyed exactly like *claim_status*, ``(sentence,
    proposition, key)``, but to the claim's own verified evidence quote(s) rather
    than its status -- narrows the coverage gate's own numeral disqualification
    (`sentence_coverage.covered_numeral_keys`): a residue whose only problem is a
    numeral (a digit or number word) no proposition covers is no longer removed
    for that reason alone when the SAME number, in either form, is verbatim in one
    of the sentence's own kept claims' evidence quotes. Defaults to ``None`` (every
    existing caller and test of this function), in which case every numeral
    disqualifies exactly as before this parameter existed."""
    from app.agents.citation_link_agent import split_paragraphs

    blocks = split_paragraphs(text)
    links_by_paragraph: dict[int, list[dict]] = {}
    for link in citation_links:
        index = _paragraph_index_for_sentence(blocks, link)
        if isinstance(index, int):
            links_by_paragraph.setdefault(index, []).append(dict(link))
    uncited_by_paragraph: dict[int, list[dict]] = {}
    for sentence in uncited_sentences:
        index = _paragraph_index_for_sentence(blocks, sentence)
        if isinstance(index, int):
            uncited_by_paragraph.setdefault(index, []).append(dict(sentence))

    # Fix B's companion guard: a link whose normalised sentence equals no fragment
    # norm of the paragraph it was bucketed into -- including one bucketed onto an
    # index outside *blocks* altogether (an untrusted, out-of-range ``paragraph_
    # index`` `_paragraph_index_for_sentence` falls back to when no block contains its
    # sentence at all) -- is invisible to the per-paragraph pass below: it is silently
    # never turned into a claim, and its own paragraph is later counted as merely
    # dangling rather than as the real loss it is. Counted here, once, over every
    # block *citation_links* was bucketed against, so this can never happen in
    # silence again even where the sentence snap (`validate_citation_link`) does not
    # reach (a link this function is handed directly, bypassing that snap, as every
    # test of this function still does).
    links_unmatched_to_sentence = 0
    for index, bucketed_links in links_by_paragraph.items():
        block = blocks[index] if 0 <= index < len(blocks) else None
        fragment_norms = (
            {
                _finalize_sentence_norm(block[start:end])
                for start, end in _sentence_fragments(block)
            }
            if block is not None
            else set()
        )
        for link in bucketed_links:
            sentence_norm = _finalize_sentence_norm(link.get("sentence") or "")
            if not sentence_norm or sentence_norm not in fragment_norms:
                links_unmatched_to_sentence += 1

    total_stats: dict[str, int] = {
        "links_unmatched_to_sentence": links_unmatched_to_sentence,
        "sentences_removed_title_echo": 0,
        "headings_removed": 0,
    }
    final_blocks: list[str] = []
    surviving_links: list[dict] = []
    surviving_uncited: list[dict] = []
    healed_sentences: dict[str, str] = {}
    # The write job's own record of why every `sentences_removed_coverage_
    # incomplete` sentence was dropped, in paragraph order -- see
    # `FinalizeResult`'s own docstring.
    coverage_incomplete_reasons: list[dict[str, Any]] = []
    # The same, for every `sentences_removed_meta_evaluation_uncited` sentence's own
    # matched span.
    meta_evaluation_reasons: list[dict[str, Any]] = []
    # The same, for every `sentences_removed_verb_initial` sentence's own before/
    # after text.
    verb_initial_reasons: list[dict[str, Any]] = []
    # The same, for every `sentences_removed_heading_shaped` sentence.
    heading_shaped_reasons: list[dict[str, Any]] = []
    # Every heading block dropped whole below, in the order they were met -- see
    # `FinalizeResult`'s own docstring.
    headings_removed_reasons: list[dict[str, Any]] = []
    # The same, for every marker-headed fragment dropped rather than merged into
    # its predecessor (`_finalize_paragraph_text`'s own `sentences_truncated_
    # fragment_repaired` comment).
    truncated_fragment_reasons: list[dict[str, Any]] = []
    opening_paragraph_seen = False
    for index, block in enumerate(blocks):
        if _is_heading_block(block):
            # The section title is the only heading a saved section ever carries
            # (it is injected separately by the caller, never part of *text*): every
            # heading the writer wrote inside its own body is dropped whole here,
            # whatever it says and wherever it sits, never kept as a heading node
            # and never demoted to a plain-text paragraph.
            total_stats["headings_removed"] += 1
            headings_removed_reasons.append({"heading": block.strip()})
            continue
        rebuilt, kept_links, stats, healed, kept_uncited = _finalize_paragraph_text(
            block,
            links_by_paragraph.get(index, []),
            uncited_by_paragraph.get(index, []),
            claim_status,
            claim_evidence_quotes,
        )
        healed_sentences.update(healed)
        # Popped out before the ordinary per-key integer aggregation below: this one
        # key is a list, never a count, and is carried separately (`FinalizeResult`'s
        # own field) instead of being summed like every other key in `stats`.
        coverage_incomplete_reasons.extend(stats.pop("coverage_incomplete_reasons", []))
        meta_evaluation_reasons.extend(stats.pop("meta_evaluation_reasons", []))
        verb_initial_reasons.extend(stats.pop("verb_initial_reasons", []))
        heading_shaped_reasons.extend(stats.pop("heading_shaped_reasons", []))
        truncated_fragment_reasons.extend(stats.pop("truncated_fragment_reasons", []))
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value
        if not rebuilt:
            continue
        is_opening_paragraph = not opening_paragraph_seen
        opening_paragraph_seen = True
        # The guarantee that an "unclassified" sentence (a citation-link retry that
        # failed, or never mentioned it) is never silently dropped outranks rule (a)
        # here: a paragraph carrying one is never known to be dangling filler, only
        # genuinely uncertain, so it survives even without a citation of its own
        # (`test_resolve_and_finalize_keep_orphaned_sentences_
        # from_writing_result_on_retry_failure`, a real, closing synthesis paragraph
        # one unclassified sentence keeps whole).
        has_unclassified = any(entry.get("unclassified") for entry in kept_uncited)
        if not kept_links and not is_opening_paragraph and not has_unclassified:
            total_stats["sentences_removed_dangling"] = (
                total_stats.get("sentences_removed_dangling", 0)
                + len(_sentence_fragments(rebuilt))
            )
            continue
        new_index = len(final_blocks)
        final_blocks.append(rebuilt)
        for link in kept_links:
            surviving_links.append({**link, "paragraph_index": new_index})
        for entry in kept_uncited:
            surviving_uncited.append({**entry, "paragraph_index": new_index})

    # A body paragraph that restates the section's own title (`_is_title_echo`) is
    # dropped whole, never delivered -- see `_is_title_echo`'s own docstring. No
    # heading can ever reach this point (every one was already dropped whole in the
    # loop above), so the only reference text this check ever compares against is
    # *section_title* itself.
    if section_title:
        echo_keep_mask = [not _is_title_echo(block, [section_title]) for block in final_blocks]
        if not all(echo_keep_mask):
            echo_index_remap: dict[int, int] = {}
            echo_filtered_blocks: list[str] = []
            for old_index, (block, keep) in enumerate(zip(final_blocks, echo_keep_mask)):
                if not keep:
                    total_stats["sentences_removed_title_echo"] += 1
                    continue
                echo_index_remap[old_index] = len(echo_filtered_blocks)
                echo_filtered_blocks.append(block)
            final_blocks = echo_filtered_blocks
            surviving_links = [
                {**link, "paragraph_index": echo_index_remap[link["paragraph_index"]]}
                for link in surviving_links
                if link["paragraph_index"] in echo_index_remap
            ]
            surviving_uncited = [
                {**entry, "paragraph_index": echo_index_remap[entry["paragraph_index"]]}
                for entry in surviving_uncited
                if entry["paragraph_index"] in echo_index_remap
            ]

    return FinalizeResult(
        text="\n\n".join(final_blocks),
        citation_links=surviving_links,
        stats=total_stats,
        healed_sentences=healed_sentences,
        uncited_sentences=surviving_uncited,
        coverage_incomplete_reasons=coverage_incomplete_reasons,
        meta_evaluation_reasons=meta_evaluation_reasons,
        verb_initial_reasons=verb_initial_reasons,
        heading_shaped_reasons=heading_shaped_reasons,
        headings_removed_reasons=headings_removed_reasons,
        truncated_fragment_reasons=truncated_fragment_reasons,
    )


def _claims_for_paragraph_text(
    paragraph_text: str,
    claims: Sequence[tuple[str, str, str, str]],
    stored_links: Sequence[Mapping[str, Any]] = (),
) -> list[dict]:
    """Link entries for `_finalize_paragraph_text` to run over one paragraph, one per
    claim whose own ``claim_sentence`` occurs in *paragraph_text*, grouped back under
    whichever of *stored_links* -- the paragraph's own ``attrs.citationLinks`` -- named
    that claim's ``(sentence, key)`` pair.

    `extract_claims_from_document` resolves a citation unit through three sources --
    the writer's own ``attrs.citationLinks`` map, the pre-existing author-year regexes,
    or the numbered-citation resolver -- and only the first of those is ever written
    back onto the paragraph's own attrs. Finalize must know about a claim regardless of
    which source resolved it, so it is driven by the same flat ``claims`` list
    `_verify_claims` was given (every claim, from every source) rather than by
    ``attrs.citationLinks`` alone, which would silently miss a regex- or
    numbered-resolved claim entirely.

    A claim whose key belongs to one of *stored_links* is folded into ONE entry per
    stored link -- copying that stored link's own ``citation_text``, ``proposition``,
    ``evidence_ids`` and every other field, narrowed to just the keys this paragraph's
    claims actually cover -- rather than into its own single-key entry. A grouped
    citation such as "(Smith, 2020; Jones, 2019)" is stored as one link naming both
    keys; extraction re-derives its own per-key ``citation_text`` ("Smith, 2020",
    without the parentheses) that never equals the stored group's rendered text, so
    matching by ``(sentence, citation_text)`` (as an earlier version of this function's
    caller did) never found it -- the group was rebuilt as two separate one-key links,
    losing the stored ``proposition`` and ``evidence_ids`` and re-exposing the whole
    sentence to the verifier on the very next pass. Matching instead by ``(sentence,
    key)`` membership survives that re-derivation. A claim whose key belongs to no
    stored link (a regex- or numbered-citation-resolved claim, never written to
    ``attrs.citationLinks`` at all) still gets its own single-key
    ``{"sentence", "keys", "citation_text"}`` entry, exactly as before this change.

    The rebuilt entry's own ``sentence`` is always the claim's own, freshly-extracted
    sentence -- never the stored link's own, possibly stale one -- because
    `_finalize_paragraph_text` looks a claim's verdict up by
    exactly that sentence (the same string `extract_claims_from_document` cut the
    claim from, which built ``claim_status``). The two only ever differ once the
    paragraph's own text has moved since the link was stored -- a mark added around
    part of the cited sentence, for instance, which makes `_extract_text_from_tiptap`
    re-join the paragraph's text nodes with an extra space -- but they are normalised-
    equal by construction (the stored link's sentence is only ever an owner at all
    because its normalised form matched this same paragraph's current, normalised
    text). Carrying the stale one forward made every such lookup miss, so the whole
    sentence -- verified, marks and all -- was removed as if it had no verdict at all."""
    # Keyed on (sentence, proposition, key), not (sentence, key) alone, so two claims
    # naming the same sentence and key but DIFFERENT propositions are rebuilt as two
    # separate links here too, on the saved-draft heal path, exactly as
    # `claims_from_citation_links` now keeps them separate on the write job's own
    # path.
    paragraph_norm = _doc_normalize_ws(paragraph_text)
    present: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for claim_text, key, citation_text, sentence in claims:
        sentence_norm = _doc_normalize_ws(sentence)
        if sentence_norm and sentence_norm in paragraph_norm:
            claim_norm = _doc_normalize_ws(claim_text)
            present.setdefault(
                (sentence_norm, claim_norm, key), (sentence, citation_text, claim_text)
            )

    key_owner: dict[tuple[str, str, str], dict] = {}
    for stored in stored_links or []:
        stored_sentence_norm = _doc_normalize_ws(stored.get("sentence") or "")
        stored_proposition_norm = _doc_normalize_ws(
            stored.get("proposition") or stored.get("sentence") or ""
        )
        for key in stored.get("keys") or []:
            key_owner.setdefault((stored_sentence_norm, stored_proposition_norm, key), stored)

    links: list[dict] = []
    grouped_keys: dict[int, list[str]] = {}
    for (sentence_norm, claim_norm, key), (sentence, citation_text, claim_text) in present.items():
        owner = key_owner.get((sentence_norm, claim_norm, key))
        if owner is None:
            # No stored link owns this claim (a regex- or numbered-citation-resolved
            # claim): `claim_text` always equals `sentence` on this path (no
            # citation-link call ever narrowed it), so leaving `proposition` unset here
            # and letting `_finalize_paragraph_text`'s own fallback (`proposition or
            # sentence`) supply it is equivalent, and keeps this synthetic entry's
            # shape unchanged for every existing reader.
            links.append({"sentence": sentence, "keys": [key], "citation_text": citation_text})
            continue
        owner_id = id(owner)
        if owner_id not in grouped_keys:
            grouped_keys[owner_id] = []
            entry = {**owner, "sentence": sentence, "keys": grouped_keys[owner_id]}
            # A draft healed by an earlier build of this module can still carry
            # ``pre_heal_sentence`` on a stored link: a pairing key belonging to a pass
            # that ended long ago, which the heal after it would read as if it
            # described that pass and so lose the verified row it named. Finalize no
            # longer writes that field at all; dropping any inherited one here means
            # the next heal of such a draft saves the link without it, so the document
            # converges on its own.
            entry.pop("pre_heal_sentence", None)
            links.append(entry)
        grouped_keys[owner_id].append(key)
    return links


def _rebuild_finalized_node_list(
    nodes: list[dict],
    claims: Sequence[tuple[str, str, str, str]],
    claim_status: Mapping[tuple[str, str, str], str],
    stats_acc: dict[str, int],
    surviving_links_acc: list[dict],
    healed_sentences_acc: dict[str, str],
    section_state: dict[str, bool] | None = None,
) -> list[dict]:
    """Recursive worker for `finalize_draft_document`: rebuilds every paragraph node
    (including one nested inside a list item or blockquote, exactly as
    `_iter_paragraph_nodes` walks for extraction) via `_finalize_paragraph_text`,
    dropping a paragraph -- or a now-empty list/blockquote around it -- that finalizes
    to nothing. A non-paragraph, non-heading node (an image, ...) passes through
    unchanged. The section title is the only heading a saved section ever carries:
    the first heading node this rebuild meets, across the whole document, passes
    through unchanged too; every heading node after that first one is dropped
    whole, counted under ``stats_acc["headings_removed"]`` (no reasons list on this
    path -- see the other list-valued reason keys' own popped-and-discarded
    treatment below).

    A paragraph whose text `_finalize_paragraph_text` did not actually change AND
    whose repaired ``citationLinks`` are byte-identical to what it already stored is
    returned as the exact same node object -- marks, ``origin``, ``blockId`` and every
    other ``attrs`` key intact -- rather than being rebuilt into a single plain-text
    node: rebuilding unconditionally, on every paragraph of every heal, would drop
    every mark, relabel the user's own paragraphs as AI-written (the editor's
    provenance indicator defaults ``origin`` to "ai" when the attr is missing), and
    fail the "healing a clean draft changes nothing" invariant on the very first heal
    of every draft the write job produced.

    A paragraph whose links changed (a citationLinks repair, e.g. attaching a link an
    earlier pass never stored, or narrowing a mixed co-citation's own keys) but whose
    text did not keeps its own ``content`` untouched too -- only ``attrs`` is updated.
    `_claims_for_paragraph_text` already folds each claim back into the one stored link
    that named it, carrying ``proposition``,
    ``evidence_ids`` and every other stored field straight through rather than
    replacing them wholesale with the finalize helper's own synthetic single-key link,
    so no further merge step is needed here. Only a paragraph whose text actually
    changed is rebuilt as plain text, which is the one case still losing marks -- a
    narrower, accepted limitation than rebuilding on every heal.

    A paragraph left with no surviving citation link at all is dropped whole, exactly
    like an empty ``new_text`` already was -- unless it is the
    document's own opening paragraph. ``section_state`` (created fresh on the outermost
    call, threaded unchanged through every recursive one) tracks that: only the very
    first paragraph the whole document delivers is ever exempt; no heading, at any
    level, grants a second one.

    A heading node of level 2 or above does NOT re-arm the exemption for whatever
    paragraph follows it, even though "a saved document can carry several sections"
    might suggest it should: `finalize_generated_section` (the write loop's own copy
    of this same rule) and `check_paragraph_has_citation` (the checker's rule 8a) both
    grant exactly one exemption per document, never one per heading. Re-arming per
    heading would matter on a document whose own sub-headings are all level 2 (this
    product's Drafts are one section each, per `finalize_generated_section`'s own
    docstring; a model that gives its own internal sub-headings level 2 rather than
    level 3 produces exactly this shape): it would let this heal keep FOUR paragraphs
    the other two mechanisms would each keep only ONE of, so a heal that stripped the
    last citation off a later paragraph would leave a rule 8a violation this heal
    believed was legitimate. Tracking exactly one exemption everywhere is also what
    lets rule (a) actually remove a bodiless-framing paragraph under a later
    sub-heading instead of forever protecting it as if it were a fresh section's own
    opening line."""
    if section_state is None:
        section_state = {"opening_pending": True, "title_heading_kept": False}
    rebuilt: list[dict] = []
    for node in nodes:
        node_type = node.get("type")
        if node_type == "heading":
            # The section title is the only heading a saved section ever carries:
            # the very first heading node this whole rebuild meets (always the
            # document's own title, node 0 of the top-level call -- a heading has
            # never been observed nested inside a list or blockquote in this
            # product) is kept unchanged; every other heading node is dropped
            # whole, counted under `headings_removed`. No reasons list on this
            # path, for the same reason the other list-valued reason keys above
            # are popped and discarded here rather than carried forward.
            if not section_state["title_heading_kept"]:
                section_state["title_heading_kept"] = True
                rebuilt.append(node)
            else:
                stats_acc["headings_removed"] = stats_acc.get("headings_removed", 0) + 1
            continue
        if node_type == "paragraph":
            text = _extract_text_from_tiptap(node)
            stripped_text = text.strip()
            if not stripped_text:
                continue
            attrs = node.get("attrs") or {}
            stored_links = attrs.get("citationLinks") or []
            stored_uncited = attrs.get("uncitedSentences") or []
            paragraph_links = _claims_for_paragraph_text(text, claims, stored_links)
            new_text, merged_links, stats, healed, surviving_uncited = (
                _finalize_paragraph_text(text, paragraph_links, stored_uncited, claim_status)
            )
            healed_sentences_acc.update(healed)
            # The write stage's own job result carries the list-valued reason keys
            # onward (see `FinalizeResult`'s own docstring); the standalone
            # verify-and-heal action this function backs has no field to carry them
            # in (`ClaimVerificationReport.finalize_stats` is typed `dict[str,
            # int]`), so they are popped and discarded here, before the ordinary
            # per-key integer aggregation, rather than carried forward or summed as
            # though they were numbers -- the checks themselves still run on this
            # path too (they live in `_finalize_paragraph_text`, shared by both
            # callers), only their own reason lists have nowhere to go here.
            stats.pop("coverage_incomplete_reasons", None)
            stats.pop("meta_evaluation_reasons", None)
            stats.pop("verb_initial_reasons", None)
            stats.pop("heading_shaped_reasons", None)
            stats.pop("truncated_fragment_reasons", None)
            for key, value in stats.items():
                stats_acc[key] = stats_acc.get(key, 0) + value
            if not new_text:
                continue
            is_opening_paragraph = section_state["opening_pending"]
            section_state["opening_pending"] = False
            # See `finalize_generated_section`'s own comment: an "unclassified"
            # sentence's own guarantee outranks rule (a).
            has_unclassified = any(entry.get("unclassified") for entry in surviving_uncited)
            if not merged_links and not is_opening_paragraph and not has_unclassified:
                stats_acc["sentences_removed_dangling"] = (
                    stats_acc.get("sentences_removed_dangling", 0)
                    + len(_sentence_fragments(new_text))
                )
                continue
            surviving_links_acc.extend(merged_links)
            # Whitespace-insensitive: `_extract_text_from_tiptap` joins adjacent text
            # nodes with a literal space, so a paragraph split by a mark right after a
            # trailing space -- e.g. `["My own paragraph, ", bold("emphasised"), ...]`
            # -- gains a double space that finalize's own whitespace collapse then
            # removes even though nothing was actually healed. Comparing normalised
            # text keeps that a no-op rather than a "changed" rebuild that would drop
            # the mark.
            text_unchanged = _doc_normalize_ws(new_text) == _doc_normalize_ws(stripped_text)
            links_unchanged = merged_links == stored_links
            # `attrs.uncitedSentences` must be refreshed just as `attrs.citationLinks`
            # is, or a sentence this pass just classified (the coverage-gap retry
            # resolved it, or it was already framing-tagged) is not persisted, and a
            # LATER heal of this same draft sees an empty ``uncitedSentences`` again
            # and re-treats it as a fresh coverage gap.
            uncited_unchanged = surviving_uncited == stored_uncited
            if text_unchanged and links_unchanged and uncited_unchanged:
                rebuilt.append(node)
                continue
            new_attrs = dict(attrs)
            if merged_links:
                new_attrs["citationLinks"] = merged_links
            else:
                new_attrs.pop("citationLinks", None)
            if surviving_uncited:
                new_attrs["uncitedSentences"] = surviving_uncited
            else:
                new_attrs.pop("uncitedSentences", None)
            # A citationLinks-only repair (e.g. attaching a link an earlier pass never
            # stored, or narrowing a mixed co-citation's own keys) never touches the
            # paragraph's own content -- only a real text change rebuilds it as plain
            # text, which is the one case still losing marks (an accepted, narrower
            # limitation than rebuilding on every heal).
            new_node: dict = (
                {**node}
                if text_unchanged
                else {**node, "content": [{"type": "text", "text": new_text}]}
            )
            if new_attrs:
                new_node["attrs"] = new_attrs
            else:
                new_node.pop("attrs", None)
            rebuilt.append(new_node)
        elif node_type in _CONTAINER_NODE_TYPES:
            children = _rebuild_finalized_node_list(
                node.get("content") or [],
                claims,
                claim_status,
                stats_acc,
                surviving_links_acc,
                healed_sentences_acc,
                section_state=section_state,
            )
            if children:
                rebuilt.append({**node, "content": children})
        else:
            rebuilt.append(node)
    return rebuilt


def _split_off_reference_section(nodes: list[dict]) -> tuple[list[dict], list[dict]]:
    """(body_nodes, reference_section_nodes): like `_split_body_and_reference_nodes`,
    but keeps the References/Bibliography/... heading itself in the returned reference
    section rather than dropping it -- that function is tuned for claim extraction,
    which never needs the heading node back; `finalize_draft_document` reconstructs the
    whole document, so the heading must survive too."""
    for i, node in enumerate(nodes):
        if node.get("type") != "heading":
            continue
        heading_text = _extract_text_from_tiptap(node).strip()
        if not _REFERENCE_LIST_HEADING_RE.match(heading_text):
            continue
        return nodes[:i], nodes[i:]
    return nodes, []


class FinalizeDocumentResult(NamedTuple):
    """What finalizing one saved draft document produces: the rebuilt Tiptap content,
    the citation links finalize kept, the removal/drop statistics, and
    ``healed_sentences``, the pre-heal-to-healed sentence map (see
    `_finalize_paragraph_text`) `surviving_verifications` needs for this pass's own
    verdict pairing -- returned alongside the document rather than stored on the links
    saved into it, so that no pass can read the previous pass's pairing key."""

    content: dict
    citation_links: list[dict]
    stats: dict[str, int]
    healed_sentences: dict[str, str]


def finalize_draft_document(
    content: dict,
    claims: Sequence[tuple[str, str, str, str]],
    claim_status: Mapping[tuple[str, str, str], str],
) -> FinalizeDocumentResult:
    """Apply the same per-sentence finalize rules `finalize_generated_section` uses
    directly to a saved Tiptap draft document: every paragraph
    node's own text is rebuilt against every claim `extract_claims_from_document`
    resolved for it (``claims``, the same list `_verify_claims` was given -- see
    `_claims_for_paragraph_text`) plus its own ``attrs.uncitedSentences``; a paragraph
    that finalizes to nothing is dropped. A references section (and every non-paragraph
    node) is returned unchanged. Returns ``(new_content, surviving_links, stats)``.

    Design amendment A4 (removing an uncited "finding" sentence) can only fire through
    this function, or through `verify_and_heal_claims` which calls it, because
    ``attrs.uncitedSentences`` is written onto a saved draft. `verify_and_heal_claims`
    writes it itself, on every run (`_close_citation_link_coverage_gap`, that
    function's own "link" step): a hand-typed uncited "finding" sentence, or one a
    stale linker run never saw, is found there and repaired before this function is
    ever called, so a paragraph's own ``attrs.uncitedSentences`` here is only ever
    empty when that step found nothing to add to it, never because nothing wrote it at
    all. A gap that step's own bounded retry could not classify -- because the retry
    call itself failed, or succeeded without ever mentioning that sentence -- is
    marked and RECORDED there, not removed: `_rebuild_finalized_node_list` persists
    ``attrs.uncitedSentences`` back onto the document exactly as it persists
    ``attrs.citationLinks``, so a sentence this pass classifies (or leaves recorded as
    still unresolved) is never treated as a fresh coverage gap on a later heal.

    A top-level heading node left with no body at all by any of the above -- the last
    node of the document, or immediately followed by another heading -- is dropped
    (`_drop_headings_with_no_body`), the same cleanup
    `finalize_generated_section` applies to its own text blocks. Checked only at the
    top level: a heading node has never been observed nested inside a list or
    blockquote in this product, so `_rebuild_finalized_node_list`'s own per-container
    recursion is not asked to repeat this check at every depth."""
    body_nodes, reference_nodes = _split_off_reference_section(_top_level_nodes(content))
    stats: dict[str, int] = {}
    surviving_links: list[dict] = []
    healed_sentences: dict[str, str] = {}
    new_body = _rebuild_finalized_node_list(
        body_nodes, claims, claim_status, stats, surviving_links, healed_sentences
    )
    new_body = _drop_headings_with_no_body(
        new_body, lambda node: node.get("type") == "heading"
    )
    new_content = {**(content or {}), "type": "doc", "content": new_body + reference_nodes}
    return FinalizeDocumentResult(
        content=new_content,
        citation_links=surviving_links,
        stats=stats,
        healed_sentences=healed_sentences,
    )


def surviving_verifications(
    claims: Sequence[tuple[str, str, str, str]],
    verifications: Sequence[Any],
    surviving_links: Sequence[Mapping[str, Any]],
    healed_sentences: Mapping[str, str],
) -> list:
    """The subset of *verifications* whose own ``(sentence, key)`` pair is still
    present among *surviving_links* -- the links `finalize_generated_section` or
    `finalize_draft_document` actually kept -- in *verifications*' own order.
    Filtering the raw *verifications* list on ``status == "verified"``
    alone is wrong: rule 2 of `_finalize_paragraph_text` removes a whole sentence, and
    every citation on it, the moment any one of its citations is not verified, so a
    verified citation that shared a sentence with an unsupported one is gone from the
    final text even though its own status still says "verified". Every entry in
    *surviving_links* only ever carries a key that finalize itself established as
    verified (rule 3), so no separate status check is needed here -- membership in
    *surviving_links* already means "verified and still in the text".

    *claims* and *verifications* must be the same paired sequence `_verify_claims` was
    given and returned: ``zip(claims, verifications)`` yields one ``((claim_text, key,
    citation_text, sentence), verification)`` pair per claim, exactly as every caller of
    `_verify_claims` already keeps them.

    A surviving link's own ``sentence`` no longer always names the sentence its claims
    were verified against: rule 3 rewrites it in place once a co-cited no_full_text
    citation is dropped from the same sentence, which would otherwise make this exact
    pairing drop the verified claim it was rewritten alongside. *healed_sentences* is
    the map finalize
    returned for that same pass, from each rewritten sentence's pre-heal wording to the
    text that replaced it; a claim whose own sentence was rewritten is looked up under
    the rewritten wording, every other claim under its own sentence exactly as before.

    The map is a required argument, not an optional one, because a caller that omits it
    after a rewrite silently drops verified rows from the final report -- the failure
    this pairing exists to prevent. It must be the map from the pass that produced
    *surviving_links* and nothing else: passing an older pass's map (which is what
    reading a pairing key stored on a link amounts to) reintroduces the same bug one
    heal later. Pass an empty map when finalize
    reported no rewrite."""
    # Keyed on (sentence, proposition, key), not (sentence, key) alone -- two links
    # surviving on the same sentence and key but with different propositions (the
    # dedup fix, edit 4) would otherwise collide into one set entry, letting a claim
    # whose OWN proposition did not survive match anyway because a DIFFERENT
    # proposition of the same key did.
    survived_pairs = {
        (link.get("sentence"), link.get("proposition") or link.get("sentence"), key)
        for link in surviving_links
        for key in (link.get("keys") or [])
    }
    result: list = []
    for (claim_text, key, _citation_text, sentence), v in zip(claims, verifications):
        if (sentence, claim_text, key) in survived_pairs:
            result.append(v)
            continue
        if sentence not in healed_sentences:
            continue
        healed = healed_sentences[sentence]
        # A claim with no distinct proposition (`claim_text == sentence`, the common
        # case for a link `_finalize_paragraph_text` never gave a narrower one) has its
        # own "proposition" implicitly fall back to the CURRENT sentence text on the
        # surviving link too (`link.get("proposition") or link.get("sentence")`), so
        # once the sentence is rewritten that fallback tracks the rewrite -- the
        # pairing key's proposition component must be looked up post-heal here too, not
        # only its sentence component.
        healed_claim_text = healed if claim_text == sentence else claim_text
        if (healed, healed_claim_text, key) in survived_pairs:
            # A rewrite (a no_full_text drop's own cleanup, or
            # a trailing excision) changes the delivered sentence's own text; the
            # verification record's own ``claim_sentence`` -- read by
            # `demo/check_delivered.py` rule 1 and by every other consumer of
            # `final_report` -- must track that rewrite too, not just the pairing
            # lookup above, or the final report keeps naming a sentence the draft no
            # longer contains verbatim.
            if v.claim_sentence != healed:
                v = v.model_copy(update={"claim_sentence": healed})
            result.append(v)
    return result


async def _close_citation_link_coverage_gap(
    draft_content: dict, paper_lookup: dict[str, Paper], draft_id: UUID
) -> None:
    """`verify_and_heal_claims`'s own "link" step for the linker coverage rule (see
    that function's docstring): every non-heading paragraph node of *
    draft_content* is checked, independently, against its own
    ``attrs.citationLinks``/``attrs.uncitedSentences`` (`unclassified_body_sentences`);
    every gap found, across the whole document, is sent back to the citation-link model
    together, in ONE retry call -- the same bound the write job's own gated loop applies
    -- and whatever it resolves, or a synthetic "unclassified" finding entry (below) for
    whatever it still leaves unresolved, is written onto the OWNING paragraph node's own
    ``attrs``, mutated in place on the dict *draft_content* itself already holds (no
    document rebuild: the caller reads the same, now-repaired, structure straight back).

    ``paragraph_index`` on every merged or synthetic entry is always ``0``: unlike the
    write job's own flat, whole-section block list, each paragraph node here is checked
    against only its own text, and `_finalize_paragraph_text` downstream is likewise
    called once per Tiptap paragraph node, never over several joined together.

    A reference section (and every non-paragraph node) is left alone, exactly as
    `extract_claims_from_document` and `finalize_draft_document` already treat it. A
    retry call that itself fails (a provider outage, a malformed structured response)
    is caught and logged, exactly like the write job's own citation-link call, and
    leaves every gap unresolved; this repair call's own cost is not currently folded
    into any provenance this action's job result reports, an accepted, narrower scope
    than the write job's own ``citation_link_calls`` list.

    The synthetic entry written for a gap the retry could not classify -- whether
    because the retry call itself failed, or because it succeeded without ever
    mentioning that particular sentence -- carries ``"unclassified": True``, which
    stops `_finalize_paragraph_text` from ever removing it. Writing such a gap as an
    unconditionally removable "finding" instead would let a single transient linker
    failure on this user-facing action delete every sentence its own prior save had
    never gotten around to persisting into ``attrs.uncitedSentences`` in the first
    place -- root-caused in `app.services.writing._build_section_tiptap_nodes`, which
    persists that attribute from the moment a section is first written, so this retry
    only ever has to reach genuinely new gaps.

    `unclassified_body_sentences` does not treat a STALE
    ``"unclassified": True`` entry as coverage, so a gap this action already tried once
    and could not classify is found again on every later call -- giving it a fresh,
    bounded retry attempt each time, exactly like a brand-new gap -- instead of being
    permanently frozen the first time the retry call itself happened to fail. The stale
    placeholder for such a gap is dropped from its owning node's own
    ``attrs.uncitedSentences`` before this pass's own resolution (or its own fresh
    placeholder) is written, never left in place beside it."""
    from app.agents.citation_link_agent import link_citations

    body_nodes, _reference_nodes = _split_off_reference_section(_top_level_nodes(draft_content))
    gaps_by_node: list[tuple[dict, dict[str, Any]]] = []
    for node in _iter_paragraph_nodes(body_nodes):
        text = _extract_text_from_tiptap(node)
        if not text.strip():
            continue
        attrs = node.get("attrs") or {}
        node_gaps = unclassified_body_sentences(
            text, attrs.get("citationLinks") or [], attrs.get("uncitedSentences") or []
        )
        gaps_by_node.extend((node, gap) for gap in node_gaps)

    if not gaps_by_node:
        return

    target_by_sentence: dict[str, dict] = {
        _finalize_sentence_norm(gap["sentence"]): node for node, gap in gaps_by_node
    }
    # Every gap this action found for a node -- including one carrying a stale
    # ``"unclassified": True`` entry from an earlier call -- is dropped from that
    # node's own ``uncitedSentences`` up front, computed once per node so a node with
    # more than one gap accumulates every resolution onto the same base list rather
    # than each gap's own filtering discarding what a sibling gap just added.
    gap_norms_by_node: dict[int, set[str]] = {}
    for node, gap in gaps_by_node:
        gap_norms_by_node.setdefault(id(node), set()).add(_finalize_sentence_norm(gap["sentence"]))
    base_uncited_by_node: dict[int, list[dict]] = {}
    for node, _gap in gaps_by_node:
        key = id(node)
        if key in base_uncited_by_node:
            continue
        norms = gap_norms_by_node[key]
        base_uncited_by_node[key] = [
            entry
            for entry in (node.get("attrs") or {}).get("uncitedSentences") or []
            if _finalize_sentence_norm(entry.get("sentence") or "") not in norms
        ]

    retry_text = "\n\n".join(gap["sentence"] for _node, gap in gaps_by_node)
    try:
        retry_result = await link_citations(retry_text, list(paper_lookup.keys()))
    except Exception:
        logger.warning(
            "Citation-link coverage retry failed for draft %s", draft_id, exc_info=True
        )
        retry_result = None

    resolved_norms: set[str] = set()
    if retry_result is not None:
        for link in retry_result.links:
            norm = _finalize_sentence_norm(link.get("sentence") or "")
            node = target_by_sentence.get(norm)
            if node is None:
                continue
            attrs = dict(node.get("attrs") or {})
            attrs["citationLinks"] = [
                *(attrs.get("citationLinks") or []), {**link, "paragraph_index": 0}
            ]
            attrs["uncitedSentences"] = base_uncited_by_node[id(node)]
            node["attrs"] = attrs
            resolved_norms.add(norm)
        for entry in retry_result.uncited_sentences:
            norm = _finalize_sentence_norm(entry.get("sentence") or "")
            node = target_by_sentence.get(norm)
            if node is None:
                continue
            base_uncited_by_node[id(node)].append({**entry, "paragraph_index": 0})
            attrs = dict(node.get("attrs") or {})
            attrs["uncitedSentences"] = base_uncited_by_node[id(node)]
            node["attrs"] = attrs
            resolved_norms.add(norm)

    for node, gap in gaps_by_node:
        norm = _finalize_sentence_norm(gap["sentence"])
        if norm in resolved_norms:
            continue
        base_uncited_by_node[id(node)].append({
            "paragraph_index": 0,
            "sentence": gap["sentence"],
            "tag": "finding",
            "unclassified": True,
        })
        attrs = dict(node.get("attrs") or {})
        attrs["uncitedSentences"] = base_uncited_by_node[id(node)]
        node["attrs"] = attrs


async def verify_and_heal_claims(
    project_id: UUID,
    draft_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: the standalone action for an already-saved (and possibly
    hand-edited) draft -- link (the citation-link map already stored on the draft's
    own paragraphs, exactly as `extract_claims_from_document` always resolved it,
    plus the pre-existing regex fallbacks for anything the document's own text no
    longer supports) -> verify -> finalize, with the identical
    `finalize_draft_document` the write job's own gated loop uses.

    This keeps its name for API compatibility (`POST /projects/{id}/verify-and-heal`
    is unchanged), but no longer only reports: the draft is
    now actually healed -- a cited sentence whose final status is not verified is
    removed, a citation of a paper with no full text is dropped on its own (and its
    sentence too when no verified citation remains on it), and every
    ``[NEEDS CITATION]`` marker is stripped -- and the healed content is saved back
    onto the draft (versioned exactly as any other content-changing
    `PUT /drafts/{id}` would be, via `app.services.draft.update_draft`).

    A citation sharing one rendered mention with a verified one -- e.g. "(Jones, 2019;
    Lee, 2021)" with only Lee verified -- is a documented exception to that drop: the
    mention's own text is left rendered as
    written, unsplit, and the stored link survives narrowed to only its verified keys,
    the same limitation `_finalize_paragraph_text`'s own mixed-status branch documents
    for the write job's gated loop. Splitting one multi-key mention's own text into a
    verified part and a dropped part is out of this feature's scope on both paths.

    The job's own result keeps the exact same flat contract every existing reader
    already parses -- every field still describes the raw verification pass,
    unfiltered -- and adds three keys: ``healed`` (always ``True`` now),
    ``finalize_stats`` (the removal/drop counts `finalize_draft_document` returned),
    and ``final_report`` -- "ONE report of the final text", restricted to the claims
    that actually survive in the saved draft (every row verified, by construction).

    Before claims are extracted, every non-heading paragraph node's own text is
    checked against that node's own ``attrs.citationLinks`` and
    ``attrs.uncitedSentences`` (`unclassified_body_sentences`); a gap left by every
    source -- a hand-typed sentence, a sentence a stale linker run never saw, or a
    paragraph saved before this action wrote ``uncitedSentences`` at all -- is sent
    back to the citation-link model once, across the whole document (one call, not one
    per paragraph, the same bound the write job's own gated loop applies), and merged
    onto the owning paragraph's own attrs. A sentence the retry still cannot classify
    is written back as a synthetic ``uncitedSentences`` entry (``tag: "finding"``,
    ``"unclassified": True``); `finalize_draft_document`'s rule 1 KEEPS it, counted
    under ``sentences_unclassified_kept``, instead of removing it, since a retry that
    never positively classified a sentence must not be trusted to delete it. This runs
    on every draft, not only one this action itself wrote: ``attrs.uncitedSentences``
    from a draft saved before `app.services.writing._build_section_tiptap_nodes`
    started persisting it is simply empty, which is exactly the "everything is a gap"
    starting state the repair call is built to handle -- and now that
    `_build_section_tiptap_nodes` does persist it, that starting state is reached only
    once per draft, not on every heal.
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session,
            job_id,
            JobStatus.running,
            progress=0.0,
            progress_message="Starting claim verification...",
        )

    try:
        # 1. Load draft content
        async with session_factory() as session:
            draft = await session.get(Draft, draft_id)
            if draft is None:
                raise ValueError(f"Draft {draft_id} not found")
            draft_content = draft.content or {}

        # 2. Load all project papers (keyed by title keywords for matching) -- built
        #    before claim extraction so a narrative citation's surname can be resolved
        #    against the real library instead of guessing from sentence position
        #    alone.
        async with session_factory() as session:
            result = await session.execute(
                select(Paper)
                .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                .where(ProjectPaper.project_id == project_id)
            )
            papers = list(result.scalars().all())

        paper_lookup = _build_paper_lookup(papers)

        # 3. Close the citation-link coverage gap on the saved draft itself, before
        # claims are extracted -- see the docstring above. Mutates each
        # gapped paragraph node's own `attrs` in place, directly on `draft_content`, so
        # the extraction and finalize steps right after this both see the repaired
        # attrs without a second fetch or a full document rebuild.
        await _close_citation_link_coverage_gap(draft_content, paper_lookup, draft_id)

        # 4. Parse claims paragraph by paragraph: the writer's own citation-link map
        # first, the pre-existing author-year regexes for a sentence with no
        # surviving link, and the numbered-citation resolver last. This is this
        # action's own "link" step: the document's own stored citationLinks (now
        # repaired), re-validated against its current text.
        claims, citation_coverage = extract_claims_from_document(draft_content, paper_lookup)

        deps = await _load_analysis_deps(project_id, session_factory)

        # 5. Verify each claim. No evidence_quotes_by_claim here: a saved draft's own
        # paragraph attrs do not yet carry evidence_ids (the frontend attaches them),
        # so this is the pre-existing, unordered code path (design section 6
        # amendment A6).
        outcome = await _verify_claims(
            claims, paper_lookup, deps, job_id, session_factory, progress_label="Verified"
        )

        if outcome["cancelled"]:
            async with session_factory() as session:
                await task_service.update_job_status(
                    session,
                    job_id,
                    JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Claim verification cancelled.",
                )
            return

        # 6. Finalize: heal the draft's own content with the same rules and the same
        # `_finalize_paragraph_text` the write job's gated loop uses (item 4).
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.running, progress=0.9,
                progress_message="Finalizing verified text...",
            )
        # Keyed on (sentence, proposition, key).
        claim_status = {
            (sentence, ct, key): v.status
            for (ct, key, _cx, sentence), v in zip(claims, outcome["verifications"])
        }
        finalize_outcome = finalize_draft_document(draft_content, claims, claim_status)
        new_content = finalize_outcome.content
        surviving_links = finalize_outcome.citation_links
        finalize_stats = finalize_outcome.stats

        async with session_factory() as session:
            from app.services import draft as draft_service

            fresh_draft = await session.get(Draft, draft_id)
            if fresh_draft is not None:
                await draft_service.update_draft(
                    session, draft_id, fresh_draft.user_id, content=new_content
                )
                await session.commit()

        # 7. Store ONE report of the final, healed text: every surviving row is
        # verified by construction (finalize never keeps anything else), plus this
        # pass's own loop statistics.
        # The main report fields are UNCHANGED in meaning: every existing reader (the
        # API route, the frontend, this file's own pre-existing test suite) keeps
        # seeing the full, unfiltered verification pass. Healing is visible in three
        # additive keys only: `healed`, `finalize_stats`, and `final_report` (the
        # clean, "every row verified" view).
        report = ClaimVerificationReport(
            draft_id=draft_id,
            verifications=outcome["verifications"],
            verified_count=outcome["verified_count"],
            unsupported_count=outcome["unsupported_count"],
            nuance_count=outcome["nuance_count"],
            abstract_only_count=outcome["abstract_only_count"],
            error_count=outcome["error_count"],
            provenance=outcome["provenance"],
            full_text_coverage=outcome["full_text_coverage"],
            needs_rewrite=outcome["nuance_count"],
            needs_removal=outcome["unsupported_count"],
            contradicted_count=outcome["contradicted_count"],
            guarded_count=outcome["guarded_count"],
            citation_coverage=CitationCoverage.model_validate(citation_coverage),
        )
        # Built from the links finalize actually kept, not from the raw verdict list
        # -- see `surviving_verifications`.
        final_verifications = surviving_verifications(
            claims,
            outcome["verifications"],
            surviving_links,
            finalize_outcome.healed_sentences,
        )

        async with session_factory() as session:
            await task_service.update_job_status(
                session,
                job_id,
                JobStatus.completed,
                progress=1.0,
                progress_message="Claim verification complete.",
                result={
                    **report.model_dump(mode="json"),
                    "verified": outcome["verified_count"],
                    "healed": True,
                    "finalize_stats": finalize_stats,
                    "final_report": {
                        "verifications": [
                            v.model_dump(mode="json") for v in final_verifications
                        ],
                        "verified_count": len(final_verifications),
                    },
                },
            )
    except Exception as e:
        logger.exception(
            "Claim verification failed for draft %s in project %s",
            draft_id,
            project_id,
        )
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e)
            )


async def verify_user_edits(
    project_id: UUID,
    draft_id: UUID,
    changed_sections: list,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: Phase B — re-verify only the changed sections after user edits.

    Writes the same flat report contract as Phase A, plus ``sections_checked``.
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session,
            job_id,
            JobStatus.running,
            progress=0.0,
            progress_message="Re-verifying edited sections...",
        )

    try:
        # 1. Load project papers for citation matching
        async with session_factory() as session:
            result = await session.execute(
                select(Paper)
                .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                .where(ProjectPaper.project_id == project_id)
            )
            papers = list(result.scalars().all())

        paper_lookup = _build_paper_lookup(papers)

        # 2. Extract claims from changed sections only. `_extract_claims` returns plain
        # (sentence, citation_key, citation_text) triples -- it never sees a citation-link
        # map, only a changed section's raw text, so there is no proposition to narrow the
        # claim to. `_verify_claims` takes a fourth, claim_sentence element; it is the
        # same sentence here, since claim_text and claim_sentence are identical
        # whenever no proposition applied.
        all_claims: list[tuple[str, str, str, str]] = []
        for section_text in changed_sections:
            if isinstance(section_text, str):
                all_claims.extend(
                    (sentence, key, citation_text, sentence)
                    for sentence, key, citation_text in _extract_claims(
                        section_text, paper_lookup
                    )
                )

        # 3. Verify each claim (same routine and same contract as Phase A)
        deps = await _load_analysis_deps(project_id, session_factory)
        outcome = await _verify_claims(
            all_claims, paper_lookup, deps, job_id, session_factory,
            progress_label="Re-verified",
        )

        if outcome["cancelled"]:
            async with session_factory() as session:
                await task_service.update_job_status(
                    session,
                    job_id,
                    JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Claim verification cancelled.",
                )
            return

        report = ClaimVerificationReport(
            draft_id=draft_id,
            verifications=outcome["verifications"],
            verified_count=outcome["verified_count"],
            unsupported_count=outcome["unsupported_count"],
            nuance_count=outcome["nuance_count"],
            abstract_only_count=outcome["abstract_only_count"],
            error_count=outcome["error_count"],
            provenance=outcome["provenance"],
            full_text_coverage=outcome["full_text_coverage"],
            needs_rewrite=outcome["nuance_count"],
            needs_removal=outcome["unsupported_count"],
            contradicted_count=outcome["contradicted_count"],
            guarded_count=outcome["guarded_count"],
        )

        # 4. Final status
        async with session_factory() as session:
            await task_service.update_job_status(
                session,
                job_id,
                JobStatus.completed,
                progress=1.0,
                progress_message="Edit verification complete.",
                result={
                    **report.model_dump(mode="json"),
                    "verified": outcome["verified_count"],
                    "sections_checked": len(changed_sections),
                },
            )
    except Exception as e:
        logger.exception(
            "Edit verification failed for draft %s in project %s",
            draft_id,
            project_id,
        )
        async with session_factory() as session:
            await task_service.update_job_status(
                session, job_id, JobStatus.failed, error=str(e)
            )
