"""Harvest naturalistic claim-to-source items from published review articles (reader-extracted
citations via ``--from-readings`` replace this module's own regex extraction as the path that
feeds the shipped build; the broader discovery, test-first split, citation and text hygiene
remain, and the regex-extraction path itself is unchanged and still importable/testable, it
simply no longer produces the items this script ships).

Interpreter: the **system** ``python`` for a real (non ``--skip-fetch``) run (backend
dependencies for the reused acquisition pipeline). The pure helpers (splitter, citation regex,
reference-entry parsing, deterministic sampling, item assembly, ``--skip-fetch`` rebuild) and
the discovery-pagination helper (exercised against a fake in-process client, no real network)
run anywhere and are unit-tested in ``tests/test_claims_harvest.py``.
Importers: ``run_hss.py`` (unchanged) reads the two output files this script writes through
its existing ``--claims``/``--cache-dir`` contract; ``tests/test_claims_harvest.py``.

PURPOSE (brief): every existing claim-verification set (SciFact, WiCE, the HSS set) is built
from a claim the *evaluation harness* constructs. This script instead harvests a claim a human
review-article author actually wrote: a sentence from a published review that carries exactly
one in-text citation, paired with the cited work's own full text, so the verifier is also
judged on naturalistic prose (with all its hedges, paraphrase and partial support) rather than
only on constructed stimuli. Whether a harvested sentence is in fact *supported* by the cited
work's text is exactly what still needs a judgment pass after this script runs -- this script
only assembles the (claim, cited-full-text) pair; it makes no verification call itself.

WHICH CANDIDATE CLAIMS: v2 found them itself, by discovering review articles (design item 1
below) and mining their body text with a citation-detecting regex pipeline (design item 2).
v3's ``--from-readings`` (:func:`fetch_from_readings`/:func:`build_review_cache_from_readings`,
just below :func:`extract_candidates`) instead takes candidates a reader already found: an
Opus reader session (or a human) reads one review article and records, per candidate claim,
the verbatim sentence, its citation, the review's own matched reference-list entry and the
cited work's title/year/authors/DOI (schema documented at that function). A reader sees context
a regex cannot -- a block-quoted participant utterance, a citation that is really about the
review's own procedure, a reference entry split across a page break the PDF extractor mangled
-- so this is the path that now feeds the shipped ``real_claims_test.jsonl``/
``real_claims_dev.jsonl``. Both paths converge on the identical review-cache shape (``doi``/
``title``/``candidates``/``stats``) and the identical :func:`build_outputs` sampling/split/
assembly stage; only how a review's candidates are FOUND differs between them.

Reuses (imports, never copies) from ``build_hss_set.py``: ``JOURNALS`` (18 applied-linguistics
journals, most already carrying a resolved OpenAlex source id), ``load_pipeline``/
``acquire_one`` (Unpaywall lookup -> ``fetch_pdf_from_url`` -> ``extract_text_from_pdf`` ->
``chunk_text``, the same ``--min-chars``/``--min-chunks`` threshold), ``doi_slug``/
``load_cache`` (the ``<doi-slug>.json`` cache file convention), ``load_excluded_dois`` (the
source-list DOI-exclusion machinery), ``public_url``/``_bare_doi``, ``_authors`` (OpenAlex
``authorships`` -> a plain name list), ``OPENALEX_WORKS``/``CROSSREF_WORKS``, and a subset of
its PDF-extraction artefact guards (``HYPHEN_BREAK_RE``, ``ADJACENT_NUMBERS_RE``,
``FOOTNOTE_MARK_RE``, ``ELLIPSIS_RE``, ``REFERENCE_RE``, ``MIN_ALPHA_RATIO`` -- everything
``is_clean_sentence`` checks except ``ATTRIBUTION_RE``, which by design rejects the "Smith
(2019) found ..." shape this harvester exists to keep). Every discovery query is this module's
own (:func:`openalex_journal_query_url` and friends below), not
``build_hss_set.list_candidates``/``openalex_query_url``: that function's query is hardcoded to
``publication_year:2019-2024`` and one source per call for the (unrelated) HSS-source-
acquisition use case, whereas review discovery here needs "2015 or later" across four
independent channels with cursor pagination; it therefore also does not reuse that function's
Crossref-on-HTTP-429 fallback (recorded as a deviation -- a real run that hits OpenAlex's daily
rate limit mid-discovery undercounts candidates for the remaining channels rather than falling
back to Crossref).

Design (six steps; every constant lives at the top of its section below):

(1) Review-article discovery via OpenAlex, four independent channels, each cursor-paginated
    (:func:`openalex_cursor_pages`) and deduplicated by DOI: (a) the 18 :data:`JOURNALS`, by
    OpenAlex *source id* (:func:`resolve_source_ids` -- most already carry one from
    ``build_hss_set``; the rest are resolved at run time via the Sources API, never
    hardcoded); (b) OpenAlex *topics* for applied linguistics / SLA / language education /
    TESOL / educational technology in language learning / foreign language teaching
    (:data:`TOPIC_NAMES`, resolved to ids via :func:`resolve_topic_ids`); (c) OpenAlex
    *subfields* (:data:`SUBFIELD_NAMES`, :func:`resolve_subfield_ids`); (d) title keywords
    (:data:`TITLE_KEYWORDS`: review, meta-analysis, synthesis, state of the art, research
    agenda, scoping review). No topic, subfield or source id is ever hardcoded: a name that
    resolves to nothing is simply absent from that channel, never a hardcoded id silently
    returning zero candidates. Every channel filters to publication year 2015 or later
    (:data:`MIN_PUBLICATION_YEAR`), ``type`` restricted to ``article`` or ``review``, open
    access with a PDF location (:func:`has_pdf_location`: ``best_oa_location.pdf_url`` OR any
    ``locations[].pdf_url``), and whose title or type marks them as a review
    (:func:`is_review_work`) and is not front matter (:func:`is_front_matter_title`: guest
    editorials, erratum/corrigendum notices). Deduplicated, sorted by DOI for determinism,
    capped at ``--max-reviews`` (default 300).
(2) Per review: fetch + extract + chunk with the reused pipeline (:func:`build_hss_set.
    acquire_one`, wrapped in a per-candidate wall-clock budget, :func:`acquire_one_with_budget`
    -- design item 6); the review's own text is cleaned (:func:`clean_extracted_text`) and
    split into sentences with a citation-safe splitter (:func:`split_sentences_real`) that
    never breaks immediately after "et al.", "e.g.", "i.e.", "cf.", "vs." or a single-letter
    initial, unlike ``common.split_sentences`` (kept local to this module rather than changing
    that shared helper). Kept sentences: 15-60 words, not from the review's own "references"
    chunk, free of PDF-extraction and page-furniture artefacts (:func:`_is_artefact_free`),
    not the citing review's own procedure or corpus (:data:`REVIEW_OWN_WORK_RE`), containing
    exactly one in-text citation of the recognised forms (:func:`find_single_citation`; the
    narrative "and"/"&"/"and ... and" forms are anchored -- :data:`_NO_MIDLIST` -- so they
    cannot start mid-list and capture a co-author's surname instead of the first author's),
    dropping sentences that begin with See/Cf./For example, are list items or headings, start
    with an orphaned "(YYYY)" fragment, or do not end in terminal punctuation.
(3) Resolve the citation: parse the review's own reference list (:func:`parse_reference_entries`,
    from the chunk whose section is literally ``"references"``, cleaned first with
    :func:`clean_extracted_text`), match the leading surname (particle-aware -- "van Lier",
    "Al Nafjan" -- like ``backend/app/services/citation_audit.py``'s own particle handling),
    the *second* surname when the citation carries one (declining a match whose entry text does
    not contain it), and year (:func:`match_reference_entry`); take the entry's own printed
    DOI, verified against Crossref's title for that DOI when both are available
    (:func:`resolve_citation`, so a bled-together or mis-OCRed DOI is not accepted unchecked),
    else the best Crossref title match above a similarity threshold
    (:func:`best_crossref_match`). Either way, the cited work's *authors* come from its own
    OpenAlex ``authorships`` (:func:`openalex_work_by_doi`, :func:`build_hss_set._authors`),
    not from Crossref, which does not always carry an author list. The cited work must not be
    the review itself, not be a journal already excluded via ``load_excluded_dois`` (the two
    HSS source lists plus the frozen v3 test set's own sources file -- see design item 5), and
    its extracted text must reach the same ``--min-chars``/``--min-chunks`` thresholds as
    ``build_hss_set`` (the same :func:`build_hss_set.acquire_one` call enforces this). A cited
    work already present in the fulltext cache is never re-fetched (design item 6,
    resumability); citation resolution itself is always redone from the review's cached chunks
    on every run, so a fix to this module's own extraction logic applies to an already-fetched
    review without needing to redownload its PDF.
(4) Sampling: at most ``--max-per-review`` (default 4) items per review; reviews are assigned to
    dev or test by a seeded shuffle (:data:`DEFAULT_SEED` = 20260910,
    :func:`shuffled_review_order`) so dev and test cite disjoint reviews
    (:func:`split_items_by_review`); a review's items go to **test first**, then dev, so the
    held-out test set is never starved just because total yield is below the dev target
    (design item 5, fixing a dev-first split that previously left the test set empty); targets
    ``--n-dev`` (20) / ``--n-test`` (60).
(5) Outputs: ``real_claims_dev.jsonl`` / ``real_claims_test.jsonl`` (item contract below) and
    ``real_claims.build.json`` (every filter count, seed, targets, per-review counts, excluded
    DOIs, sha256 of each output file). Caches under ``data/real_claims_reviews/`` (one review's
    fetched text + its resolved citation candidates + its filter-funnel stats) and
    ``data/real_claims_fulltext/`` (one cited work's fetched text, ``build_hss_set``-shaped),
    both gitignored.
(6) Acquisition robustness: reviews are fetched with bounded concurrency (``--concurrency``,
    default 6, an ``asyncio.Semaphore``), every HTTP call in this module carries explicit
    connect and read timeouts (:func:`_http_timeout`), and every call into the reused
    acquisition pipeline is wrapped in a per-candidate wall-clock budget
    (:func:`acquire_one_with_budget`) that turns a hang (observed on Windows/httpx for one
    candidate during a prior real run of this harvester) into a recorded
    ``"timeout_budget_exceeded"`` skip instead of blocking the whole harvest indefinitely.
    A review or cited work already present in its cache directory is never re-fetched on a
    later run (resumability).

Item contract: matches ``run_hss.py``'s ``ITEM_KEYS`` exactly (``item_id``, ``rule``,
``expected``, ``claim``, ``source_doi``, ``chunk_doi``, ``chunk_index``, ``alteration``) plus
``title``/``authors``, so ``run_hss.py --claims real_claims_test.jsonl --cache-dir
data/real_claims_fulltext`` works unchanged: ``chunk_doi`` = ``cited_doi`` (its cache payload's
chunks are what ``run_hss.py`` loads and hands to the verifier), ``source_doi`` = ``review_doi``,
``title``/``authors`` = the cited work's (not the review's) metadata. ``rule`` is the constant
``"real"`` (cosmetic: only so ``run_hss.py``'s own per-rule print groups these items instead of
``None``). ``expected`` is ``null`` -- unlike every synthetic set, a real citation's true support
status is not known by construction; that judgment is a separate pass this script does not make,
and any accuracy figure ``run_hss.py`` prints over this set is meaningless until it does.

Two-stage acquisition/build split (mirrors ``build_hss_set.py``'s ``acquire_sources`` /
``build_items_from_cache`` split): stage 1 (network, skipped by ``--skip-fetch``) either
discovers reviews and resolves+fetches citations itself (:func:`fetch_reviews`, the regex-
extraction path) or, given ``--from-readings <path>``, resolves+fetches citations from a
readings file's rows instead (:func:`fetch_from_readings`) -- either way writing one
review-cache file per review (``doi``/``title``/``candidates``/``stats``/``extraction``, plus,
on the regex-extraction path only, the review's own fetched text and chunks) and one
fulltext-cache file per cited work. Stage 2 (:func:`build_outputs`, pure) rebuilds the two
output jsonl files and the build json purely from those caches, so ``--skip-fetch`` (or a
second run over the same caches, from either stage-1 path) is deterministic and reproducible
without any network access. Every cached candidate and its review
payload carry an ``extraction`` field (``"regex"`` or ``"readings"``, defaulted to ``"regex"``
for a cache file predating the field) so :func:`build_outputs`'s own ``extraction`` filter can
isolate one path's items from a cache directory that holds both; the
fulltext cache is passed through :func:`build_hss_set.check_cache_identity` before use, closing
the wrong-work identity check's own gap on an already-cached cited work; and
a readings row's claim is cross-checked against its own reader-recorded citation year/authors
before its citation is trusted. That cross-check's
citation is built straight from the ``CITATION_RE`` match rather than routed through
``find_single_citation``'s unrelated selection filters, which could otherwise silently disable the whole
check, and its surname comparison requires an exact parsed-entry surname
match rather than a bare substring, so a short surname cannot satisfy it inside an
unrelated word of the reference entry's own title.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from build_hss_set import (  # noqa: E402
    ADJACENT_NUMBERS_RE,
    CROSSREF_WORKS,
    DEFAULT_MIN_CHARS,
    DEFAULT_MIN_CHUNKS,
    ELLIPSIS_RE,
    FOOTNOTE_MARK_RE,
    HYPHEN_BREAK_RE,
    JOURNALS,
    MIN_ALPHA_RATIO,
    OPENALEX_WORKS,
    REFERENCE_RE,
    _authors,
    _bare_doi,
    acquire_one,
    check_cache_identity,
    check_fetched_text_identity,
    doi_slug,
    load_cache,
    load_excluded_dois,
    load_pipeline,
    public_url,
)
from common import (  # noqa: E402
    append_jsonl,
    export_env_from_dotenv,
    load_dotenv_values,
    now_iso,
    read_json,
    read_jsonl,
    resolve_path_args,
    token_jaccard,
    write_json,
)

REPO_ROOT = HERE.parent.parent
DEFAULT_REVIEWS_CACHE_DIR = HERE / "data" / "real_claims_reviews"
DEFAULT_FULLTEXT_CACHE_DIR = HERE / "data" / "real_claims_fulltext"
DEFAULT_DEV_OUT = HERE / "real_claims_dev.jsonl"
DEFAULT_TEST_OUT = HERE / "real_claims_test.jsonl"
DEFAULT_BUILD_OUT = HERE / "real_claims.build.json"
DEFAULT_HSS_SOURCES = HERE / "hss_sources.json"
DEFAULT_HSS_DEV_SOURCES = HERE / "hss_dev_sources.json"
DEFAULT_HSS_TEST_V3_SOURCES = HERE / "hss_test_v3_sources.json"

DEFAULT_SEED = 20260910
DEFAULT_N_DEV = 20
DEFAULT_N_TEST = 60
DEFAULT_MAX_REVIEWS = 300
DEFAULT_MAX_PER_REVIEW = 4
DEFAULT_CONCURRENCY = 6
MIN_SENTENCE_WORDS = 15
MAX_SENTENCE_WORDS = 60
CROSSREF_TITLE_SIMILARITY_THRESHOLD = 0.8
# A Crossref title-search candidate whose own publication year is
# more than this many years from the reference entry's own printed year is never picked, no
# matter how similar its title -- real-test-06's wrong match (a 2022 paper by the same two
# authors, same subject, matched at 0.667 similarity against a 2018 entry) had no year to
# check against before this guard existed.
CROSSREF_YEAR_TOLERANCE = 1
MIN_PUBLICATION_YEAR = 2015
PER_CANDIDATE_BUDGET_S = 90.0
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 30.0
# The --from-readings path: a reader (a human, or an Opus reader session)
# already read the review and copied its own reference-list entry verbatim, so the DOI
# resolution here only needs to SANITY-CHECK that reader's work, not re-derive it from
# scratch the way resolve_citation's own regex-extraction path must. Both thresholds are
# therefore looser than resolve_citation's (0.8 title-search, no sanity check at all on its
# printed-DOI path): READINGS_DOI_SANITY_THRESHOLD guards only against a reader-transcribed
# DOI that resolves to a clearly different work (a typo, a bled reference-list entry), and
# READINGS_TITLE_SIMILARITY_THRESHOLD (used only when no cited_doi was printed in the
# reference list at all) still requires a real title match, just not the regex path's
# stricter bar meant to compensate for having no independent read of the source.
READINGS_DOI_SANITY_THRESHOLD = 0.5
READINGS_TITLE_SIMILARITY_THRESHOLD = 0.6

# OpenAlex topic/subfield display names, resolved to ids at run time (design item 1) rather
# than hardcoded: no such id is documented anywhere in this repository, and a wrong hardcoded
# id would silently return zero candidates rather than failing loudly.
TOPIC_NAMES: tuple[str, ...] = (
    "Applied Linguistics",
    "Second Language Acquisition",
    "Language Education",
    "Teaching English to Speakers of Other Languages",
    "Educational Technology in Language Learning",
    "Foreign Language Teaching",
)
SUBFIELD_NAMES: tuple[str, ...] = ("Linguistics and Language", "Education")
TITLE_KEYWORDS: tuple[str, ...] = (
    "review", "meta-analysis", "synthesis", "state of the art", "research agenda",
    "scoping review",
)
SELECT_FIELDS = (
    "id,doi,title,publication_year,type,open_access,best_oa_location,primary_location,"
    "locations,authorships"
)
DEFAULT_PAGE_SIZE = 200
DEFAULT_MAX_PAGES_PER_CHANNEL = 5


def _http_timeout() -> Any:
    """``httpx.Timeout`` with explicit connect and read bounds (design item 6): a real run
    hung indefinitely on one candidate under httpx on Windows with no timeout specified, so
    every ``httpx.AsyncClient`` this module creates uses this instead of a single float."""
    import httpx

    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S, write=READ_TIMEOUT_S,
        pool=CONNECT_TIMEOUT_S,
    )


# -------------------------------------------------------------------------------------
# Review-article filters (design item 1)
# -------------------------------------------------------------------------------------

REVIEW_TITLE_RE = re.compile(
    r"\b(?:review|meta-analys(?:is|es)|meta\s+analysis|state\s+of\s+the\s+art|synthesis|"
    r"research\s+agenda)\b",
    re.IGNORECASE,
)
REVIEW_TYPE_VALUES: frozenset[str] = frozenset({"review"})
# Front matter/editorial titles: a guest-editorial
# introduction or an erratum/corrigendum/retraction notice is not a research article, and
# almost every declarative sentence in one summarises somebody ELSE's contribution -- exactly
# the attributed-finding shape this harvester must not mine. Checked before the review-title
# regex above, so a title containing both ("Editorial: a review of ...") is still excluded.
FRONT_MATTER_TITLE_RE = re.compile(
    r"\b(?:guest editors?|word from the editors?|editorial|erratum|corrigendum|retraction|"
    r"editors?'? introduction|special issue introduction)\b",
    re.IGNORECASE,
)


def is_review_title(title: str | None) -> bool:
    return bool(title) and REVIEW_TITLE_RE.search(str(title)) is not None


def is_front_matter_title(title: str | None) -> bool:
    return bool(title) and FRONT_MATTER_TITLE_RE.search(str(title)) is not None


def is_review_work(work: Mapping[str, Any]) -> bool:
    """Title contains a review marker, or OpenAlex's own ``type`` says ``"review"`` -- unless
    the title marks the work as front matter (:data:`FRONT_MATTER_TITLE_RE`), which is
    excluded regardless of the ``type`` OpenAlex recorded for it."""
    title = work.get("title")
    if is_front_matter_title(title):
        return False
    return is_review_title(title) or work.get("type") in REVIEW_TYPE_VALUES


def has_pdf_location(work: Mapping[str, Any]) -> bool:
    """True when ``best_oa_location.pdf_url`` is present, OR any entry of ``locations`` (design
    item 1: "``best_oa_location`` or any location with pdf_url") carries one. Real open access
    is only established later, by Unpaywall in ``acquire_one``; this is a cheap client-side
    pre-filter on the OpenAlex response alone."""
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        return True
    for loc in work.get("locations") or []:
        if isinstance(loc, Mapping) and loc.get("pdf_url"):
            return True
    return False


def _year_filter() -> str:
    return f"publication_year:>{MIN_PUBLICATION_YEAR - 1}"


def openalex_journal_query_url(
    source_id: str, mailto: str, *, cursor: str = "*", per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """One of the 18 :data:`JOURNALS`, by OpenAlex *source id* (design item 1a) -- more
    reliable than filtering by ISSN, since a journal can carry a print/electronic ISSN pair
    that does not always match the one OpenAlex indexed the source under."""
    filt = (
        f"primary_location.source.id:{source_id},open_access.is_oa:true,type:article|review,"
        + _year_filter()
    )
    return (
        f"{OPENALEX_WORKS}?filter={filt}&select={SELECT_FIELDS}&per-page={per_page}"
        f"&cursor={cursor}&mailto={mailto}"
    )


def topic_search_url(name: str, mailto: str) -> str:
    return f"https://api.openalex.org/topics?search={quote(name)}&per-page=1&mailto={mailto}"


def openalex_topic_query_url(
    topic_id: str, mailto: str, *, cursor: str = "*", per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    filt = (
        f"primary_topic.id:{topic_id},open_access.is_oa:true,type:article|review,"
        + _year_filter()
    )
    return (
        f"{OPENALEX_WORKS}?filter={filt}&select={SELECT_FIELDS}&per-page={per_page}"
        f"&cursor={cursor}&mailto={mailto}"
    )


def subfield_search_url(name: str, mailto: str) -> str:
    return f"https://api.openalex.org/subfields?search={quote(name)}&per-page=1&mailto={mailto}"


def openalex_subfield_query_url(
    subfield_id: str, mailto: str, *, cursor: str = "*", per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    filt = (
        f"primary_topic.subfield.id:{subfield_id},open_access.is_oa:true,type:article|review,"
        + _year_filter()
    )
    return (
        f"{OPENALEX_WORKS}?filter={filt}&select={SELECT_FIELDS}&per-page={per_page}"
        f"&cursor={cursor}&mailto={mailto}"
    )


def openalex_title_keyword_query_url(
    keyword: str, subfield_id: str, mailto: str, *, cursor: str = "*",
    per_page: int = DEFAULT_PAGE_SIZE,
) -> str:
    """A title-keyword match (design item 1c) is scoped to *subfield_id* (one of
    :data:`SUBFIELD_NAMES`, resolved once and reused here): an unscoped ``title.search:review``
    matches every open-access review in OpenAlex regardless of discipline (medicine, physics,
    ...), which crowded out the on-topic applied-linguistics candidates entirely once the
    combined, DOI-sorted pool was capped at ``--max-reviews`` (measured live: a 5-review cap
    returned five JAMA/Cochrane medical reviews and zero applied-linguistics ones)."""
    filt = (
        f"title.search:{quote(keyword)},primary_topic.subfield.id:{subfield_id},"
        "open_access.is_oa:true,type:article|review," + _year_filter()
    )
    return (
        f"{OPENALEX_WORKS}?filter={filt}&select={SELECT_FIELDS}&per-page={per_page}"
        f"&cursor={cursor}&mailto={mailto}"
    )


def source_lookup_url(issn: str, mailto: str) -> str:
    return f"https://api.openalex.org/sources?filter=issn:{issn}&select=id&per-page=1&mailto={mailto}"


def openalex_work_by_doi_url(doi: str, mailto: str) -> str:
    return f"{OPENALEX_WORKS}/https://doi.org/{doi}?mailto={mailto}"


# -------------------------------------------------------------------------------------
# Citation detection (design item 2): recognised APA in-text forms.
# -------------------------------------------------------------------------------------

# Lower-case surname particles kept with the surname ("van Lier", "de Bot", "Al Nafjan"): same
# list and technique as backend/app/services/citation_audit.py's _PARTICLES/_PARTICLE_RE/
# _SURNAME (reimplemented locally, not imported -- that module is a general APA-citation
# *auditor* for generated text, not owned by this script; see normalize_surname below, which
# already documents the same reimplement-not-import choice).
_PARTICLES = frozenset(
    "van von de del della der den du da das dos di la le el al ter ten op bin ibn".split()
)
_PARTICLE_RE = r"(?i:" + "|".join(sorted(_PARTICLES, key=len, reverse=True)) + r")"
_NAME = rf"(?:{_PARTICLE_RE}\s+)*[A-Z][A-Za-zÀ-ÿ'\-]+"
_YEAR = r"(?:19|20)\d{2}"
# A narrative surname (outside parentheses: "Smith (2020)", "Smith and Jones (2020)") must not
# start immediately after a co-author separator (" and ", " & "), or the regex would capture
# the LAST author of a multi-author narrative citation as if it were a lone single citation --
# e.g. "Crosthwaite and Baisa (2023)" would otherwise match "Baisa (2023)" alone, which (when
# Baisa also has an unrelated same-year reference-list entry of their own) silently resolves
# the claim to the wrong paper. A plain comma is deliberately NOT one of the guarded
# separators: ordinary prose constantly precedes a citation with a comma that has nothing to
# do with an author list ("Building on this foundation, Aryadoust et al. (2022) showed ..."),
# and APA style itself never joins two authors with a bare comma (only "and"/"&"), so the
# residual risk a comma guard would close is negligible next to the recall it costs.
# Parenthetical forms need no such guard: they can only start right after the pattern's own
# literal "(", which a co-author separator cannot precede.
_NO_MIDLIST = r"(?<!\band\s)(?<!&\s)"
# Alternation order matters here (unlike the four purely-parenthetical branches, which remain
# mutually exclusive by their distinct post-surname literal): the multi-author narrative
# branches must be tried before the single-surname narrative branch so that, at the first
# surname's start position, the longer, correct-shape match is taken instead of stopping short.
CITATION_RE = re.compile(
    rf"\((?P<p2_s1>{_NAME}) and (?P<p2_s2>{_NAME}), (?P<p2_year>{_YEAR})\)"
    rf"|\((?P<pe_s1>{_NAME}) et al\., (?P<pe_year>{_YEAR})\)"
    rf"|\((?P<ps_s1>{_NAME}), (?P<ps_year>{_YEAR})\)"
    rf"|{_NO_MIDLIST}(?P<ne3_s1>{_NAME}), (?P<ne3_s2>{_NAME}), and (?:{_NAME}) "
    rf"\((?P<ne3_year>{_YEAR})\)"
    rf"|{_NO_MIDLIST}(?P<ne2_s1>{_NAME}) and (?P<ne2_s2>{_NAME}) \((?P<ne2_year>{_YEAR})\)"
    rf"|{_NO_MIDLIST}(?P<ne2amp_s1>{_NAME}) & (?P<ne2amp_s2>{_NAME}) \((?P<ne2amp_year>{_YEAR})\)"
    rf"|{_NO_MIDLIST}(?P<ne_s1>{_NAME}) et al\. \((?P<ne_year>{_YEAR})\)"
    rf"|{_NO_MIDLIST}(?P<ns_s1>{_NAME}) \((?P<ns_year>{_YEAR})\)"
)
# Any 4-digit 19xx/20xx token, used as a cheap "exactly one citation" guard: a sentence with a
# second citation in an unrecognised form (e.g. an ampersand list) still carries a second year
# token even though CITATION_RE itself only matches the recognised one.
YEAR_TOKEN_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
LEADING_DROP_RE = re.compile(r"^\s*(?:See\b|Cf\.|For\s+example\b)")
# A multi-level section number ("4.3 General characteristics") is a list/heading item, but an
# earlier version of this pattern did not recognise it because its numeric branch required a single
# level of digits followed by a mandatory "." or ")" before the whitespace -- "4.3" has a "."
# followed by another digit ("3"), not whitespace, so the old pattern simply did not match.
# The trailing "[.)]" is now optional too, since a bare multi-level number ("4.3") often
# carries no punctuation of its own before the heading text.
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-•*]\s|\(?\d{1,2}(?:\.\d{1,2}){0,3}[.)]?\s|\(?[a-hA-H][.)]\s)"
)
HEADING_RE = re.compile(
    r"^(?:Introduction|Background|Method(?:s|ology)?|Results?|Findings|Discussion|"
    r"Conclusions?|Abstract|Summary|Overview|Literature\s+Review|Theoretical\s+Framework|"
    r"References)\s+[A-Z]"
)
# Belt-and-braces guards: even with the citation-safe
# splitter below, a candidate sentence starting with an orphaned "(YYYY)" (the year half of a
# citation split away from its authors) or not ending in terminal punctuation is never a claim.
_ORPHAN_YEAR_START_RE = re.compile(r"^\(\s*(?:19|20)\d{2}[a-z]?\)")
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]['\")\]]*$")
# A sentence about the CITING review's own procedure or corpus: the
# citation there is a methods reference, not a claim about what the cited work itself found.
# The pattern covers first-person-plural ACTIVE verbs and intercoder-reliability phrasing,
# the passive voice ("was assessed using ...", real-test-01), and a broader first-person
# methods verb ("we applied ...", real-test-05). It matches only 5 of 1,157 accepted sentences
# corpus-wide, so it costs almost no recall.
REVIEW_OWN_WORK_RE = re.compile(
    r"\b(?:the (?:present|current) (?:review|study|paper|article|systematic review)|"
    r"our (?:corpus|sample|search|coding|analysis|study|review)|"
    r"in our (?:corpus|sample|search|coding|analysis|study)|"
    r"we (?:conducted|coded|searched|screened|collected)|"
    r"we\s+\w+ed\b|"
    r"(?:was|were)\s+(?:assessed|coded|extracted|screened|rated|scored)\s+"
    r"(?:using|with|following)|"
    r"the included studies|inclusion criteria|data extraction|"
    r"inter-?coder reliability|inter-?rater reliability|"
    r"cohen'?s kappa|kappa of \d)",
    re.IGNORECASE,
)
# A running header or page-furniture fragment glued inside an otherwise plausible sentence:
# "Page 5 of 25", "Downloaded from ...", or a
# journal-style "(YEAR) volume:page" burst such as "(2026) 11:70".
PAGE_FURNITURE_RE = re.compile(
    r"\bPage\s+\d+\s+of\s+\d+\b|\bDownloaded from\b|\(\d{4}\)\s*\d{1,3}:\d{1,4}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationMatch:
    text: str
    surname: str
    surname2: str | None
    connector: str  # "single" | "and" | "et_al"
    year: str


def _citation_from_match(m: re.Match[str]) -> CitationMatch:
    """``CitationMatch`` for one already-established :data:`CITATION_RE` match, dispatching on
    whichever named group fired (mirrors :data:`CITATION_RE`'s own alternation order). Factored
    out of :func:`find_single_citation` so a caller that has
    already established -- by its own means -- that a string carries exactly one recognised
    citation (e.g. :func:`validate_reading_row`, from ``next(CITATION_RE.finditer(claim))``) can
    build the same :class:`CitationMatch` without going through :func:`find_single_citation`'s
    unrelated SELECTION filters (word count, artefact guards, :data:`REVIEW_OWN_WORK_RE`,
    heading/list-item shape, terminal punctuation, ...), which return ``None`` for reasons that
    have nothing to do with which citation the text carries."""
    gd = m.groupdict()
    text = m.group(0)
    if gd.get("p2_s1"):
        return CitationMatch(text, gd["p2_s1"], gd["p2_s2"], "and", gd["p2_year"])
    if gd.get("pe_s1"):
        return CitationMatch(text, gd["pe_s1"], None, "et_al", gd["pe_year"])
    if gd.get("ps_s1"):
        return CitationMatch(text, gd["ps_s1"], None, "single", gd["ps_year"])
    if gd.get("ne3_s1"):
        return CitationMatch(text, gd["ne3_s1"], gd["ne3_s2"], "and", gd["ne3_year"])
    if gd.get("ne2_s1"):
        return CitationMatch(text, gd["ne2_s1"], gd["ne2_s2"], "and", gd["ne2_year"])
    if gd.get("ne2amp_s1"):
        return CitationMatch(text, gd["ne2amp_s1"], gd["ne2amp_s2"], "and", gd["ne2amp_year"])
    if gd.get("ne_s1"):
        return CitationMatch(text, gd["ne_s1"], None, "et_al", gd["ne_year"])
    return CitationMatch(text, gd["ns_s1"], None, "single", gd["ns_year"])


# PDF-extraction artefact guards reused (imported, never copied) from build_hss_set.py's
# is_clean_sentence: a hyphen line-break ("com- pare"), digit-soup ("post .17 45 .183"), a
# footnote mark glued to a word ("the UK.2 Most"), a trailing ellipsis, a reference-list
# fragment bled into the sentence, or page furniture (:data:`PAGE_FURNITURE_RE`), plus its
# alpha-character-ratio floor. ``HYPHEN_BREAK_RE`` is defence in depth here: normal sentences
# have their hyphen/soft-hyphen line breaks *repaired* before this point
# (:func:`clean_extracted_text`), not merely flagged, so this only fires on a case that repair
# missed. ATTRIBUTION_RE is deliberately excluded: it rejects the "Smith (2019) found ..."
# shape this harvester exists to keep, unlike build_hss_set's own (differently purposed) HSS-
# claim extraction.
_ARTEFACT_RES: tuple[re.Pattern[str], ...] = (
    HYPHEN_BREAK_RE, ADJACENT_NUMBERS_RE, FOOTNOTE_MARK_RE, ELLIPSIS_RE, REFERENCE_RE,
    PAGE_FURNITURE_RE,
)


def _is_artefact_free(sentence: str) -> bool:
    """``False`` when *sentence* trips one of :data:`_ARTEFACT_RES` or falls below
    ``build_hss_set.MIN_ALPHA_RATIO``'s alpha-character-ratio floor."""
    if any(pattern.search(sentence) for pattern in _ARTEFACT_RES):
        return False
    letters = sum(1 for ch in sentence if ch.isalpha() or ch == " ")
    return letters / max(len(sentence), 1) >= MIN_ALPHA_RATIO


def find_single_citation(sentence: str) -> CitationMatch | None:
    """The sentence's one recognised citation, or ``None`` when the sentence should be
    dropped: out of the 15-60 word range, opens with See/Cf./For example, is a list item or a
    heading, starts with an orphaned "(YYYY)" fragment, does not end in terminal punctuation,
    is about the citing review's own procedure or corpus (:data:`REVIEW_OWN_WORK_RE`), carries
    a PDF-extraction or page-furniture artefact (:func:`_is_artefact_free`), or does not
    contain exactly one citation of the recognised forms (:data:`CITATION_RE`, guarded by
    :data:`YEAR_TOKEN_RE` so a second, unrecognised-form citation elsewhere in the sentence
    still disqualifies it)."""
    stripped = sentence.strip()
    if not stripped:
        return None
    if LEADING_DROP_RE.match(stripped) or LIST_ITEM_RE.match(stripped) or HEADING_RE.match(
        stripped
    ):
        return None
    if _ORPHAN_YEAR_START_RE.match(stripped):
        return None
    if not _TERMINAL_PUNCT_RE.search(stripped):
        return None
    if _blocks_sentence_break(stripped):
        # ends in "et al."/"e.g."/"i.e."/"cf."/"vs."/a bare initial: a truncated fragment, not
        # a complete sentence, even though its trailing period satisfies the terminal-
        # punctuation check above on its own.
        return None
    if REVIEW_OWN_WORK_RE.search(stripped):
        return None
    if not _is_artefact_free(sentence):
        return None
    words = stripped.split()
    if not (MIN_SENTENCE_WORDS <= len(words) <= MAX_SENTENCE_WORDS):
        return None
    if len(YEAR_TOKEN_RE.findall(stripped)) != 1:
        return None
    matches = list(CITATION_RE.finditer(stripped))
    if len(matches) != 1:
        return None
    return _citation_from_match(matches[0])


# -------------------------------------------------------------------------------------
# Sentence splitting (design item 3): a citation-safe splitter kept local to this module
# rather than changing ``common.split_sentences``, which other evaluation harnesses depend on.
# -------------------------------------------------------------------------------------

_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# "al." / "cf." / "vs." as a WHOLE word (\b keeps this from firing on "verbal." or "versus.");
# "e.g."/"i.e." checked separately below since both already end in a literal period. A page
# (or chapter/section/figure/table) locator abbreviation -- "(p.", "(pp.", "(chap.", "(sec.",
# "(fig.", "(tab." -- is never a sentence end either: the
# generic boundary regex's lookahead class includes digits, so "...processes" (p. 10)." was
# previously cut between "(p." and "10)." (real-test-09's exact shipped defect) because "1" of
# "10)" satisfied the digit lookahead right after "(p."'s own trailing period.
_NO_SPLIT_ABBREV_RE = re.compile(
    r"\bal\.$|\bcf\.$|\bvs\.$|\(p{1,2}\.$|\(chap\.$|\(sec\.$|\(fig\.$|\(tab\.$", re.IGNORECASE
)
# A single capital letter preceded by start-of-string, whitespace, a comma or an opening
# parenthesis ("Smith, J." / "(J. Smith"), never a run inside a longer capitalised word ("USA.").
_NO_SPLIT_INITIAL_RE = re.compile(r"(?:^|[\s,(])[A-Z]\.$")


def _blocks_sentence_break(candidate: str) -> bool:
    """True when *candidate* (the sentence-so-far, up to and including the '.'/'!'/'?' the
    generic boundary regex below would otherwise split on) ends in one of the abbreviations
    the splitter must never break after: 'et al.', 'e.g.', 'i.e.', 'cf.', 'vs.', or a single-
    letter initial. Checked on the rstripped candidate so trailing source whitespace never
    hides a match."""
    s = candidate.rstrip()
    if not s:
        return False
    low = s.lower()
    if low.endswith("e.g.") or low.endswith("i.e."):
        return True
    if _NO_SPLIT_ABBREV_RE.search(low):
        return True
    return bool(_NO_SPLIT_INITIAL_RE.search(s))


def split_sentences_real(text: str) -> list[str]:
    """Sentence splitter for review body text (design item 3): the same period/!/?-plus-
    capital boundary as ``common.split_sentences``, except it never breaks immediately after
    'et al.', 'e.g.', 'i.e.', 'cf.', 'vs.' or a single-letter initial (:func:
    `_blocks_sentence_break`) -- exactly the shapes that would otherwise cut a narrative
    citation like 'Aryadoust et al. (2022)' into an orphaned author fragment and an orphaned
    '(2022)' fragment."""
    normalised = re.sub(r"\s+", " ", text or "").strip()
    if not normalised:
        return []
    sentences: list[str] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(normalised):
        candidate = normalised[start:m.start()]
        if _blocks_sentence_break(candidate):
            continue
        sentences.append(candidate.strip())
        start = m.start()
    tail = normalised[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


# -------------------------------------------------------------------------------------
# Text hygiene (design item 4): PDF zero-width characters stripped and hyphen line breaks
# (ASCII and soft-hyphen U+00AD) undone before any sentence splitting or reference parsing,
# applied to every chunk's text, not only the references chunk: a soft hyphen handled only
# inside the references cleanup would survive, un-repaired, into claim text.
# -------------------------------------------------------------------------------------

# Zero-width Unicode characters some PDF extractors (Springer/Nature house style observed in
# practice) insert inside a reference's own printed DOI/URL, e.g. the literal string
# "https://​doi.​org/​10.​1016/j.​joi.​2017.​09.​007":
# zero-width space, zero-width non-joiner, zero-width joiner, zero-width no-break space (BOM),
# soft hyphen. Left unstripped, ``10.​1016`` never matches _DOI_RE's literal ``10.\d`` and
# every citation for that publisher falls onto the lower-precision Crossref title path.
_ZERO_WIDTH_RE = re.compile("[​‌‍﻿­]")
# A soft-hyphen (U+00AD) PDF line break between two letters, with or without the trailing
# space some extractors also emit ("cor­ roborated" / "pro­ficiency"): joined away, not merely
# stripped, since stripping the character alone still leaves the separating space behind.
_SOFT_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])­\s?([A-Za-z])")
# An ASCII hyphen line-break surviving into the extracted text ("joi-\nnal" from a line
# actually wrapped after "joi-"): keyed on the real newline the PDF extractor left behind, not
# a literal space, since a hyphen followed by a plain space is
# not a line break at all, it is either an ordinary hyphenated compound ("co-citation") or an
# APA suspended hyphenation ("pre- and post-implementation"), and the old space-keyed regex
# wrongly joined the latter into a non-word ("preand post-implementation", real-test-14)
# while never repairing a real hyphen-newline break at all (measured: 8,735 such breaks across
# the 81 cached reviews, none of them repaired). Never joins across the break when the
# following word is one of the suspended-hyphenation connectors (and/or/to/but/nor), so a
# suspended hyphenation that happens to fall exactly at a line wrap is still left alone.
_LINEBREAK_HYPHEN_RE = re.compile(
    r"([A-Za-z])-\s*\n\s*(?!(?:and|or|to|but|nor)\b)([A-Za-z])", re.IGNORECASE
)


def clean_extracted_text(text: str) -> str:
    """*text* with PDF zero-width characters stripped and hyphen line breaks undone (both the
    ASCII hyphen-newline form and the soft-hyphen U+00AD form, the latter with or without a
    trailing space), applied before any sentence splitting or reference-list parsing (design
    item 4). A human author's own claim text is never rejected for carrying this artefact once
    it is repaired here; :data:`_ARTEFACT_RES`'s ``HYPHEN_BREAK_RE`` stays as defence in depth
    for a shape this cleanup misses."""
    cleaned = _SOFT_HYPHEN_BREAK_RE.sub(r"\1\2", text or "")
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", cleaned)
    return cleaned


# A physical-line heading guard here (``_drop_heading_lines``/``_looks_like_heading_line``)
# once dropped any short, unpunctuated line followed by a line starting with a capital
# letter, on the theory that this shape is a section heading glued to the sentence that
# follows it. Measured over 81 cached reviews this deleted 18,378 of 94,022 physical lines
# (19.5 percent), 16,356 of them ordinary mid-sentence wrapped continuations (the "next line starts
# with a capital" condition fires constantly on proper nouns, not just headings), and spliced
# the surviving fragments into sentences that never existed in the source -- worse than the
# glued-heading defect it replaced, because the result reads as a plausible claim instead of
# an obviously broken one. Removed outright rather than repaired: this module's own
# regex-extraction path (:func:`extract_candidates`) no longer feeds the shipped build (see
# ``--from-readings``/:func:`build_review_cache_from_readings` below and the module
# docstring), so the same-line heading vocabulary (:data:`HEADING_RE`, inside
# :func:`find_single_citation`) is the only heading guard left. It is safe on its own: it
# only ever drops a heading fused to the START of the very sentence being evaluated, never a
# physically separate body line the sentence never actually contained.


def find_references_chunk(chunks: Sequence[Mapping[str, Any]]) -> str | None:
    """Text of the chunk whose ``section`` is exactly ``"references"`` (``fulltext.
    detect_sections`` names a section by its heading line, lower-cased), or ``None`` when no
    such chunk exists (e.g. the PDF's reference-list heading was not detected as its own
    line)."""
    for chunk in chunks or []:
        if str(chunk.get("section", "")).strip().casefold() == "references":
            return str(chunk.get("text") or "")
    return None


# -------------------------------------------------------------------------------------
# Reference-list parsing and matching (design item 3)
# -------------------------------------------------------------------------------------

_REF_HEADING_RE = re.compile(r"^references$", re.IGNORECASE)
# An entry-start line: "Surname, I." / "Surname, I. I." / "Surname, I. I. I.", 1-3 initials
# (APA caps a citation's shown authors, never the reference list itself, but every entry still
# starts with exactly one leading surname + initial(s) before the next author or the year).
_REF_START_RE = re.compile(rf"^{_NAME},\s?(?:[A-Z]\.\s?){{1,3}}")
_REF_YEAR_RE = re.compile(r"\((?P<year>(?:19|20)\d{2}[a-z]?)\)")
_REF_SURNAME_RE = re.compile(rf"^(?P<surname>{_NAME}),")
_DOI_RE = re.compile(
    r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(?P<doi>10\.\d{4,9}/\S+)", re.IGNORECASE
)


def split_reference_lines(text: str) -> list[str]:
    """One string per reference entry: a physical line that does not itself start a new entry
    (:data:`_REF_START_RE`) is a wrapped continuation of the previous one, merged with a
    space. The literal "References" heading line is dropped."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln and not _REF_HEADING_RE.match(ln)]
    entries: list[str] = []
    for ln in lines:
        if entries and not _REF_START_RE.match(ln):
            entries[-1] = f"{entries[-1]} {ln}"
        else:
            entries.append(ln)
    return entries


def parse_reference_entry(raw: str) -> dict[str, Any] | None:
    """``{surname, year, year_raw, title, doi, raw}`` for one merged entry string, or ``None``
    when it does not even carry a leading surname and a parenthesised year (not a reference
    line at all -- e.g. a running header wrapped into the references chunk)."""
    surname_m = _REF_SURNAME_RE.match(raw)
    year_m = _REF_YEAR_RE.search(raw)
    if surname_m is None or year_m is None:
        return None
    after_year = raw[year_m.end():].lstrip(". ")
    title_m = re.match(r"([^.]+)\.", after_year)
    title = title_m.group(1).strip() if title_m else None
    doi_m = _DOI_RE.search(raw)
    doi = doi_m.group("doi").rstrip(").,;") if doi_m else None
    year_raw = year_m.group("year")
    return {
        "surname": surname_m.group("surname"),
        "year": year_raw[:4],
        "year_raw": year_raw,
        "title": title,
        "doi": doi,
        "raw": raw,
    }


def parse_reference_entries(text: str) -> list[dict[str, Any]]:
    out = []
    for raw in split_reference_lines(clean_extracted_text(text)):
        entry = parse_reference_entry(raw)
        if entry is not None:
            out.append(entry)
    return out


def normalize_surname(name: str) -> str:
    """Case- and diacritic-insensitive comparison key (mirrors
    ``backend/app/services/citation_audit.py``'s ``_normalize``, reimplemented locally rather
    than imported: that module is a general APA-citation *auditor*, not owned by this
    script)."""
    decomposed = unicodedata.normalize("NFKD", name or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold().strip()


def match_reference_entry(
    surname: str, surname2: str | None, year: str, entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """The one reference entry whose leading surname and year match, or ``None`` when zero or
    more than one entry matches (an ambiguous match, e.g. "2019a"/"2019b" by the same author
    cited without a year suffix, is never guessed at). When the citation carries a *second*
    surname (a two-author or three-author narrative/parenthetical form), the matched entry's
    own raw text must contain it too: a wrapped
    or non-APA entry that happens to share first-surname-plus-year with an unrelated entry is
    otherwise silently accepted. An "et al." citation carries no second surname, so this check
    is skipped for it -- unchanged behaviour."""
    key = normalize_surname(surname)
    matches = [
        e for e in entries
        if e and normalize_surname(str(e.get("surname") or "")) == key
        and str(e.get("year") or "")[:4] == year[:4]
    ]
    if len(matches) != 1:
        return None
    match = matches[0]
    if surname2:
        key2 = normalize_surname(surname2)
        if key2 not in normalize_surname(str(match.get("raw") or "")):
            return None
    return match


# -------------------------------------------------------------------------------------
# Crossref fallback (design item 3: no printed DOI -> title similarity) and OpenAlex
# authorship enrichment (design item 3: the cited work's authors come from OpenAlex, not
# Crossref, which does not always carry an author list).
# -------------------------------------------------------------------------------------


def crossref_title_query_url(title: str, mailto: str, rows: int = 5) -> str:
    """``select`` carries ``author`` because the narrower
    ``DOI,title,type`` selection would structurally guarantee an empty author list for every
    candidate this query could ever return, since :func:`_crossref_authors` reads
    ``message["author"]``, a field Crossref never includes unless it is explicitly selected.
    ``issued`` is selected so
    :func:`crossref_search_to_candidates` can carry each candidate's own publication year
    through for :func:`best_crossref_match`'s year guard."""
    return (
        f"{CROSSREF_WORKS}?query.bibliographic={quote(title)}&rows={rows}"
        f"&select=DOI,title,type,author,issued&mailto={mailto}"
    )


def crossref_doi_url(doi: str, mailto: str) -> str:
    return f"{CROSSREF_WORKS}/{doi}?mailto={mailto}"


def _crossref_authors(message: Mapping[str, Any]) -> list[str]:
    """``["Given Family", ...]`` from a Crossref ``author`` list (same convention as
    ``build_hss_set.crossref_to_works``: joined given+family, blanks dropped, reimplemented
    locally rather than imported since that function returns a whole OpenAlex-shaped work
    dict for a different caller, not just a name list)."""
    return [
        " ".join(x for x in (a.get("given"), a.get("family")) if x)
        for a in message.get("author") or []
        if a.get("given") or a.get("family")
    ]


def _crossref_issued_year(item: Mapping[str, Any]) -> str | None:
    """The first ``date-parts`` year of a Crossref ``issued`` field (``{"issued":
    {"date-parts": [[2022, 10, 1]]}}``), or ``None`` when the item carries no usable issued
    date. Needed so :func:`best_crossref_match` can guard against a same-subject,
    wrong-year candidate."""
    parts = (item.get("issued") or {}).get("date-parts") or []
    first = parts[0] if parts else None
    year = first[0] if first else None
    return str(year) if year else None


def crossref_search_to_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for it in (payload.get("message") or {}).get("items") or []:
        title = (it.get("title") or [None])[0]
        doi = it.get("DOI")
        if not title or not doi:
            continue
        out.append({"doi": doi, "title": title, "type": it.get("type"),
                    "authors": _crossref_authors(it), "year": _crossref_issued_year(it)})
    return out


def best_crossref_match(
    entry_title: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    entry_year: str | None = None,
    threshold: float = CROSSREF_TITLE_SIMILARITY_THRESHOLD,
) -> dict[str, Any] | None:
    """Highest token-Jaccard title match at or above *threshold*, or ``None``. When
    *entry_year* is given (the reference entry's own printed year), a candidate whose own
    ``year`` is more than :data:`CROSSREF_YEAR_TOLERANCE` years away is skipped outright,
    regardless of title similarity -- a candidate with no ``year`` of
    its own is not skipped by this guard, since no comparison is then possible either way. A
    caller that never passes *entry_year* (any pre-existing caller) keeps matching on title
    similarity alone, unchanged."""
    best: Mapping[str, Any] | None = None
    best_score = 0.0
    for c in candidates:
        title = c.get("title")
        if not title:
            continue
        if entry_year and c.get("year"):
            try:
                year_gap = abs(int(str(c["year"])[:4]) - int(str(entry_year)[:4]))
            except ValueError:
                year_gap = 0
            if year_gap > CROSSREF_YEAR_TOLERANCE:
                continue
        score = token_jaccard(entry_title, str(title))
        if score > best_score:
            best_score, best = score, c
    if best is not None and best_score >= threshold:
        return {**best, "similarity": best_score}
    return None


async def crossref_lookup_by_title(title: str, mailto: str) -> list[dict[str, Any]]:
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        resp = await client.get(crossref_title_query_url(title, mailto))
        if resp.status_code != 200:
            return []
        return crossref_search_to_candidates(resp.json())


async def crossref_lookup_by_doi(doi: str, mailto: str) -> dict[str, Any] | None:
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        resp = await client.get(crossref_doi_url(doi, mailto))
        if resp.status_code != 200:
            return None
        message = (resp.json() or {}).get("message") or {}
        title = (message.get("title") or [None])[0]
        return {
            "doi": message.get("DOI"), "title": title, "type": message.get("type"),
            "authors": _crossref_authors(message),
        }


async def openalex_work_by_doi(doi: str, mailto: str) -> dict[str, Any] | None:
    """The OpenAlex work record for *doi* (used only for its ``authorships``), or ``None`` on
    any non-200 response -- a lookup failure simply leaves the cited work's authors empty
    rather than aborting resolution."""
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        resp = await client.get(openalex_work_by_doi_url(doi, mailto))
        if resp.status_code != 200:
            return None
        return resp.json()


async def resolve_citation(
    entry: Mapping[str, Any], *, mailto: str
) -> dict[str, Any] | None:
    """``{doi, title, year, type, resolution, similarity, authors}`` for one reference entry:
    its own printed DOI (verified against Crossref's title for that DOI) when present, else the
    best Crossref title match; ``None`` when neither yields a usable, title-verified DOI
    (design item 3). The printed-DOI path is not accepted unchecked: when Crossref returns a
    title for *doi* and the entry itself was parsed with a title, the two are compared
    (:func:`common.token_jaccard`) and the DOI is rejected below
    :data:`CROSSREF_TITLE_SIMILARITY_THRESHOLD` -- a bled-together or mis-OCRed DOI (two
    reference entries merged into one, see :func:`split_reference_lines`) then points Crossref
    at a different work than the one the entry's own title describes, and is caught here rather
    than accepted on the strength of the DOI alone. Neither title is available (a Crossref
    lookup failure, or no title parsed from the entry) -> no comparison is possible, so the
    printed DOI is accepted as before (best effort, not a regression from the prior behaviour).
    OpenAlex authorship enrichment is deferred to :func:`extract_candidates` (spent only on a
    candidate that is actually kept, design item 6), but Crossref's own ``author`` list -- free,
    already fetched for the title check above -- is carried through here as an ``"authors"``
    fallback for when OpenAlex has nothing (over budget, not indexed, ...): the alternative is
    shipping an empty author list, which is a defect. A resolved title containing a Unicode
    replacement character (U+FFFD) is rejected
    outright (real-run finding, 2026-09-07: Crossref returned one title with a genuinely
    malformed byte sequence for a Latin-script dash, decoded by httpx as U+FFFD) -- a corrupted
    title is not accepted on the strength of a DOI or a similarity score alone. The
    crossref-title path also passes the entry's own printed year through to
    :func:`best_crossref_match`: a same-subject, same-authors
    candidate from a different year is rejected
    there before the raised :data:`CROSSREF_TITLE_SIMILARITY_THRESHOLD` (0.8) is even
    consulted."""
    doi = entry.get("doi")
    if doi:
        work = await crossref_lookup_by_doi(doi, mailto)
        work_title = (work or {}).get("title")
        entry_title = entry.get("title")
        similarity = None
        if work_title and entry_title:
            similarity = token_jaccard(entry_title, str(work_title))
            if similarity < CROSSREF_TITLE_SIMILARITY_THRESHOLD:
                return None
        title = work_title or entry_title
        if title and "�" in title:
            return None
        wtype = (work or {}).get("type")
        return {
            "doi": _bare_doi(doi), "title": title, "year": entry.get("year"), "type": wtype,
            "resolution": "printed_doi", "similarity": similarity,
            "authors": (work or {}).get("authors") or [],
        }
    title = entry.get("title")
    if not title:
        return None
    candidates = await crossref_lookup_by_title(title, mailto)
    match = best_crossref_match(title, candidates, entry_year=entry.get("year"))
    if match is None:
        return None
    match_title = match.get("title")
    if match_title and "�" in match_title:
        return None
    return {
        "doi": _bare_doi(match.get("doi")), "title": match_title,
        "year": entry.get("year"), "type": match.get("type"), "resolution": "crossref_title",
        "similarity": match.get("similarity"), "authors": match.get("authors") or [],
    }


# -------------------------------------------------------------------------------------
# Acquisition-outcome bucketing: reuses build_hss_set.acquire_one's own reason vocabulary
# -------------------------------------------------------------------------------------


def acquire_outcome_buckets(reason: str | None) -> dict[str, int]:
    """``{"oa", "fetched", "kept"}`` 0/1 increments for one ``acquire_one`` outcome, from its
    ``reason`` (``None`` on success): ``"not_oa"`` fails every bucket; ``"html_not_pdf"``
    passed the OA lookup but the fetched bytes were not a PDF; ``"min_chars"``/``"min_chunks"``
    passed OA and a successful PDF fetch+extraction but the text was too short/thin;
    ``"timeout_budget_exceeded"`` and any other reason (an exception, ``"http_403"``, ...)
    reached the fetch attempt after a successful OA lookup but failed there, so it counts as
    ``oa`` only, exactly like ``html_not_pdf``."""
    if reason == "not_oa":
        return {"oa": 0, "fetched": 0, "kept": 0}
    if reason == "html_not_pdf":
        return {"oa": 1, "fetched": 0, "kept": 0}
    if reason in ("min_chars", "min_chunks"):
        return {"oa": 1, "fetched": 1, "kept": 0}
    if reason is None:
        return {"oa": 1, "fetched": 1, "kept": 1}
    return {"oa": 1, "fetched": 0, "kept": 0}


# -------------------------------------------------------------------------------------
# Per-candidate wall-clock budget (design item 6)
# -------------------------------------------------------------------------------------


async def acquire_one_with_budget(
    work: Mapping[str, Any], pipeline: Mapping[str, Any], *, email: str, min_chars: int,
    min_chunks: int, budget_s: float = PER_CANDIDATE_BUDGET_S,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """:func:`build_hss_set.acquire_one`, wrapped in a *budget_s*-second wall-clock budget: a
    fetch or PDF-extraction step that hangs inside the reused pipeline (observed on
    Windows/httpx for one real candidate during a prior real run of this harvester) is turned
    into a recorded ``"timeout_budget_exceeded"`` skip instead of blocking the whole harvest
    indefinitely. ``asyncio.wait_for`` cancels the inner call on timeout, so no task is left
    running in the background. Any OTHER exception (a real run hit a ``UnicodeEncodeError``
    from a ``print()`` inside the reused pipeline meeting a non-ASCII character on a non-UTF-8
    Windows console codepage) is caught the same way: one candidate's failure must never
    propagate through ``asyncio.gather`` and abort every other review already in flight."""
    try:
        return await asyncio.wait_for(
            acquire_one(work, pipeline, email=email, min_chars=min_chars, min_chunks=min_chunks),
            timeout=budget_s,
        )
    except TimeoutError:
        return None, None, "timeout_budget_exceeded"
    except Exception as exc:  # noqa: BLE001 - one candidate's failure must not abort the harvest
        return None, None, f"{type(exc).__name__}: {exc}"


# -------------------------------------------------------------------------------------
# Deterministic sampling (design item 5)
# -------------------------------------------------------------------------------------


def shuffled_review_order(review_dois: Sequence[str], seed: int) -> list[str]:
    """Sorted (so the result never depends on caller iteration order), then shuffled with a
    seeded ``random.Random`` -- deterministic and reproducible for a given *seed*."""
    order = sorted(set(review_dois))
    random.Random(seed).shuffle(order)
    return order


def split_items_by_review(
    review_order: Sequence[str],
    items_by_review: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    n_dev: int,
    n_test: int,
    max_per_review: int = DEFAULT_MAX_PER_REVIEW,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Greedily assign each review's (capped at *max_per_review*) items to **test first, then
    dev**, in *review_order*: a review contributes to exactly one split, never both, so dev and
    test cite disjoint reviews. Test is filled first (design item 5, fixing a dev-first split
    that previously left the held-out test set starved whenever total yield was below the dev
    target) so the frozen test set this harvester exists to produce is never empty just
    because dev happened to be filled first. A review reached after both targets are already
    met contributes nothing (its DOI is simply absent from the returned ``review_split``)."""
    dev_items: list[dict[str, Any]] = []
    test_items: list[dict[str, Any]] = []
    review_split: dict[str, str] = {}
    for doi in review_order:
        items = list(items_by_review.get(doi) or [])[:max_per_review]
        if not items:
            continue
        if len(test_items) < n_test:
            test_items.extend(items)
            review_split[doi] = "test"
        elif len(dev_items) < n_dev:
            dev_items.extend(items)
            review_split[doi] = "dev"
    return dev_items, test_items, review_split


# -------------------------------------------------------------------------------------
# Item assembly (design item 5 / the run_hss.py contract)
# -------------------------------------------------------------------------------------


def build_item_row(
    *,
    sentence_info: Mapping[str, Any],
    review: Mapping[str, Any],
    cited: Mapping[str, Any],
) -> dict[str, Any]:
    """One output row, without ``item_id`` (assigned by the caller once dev/test order is
    known). Carries both this script's own descriptive fields (``citation_text``,
    ``cited_doi``, ``cited_title``, ``cited_year``, ``review_doi``, ``review_title``,
    ``context_before``) and ``run_hss.py``'s ``ITEM_KEYS`` fields (``rule``, ``expected``,
    ``claim``, ``source_doi``, ``chunk_doi``, ``chunk_index``, ``alteration``) plus
    ``title``/``authors`` -- ``chunk_doi``/``title``/``authors`` describe the *cited* work (the
    paper whose chunks ``run_hss.py`` hands to the verifier), ``source_doi`` the review.
    ``extraction`` is copied straight through from
    *sentence_info* -- :func:`build_outputs` is the one that resolves a candidate lacking its
    own ``extraction`` key to ``"regex"`` before calling this function, so every row this
    function returns names the path (``"regex"`` or ``"readings"``) that actually produced
    it."""
    cited_doi = sentence_info["cited_doi"]
    review_doi = review.get("doi")
    return {
        "rule": "real",
        "expected": None,
        "claim": sentence_info["sentence"],
        "citation_text": sentence_info.get("citation_text"),
        "context_before": sentence_info.get("context_before") or "",
        "cited_doi": cited_doi,
        "cited_title": cited.get("title") or sentence_info.get("cited_title"),
        "cited_year": sentence_info.get("cited_year"),
        "review_doi": review_doi,
        "review_title": review.get("title"),
        "chunk_doi": cited_doi,
        "chunk_index": None,
        "alteration": None,
        "source_doi": review_doi,
        "title": cited.get("title") or sentence_info.get("cited_title"),
        "authors": cited.get("authors"),
        "resolution": sentence_info.get("resolution"),
        "similarity": sentence_info.get("similarity"),
        "extraction": sentence_info.get("extraction"),
    }


def _candidate_extraction(payload: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    """The path (``"regex"`` or ``"readings"``) that produced *candidate*: its own
    ``extraction`` field when stamped (:func:`_process_review`,
    :func:`build_review_cache_from_readings`), else the review payload's own -- a cache file
    predating this stamp (payload and candidate both missing the key) is treated as
    ``"regex"``, the only producer that existed before ``--from-readings``, so a
    ``--skip-fetch`` rebuild over an un-stamped cache neither crashes nor silently mislabels
    its own items as ``"readings"``."""
    return str(candidate.get("extraction") or payload.get("extraction") or "regex")


def build_outputs(
    reviews_cache_dir: Path,
    fulltext_cache_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    n_dev: int = DEFAULT_N_DEV,
    n_test: int = DEFAULT_N_TEST,
    max_per_review: int = DEFAULT_MAX_PER_REVIEW,
    excluded: set[str] | None = None,
    extraction: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pure rebuild of ``(dev_items, test_items, build_info)`` from the two local caches --
    the ``--skip-fetch`` path, and the second half of a normal (fetching) run. A review-cache
    payload's ``candidates`` (already resolved and, where a fetch succeeded, cached under
    *fulltext_cache_dir*) are filtered again here to the cited DOI actually having a fulltext
    cache entry, not being *excluded* and not being the review itself (defensive: a normal run
    already enforces this before caching a candidate, so this only matters for hand-edited or
    partial caches), then capped at *max_per_review*. Reviews with at least one usable
    candidate are shuffled (:func:`shuffled_review_order`) and split
    (:func:`split_items_by_review`, test first); every ``stats`` sub-dict of every cached
    review is summed into the build info regardless of which reviews ended up contributing
    items, so the funnel counts describe every review this harvester has ever fetched, not
    just the sampled ones.

    *fulltext_cache_dir*'s cache is passed through
    :func:`build_hss_set.check_cache_identity` before it is used for anything, the same
    wrong-work identity check :func:`build_hss_set.build_items_from_cache` already applies by
    default, so a cited work whose own recorded title/authors do not match its own fetched
    text is dropped here regardless of which acquisition path (or which prior run's
    ``was_cached`` reuse) put it in the cache; ``info["wrong_work_sources"]`` records what was
    dropped and why, mirroring that function's own field.

    *extraction* (``"regex"`` | ``"readings"`` | ``None``),
    when given, keeps only candidates :func:`_candidate_extraction` resolves to that value,
    so a ``--from-readings`` run's own cache directory, which may still hold candidates a prior
    regex-extraction run already cached there, never silently contributes regex-path items to
    a build meant to be reader-extracted evidence. ``info["n_items_by_extraction"]`` always
    reports the FULL usable pool's breakdown by path, regardless of *extraction*, so a filtered
    build still discloses how many candidates of the OTHER path exist in the same cache
    directory.
    """
    excluded = excluded or set()
    reviews = load_cache(reviews_cache_dir)
    fulltext = load_cache(fulltext_cache_dir)
    fulltext, wrong_work_sources = check_cache_identity(fulltext)
    stats_totals = dict.fromkeys(
        ("n_sentences_seen", "n_single_citation", "n_resolved", "n_oa", "n_fetched", "n_kept"),
        0,
    )
    items_by_review: dict[str, list[dict[str, Any]]] = {}
    per_review: dict[str, dict[str, Any]] = {}
    extraction_counts: Counter[str] = Counter()
    for doi, payload in reviews.items():
        stats = payload.get("stats") or {}
        for key in stats_totals:
            stats_totals[key] += int(stats.get(key) or 0)
        raw_candidates = payload.get("candidates") or []
        usable_all = [
            c for c in raw_candidates
            if c.get("cited_doi") in fulltext
            and c["cited_doi"] not in excluded
            and c["cited_doi"] != doi
        ]
        for c in usable_all:
            extraction_counts[_candidate_extraction(payload, c)] += 1
        usable = usable_all if extraction is None else [
            c for c in usable_all if _candidate_extraction(payload, c) == extraction
        ]
        usable = usable[:max_per_review]
        per_review[doi] = {
            "title": payload.get("title"),
            "n_candidates": len(raw_candidates),
            "n_used": len(usable),
        }
        if usable:
            items_by_review[doi] = [
                build_item_row(
                    sentence_info={**c, "extraction": _candidate_extraction(payload, c)},
                    review=payload, cited=fulltext[c["cited_doi"]],
                )
                for c in usable
            ]
    order = shuffled_review_order(list(items_by_review), seed)
    dev_raw, test_raw, review_split = split_items_by_review(
        order, items_by_review, n_dev=n_dev, n_test=n_test, max_per_review=max_per_review,
    )
    dev_items = [{"item_id": f"real-dev-{i:02d}", **row} for i, row in enumerate(dev_raw, 1)]
    test_items = [{"item_id": f"real-test-{i:02d}", **row} for i, row in enumerate(test_raw, 1)]
    info = {
        "seed": seed,
        "n_dev_target": n_dev,
        "n_test_target": n_test,
        "n_dev": len(dev_items),
        "n_test": len(test_items),
        "max_per_review": max_per_review,
        "n_reviews_cached": len(reviews),
        "n_reviews_with_items": len(items_by_review),
        **stats_totals,
        "per_review": per_review,
        "review_split": review_split,
        "excluded_dois": sorted(excluded),
        "n_excluded": len(excluded),
        "wrong_work_sources": wrong_work_sources,
        "n_wrong_work_sources": len(wrong_work_sources),
        "extraction_filter": extraction,
        "n_items_by_extraction": dict(extraction_counts),
    }
    return dev_items, test_items, info


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_item_id_cited_dois(
    item_ids: Sequence[str], *paths: Path
) -> dict[str, str | None]:
    """``{item_id: cited_doi}`` for *item_ids* found in *paths* (each an existing-contract
    jsonl file, one JSON object per line with ``item_id``/``cited_doi``), used by
    ``--exclude-item-ids`` to translate a vetoed item from a *previous* run's output into the
    DOI-exclusion mechanism ``build_outputs`` already has (``excluded``): neither a
    ``--veto-ids`` flag nor an item-id-keyed exclusion field exists anywhere in this build
    json, and this script has no other durable identity for a harvested item across a rebuild
    (``item_id`` itself is just its position in the shuffled dev/test split, reassigned every
    run), so the smallest correct fix is to resolve the id against the file(s) it was minted
    into before those files are overwritten by this same run and exclude its ``cited_doi``
    (the same DOI-set mechanism the source-list files already use) -- consistent with
    ``extract_candidates``'s own ``seen_cited_dois`` rule that a given cited work contributes
    at most one candidate per review. A *path* that does not exist yet is skipped without
    error; an id absent from every path is simply absent from the result (the caller decides
    whether that is an error)."""
    wanted = set(item_ids)
    found: dict[str, str | None] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("item_id")
            if iid in wanted and iid not in found:
                found[iid] = row.get("cited_doi")
    return found


# -------------------------------------------------------------------------------------
# Acquisition (network; no LLM). Not unit tested directly (mirrors build_hss_set.py's own
# acquisition functions, exercised only at real run time), except for the pure URL builders,
# filters and the cursor-pagination helper above/below.
# -------------------------------------------------------------------------------------


async def resolve_topic_ids(names: Sequence[str], mailto: str) -> dict[str, str]:
    """``{display_name: topic_id}`` for the first OpenAlex Topics search hit per name; a name
    with no hits is simply absent (not an error -- the caller then queries one fewer topic)."""
    import httpx

    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        for name in names:
            resp = await client.get(topic_search_url(name, mailto))
            if resp.status_code != 200:
                continue
            results = resp.json().get("results") or []
            if results:
                out[name] = str(results[0]["id"]).rsplit("/", 1)[-1]
    return out


async def resolve_subfield_ids(names: Sequence[str], mailto: str) -> dict[str, str]:
    """``{display_name: subfield_id}``, the OpenAlex Subfields API equivalent of
    :func:`resolve_topic_ids`."""
    import httpx

    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        for name in names:
            resp = await client.get(subfield_search_url(name, mailto))
            if resp.status_code != 200:
                continue
            results = resp.json().get("results") or []
            if results:
                out[name] = str(results[0]["id"]).rsplit("/", 1)[-1]
    return out


async def resolve_source_ids(
    journals: Sequence[Mapping[str, str]], mailto: str
) -> dict[str, str]:
    """``{journal_name: openalex_source_id}``: a :data:`JOURNALS` entry that already carries a
    non-empty ``openalex`` id (most of them, verified at ``build_hss_set``'s v3 addition time)
    is reused without a network call; the rest are resolved here via the OpenAlex Sources API
    by ISSN. A journal that resolves to nothing is simply absent (that channel then
    contributes zero candidates for it, rather than failing the whole discovery run)."""
    out: dict[str, str] = {}
    to_resolve = []
    for j in journals:
        sid = (j.get("openalex") or "").strip()
        if sid:
            out[j["name"]] = sid
        else:
            to_resolve.append(j)
    if not to_resolve:
        return out
    import httpx

    async with httpx.AsyncClient(timeout=_http_timeout()) as client:
        for j in to_resolve:
            resp = await client.get(source_lookup_url(j["issn"], mailto))
            if resp.status_code != 200:
                continue
            results = resp.json().get("results") or []
            if results:
                out[j["name"]] = str(results[0]["id"]).rsplit("/", 1)[-1]
    return out


async def openalex_cursor_pages(
    client: Any,
    url_for_cursor: Callable[[str], str],
    *,
    max_pages: int = DEFAULT_MAX_PAGES_PER_CHANNEL,
) -> list[dict[str, Any]]:
    """All ``results`` across up to *max_pages* cursor-paginated OpenAlex works pages (design
    item 1): *url_for_cursor* builds the request URL for a given cursor value, starting at
    ``"*"`` and following each page's ``meta.next_cursor`` until it is falsy or *max_pages* is
    reached. *client* only needs an async ``get(url) -> response`` with ``.status_code`` and
    ``.json()`` -- a real ``httpx.AsyncClient`` in production, a small fake in tests, so this
    function's own pagination logic is exercised with no real network client involved. A
    non-200 response ends pagination for this channel (its own results so far are kept) rather
    than raising, so one channel's failure never aborts discovery."""
    out: list[dict[str, Any]] = []
    cursor = "*"
    for _ in range(max_pages):
        resp = await client.get(url_for_cursor(cursor))
        if resp.status_code != 200:
            break
        payload = resp.json()
        out.extend(payload.get("results") or [])
        cursor = (payload.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    return out


async def list_review_candidates_by_source(
    client: Any, journal: Mapping[str, str], source_id: str, mailto: str
) -> list[dict[str, Any]]:
    results = await openalex_cursor_pages(
        client, lambda cursor: openalex_journal_query_url(source_id, mailto, cursor=cursor),
    )
    return [
        {**w, "doi": _bare_doi(w.get("doi")), "journal": journal["name"],
         "issn": journal.get("issn"),
         "selected_from": public_url(openalex_journal_query_url(source_id, mailto))}
        for w in results if w.get("doi")
    ]


async def list_review_candidates_by_topic(
    client: Any, topic_id: str, topic_name: str, mailto: str
) -> list[dict[str, Any]]:
    results = await openalex_cursor_pages(
        client, lambda cursor: openalex_topic_query_url(topic_id, mailto, cursor=cursor),
    )
    return [
        {**w, "doi": _bare_doi(w.get("doi")), "journal": f"(topic: {topic_name})", "issn": None,
         "selected_from": public_url(openalex_topic_query_url(topic_id, mailto))}
        for w in results if w.get("doi")
    ]


async def list_review_candidates_by_subfield(
    client: Any, subfield_id: str, subfield_name: str, mailto: str
) -> list[dict[str, Any]]:
    results = await openalex_cursor_pages(
        client, lambda cursor: openalex_subfield_query_url(subfield_id, mailto, cursor=cursor),
    )
    return [
        {**w, "doi": _bare_doi(w.get("doi")), "journal": f"(subfield: {subfield_name})",
         "issn": None,
         "selected_from": public_url(openalex_subfield_query_url(subfield_id, mailto))}
        for w in results if w.get("doi")
    ]


async def list_review_candidates_by_keyword(
    client: Any, keyword: str, subfield_id: str, subfield_name: str, mailto: str
) -> list[dict[str, Any]]:
    results = await openalex_cursor_pages(
        client,
        lambda cursor: openalex_title_keyword_query_url(
            keyword, subfield_id, mailto, cursor=cursor,
        ),
    )
    label = f"(title keyword: {keyword!r} in {subfield_name})"
    selected_from = public_url(openalex_title_keyword_query_url(keyword, subfield_id, mailto))
    return [
        {**w, "doi": _bare_doi(w.get("doi")), "journal": label, "issn": None,
         "selected_from": selected_from}
        for w in results if w.get("doi")
    ]


async def discover_reviews(
    mailto: str,
    *,
    max_reviews: int = DEFAULT_MAX_REVIEWS,
    journals: Sequence[Mapping[str, str]] = JOURNALS,
    topic_names: Sequence[str] = TOPIC_NAMES,
    subfield_names: Sequence[str] = SUBFIELD_NAMES,
    title_keywords: Sequence[str] = TITLE_KEYWORDS,
) -> list[dict[str, Any]]:
    """Deduplicated, DOI-sorted, review-filtered, OA+PDF-filtered candidate works across the
    four discovery channels (design item 1), capped at *max_reviews*."""
    import httpx

    seen: dict[str, dict[str, Any]] = {}

    def _add(works: Sequence[Mapping[str, Any]]) -> None:
        for w in works:
            if w["doi"] not in seen and is_review_work(w) and has_pdf_location(w):
                seen[w["doi"]] = dict(w)

    async with httpx.AsyncClient(timeout=_http_timeout(), follow_redirects=True) as client:
        source_ids = await resolve_source_ids(journals, mailto)
        for journal in journals:
            source_id = source_ids.get(journal["name"])
            if not source_id:
                print(f"  no OpenAlex source id resolved for {journal['name']}; skipping")
                continue
            try:
                works = await list_review_candidates_by_source(
                    client, journal, source_id, mailto,
                )
            except Exception as exc:  # noqa: BLE001 - one channel's failure must not abort the rest
                print(f"  OpenAlex journal query failed for {journal['name']}: {exc}")
                works = []
            _add(works)
        topic_ids = await resolve_topic_ids(topic_names, mailto)
        for name, topic_id in topic_ids.items():
            try:
                works = await list_review_candidates_by_topic(client, topic_id, name, mailto)
            except Exception as exc:  # noqa: BLE001
                print(f"  OpenAlex topic query failed for {name}: {exc}")
                works = []
            _add(works)
        subfield_ids = await resolve_subfield_ids(subfield_names, mailto)
        for name, subfield_id in subfield_ids.items():
            try:
                works = await list_review_candidates_by_subfield(
                    client, subfield_id, name, mailto,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  OpenAlex subfield query failed for {name}: {exc}")
                works = []
            _add(works)
        # Design item 1c: title-keyword matches are scoped to the same subfields as channel
        # (b) above (see openalex_title_keyword_query_url's docstring for why an unscoped
        # search crowds out every on-topic candidate with off-topic reviews from every field).
        for name, subfield_id in subfield_ids.items():
            for keyword in title_keywords:
                try:
                    works = await list_review_candidates_by_keyword(
                        client, keyword, subfield_id, name, mailto,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  OpenAlex title-keyword query failed for {keyword!r} in {name}: {exc}")
                    works = []
                _add(works)
    ordered = sorted(seen.values(), key=lambda w: w["doi"])
    return ordered[:max_reviews]


async def extract_candidates(
    review_payload: Mapping[str, Any],
    *,
    mailto: str,
    pipeline: Mapping[str, Any],
    excluded: set[str],
    fulltext_cache_dir: Path,
    min_chars: int,
    min_chunks: int,
    review_doi: str,
    max_per_review: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One review's resolved, fetched candidates plus its filter-funnel ``stats`` (design item
    2-3). Stops attempting new fetches once *max_per_review* candidates are kept (still counts
    every sentence seen and every citation resolved above that cap, so the funnel reflects the
    whole review, not just the part that was worth fetching). Always recomputed from the
    review's cached chunks (design item 6): a fix to this module's own extraction logic
    therefore applies to an already-fetched review without needing to redownload its PDF. The
    review's own "references" chunk is never scanned for claims,
    it is only parsed for entries, above.

    The cited work's OpenAlex authorships (design item 3) are fetched only AFTER
    :func:`acquire_one_with_budget` actually succeeds for a NEW (not-already-cached) cited
    work, never for every ``resolve_citation`` result: a real run resolved 467 citations but
    kept only 28, and fetching OpenAlex authorships eagerly for all 467 exhausted OpenAlex's
    per-request daily budget in a single run, well short of covering the discovery pool this
    module can otherwise reach. An already-cached cited work's authors are trusted as already
    correct from when it was first fetched -- no OpenAlex call on that path either."""
    chunks = review_payload.get("chunks") or []
    references = parse_reference_entries(find_references_chunk(chunks) or "")
    stats = dict.fromkeys(
        ("n_sentences_seen", "n_single_citation", "n_resolved", "n_oa", "n_fetched", "n_kept"),
        0,
    )
    kept: list[dict[str, Any]] = []
    seen_cited_dois: set[str] = set()
    for chunk in chunks:
        if str(chunk.get("section", "")).strip().casefold() == "references":
            continue
        cleaned = clean_extracted_text(str(chunk.get("text") or ""))
        sentences = split_sentences_real(cleaned)
        for i, sentence in enumerate(sentences):
            stats["n_sentences_seen"] += 1
            citation = find_single_citation(sentence)
            if citation is None:
                continue
            stats["n_single_citation"] += 1
            entry = match_reference_entry(
                citation.surname, citation.surname2, citation.year, references,
            )
            if entry is None:
                continue
            resolved = await resolve_citation(entry, mailto=mailto)
            if resolved is None or not resolved.get("doi"):
                continue
            cited_doi = resolved["doi"]
            if (
                cited_doi == review_doi
                or cited_doi in excluded
                or cited_doi in seen_cited_dois
                or resolved.get("type") not in (None, "journal-article")
            ):
                continue
            stats["n_resolved"] += 1
            if len(kept) >= max_per_review:
                continue
            cached_path = fulltext_cache_dir / f"{doi_slug(cited_doi)}.json"
            was_cached = cached_path.exists()
            if was_cached:
                payload = read_json(cached_path)
                buckets = {"oa": 1, "fetched": 1, "kept": 1}
                # An already-cached cited work whose authors
                # were never successfully populated (e.g. cached back when
                # crossref_title_query_url's select list omitted "author") is not frozen empty
                # forever: it is enriched the same way a freshly-fetched one is (OpenAlex
                # first, the resolved entry's own Crossref authors as a fallback) and the
                # cache file rewritten, but ONLY when there is something to fix (an already-
                # correct cache costs no extra OpenAlex call, matching
                # test_extract_candidates_reuses_a_cached_cited_work_without_any_openalex_call).
                if not payload.get("authors"):
                    oa_work = await openalex_work_by_doi(cited_doi, mailto)
                    oa_authors = _authors(oa_work) if oa_work else []
                    payload["authors"] = oa_authors or (resolved.get("authors") or [])
                    write_json(cached_path, payload)
            else:
                work = {
                    "doi": cited_doi, "title": resolved.get("title"),
                    "publication_year": resolved.get("year"),
                    "journal": "(reference resolution)", "issn": None,
                    "selected_from": "reference-resolution",
                    "authorships": [],
                }
                _source, payload, reason = await acquire_one_with_budget(
                    work, pipeline, email=mailto, min_chars=min_chars, min_chunks=min_chunks,
                )
                buckets = acquire_outcome_buckets(reason)
                if payload is not None:
                    oa_work = await openalex_work_by_doi(cited_doi, mailto)
                    oa_authors = _authors(oa_work) if oa_work else []
                    payload["authors"] = oa_authors or (resolved.get("authors") or [])
            stats["n_oa"] += buckets["oa"]
            stats["n_fetched"] += buckets["fetched"]
            stats["n_kept"] += buckets["kept"]
            if payload is None:
                continue
            if not was_cached:
                write_json(cached_path, payload)
            seen_cited_dois.add(cited_doi)
            kept.append({
                "sentence": sentence,
                "context_before": sentences[i - 1] if i > 0 else "",
                "citation_text": citation.text,
                "cited_doi": cited_doi,
                "cited_title": resolved.get("title"),
                "cited_year": resolved.get("year"),
                "resolution": resolved.get("resolution"),
                "similarity": resolved.get("similarity"),
            })
    return kept, stats


# -------------------------------------------------------------------------------------
# The --from-readings path builds review caches from a reader-extracted-citations
# readings file instead of this module's own regex extraction (design items 2-3 above). A
# reader (an Opus reader session, or a human) reads one review article and, for each candidate
# claim, records the verbatim sentence, its citation, the review's own matched reference-list
# entry and the cited work's own title/year/authors/DOI (when printed) -- exactly the
# judgment call ``find_single_citation``/``parse_reference_entries``/``match_reference_entry``
# make mechanically above, made instead by a reader who can see context a regex cannot (a
# block-quoted participant utterance, a citation that is really about the review's own
# procedure, a reference entry split across a page break the PDF extractor mangled). This
# path still reuses the SAME acquisition code as the regex path (:func:`acquire_one_with_budget`,
# so the wrong-work identity check -- title AND first author against the freshly extracted PDF
# text -- runs on a fresh fetch unchanged, and also on an
# ALREADY-cached cited work too: :func:`build_review_cache_from_readings`'s own ``was_cached``
# branch re-applies :func:`build_hss_set.check_fetched_text_identity` before accepting the
# reused payload, rather than trusting a filename match alone) and writes the SAME review-cache
# shape (``doi``/``title``/``candidates``/``stats``, plus
# an ``extraction`` field on the payload and on every candidate) that :func:`build_outputs`
# already knows how to sample, split and assemble into items, so nothing downstream of the
# cache needs to know which path produced it -- except when a caller explicitly asks
# :func:`build_outputs` to isolate one path via its own ``extraction`` filter.
#
# Readings-file schema (``evaluation/claims/real_claims_readings.jsonl``, one JSON object per
# line, gitignored like the review/fulltext caches -- committed only when a specific run's
# output is meant to be reproducible from the repository): ``review_doi``, ``review_title``,
# ``claim`` (the verbatim sentence, citation kept in place), ``citation_text`` (the verbatim
# in-text citation), ``reference_entry`` (the matched reference-list entry, verbatim),
# ``cited_title``, ``cited_year``, ``cited_authors`` (a plain list of names), ``cited_doi``
# (``null``/absent when the reference entry printed none), ``context_before`` (the sentence
# immediately preceding the claim in the review, or ``""``), ``reader_notes`` (free text, not
# read by this module).
# -------------------------------------------------------------------------------------


def group_readings_by_review(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """``{review_doi: [rows...]}``, preserving each review's row order, one review-cache file
    being written per key (:func:`fetch_from_readings`). A row with no usable ``review_doi``
    (missing, blank, or unparseable) is dropped: there is nothing to cache it under, and no
    review the resulting item could ever cite as its ``source_doi``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        doi = _bare_doi(row.get("review_doi"))
        if not doi:
            continue
        out.setdefault(doi, []).append(dict(row))
    return out


def validate_reading_row(row: Mapping[str, Any]) -> list[str]:
    """Validation-error strings for one readings-file row (an empty list means the row is
    usable): the ``claim`` is 15-60 words (:data:`MIN_SENTENCE_WORDS`/:data:`MAX_SENTENCE_WORDS`
    -- the same length contract a regex-extracted sentence must meet, so a reader-selected
    claim is held to the same bound), carries exactly one recognised citation
    (:data:`YEAR_TOKEN_RE`/:data:`CITATION_RE`, the same pair :func:`find_single_citation` uses,
    so a claim with a second, unrecognised-form citation is rejected here exactly as it would
    be on the regex path), and a non-blank ``reference_entry`` is present -- the matched
    reference-list entry the reader copied verbatim, without which
    :func:`resolve_reading_citation` has nothing to sanity-check a printed DOI against or
    search Crossref by title for.

    None of the checks above look at WHICH citation the claim
    carries, only that it carries exactly one recognised form -- the regex path's own analogue
    (:func:`match_reference_entry`, matching leading surname, second surname and year against
    the review's own parsed reference list) has no counterpart here, exactly where the producer
    changes from a regex to a reader session that can mis-pair a claim with an unrelated cited
    work. Observed on a live CLI run: a row whose claim read "... a pattern Kalyuga (2011) also
    observed ..." was paired with a ``reference_entry``/``cited_authors``/``cited_year`` for
    Paas and Van Merrienboer (1994) and sailed through unnoticed. So the claim's own citation is
    cross-checked against the reader's own ``cited_year`` (more than one year apart is an error,
    skipped only when ``cited_year`` is absent) and against ``reference_entry``/``cited_authors``
    (the citation's surname, and its second surname when it carries one, must appear in at
    least one of them).

    That citation is built directly from the
    :data:`CITATION_RE` match already established above (via :func:`_citation_from_match`),
    never from :func:`find_single_citation`. ``find_single_citation`` is a SELECTION filter for
    regex-*extracted* sentences -- it also returns ``None`` for a PDF-extraction artefact
    (:func:`_is_artefact_free`: an ordinary suspended hyphenation like "pre- and post-test"
    trips its ``HYPHEN_BREAK_RE``, ``[A-Za-z]- [A-Za-z]``, exactly as this module's own
    :data:`_LINEBREAK_HYPHEN_RE` comment already documents as a legitimate, non-artefact shape),
    an unrelated same-sentence digit pair (``ADJACENT_NUMBERS_RE``), the citing review's own
    methods prose (:data:`REVIEW_OWN_WORK_RE`), a heading/list-item shape, or a non-terminal
    ending -- none of which say anything about whether the claim's citation matches this row's
    recorded reference. Routing the cross-check through it therefore silently disabled the
    entire check for any reader-copied claim that happened to trip one of those unrelated
    filters. A citation is built only when the elif branch above already confirms exactly one
    :data:`CITATION_RE` match; the genuine no-recognised-citation case is already its own error
    there.

    The surname check below does not accept a bare substring
    match. ``reference_entry`` is parsed with :func:`parse_reference_entry` and the citation's
    surname must equal that entry's own leading surname exactly (case/diacritic-insensitive,
    via :func:`normalize_surname`) -- the same rule :func:`match_reference_entry` applies on the
    regex path. A short surname (dense in this corpus: Li, He, Wu, Xu, Yu, Ma, Ng, An, Lee) is
    otherwise a substring of an unrelated word in the entry's own title or journal name (e.g.
    "Li" inside "Applied Linguistics"), which a raw ``in`` containment check cannot tell apart
    from a real match. The word-boundary containment check
    (``re.search(rf"\b{surname_key}\b", haystack)``) is used only as a fallback when
    ``reference_entry`` fails to parse at all (no leading "Surname, I." plus a parenthesised
    year), and the second-surname branch -- for which ``parse_reference_entry`` has no
    counterpart field -- always uses that same word-boundary form."""
    errors: list[str] = []
    claim = str(row.get("claim") or "").strip()
    words = claim.split()
    if not (MIN_SENTENCE_WORDS <= len(words) <= MAX_SENTENCE_WORDS):
        errors.append(
            f"claim has {len(words)} words, outside {MIN_SENTENCE_WORDS}-{MAX_SENTENCE_WORDS}"
        )
    citation_matches = list(CITATION_RE.finditer(claim))
    if len(YEAR_TOKEN_RE.findall(claim)) != 1:
        errors.append("claim does not carry exactly one citation year token")
    elif len(citation_matches) != 1:
        errors.append("claim does not carry exactly one recognised citation form")
    if not str(row.get("reference_entry") or "").strip():
        errors.append("reference_entry is missing")
    citation = _citation_from_match(citation_matches[0]) if len(citation_matches) == 1 else None
    if citation is not None:
        cited_year = str(row.get("cited_year") or "").strip()
        if cited_year:
            try:
                year_gap = abs(int(citation.year[:4]) - int(cited_year[:4]))
            except ValueError:
                year_gap = 0
            if year_gap > 1:
                errors.append(
                    f"claim's own citation year {citation.year} does not match this row's "
                    f"cited_year {cited_year}"
                )
        reference_entry = str(row.get("reference_entry") or "")
        cited_authors = [str(a) for a in (row.get("cited_authors") or []) if a]
        haystack = normalize_surname(f"{reference_entry} {' '.join(cited_authors)}")
        parsed_entry = parse_reference_entry(reference_entry)
        surname_key = normalize_surname(citation.surname)
        if parsed_entry is not None:
            surname_ok = surname_key == normalize_surname(str(parsed_entry.get("surname") or ""))
        else:
            surname_ok = bool(re.search(rf"\b{re.escape(surname_key)}\b", haystack))
        if surname_key and not surname_ok:
            errors.append(
                f"claim's own citation surname {citation.surname!r} not found in this row's "
                "reference_entry or cited_authors"
            )
        elif citation.surname2:
            surname2_key = normalize_surname(citation.surname2)
            if surname2_key and not re.search(rf"\b{re.escape(surname2_key)}\b", haystack):
                errors.append(
                    f"claim's own citation second surname {citation.surname2!r} not found in "
                    "this row's reference_entry or cited_authors"
                )
    return errors


async def resolve_reading_citation(
    row: Mapping[str, Any], *, mailto: str
) -> dict[str, Any] | None:
    """``{doi, title, resolution, similarity}`` for one readings-file row's cited work, or
    ``None`` when resolution fails -- the readings-path analogue of :func:`resolve_citation`,
    at the looser thresholds documented at :data:`READINGS_DOI_SANITY_THRESHOLD`/
    :data:`READINGS_TITLE_SIMILARITY_THRESHOLD`. When ``cited_doi`` is printed, it is
    sanity-checked against Crossref's own title for that DOI (rejected below the sanity
    threshold; accepted, as before, when no title comparison is possible at all -- a Crossref
    lookup failure or a row with no ``cited_title`` of its own). Otherwise, when only
    ``cited_title`` is given, the best Crossref title match at or above the title-search
    threshold is used, with the same year guard :func:`best_crossref_match` already applies
    when ``cited_year`` is given. A resolved title containing a Unicode replacement character
    (U+FFFD) is rejected outright, mirroring :func:`resolve_citation`'s own guard against a
    genuinely malformed Crossref response."""
    cited_doi = row.get("cited_doi")
    cited_title = row.get("cited_title")
    if cited_doi:
        cited_doi = _bare_doi(cited_doi)
        work = await crossref_lookup_by_doi(cited_doi, mailto)
        work_title = (work or {}).get("title")
        similarity = None
        if work_title and cited_title:
            similarity = token_jaccard(cited_title, str(work_title))
            if similarity < READINGS_DOI_SANITY_THRESHOLD:
                return None
        title = work_title or cited_title
        if title and "�" in title:
            return None
        return {
            "doi": cited_doi, "title": title, "resolution": "printed_doi",
            "similarity": similarity,
        }
    if not cited_title:
        return None
    candidates = await crossref_lookup_by_title(cited_title, mailto)
    match = best_crossref_match(
        cited_title, candidates, entry_year=row.get("cited_year"),
        threshold=READINGS_TITLE_SIMILARITY_THRESHOLD,
    )
    if match is None:
        return None
    match_title = match.get("title")
    if match_title and "�" in match_title:
        return None
    return {
        "doi": _bare_doi(match.get("doi")), "title": match_title,
        "resolution": "crossref_title", "similarity": match.get("similarity"),
    }


async def build_review_cache_from_readings(
    review_doi: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    mailto: str,
    pipeline: Mapping[str, Any],
    excluded: set[str],
    fulltext_cache_dir: Path,
    min_chars: int,
    min_chunks: int,
    max_per_review: int,
) -> dict[str, Any]:
    """``{doi, title, candidates, stats}`` for one review's readings rows -- the
    ``--from-readings`` analogue of :func:`extract_candidates`, producing the identical cache
    shape (the same six ``stats`` keys, the same per-candidate fields) so
    :func:`build_outputs`/:func:`build_from_cache` need no readings-specific code at all.
    ``n_sentences_seen``/``n_single_citation`` count readings rows rather than split
    sentences (one row IS one candidate claim here, there is no splitting), but mean the same
    thing at each stage of the funnel: total rows read, rows that passed
    :func:`validate_reading_row`, citations resolved to a usable DOI, and the OA/fetch/kept
    buckets :func:`acquire_outcome_buckets` already computes from the reused acquisition
    pipeline's own reason vocabulary -- including a ``"wrong_work: ..."`` reason from the
    wrong-work identity check inside :func:`acquire_one`, which buckets as ``oa`` only,
    exactly like a fetch that never happens for any other reason after a successful OA
    lookup."""
    stats = dict.fromkeys(
        ("n_sentences_seen", "n_single_citation", "n_resolved", "n_oa", "n_fetched", "n_kept"),
        0,
    )
    kept: list[dict[str, Any]] = []
    seen_cited_dois: set[str] = set()
    review_title = None
    for row in rows:
        stats["n_sentences_seen"] += 1
        if review_title is None and row.get("review_title"):
            review_title = row.get("review_title")
        if validate_reading_row(row):
            continue
        stats["n_single_citation"] += 1
        resolved = await resolve_reading_citation(row, mailto=mailto)
        if resolved is None or not resolved.get("doi"):
            continue
        cited_doi = resolved["doi"]
        if cited_doi == review_doi or cited_doi in excluded or cited_doi in seen_cited_dois:
            continue
        stats["n_resolved"] += 1
        if len(kept) >= max_per_review:
            continue
        cited_authors = [a for a in (row.get("cited_authors") or []) if a]
        cached_path = fulltext_cache_dir / f"{doi_slug(cited_doi)}.json"
        was_cached = cached_path.exists()
        if was_cached:
            payload = read_json(cached_path)
            # This branch does not accept a cached payload on
            # the strength of its DOI-slug filename alone; it also checks whether the text
            # actually fetched under that name describes what it claims to. The docstring's
            # own promise that this path "reuses the SAME acquisition code as the regex path
            # ... so the wrong-work identity check ... runs unchanged" held only on the else
            # branch below. Observed on a live CLI run: a cache entry carrying one work's
            # DOI/title/metadata but another work's chunks (a mis-fetch from a prior run) was
            # reused unchecked and shipped. Checked here with the reader's own cited_authors
            # preferred over the cached payload's own (possibly still-empty, e.g. a regex-path
            # cache fetched before authorship enrichment existed) authors list.
            identity_ok, detected = check_fetched_text_identity(
                expected_title=str(payload.get("title") or ""),
                extracted_text=" ".join(
                    str(c.get("text", "")) for c in payload.get("chunks") or []
                ),
                expected_authors=cited_authors or payload.get("authors"),
            )
            if not identity_ok:
                buckets = acquire_outcome_buckets(f"wrong_work: {detected}")
                payload = None
            else:
                buckets = {"oa": 1, "fetched": 1, "kept": 1}
                if not payload.get("authors") and cited_authors:
                    payload["authors"] = cited_authors
                    write_json(cached_path, payload)
        else:
            work = {
                "doi": cited_doi,
                "title": resolved.get("title") or row.get("cited_title"),
                "publication_year": row.get("cited_year"),
                "journal": "(reader-extracted citation)",
                "issn": None,
                "selected_from": "reading",
                # The identity check inside acquire_one (check_fetched_text_identity) compares
                # the freshly fetched PDF text's first author against work["authorships"] via
                # build_hss_set._authors -- populating it from the reading's own cited_authors
                # (rather than the empty list the regex path passes here, since it has no
                # author list until AFTER a successful fetch) is what makes the identity check
                # verify the first author too, not only the title, on this path.
                "authorships": [{"author": {"display_name": n}} for n in cited_authors],
            }
            _source, payload, reason = await acquire_one_with_budget(
                work, pipeline, email=mailto, min_chars=min_chars, min_chunks=min_chunks,
            )
            buckets = acquire_outcome_buckets(reason)
            if payload is not None and not payload.get("authors"):
                payload["authors"] = cited_authors
        stats["n_oa"] += buckets["oa"]
        stats["n_fetched"] += buckets["fetched"]
        stats["n_kept"] += buckets["kept"]
        if payload is None:
            continue
        if not was_cached:
            write_json(cached_path, payload)
        seen_cited_dois.add(cited_doi)
        kept.append({
            "sentence": row.get("claim"),
            "context_before": row.get("context_before") or "",
            "citation_text": row.get("citation_text"),
            "cited_doi": cited_doi,
            "cited_title": resolved.get("title") or row.get("cited_title"),
            "cited_year": row.get("cited_year"),
            "resolution": resolved.get("resolution"),
            "similarity": resolved.get("similarity"),
            # Stamped on every candidate (not only the
            # payload below) so build_outputs can tell a readings-path item from a regex-path
            # one even inside a cache dir that mixes both, without which a --from-readings run
            # over a cache directory that already held regex-extraction review caches silently
            # shipped a "reader-extracted" test set that was mostly regex-extracted.
            "extraction": "readings",
        })
    return {
        "doi": review_doi,
        "title": review_title,
        # build_hss_set.load_cache (reused unchanged by build_outputs) only keeps a cache file
        # whose payload carries a truthy "chunks" -- this path never fetches the citing
        # review's own PDF (a reader already read it), so this placeholder exists purely to
        # satisfy that shared filter; nothing downstream of load_cache ever reads it back.
        "chunks": [{"section": "readings", "text": ""}],
        "candidates": kept,
        "stats": stats,
        "extraction": "readings",
    }


async def fetch_from_readings(
    *,
    readings_path: Path,
    mailto: str,
    excluded: set[str],
    reviews_cache_dir: Path,
    fulltext_cache_dir: Path,
    min_chars: int,
    min_chunks: int,
    max_per_review: int,
) -> int:
    """Stage 1 for ``--from-readings``: reads *readings_path* (one JSON object per line, the
    schema documented above), groups its rows by ``review_doi``
    (:func:`group_readings_by_review`), resolves and fetches each review's candidates
    (:func:`build_review_cache_from_readings`) and writes one review-cache payload per
    ``review_doi`` under *reviews_cache_dir* -- in EXACTLY the shape
    :func:`extract_candidates` produces, so :func:`build_outputs`/``--skip-fetch`` need no
    readings-specific code at all afterwards. Returns the number of distinct reviews present
    in the readings file.

    Raises ``SystemExit`` when *readings_path*
    yields zero rows (missing file, empty file, or every line blank) or groups into zero
    reviews (every row lacked a usable ``review_doi``), before ``main`` ever reaches
    :func:`build_from_cache` -- a mistyped or empty ``--from-readings`` path must fail loudly
    rather than silently truncating the shipped ``real_claims_test.jsonl``/
    ``real_claims_dev.jsonl`` to zero rows with exit code 0."""
    all_rows = read_jsonl(readings_path)
    if not all_rows:
        raise SystemExit(
            f"--from-readings: {readings_path} yielded zero rows (missing, empty, or every "
            "line blank)"
        )
    by_review = group_readings_by_review(all_rows)
    if not by_review:
        raise SystemExit(
            f"--from-readings: {readings_path} yielded zero grouped reviews (no row had a "
            "usable review_doi)"
        )
    pipeline = load_pipeline()
    reviews_cache_dir.mkdir(parents=True, exist_ok=True)
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    for review_doi, rows in by_review.items():
        payload = await build_review_cache_from_readings(
            review_doi, rows, mailto=mailto, pipeline=pipeline, excluded=excluded,
            fulltext_cache_dir=fulltext_cache_dir, min_chars=min_chars, min_chunks=min_chunks,
            max_per_review=max_per_review,
        )
        write_json(reviews_cache_dir / f"{doi_slug(review_doi)}.json", payload)
        print(
            f"  {review_doi}: {len(payload['candidates'])} usable citation(s) from readings, "
            f"stats={payload['stats']}"
        )
    return len(by_review)


async def _process_review(
    work: Mapping[str, Any],
    *,
    mailto: str,
    pipeline: Mapping[str, Any],
    excluded: set[str],
    reviews_cache_dir: Path,
    fulltext_cache_dir: Path,
    min_chars: int,
    min_chunks: int,
    max_per_review: int,
    semaphore: asyncio.Semaphore,
) -> None:
    """Fetch (or reuse a cached fetch of) one review, then always recompute its candidates
    (design item 6: resumability for the expensive PDF fetch, freshness for the cheap, pure
    extraction logic). The whole body is one failure domain: an unexpected exception anywhere
    in it (this module's own extraction logic, or the reused acquisition pipeline reached
    outside :func:`acquire_one_with_budget`'s own try/except, e.g. inside ``read_json`` on a
    corrupted cache file) is logged and this one review is skipped, never propagated through
    ``asyncio.gather`` to abort every other review already in flight for the same run.

    A review whose cache payload already carries
    ``extraction == "readings"`` is left untouched and never reaches ``extract_candidates``.
    Both paths default to the same cache directory, and this function's own resumability
    design -- reprocess a cached review's candidates from its own cached text, even one this
    run's discovery did not return -- would otherwise re-run regex extraction over a readings
    payload's placeholder ``chunks`` (``[{"section": "readings", "text": ""}]``), yielding zero
    candidates and silently overwriting the payload's ``extraction`` back to ``"regex"``. A
    regex-path run must never destroy a readings-path review cache this way."""
    review_doi = work["doi"]
    cache_path = reviews_cache_dir / f"{doi_slug(review_doi)}.json"
    try:
        if cache_path.exists():
            payload = read_json(cache_path)
            if payload.get("extraction") == "readings":
                print(f"  {review_doi}: skipping (cached from --from-readings, not regex)")
                return
        else:
            async with semaphore:
                _source, payload, reason = await acquire_one_with_budget(
                    work, pipeline, email=mailto, min_chars=min_chars, min_chunks=min_chunks,
                )
            if payload is None:
                print(f"  {review_doi}: review fetch failed ({reason})")
                return
        candidates, stats = await extract_candidates(
            payload, mailto=mailto, pipeline=pipeline, excluded=excluded,
            fulltext_cache_dir=fulltext_cache_dir, min_chars=min_chars, min_chunks=min_chunks,
            review_doi=review_doi, max_per_review=max_per_review,
        )
        # Stamped here (extract_candidates' own caller), not
        # inside extract_candidates itself, so every candidate this module's regex-extraction
        # path has ever produced -- including one already cached before this stamp existed,
        # via build_outputs' own "no key means regex" fallback -- is unambiguously
        # distinguishable from a --from-readings candidate once both share one cache directory.
        for c in candidates:
            c["extraction"] = "regex"
        payload["extraction"] = "regex"
        payload["candidates"] = candidates
        payload["stats"] = stats
        write_json(cache_path, payload)
        print(f"  {review_doi}: {len(candidates)} usable citation(s), stats={stats}")
    except Exception as exc:  # noqa: BLE001 - one review's failure must not abort the harvest
        print(f"  {review_doi}: unexpected failure, skipped ({type(exc).__name__}: {exc})")


async def fetch_reviews(
    *,
    mailto: str,
    excluded: set[str],
    reviews_cache_dir: Path,
    fulltext_cache_dir: Path,
    max_reviews: int,
    max_per_review: int,
    min_chars: int,
    min_chunks: int,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Stage 1: discover, fetch (bounded concurrency, design item 6), resolve and cache.
    Returns the number of reviews discovered.

    Every review already present in *reviews_cache_dir* is reprocessed too, even one this
    run's own :func:`discover_reviews` call does not return (a narrower result on a given run,
    e.g. an OpenAlex channel over its request budget, or a review that no longer matches a
    tightened filter): :func:`_process_review` never re-fetches an already-cached review's own
    text, only recomputes its candidates from it, so this costs no extra network call and
    keeps the module docstring's own promise -- "a fix to this module's own extraction logic
    applies to an already-fetched review without needing to redownload its PDF" -- true even
    when live discovery is degraded or absent for the run that applies the fix."""
    pipeline = load_pipeline()
    reviews_cache_dir.mkdir(parents=True, exist_ok=True)
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    discovered = await discover_reviews(mailto, max_reviews=max_reviews)
    print(f"discovered {len(discovered)} candidate review articles")
    discovered_dois = {w["doi"] for w in discovered}
    already_cached = [
        {"doi": doi} for doi in load_cache(reviews_cache_dir) if doi not in discovered_dois
    ]
    if already_cached:
        print(
            f"reprocessing {len(already_cached)} previously-cached review(s) not returned by "
            "this run's discovery"
        )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    await asyncio.gather(*(
        _process_review(
            work, mailto=mailto, pipeline=pipeline, excluded=excluded,
            reviews_cache_dir=reviews_cache_dir, fulltext_cache_dir=fulltext_cache_dir,
            min_chars=min_chars, min_chunks=min_chunks, max_per_review=max_per_review,
            semaphore=semaphore,
        )
        for work in (*discovered, *already_cached)
    ))
    return len(discovered)


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reviews-cache-dir", type=Path, default=DEFAULT_REVIEWS_CACHE_DIR)
    ap.add_argument("--fulltext-cache-dir", type=Path, default=DEFAULT_FULLTEXT_CACHE_DIR)
    ap.add_argument("--out-dev", type=Path, default=DEFAULT_DEV_OUT)
    ap.add_argument("--out-test", type=Path, default=DEFAULT_TEST_OUT)
    ap.add_argument("--build-out", type=Path, default=DEFAULT_BUILD_OUT)
    ap.add_argument("--hss-sources", type=Path, default=DEFAULT_HSS_SOURCES)
    ap.add_argument("--hss-dev-sources", type=Path, default=DEFAULT_HSS_DEV_SOURCES)
    ap.add_argument(
        "--hss-test-v3-sources", type=Path, default=DEFAULT_HSS_TEST_V3_SOURCES,
        help="exclude the frozen v3 test set's own sources too (design item 3's third "
        "exclusion list, alongside --hss-sources/--hss-dev-sources)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-dev", type=int, default=DEFAULT_N_DEV)
    ap.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    ap.add_argument("--max-reviews", type=int, default=DEFAULT_MAX_REVIEWS)
    ap.add_argument("--max-per-review", type=int, default=DEFAULT_MAX_PER_REVIEW)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--min-chunks", type=int, default=DEFAULT_MIN_CHUNKS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--dry-run", action="store_true",
                     help="print the discovery plan and targets; no network, nothing written")
    ap.add_argument("--skip-fetch", action="store_true",
                     help="rebuild the two output files from the local caches only")
    ap.add_argument(
        "--from-readings", type=Path, default=None,
        help="build review caches from a reader-extracted-citations readings jsonl file "
        "(see the module docstring's readings-file schema) instead of the regex-extraction "
        "discovery pipeline; still writes one review-cache payload per review_doi under "
        "--reviews-cache-dir in the same shape, so --skip-fetch and the dev/test/build-json "
        "output stage work unchanged afterwards",
    )
    ap.add_argument(
        "--exclude-item-ids", default=None,
        help="comma-separated item_id values (e.g. 'real-dev-06,real-dev-07') found in the "
        "CURRENT --out-dev/--out-test files; each one's cited_doi is looked up there before "
        "this run overwrites them and added to the DOI exclusion set, so a vetoed item's "
        "cited work is never re-selected on rebuild (see load_item_id_cited_dois)",
    )
    return resolve_path_args(ap.parse_args(argv))


def _write_jsonl_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write *rows* as jsonl, always creating *path* (even empty) so :func:`sha256_file` never
    hits a missing file when a build yields zero dev or zero test items."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    append_jsonl(path, rows)


def build_from_cache(
    args: argparse.Namespace, excluded: set[str], *, extraction: str | None = None,
) -> None:
    dev_items, test_items, info = build_outputs(
        args.reviews_cache_dir, args.fulltext_cache_dir, seed=args.seed, n_dev=args.n_dev,
        n_test=args.n_test, max_per_review=args.max_per_review, excluded=excluded,
        extraction=extraction,
    )
    _write_jsonl_file(args.out_dev, dev_items)
    _write_jsonl_file(args.out_test, test_items)
    info["out_dev_sha256"] = sha256_file(args.out_dev)
    info["out_test_sha256"] = sha256_file(args.out_test)
    info["built_at"] = now_iso()
    write_json(args.build_out, info)
    print(
        f"wrote {args.out_dev} ({len(dev_items)} items) and {args.out_test} "
        f"({len(test_items)} items) and {args.build_out}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    import asyncio as _asyncio

    # Real-run finding (2026-09-07): a non-UTF-8 Windows console codepage (observed: GBK)
    # cannot encode a non-ASCII character (an author's diacritic, a "©") that a print()
    # anywhere in this module or the reused acquisition pipeline emits, raising
    # UnicodeEncodeError and (before acquire_one_with_budget's broader except clause and this
    # fix) aborting the whole harvest outright. Reconfiguring here fixes it at the source for
    # every print in the process, not just the ones this module's own exception handling
    # already tolerates. Guarded: a test's ``capsys``-captured stream does not always support
    # ``reconfigure``, and that must never be a reason for this CLI to fail.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    if args.dry_run:
        print(
            "dry run -- plan only, no network:\n"
            f"  journals: {len(JOURNALS)}; topics: {TOPIC_NAMES}\n"
            f"  subfields: {SUBFIELD_NAMES}; title keywords: {TITLE_KEYWORDS}\n"
            f"  max_reviews={args.max_reviews} max_per_review={args.max_per_review} "
            f"concurrency={args.concurrency}\n"
            f"  targets: n_dev={args.n_dev} n_test={args.n_test} seed={args.seed}\n"
            f"  min_chars={args.min_chars} min_chunks={args.min_chunks}\n"
            f"  reviews_cache_dir={args.reviews_cache_dir}\n"
            f"  fulltext_cache_dir={args.fulltext_cache_dir}"
        )
        return 0
    excluded = load_excluded_dois(
        [args.hss_sources, args.hss_dev_sources, args.hss_test_v3_sources]
    )
    if args.exclude_item_ids:
        wanted_ids = [x.strip() for x in args.exclude_item_ids.split(",") if x.strip()]
        id_to_doi = load_item_id_cited_dois(wanted_ids, args.out_dev, args.out_test)
        missing = [i for i in wanted_ids if i not in id_to_doi]
        if missing:
            raise SystemExit(
                f"--exclude-item-ids: no cited_doi found for {missing} in "
                f"{args.out_dev} / {args.out_test}"
            )
        excluded |= {_bare_doi(doi) for doi in id_to_doi.values() if doi}
    if not args.skip_fetch:
        mailto = load_dotenv_values(REPO_ROOT / ".env").get("OPENALEX_EMAIL") or ""
        if not mailto:
            raise SystemExit(
                "OPENALEX_EMAIL missing from .env (needed for OpenAlex/Unpaywall/Crossref)"
            )
        export_env_from_dotenv(("UNPAYWALL_EMAIL", "OPENALEX_EMAIL", "DATABASE_URL"))
        if args.from_readings:
            # A bare `type=Path` argument has no
            # existence check of its own, and read_jsonl silently returns [] for a path that
            # does not exist, so without this a mistyped path used to reach build_from_cache
            # with zero items and exit 0. fetch_from_readings raises on zero rows too (the
            # empty-but-present-file case); this is the distinct "wrong path entirely" case.
            if not args.from_readings.exists():
                raise SystemExit(f"--from-readings: file not found: {args.from_readings}")
            _asyncio.run(fetch_from_readings(
                readings_path=args.from_readings, mailto=mailto, excluded=excluded,
                reviews_cache_dir=args.reviews_cache_dir,
                fulltext_cache_dir=args.fulltext_cache_dir, min_chars=args.min_chars,
                min_chunks=args.min_chunks, max_per_review=args.max_per_review,
            ))
        else:
            _asyncio.run(fetch_reviews(
                mailto=mailto, excluded=excluded, reviews_cache_dir=args.reviews_cache_dir,
                fulltext_cache_dir=args.fulltext_cache_dir, max_reviews=args.max_reviews,
                max_per_review=args.max_per_review, min_chars=args.min_chars,
                min_chunks=args.min_chunks, concurrency=args.concurrency,
            ))
    # A --from-readings run isolates its own build to
    # readings-path candidates only, so a cache directory that already held (or later
    # accumulates) regex-extraction review caches never silently contributes regex-path items
    # to a build meant to be reader-extracted evidence (info["n_items_by_extraction"] still
    # discloses both counts either way).
    build_from_cache(args, excluded, extraction="readings" if args.from_readings else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
