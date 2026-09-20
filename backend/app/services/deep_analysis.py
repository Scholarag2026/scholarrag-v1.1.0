"""Background service: deep-analyze all papers in a project library."""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.deep_analysis_agent import (
    _resolve_instructions,
    analyze_paper_text,
    build_abstract_only_analysis,
)
from app.config import settings
from app.models.analysis_job import JobStatus
from app.models.evidence import Evidence
from app.models.paper import Paper
from app.models.project_paper import ProjectPaper
from app.schemas.provenance import prompt_version as hash_prompt_version
from app.services import task as task_service

logger = logging.getLogger(__name__)

#: A quote shorter than this can drop the negation or qualifier that made
#: the source sentence true, so it is rejected regardless of how it locates in the chunk.
MIN_EVIDENCE_WORDS = 8

#: Sentence-ending punctuation, and the brackets and quotation marks that may sit either
#: side of a sentence. A sentence can end "...necessary.”" or "...(see Table 1)." and can
#: begin "“The results...", so both ends of a candidate span are read past these.
_TERMINALS = ".!?"
_CLOSERS = "\"')]}»›”’"
_OPENERS = "\"'([{«‹“‘"

#: A real sentence boundary in extracted text: terminal punctuation, any closing quotes
#: or brackets, then whitespace.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?][\"')\]}»›”’]*\s")

#: The same boundary shape, anchored at the very end of the text in front of a candidate
#: span. Used by ``_span_starts_a_sentence`` to decide whether the terminal punctuation
#: immediately before a span is a real sentence end: a decimal point, the last period of
#: an author initial ("J.R.") or an abbreviation is never followed by whitespace, so none
#: of those match here even though the last character before the span is one of
#: ``_TERMINALS``.
_TRAILING_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?][\"')\]}»›”’]*\s+$")

#: A word broken across a line by the typesetter ("feed-\nback"), which pymupdf hands
#: over with the hyphen and the line break still in it.
_LINE_BREAK_HYPHEN_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

#: U+00AD SOFT HYPHEN: invisible when rendered, a character like any other to a string
#: comparison.
_SOFT_HYPHEN = "­"

#: How many words in front of a candidate span ``_lead_in_is_not_prose`` reads before
#: deciding whether they are a heading, a table row or a gutter rather than the first
#: half of a sentence.
_LEAD_IN_TOKENS = 4

#: Punctuation a heading never ends in, but the first half of a sentence often does.
_CLAUSE_JOINING_PUNCTUATION = ",;:-–—(/&["

#: Words a heading never ends in either. Read by ``_starts_a_line_below_a_heading``: the
#: line above a candidate span ending in one of these is a sentence a hard line wrap cut
#: open, not a heading. Closed classes only (articles, conjunctions, subordinators,
#: prepositions, auxiliaries, relative and demonstrative pronouns), since a heading can
#: end in any content word at all ("3.4 Data analysis").
_CONTINUATION_WORDS = frozenset("""
a an the and or but if when while because although though unless since as than whether
that which who whom whose of in on at by for with from to into about over under between
among per via this these those it its their his her our your my is are was were be been
being has have had do does did not no both either neither such all any more most other
another same
""".split())


def analysis_prompt_version(expertise_level: str | None = None) -> str:
    """The sha256 ``prompt_version`` of the exact analysis instructions used for
    *expertise_level* -- a pure function of the prompt text, independent of any one
    call's own provenance, so it can be recorded on every evidence row a paper's
    analysis produces even when the caller (``analyze_all_papers`` below) uses the
    provenance-discarding ``analyze_paper_text`` wrapper rather than
    ``analyze_paper_text_with_provenance``."""
    return hash_prompt_version(_resolve_instructions(expertise_level))


def normalise_ws(text: str | None) -> str:
    """Collapse all whitespace (including newlines) to single spaces and strip the
    ends. Shared by every evidence-acceptance check below so the same normalised text
    is compared and stored everywhere."""
    return " ".join((text or "").split())


def normalise_source_text(text: str | None) -> str:
    """A chunk's own text, cleaned of the artefacts PDF extraction leaves behind but
    otherwise unchanged: NFKC (so an "fi" ligature is the two letters it renders as),
    soft hyphens dropped, a word broken across a line rejoined, whitespace collapsed.

    This is what an accepted evidence quote is stored as -- applied to the span located
    in the chunk, never to the string the model typed -- so a stored quote is the
    paper's own wording by construction. Unlike ``fold_for_match`` below it changes no
    letter, digit or punctuation mark and does not fold case, so it is not a comparison
    form: two texts that differ only in typography still differ here."""
    normalised = unicodedata.normalize("NFKC", text or "")
    normalised = normalised.replace(_SOFT_HYPHEN, "")
    normalised = _LINE_BREAK_HYPHEN_RE.sub(r"\1\2", normalised)
    return normalise_ws(normalised)


@lru_cache(maxsize=None)
def _fold_char(character: str) -> str:
    """One character's contribution to ``fold_for_match``: NFKC, then NFKD with every
    combining mark dropped, then letters and digits only, case-folded. Cached because a
    chunk of 120,000 characters draws on an alphabet of a few hundred."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", character))
    return "".join(c for c in decomposed if c.isalpha() or c.isdigit()).casefold()


def fold_with_offsets(text: str | None) -> tuple[str, list[int]]:
    """``fold_for_match(text)`` plus, for every character of the folded form, the index
    in *text* of the character it came from.

    The offsets are what makes it possible to fold both sides of a comparison this hard
    and still hand back the source's own characters: a quote is located in the folded
    chunk, and the match's start and end are mapped straight back onto the raw chunk to
    read the span's real text and to look at the punctuation around it."""
    folded: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(text or ""):
        for kept in _fold_char(character):
            folded.append(kept)
            offsets.append(index)
    return "".join(folded), offsets


def fold_for_match(text: str | None) -> str:
    """Canonical comparison form for evidence acceptance: NFKC and NFKD-with-marks-
    dropped, then letters and digits only, case-folded.

    The same canonical form ``app.services.fulltext._normalise_for_match`` already
    compares a claim's quote against the full text with, computed character by character
    here so ``fold_with_offsets`` can keep the index each kept character came from. Every
    typographic and PDF-extraction difference between what the model typed and what the
    extractor produced -- curly against straight quotes, an "fi" ligature split by an
    injected space, an en dash against a hyphen, a word broken across a line, a soft
    hyphen, a non-breaking space -- vanishes identically on both sides, so a genuine copy
    compares equal to its own source. The letters and digits themselves are never
    reordered or altered, so a paraphrase still compares unequal.

    Without this normalisation, an exact copy can still be rejected purely because of
    typography of this kind.
    """
    return fold_with_offsets(text)[0]


def _looks_like_sentence_start(character: str) -> bool:
    """True when *character* can open a sentence: an upper-case letter, a digit, or a
    letter from a script that has no case at all."""
    if character.isdigit():
        return True
    if not character.isalpha():
        return False
    return character.isupper() or character.lower() == character.upper()


def _lead_in_is_not_prose(prefix: str) -> bool:
    """True when what stands between the last real sentence boundary in *prefix* and the
    end of *prefix* is not the first half of a sentence.

    Extracted text glues material that carries no full stop of its own onto the front of
    the sentence that follows it: a numbered heading ("3.2 Procedures"), the last row of
    a table ("Total 100% 100%"), a line-number gutter ("1 2 3 ... 25"), a figure caption.
    A sentence that follows one of these is still a whole sentence, but a splitter that
    only knows about full stops cannot see where it begins -- which is why 8 of 78 items
    proposed for the seed papers were rejected while being exact copies of a whole
    sentence.

    The discriminator is a lower-case word in the last ``_LEAD_IN_TOKENS`` words before
    the span: headings, table rows and gutters run out in a capitalised word, a number or
    a percentage ("... Total 100% 100% 100%"), while the first half of an English
    sentence carries a lower-case function word within a word or two of where it was cut
    ("Figure 4, from the record, shows that ..."). Only the last few words are looked at
    because a table's own header rows are full of lower-case words that say nothing about
    where the sentence after the table begins.

    It is a heuristic, and it is deliberately paired in ``_span_starts_a_sentence`` with a
    second condition -- the span itself has to begin the way a sentence begins -- so that
    accepting a fragment needs both no lower-case word in front of the span and a
    capitalised word or a digit starting it. What it can still let through is a cut made
    at a capitalised word with no function word near it ("... reported by Smith | The
    gain was largest ..."), a known limitation of this heuristic.
    """
    boundaries = list(_SENTENCE_BOUNDARY_RE.finditer(prefix))
    lead_in = prefix[boundaries[-1].end():] if boundaries else prefix
    for token in lead_in.split()[-_LEAD_IN_TOKENS:]:
        letters = [c for c in token if c.isalpha()]
        if letters and letters[0].islower():
            return False
    return True


def _starts_a_line_below_a_heading(text: str, start: int) -> bool:
    """True when ``text[start]`` is the first character of a physical line and the line
    above it does not run into it.

    A numbered section heading is a line of its own in extracted text, and it often ends
    in a lower-case word ("3.4 Data analysis", "6.3 Limitations and suggestions for
    future research"), which is exactly what ``_lead_in_is_not_prose`` reads as the first
    half of a sentence. The line above is taken to run into the span when it ends in a
    word that cannot end a heading either: punctuation that opens or joins a clause, or
    one of ``_CONTINUATION_WORDS``. That is what separates a heading from a sentence a
    hard line wrap cut open ("Figure 4, from the record, shows that \\n Flora
    resubmitted ...").

    Measured on the twelve seed papers: this recovered 12 further proposed items, every
    one of them the first sentence under
    a section heading or a figure label, and accepted nothing else.
    """
    line_start = text.rfind("\n", 0, start)
    if line_start == -1:
        return False
    if text[line_start + 1:start].strip():
        return False
    above = text[:line_start].rstrip()
    line_above = above[above.rfind("\n") + 1:].strip()
    if not line_above:
        return False
    last_word = line_above.split()[-1]
    if last_word[-1] in _CLAUSE_JOINING_PUNCTUATION:
        return False
    return last_word.strip(_CLOSERS + _OPENERS + ".,;:").casefold() not in _CONTINUATION_WORDS


def _span_starts_a_sentence(text: str, start: int) -> bool:
    """True when ``text[start]`` is where a sentence begins: at the very start of the
    chunk, straight after a real sentence boundary (terminal punctuation, any closing
    quotes or brackets, then whitespace -- the same shape ``_SENTENCE_BOUNDARY_RE``
    requires, via ``_TRAILING_SENTENCE_BOUNDARY_RE``, so a decimal point or the last
    period of an author initial with no space after it is never treated as one), after a
    heading, table row or gutter that carries no punctuation of its own
    (``_lead_in_is_not_prose``), or at the start of a line below a heading line
    (``_starts_a_line_below_a_heading``).

    The last two branches both also require the span itself to begin the way a sentence
    begins, so accepting a fragment takes two independent mistakes."""
    prefix = text[:start]
    stripped = prefix.rstrip()
    if not stripped:
        return True
    if _TRAILING_SENTENCE_BOUNDARY_RE.search(prefix):
        return True
    if not _looks_like_sentence_start(text[start]):
        return False
    return _lead_in_is_not_prose(prefix) or _starts_a_line_below_a_heading(text, start)


def _sentence_end_after(text: str, end: int) -> int | None:
    """The index just past the sentence-ending punctuation that closes the span ending at
    *end*, or ``None`` when the span stops in the middle of a sentence.

    *end* is one past the span's last letter or digit, because ``fold_for_match`` keeps
    nothing else, so the punctuation that closes the sentence can be several characters
    further on: the "4" of "(8.4%)." is followed by "%", ")" and only then the full stop.
    Everything up to the next letter, digit or space is read here, and the span is a
    sentence end only if a full stop, question mark or exclamation mark is among it. The
    returned index includes that punctuation, so the stored quote keeps the mark it ends
    on."""
    index = end
    seen_terminal = False
    while index < len(text):
        character = text[index]
        if character.isspace():
            break
        if _fold_char(character):
            # A letter or a digit: the span stops inside a word or a number.
            return None
        if character in _TERMINALS:
            seen_terminal = True
        index += 1
    return index if seen_terminal else None


def _span_start_with_opening_quote(text: str, start: int) -> int:
    """*start*, widened left over an opening quotation mark or bracket that the fold
    dropped, so a quoted sentence keeps its own opening mark when it is stored."""
    index = start
    while index > 0 and text[index - 1] in _OPENERS:
        index -= 1
    if index > 0 and not text[index - 1].isspace():
        return start
    return index


class QuoteLocation(NamedTuple):
    """Where a proposed evidence quote really is: the chunk code found it in (not
    necessarily the one the model named) and the source's own text for that span."""

    chunk_index: int
    quote: str


def locate_quote_in_chunk(quote: str | None, chunk_text: str | None) -> str | None:
    """The source's own text for *quote* in *chunk_text*, or ``None``.

    A span is accepted only when it is a verbatim copy under ``fold_for_match``, begins
    where a sentence begins and ends where a sentence ends, so what comes back is always
    one or more whole sentences of the chunk (design section 6 amendment A2: a fragment
    can drop the negation or the qualifier that made the source sentence true). Every
    occurrence is tried, so a sentence repeated in a chunk is located by whichever
    occurrence sits on sentence boundaries."""
    folded_quote = fold_for_match(quote)
    if not folded_quote:
        return None
    folded_chunk, offsets = fold_with_offsets(chunk_text)
    position = folded_chunk.find(folded_quote)
    while position != -1:
        raw_start = offsets[position]
        raw_end = offsets[position + len(folded_quote) - 1] + 1
        end = _sentence_end_after(chunk_text or "", raw_end)
        if end is not None and _span_starts_a_sentence(chunk_text or "", raw_start):
            start = _span_start_with_opening_quote(chunk_text or "", raw_start)
            return normalise_source_text((chunk_text or "")[start:end])
        position = folded_chunk.find(folded_quote, position + 1)
    return None


def locate_quote(quote: str | None, chunks: list, named_index: object) -> QuoteLocation | None:
    """Where *quote* is in *chunks*, looking in the chunk the model named first and then
    in every other chunk in order.

    ``chunk_index`` is a hint, not a gate. Of 78 evidence items proposed for the
    twelve seed papers, 35 named a chunk that does not exist -- every item of a paper
    stored as one 76,000-character chunk was numbered as if the paper had eighteen -- and
    21 of those were verbatim, whole-sentence copies sitting in a chunk of the same
    paper. Searching the whole paper costs one pass over text already in memory and keeps
    the decision with code: the stored ``chunk_index`` is the chunk the quote was found
    in, so it still points at the text the quote came from."""
    order: list[int] = []
    if isinstance(named_index, int) and 0 <= named_index < len(chunks):
        order.append(named_index)
    order.extend(i for i in range(len(chunks)) if i not in order)
    for index in order:
        chunk = chunks[index]
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        located = locate_quote_in_chunk(quote, chunk_text)
        if located is not None:
            return QuoteLocation(chunk_index=index, quote=located)
    return None


def quote_is_complete_sentences(quote_norm: str, chunk_text: str) -> bool:
    """True when *quote_norm* is a verbatim span of *chunk_text* covering one or more
    whole sentences (design section 6 amendment A2). Kept as the single-chunk predicate
    the acceptance rule is built on; ``locate_quote`` applies it to every chunk of a
    paper and also hands back the source's own text for the span."""
    return locate_quote_in_chunk(quote_norm, chunk_text) is not None


def normalise_concepts(concepts: object) -> list[str]:
    """Concept tags lower-cased, trimmed, and deduplicated in first-seen order."""
    seen: list[str] = []
    for concept in concepts or []:
        normalised = str(concept).strip().lower()
        if normalised and normalised not in seen:
            seen.append(normalised)
    return seen


def validate_evidence_items(items: object, chunks: list) -> tuple[list[dict], int]:
    """Accept only the evidence items code can verify against *chunks*.

    An item is accepted only when its ``quote`` is at least ``MIN_EVIDENCE_WORDS`` words
    and ``locate_quote`` finds it, as a verbatim whole-sentence span under
    ``fold_for_match``, in one of *chunks* -- the chunk the model named if the quote is
    there, otherwise any other chunk of the same paper. What is stored is the
    source's own text for that span and the index of the chunk it was found in, not the
    string the model typed and not the index it claimed, so an accepted quote is verbatim
    from the paper by construction. Every other item is rejected and only counted, never
    stored. Returns ``(accepted_rows, rejected_count)`` where each accepted row is a
    plain dict ready for ``Evidence(paper_id=..., prompt_version=..., **row)``.

    ``items`` tolerates whatever shape ``DeepPaperAnalysis.model_dump()["evidence"]``
    produces (a list of plain dicts) as well as a missing or ``None`` value (an older
    stored analysis, or a caller's fake result with no ``evidence`` key at all).
    """
    accepted: list[dict] = []
    rejected = 0
    for item in items or []:
        quote = item.get("quote") if isinstance(item, dict) else None
        if len(normalise_ws(quote).split()) < MIN_EVIDENCE_WORDS:
            rejected += 1
            continue
        located = locate_quote(
            quote, chunks, item.get("chunk_index") if isinstance(item, dict) else None
        )
        if located is None or len(located.quote.split()) < MIN_EVIDENCE_WORDS:
            rejected += 1
            continue
        chunk = chunks[located.chunk_index]
        accepted.append({
            "quote": located.quote,
            "chunk_index": located.chunk_index,
            "section": chunk.get("section") if isinstance(chunk, dict) else None,
            "finding": (item.get("finding") or "").strip(),
            "kind": item.get("kind"),
            "origin": item.get("origin"),
            "concepts": normalise_concepts(item.get("concepts")),
        })
    return accepted, rejected


async def replace_paper_evidence(
    session: AsyncSession,
    paper_id: UUID,
    accepted_rows: list[dict],
    prompt_version_str: str,
) -> None:
    """Replace *paper_id*'s entire evidence table row set with *accepted_rows*,
    so a paper re-analysed with
    a newer prompt never carries a mix of two runs' evidence. Does not commit; the
    caller commits alongside the rest of that analysis's persistence so both succeed or
    fail together.
    """
    await session.execute(delete(Evidence).where(Evidence.paper_id == paper_id))
    for row in accepted_rows:
        session.add(Evidence(paper_id=paper_id, prompt_version=prompt_version_str, **row))


def full_text_from_chunks(chunks: list) -> str:
    """Join a paper's ``fulltext_chunks`` into one string, in order.

    Neither ``analyze_all_papers`` below nor the write job's own grounding step
    (``app.services.writing._ground_selected_papers``) calls this any more: both now
    pass the paper's raw chunks straight to the analysis agent, which
    shows them numbered so it can name a ``chunk_index`` for each evidence item, rather
    than one joined string. Kept as the one place a plain, ordered full-text string is
    ever assembled from chunks, for any caller that genuinely wants that (unlike the
    analysis agent's own numbered-chunk view, ``_build_numbered_chunks``).
    """
    return "\n\n".join(
        c["text"] for c in (chunks or []) if isinstance(c, dict) and "text" in c
    )


def format_authors(authors: object) -> str:
    """Author names as one display string.

    Tolerates a list of plain strings, a list of ``{"name": ...}`` dicts, a mix of the
    two, or anything else (stringified as-is). Shared by ``analyze_all_papers`` and the
    write job's grounding step.
    """
    if isinstance(authors, list):
        return ", ".join(
            a if isinstance(a, str) else (a.get("name", "") if isinstance(a, dict) else str(a))
            for a in authors
        )
    return str(authors) if authors else ""


def is_full_text_analysis(analysis: dict | None) -> bool:
    """True when *analysis* carries real full-text analysis content, as opposed to the
    abstract-only placeholder ``build_abstract_only_analysis`` produces (every field but
    ``key_findings`` left empty).

    Used by the write job's grounding step (``app.services.writing.generate_section``)
    to decide whether a paper that already has ``fulltext_chunks`` still
    needs to be analysed with its full text before a section is written from it.
    """
    if not analysis:
        return False
    return bool(
        analysis.get("methodology")
        or analysis.get("theoretical_framework")
        or analysis.get("claims_and_evidence")
        or analysis.get("builds_on")
        or analysis.get("limitations")
        or analysis.get("future_directions")
        or analysis.get("key_quotes_with_citations")
        or analysis.get("themes")
    )


async def analyze_all_papers(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    expertise_level: str | None = None,
) -> None:
    """Background task: deep-analyze all papers in a project library.

    For each paper:
    - Skips if deep_analysis already stored (idempotent).
    - Uses full-text analysis when fulltext_status == "acquired".
    - Falls back to abstract-only analysis otherwise.
    - Commits after each paper so progress survives crashes.
    """
    async with session_factory() as session:
        await task_service.update_job_status(
            session,
            job_id,
            JobStatus.running,
            progress=0.0,
            progress_message="Starting deep analysis...",
        )

    try:
        # Load all papers for this project
        async with session_factory() as session:
            result = await session.execute(
                select(Paper)
                .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
                .where(ProjectPaper.project_id == project_id)
            )
            papers = list(result.scalars().all())

        total = len(papers)
        semaphore = asyncio.Semaphore(max(1, settings.analysis_concurrency))
        progress_lock = asyncio.Lock()
        state = {"analyzed": 0, "completed": 0, "cancelled": False}

        async def _record_progress() -> None:
            async with progress_lock:
                state["completed"] += 1
                done = state["completed"]
                every = max(1, settings.job_progress_update_every)
                if done % every and done != total:
                    return
                async with session_factory() as progress_session:
                    if await task_service.should_abort(progress_session, job_id):
                        state["cancelled"] = True
                        return
                    await task_service.update_job_status(
                        progress_session,
                        job_id,
                        JobStatus.running,
                        progress=done / total if total else 1.0,
                        progress_message=f"Analyzed {state['analyzed']}/{total} papers",
                    )

        async def _analyze_one(paper_id: UUID) -> None:
            # `_record_progress()` must NEVER be awaited while a DB session from this
            # function is still open — it takes an in-process lock and then opens its
            # own session, which deadlocks the pool once concurrent tasks outnumber
            # available connections. So every session block below only extracts plain
            # values into locals; progress is recorded exactly once, in the `finally`,
            # strictly after all sessions opened by this call have been closed. Every
            # failure branch is caught locally so one bad paper never aborts the job.
            try:
                missing = False
                already = False
                title = authors_str = abstract = ""
                year = None
                meta: dict = {}
                fulltext_status = None

                async with session_factory() as session:
                    paper = (
                        await session.execute(select(Paper).where(Paper.id == paper_id))
                    ).scalar_one_or_none()
                    if paper is None:
                        missing = True
                    else:
                        already = bool(paper.metadata_ and "deep_analysis" in paper.metadata_)
                        title = paper.title or ""
                        authors_str = format_authors(paper.authors or [])
                        year = paper.year
                        meta = dict(paper.metadata_ or {})
                        fulltext_status = meta.get("fulltext_status")
                        abstract = paper.abstract or ""

                if missing:
                    return

                # Idempotency check — cheap, before any LLM call.
                if already:
                    logger.debug(
                        "Skipping paper %s — deep_analysis already present", paper_id
                    )
                    state["analyzed"] += 1
                    return

                analysis_dict = None
                evidence_rows: list[dict] | None = None
                if fulltext_status == "acquired":
                    async with semaphore:
                        if state["cancelled"]:
                            return
                        try:
                            chunks = meta.get("fulltext_chunks", [])
                            deep_result = await analyze_paper_text(
                                title, authors_str, year, chunks,
                                expertise_level=expertise_level,
                            )
                            analysis_dict = deep_result.model_dump()
                            raw_evidence = analysis_dict.pop("evidence", [])
                            evidence_rows, _rejected = validate_evidence_items(
                                raw_evidence, chunks
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to deep-analyze paper %s: %s", paper_id, exc
                            )
                            return
                else:
                    try:
                        analysis_dict = build_abstract_only_analysis(
                            title, authors_str, year, abstract
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to build abstract-only analysis for paper %s: %s",
                            paper_id,
                            exc,
                        )
                        return

                # Commit per paper so progress survives a crash.
                try:
                    async with session_factory() as session:
                        paper = (
                            await session.execute(
                                select(Paper).where(Paper.id == paper_id)
                            )
                        ).scalar_one_or_none()
                        if paper is None:
                            return
                        metadata = dict(paper.metadata_ or {})
                        metadata["deep_analysis"] = analysis_dict
                        paper.metadata_ = metadata
                        if evidence_rows is not None:
                            await replace_paper_evidence(
                                session, paper_id, evidence_rows,
                                analysis_prompt_version(expertise_level),
                            )
                        await session.commit()
                except Exception as exc:
                    logger.warning(
                        "Failed to persist deep analysis for paper %s: %s", paper_id, exc
                    )
                    return

                state["analyzed"] += 1
                logger.info(
                    "Deep-analyzed paper %s (%d/%d)", paper_id, state["analyzed"], total
                )
            except Exception:
                logger.exception("Unexpected error deep-analyzing paper %s", paper_id)
            finally:
                await _record_progress()

        results = await asyncio.gather(
            *(_analyze_one(p.id) for p in papers), return_exceptions=True
        )
        for paper, outcome in zip(papers, results):
            if isinstance(outcome, BaseException):
                logger.error(
                    "Deep analysis task for paper %s raised unexpectedly: %s",
                    paper.id,
                    outcome,
                    exc_info=outcome,
                )

        if state["cancelled"]:
            async with session_factory() as session:
                await task_service.update_job_status(
                    session,
                    job_id,
                    JobStatus.cancelled,
                    progress=0.0,
                    progress_message="Deep analysis cancelled.",
                )
            return

        analyzed = state["analyzed"]

        # Mark job completed
        async with session_factory() as session:
            await task_service.update_job_status(
                session,
                job_id,
                JobStatus.completed,
                progress=1.0,
                progress_message=f"Deep analysis complete: {analyzed}/{total} papers analyzed",
                result={"analyzed": analyzed, "total": total},
            )

    except Exception as exc:
        logger.exception("Deep analysis job %s failed: %s", job_id, exc)
        async with session_factory() as session:
            await task_service.update_job_status(
                session,
                job_id,
                JobStatus.failed,
                error=str(exc),
            )
