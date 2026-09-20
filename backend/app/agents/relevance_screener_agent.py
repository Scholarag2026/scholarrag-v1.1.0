"""Relevance screener agent -- protocol-eligibility screening with three statuses.

The screener asks whether the shown title and abstract fail a numbered criterion of the
review's own protocol, rather than whether a record is topically relevant. Two prompt
versions are kept side by side, the same pattern used by ``claim_verification_agent.py``:
``SCREENER_PROMPT_V1`` is the original PRISMA-worded binary prompt, frozen byte-exact so
``prompt_version(SCREENER_PROMPT_V1)`` stays ``sha256:3c422c45cd8b`` for anyone re-scoring
the v1 evaluation runs. ``SCREENER_PROMPT_V2`` is a protocol-eligibility prompt: default
INCLUDE, an EXCLUDE must name a criterion id and a verbatim quote, and a record the shown
text cannot decide is NEEDS_REVIEW rather than a soft EXCLUDE; kept byte-identical so
``prompt_version(SCREENER_PROMPT_V2)`` stays ``sha256:fb9de89e0534`` for anyone re-scoring
the v2 evaluation runs. ``SCREENER_PROMPT_V3`` adds the anchor ledger and the title-only
TOPIC exclude on top of v2's rules, and is the version production actually runs:
``SCREENER_PROMPT`` names whichever version is currently active (v3);
``SCREENER_PROMPT_VERSION`` is derived from it.

The abstract shown to the model is capped at ``ABSTRACT_CHAR_LIMIT`` characters. Both the
truncation *style* and the record framing around it (criteria headers, per-record layout)
are bound to the prompt version, not to the cap: v1 reproduces the originally committed
layout and ``abstract[:limit] + "..."`` behaviour byte for byte; v2 numbers the criteria and
shows an abstract whole at or under the limit and, above it, a head, a cut marker, and a
tail (a 7:3 split of the limit). ``render_abstract`` and ``build_shown_texts`` are exposed
so a harness can reproduce, and a guard can verify, exactly what the model was shown.
``screen_papers`` takes ``prompt_version`` and ``abstract_limit`` keyword-only, defaulting
to the active v2 prompt and the full cap, so a caller can run either version end to end.

``apply_decision_guard`` is a deterministic, model-free check applied inside
``screen_papers`` to every v2 decision (v1 has no criterion or quote to check, so the guard
is skipped for it): an EXCLUDE whose criterion id is not one of the criteria shown to the
model, or whose quote is not a verbatim substring of that record's shown text, is demoted to
NEEDS_REVIEW. With no protocol criteria at all, the criterion-id half of the anchor is
dropped and a verbatim quote alone anchors the EXCLUDE. A full-text *inclusion* criterion is
never tested at abstract stage at all -- an EXCLUDE naming one is always unanchored,
whatever its quote says -- while a full-text *exclusion* criterion can ground a
NEEDS_REVIEW (never an EXCLUDE) only behind a verbatim quote of explicit contrary evidence;
the same rule applies whether the demotion came from the guard (an EXCLUDE the model should
not have made) or the model already answered NEEDS_REVIEW itself. The guard also checks
first, ahead of every other check below, whether a record has no abstract at all -- a title
alone can never ground an exclusion, whatever criterion the EXCLUDE names. An EXCLUDE there
is demoted to NEEDS_REVIEW with reason ``"no_abstract"`` when its quote is a verbatim
substring of the shown text (including a quote copied only from the title), and with reason
``"unanchored_exclude"`` when the quote is fabricated: the same distinction the
full-text-exclusion check above makes, so a no-abstract EXCLUDE that also invents its quote
is not owed the gentler label. The guard never produces any other transition -- it never
turns an INCLUDE or a NEEDS_REVIEW into anything else, and it never produces an INCLUDE
from an EXCLUDE. Every
guard conversion is counted on the batch result as ``guard_conversions``, with the
per-record flags in ``guard_applied``.

Every decision carries a short reason and every batch carries the provenance of the LLM
call that produced it (which model actually answered, temperature, prompt version, token
usage). A failed call raises :class:`ScreeningError` so the pipeline can record the batch
as *unscreened* instead of silently including everything.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai import Agent, PromptedOutput

logger = logging.getLogger(__name__)

from app.agents.model_config import (
    AGENT_RETRIES,
    DETERMINISTIC_FAST_MODEL_SETTINGS,
    SECOND_PASS_V2_MODEL_SETTINGS,
    build_deepseek_model,
    prompt_version,
)
from app.config import settings
from app.schemas.provenance import LLMCallProvenance, provenance_from_run

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

#: The original binary INCLUDE/EXCLUDE prompt, frozen byte-exact. Kept exported so
#: ``prompt_version(SCREENER_PROMPT_V1)`` stays ``sha256:3c422c45cd8b`` for anyone
#: re-scoring the v1 evaluation runs.
SCREENER_PROMPT_V1 = """\
You are a systematic review screener operating under the PRISMA framework for title-and-abstract
screening.

For each paper listed, decide INCLUDE or EXCLUDE based on whether the paper DIRECTLY addresses
or substantively contributes to the user's specific research topic.

Rules:
- INCLUDE if the paper's title and abstract show it directly discusses, applies, or investigates
  the specific topic. The paper must have a clear, direct connection — not just share a broad
  domain.
- EXCLUDE if the paper is about a different topic that merely shares some keywords or a general
  field. For example, if the topic is "AI in translation studies", a paper about general deep
  learning architectures or computer vision should be EXCLUDED even though it involves AI.
- When genuinely uncertain whether the paper addresses the specific topic, INCLUDE.
- A paper that uses methods FROM the topic's domain but APPLIES them elsewhere is NOT relevant.
  (e.g., a paper using NLP for medical diagnosis is NOT relevant to "AI in translation studies")
- Judge each paper on its title AND abstract together.

Output valid JSON with a "decisions" list containing exactly one object per paper, in the same
order as the input list. Each object has:
- "decision": "INCLUDE" or "EXCLUDE"
- "reason": at most 20 words naming the evidence in the title/abstract that drove the decision"""

#: Version hash of the frozen v1 prompt.
SCREENER_PROMPT_V1_VERSION = prompt_version(SCREENER_PROMPT_V1)

#: The active protocol-eligibility prompt. Default INCLUDE; an EXCLUDE
#: must name a criterion id and quote the shown text verbatim; a record the shown text
#: cannot decide is NEEDS_REVIEW. Flip ``SCREENER_PROMPT`` (and only that alias) to roll
#: back to v1 -- and flip the renderer call sites to ``prompt_version="v1"`` with it, since
#: the truncation style is bound to the prompt version, not to the cap.
SCREENER_PROMPT_V2 = """\
You are a systematic review screener performing title-and-abstract screening against a
written eligibility protocol.

The user turn gives the research question, the numbered inclusion criteria I1, I2, ... and
the numbered exclusion criteria E1, E2, ..., then the records.

Rules:
- Default to INCLUDE. A record is INCLUDE unless the shown title and abstract give you a
  specific ground to exclude it, or leave you unable to decide.
- EXCLUDE only when the shown text itself fails a numbered criterion. An EXCLUDE must name
  exactly one criterion id (for example "E3" or "I2") and must quote a phrase copied
  character for character from the shown title or abstract that shows the failure. Do not
  paraphrase the quote and do not quote text you were not shown. Before returning an
  EXCLUDE, check that the phrase you quoted states the failure you wrote in the reason; if
  it does not, you do not have a ground.
- When a criterion gives a category by listing kinds or examples, whether with "included",
  "such as", "for example", or a parenthetical, the list shows what the category covers;
  it is not the whole of it. A feature of the shown text that is of a listed kind
  satisfies the criterion even when it is called something else, and a therapy or measure
  satisfies a definition when the shown text says it has the component the definition
  names, whatever the therapy or measure is called. EXCLUDE on such a criterion only when
  the shown text names something plainly outside the category; when you cannot tell,
  answer NEEDS_REVIEW. This bullet governs only a criterion that is not listed under a
  Full-text heading; a criterion listed under a Full-text heading is governed by the two
  Full-text bullets below instead, not by this one.
- A criterion phrased as an absence ("did not use", "did not report", "no X") is failed
  only when the shown text shows the thing missing from the study as a whole. If the shown
  text names the thing anywhere, in any arm, condition, measure or outcome, the criterion
  is met, whatever else the study also contains. If the shown text does not mention the
  thing at all, you cannot tell: answer NEEDS_REVIEW, not EXCLUDE. This bullet governs only
  a criterion that is not listed under a Full-text heading; silence about a criterion
  listed under a Full-text heading is governed by the two Full-text bullets below instead.
- Judge the numbered criteria that are not listed under a Full-text heading, and the
  research question, before anything else. If the shown text fails one of those criteria,
  or the record is off topic, answer EXCLUDE on that ground and stop; do not consider a
  full-text criterion. A record is off topic when the shown text is about a different
  population, phenomenon or outcome from the one the research question names. It is not
  off topic merely because the shown text fails to confirm a detail some criterion asks
  about: if your reason for calling a record off topic could be written as "criterion Cn
  is not confirmed", then judge Cn under its own rules instead, and if Cn is a full-text
  criterion, do not EXCLUDE at all. A record with no abstract can never ground an EXCLUDE
  on any criterion, and can never be judged off topic: answer NEEDS_REVIEW. An off-topic
  EXCLUDE names the criterion the mismatch fails, or "TOPIC" when no criterion covers it,
  and quotes the shown text verbatim like any other EXCLUDE. Only a record that passes
  every one of those criteria and is on topic is judged against the Full-text criteria
  below.
- A criterion listed under Full-text inclusion criteria cannot be decided from a title and
  abstract. Judge the record on its remaining criteria and the research topic; if it
  passes, answer INCLUDE and list any such criterion the shown text does not already
  confirm in "to_confirm". Silence about one of these is not a reason to EXCLUDE and not a
  reason for NEEDS_REVIEW.
- A criterion listed under Full-text exclusion criteria can never ground an EXCLUDE.
  Answer NEEDS_REVIEW, naming the criterion in "criterion" and quoting the evidence
  verbatim in "quote", only when the shown text states explicit contrary evidence against
  it, for example naming a different instrument as the only one used. Silence about one of
  these has no effect on your decision.
- NEEDS_REVIEW when the shown text cannot decide: no abstract, a cut abstract, or a record
  on the right subject whose shown text does not say whether it meets the criteria.
  NEEDS_REVIEW is not a soft EXCLUDE; it means a human must read more than you were shown.
- A cut abstract is marked with [...]. Absence of evidence in a cut abstract is not evidence
  against a criterion; use NEEDS_REVIEW, not EXCLUDE.
- Judge each record on its title and abstract together.
- Screen each record on its own. Do not compare records and do not aim at any number of
  inclusions.

Output valid JSON with a "decisions" list containing exactly one object per record, in the
same order as the input list. Each object has:
- "decision": "INCLUDE", "EXCLUDE" or "NEEDS_REVIEW"
- "criterion": the criterion id for an EXCLUDE, or for a NEEDS_REVIEW that turns on a
  failed criterion; "" for INCLUDE and for a NEEDS_REVIEW naming no criterion (for
  example, no abstract)
- "quote": for an EXCLUDE, and for any NEEDS_REVIEW that names a criterion, a phrase
  copied verbatim from the shown title or abstract, "" otherwise
- "to_confirm": on an INCLUDE, the full-text inclusion criterion ids the shown text does
  not already confirm; [] otherwise
- "reason": at most 20 words naming the evidence that drove the decision"""

#: ``SCREENER_PROMPT_V2`` stays byte-exact above -- this is a new prompt
#: beside it, not an edit to it, so ``prompt_version(SCREENER_PROMPT_V2)`` is unchanged and
#: every v2 evaluation run stays reproducible. Two additions over v2:
#:
#: Element 1 (anchored INCLUDE). On an INCLUDE, the model also returns an ``"anchors"``
#: list: one entry per slot -- ``population`` (who or what the study itself studied),
#: ``subject`` (the intervention, exposure or phenomenon the research question names) and
#: ``outcome`` (a result the study reports, of the kind the research question asks about) --
#: each either a verbatim quote for that slot or the literal ``"not_established"``. An
#: INCLUDE was previously unfalsifiable: the v2 schema asks for a criterion id and a quote
#: only on an EXCLUDE or a criterion-bearing NEEDS_REVIEW (module docstring), so nothing in
#: the loop ever asked whether the shown text actually supports the reasons a record was
#: kept. The ledger is emitted on every INCLUDE regardless of which slots a given screening
#: run treats as anchored (:func:`apply_decision_guard`'s ``anchored_slots``), so the anchor
#: set can be chosen, and re-scored, after the fact from the same run rows.
#:
#: Element 3 (title-only TOPIC exclude). v2 forbids judging a no-abstract record off topic
#: at all ("A record with no abstract can never ground an EXCLUDE on any criterion, and can
#: never be judged off topic: answer NEEDS_REVIEW"), because a title cannot establish that a
#: numbered criterion is *failed*. It can establish that the record is about something else
#: entirely, which is a different judgement (a 140-record sample: the human majority excludes
#: 15 of 31 no-abstract records on the title alone, 12 of them because the title plainly
#: names a different population or phenomenon). v3 allows exactly that one
#: narrow exception: an EXCLUDE naming criterion ``"TOPIC"`` on a no-abstract record, quoting
#: the title verbatim. :func:`apply_decision_guard`'s no-abstract rule gains the matching
#: carve-out; every other no-abstract EXCLUDE is demoted exactly as under v2.
SCREENER_PROMPT_V3 = """\
You are a systematic review screener performing title-and-abstract screening against a
written eligibility protocol.

The user turn gives the research question, the numbered inclusion criteria I1, I2, ... and
the numbered exclusion criteria E1, E2, ..., then the records.

Rules:
- Default to INCLUDE. A record is INCLUDE unless the shown title and abstract give you a
  specific ground to exclude it, or leave you unable to decide.
- EXCLUDE only when the shown text itself fails a numbered criterion. An EXCLUDE must name
  exactly one criterion id (for example "E3" or "I2") and must quote a phrase copied
  character for character from the shown title or abstract that shows the failure. Do not
  paraphrase the quote and do not quote text you were not shown. Before returning an
  EXCLUDE, check that the phrase you quoted states the failure you wrote in the reason; if
  it does not, you do not have a ground.
- When a criterion gives a category by listing kinds or examples, whether with "included",
  "such as", "for example", or a parenthetical, the list shows what the category covers;
  it is not the whole of it. A feature of the shown text that is of a listed kind
  satisfies the criterion even when it is called something else, and a therapy or measure
  satisfies a definition when the shown text says it has the component the definition
  names, whatever the therapy or measure is called. EXCLUDE on such a criterion only when
  the shown text names something plainly outside the category; when you cannot tell,
  answer NEEDS_REVIEW. This bullet governs only a criterion that is not listed under a
  Full-text heading; a criterion listed under a Full-text heading is governed by the two
  Full-text bullets below instead, not by this one.
- A criterion phrased as an absence ("did not use", "did not report", "no X") is failed
  only when the shown text shows the thing missing from the study as a whole. If the shown
  text names the thing anywhere, in any arm, condition, measure or outcome, the criterion
  is met, whatever else the study also contains. If the shown text does not mention the
  thing at all, you cannot tell: answer NEEDS_REVIEW, not EXCLUDE. This bullet governs only
  a criterion that is not listed under a Full-text heading; silence about a criterion
  listed under a Full-text heading is governed by the two Full-text bullets below instead.
- Judge the numbered criteria that are not listed under a Full-text heading, and the
  research question, before anything else. If the shown text fails one of those criteria,
  or the record is off topic, answer EXCLUDE on that ground and stop; do not consider a
  full-text criterion. A record is off topic when the shown text is about a different
  population, phenomenon or outcome from the one the research question names. It is not
  off topic merely because the shown text fails to confirm a detail some criterion asks
  about: if your reason for calling a record off topic could be written as "criterion Cn
  is not confirmed", then judge Cn under its own rules instead, and if Cn is a full-text
  criterion, do not EXCLUDE at all. A record with no abstract can never ground an EXCLUDE on
  a numbered criterion. On a record with no abstract, answer EXCLUDE with criterion "TOPIC"
  only when the title itself plainly names a population or a phenomenon other than the one
  the research question names, quoting the title verbatim; otherwise answer NEEDS_REVIEW.
  An off-topic EXCLUDE names the criterion the mismatch fails, or "TOPIC" when no criterion
  covers it, and quotes the shown text verbatim like any other EXCLUDE. Only a record that
  passes every one of those criteria and is on topic is judged against the Full-text
  criteria below.
- A criterion listed under Full-text inclusion criteria cannot be decided from a title and
  abstract. Judge the record on its remaining criteria and the research topic; if it
  passes, answer INCLUDE and list any such criterion the shown text does not already
  confirm in "to_confirm". A record whose "to_confirm" is not empty is queued for
  full-text review rather than included outright, whatever else its shown text says --
  name only what is genuinely still unconfirmed. Silence about one of these is not a
  reason to EXCLUDE and not a reason for NEEDS_REVIEW.
- A criterion listed under Full-text exclusion criteria can never ground an EXCLUDE.
  Answer NEEDS_REVIEW, naming the criterion in "criterion" and quoting the evidence
  verbatim in "quote", only when the shown text states explicit contrary evidence against
  it, for example naming a different instrument as the only one used. Silence about one of
  these has no effect on your decision.
- On an INCLUDE, also return an "anchors" list with exactly three entries, one per slot:
  "population" (the phrase naming who or what the study itself studied), "subject" (the
  phrase naming the intervention, exposure or phenomenon the research question names), and
  "outcome" (the phrase naming a result the study reports, of the kind the research
  question asks about). Each entry either quotes the shown text character for character for
  that slot, or, when the shown text does not establish it, answers "not_established" for
  that slot instead. Do not paraphrase a quote and do not quote text you were not shown.
- NEEDS_REVIEW when the shown text cannot decide: no abstract, a cut abstract, or a record
  on the right subject whose shown text does not say whether it meets the criteria.
  NEEDS_REVIEW is not a soft EXCLUDE; it means a human must read more than you were shown.
- A cut abstract is marked with [...]. Absence of evidence in a cut abstract is not evidence
  against a criterion; use NEEDS_REVIEW, not EXCLUDE.
- Judge each record on its title and abstract together.
- Screen each record on its own. Do not compare records and do not aim at any number of
  inclusions.

Output valid JSON with a "decisions" list containing exactly one object per record, in the
same order as the input list. Each object has:
- "decision": "INCLUDE", "EXCLUDE" or "NEEDS_REVIEW"
- "criterion": the criterion id for an EXCLUDE, or for a NEEDS_REVIEW that turns on a
  failed criterion; "" for INCLUDE and for a NEEDS_REVIEW naming no criterion (for
  example, no abstract)
- "quote": for an EXCLUDE, and for any NEEDS_REVIEW that names a criterion, a phrase
  copied verbatim from the shown title or abstract, "" otherwise
- "to_confirm": on an INCLUDE, the full-text inclusion criterion ids the shown text does
  not already confirm; a non-empty list means this record is queued for full-text review,
  not included outright; [] otherwise
- "anchors": on an INCLUDE only, a list of {"slot": "population"|"subject"|"outcome",
  "quote": a phrase copied verbatim from the shown title or abstract for that slot, or
  "not_established"}; [] otherwise
- "reason": at most 20 words naming the evidence that drove the decision"""

#: Version hash of the v3 prompt.
SCREENER_PROMPT_V3_VERSION = prompt_version(SCREENER_PROMPT_V3)

#: The active prompt: v3, with the anchor ledger, the title-only TOPIC exclude and (via
#: ``apply_decision_guard``'s unconditional full-text-to-confirm check, ``apply_type_demotion``,
#: ``apply_table_of_contents_demotion`` and the inclusion-only second pass, all wired into
#: ``app.services.smart_search.run_smart_search``) applying to every production Smart Search
#: job, not only an evaluation harness run given ``--prompt-version v3`` explicitly.
#: ``SCREENER_PROMPT_V2`` stays byte-identical and its own digest test still
#: pins ``sha256:fb9de89e0534``, so every v2 evaluation run stays reproducible; flip this one
#: constant (and only this constant) to roll back to v2, or to v1.
SCREENER_PROMPT = SCREENER_PROMPT_V3

#: Version hash of the active system prompt, recorded in every screening provenance record.
SCREENER_PROMPT_VERSION = prompt_version(SCREENER_PROMPT)

#: Which renderer style ``_build_screening_prompt`` and ``build_shown_texts`` use by
#: default. Kept alongside ``SCREENER_PROMPT`` because the style is bound to the version.
#: v3 reuses v2's rendering byte for byte (module docstring), so this only ever needs to be
#: "v1" or "v2"/"v3" -- both of the latter share one renderer branch.
_ACTIVE_PROMPT_VERSION = "v3"

#: System prompt text by ``prompt_version``, so :func:`screen_papers` and
#: :func:`get_relevance_screener_agent` can select the matching one: a v1 call must send
#: the v1 system prompt, not v2's.
_PROMPTS_BY_VERSION: dict[str, str] = {
    "v1": SCREENER_PROMPT_V1,
    "v2": SCREENER_PROMPT_V2,
    "v3": SCREENER_PROMPT_V3,
}

# --------------------------------------------------------------------------------------
# Abstract cap
# --------------------------------------------------------------------------------------

#: The abstract character budget shown to the model, set to 10,000: this shows every
#: abstract-bearing record whole on the development set and the held-out set (``Fong_2021``
#: 1.0000, ``van_Dis_2020`` 1.0000) and leaves exactly one over-long abstract each on the
#: three legacy or withdrawn corpora. This also fixes the cut-abstract rule for records a
#: smaller cap cut for no reason (record 856's only LASSI mention sat at character 3,265,
#: inside a 3,000-character cut region). One constant: everything else derives from it.
ABSTRACT_CHAR_LIMIT = 10000

#: " [...] ", the marker placed between the kept head and tail of a v2-style cut abstract.
ABSTRACT_CUT_MARKER = " [...] "

#: "(no abstract)", the placeholder :func:`build_shown_texts` and the v1 branch of
#: :func:`_build_screening_prompt` render in place of an empty or missing abstract. The
#: guard rule (:func:`_shown_abstract_is_missing`) keys off this exact string, so it is
#: named once here rather than repeated as a literal at each render and check site.
NO_ABSTRACT_PLACEHOLDER = "(no abstract)"


def _head_tail_split(limit: int) -> tuple[int, int]:
    """7:3 split of a character budget between a kept head and a kept tail."""
    head = round(limit * 7 / 10)
    return head, limit - head


ABSTRACT_HEAD_CHARS, ABSTRACT_TAIL_CHARS = _head_tail_split(ABSTRACT_CHAR_LIMIT)

MAX_REASON_LENGTH = 300
MAX_CRITERION_LENGTH = 8
MAX_QUOTE_LENGTH = 300
NO_DECISION_REASON = "no decision returned"

#: Appended to a decision's reason when the guard demotes an unanchored EXCLUDE.
GUARD_DEMOTION_NOTE = "guard: EXCLUDE not anchored to a shown criterion id and verbatim quote"

#: One cached agent per prompt version (``"v1"``, ``"v2"``), so a v1 call and a v2 call
#: never overwrite each other's cached system prompt.
_agents: dict[str, Agent[None, ScreeningDecisions]] = {}


def render_abstract(abstract: str, *, prompt_version: str, limit: int) -> str:
    """Render an abstract for the user turn.

    The truncation style is bound to the prompt version, not to the cap, so the version
    and the limit are taken as separate arguments.

    v1 reproduces the originally committed behaviour exactly: ``abstract[:limit] + "..."``
    when the abstract is longer than ``limit``, untouched otherwise. A v1 render at
    ``limit=500`` is therefore byte-identical to the six committed v1 runs.

    v2 shows the abstract whole when it is at or under ``limit``, and otherwise shows a
    head, :data:`ABSTRACT_CUT_MARKER`, and a tail split 7:3 of ``limit``. v3 reuses the v2
    rendering byte for byte, only its system prompt text differs.
    """
    abstract = abstract or ""
    if prompt_version == "v1":
        if len(abstract) > limit:
            return abstract[:limit] + "..."
        return abstract
    if prompt_version in ("v2", "v3"):
        if len(abstract) <= limit:
            return abstract
        head_chars, tail_chars = _head_tail_split(limit)
        head = abstract[:head_chars]
        tail = abstract[-tail_chars:] if tail_chars > 0 else ""
        return f"{head}{ABSTRACT_CUT_MARKER}{tail}"
    raise ValueError(f"unknown prompt_version: {prompt_version!r} (expected 'v1', 'v2' or 'v3')")


def build_shown_texts(
    papers: list[dict],
    *,
    prompt_version: str = _ACTIVE_PROMPT_VERSION,
    limit: int = ABSTRACT_CHAR_LIMIT,
) -> list[str]:
    """Per paper, the title and abstract exactly as rendered into the user turn.

    ``apply_decision_guard`` and any harness both read this instead of re-deriving the
    rendering, so neither can drift from what the model actually saw. The v1 user turn
    indents each "Title:"/"Abstract:" line by two spaces (``_build_screening_prompt``'s
    ``"Paper N:"`` layout); this helper matches that margin when ``prompt_version`` is
    ``"v1"`` so its return value is a literal substring of the v1 prompt too, not only
    the v2 one.
    """
    indent = "  " if prompt_version == "v1" else ""
    texts: list[str] = []
    for paper in papers:
        title = paper.get("title") or "(no title)"
        abstract = (paper.get("abstract") or "").strip() or NO_ABSTRACT_PLACEHOLDER
        abstract = render_abstract(abstract, prompt_version=prompt_version, limit=limit)
        texts.append(f"{indent}Title: {title}\n{indent}Abstract: {abstract}")
    return texts


# --------------------------------------------------------------------------------------
# The verbatim-quote check
# --------------------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")

# NFKC alone folds none of these -- the dash and quote variants are canonical (not
# compatibility) code points -- which is why the map is explicit. Mirrored
# byte-for-byte in ``evaluation/common.py``'s ``_UNICODE_FOLD_MAP`` because the backend
# cannot import ``evaluation/``.
_UNICODE_FOLD_MAP: dict[str, str] = {
    # hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar, minus sign
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-",
    # left/right single quotation mark, single low-9 quotation mark, modifier letter apostrophe
    "‘": "'", "’": "'", "‛": "'", "ʼ": "'",
    # left/right double quotation mark, double high-reversed-9 quotation mark
    "“": '"', "”": '"', "‟": '"',
    # no-break space, thin space, narrow no-break space
    " ": " ", " ": " ", " ": " ",
    # horizontal ellipsis
    "…": "...",
}
_UNICODE_FOLD_RE = re.compile("|".join(re.escape(k) for k in _UNICODE_FOLD_MAP))


def _normalise_unicode_punctuation(text: str) -> str:
    """NFKC-normalise, then fold Unicode dashes/quotes/spaces/ellipsis to ASCII.

    Mirrors ``evaluation/common.py:normalise_unicode_punctuation`` exactly; the backend
    keeps a private copy because it cannot import ``evaluation/``.
    """
    text = unicodedata.normalize("NFKC", text or "")
    return _UNICODE_FOLD_RE.sub(lambda m: _UNICODE_FOLD_MAP[m.group(0)], text)


def _normalise_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _quote_is_verbatim(quote: str | None, shown_text: str | None) -> bool:
    """Unicode-folded, whitespace-normalised, casefolded substring containment.

    Mirrors ``evaluation/common.py:348 quote_is_verbatim(..., casefold=True)`` exactly;
    the backend keeps a private copy because it cannot import ``evaluation/``.
    """
    if not quote or not shown_text:
        return False
    q = _normalise_whitespace(_normalise_unicode_punctuation(quote)).casefold()
    c = _normalise_whitespace(_normalise_unicode_punctuation(shown_text)).casefold()
    if not q:
        return False
    return q in c


class ScreeningError(Exception):
    """The screener could not produce decisions for a batch (LLM error, invalid output)."""


#: The three anchor-ledger slots ``SCREENER_PROMPT_V3`` asks for on an INCLUDE, chosen to
#: catch unfalsifiable inclusions (17 found in a 40-record sample). Not a closed vocabulary
#: the model is constrained to (``AnchorEntry.slot`` is a plain string, not a ``Literal``): an
#: unrecognised slot name is simply invisible to :func:`apply_decision_guard`'s
#: ``anchored_slots`` check rather than failing the whole batch's structured output.
ANCHOR_SLOTS: tuple[str, ...] = ("population", "subject", "outcome")

#: The literal value an :class:`AnchorEntry`'s ``quote`` carries when the shown text does
#: not establish that slot -- distinct from ``""`` (a model returning an empty string is
#: truncation or a mistake, not a considered "not established" answer, and would otherwise
#: fail the verbatim check the same way any other empty quote does).
NOT_ESTABLISHED = "not_established"


class AnchorEntry(BaseModel):
    """One line of the anchor ledger: a slot, and either a verbatim quote for it or
    the literal :data:`NOT_ESTABLISHED`."""

    slot: str = Field(default="", max_length=16)
    quote: str = Field(default="", max_length=MAX_QUOTE_LENGTH)

    @field_validator("slot", mode="before")
    @classmethod
    def _normalise_slot(cls, value: object) -> object:
        if value is None:
            return ""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("quote", mode="before")
    @classmethod
    def _normalise_quote(cls, value: object) -> object:
        """Truncate like any other quote, but recognise :data:`NOT_ESTABLISHED` under any
        casing/whitespace first, so ``"Not_Established "`` still reads as the sentinel
        rather than as a (fabricated) quote of that literal text."""
        if value is None:
            return ""
        if isinstance(value, str):
            if value.strip().casefold() == NOT_ESTABLISHED:
                return NOT_ESTABLISHED
            if len(value) > MAX_QUOTE_LENGTH:
                return value[:MAX_QUOTE_LENGTH]
        return value


class PaperDecision(BaseModel):
    """One screening decision: INCLUDE, EXCLUDE with a criterion and a quote, or
    NEEDS_REVIEW."""

    decision: Literal["INCLUDE", "EXCLUDE", "NEEDS_REVIEW"]
    criterion: str = Field(
        default="",
        max_length=MAX_CRITERION_LENGTH,
        description=(
            "The criterion id for an EXCLUDE, or for a NEEDS_REVIEW that turns on a failed "
            "criterion (for example 'E3'); '' otherwise (for example, no abstract)."
        ),
    )
    quote: str = Field(
        default="",
        max_length=MAX_QUOTE_LENGTH,
        description=(
            "For an EXCLUDE, and for any NEEDS_REVIEW that names a criterion, a phrase "
            "copied verbatim from the shown text; '' otherwise."
        ),
    )
    reason: str = Field(
        default="",
        max_length=MAX_REASON_LENGTH,
        description="At most 20 words naming the evidence that drove the decision.",
    )
    #: On an INCLUDE, the full-text *inclusion* criterion ids the shown
    #: text does not already confirm; ``[]`` on every other decision. ``screen_papers``
    #: post-processes this (intersects with the protocol's own full-text inclusion ids,
    #: defaults to the full set on an INCLUDE the model left empty, forces ``[]`` on every
    #: non-INCLUDE) -- this field is only what the model itself returned.
    to_confirm: list[str] = Field(default_factory=list)
    #: On an INCLUDE from ``SCREENER_PROMPT_V3``, one
    #: :class:`AnchorEntry` per :data:`ANCHOR_SLOTS` slot; ``[]`` on v1/v2 (neither prompt
    #: asks for it) and on every non-INCLUDE. This field is only what the model itself
    #: returned -- :func:`apply_decision_guard` reads it but never writes it, so a demoted
    #: decision keeps its original ledger for audit the same way ``criterion``/``quote`` do.
    anchors: list[AnchorEntry] = Field(default_factory=list)

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("criterion", mode="before")
    @classmethod
    def _normalise_criterion(cls, value: object) -> object:
        """Strip, upper-case, then truncate -- the same order ``_normalise_decision``
        applies to ``decision``. Without this, a model
        that answers ``"e1"`` or ``" E1"`` fails ``criterion in known_ids`` in
        :func:`apply_decision_guard` for a reason that has nothing to do with screening
        quality, demoting every such EXCLUDE to NEEDS_REVIEW."""
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.strip().upper()
            if len(value) > MAX_CRITERION_LENGTH:
                value = value[:MAX_CRITERION_LENGTH]
        return value

    @field_validator("quote", mode="before")
    @classmethod
    def _truncate_quote(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str) and len(value) > MAX_QUOTE_LENGTH:
            return value[:MAX_QUOTE_LENGTH]
        return value

    @field_validator("reason", mode="before")
    @classmethod
    def _truncate_reason(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str) and len(value) > MAX_REASON_LENGTH:
            return value[:MAX_REASON_LENGTH]
        return value

    @field_validator("to_confirm", mode="before")
    @classmethod
    def _normalise_to_confirm(cls, value: object) -> object:
        """Strip/upper-case each id the same way ``criterion`` is normalised, and drop any
        blank entry, so a model that answers ``["i1", " I3 ", ""]`` still matches the
        protocol's own ``"I1"``/``"I3"`` ids in :func:`screen_papers`."""
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            item = item.strip().upper()
            if len(item) > MAX_CRITERION_LENGTH:
                item = item[:MAX_CRITERION_LENGTH]
            if item:
                out.append(item)
        return out


class ScreeningDecisions(BaseModel):
    """Structured output from the relevance screener agent."""

    decisions: list[PaperDecision] = Field(
        ...,
        description="One decision object per paper, in input order.",
    )


class ScreeningBatchResult(BaseModel):
    """Decisions for one batch, parallel to the input papers, plus the call's provenance."""

    statuses: list[str]
    criteria_ids: list[str]
    quotes: list[str]
    reasons: list[str]
    provenance: LLMCallProvenance
    #: How many decisions the guard demoted from EXCLUDE to NEEDS_REVIEW in this batch, and
    #: which positions, since a demoted row is only weakly inferable from the status alone.
    guard_conversions: int = 0
    guard_applied: list[bool] = []
    #: Which guard demotion, if any, produced this record's decision: ``""`` (no demotion),
    #: ``"no_abstract"``,
    #: ``"full_text_criterion"``, ``"cut_abstract"``, ``"unanchored_exclude"`` or
    #: ``"unquoted_criterion"``. Always parallel to ``statuses`` -- defaulted to a same-length
    #: list of ``""`` when omitted, so every existing caller that never mentions it still
    #: constructs cleanly, but a caller that does supply it must get the length right.
    guard_reasons: list[str] = Field(default_factory=list)
    #: How many of this batch's decisions the model did not return at all, and were padded
    #: with NEEDS_REVIEW. A harness scoring a
    #: ``--prompt-version v1`` run asserts this is zero before treating the run as
    #: comparable to the six originally committed v1 runs, which padded a short tail with
    #: INCLUDE rather than NEEDS_REVIEW.
    padded: int = 0
    #: Provenance of the one re-ask call, set only when
    #: a batch response left at least one paper undecided and :func:`_reask_missing_decisions`
    #: got a usable response back; ``None`` when nothing was missing, or when the re-ask call
    #: itself raised. Kept as a second record rather than merged into ``provenance``, so a
    #: model or fingerprint change between a batch's two calls stays visible instead of being
    #: silently overwritten or dropped -- a caller totalling the batch's real cost adds this
    #: record's ``input_tokens``/``output_tokens`` to ``provenance``'s.
    reask_provenance: LLMCallProvenance | None = None
    #: ``"<ExceptionClass>: <message>"`` when the one re-ask call itself raised (a transport
    #: or model failure), ``None`` when nothing was missing or the re-ask call answered.
    #: Without this, a record still padded because its re-ask
    #: call failed outright is indistinguishable in the results file from one still padded
    #: because the model quietly skipped it again -- a harness chasing the latter with
    #: further re-ask passes is doing useful work, chasing the former is not.
    reask_error: str | None = None
    #: Per record, the full-text inclusion criterion ids still to be
    #: confirmed at full text -- non-empty only on an INCLUDE, already intersected with the
    #: protocol's own full-text inclusion ids and defaulted to the full set when the model
    #: left an INCLUDE's own list empty (see :func:`screen_papers`). Always parallel to
    #: ``statuses``; defaulted to a same-length list of ``[]`` when omitted.
    to_confirm: list[list[str]] = Field(default_factory=list)
    #: Per record, the raw anchor ledger the model
    #: returned (``[{"slot": ..., "quote": ...}, ...]``), kept whatever ``anchored_slots``
    #: the guard checked against or left unchecked -- non-empty only on an INCLUDE from
    #: ``SCREENER_PROMPT_V3``. Always parallel to ``statuses``; defaulted to a same-length
    #: list of ``[]`` when omitted, the same convention ``to_confirm`` uses.
    anchors: list[list[dict[str, str]]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_include(cls, data: object) -> object:
        """Compatibility shim for the ``include: list[bool]`` keyword some test fixtures still
        build ``ScreeningBatchResult(include=..., reasons=..., provenance=...)`` with. When
        ``statuses``
        is absent and ``include`` is present, ``statuses`` is derived from ``include`` and
        ``criteria_ids``/``quotes`` are filled with one empty string per record. Delete this
        shim (and the test that covers it) once those fixtures move to ``statuses=``.
        """
        if isinstance(data, dict) and "statuses" not in data and "include" in data:
            include = data["include"]
            data = {key: value for key, value in data.items() if key != "include"}
            data["statuses"] = ["INCLUDE" if flag else "EXCLUDE" for flag in include]
            data.setdefault("criteria_ids", [""] * len(include))
            data.setdefault("quotes", [""] * len(include))
        if isinstance(data, dict) and not data.get("guard_reasons"):
            statuses = data.get("statuses")
            if isinstance(statuses, list):
                data = {**data, "guard_reasons": [""] * len(statuses)}
        if isinstance(data, dict) and not data.get("to_confirm"):
            statuses = data.get("statuses")
            if isinstance(statuses, list):
                data = {**data, "to_confirm": [[] for _ in statuses]}
        if isinstance(data, dict) and not data.get("anchors"):
            statuses = data.get("statuses")
            if isinstance(statuses, list):
                data = {**data, "anchors": [[] for _ in statuses]}
        return data

    @property
    def include(self) -> list[bool]:
        """``status == "INCLUDE"`` per record; the only automatic route into the library."""
        return [status == "INCLUDE" for status in self.statuses]

    @model_validator(mode="after")
    def _validate_parallel_lengths(self) -> "ScreeningBatchResult":
        """The five per-record lists must stay parallel to each other. Without this,
        ``smart_search.py``'s zip over the batch and these lists truncates silently on a
        mismatch and drops records from the flow identity ``_reconciles`` asserts."""
        n = len(self.statuses)
        if len(self.criteria_ids) != n or len(self.quotes) != n or len(self.reasons) != n:
            raise ValueError(
                "statuses, criteria_ids, quotes and reasons must be the same length "
                f"(got {n}, {len(self.criteria_ids)}, {len(self.quotes)}, {len(self.reasons)})"
            )
        if len(self.guard_reasons) != n:
            raise ValueError(
                "guard_reasons must be the same length as statuses "
                f"(got {len(self.guard_reasons)} for {n})"
            )
        if len(self.to_confirm) != n:
            raise ValueError(
                "to_confirm must be the same length as statuses "
                f"(got {len(self.to_confirm)} for {n})"
            )
        if len(self.anchors) != n:
            raise ValueError(
                "anchors must be the same length as statuses "
                f"(got {len(self.anchors)} for {n})"
            )
        if self.guard_applied and len(self.guard_applied) != n:
            raise ValueError(
                "guard_applied must be empty or the same length as statuses "
                f"(got {len(self.guard_applied)} for {n})"
            )
        return self


#: ``ScreeningBatchResult.guard_reasons`` / return value of :func:`apply_decision_guard`.
#: An EXCLUDE on a record whose rendered abstract is
#: :data:`NO_ABSTRACT_PLACEHOLDER`. The *check* (is the abstract missing at all) runs first
#: and unconditionally, ahead of every other guard reason below -- a record with no abstract
#: has nothing to test any criterion against, on topic or off. This *label* is given only
#: when the EXCLUDE's quote is a verbatim substring of the shown text -- which still covers
#: a quote copied only from the record's *title*, since the title is part of the shown text
#: and would otherwise satisfy the ordinary anchor check at the bottom of the function -- and
#: not when the quote is fabricated, which instead gets :data:`GUARD_REASON_UNANCHORED_EXCLUDE`,
#: since a fabricated quote is not owed the gentler label.
GUARD_REASON_NO_ABSTRACT = "no_abstract"
GUARD_REASON_FULL_TEXT_CRITERION = "full_text_criterion"
GUARD_REASON_CUT_ABSTRACT = "cut_abstract"
GUARD_REASON_UNANCHORED_EXCLUDE = "unanchored_exclude"
#: A model-native NEEDS_REVIEW that names
#: any non-empty criterion id -- a full-text *exclusion* id, an ordinary abstract-stage id,
#: or a full-text *inclusion* id -- but does not back it with a non-empty, verbatim quote.
#: Kept on the decision for audit (unlike an ordinary undecided record, this one at least
#: named a criterion) but not trusted as the amendment's contrary-evidence case. Applying
#: this uniformly, rather than only to a full-text exclusion id, keeps every criterion-naming
#: NEEDS_REVIEW with no verbatim quote out of the undecidable/no-abstract bucket, where it
#: would be indistinguishable from a record the model legitimately could not decide.
GUARD_REASON_UNQUOTED_CRITERION = "unquoted_criterion"
#: An INCLUDE demoted because an anchored slot
#: (:data:`ANCHOR_SLOTS`, chosen per run by ``anchored_slots``) has no entry in the model's
#: own ledger, an entry answering :data:`NOT_ESTABLISHED`, or an entry whose quote is not a
#: verbatim substring of the shown text. The only demotion :func:`apply_decision_guard`
#: applies to an INCLUDE; it still never excludes and never promotes.
GUARD_REASON_UNANCHORED_INCLUDE = "unanchored_include"
#: An INCLUDE demoted by the separate inclusion-only
#: second pass (:func:`confirm_inclusions` / :func:`apply_second_pass_guard`), not by
#: :func:`apply_decision_guard` itself, because it answered ``NOT_ESTABLISHED`` for at least
#: one slot on a direct, one-at-a-time re-ask. Named here, beside the other guard reasons,
#: because it is written into the same ``guard_reason`` column of a screening record.
GUARD_REASON_NOT_ESTABLISHED = "not_established"
#: A record that passes every
#: abstract-stage criterion and is on topic, but carries one or more unconfirmed full-text
#: *inclusion* criteria (:func:`_resolve_to_confirm` on the model's own, pre-guard decision
#: is non-empty), is NEEDS_REVIEW, never INCLUDE -- a full-text inclusion criterion is by
#: definition undecidable from a title and abstract, so an INCLUDE naming one still open is
#: routed to the human queue rather than shipped as a clean inclusion, whatever the prompt
#: itself asked the model to do. Structural: this fires even when the model ignores the v3
#: prompt's own description of the same rule (see ``SCREENER_PROMPT_V3``'s full-text
#: inclusion bullet).
GUARD_REASON_FULL_TEXT_TO_CONFIRM = "full_text_to_confirm"
#: An INCLUDE demoted
#: because OpenAlex's own ``type``/``is_paratext`` marks the record as non-article
#: (:data:`NONARTICLE_TYPES`) or a chapter-list abstract (:func:`_is_table_of_contents`).
#: Applied by a caller *after* ``screen_papers`` returns (:func:`apply_type_demotion`,
#: :func:`apply_table_of_contents_demotion`) -- ``screen_papers`` itself has no OpenAlex
#: metadata to check, only the shown title and abstract.
GUARD_REASON_NONARTICLE_TYPE = "nonarticle_type"
GUARD_REASON_TABLE_OF_CONTENTS = "table_of_contents"

#: The reason-string suffix appended for each guard reason, kept distinct so an exported
#: screening record names *why* a record was routed, not merely that it was.
_GUARD_NOTES: dict[str, str] = {
    GUARD_REASON_NO_ABSTRACT: (
        "guard: EXCLUDE routed to NEEDS_REVIEW -- no abstract is shown, so the title alone "
        "cannot ground an exclusion"
    ),
    GUARD_REASON_FULL_TEXT_CRITERION: (
        "guard: EXCLUDE routed to NEEDS_REVIEW -- this criterion cannot be decided from a "
        "title and abstract"
    ),
    GUARD_REASON_CUT_ABSTRACT: (
        "guard: EXCLUDE routed to NEEDS_REVIEW -- this criterion tests absence and the shown "
        "abstract was cut"
    ),
    GUARD_REASON_UNANCHORED_EXCLUDE: GUARD_DEMOTION_NOTE,
    GUARD_REASON_UNANCHORED_INCLUDE: (
        "guard: INCLUDE routed to NEEDS_REVIEW -- an anchored slot has no verbatim quote, "
        "or was left not_established"
    ),
    GUARD_REASON_NOT_ESTABLISHED: (
        "second pass: INCLUDE routed to NEEDS_REVIEW -- the inclusion-only re-ask answered "
        "not_established for a slot"
    ),
    GUARD_REASON_FULL_TEXT_TO_CONFIRM: (
        "guard: INCLUDE routed to NEEDS_REVIEW -- one or more full-text inclusion "
        "criteria are not yet confirmed by the shown text"
    ),
    GUARD_REASON_NONARTICLE_TYPE: (
        "guard: INCLUDE routed to NEEDS_REVIEW -- OpenAlex marks this record a "
        "non-article type"
    ),
    GUARD_REASON_TABLE_OF_CONTENTS: (
        "guard: INCLUDE routed to NEEDS_REVIEW -- the shown abstract reads as a table of "
        "contents"
    ),
}


def _shown_abstract_is_missing(shown_text: str | None) -> bool:
    """True when the record's rendered abstract is :data:`NO_ABSTRACT_PLACEHOLDER`:
    the record has no abstract at all, empty or missing alike, since
    :func:`build_shown_texts` renders both the same way. ``shown_text`` always ends with the
    abstract line and nothing else, so a suffix check is exact, not a heuristic -- it cannot
    be fooled by a real abstract that merely mentions the placeholder text mid-sentence."""
    return (shown_text or "").endswith(f"Abstract: {NO_ABSTRACT_PLACEHOLDER}")


def _shown_title(shown_text: str | None) -> str:
    """The title line of ``shown_text``, stripped of its ``"Title: "``/indent prefix.
    ``build_shown_texts`` always renders
    ``f"{indent}Title: {title}\\n{indent}Abstract: {abstract}"``, so splitting on the first
    ``"\\nAbstract:"`` and dropping the ``"Title: "`` prefix recovers exactly the title text
    the model was shown, whatever the record's abstract (or lack of one) contains."""
    text = shown_text or ""
    title_line = text.split("\nAbstract:", 1)[0]
    return title_line.split("Title:", 1)[1].strip() if "Title:" in title_line else title_line


def _unanchored_include_slots(
    decision: PaperDecision, shown_text: str, anchored_slots: frozenset[str]
) -> bool:
    """True when at least one of ``anchored_slots`` has no entry in ``decision.anchors``, an
    entry answering :data:`NOT_ESTABLISHED`, or an entry whose quote is not a verbatim
    substring of ``shown_text``. A slot named more than
    once in the model's ledger is read as its last entry, the same "last one wins" rule
    ``PaperDecision``'s own field validators apply nowhere else needs, because nothing else
    here builds a dict from a list the model controls."""
    by_slot = {entry.slot: entry.quote for entry in decision.anchors}
    for slot in anchored_slots:
        quote = by_slot.get(slot)
        if quote is None or quote == NOT_ESTABLISHED:
            return True
        if not _quote_is_verbatim(quote, shown_text):
            return True
    return False


def apply_decision_guard(
    decision: PaperDecision,
    shown_text: str,
    known_ids: set[str],
    *,
    full_text_inclusion_ids: set[str] = frozenset(),
    full_text_exclusion_ids: set[str] = frozenset(),
    absence_ids: set[str] = frozenset(),
    anchored_slots: frozenset[str] = frozenset(),
) -> tuple[PaperDecision, str]:
    """Demote an EXCLUDE that should not stand at title-and-abstract stage, or an INCLUDE
    left unanchored on a slot this run treats as anchored, or an INCLUDE still carrying an
    unconfirmed full-text inclusion criterion; never any other transition. Returns
    ``(decision, guard_reason)``, ``guard_reason`` one of ``""`` (no demotion),
    ``"no_abstract"``, ``"full_text_criterion"``, ``"cut_abstract"``, ``"unanchored_exclude"``,
    ``"unquoted_criterion"``, ``"unanchored_include"`` or ``"full_text_to_confirm"``.

    Checked first among the
    INCLUDE checks, ahead of the anchor check -- when :func:`_resolve_to_confirm` on the
    model's own, pre-guard decision is non-empty (at least one full-text inclusion
    criterion is not yet confirmed by the shown text), the decision is demoted to
    ``"full_text_to_confirm"``. A full-text inclusion criterion is by definition
    undecidable from a title and abstract, so this fires structurally, whatever the prompt
    itself asked the model to do, and whatever the anchor ledger says. A no-op when the
    protocol names no full-text inclusion criterion at all (``full_text_inclusion_ids``
    empty, the default), so every caller predating this full-text check is unaffected.

    On an INCLUDE, when ``anchored_slots`` names at least
    one of :data:`ANCHOR_SLOTS`, the decision is demoted to ``"unanchored_include"`` unless
    every named slot has an entry in ``decision.anchors`` whose quote is a verbatim substring
    of ``shown_text`` (:func:`_unanchored_include_slots`). ``anchored_slots`` empty (the
    default) is a no-op -- every existing caller that never mentions it keeps today's
    behaviour, INCLUDE always returned unchanged.

    The no-abstract check below gains one carve-out ahead
    of its general rule -- an EXCLUDE naming criterion ``"TOPIC"`` whose quote is a verbatim
    substring of the record's *title* (:func:`_shown_title`) stands unchanged, because a
    title can establish that a record is about something else even though it cannot
    establish that a numbered criterion is failed. Every other no-abstract EXCLUDE is
    demoted exactly as before.

    Checked first, ahead of every other EXCLUDE check below, an EXCLUDE on a
    record with no abstract (:func:`_shown_abstract_is_missing`) is always demoted to
    NEEDS_REVIEW, regardless of the criterion it names. This is the one precedence
    exception to the ordering documented below: a record with no abstract cannot be judged
    on any criterion, full-text or abstract-stage, on topic or off, from the title alone, so
    nothing later in this function ever sees such a record. The *label* given to that
    demotion follows the same rule the full-text-exclusion check applies just below --
    ``"no_abstract"`` when the quote is a verbatim substring of the shown text, which still
    covers an EXCLUDE quoting only the title (the title is still part of ``shown_text`` and
    would otherwise pass the ordinary anchor check at the bottom unchanged), and
    ``"unanchored_exclude"`` when the quote is fabricated, so a no-abstract EXCLUDE that also
    invents its quote is not owed the gentler label.

    The two kinds of full-text criterion are treated differently, which is why they arrive
    as two separate sets rather than a single combined set of full-text ids:

    A full-text **inclusion** criterion is never tested at abstract stage at all, so an
    EXCLUDE naming one is a rule the prompt forbids outright: it is always demoted to
    ``"unanchored_exclude"``, whatever its quote says. There is no "verbatim quote against a
    full-text inclusion criterion" case to reward, because the model was never supposed to
    ground an EXCLUDE on that criterion in the first place.

    A full-text **exclusion** criterion can ground a NEEDS_REVIEW, never an EXCLUDE, and only
    behind a verbatim quote of explicit contrary evidence:

    - An EXCLUDE naming one is demoted to ``"full_text_criterion"`` when its quote is
      verbatim, and to ``"unanchored_exclude"`` otherwise -- an EXCLUDE that both breaks the
      full-text rule and fabricates its quote is not owed the gentler label.
    - A NEEDS_REVIEW naming one (the model already answered NEEDS_REVIEW itself, so nothing
      is demoted; the decision is returned unchanged, by identity) is attributed
      ``"full_text_criterion"`` only when its quote is non-empty and verbatim, and
      ``"unquoted_criterion"`` otherwise -- the criterion id and quote stay on the decision
      either way, for audit.

    A NEEDS_REVIEW naming *any other* non-empty criterion id -- an ordinary abstract-stage
    id, or a full-text **inclusion** id (the non-empty-and-verbatim quote test is not
    narrowed to ``full_text_exclusion_ids``, since the protocol still requires a criterion id
    and a verbatim quote at abstract stage, and the prompt forbids a full-text-inclusion
    NEEDS_REVIEW outright) -- is
    attributed ``"unquoted_criterion"`` the same way whenever its quote is empty or not
    verbatim, so it never falls through to guard_reason ``""`` and hides inside the
    undecidable/no-abstract bucket alongside a record the model legitimately could not
    decide. Only a full-text **exclusion** id backed by a verbatim quote earns the
    ``"full_text_criterion"`` label; every other criterion-naming NEEDS_REVIEW with a
    verbatim quote is left unattributed (``""``), the same as before. A NEEDS_REVIEW naming
    no criterion at all is always ``""`` -- there is no quote to check it against.

    Beyond the full-text checks:

    - ``decision.criterion in absence_ids`` and :data:`ABSTRACT_CUT_MARKER` is in
      ``shown_text``: an absence criterion (the source review's prompt has stated the rule
      since S1) cannot be grounded on a cut abstract, because the missing tail, not the
      record, may be why nothing was found.
    - the pre-existing unanchored-quote/unknown-criterion-id test.

    ``criterion`` and ``quote`` are kept on a demoted decision for audit, so a routed record
    is as auditable as an EXCLUDE. An INCLUDE is always returned unchanged with guard_reason
    ``""``, and this function never returns an INCLUDE.
    """
    if decision.decision == "NEEDS_REVIEW":
        if not decision.criterion:
            return decision, ""
        if decision.quote and _quote_is_verbatim(decision.quote, shown_text):
            if decision.criterion in full_text_exclusion_ids:
                return decision, GUARD_REASON_FULL_TEXT_CRITERION
            return decision, ""
        return decision, GUARD_REASON_UNQUOTED_CRITERION
    if decision.decision == "INCLUDE":
        # Checked first, ahead of
        # the anchor check below -- a full-text inclusion criterion is by definition
        # undecidable from a title and abstract, so a record still carrying one open never
        # ships as a clean INCLUDE, whatever the anchor ledger says.
        if _resolve_to_confirm(decision, full_text_inclusion_ids):
            note = _GUARD_NOTES[GUARD_REASON_FULL_TEXT_TO_CONFIRM]
            reason = decision.reason.strip()
            reason = f"{reason} ({note})" if reason else note
            if len(reason) > MAX_REASON_LENGTH:
                reason = reason[:MAX_REASON_LENGTH]
            return (
                decision.model_copy(update={"decision": "NEEDS_REVIEW", "reason": reason}),
                GUARD_REASON_FULL_TEXT_TO_CONFIRM,
            )
        if anchored_slots and _unanchored_include_slots(decision, shown_text, anchored_slots):
            note = _GUARD_NOTES[GUARD_REASON_UNANCHORED_INCLUDE]
            reason = decision.reason.strip()
            reason = f"{reason} ({note})" if reason else note
            if len(reason) > MAX_REASON_LENGTH:
                reason = reason[:MAX_REASON_LENGTH]
            return (
                decision.model_copy(update={"decision": "NEEDS_REVIEW", "reason": reason}),
                GUARD_REASON_UNANCHORED_INCLUDE,
            )
        return decision, ""
    if decision.decision != "EXCLUDE":
        return decision, ""
    if _shown_abstract_is_missing(shown_text):
        if decision.criterion == TOPIC_CRITERION_ID and _quote_is_verbatim(
            decision.quote, _shown_title(shown_text)
        ):
            # A title-only TOPIC exclude stands.
            return decision, ""
        if _quote_is_verbatim(decision.quote, shown_text):
            guard_reason = GUARD_REASON_NO_ABSTRACT
        else:
            guard_reason = GUARD_REASON_UNANCHORED_EXCLUDE
    elif decision.criterion in full_text_exclusion_ids:
        if _quote_is_verbatim(decision.quote, shown_text):
            guard_reason = GUARD_REASON_FULL_TEXT_CRITERION
        else:
            guard_reason = GUARD_REASON_UNANCHORED_EXCLUDE
    elif decision.criterion in full_text_inclusion_ids:
        guard_reason = GUARD_REASON_UNANCHORED_EXCLUDE
    elif decision.criterion in absence_ids and ABSTRACT_CUT_MARKER in (shown_text or ""):
        guard_reason = GUARD_REASON_CUT_ABSTRACT
    else:
        # ``known_ids`` is never empty on its own, since
        # ``_known_criterion_ids`` always adds the reserved ``TOPIC_CRITERION_ID``. The
        # empty-protocol fallback (module docstring) must key off whether the *protocol
        # itself* names any numbered criterion, not off
        # whether the whole vocabulary (protocol ids plus the always-present TOPIC id) is
        # empty -- otherwise a compliant EXCLUDE with an empty criterion id and a verbatim
        # quote, the default product path with no protocol supplied, is wrongly demoted.
        protocol_ids = known_ids - {TOPIC_CRITERION_ID}
        criterion_ok = decision.criterion in known_ids if protocol_ids else True
        if criterion_ok and _quote_is_verbatim(decision.quote, shown_text):
            return decision, ""
        guard_reason = GUARD_REASON_UNANCHORED_EXCLUDE
    note = _GUARD_NOTES[guard_reason]
    reason = decision.reason.strip()
    reason = f"{reason} ({note})" if reason else note
    if len(reason) > MAX_REASON_LENGTH:
        reason = reason[:MAX_REASON_LENGTH]
    return decision.model_copy(update={"decision": "NEEDS_REVIEW", "reason": reason}), guard_reason


# --------------------------------------------------------------------------------------
# Two narrow deterministic post-model devices, applied by a caller *after* screen_papers
# returns (the harness or an evaluation script), never inside screen_papers itself --
# OpenAlex's own type/is_paratext are not shown to the model and are not part of the
# screening prompt.
# Each demotes an INCLUDE to NEEDS_REVIEW only; neither ever excludes and neither ever
# promotes, the same contract apply_decision_guard keeps.
# --------------------------------------------------------------------------------------

#: The narrow OpenAlex work types an INCLUDE is demoted for, measured to cost nothing on a
#: 140 human-judged demo sample and no final inclusion in the reference dataset used to
#: measure it. Deliberately excludes ``conference-paper``, ``dissertation``, ``report`` and
#: ``other`` -- measurement showed both directions of error on those types (S-115 typed
#: ``conference-paper`` and majority-excluded, S-111 typed ``other`` and majority-included),
#: so they are left to the model and the second-pass judge rather than a blanket rule.
NONARTICLE_TYPES: frozenset[str] = frozenset(
    {
        "paratext",
        "book",
        "book-review",
        "reference-entry",
        "conference-abstract",
        "editorial",
        "erratum",
        "letter",
        "peer-review",
        "dataset",
        "libguides",
    }
)


def apply_type_demotion(
    status: str, *, work_type: str | None, is_paratext: bool | None
) -> tuple[str, str]:
    """Demote an INCLUDE to NEEDS_REVIEW when OpenAlex marks the record ``is_paratext`` or
    a member of :data:`NONARTICLE_TYPES`; a no-op on any other status, and on an INCLUDE
    whose type is unknown, empty, or outside the narrow list. Returns ``(status,
    guard_reason)``, ``guard_reason`` :data:`GUARD_REASON_NONARTICLE_TYPE` or ``""``.
    """
    if status != "INCLUDE":
        return status, ""
    if bool(is_paratext) or (work_type in NONARTICLE_TYPES):
        return "NEEDS_REVIEW", GUARD_REASON_NONARTICLE_TYPE
    return status, ""


#: A numbered-chapter pattern
#: ("1. Introduction", "12. Conclusion"). Matched against the shown abstract text only.
_TOC_CHAPTER_RE = re.compile(r"\d{1,2}\.\s+[A-Z]")
#: A "Part I"/"Part II"/"Part 1" marker, roman or arabic.
_TOC_PART_RE = re.compile(r"\bPart\s+(?:[IVXLCDM]+|\d+)\b")


def _is_table_of_contents(shown_text: str | None) -> bool:
    """True when the shown text reads as a chapter list rather than an abstract: four or
    more numbered chapter matches, or two or more together with a "Part" marker, or two or
    more "Part" markers alone."""
    text = shown_text or ""
    chapters = len(_TOC_CHAPTER_RE.findall(text))
    parts = len(_TOC_PART_RE.findall(text))
    if chapters >= 4:
        return True
    if chapters >= 2 and parts >= 1:
        return True
    if parts >= 2:
        return True
    return False


def apply_table_of_contents_demotion(status: str, shown_text: str | None) -> tuple[str, str]:
    """Demote an INCLUDE to NEEDS_REVIEW when :func:`_is_table_of_contents` reads the shown
    text as a chapter list; a no-op on any other status. Returns ``(status, guard_reason)``,
    ``guard_reason`` :data:`GUARD_REASON_TABLE_OF_CONTENTS` or ``""``.
    """
    if status != "INCLUDE":
        return status, ""
    if _is_table_of_contents(shown_text):
        return "NEEDS_REVIEW", GUARD_REASON_TABLE_OF_CONTENTS
    return status, ""


#: The reserved criterion id an off-topic EXCLUDE names when no
#: numbered criterion covers the population/phenomenon/outcome mismatch. Always a member of
#: :func:`_known_criterion_ids`'s returned set, in neither full-text set and in no absence
#: set, so on a record with an abstract it reaches only the guard's final (ordinary
#: unanchored-quote) branch: a verbatim quote anchors it like any other EXCLUDE, it is never
#: a NEEDS_REVIEW ground, and it can never enter ``to_confirm`` (that list is intersected
#: against full-text inclusion ids only). On a record with no abstract the prompt
#: forbids naming ``TOPIC`` at all (a no-abstract record can never be judged off topic), and
#: the guard's no-abstract check runs first regardless, so an EXCLUDE naming ``TOPIC`` there
#: is demoted to ``"no_abstract"``, never reaching this branch.
TOPIC_CRITERION_ID = "TOPIC"


def _known_criterion_ids(
    inclusion_criteria: list[str] | None, exclusion_criteria: list[str] | None
) -> set[str]:
    """Every criterion id the model may anchor an EXCLUDE to: the protocol's own I{n}/E{n}
    ids plus the reserved :data:`TOPIC_CRITERION_ID`, which names an
    off-topic EXCLUDE that no numbered criterion covers -- always known, whether or not the
    protocol names any criterion at all."""
    ids = {f"I{i}" for i in range(1, len(inclusion_criteria or []) + 1)}
    ids |= {f"E{i}" for i in range(1, len(exclusion_criteria or []) + 1)}
    ids.add(TOPIC_CRITERION_ID)
    return ids


def _stage_ids(
    criteria: list[str] | None, stages: list[str] | None, *, prefix: str, stage: str
) -> set[str]:
    """Criterion ids of one side (inclusion or exclusion) whose parallel stage list marks
    them ``stage``. A criterion beyond the end of its stage list, or
    with no stage list at all, defaults to ``"abstract"`` -- the same default
    :func:`evaluation.common.load_protocol` applies."""
    ids: set[str] = set()
    for i in range(len(criteria or [])):
        s = stages[i] if stages and i < len(stages) else "abstract"
        if s == stage:
            ids.add(f"{prefix}{i + 1}")
    return ids


def full_text_inclusion_criterion_ids(
    inclusion_criteria: list[str] | None, inclusion_stages: list[str] | None
) -> set[str]:
    """Inclusion-side criterion ids marked ``stage: "full_text"``: never
    tested at abstract stage, deferred to full text with a to-confirm note on an INCLUDE."""
    return _stage_ids(inclusion_criteria, inclusion_stages, prefix="I", stage="full_text")


def full_text_exclusion_criterion_ids(
    exclusion_criteria: list[str] | None, exclusion_stages: list[str] | None
) -> set[str]:
    """Exclusion-side criterion ids marked ``stage: "full_text"``: can
    never ground an EXCLUDE, and can ground a NEEDS_REVIEW only behind a verbatim quote of
    explicit contrary evidence (:func:`apply_decision_guard`). ``run_smart_search``'s
    ``needs_review_by_reason`` bucket uses the same set so a compliant model routing (no
    guard demotion, ``guard_reason == ""``) is never folded into ``"undecidable"``."""
    return _stage_ids(exclusion_criteria, exclusion_stages, prefix="E", stage="full_text")


def _absence_criterion_ids(
    inclusion_criteria: list[str] | None,
    exclusion_criteria: list[str] | None,
    inclusion_absence: list[bool] | None,
    exclusion_absence: list[bool] | None,
) -> set[str]:
    """Criterion ids that test absence: default False for an inclusion
    criterion, True for an exclusion criterion, unless the protocol overrides it."""
    ids: set[str] = set()
    for i in range(len(inclusion_criteria or [])):
        absent = (
            inclusion_absence[i] if inclusion_absence and i < len(inclusion_absence) else False
        )
        if absent:
            ids.add(f"I{i + 1}")
    for i in range(len(exclusion_criteria or [])):
        absent = (
            exclusion_absence[i] if exclusion_absence and i < len(exclusion_absence) else True
        )
        if absent:
            ids.add(f"E{i + 1}")
    return ids


def _resolve_to_confirm(
    decision: PaperDecision, full_text_inclusion_ids: set[str]
) -> list[str]:
    """On an INCLUDE, the model's own ``to_confirm`` list intersected
    with the protocol's full-text inclusion ids, sorted; when that intersection is empty
    (the model omitted the field, emptied it, or named ids outside the protocol) it defaults
    to the full set, since a full-text inclusion criterion always needs confirming unless the
    model actually named it. ``[]`` on every other decision, regardless of what the model
    returned -- to_confirm is only ever meaningful on an INCLUDE.
    """
    if decision.decision != "INCLUDE":
        return []
    kept = sorted({c for c in decision.to_confirm if c in full_text_inclusion_ids})
    return kept if kept else sorted(full_text_inclusion_ids)


def get_relevance_screener_agent(
    *, prompt_version: str = _ACTIVE_PROMPT_VERSION
) -> Agent[None, ScreeningDecisions]:
    """Lazily create the relevance screener agent for one prompt version.

    Avoids requiring an API key at import time. One agent is cached per ``prompt_version``,
    so a v1 call gets an agent instructed with ``SCREENER_PROMPT_V1`` and a v2 call gets one
    instructed with ``SCREENER_PROMPT_V2``, without either overwriting the other's cached
    instance.
    """
    if prompt_version not in _PROMPTS_BY_VERSION:
        raise ValueError(f"unknown prompt_version: {prompt_version!r} (expected 'v1', 'v2' or 'v3')")
    if prompt_version not in _agents:
        _agents[prompt_version] = Agent(
            build_deepseek_model(settings.deepseek_model),
            deps_type=None,
            output_type=ScreeningDecisions,
            instructions=_PROMPTS_BY_VERSION[prompt_version],
            model_settings=DETERMINISTIC_FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agents[prompt_version]


def _build_screening_prompt(
    user_query: str,
    papers: list[dict],
    inclusion_criteria: list[str] | None = None,
    exclusion_criteria: list[str] | None = None,
    *,
    prompt_version: str = _ACTIVE_PROMPT_VERSION,
    limit: int = ABSTRACT_CHAR_LIMIT,
    inclusion_stages: list[str] | None = None,
    exclusion_stages: list[str] | None = None,
) -> str:
    """Assemble the user-turn prompt: the research question, the criteria, then every
    record's title and abstract exactly as rendered.

    The framing is bound to ``prompt_version`` the same way the abstract truncation style
    is: v1 reproduces the originally committed layout byte for byte -- "INCLUDE papers
    that:" / "EXCLUDE papers that:" bullets, and
    "Paper N:" with two-space-indented "Title:" / "Abstract:" lines -- so a v1 run at
    ``limit=500`` against an unchanged protocol is byte-identical to the six committed v1
    runs. v2 numbers the criteria ``I1..`` and ``E1..`` in protocol order and renders each
    record as "Record N:" followed by the unindented text :func:`build_shown_texts` returns
    for it.

    ``inclusion_stages``/``exclusion_stages`` (v2 only) are the parallel
    per-criterion stage lists (``"abstract"``/``"full_text"``). Two headings are appended
    after the exclusion block, each only when that side has a full-text member: a
    ``Full-text inclusion criteria: ...`` line naming the deferred ids, followed by the
    deferral-and-to-confirm instruction, then a ``Full-text exclusion criteria: ...`` line,
    followed by the contrary-evidence instruction. Either heading is omitted entirely when
    its side has no full-text criterion, so a caller that never mentions stages (every
    pre-S8c call site) renders exactly as before.

    Immediately before those two headings, an ``Order of judgement:
    ...`` line states that the criteria above (and the research question) are judged
    first, and is rendered under the same condition as the headings -- at least one side
    has a full-text member -- so a caller with no full-text criterion still renders
    exactly the prompt without it.
    """
    if prompt_version == "v1":
        parts: list[str] = [f"Research topic: {user_query}"]
        if inclusion_criteria:
            parts.append("\nINCLUDE papers that:")
            for criterion in inclusion_criteria:
                parts.append(f"  - {criterion}")
        if exclusion_criteria:
            parts.append("\nEXCLUDE papers that:")
            for criterion in exclusion_criteria:
                parts.append(f"  - {criterion}")
        parts.append("")
        parts.append("Papers to screen (title and abstract):")
        for i, paper in enumerate(papers, start=1):
            title = paper.get("title") or "(no title)"
            abstract = render_abstract(
                paper.get("abstract") or NO_ABSTRACT_PLACEHOLDER, prompt_version="v1", limit=limit
            )
            parts.append(f"\nPaper {i}:")
            parts.append(f"  Title: {title}")
            parts.append(f"  Abstract: {abstract}")
        return "\n".join(parts)

    if prompt_version in ("v2", "v3"):
        parts = [f"Research topic: {user_query}"]
        if inclusion_criteria:
            parts.append("\nInclusion criteria:")
            for i, criterion in enumerate(inclusion_criteria, start=1):
                parts.append(f"  I{i}: {criterion}")
        if exclusion_criteria:
            parts.append("\nExclusion criteria:")
            for i, criterion in enumerate(exclusion_criteria, start=1):
                parts.append(f"  E{i}: {criterion}")
        if not inclusion_criteria and not exclusion_criteria:
            # With no protocol at all, do not
            # promise numbered criteria the user turn does not supply.
            parts.append(
                "\nNo inclusion or exclusion criteria were provided for this screen. "
                "EXCLUDE only when the shown title and abstract itself gives a clear, "
                "quotable reason; leave the criterion id empty."
            )
        full_text_inclusion_ids = sorted(
            full_text_inclusion_criterion_ids(inclusion_criteria, inclusion_stages),
            key=lambda cid: int(cid[1:]),
        )
        full_text_exclusion_ids = sorted(
            full_text_exclusion_criterion_ids(exclusion_criteria, exclusion_stages),
            key=lambda cid: int(cid[1:]),
        )
        if full_text_inclusion_ids or full_text_exclusion_ids:
            # States, before either full-text heading, that the
            # criteria above (and the research question) are judged first -- rendered only
            # when at least one side has a full-text member, so a caller that never marks a
            # criterion full-text still renders exactly the prompt without it.
            parts.append(
                "\nOrder of judgement: decide the criteria above and the research question "
                "first. Only a record that passes them all and is on topic is judged "
                "against the criteria below."
            )
        if full_text_inclusion_ids:
            parts.append(
                f"\nFull-text inclusion criteria: {', '.join(full_text_inclusion_ids)}"
            )
            parts.append(
                "These cannot be decided from a title and abstract. Judge the record on "
                "its remaining criteria and the research topic; if it passes, INCLUDE it "
                'and list any of these ids the shown text does not confirm in "to_confirm".'
            )
        if full_text_exclusion_ids:
            parts.append(
                f"\nFull-text exclusion criteria: {', '.join(full_text_exclusion_ids)}"
            )
            parts.append(
                "These can never ground an EXCLUDE. NEEDS_REVIEW, naming the criterion and "
                "quoting the text verbatim, only if the shown text states explicit contrary "
                "evidence against one; otherwise its silence has no effect."
            )
        parts.append("")
        parts.append("Records to screen (title and abstract):")
        shown_texts = build_shown_texts(papers, prompt_version=prompt_version, limit=limit)
        for i, text in enumerate(shown_texts, start=1):
            parts.append(f"\nRecord {i}:\n{text}")
        return "\n".join(parts)

    raise ValueError(f"unknown prompt_version: {prompt_version!r} (expected 'v1', 'v2' or 'v3')")


async def _reask_missing_decisions(
    agent: Agent[None, ScreeningDecisions],
    user_query: str,
    missing_papers: list[dict],
    inclusion_criteria: list[str] | None,
    exclusion_criteria: list[str] | None,
    *,
    prompt_version: str,
    limit: int,
    inclusion_stages: list[str] | None,
    exclusion_stages: list[str] | None,
) -> tuple[list[PaperDecision], Any | None, str | None]:
    """Ask the same agent once more, for only the records a batch response left undecided.

    A fresh, smaller prompt is built with :func:`_build_screening_prompt` (unchanged: the
    prompt text and framing are exactly what a batch of this size would otherwise get) over
    just ``missing_papers``. Returns ``(decisions, run_result, error)``: ``decisions`` is
    whatever this second call gives back, cut to at most ``len(missing_papers)`` -- possibly
    the full set (nothing left to pad), possibly still short, possibly empty on a call
    failure. ``run_result`` is the raw ``AgentRunResult`` of a call that answered, so
    :func:`screen_papers` can build its provenance: the re-ask's own tokens, model and
    fingerprint must stay visible, not be dropped; ``None`` when the call itself raised,
    since there is nothing to build a provenance record from. ``error`` is
    ``"<ExceptionClass>: <message>"`` when the call itself raised, ``None`` otherwise: a
    still-padded record caused by a transport or model failure is otherwise
    indistinguishable from one caused by the model quietly skipping a record again, so
    nothing would show which happened. This function never pads and never raises --
    :func:`screen_papers` pads whatever position, if any, this still leaves open.
    """
    prompt = _build_screening_prompt(
        user_query,
        missing_papers,
        inclusion_criteria,
        exclusion_criteria,
        prompt_version=prompt_version,
        limit=limit,
        inclusion_stages=inclusion_stages,
        exclusion_stages=exclusion_stages,
    )
    try:
        result = await agent.run(prompt)
    except Exception as exc:
        return [], None, f"{type(exc).__name__}: {exc}"
    return list(result.output.decisions)[: len(missing_papers)], result, None


async def screen_papers(
    user_query: str,
    papers: list[dict],
    inclusion_criteria: list[str] | None = None,
    exclusion_criteria: list[str] | None = None,
    *,
    prompt_version: str = _ACTIVE_PROMPT_VERSION,
    abstract_limit: int = ABSTRACT_CHAR_LIMIT,
    inclusion_stages: list[str] | None = None,
    exclusion_stages: list[str] | None = None,
    inclusion_absence: list[bool] | None = None,
    exclusion_absence: list[bool] | None = None,
    anchored_slots: frozenset[str] = frozenset(),
) -> ScreeningBatchResult:
    """Screen a batch of papers against a written eligibility protocol.

    ``prompt_version`` selects the system prompt (``SCREENER_PROMPT_V1`` or ``_V2``), the
    matching user-turn framing and abstract truncation style of :func:`_build_screening_prompt`,
    and whether :func:`apply_decision_guard` runs at all: v1 asks the model for no criterion
    and no quote, so there is nothing for the guard to anchor against, and it is skipped
    entirely for a v1 call. ``abstract_limit`` overrides the numeric abstract cap only; the
    truncation style stays bound to ``prompt_version``.

    ``anchored_slots`` (v3 only) names which of the three
    anchor-ledger slots (``"population"``, ``"subject"``, ``"outcome"``) the guard demands a
    verbatim, established entry for on every INCLUDE; empty (the default) means the guard
    never demotes an INCLUDE on this ground, whatever the model's own ledger says -- the
    ledger is still returned on ``ScreeningBatchResult.anchors`` either way, so a caller can
    choose, and later re-choose, the anchor set without another model call.

    ``inclusion_stages``/``exclusion_stages`` (v2 only) are the parallel
    ``"abstract"``/``"full_text"`` stage of each criterion; a criterion beyond the end of its
    stage list, or when no stage list is given at all, defaults to ``"abstract"``, so an
    existing caller that never mentions stages is unaffected. ``inclusion_absence``/
    ``exclusion_absence`` are the parallel "tests that something is not present" flags,
    defaulting to False for an inclusion criterion and True for an exclusion criterion (the
    same default :func:`evaluation.common.load_protocol` applies) when not given.

    Default INCLUDE (v2): a record is INCLUDE unless the shown text gives a specific ground
    to exclude it, or leaves the decision undecidable. An EXCLUDE must be anchored to a
    verbatim quote from the shown text, and, when the protocol names at least one criterion,
    to a numbered criterion id too; :func:`apply_decision_guard` demotes any EXCLUDE that is
    not, or that names a full-text-stage criterion (inclusion-side always, exclusion-side
    unless its quote is verbatim), or that anchors an absence criterion in a
    cut abstract, or that is made on a record with no abstract at all (checked
    first, unconditionally, since a title alone cannot ground an exclusion). A paper the
    model returned no decision for is re-asked once, in a batch of just the still-undecided
    papers, via :func:`_reask_missing_decisions`;
    only a paper this second call still does not cover is NEEDS_REVIEW with the visible
    reason ``"no decision returned"`` -- never a silent INCLUDE, and never padded before the
    re-ask has had its one chance. A re-ask that answers is a second paid call, so its
    tokens, reported model and fingerprint are returned on ``reask_provenance``
    rather than merged into, or dropped from, ``provenance``;
    a caller totalling the batch's real cost adds the two records' ``input_tokens``/
    ``output_tokens`` together.

    ``to_confirm`` is resolved per record by :func:`_resolve_to_confirm`
    after the guard runs: ``[]`` on every non-INCLUDE, and on an INCLUDE the model's own list
    intersected with the protocol's full-text inclusion ids, defaulting to the full set when
    that comes up empty.

    Raises:
        ScreeningError: on any failure -- an empty batch, an unknown ``prompt_version``, a
            model call that raises, a response with no decisions at all, or an error while
            building the result. The caller decides what to do with the batch; nothing is
            silently included.
    """
    if not papers:
        raise ScreeningError("empty batch: nothing to screen")

    try:
        system_prompt = _PROMPTS_BY_VERSION.get(prompt_version)
        if system_prompt is None:
            raise ValueError(
                f"unknown prompt_version: {prompt_version!r} (expected 'v1', 'v2' or 'v3')"
            )

        prompt = _build_screening_prompt(
            user_query,
            papers,
            inclusion_criteria,
            exclusion_criteria,
            prompt_version=prompt_version,
            limit=abstract_limit,
            inclusion_stages=inclusion_stages,
            exclusion_stages=exclusion_stages,
        )
        agent = get_relevance_screener_agent(prompt_version=prompt_version)
        result = await agent.run(prompt)

        decisions = list(result.output.decisions)[: len(papers)]
        if not decisions:
            # Zero decisions is a failed call, not a short tail to pad.
            raise ScreeningError("model returned no decisions")
        reask_run_result: Any | None = None
        reask_error: str | None = None
        missing = len(papers) - len(decisions)
        if missing > 0:
            # Re-ask once, in a batch of just the still-undecided records, before padding
            # anything: a short response is far more often the model
            # skipping a few records than a genuine "cannot decide", so a fresh, smaller
            # call recovers most of them. Only whatever this single extra call still does
            # not cover is padded, and the reported ``padded`` count reflects that, not the
            # original shortfall.
            missing_papers = papers[len(decisions) :]
            reask_decisions, reask_run_result, reask_error = await _reask_missing_decisions(
                agent,
                user_query,
                missing_papers,
                inclusion_criteria,
                exclusion_criteria,
                prompt_version=prompt_version,
                limit=abstract_limit,
                inclusion_stages=inclusion_stages,
                exclusion_stages=exclusion_stages,
            )
            decisions.extend(reask_decisions)
            missing = len(papers) - len(decisions)
            if missing > 0:
                decisions.extend(
                    PaperDecision(decision="NEEDS_REVIEW", reason=NO_DECISION_REASON)
                    for _ in range(missing)
                )

        if prompt_version == "v1":
            # v1 asks for no criterion and no quote, so the guard has nothing to check.
            guarded = decisions
            guard_applied = [False] * len(decisions)
            guard_reasons = [""] * len(decisions)
            full_text_inclusion_ids: set[str] = set()
        else:
            known_ids = _known_criterion_ids(inclusion_criteria, exclusion_criteria)
            full_text_inclusion_ids = full_text_inclusion_criterion_ids(
                inclusion_criteria, inclusion_stages,
            )
            full_text_exclusion_ids = full_text_exclusion_criterion_ids(
                exclusion_criteria, exclusion_stages,
            )
            absence_ids = _absence_criterion_ids(
                inclusion_criteria, exclusion_criteria, inclusion_absence, exclusion_absence,
            )
            shown_texts = build_shown_texts(
                papers, prompt_version=prompt_version, limit=abstract_limit
            )
            guard_results = [
                apply_decision_guard(
                    decision, shown_text, known_ids,
                    full_text_inclusion_ids=full_text_inclusion_ids,
                    full_text_exclusion_ids=full_text_exclusion_ids,
                    absence_ids=absence_ids,
                    anchored_slots=anchored_slots,
                )
                for decision, shown_text in zip(decisions, shown_texts, strict=True)
            ]
            guarded = [pair[0] for pair in guard_results]
            guard_reasons = [pair[1] for pair in guard_results]
            guard_applied = [
                pre.decision != post.decision
                for pre, post in zip(decisions, guarded, strict=True)
            ]

        statuses = [d.decision for d in guarded]
        criteria_ids = [d.criterion for d in guarded]
        quotes = [d.quote for d in guarded]
        reasons = [d.reason for d in guarded]
        # Resolved from the
        # pre-guard ``decisions`` (the model's own answer), not ``guarded`` -- a record the
        # guard just demoted to NEEDS_REVIEW for ``full_text_to_confirm`` keeps its
        # unconfirmed criterion ids here for audit, the same way a demoted EXCLUDE keeps its
        # ``criterion``/``quote``; ``guarded[i].decision`` is already "NEEDS_REVIEW" at that
        # point, and ``_resolve_to_confirm`` only ever returns non-``[]`` for an INCLUDE.
        to_confirm = [_resolve_to_confirm(d, full_text_inclusion_ids) for d in decisions]
        # The raw slot ledger the model returned, kept
        # regardless of what the guard did with the decision (``anchored_slots`` may be
        # empty, in which case the guard never demotes on it at all) -- an evaluation
        # harness re-scores every anchor configuration offline from this alone, at no
        # further model cost. Always ``[]`` on v1/v2 (``PaperDecision.anchors`` defaults to
        # empty and nothing in either prompt asks the model to fill it).
        anchors = [[{"slot": a.slot, "quote": a.quote} for a in d.anchors] for d in guarded]

        provenance = provenance_from_run(
            "relevance_screener",
            result,
            model_configured=settings.deepseek_model,
            temperature=DETERMINISTIC_FAST_MODEL_SETTINGS.get("temperature"),
            prompt=system_prompt,
            guard_conversions=sum(guard_applied),
        )
        # A re-ask is a second paid call, so its own
        # tokens, model and fingerprint are recorded on a second provenance record rather
        # than dropped -- ``provenance`` alone would otherwise describe only one of the two
        # calls that produced this batch.
        reask_provenance = (
            provenance_from_run(
                "relevance_screener",
                reask_run_result,
                model_configured=settings.deepseek_model,
                temperature=DETERMINISTIC_FAST_MODEL_SETTINGS.get("temperature"),
                prompt=system_prompt,
            )
            if reask_run_result is not None
            else None
        )
        return ScreeningBatchResult(
            statuses=statuses,
            criteria_ids=criteria_ids,
            quotes=quotes,
            reasons=reasons,
            provenance=provenance,
            reask_provenance=reask_provenance,
            guard_conversions=sum(guard_applied),
            guard_applied=guard_applied,
            guard_reasons=guard_reasons,
            to_confirm=to_confirm,
            padded=max(missing, 0),
            reask_error=reask_error,
            anchors=anchors,
        )
    except ScreeningError:
        raise
    except Exception as exc:
        raise ScreeningError(f"{type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------------------
# The inclusion-only second pass
# --------------------------------------------------------------------------------------
#
# After the batch pass and the guard, the records still standing as INCLUDE are sent back
# once more, in small batches, and asked the three anchor-ledger questions directly -- does
# the shown text name the study's own population, the subject, and a reported outcome. This
# is where a sufficiency judgement a quote check cannot make gets made: the ledger only
# checks that a quoted phrase exists and is verbatim, not that it actually establishes the
# slot (record S-087 is an example: an abstract full of L2 writing vocabulary whose
# participants are teachers, not learners). The pass can only confirm or demote to
# NEEDS_REVIEW -- it never
# excludes and never promotes -- and it runs on inclusions only, so it is cheap regardless
# of corpus size.

SECOND_PASS_PROMPT_V1 = """\
You already screened each of these records as INCLUDE against a review's eligibility
protocol. For each record, decide from its shown title and abstract alone whether the shown
text itself establishes three things: the study's own population (who or what the study
itself studied), the subject (the intervention, exposure or phenomenon the research
question names), and a reported outcome (a result the study reports, of the kind the
research question asks about).

Rules:
- Judge only whether the shown text establishes each slot; do not use outside knowledge and
  do not guess from the venue, the authors or the journal.
- When the shown text establishes all three slots, answer "confirmed" and quote each slot's
  phrase character for character from the shown text.
- When the shown text does not establish at least one slot, answer "not_established" and
  name the first such slot in "missing_slot"; still quote any slot the shown text does
  establish.
- Do not paraphrase a quote and do not quote text you were not shown.

Output valid JSON with an "answers" list containing exactly one object per record, in the
same order as the input list. Each object has:
- "verdict": "confirmed" or "not_established"
- "missing_slot": the first slot ("population", "subject" or "outcome") the shown text does
  not establish; "" when "verdict" is "confirmed"
- "anchors": a list of objects, one per slot the shown text establishes, each
  {"slot": "population"|"subject"|"outcome", "quote": a phrase copied verbatim from the
  shown title or abstract}"""

#: Version hash of the second-pass prompt, recorded on its own provenance record the same
#: way ``SCREENER_PROMPT_VERSION`` is for the batch pass.
SECOND_PASS_PROMPT_VERSION = prompt_version(SECOND_PASS_PROMPT_V1)

#: Padded second-pass answer's verdict (mirrors ``NO_DECISION_REASON``'s "never
#: default to a lenient outcome silently"): a record the model did not answer at all is
#: treated as unconfirmed, not confirmed, so it is demoted rather than silently trusted.
NO_SECOND_PASS_ANSWER_REASON = "no second-pass answer returned"


class SecondPassAnswer(BaseModel):
    """One record's answer to the three anchor-ledger questions."""

    verdict: Literal["confirmed", "not_established"]
    missing_slot: str = Field(default="", max_length=16)
    anchors: list[AnchorEntry] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("missing_slot", mode="before")
    @classmethod
    def _normalise_missing_slot(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.strip().lower()
            if len(value) > 16:
                value = value[:16]
        return value


class SecondPassAnswers(BaseModel):
    """Structured output from the second-pass agent."""

    answers: list[SecondPassAnswer] = Field(
        ..., description="One answer object per record, in input order."
    )


# --------------------------------------------------------------------------------------
# A second, criterion-instantiated judge beside the untouched
# ``SECOND_PASS_PROMPT_V1`` above. The V1 judge never sees the protocol at all -- it
# asks whether a "population", a "subject" and an "outcome" exist in the abstract's own
# terms, which a record can satisfy while still answering the reviewers' actual objection
# (a study of what teachers do is a study of teachers, even when the material studied is
# learners' writing). ``SECOND_PASS_PROMPT_V2`` is shown the research question and the same
# numbered criteria the batch pass rendered, and asks three questions per record: does the
# shown text establish the population *the criteria name*, does it establish an outcome
# *the criteria name* for that population, and is the record a single study or a synthesis
# of studies (allowed wherever the protocol's own criteria allow one) as against an
# overview, editorial, whole book or chapter list. Any "not established" demotes the
# INCLUDE the same way the V1 judge's does; it never excludes and never promotes.
# --------------------------------------------------------------------------------------

SECOND_PASS_PROMPT_V2 = """\
You already screened each of these records as INCLUDE against a review's eligibility
protocol. The user turn gives the research question, the same numbered inclusion and
exclusion criteria the batch pass saw, then the records.

For each record, answer three questions from its shown title and abstract alone.

1. Population. Name the participants or units of analysis the shown text says the study
   itself studied, quoting a phrase character for character from the shown text, then
   answer whether those participants are themselves the population the criteria name. A
   study of what teachers do is a study of teachers, even when the material studied is
   learners' writing.
2. Outcome. Name a result the shown text says the study reports, quoting a phrase
   character for character from the shown text, then answer whether that result is one of
   the outcomes the criteria name, reported for the population of question 1. A
   description of what an intervention consists of, a count of publications, or a result
   about a different population is not one of those outcomes.
3. Study type. Answer whether the record is a single empirical study, or a synthesis of
   empirical studies (allowed wherever the criteria allow one), as against an overview of a
   field, an editorial, a whole book, or a chapter list.

When the answer to question 3 is "synthesis", answer questions 1 and 2 about the primary
studies the synthesis describes, not about the synthesis itself.

Rules:
- Judge only whether the shown text establishes each answer; do not use outside knowledge
  and do not guess from the venue, the authors or the journal.
- Do not paraphrase a quote and do not quote text you were not shown.
- When the shown text does not establish an answer, answer "not_established" for that
  question and leave its quote empty.

Output valid JSON with an "answers" list containing exactly one object per record, in the
same order as the input list. Each object has:
- "population": {"established": "established" or "not_established", "quote": a phrase
  copied verbatim from the shown title or abstract, "" when not established}
- "outcome": {"established": "established" or "not_established", "quote": a phrase
  copied verbatim from the shown title or abstract, "" when not established}
- "study_type": "study", "synthesis" or "not_established" (an overview, an editorial, a
  whole book, or a chapter list)"""

#: Version hash of the v2 second-pass prompt.
SECOND_PASS_PROMPT_V2_VERSION = prompt_version(SECOND_PASS_PROMPT_V2)


class SecondPassSlotAnswer(BaseModel):
    """One of the V2 judge's population/outcome answers: established or not, plus its
    verbatim quote."""

    established: Literal["established", "not_established"] = "not_established"
    quote: str = Field(default="", max_length=MAX_QUOTE_LENGTH)

    @field_validator("established", mode="before")
    @classmethod
    def _normalise_established(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("quote", mode="before")
    @classmethod
    def _truncate_quote(cls, value: object) -> object:
        if value is None:
            return ""
        if isinstance(value, str) and len(value) > MAX_QUOTE_LENGTH:
            return value[:MAX_QUOTE_LENGTH]
        return value


class SecondPassAnswerV2(BaseModel):
    """One record's answer to the v2 judge's three criterion-instantiated questions."""

    population: SecondPassSlotAnswer = Field(default_factory=SecondPassSlotAnswer)
    outcome: SecondPassSlotAnswer = Field(default_factory=SecondPassSlotAnswer)
    study_type: Literal["study", "synthesis", "not_established"] = "not_established"

    @field_validator("study_type", mode="before")
    @classmethod
    def _normalise_study_type(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value


class SecondPassAnswersV2(BaseModel):
    """Structured output from the v2 second-pass agent."""

    answers: list[SecondPassAnswerV2] = Field(
        ..., description="One answer object per record, in input order."
    )


class SecondPassBatchResultV2(BaseModel):
    """Round 2 second-pass answers for one batch, parallel to the input papers, plus
    provenance."""

    answers: list[SecondPassAnswerV2]
    provenance: LLMCallProvenance
    #: How many records this batch's response left unanswered, padded to an all-slots
    #: ``"not_established"`` answer (never silently confirmed).
    padded: int = 0


class SecondPassBatchResult(BaseModel):
    """Second-pass answers for one batch, parallel to the input papers, plus provenance."""

    verdicts: list[str]
    missing_slots: list[str] = Field(default_factory=list)
    anchors: list[list[dict[str, str]]] = Field(default_factory=list)
    provenance: LLMCallProvenance
    #: How many records this batch's response left unanswered, padded to "not_established"
    #: (never silently "confirmed") -- mirrors ``ScreeningBatchResult.padded``.
    padded: int = 0

    @model_validator(mode="after")
    def _validate_parallel_lengths(self) -> "SecondPassBatchResult":
        n = len(self.verdicts)
        if len(self.missing_slots) != n or len(self.anchors) != n:
            raise ValueError(
                "verdicts, missing_slots and anchors must be the same length "
                f"(got {n}, {len(self.missing_slots)}, {len(self.anchors)})"
            )
        return self


#: Values ``settings.screener_second_pass_output_mode`` accepts. "tool" is pydantic-ai's own
#: tool-calling structured output (a forced ``tool_choice`` -- this module's own behaviour
#: everywhere else, and still available here for a judge model that does not reject it).
#: "prompted" (the shipped default for the V2 second pass) puts the ``SecondPassAnswerV2``
#: schema in the prompt instead and asks the model to answer in plain JSON text, which
#: pydantic-ai validates against that same schema after the fact -- no ``tool_choice`` is
#: ever sent. Exists because DeepSeek's own API rejects ``tool_choice`` outright while
#: thinking mode is on for some served models (``deepseek-flash``, this setting's own
#: default judge model, confirmed by a smoke measurement) but not others; "prompted" is the
#: way to keep reasoning on for a model that rejects "tool" mode. Applies to the V2 second
#: pass only -- the V1 second pass always uses "tool" mode, unaffected.
SECOND_PASS_OUTPUT_MODES: tuple[str, ...] = ("tool", "prompted")

#: Reported on the V2 second pass's own provenance when no reasoning-effort level is
#: explicitly configured (the shipped default: ``SECOND_PASS_V2_MODEL_SETTINGS`` leaves
#: pydantic-ai's own generic ``thinking`` setting unset, so no ``reasoning_effort`` request
#: parameter is sent at all, and the served model's own default reasoning effort applies).
#: DeepSeek's own thinking-mode guide names "high" as that default.
SECOND_PASS_REASONING_EFFORT_DEFAULT = "high"

#: One cached agent per ``(prompt_version, model_name, output_mode)``, so a V1 call at
#: ``settings.deepseek_model``, a V2 "tool" call and a V2 "prompted" call at
#: ``settings.screener_second_pass_model`` (or an explicit override) never overwrite each
#: other's cached instance -- the same pattern :data:`_agents` uses for the batch pass.
#: ``output_mode`` is always ``"tool"`` for ``prompt_version="v1"``, which this setting does
#: not apply to.
_second_pass_agents: dict[tuple[str, str, str], Agent] = {}


def get_second_pass_agent(
    *, prompt_version: str = "v1", model: str | None = None
) -> Agent[None, SecondPassAnswers] | Agent[None, SecondPassAnswersV2]:
    """Lazily create the second-pass agent for one ``(prompt_version, model, output_mode)``
    combination.

    ``prompt_version="v1"`` (the default) is the anchor-ledger
    re-ask at ``settings.deepseek_model``, on ``DETERMINISTIC_FAST_MODEL_SETTINGS`` (thinking
    disabled, needed for tool-calling structured output to work on that model at all).
    ``prompt_version="v2"`` is the research-question-and-criteria-aware
    judge; ``model`` defaults to ``settings.screener_second_pass_model`` when not given, on
    ``SECOND_PASS_V2_MODEL_SETTINGS`` -- reasoning left on, a wider timeout -- since this
    specific judgement is measurably worse with reasoning disabled, unlike every other
    agent in this module (see that constant's own comment).
    Its own output mode is ``settings.screener_second_pass_output_mode`` (see
    :data:`SECOND_PASS_OUTPUT_MODES`); "tool" builds the agent exactly as before, "prompted"
    wraps ``SecondPassAnswersV2`` in :class:`pydantic_ai.PromptedOutput` so no ``tool_choice``
    is sent. Both modes return the identical Pydantic model from ``result.output``, so every
    caller downstream of ``agent.run`` needs no mode-specific handling at all.
    """
    if prompt_version not in ("v1", "v2"):
        raise ValueError(f"unknown second-pass prompt_version: {prompt_version!r}")
    model_name = model or (
        settings.deepseek_model if prompt_version == "v1" else settings.screener_second_pass_model
    )
    output_mode = settings.screener_second_pass_output_mode if prompt_version == "v2" else "tool"
    if output_mode not in SECOND_PASS_OUTPUT_MODES:
        raise ValueError(f"unknown screener_second_pass_output_mode: {output_mode!r}")
    key = (prompt_version, model_name, output_mode)
    if key not in _second_pass_agents:
        prompt_text = SECOND_PASS_PROMPT_V1 if prompt_version == "v1" else SECOND_PASS_PROMPT_V2
        if prompt_version == "v1":
            output_type: Any = SecondPassAnswers
        elif output_mode == "prompted":
            output_type = PromptedOutput(SecondPassAnswersV2)
        else:
            output_type = SecondPassAnswersV2
        agent_settings = (
            DETERMINISTIC_FAST_MODEL_SETTINGS if prompt_version == "v1"
            else SECOND_PASS_V2_MODEL_SETTINGS
        )
        _second_pass_agents[key] = Agent(
            build_deepseek_model(model_name),
            deps_type=None,
            output_type=output_type,
            instructions=prompt_text,
            model_settings=agent_settings,
            retries=AGENT_RETRIES,
        )
    return _second_pass_agents[key]


def _build_second_pass_prompt(papers: list[dict], *, limit: int = ABSTRACT_CHAR_LIMIT) -> str:
    """The user turn for the V1 second pass: every record already standing as INCLUDE,
    shown exactly as :func:`build_shown_texts` renders it for the v2/v3 framing, so a quote
    is verbatim against the same text the batch pass itself saw."""
    shown_texts = build_shown_texts(papers, prompt_version="v2", limit=limit)
    parts = ["Records already screened as INCLUDE:"]
    for i, text in enumerate(shown_texts, start=1):
        parts.append(f"\nRecord {i}:\n{text}")
    return "\n".join(parts)


def _build_second_pass_prompt_v2(
    papers: list[dict],
    research_question: str,
    inclusion_criteria: list[str] | None,
    exclusion_criteria: list[str] | None,
    *,
    limit: int = ABSTRACT_CHAR_LIMIT,
) -> str:
    """The user turn for the v2 second pass: the research question, the same
    ``I{n}``/``E{n}`` numbered criteria :func:`_build_screening_prompt` renders for the
    batch pass, then every record already standing as INCLUDE."""
    parts = [f"Research question: {research_question}"]
    if inclusion_criteria:
        parts.append("\nInclusion criteria:")
        for i, criterion in enumerate(inclusion_criteria, start=1):
            parts.append(f"  I{i}: {criterion}")
    if exclusion_criteria:
        parts.append("\nExclusion criteria:")
        for i, criterion in enumerate(exclusion_criteria, start=1):
            parts.append(f"  E{i}: {criterion}")
    parts.append("\nRecords already screened as INCLUDE:")
    shown_texts = build_shown_texts(papers, prompt_version="v2", limit=limit)
    for i, text in enumerate(shown_texts, start=1):
        parts.append(f"\nRecord {i}:\n{text}")
    return "\n".join(parts)


async def confirm_inclusions(
    papers: list[dict],
    *,
    abstract_limit: int = ABSTRACT_CHAR_LIMIT,
    research_question: str = "",
    inclusion_criteria: list[str] | None = None,
    exclusion_criteria: list[str] | None = None,
    model: str | None = None,
) -> SecondPassBatchResult | SecondPassBatchResultV2:
    """Ask the second-pass questions directly for a batch of INCLUDE records.

    ``research_question`` empty (the default) keeps V1's behaviour byte for byte: the
    anchor-ledger re-ask (``SECOND_PASS_PROMPT_V1``), at ``settings.deepseek_model``, never
    told the protocol at all, returning a :class:`SecondPassBatchResult`. A non-empty
    ``research_question`` switches to the
    criterion-instantiated judge (``SECOND_PASS_PROMPT_V2``), shown ``inclusion_criteria``/
    ``exclusion_criteria`` exactly as the batch pass numbered them, at ``model`` or
    ``settings.screener_second_pass_model`` when ``model`` is not given, returning a
    :class:`SecondPassBatchResultV2`.

    Raises :class:`ScreeningError` on any failure (empty batch, a model call that raises, a
    response with no answers at all) -- the caller decides what to do with the batch, the
    same contract :func:`screen_papers` keeps. A response short by a few records is padded
    to an all-``"not_established"`` answer (never a confirmed one), the safe default the
    rest of this module also uses for anything the model did not actually answer.
    """
    if not papers:
        raise ScreeningError("empty batch: nothing to confirm")
    try:
        if research_question.strip():
            agent = get_second_pass_agent(prompt_version="v2", model=model)
            prompt = _build_second_pass_prompt_v2(
                papers,
                research_question,
                inclusion_criteria,
                exclusion_criteria,
                limit=abstract_limit,
            )
            result = await agent.run(prompt)
            answers_v2 = list(result.output.answers)[: len(papers)]
            if not answers_v2:
                raise ScreeningError("second pass returned no answers")
            missing_v2 = len(papers) - len(answers_v2)
            if missing_v2 > 0:
                answers_v2.extend(SecondPassAnswerV2() for _ in range(missing_v2))
            # DeepSeek's own
            # thinking-mode guide states that temperature is not supported in thinking
            # mode, so the temperature=0.0 this call's own ModelSettings still sends is
            # silently ignored by the API -- reporting it as this call's own temperature
            # would be false. output_mode/reasoning_effort take its place instead.
            provenance_v2 = provenance_from_run(
                "relevance_screener_second_pass_v2",
                result,
                model_configured=model or settings.screener_second_pass_model,
                temperature=None,
                prompt=SECOND_PASS_PROMPT_V2,
                output_mode=settings.screener_second_pass_output_mode,
                reasoning_effort=SECOND_PASS_REASONING_EFFORT_DEFAULT,
            )
            return SecondPassBatchResultV2(
                answers=answers_v2,
                provenance=provenance_v2,
                padded=max(missing_v2, 0),
            )
        agent = get_second_pass_agent()
        prompt = _build_second_pass_prompt(papers, limit=abstract_limit)
        result = await agent.run(prompt)
        answers = list(result.output.answers)[: len(papers)]
        if not answers:
            raise ScreeningError("second pass returned no answers")
        missing = len(papers) - len(answers)
        if missing > 0:
            answers.extend(
                SecondPassAnswer(verdict="not_established", missing_slot="")
                for _ in range(missing)
            )
        verdicts = [a.verdict for a in answers]
        missing_slots = [a.missing_slot for a in answers]
        anchors = [[{"slot": e.slot, "quote": e.quote} for e in a.anchors] for a in answers]
        provenance = provenance_from_run(
            "relevance_screener_second_pass",
            result,
            model_configured=settings.deepseek_model,
            temperature=DETERMINISTIC_FAST_MODEL_SETTINGS.get("temperature"),
            prompt=SECOND_PASS_PROMPT_V1,
        )
        return SecondPassBatchResult(
            verdicts=verdicts,
            missing_slots=missing_slots,
            anchors=anchors,
            provenance=provenance,
            padded=max(missing, 0),
        )
    except ScreeningError:
        raise
    except Exception as exc:
        raise ScreeningError(f"{type(exc).__name__}: {exc}") from exc


def apply_second_pass_guard(
    status: str,
    verdict: str,
    *,
    answers: "SecondPassAnswer | SecondPassAnswerV2 | None" = None,
    shown_text: str | None = None,
) -> tuple[str, str]:
    """Demote an INCLUDE ``status`` to NEEDS_REVIEW when the second pass does not confirm
    it; return ``status`` unchanged, with guard_reason ``""``, in every other case. Never
    excludes and never promotes, and is a no-op on any ``status`` other than ``"INCLUDE"``
    -- the second pass only ever runs on records already standing as INCLUDE, but a caller
    that hands it a differently-routed record (for example one the offline grid re-derives
    under a different anchor set) gets that status back untouched rather than a reason
    meant only for an INCLUDE.

    Three ways to call this, from narrowest to widest:

    - ``verdict`` alone (``answers=None``, the original contract, unchanged): demotes
      on ``verdict == "not_established"``, whatever that string came from.
    - ``answers`` a V1 :class:`SecondPassAnswer` plus ``shown_text``: a ``"confirmed"``
      verdict is trusted only when every one of :data:`ANCHOR_SLOTS` has a
      non-``not_established`` entry in ``answers.anchors`` whose quote is a verbatim
      substring of ``shown_text``; a quote that is not verbatim is treated as
      ``not_established`` for that slot, the same rule :func:`apply_decision_guard`
      applies to the batch pass's own anchor ledger.
    - ``answers`` a V2 :class:`SecondPassAnswerV2` plus ``shown_text``: demotes when
      ``population`` or ``outcome`` is not ``"established"`` or its quote is not verbatim
      in ``shown_text``, or when ``study_type == "not_established"``. ``study_type in
      ("study", "synthesis")`` never demotes on its own -- a synthesis is allowed wherever
      the protocol's own criteria allow one, per the prompt's own synthesis rule.

    In every branch the guard_reason on a demotion is :data:`GUARD_REASON_NOT_ESTABLISHED`;
    which question failed is available to the caller from ``answers`` directly, so
    ``needs_review_by_reason`` keeps one stable key across both judge versions.
    """
    if status != "INCLUDE":
        return status, ""
    if isinstance(answers, SecondPassAnswerV2):
        for slot in (answers.population, answers.outcome):
            if slot.established != "established" or not _quote_is_verbatim(
                slot.quote, shown_text
            ):
                return "NEEDS_REVIEW", GUARD_REASON_NOT_ESTABLISHED
        if answers.study_type not in ("study", "synthesis"):
            return "NEEDS_REVIEW", GUARD_REASON_NOT_ESTABLISHED
        return status, ""
    if isinstance(answers, SecondPassAnswer):
        if answers.verdict != "confirmed":
            return "NEEDS_REVIEW", GUARD_REASON_NOT_ESTABLISHED
        quotes = {entry.slot: entry.quote for entry in answers.anchors}
        for slot_name in ANCHOR_SLOTS:
            quote = quotes.get(slot_name)
            if not quote or quote == NOT_ESTABLISHED or not _quote_is_verbatim(
                quote, shown_text
            ):
                return "NEEDS_REVIEW", GUARD_REASON_NOT_ESTABLISHED
        return status, ""
    if verdict == "not_established":
        return "NEEDS_REVIEW", GUARD_REASON_NOT_ESTABLISHED
    return status, ""


# --------------------------------------------------------------------------------------
# The inclusion-only second pass runs as
# a concurrent stage, judging a whole round's own INCLUDE candidates together instead of
# one sequential call per original batch of ten. A reasoning-on second-pass call was
# observed to take 43 to 315s each; sequentially awaiting one per batch of ten inside
# app.services.smart_search's own per-batch loop meant a production job with a few hundred
# candidates could spend hours there against a 30-minute search budget.
# --------------------------------------------------------------------------------------

#: Records per second-pass call under the concurrent stage. Batching is kept (a single call
#: still judges several records together, the original per-record cost saving); five keeps
#: one call's own footprint small relative to a round's whole candidate set, so a slow or
#: failed call demotes a small, bounded slice rather than the round's entire set.
SECOND_PASS_STAGE_BATCH_SIZE = 5

#: Guard reason for a second-pass candidate the stage never got an answer for -- whether its
#: own call raised, its own 420s timeout elapsed (SECOND_PASS_V2_MODEL_SETTINGS), or the
#: stage's own time budget ran out before its chunk could even be launched. One string for
#: all three: a caller (or a reader of the exported screening record) cannot tell "ran out
#: of stage budget" from "the model call itself failed", and does not need to -- both mean
#: "the second pass could not reach this record", exactly what this reason has always named.
GUARD_REASON_SECOND_PASS_UNAVAILABLE = "second_pass_unavailable"


@dataclass
class SecondPassStageDecision:
    """One candidate's own outcome from :func:`run_second_pass_stage`, in the candidate's
    own input position."""

    status: str  # "INCLUDE" or "NEEDS_REVIEW"
    guard_reason: str
    #: The per-question answer (``population``/``outcome``/``study_type``), or ``None`` when
    #: the second pass never reached this record at all (its own chunk failed, timed out, or
    #: was never launched for lack of stage budget).
    second_pass: dict[str, str] | None = None
    #: ``"<ExceptionType>: <message>"`` when this record's own chunk raised, so the cause
    #: of a ``second_pass_unavailable`` guard reason is
    #: recoverable, not just its count; ``None`` for a confirmed/demoted answer or a chunk
    #: never launched for lack of stage budget (no exception occurred there at all).
    error: str | None = None


@dataclass
class SecondPassStageCall:
    """One actual model call the stage made, kept individually (never merged across chunks)
    so a caller's own per-call latency/token/model/fingerprint reporting is identical in
    shape to the one-call-per-batch code this stage replaces.

    ``indices`` are this call's own candidates' positions in
    :func:`run_second_pass_stage`'s own input order, so a caller that must attribute this
    call to specific records (evaluation/screening/run_screening.py's own per-row
    ``second_pass_call_id``/token columns) does not have to re-derive chunk boundaries
    itself; app.services.smart_search does not need this field, since it reads
    ``decisions`` directly in candidate order instead."""

    provenance: LLMCallProvenance
    latency_s: float
    n_records: int
    indices: list[int] = dc_field(default_factory=list)


@dataclass
class SecondPassStageResult:
    """Every candidate's own decision, parallel to :func:`run_second_pass_stage`'s own
    input, plus the calls that produced them and the stage's own wall-clock time."""

    decisions: list[SecondPassStageDecision]
    calls: list[SecondPassStageCall] = dc_field(default_factory=list)
    #: Wall-clock time for the whole stage (one ``time.monotonic()``-like span around the
    #: concurrent gather), not the sum of the calls' own latencies -- those overlap under
    #: concurrency and would double-count the time actually spent.
    wall_time_s: float = 0.0
    #: Candidates whose own chunk was never launched because the stage's own deadline had
    #: already passed when its turn came (routed to NEEDS_REVIEW with
    #: GUARD_REASON_SECOND_PASS_UNAVAILABLE, same as a chunk that raised or timed out).
    n_skipped_budget: int = 0


async def run_second_pass_stage(
    candidates: list[dict],
    shown_texts: list[str],
    *,
    research_question: str,
    inclusion_criteria: list[str] | None,
    exclusion_criteria: list[str] | None,
    model: str | None = None,
    concurrency: int | None = None,
    batch_size: int = SECOND_PASS_STAGE_BATCH_SIZE,
    stage_deadline: float | None = None,
    now_fn: Callable[[], float] = time.monotonic,
    judge: Callable[..., Awaitable[Any]] | None = None,
    should_abort: Callable[[], Awaitable[bool]] | None = None,
    on_chunk_done: Callable[[], Awaitable[None]] | None = None,
) -> SecondPassStageResult:
    """Judge every one of ``candidates`` (an INCLUDE survivor of the batch pass and the two
    deterministic demotions) with the inclusion-only second pass, concurrently.

    Splits ``candidates`` into chunks of ``batch_size`` (default
    :data:`SECOND_PASS_STAGE_BATCH_SIZE`) and runs up to ``concurrency`` (default
    ``settings.screener_second_pass_concurrency`` when not given) of ``judge`` calls
    (default :func:`confirm_inclusions`, overridable so a test can inject a fake judge with
    the same call shape and no model or network) concurrently, bounded by an
    ``asyncio.Semaphore``. Each chunk's own answers are scored with
    :func:`apply_second_pass_guard`, unchanged from the sequential call this replaces.

    ``stage_deadline`` (an absolute ``now_fn()``-comparable time; ``None`` means unbounded)
    is checked before a chunk is launched, not mid-call: a chunk already in flight always
    finishes (a call already paid for is not abandoned), but a chunk that would start on or
    after the deadline is never sent at all -- its own candidates are routed to
    NEEDS_REVIEW with :data:`GUARD_REASON_SECOND_PASS_UNAVAILABLE`, counted in
    ``n_skipped_budget``. The check runs twice per chunk (once before it queues for the
    semaphore, once after acquiring it) since a chunk queued behind a slow call can cross
    the deadline while it waits its turn.

    ``should_abort``, an optional async predicate, is checked
    only at the second of those two points, after the semaphore is acquired: the first,
    pre-semaphore point checks the deadline alone, a bare
    monotonic-clock comparison, never ``should_abort``, since every chunk reaches that point
    at once (``asyncio.gather`` starts them all together, before any of them has acquired
    the semaphore) -- checking ``should_abort`` there too would run one session checkout per
    chunk concurrently, unbounded by ``concurrency``, rather than at most
    ``effective_concurrency`` many at once the way every actual judge call already is. A
    chunk that has not yet started its own judge call when ``should_abort`` returns ``True``
    is routed to NEEDS_REVIEW with :data:`GUARD_REASON_SECOND_PASS_UNAVAILABLE` and never
    sent, the same as a deadline miss -- a job cancelled mid-stage stops launching new calls
    (at the next chunk boundary, not mid-call) rather than waiting out the rest of its own
    budget. A chunk already in flight when cancellation is noticed still finishes cleanly
    (bounded by its own timeout), the same "a call already paid for is not abandoned" rule
    the deadline already follows; ``run_smart_search`` is what turns a cancelled job's
    remaining candidates into NEEDS_REVIEW outright once this returns, rather than waiting
    for this function to launch them all only to skip every one.

    ``on_chunk_done``, when given, is awaited exactly once per
    chunk -- whatever ended it (a completed call, a failure, a deadline miss or an abort) --
    so a caller can drive an incremental progress update while a stage with many chunks is
    still running, instead of the interface going quiet until the whole stage finishes.

    Returns one :class:`SecondPassStageDecision` per candidate, in input order, regardless
    of which chunk produced it, one :class:`SecondPassStageCall` per chunk actually sent,
    and the stage's own wall-clock time (measured once around the whole concurrent gather,
    not summed from the calls' own latencies, which overlap under concurrency).
    """
    # Resolved here, not bound as the parameter's own default value: a default of
    # ``confirm_inclusions`` would capture that name's value once, at import time, so a
    # test's ``patch.object(relevance_screener_agent, "confirm_inclusions", ...)`` would
    # have no effect on a caller that never passes ``judge`` explicitly. Looking the name
    # up here, at call time, is what makes that patch (the same one this module's own
    # callers already use for every other agent function) actually take effect.
    judge_fn = judge if judge is not None else confirm_inclusions
    n = len(candidates)
    decisions: list[SecondPassStageDecision | None] = [None] * n
    calls: list[SecondPassStageCall] = []
    skipped = 0
    effective_concurrency = max(
        1, concurrency if concurrency is not None else settings.screener_second_pass_concurrency
    )
    semaphore = asyncio.Semaphore(effective_concurrency)
    stage_start = now_fn()
    chunks = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

    def _route_unavailable(indices: list[int], *, error: str | None = None) -> None:
        for idx in indices:
            decisions[idx] = SecondPassStageDecision(
                status="NEEDS_REVIEW",
                guard_reason=GUARD_REASON_SECOND_PASS_UNAVAILABLE,
                error=error,
            )

    def _deadline_passed() -> bool:
        return stage_deadline is not None and now_fn() >= stage_deadline

    async def _stop_launching() -> bool:
        return _deadline_passed() or (should_abort is not None and await should_abort())

    async def _run_chunk(indices: list[int]) -> None:
        try:
            await _run_chunk_inner(indices)
        finally:
            if on_chunk_done is not None:
                await on_chunk_done()

    async def _run_chunk_inner(indices: list[int]) -> None:
        nonlocal skipped
        # This first check is the deadline alone, never
        # ``should_abort`` -- every chunk reaches this point at once (asyncio.gather starts
        # them all together, before any of them has acquired the semaphore), so a
        # ``should_abort`` here would run one session checkout per chunk concurrently,
        # unbounded by ``concurrency``. The deadline check is a bare monotonic-clock
        # comparison, not I/O, so running it unbounded costs nothing.
        if _deadline_passed():
            skipped += len(indices)
            _route_unavailable(indices)
            return
        async with semaphore:
            # Both the deadline and ``should_abort`` are checked here, after acquiring the
            # semaphore: at most ``effective_concurrency`` chunks hold the semaphore at
            # once, so at most that many ``should_abort`` session checkouts run at once too.
            if await _stop_launching():
                skipped += len(indices)
                _route_unavailable(indices)
                return
            chunk_papers = [candidates[idx] for idx in indices]
            call_start = now_fn()
            try:
                result = await judge_fn(
                    chunk_papers,
                    research_question=research_question,
                    inclusion_criteria=inclusion_criteria,
                    exclusion_criteria=exclusion_criteria,
                    model=model,
                )
            except Exception as exc:
                # Logged here (the one place that always runs,
                # whatever the caller is) and carried on the decision itself, so the cause
                # of a second_pass_unavailable guard reason is recoverable from the logs or
                # from the result, not just its count.
                error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "run_second_pass_stage: chunk of %d record(s) failed (%s)",
                    len(indices), error,
                )
                _route_unavailable(indices, error=error)
                return
            calls.append(
                SecondPassStageCall(
                    provenance=result.provenance,
                    latency_s=now_fn() - call_start,
                    n_records=len(indices),
                    indices=list(indices),
                )
            )
            answers = list(result.answers)
            for idx, answer in zip(indices, answers):
                new_status, reason = apply_second_pass_guard(
                    "INCLUDE", "", answers=answer, shown_text=shown_texts[idx],
                )
                second_pass: dict[str, str] | None = None
                if hasattr(answer, "population") and hasattr(answer, "outcome"):
                    second_pass = {
                        "population": answer.population.established,
                        "outcome": answer.outcome.established,
                        "study_type": answer.study_type,
                    }
                decisions[idx] = SecondPassStageDecision(
                    status=new_status, guard_reason=reason, second_pass=second_pass,
                )
            # A chunk answer short of its own input (padded by confirm_inclusions to
            # "not_established", never silently confirmed) still leaves every index
            # assigned above; nothing further to do for a short answers list here.

    await asyncio.gather(*(_run_chunk(idx_list) for idx_list in chunks))

    wall_time = now_fn() - stage_start
    # Every candidate must end up with a decision; this is a defensive fallback only (every
    # branch above assigns one), so a bug here demotes to NEEDS_REVIEW rather than crashing
    # a caller's zip or, worse, silently promoting an unconfirmed INCLUDE.
    final_decisions = [
        d if d is not None
        else SecondPassStageDecision(
            status="NEEDS_REVIEW", guard_reason=GUARD_REASON_SECOND_PASS_UNAVAILABLE,
        )
        for d in decisions
    ]
    return SecondPassStageResult(
        decisions=final_decisions, calls=calls, wall_time_s=wall_time, n_skipped_budget=skipped,
    )
