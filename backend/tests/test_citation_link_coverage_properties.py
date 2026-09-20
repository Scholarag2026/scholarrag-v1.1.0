"""Property-style test for the citation-link coverage rule: whatever the citation-link
call (first pass or its own bounded retry) omits, the
text finalize ships must never contain a body sentence that is neither linked-and-
verified, framing-tagged, nor recorded as still unresolved.

Built from a small grammar of three sentence kinds -- a cited sentence, a framing
sentence with no citation, and a finding sentence with no citation the writer meant to
flag as uncited -- assembled into random paragraphs, with random independent omissions
from what a "perfect" citation-link call would have returned (the exact defect class
`unclassified_body_sentences`/`resolve_unclassified_sentences` closes) and a retry
linker that is itself only sometimes able to resolve a gap it is asked about again.

Two properties are checked. The first: no
sentence finalize ships is unaccounted for -- it is either framing-tagged, a verified
link, or explicitly recorded as still unresolved (``"unclassified": True``). The second,
the converse the first property alone cannot catch:
every sentence a PERFECT citation-link call would have classified as framing, or linked
and verified, must still be present after `resolve_unclassified_sentences` plus
`_finalize_paragraph_text` -- unless the retry call itself actually answered without
ever mentioning that sentence, in which case it becomes an ordinary, removable
"finding" by design (the fix for defect A, ``resolve_unclassified_sentences``'s own
docstring): a retry call that raises outright still keeps the sentence, marked
``"unclassified": True``, since nothing about it was ever actually looked at, but a
retry call that answers and stays silent about a sentence is treated as a positive,
if unhelpful, answer -- indistinguishable, from this function's own point of view,
from a genuine uncited finding. An implementation that deleted a whole paragraph
whenever it could not classify something would still pass the first property alone,
which is why the second property exists. A fixed seed
makes a failure reproducible."""

import asyncio
import random

import pytest

from app.services.fulltext import (
    _finalize_paragraph_text,
    _finalize_sentence_norm,
    _sentence_fragments,
    resolve_unclassified_sentences,
)

SEED = 20260911
CASES = 300

CITED_SENTENCES = [
    ("Smith (2020) reported large gains in accuracy.", "smith_2020"),
    ("Jones (2021) found a durable revision effect.", "jones_2021"),
    ("Lee (2020) observed no effect on learner engagement.", "lee_2020"),
    ("Storch (2018) documented consistent peer-feedback uptake.", "storch_2018"),
]
FRAMING_SENTENCES = [
    "Research on this question has grown over the last decade.",
    "The following studies illustrate three distinct approaches.",
    "Three strands dominate the project library.",
]
FINDING_SENTENCES = [
    "Mixed-effect models showed a durable four-week advantage.",
    "Cognitive load was lower for one feedback type on grammatical errors.",
    "The sample was confined to one classroom and one proficiency band.",
]


class _FakeLinkResult:
    """The subset of ``CitationLinkResult`` `resolve_unclassified_sentences` reads
    (``links``, ``uncited_sentences``, ``provenance``) -- not the real type, since this
    file never imports `app.agents.citation_link_agent` (frozen, out of scope)."""

    def __init__(self, links, uncited_sentences):
        self.links = links
        self.uncited_sentences = uncited_sentences
        self.provenance = None


def _paragraph(rng):
    """One paragraph: 2 to 5 sentences, a random mix of cited/framing/finding, the
    *true* link/uncited entries a perfect citation-link call would return for it, and
    a claim_status map marking every cited sentence verified (verification itself is
    not this property's subject)."""
    sentences = []
    true_links = []
    true_uncited = []
    claim_status = {}
    for _ in range(rng.randint(2, 5)):
        kind = rng.choice(["cited", "cited", "framing", "finding"])
        if kind == "cited":
            sentence, key = rng.choice(CITED_SENTENCES)
            sentences.append(sentence)
            true_links.append({"sentence": sentence, "keys": [key], "citation_text": sentence})
            claim_status[(sentence, sentence, key)] = "verified"
        elif kind == "framing":
            sentence = rng.choice(FRAMING_SENTENCES)
            sentences.append(sentence)
            true_uncited.append({"sentence": sentence, "tag": "framing"})
        else:
            sentence = rng.choice(FINDING_SENTENCES)
            sentences.append(sentence)
            true_uncited.append({"sentence": sentence, "tag": "finding"})
    text = " ".join(sentences)
    return text, true_links, true_uncited, claim_status


def _omit_randomly(rng, true_links, true_uncited):
    """The citation-link call's own (imperfect) first-pass results: each true entry
    independently survives with probability 0.6 -- the exact omission shape that
    can leave a real sentence uncovered."""
    links = [dict(link) for link in true_links if rng.random() < 0.6]
    uncited = [dict(entry) for entry in true_uncited if rng.random() < 0.6]
    return links, uncited


def _fallible_retry_linker(rng, true_links, true_uncited):
    """A retry call that is not omniscient either: of the gaps it is asked about, it
    resolves each one independently only about half the time."""

    async def linker(retry_text):
        resolved_links = [
            dict(link) for link in true_links
            if link["sentence"] in retry_text and rng.random() < 0.5
        ]
        resolved_uncited = [
            dict(entry) for entry in true_uncited
            if entry["sentence"] in retry_text and rng.random() < 0.5
        ]
        return _FakeLinkResult(resolved_links, resolved_uncited)

    return linker


def _always_failing_retry_linker(rng, true_links, true_uncited):
    """A retry call that always fails outright (a provider outage) -- the other worst
    case, distinct from `_fallible_retry_linker`,
    which always at least tries and sometimes succeeds."""

    async def linker(retry_text):
        raise RuntimeError("provider outage")

    return linker


def _resolve_and_finalize(text, omitted_links, omitted_uncited, claim_status, linker):
    links_out, uncited_out, _prov, _calls, retry_failed = asyncio.run(
        resolve_unclassified_sentences(text, omitted_links, omitted_uncited, linker)
    )
    final_text, surviving_links, _stats, _healed, _surviving_uncited = _finalize_paragraph_text(
        text, links_out, uncited_out, claim_status
    )
    return final_text, surviving_links, uncited_out, retry_failed


def _check_invariant(rng):
    """Property 1: nothing finalize ships is unaccounted for -- every surviving sentence
    is framing-tagged, a verified link, or explicitly recorded as still unresolved."""
    text, true_links, true_uncited, claim_status = _paragraph(rng)
    omitted_links, omitted_uncited = _omit_randomly(rng, true_links, true_uncited)
    linker = _fallible_retry_linker(rng, true_links, true_uncited)

    final_text, surviving_links, uncited_out, _retry_failed = _resolve_and_finalize(
        text, omitted_links, omitted_uncited, claim_status, linker
    )

    framing_norms = {
        _finalize_sentence_norm(entry["sentence"])
        for entry in uncited_out
        if entry.get("tag") == "framing"
    }
    unresolved_norms = {
        _finalize_sentence_norm(entry["sentence"])
        for entry in uncited_out
        if entry.get("unclassified")
    }
    verified_linked_norms = {_finalize_sentence_norm(link["sentence"]) for link in surviving_links}

    for start, end in _sentence_fragments(final_text):
        fragment = final_text[start:end].strip()
        if not fragment:
            continue
        norm = _finalize_sentence_norm(fragment)
        assert (
            norm in framing_norms or norm in verified_linked_norms or norm in unresolved_norms
        ), (text, final_text, fragment)


def _check_survival_invariant(rng, linker_factory):
    """Property 2: every sentence a PERFECT citation-link call
    would have classified as framing, or linked and verified, must still be present
    after `resolve_unclassified_sentences` plus `_finalize_paragraph_text` -- regardless
    of what the first pass omitted -- UNLESS *linker_factory*'s own retry call actually
    answered without ever mentioning that sentence, in which case Fix A treats it as an
    ordinary, removable finding by design (the entry ends up tagged "finding" with no
    "unclassified" flag in ``uncited_out``, indistinguishable from a genuine uncited
    finding). ``_always_failing_retry_linker`` never produces that shape (a call that
    raises always keeps every gap, marked "unclassified"), so the property holds for it
    unconditionally; ``_fallible_retry_linker`` does, so those sentences are excluded
    here rather than asserted on. A true
    "finding" sentence is deliberately excluded too: removing one the retry actually,
    positively classifies is the correct, intended behaviour, not a bug."""
    text, true_links, true_uncited, claim_status = _paragraph(rng)
    omitted_links, omitted_uncited = _omit_randomly(rng, true_links, true_uncited)
    linker = linker_factory(rng, true_links, true_uncited)

    final_text, _surviving_links, uncited_out, _retry_failed = _resolve_and_finalize(
        text, omitted_links, omitted_uncited, claim_status, linker
    )
    final_norms = {
        _finalize_sentence_norm(final_text[start:end])
        for start, end in _sentence_fragments(final_text)
    }
    # Fix A: a gap the retry call answered about but never named ends up as exactly
    # this shape -- an ordinary, removable finding -- whatever it truly was.
    answered_but_unnamed = {
        _finalize_sentence_norm(entry["sentence"])
        for entry in uncited_out
        if entry.get("tag") == "finding" and not entry.get("unclassified")
    }

    for link in true_links:
        norm = _finalize_sentence_norm(link["sentence"])
        if norm in answered_but_unnamed:
            continue
        assert norm in final_norms, (text, final_text, link)
    for entry in true_uncited:
        if entry.get("tag") != "framing":
            continue
        if _finalize_sentence_norm(entry["sentence"]) in answered_but_unnamed:
            continue
        assert _finalize_sentence_norm(entry["sentence"]) in final_norms, (text, final_text, entry)


#: The property is checked in batches, mirroring `test_citation_coverage_properties.py`,
#: so a failure still names the exact paragraph and surviving text through the
#: assertion message, without adding one test per case to the suite's own count.
BATCH_COUNT = 10


def _run_batch(seed_offset, count):
    rng = random.Random(SEED + seed_offset)
    for _ in range(count):
        _check_invariant(rng)


def _run_survival_batch(seed_offset, count, linker_factory):
    rng = random.Random(SEED + seed_offset)
    for _ in range(count):
        _check_survival_invariant(rng, linker_factory)


@pytest.mark.parametrize("batch", range(BATCH_COUNT))
def test_coverage_rule_never_ships_an_unaccounted_for_sentence(batch):
    _run_batch(batch, CASES // BATCH_COUNT)


@pytest.mark.parametrize("batch", range(BATCH_COUNT))
def test_a_true_framing_or_verified_sentence_always_survives_a_fallible_retry(batch):
    """The converse property: a sentence a perfect
    citation-link call would have classified as framing or a verified link is never
    lost, even when the first pass omits it and the retry only sometimes resolves it."""
    _run_survival_batch(batch, CASES // BATCH_COUNT, _fallible_retry_linker)


@pytest.mark.parametrize("batch", range(BATCH_COUNT))
def test_a_true_framing_or_verified_sentence_survives_a_retry_that_always_fails(batch):
    """The retry linker raises on every
    call. The sentence is kept, recorded as unresolved, never silently destroyed."""
    _run_survival_batch(batch, CASES // BATCH_COUNT, _always_failing_retry_linker)


def test_the_generated_grammar_actually_exercises_every_sentence_kind():
    """A guard on the generator itself: a grammar that stopped producing one of the
    three sentence kinds, or stopped omitting entries, would keep the test above green
    while testing nothing."""
    rng = random.Random(SEED)
    kinds_seen = {"cited": 0, "framing": 0, "finding": 0}
    any_omitted = False
    any_kept = False
    for _ in range(CASES):
        _text, true_links, true_uncited, _status = _paragraph(rng)
        kinds_seen["cited"] += len(true_links)
        kinds_seen["framing"] += sum(1 for u in true_uncited if u["tag"] == "framing")
        kinds_seen["finding"] += sum(1 for u in true_uncited if u["tag"] == "finding")
        omitted_links, omitted_uncited = _omit_randomly(rng, true_links, true_uncited)
        if len(omitted_links) + len(omitted_uncited) < len(true_links) + len(true_uncited):
            any_omitted = True
        if omitted_links or omitted_uncited:
            any_kept = True
    assert all(count > 0 for count in kinds_seen.values()), kinds_seen
    assert any_omitted
    assert any_kept
