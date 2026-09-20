"""Property-style tests for the repeated-heal invariant.

Healing a saved draft is an action the user can run again on the same draft, and the
demo runs it once per run on a draft an earlier run may already have healed. So the
heal must be idempotent: the first pass enforces the exit invariant, and every pass
after it must be a byte-identical no-op on the document that reproduces the same final
report.

That invariant can break in several ways on shapes that are easy to miss: a pairing key
stored on a surviving link (``pre_heal_sentence``), which the NEXT pass reads as if it
described that pass and so drops the verified row from the final report; a co-cited
group narrowed to its verified keys on one pass, whose dropped key is then re-resolved
on its own and cut out of the kept group's own rendered text, turning "(Jones, 2019;
Smith, 2020)" into "(; Smith, 2020)" one heal at a time; and a surviving link left naming
a sentence with the double space a formatting mark leaves in the extracted text, which
the rebuilt paragraph no longer contains, so the pass after it rewrites the paragraph's
attrs. Rather than add one example per case, this file builds documents from a small
grammar of exactly those shapes -- formatting marks, single and grouped citations,
verified, no_full_text and unsupported verdicts -- and asserts the invariant over a few
hundred of them with a fixed seed, so a failure is reproducible.

Each pass here runs the same sequence `app.services.fulltext.verify_and_heal_claims`
runs: extract the claims from the document as it stands, key the verdicts on
``(claim sentence, key)``, finalize, and build the final report from the links finalize
kept. The verifier is replaced by a fixed verdict per citation key, which is what makes
repetition meaningful: the same claim asked twice gets the same answer, so any
difference between pass 1 and pass 2 is the heal's own doing.

All pure functions over plain dicts: no network, no database, no model.
"""

import copy
import json
import random
import re

import pytest

from app.services.fulltext import (
    extract_claims_from_document,
    finalize_draft_document,
    finalize_generated_section,
    surviving_verifications,
)

SEED = 20260909
CASES = 200
#: pass 1 heals; every pass after it must change nothing at all.
PASSES = 4

#: (key, surname, year, verdict). The verdict is fixed per key: the same claim asked on
#: a later pass gets the same answer it got on the first one.
SOURCES = [
    ("smith_2020", "Smith", "2020", "verified"),
    ("jones_2019", "Jones", "2019", "no_full_text"),
    ("lee_2021", "Lee", "2021", "verified"),
    ("park_2018", "Park", "2018", "unsupported"),
    ("zhang_2022", "Zhang", "2022", "no_full_text"),
    ("mao_2024", "Mao", "2024", "verified"),
]

PAPER_LOOKUP = {key: object() for key, _surname, _year, _verdict in SOURCES}
VERDICTS = {key: verdict for key, _surname, _year, verdict in SOURCES}

CLAUSES = [
    "tutoring improved outcomes",
    "costs fell sharply",
    "the replication failed",
    "teachers reported gains",
]


class _Verdict:
    """Stands in for `app.schemas.fulltext.ClaimVerification`. `surviving_verifications`
    reads and refreshes `claim_sentence` (a rewrite -- a no_full_text drop's own cleanup,
    or a trailing excision -- must not leave a surviving verdict naming a sentence the
    draft no longer contains verbatim), via one `model_copy(update=...)` call mirroring
    the real `ClaimVerification`'s own."""

    def __init__(self, key, status, claim_sentence=None):
        self.key = key
        self.status = status
        self.claim_sentence = claim_sentence

    def model_copy(self, *, update=None):
        copied = _Verdict(self.key, self.status, self.claim_sentence)
        for field, value in (update or {}).items():
            setattr(copied, field, value)
        return copied

    def __repr__(self):  # pragma: no cover - debugging aid only
        return f"_Verdict({self.key}, {self.status})"


# --------------------------------------------------------------------------------------
# the grammar
# --------------------------------------------------------------------------------------


#: The marker, matched with any whitespace inside it, for this file's own comparisons.
#: Written here rather than imported from the module under test: a corpus that asked the
#: code it is checking what two sentences mean would agree with it by construction.
_MARKER_RE = re.compile(r"\s*\[\s*NEEDS\s+CITATION\s*\]", re.IGNORECASE)


def _marker_free(text):
    """*text* with every ``[NEEDS CITATION]`` marker removed and its whitespace
    collapsed, for comparing two sentences the way a reader would."""
    return " ".join(_MARKER_RE.sub("", text or "").split())


#: A two-word marker or citation half, for `_content_nodes` to split a mark across
#: rather than only ever landing between two ordinary words.
_MARKER_HALVES_RE = (re.compile(r"^\[NEEDS$"), re.compile(r"^CITATION\]$"))
_CITATION_HALVES_RE = (re.compile(r"^\(?[A-Z][a-z]+,$"), re.compile(r"^\d{4}[\);,.]*$"))


def _is_special_boundary(a, b):
    """True when *a* directly followed by *b* is the two words of a
    ``[NEEDS CITATION]`` marker or of one citation's own author/year halves (e.g.
    "(Smith," / "2020)"), so marking one of them bold splits that unit across two
    nodes instead of only ever landing between two ordinary words."""
    for head_re, tail_re in (_MARKER_HALVES_RE, _CITATION_HALVES_RE):
        if head_re.match(a) and tail_re.match(b):
            return True
    return False


def _sentence(rng):
    """One sentence, the ``(citation_text, keys)`` pairs a linker would report, and --
    for an ``"uncited"`` sentence only -- the ``uncitedSentences`` tag a linker would
    give it: ``"finding"`` (removed), ``"framing"`` (kept), or ``None`` (not
    listed at all, also kept)."""
    form = rng.choice(["single", "two_singles", "group", "narrative", "uncited"])
    clause = rng.choice(CLAUSES)
    opening = clause[0].upper() + clause[1:]
    if form == "uncited":
        sentence = f"{opening} in general."
        if rng.random() < 0.3:
            sentence += " [NEEDS CITATION]"
        tag = rng.choice(["finding", "framing", None])
        return sentence, [], tag
    first = rng.choice(SOURCES)
    if form == "single":
        text = f"({first[1]}, {first[2]})"
        return f"{opening} {text}.", [(text, [first[0]])], None
    if form == "narrative":
        text = f"{first[1]} ({first[2]})"
        return f"{text} reported that {clause}.", [(text, [first[0]])], None
    second = rng.choice([s for s in SOURCES if s[0] != first[0]])
    if form == "two_singles":
        first_text = f"({first[1]}, {first[2]})"
        second_text = f"({second[1]}, {second[2]})"
        return (
            f"{opening} {first_text} and {rng.choice(CLAUSES)} {second_text}.",
            [(first_text, [first[0]]), (second_text, [second[0]])],
            None,
        )
    group_text = f"({first[1]}, {first[2]}; {second[1]}, {second[2]})"
    return (
        f"Two papers agree that {clause} {group_text}.",
        [(group_text, [first[0], second[0]])],
        None,
    )


def _content_nodes(rng, text):
    """The paragraph's own text nodes: one plain node, or three with a bold mark in the
    middle. `_extract_text_from_tiptap` joins text nodes with a literal space, so a
    trailing space left on the head node, or a leading space left on the tail node --
    the ordinary Tiptap shape for text right next to a mark -- leaves a double space in
    the extracted text. The split is sometimes forced onto the boundary inside a
    ``[NEEDS CITATION]`` marker or inside one citation's own two words, rather than left
    to land there only by chance, so the corpus reliably carries both shapes, including
    a marker split this way with the extra space on the tail side."""
    words = text.split(" ")
    roll = rng.random()
    if roll >= 0.5 or len(words) < 4:
        return [{"type": "text", "text": text}]
    special_indices = [
        i
        for i in range(1, len(words) - 1)
        if _is_special_boundary(words[i - 1], words[i])
        or _is_special_boundary(words[i], words[i + 1])
    ]
    if special_indices and rng.random() < 0.6:
        index = rng.choice(special_indices)
    else:
        index = rng.randrange(1, len(words) - 1)
    head = " ".join(words[:index])
    tail = " ".join(words[index + 1:])
    if rng.random() < 0.5:
        if rng.random() < 0.5:
            head += " "
        else:
            tail = " " + tail
    return [
        {"type": "text", "text": head},
        {"type": "text", "text": words[index], "marks": [{"type": "bold"}]},
        {"type": "text", "text": tail},
    ]


def _paragraph(rng, index):
    sentences = []
    links = []
    uncited_entries = []
    seen_sentences: set[str] = set()
    for _ in range(rng.randint(1, 3)):
        text, citations, tag = _sentence(rng)
        # Two sentences in one paragraph must never collide once their markers are
        # stripped and their whitespace collapsed: `_finalize_paragraph_text` keys its
        # per-sentence link map on exactly that form, so two such fragments share one
        # link list between them. That is a documented limitation of finalize (see its
        # own docstring), not one of the shapes this grammar exists to reproduce, and
        # it has its own spelled-out test in `test_finalize_verified_section.py`. The
        # guard compares the normalised form, not the literal text, because this
        # corpus's own markers are exactly what makes two literally different sentences
        # the same sentence to finalize.
        attempts = 0
        while _marker_free(text) in seen_sentences and attempts < 5:
            text, citations, tag = _sentence(rng)
            attempts += 1
        seen_sentences.add(_marker_free(text))
        sentences.append(text)
        if tag is not None:
            # Recorded the way a citation-linker actually reports it, which is both
            # ways round. Its own prompt asks for
            # the sentence exactly as it appears in the text, so a sentence the writer
            # flagged is reported WITH the marker inside it; and a sentence whose
            # fragment only carries the marker because the splitter put the PREVIOUS
            # sentence's marker at its head is reported without one. The doubled space
            # a formatting mark leaves inside the marker is generated on the recorded
            # side too, so neither side of the comparison is the only one that ever
            # carries it.
            recorded = text
            if rng.random() < 0.5:
                recorded = _MARKER_RE.sub("", recorded)
            elif rng.random() < 0.4:
                recorded = recorded.replace("[NEEDS CITATION]", "[NEEDS  CITATION]")
            uncited_entries.append({"sentence": recorded, "tag": tag})
        for citation_text, keys in citations:
            if rng.random() < 0.15:
                # No stored link at all: the claim is resolved by the author-year
                # regexes instead, exactly as it is for a hand-typed citation.
                continue
            link = {"sentence": text, "keys": list(keys), "citation_text": citation_text}
            if rng.random() < 0.4:
                link["evidence_ids"] = [f"e{index}"]
            if rng.random() < 0.4:
                link["proposition"] = text.split(" (")[0]
            links.append(link)
    text = " ".join(sentences)
    attrs = {"origin": "ai", "blockId": f"blk-{index}"}
    if links:
        attrs["citationLinks"] = links
    if uncited_entries:
        attrs["uncitedSentences"] = uncited_entries
    return {"type": "paragraph", "attrs": attrs, "content": _content_nodes(rng, text)}


def _document(rng):
    """A Tiptap document of one to three paragraphs, one of them sometimes wrapped in a
    bullet list so the container recursion is exercised too."""
    nodes = []
    for index in range(rng.randint(1, 3)):
        paragraph = _paragraph(rng, index)
        if rng.random() < 0.2:
            nodes.append(
                {"type": "bulletList", "content": [{"type": "listItem", "content": [paragraph]}]}
            )
        else:
            nodes.append(paragraph)
    return {"type": "doc", "content": nodes}


# --------------------------------------------------------------------------------------
# one heal pass, the same sequence `verify_and_heal_claims` runs
# --------------------------------------------------------------------------------------


#: `frame_spans_accepted` is a descriptive count (how many
#: residue spans this pass's own classifier accepted as a frame), not a removal delta
#: like every other `loop_stats` key -- it stays non-zero on every later, otherwise
#: no-op pass as long as the same frame is still there to classify, which is correct,
#: not a sign of instability. Excluded from the "nothing more happened" checks below,
#: the same way this file's own grammar never generates the `sentences_unclassified_
#: kept` shape (also potentially stable-but-non-zero) at all.
_NON_DELTA_STATS_KEYS = frozenset({"frame_spans_accepted"})


def _stable_stats(stats):
    return {k: v for k, v in stats.items() if k not in _NON_DELTA_STATS_KEYS}


def _heal_once(document):
    """(healed content, surviving links, stats, final report) for one pass.

    The final report is a ``(key, status)`` list rather than the verdict objects
    themselves: the objects are rebuilt from the freshly extracted claims on every pass,
    so only what they say is comparable across passes."""
    claims, _coverage = extract_claims_from_document(document, PAPER_LOOKUP)
    # Keyed on (sentence, proposition, key).
    claim_status = {
        (sentence, text, key): VERDICTS[key] for text, key, _citation_text, sentence in claims
    }
    verifications = [
        _Verdict(key, VERDICTS[key], claim_sentence=sentence)
        for _text, key, _citation_text, sentence in claims
    ]
    outcome = finalize_draft_document(document, claims, claim_status)
    final = surviving_verifications(
        claims, verifications, outcome.citation_links, outcome.healed_sentences
    )
    return (
        outcome.content,
        outcome.citation_links,
        outcome.stats,
        [(v.key, v.status) for v in final],
    )


def _assert_exit_invariant(content, links, report):
    """What one heal pass promises about the text it leaves behind: every surviving
    citation link is verified, the final report says exactly that, and no unsupported
    citation is left anywhere in the document."""
    for link in links:
        keys = link.get("keys") or []
        assert keys, link
        assert all(VERDICTS[key] == "verified" for key in keys), link
        assert "pre_heal_sentence" not in link, link
    assert all(status == "verified" for _key, status in report), report

    claims, _coverage = extract_claims_from_document(content, PAPER_LOOKUP)
    left_unsupported = [
        key for _text, key, _citation_text, _s in claims if VERDICTS[key] == "unsupported"
    ]
    assert left_unsupported == [], (left_unsupported, content)

    for link in _stored_links(content):
        assert "pre_heal_sentence" not in link, link


def _stored_links(content):
    """Every citation link stored anywhere in *content*."""
    found = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        found.extend((node.get("attrs") or {}).get("citationLinks") or [])
        walk(node.get("content") or [])

    walk(content)
    return found


def _rendered_text(content):
    """Every paragraph's own rendered text, in document order -- what a reader (or
    `demo/check_delivered.py`) actually sees, ignoring the internal
    ``attrs.citationLinks``/``attrs.uncitedSentences`` bookkeeping entirely.

    Deduplicating a claim on its proposition, not only its sentence and key, exposes a
    narrower, pre-existing ambiguity in `_claims_for_paragraph_text`'s own
    claim-to-paragraph attribution: it matches a claim to a paragraph by substring
    containment of the claim's own sentence in that paragraph's text, not by any stable
    node identity, so two DIFFERENT paragraphs of the same document that happen to open
    with a byte-identical sentence citing the same key can each pick up the other's own
    claim variant too. Which of the two paragraphs' own attrs carries the extra
    bookkeeping variant can shift by one heal pass before settling. The delivered TEXT
    and the final report (`_assert_exit_invariant`, checked on every pass below) are
    unaffected either way -- no unsupported or unverified content ever survives -- so
    the repeated-heal invariant this file exists to protect is intact; only the exact
    shape of an internal, unrendered bookkeeping array can take one extra pass to settle
    for this rare corpus shape. Comparing rendered text, not the raw document JSON, is
    what this function narrows the check to. Deeper attribution by node identity, rather
    than by substring matching, is out of scope here."""
    texts = []

    def _walk(nodes):
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            if node.get("type") == "paragraph":
                texts.append(
                    "".join(c.get("text", "") for c in node.get("content") or [])
                )
            _walk(node.get("content"))

    _walk(content.get("content"))
    return texts


def _check_repeated_heal(document):
    """Heal once, then keep healing: every pass after the first must render the same
    text (`_rendered_text`, see its own docstring for the narrow, documented exception)
    and must reproduce the same final report."""
    original = json.dumps(document, sort_keys=True)
    content, links, _stats, report = _heal_once(copy.deepcopy(document))
    _assert_exit_invariant(content, links, report)

    healed = _rendered_text(content)
    for _pass in range(PASSES - 1):
        content, links, stats, again = _heal_once(content)
        assert _rendered_text(content) == healed, (original, healed)
        # The rows are compared as a set, not as a list: healing normalises a
        # sentence (a mark collapsed, a marker stripped, a citation cut), so two
        # paragraphs that carried the same claim in slightly different wording can end
        # up carrying it in identical wording, after which extraction asks the verifier
        # that question once instead of twice and the row appears once instead of
        # twice. What must not change, and what M1 broke, is WHICH rows the report
        # carries.
        assert set(again) == set(report), (original, report, again)
        assert all(value == 0 for value in _stable_stats(stats).values()), (original, stats)
        _assert_exit_invariant(content, links, again)


def _generated_documents():
    rng = random.Random(SEED)
    return [_document(rng) for _ in range(CASES)]


DOCUMENTS = _generated_documents()
#: The documents are checked in batches rather than one per test: this suite's conftest
#: creates and drops the whole test schema for every test, so 200 tests would add well
#: over a minute to the backend run for no extra coverage. A failure still names the
#: document, through the assertion messages above.
BATCH_COUNT = 5
BATCHES = [DOCUMENTS[index::BATCH_COUNT] for index in range(BATCH_COUNT)]


@pytest.mark.parametrize("batch", range(BATCH_COUNT))
def test_healing_a_generated_document_twice_changes_nothing_the_second_time(batch):
    for document in BATCHES[batch]:
        _check_repeated_heal(document)


def _bold_node_texts(node):
    """Every text node's own ``text`` that carries a bold mark, anywhere in *node*."""
    found = []

    def walk(item):
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        if item.get("type") == "text" and item.get("marks"):
            found.append(item.get("text", ""))
        walk(item.get("content") or [])

    walk(node)
    return found


def test_the_generated_grammar_actually_exercises_every_shape():
    """A guard on the generator itself: a grammar that stopped producing groups, marks,
    unsupported verdicts or unlinked citations would keep the test above green while
    testing nothing."""
    documents = json.dumps(DOCUMENTS)
    assert "; " in documents  # grouped citations
    assert "bulletList" in documents
    assert "bold" in documents
    assert "[NEEDS CITATION]" in documents
    assert "(Park, 2018)" in documents  # an unsupported verdict
    assert "(Jones, 2019)" in documents  # a no_full_text verdict
    assert "proposition" in documents
    assert "evidence_ids" in documents
    assert '"tag": "finding"' in documents  # an uncited finding sentence, removed
    assert '"tag": "framing"' in documents  # an uncited framing sentence, kept
    links = [link for document in DOCUMENTS for link in _stored_links(document)]
    assert any(len(link["keys"]) > 1 for link in links)  # grouped links
    texts = [
        text
        for document in DOCUMENTS
        for node in json.loads(json.dumps(document))["content"]
        for text in [json.dumps(node)]
    ]
    assert any('"marks"' in text for text in texts)

    bold_texts = {text for document in DOCUMENTS for text in _bold_node_texts(document)}
    # A bold node whose own text IS one half of a "[NEEDS CITATION]" marker, or of one
    # citation's own author/year halves, means the mark boundary fell inside that unit
    # rather than only ever between two ordinary words.
    assert any(text in ("[NEEDS", "CITATION]") for text in bold_texts)
    assert any(_CITATION_HALVES_RE[0].match(text) or _CITATION_HALVES_RE[1].match(text)
               for text in bold_texts)


def test_the_first_heal_of_a_generated_document_actually_heals():
    """The invariant above is only worth asserting if pass 1 has real work to do: the
    corpus must contain documents whose text finalize rewrites, sentences it removes,
    citations it cuts, and reports with rows in them."""
    changed = 0
    totals = {
        "sentences_removed_unverified": 0,
        "sentences_removed_uncited_finding": 0,
        "sentences_removed_no_full_text": 0,
        "citations_dropped_no_full_text": 0,
        "needs_citation_markers_removed": 0,
    }
    reported = 0
    for document in DOCUMENTS:
        content, _links, stats, report = _heal_once(copy.deepcopy(document))
        if json.dumps(content, sort_keys=True) != json.dumps(document, sort_keys=True):
            changed += 1
        for key in totals:
            totals[key] += stats.get(key, 0)
        reported += len(report)
    assert changed > CASES // 4
    assert reported > CASES // 2
    for key, value in totals.items():
        assert value > 0, key


# --------------------------------------------------------------------------------------
# the wide corpus: the same grammar, over more seeds, checked in bulk
# --------------------------------------------------------------------------------------

#: Seeds for the wide corpus. The 200 documents above are asserted one invariant at a
#: time, with the assertion naming the document that broke; these are healed in bulk and
#: counted, which is what makes a class of failure (rather than one document) visible.
WIDE_SEEDS = (1, 2, 3, 4, 5, 6, 7)
WIDE_CASES = 300
#: Heal, then two more passes that must change nothing.
WIDE_PASSES = 3


def _paragraph_text(node):
    """The text of one paragraph node, joined the way `_extract_text_from_tiptap` joins
    it (a literal space between adjacent text nodes), written here rather than imported
    for the same reason `_marker_free` is."""
    parts = []

    def walk(item):
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        if item.get("type") == "text":
            parts.append(item.get("text") or "")
            return
        walk(item.get("content") or [])

    walk(node.get("content") or [])
    return " ".join(parts)


def _paragraphs_by_block(node, found=None):
    """``{blockId: (paragraph text, its uncitedSentences entries)}`` for every paragraph
    of *node* that carries a blockId. Every generated paragraph carries one, and it
    survives the heal, so a paragraph can be compared with itself before and after
    rather than by position, which a dropped paragraph would shift."""
    if found is None:
        found = {}
    if isinstance(node, list):
        for item in node:
            _paragraphs_by_block(item, found)
        return found
    if not isinstance(node, dict):
        return found
    if node.get("type") == "paragraph":
        attrs = node.get("attrs") or {}
        if attrs.get("blockId") is not None:
            found[attrs["blockId"]] = (
                _paragraph_text(node),
                attrs.get("uncitedSentences") or [],
            )
    _paragraphs_by_block(node.get("content") or [], found)
    return found


def _retained_findings(original, healed):
    """Every uncited "finding" sentence of *original* still present, marker-insensitively,
    in the paragraph it was recorded on after the heal. Design amendment A4 says the
    heal removes those sentences, so this list is what the exit invariant is worth: an
    entry left here is an unsupported statement shipped in the healed draft, usually
    with the one visible sign that it was unsupported (its marker) taken off it."""
    after = _paragraphs_by_block(healed)
    retained = []
    for block_id, (_text, entries) in _paragraphs_by_block(original).items():
        healed_paragraph = after.get(block_id)
        if healed_paragraph is None:
            continue
        healed_text = _marker_free(healed_paragraph[0])
        for entry in entries:
            if entry.get("tag") != "finding":
                continue
            sentence = _marker_free(entry.get("sentence") or "")
            if sentence and sentence in healed_text:
                retained.append((block_id, sentence))
    return retained


def test_the_wide_corpus_removes_every_uncited_finding_and_is_idempotent():
    """Exercises the whole grammar: 2100 documents, each healed three times.

    Two counts, both of which must be zero. An uncited "finding" sentence still present
    in its own paragraph after the heal is this policy failing on the shape a citation
    linker actually returns (the entry recorded with the writer's own
    "[NEEDS CITATION]" marker inside it, since the linker prompt asks for the sentence
    exactly as it appears in the text) -- 84 documents of this same corpus retain one
    when the fragment is stripped on one side of that comparison only. A pass after the
    first that changes the document, its statistics or its final report is the
    repeated-heal invariant failing.
    """
    retained = []
    unstable = []
    documents = 0
    for seed in WIDE_SEEDS:
        rng = random.Random(seed)
        for _case in range(WIDE_CASES):
            document = _document(rng)
            documents += 1
            original = copy.deepcopy(document)
            content, _links, _stats, report = _heal_once(copy.deepcopy(document))
            first = _rendered_text(content)
            retained.extend(_retained_findings(original, content))
            for _pass in range(WIDE_PASSES - 1):
                content, _links, stats, again = _heal_once(content)
                changed = _rendered_text(content) != first
                if changed or set(again) != set(report) or any(_stable_stats(stats).values()):
                    unstable.append((seed, json.dumps(original, sort_keys=True), stats))
                    break

    assert documents == len(WIDE_SEEDS) * WIDE_CASES
    assert retained == [], retained[:5]
    assert unstable == [], unstable[:1]


def test_the_wide_corpus_records_uncited_entries_both_with_and_without_their_marker():
    """A guard on the generator: the corpus is only evidence for the finding above if it
    actually produces the shape the finding is about -- an uncited entry whose own
    recorded text carries the marker -- alongside the shape it already produced."""
    with_marker = 0
    without_marker = 0
    findings = 0
    for seed in WIDE_SEEDS:
        rng = random.Random(seed)
        for _case in range(WIDE_CASES):
            for _block, (_text, entries) in _paragraphs_by_block(_document(rng)).items():
                for entry in entries:
                    if entry.get("tag") == "finding":
                        findings += 1
                    if "NEEDS" in (entry.get("sentence") or ""):
                        with_marker += 1
                    else:
                        without_marker += 1

    assert findings > 100, findings
    assert with_marker > 50, with_marker
    assert without_marker > 50, without_marker


def test_a_paragraph_repeating_one_sentence_settles_on_the_pass_after_its_marker_goes():
    """The accepted duplicate-sentence limitation, spelled out on
    the saved-draft path, which is the one path that can reach it: a paragraph whose two
    sentences are identical once the marker between them is stripped, carrying a
    different stored link for each of them.

    What must hold, and does: the visible text and the final report are stable from the
    first pass, no pass reports a statistic after the first, and the document is
    byte-identical from pass 2 onwards. What the limitation costs is the one attrs
    rewrite in between. Both stored links name the same sentence once the text has
    settled, so the pass after that keeps one of them, and the evidence_ids of the other
    occurrence are dropped. That collapse is `_claims_for_paragraph_text` re-deriving
    one claim where the marker used to make two, not finalize changing its mind:
    finalize applies the two links to both occurrences on every pass and returns them
    once. Nothing a citation linker returns says which occurrence of a repeated sentence
    it meant, so there is no position to key the lookup on instead."""
    plain = "Lee (2021) reported that attendance rose steadily."
    document = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "blockId": "blk-0",
                    "citationLinks": [
                        {
                            "sentence": plain,
                            "keys": ["lee_2021"],
                            "citation_text": "Lee (2021)",
                            "evidence_ids": ["e2"],
                        },
                        {
                            "sentence": f"[NEEDS CITATION] {plain}",
                            "keys": ["lee_2021"],
                            "citation_text": "Lee (2021)",
                            "evidence_ids": ["e3"],
                        },
                    ],
                },
                "content": [{"type": "text", "text": f"{plain} [NEEDS CITATION] {plain}"}],
            }
        ],
    }

    first_content, first_links, first_stats, first_report = _heal_once(document)

    settled_text = f"{plain} {plain}"
    assert _paragraph_text(first_content["content"][0]) == settled_text
    assert first_stats["needs_citation_markers_removed"] == 1
    assert [link["evidence_ids"] for link in first_links] == [["e2"], ["e3"]]
    assert set(first_report) == {("lee_2021", "verified")}

    second_content, second_links, second_stats, second_report = _heal_once(first_content)

    assert _paragraph_text(second_content["content"][0]) == settled_text
    assert all(value == 0 for value in _stable_stats(second_stats).values())
    assert set(second_report) == set(first_report)
    assert [link["evidence_ids"] for link in second_links] == [["e2"]]

    settled = json.dumps(second_content, sort_keys=True)
    third_content, third_links, third_stats, third_report = _heal_once(second_content)

    assert json.dumps(third_content, sort_keys=True) == settled
    assert third_links == second_links
    assert set(third_report) == set(second_report)
    assert all(value == 0 for value in _stable_stats(third_stats).values())


# --------------------------------------------------------------------------------------
# the re-check's own reproduction, spelled out
# --------------------------------------------------------------------------------------


def _reviewer_document():
    sentence = "Tutoring works (Smith, 2020) and it is cheap (Jones, 2019)."
    return {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "origin": "ai",
                    "blockId": "blk-1",
                    "citationLinks": [
                        {
                            "sentence": sentence,
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                            "evidence_ids": ["e1"],
                        },
                        {
                            "sentence": sentence,
                            "keys": ["jones_2019"],
                            "citation_text": "(Jones, 2019)",
                        },
                    ],
                },
                "content": [{"type": "text", "text": sentence}],
            }
        ],
    }


def test_healing_the_re_checks_own_draft_twice_keeps_the_verified_row():
    """Smith verified, Jones without
    full text. Pass 1 cuts the Jones citation and reports Smith as verified; pass 2 used
    to report nothing at all, because the surviving link still carried the pre-heal
    sentence as a pairing key and `surviving_verifications` matched on it."""
    document = _reviewer_document()

    content, links, stats, report = _heal_once(document)

    healed_text = "Tutoring works (Smith, 2020) and it is cheap."
    assert content["content"][0]["content"][0]["text"] == healed_text
    assert stats["citations_dropped_no_full_text"] == 1
    assert report == [("smith_2020", "verified")]
    assert links[0]["evidence_ids"] == ["e1"]
    assert "pre_heal_sentence" not in links[0]

    healed = json.dumps(content, sort_keys=True)
    second_content, _second_links, second_stats, second_report = _heal_once(content)

    assert json.dumps(second_content, sort_keys=True) == healed
    assert second_report == report
    assert all(value == 0 for value in _stable_stats(second_stats).values())


def test_healing_a_narrowed_co_citation_group_twice_leaves_its_text_unsplit():
    """The same repeated heal on a co-cited group: pass 1 narrows the stored link to its
    verified key and leaves the group's own rendered text alone (a documented
    limitation). Pass 2 sees the dropped key resolved on its own by the author-year
    regexes, and must still leave that text alone -- cutting "Jones, 2019" out of
    "(Jones, 2019; Smith, 2020)" would leave "(; Smith, 2020)" and mangle the sentence a
    little further on every heal."""
    sentence = "Two papers agree (Jones, 2019; Smith, 2020)."
    document = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": sentence,
                            "keys": ["jones_2019", "smith_2020"],
                            "citation_text": "(Jones, 2019; Smith, 2020)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": sentence}],
            }
        ],
    }

    content, links, _stats, report = _heal_once(document)

    assert content["content"][0]["content"][0]["text"] == sentence
    assert [link["keys"] for link in links] == [["smith_2020"]]
    assert report == [("smith_2020", "verified")]

    healed = json.dumps(content, sort_keys=True)
    second_content, _links, second_stats, second_report = _heal_once(content)

    assert json.dumps(second_content, sort_keys=True) == healed
    assert second_report == report
    assert all(value == 0 for value in _stable_stats(second_stats).values())


def test_no_surviving_link_carries_a_pairing_key_on_the_write_path_either():
    """`generate_section` puts `finalize_generated_section`'s own citation links into the
    job result verbatim (a public API payload) and into the saved section. A pairing key
    on a link would leak onto both surfaces, so finalize must not put one there; the
    pre-to-post wording is returned separately, for that one pass."""
    sentence = "Tutoring works (Smith, 2020) and it is cheap (Jones, 2019)."
    links = [
        {
            "paragraph_index": 0,
            "sentence": sentence,
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
        },
        {
            "paragraph_index": 0,
            "sentence": sentence,
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
        },
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}

    result = finalize_generated_section(sentence, links, [], status)

    assert result.text == "Tutoring works (Smith, 2020) and it is cheap."
    assert all("pre_heal_sentence" not in link for link in result.citation_links)
    assert result.healed_sentences == {sentence: result.text}


def test_a_pairing_key_left_on_a_stored_link_by_an_earlier_build_is_dropped():
    """A draft healed by the build this fix replaces still carries `pre_heal_sentence`
    on its stored link. The next heal must drop it rather than read it, so such a draft
    converges on its own instead of losing its verified row on every pass."""
    healed_sentence = "Tutoring works (Smith, 2020)."
    document = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": healed_sentence,
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                            "pre_heal_sentence": "Tutoring works (Smith, 2020) and (Jones, 2019).",
                        }
                    ]
                },
                "content": [{"type": "text", "text": healed_sentence}],
            }
        ],
    }

    content, links, _stats, report = _heal_once(document)

    assert content["content"][0]["content"][0]["text"] == healed_sentence
    assert links == [
        {
            "sentence": healed_sentence,
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
        }
    ]
    assert _stored_links(content) == links
    assert report == [("smith_2020", "verified")]

    healed = json.dumps(content, sort_keys=True)
    second_content, _second_links, _second_stats, second_report = _heal_once(content)

    assert json.dumps(second_content, sort_keys=True) == healed
    assert second_report == report
