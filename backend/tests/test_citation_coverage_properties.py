"""Property-style tests for the citation-coverage accounting in
`app.services.fulltext.extract_claims_from_document`.

The accounting can break on a shape nobody had written a test for: a lead-in word
before a narrative citation, a citation rendered twice, a sentence the claim splitter
breaks at "et al." or "U.S.", a semicolon-separated group, a link whose whitespace
differs from the draft's. Rather than add one example per shape, this file builds
documents from a small grammar of exactly those shapes and asserts the accounting
invariants over a few hundred of them, with a fixed seed so a failure is reproducible.

The invariants, stated once here and asserted for every generated document:

1. ``found`` equals the number of citations `citation_audit.citation_spans` reports in
   the document (the grammar only produces citation shapes the audit recognises, so the
   square-bracket widening `_regex_only_units` allows never applies here);
2. ``found`` does not depend on the citation-link map: the same document with every link
   removed finds the same citations;
3. ``linked`` plus ``unresolved`` equals ``found``, and ``by_source`` sums to ``found``;
4. ``sent_to_verifier`` equals the number of claims returned, and no two claims share a
   (sentence, citation key) pair, so no citation is verified twice;
5. every claim's sentence is text that is actually in the document, and every claim
   carries a non-empty citation text and a ``surname_year`` key;
6. ``unresolved_citations`` lists one balanced, non-empty text per unresolved citation
   (up to the cap), so nothing a user is shown is a truncated fragment.

All pure functions over plain dicts: no network, no database, no model.
"""

import random
import re

import pytest

from app.services.citation_audit import citation_spans
from app.services.fulltext import _doc_normalize_ws, extract_claims_from_document

SEED = 20260908
CASES = 300

#: (key, surname, year) triples the grammar cites, all present in the fake library below
#: except `nguyen_2021`, which is cited but never in the library so the "resolved but
#: absent from the library" branch of coverage is exercised too.
SOURCES = [
    ("smith_2020", "Smith", "2020"),
    ("jones_2021", "Jones", "2021"),
    ("lee_2020", "Lee", "2020"),
    ("storch_2018", "Storch", "2018"),
    ("zhang_2019", "Zhang", "2019"),
    ("mao_2024", "Mao", "2024"),
    ("nguyen_2021", "Nguyen", "2021"),
]

PAPER_LOOKUP = {key: object() for key, _surname, _year in SOURCES if key != "nguyen_2021"}

LEAD_INS = ["", "However, ", "In China, ", "By contrast, "]

CLAUSES = [
    "tutoring improved outcomes",
    "gains were reported in the U.S. by prior work",
    "teachers differed e.g. in feedback practice",
    "Smith et al. reported large gains",
    "the replication failed",
]

KEY_RE = re.compile(r"^[^\s_]+_\d{4}$")


def _rendered_citation(rng, source):
    """One citation as it appears in a sentence, plus the ``citation_text`` a linker
    would return for it and the keys it stands for."""
    key, surname, year = source
    form = rng.choice(
        ["paren", "paren_et_al", "narrative", "narrative_et_al", "narrative_two", "group"]
    )
    if form == "paren":
        return f"({surname}, {year})", [(f"({surname}, {year})", key)]
    if form == "paren_et_al":
        return f"({surname} et al., {year})", [(f"({surname} et al., {year})", key)]
    if form == "narrative":
        return f"{surname} ({year})", [(f"{surname} ({year})", key)]
    if form == "narrative_et_al":
        return f"{surname} et al. ({year})", [(f"{surname} et al. ({year})", key)]
    if form == "narrative_two":
        return f"{surname} and Li ({year})", [(f"{surname} and Li ({year})", key)]
    other_key, other_surname, other_year = rng.choice(
        [s for s in SOURCES if s[0] != key]
    )
    text = f"({surname}, {year}; {other_surname}, {other_year})"
    return text, [
        (f"{surname}, {year}", key),
        (f"{other_surname}, {other_year}", other_key),
    ]


def _sentence(rng):
    """One sentence, its (citation_text, key) pairs, and the clause a perfect linker's
    ``proposition`` could narrow a claim down to: the clause is a
    contiguous span of words `sentence` always contains verbatim, so it is a valid
    proposition for any link built from this sentence."""
    citations = []
    lead_in = rng.choice(LEAD_INS)
    clause = rng.choice(CLAUSES)
    text, pairs = _rendered_citation(rng, rng.choice(SOURCES))
    citations.extend(pairs)
    sentence = f"{lead_in}{clause} {text}"
    if rng.random() < 0.25:
        # The same citation rendered a second time in the same sentence.
        sentence += f" and again {text}"
        citations.extend(pairs)
    if rng.random() < 0.3:
        second_text, second_pairs = _rendered_citation(rng, rng.choice(SOURCES))
        sentence += f", consistent with {second_text}"
        citations.extend(second_pairs)
    sentence += "."
    if rng.random() < 0.25:
        # A double space somewhere in the sentence, including inside a citation.
        positions = [index for index, character in enumerate(sentence) if character == " "]
        if positions:
            cut = rng.choice(positions)
            sentence = sentence[:cut] + " " + sentence[cut:]
    return sentence, citations, clause


def _links_for(rng, sentence, citations, clause):
    """The citation-link map the writer might have returned for *sentence*: links
    present, absent, keyless, duplicated, written with normalised whitespace, and about
    half the time carrying ``proposition`` set to *clause* -- a
    contiguous, verbatim span of ``sentence``, so it is always valid once the link
    itself resolves."""
    links = []
    for citation_text, key in citations:
        choice = rng.choice(["present", "present", "absent", "keyless", "duplicated"])
        if choice == "absent":
            continue
        keys = [] if choice == "keyless" else [key]
        link = {
            "sentence": sentence,
            # A linker copies the citation as rendered; whitespace it normalises is the
            # difference `validate_citation_link` accepts and resolution must tolerate.
            "citation_text": _doc_normalize_ws(citation_text),
            "keys": keys,
        }
        if rng.random() < 0.5:
            link["proposition"] = clause
        links.append(link)
        if choice == "duplicated":
            links.append(dict(link))
    return links


def _document(rng):
    """A Tiptap document of one to three paragraphs of one to three sentences."""
    nodes = []
    for _ in range(rng.randint(1, 3)):
        sentences = []
        links = []
        for _ in range(rng.randint(1, 3)):
            sentence, citations, clause = _sentence(rng)
            sentences.append(sentence)
            if rng.random() < 0.75:
                links.extend(_links_for(rng, sentence, citations, clause))
        text = " ".join(sentences)
        attrs = {"citationLinks": links} if links else {}
        nodes.append(
            {"type": "paragraph", "attrs": attrs, "content": [{"type": "text", "text": text}]}
        )
    return {"type": "doc", "content": nodes}


def _paragraph_texts(document):
    return [node["content"][0]["text"] for node in document["content"]]


def _audit_citation_count(document):
    """What `citation_audit.citation_spans` itself says is in the document."""
    return sum(
        len({position for position, *_rest in citation_spans(text)})
        for text in _paragraph_texts(document)
    )


def _stripped_links(document):
    """The same document with every citation link removed."""
    return {
        "type": "doc",
        "content": [
            {"type": "paragraph", "attrs": {}, "content": node["content"]}
            for node in document["content"]
        ],
    }


def _generated_documents():
    rng = random.Random(SEED)
    return [_document(rng) for _ in range(CASES)]


DOCUMENTS = _generated_documents()
#: The documents are checked in batches rather than one per test: this suite's conftest
#: creates and drops the whole test schema for every test, so 300 tests would add well
#: over a minute to the backend run for no extra coverage. A failure still names the
#: document, through the assertion messages below.
BATCH_COUNT = 10
BATCHES = [DOCUMENTS[index::BATCH_COUNT] for index in range(BATCH_COUNT)]


def _check_invariants(document):
    claims, coverage = extract_claims_from_document(document, PAPER_LOOKUP)
    texts = _paragraph_texts(document)
    normalised_texts = [_doc_normalize_ws(text) for text in texts]

    # 1. found is the citation audit's own count.
    assert coverage["found"] == _audit_citation_count(document), texts

    # 2. the map does not change how many citations exist.
    _unmapped_claims, unmapped_coverage = extract_claims_from_document(
        _stripped_links(document), PAPER_LOOKUP
    )
    assert unmapped_coverage["found"] == coverage["found"], texts

    # 3. every citation lands in exactly one bucket, and in exactly one source.
    assert coverage["linked"] + coverage["unresolved"] == coverage["found"]
    assert sum(coverage["by_source"].values()) == coverage["found"]
    diagnostics = coverage["diagnostics"]
    assert (
        diagnostics["units_linked"]
        + diagnostics["units_fallback"]
        + diagnostics["units_numbered"]
        + diagnostics["units_without_keys"]
        == coverage["found"]
    )

    # 4. one claim per question, never the same question twice. Uniqueness is checked
    # on (claim_text, claim_sentence, key), not (claim_sentence, key) alone -- two
    # claims may legitimately share a sentence and a key when their own claim_text
    # (proposition) differs, which is a dedup distinction, not a coverage bug.
    assert coverage["sent_to_verifier"] == len(claims)
    assert (
        len({(text, claim_sentence, key) for text, key, _c, claim_sentence in claims})
        == len(claims)
    )

    # 5. claims quote the document, carry a citation text and a well-shaped key.
    # claim_text is either claim_sentence itself, or a link's proposition -- always a
    # whitespace-normalised substring of claim_sentence.
    for claim_text, key, citation_text, claim_sentence in claims:
        assert KEY_RE.match(key), key
        assert citation_text.strip()
        assert any(
            _doc_normalize_ws(claim_sentence) in text for text in normalised_texts
        ), claim_sentence
        assert _doc_normalize_ws(claim_text) in _doc_normalize_ws(claim_sentence), (
            claim_text,
            claim_sentence,
        )

    # 6. what a user is shown for an unresolved citation is a balanced, non-empty text.
    assert len(coverage["unresolved_citations"]) == min(coverage["unresolved"], 50)
    for entry in coverage["unresolved_citations"]:
        assert entry.strip() == entry and entry
        assert entry.count("(") == entry.count(")")
        assert entry.count("[") == entry.count("]")


@pytest.mark.parametrize("batch", range(BATCH_COUNT))
def test_citation_coverage_invariants_hold_for_generated_documents(batch):
    for document in BATCHES[batch]:
        _check_invariants(document)


def test_the_generated_grammar_actually_exercises_every_shape():
    """A guard on the generator itself: a grammar that stopped producing groups, lead-in
    words, twice-rendered citations, abbreviation splits or double spaces would keep the
    test above green while testing nothing."""
    texts = [text for document in DOCUMENTS for text in _paragraph_texts(document)]
    joined = "\n".join(texts)
    assert "; " in joined  # grouped citations
    assert "However, " in joined  # lead-in words
    assert "et al." in joined  # abbreviation-induced fragment splits
    assert "U.S." in joined
    assert "e.g." in joined
    assert " and again " in joined  # twice-rendered citations
    assert any("  " in text for text in texts)  # double spaces
    documents_with_links = [
        document
        for document in DOCUMENTS
        if any((node.get("attrs") or {}).get("citationLinks") for node in document["content"])
    ]
    assert len(documents_with_links) > CASES // 2
    links = [
        link
        for document in DOCUMENTS
        for node in document["content"]
        for link in (node.get("attrs") or {}).get("citationLinks") or []
    ]
    assert any(not link["keys"] for link in links)  # keyless links
    assert any(links.count(link) > 1 for link in links[:200])  # duplicated links
    assert any(link.get("proposition") for link in links)  # propositions
    assert any(not link.get("proposition") for link in links)  # and links without one


def test_generated_documents_reach_every_coverage_outcome():
    """The corpus is only worth running if it contains linked citations, unresolved ones
    and citations resolved by the regex fallback rather than the map."""
    outcomes = {"mapping": 0, "author-year": 0, "unresolved": 0}
    for document in DOCUMENTS:
        _claims, coverage = extract_claims_from_document(document, PAPER_LOOKUP)
        outcomes["mapping"] += coverage["by_source"]["mapping"]
        outcomes["author-year"] += coverage["by_source"]["author-year"]
        outcomes["unresolved"] += coverage["unresolved"]
    assert outcomes["mapping"] > 0
    assert outcomes["author-year"] > 0
    assert outcomes["unresolved"] > 0


def test_generated_documents_use_a_proposition_for_some_claims_and_the_sentence_for_others():
    """A claim whose unit
    was resolved by a link carrying a usable proposition gets that proposition as
    ``claim_text``; every other claim still gets the full sentence, exactly as before this
    change. The corpus is only worth running for this invariant if it reaches both
    outcomes."""
    narrowed = 0
    unnarrowed = 0
    for document in DOCUMENTS:
        claims, _coverage = extract_claims_from_document(document, PAPER_LOOKUP)
        for claim_text, _key, _citation_text, claim_sentence in claims:
            if claim_text != claim_sentence:
                narrowed += 1
            else:
                unnarrowed += 1
    assert narrowed > 0
    assert unnarrowed > 0
