"""A citation link's ``proposition`` narrows the claim text sent to the verifier.

``app.services.fulltext.extract_claims_from_document`` returns a
``(claim_text, citation_key, citation_text, claim_sentence)`` quadruple per claim.
``claim_sentence`` is always the full sentence (fragment) the claim was cut from;
``claim_text`` is the link's own ``proposition`` -- a verbatim, contiguous span of that
sentence -- whenever the link that resolved the unit carried a non-empty one that still
occurs in the sentence after whitespace normalisation, and the sentence itself otherwise.

All pure-function tests against plain Tiptap-document dicts; no network, no database.
"""

from __future__ import annotations

from app.services.fulltext import extract_claims_from_document


def _doc(text: str, citation_links: list[dict]) -> dict:
    return {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {"citationLinks": citation_links},
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


def test_a_valid_proposition_becomes_claim_text_while_claim_sentence_stays_the_sentence():
    sentence = "Tutoring improves outcomes for most students (Smith, 2020)."
    proposition = "Tutoring improves outcomes for most students"
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["smith_2020"],
                "citation_text": "(Smith, 2020)",
                "proposition": proposition,
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"smith_2020": object()})
    assert claims == [(proposition, "smith_2020", "(Smith, 2020)", sentence)]


def test_a_link_with_no_proposition_key_still_sends_the_sentence_as_before():
    sentence = "Tutoring improves outcomes (Jones, 2021)."
    doc = _doc(
        sentence,
        [{"sentence": sentence, "keys": ["jones_2021"], "citation_text": "(Jones, 2021)"}],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]


def test_an_empty_proposition_falls_back_to_the_sentence():
    sentence = "Tutoring improves outcomes (Jones, 2021)."
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["jones_2021"],
                "citation_text": "(Jones, 2021)",
                "proposition": "",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]


def test_a_whitespace_only_proposition_falls_back_to_the_sentence():
    sentence = "Tutoring improves outcomes (Jones, 2021)."
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["jones_2021"],
                "citation_text": "(Jones, 2021)",
                "proposition": "   ",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]


def test_a_proposition_absent_from_the_paragraph_falls_back_to_the_sentence():
    """A proposition the paragraph does not render at all -- the model hallucinated it, or
    the user has since edited the paragraph -- is exactly as stale as a link's ``sentence``
    the paragraph no longer contains, and is discarded the same way."""
    sentence = "Tutoring improves outcomes (Jones, 2021)."
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["jones_2021"],
                "citation_text": "(Jones, 2021)",
                "proposition": "this text never appears anywhere in the draft",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]


def test_a_proposition_valid_at_paragraph_level_but_outside_its_own_fragment_falls_back():
    """`_assign_links_to_units` only has the whole paragraph to check a proposition
    against; a unit's own ``sentence`` (a claim-splitter fragment) is not known until
    later. An abbreviation such as "e.g." splits one link's own sentence into two
    fragments, so a proposition spanning that split is a real paragraph substring that is
    not a substring of the one fragment the citation's own unit lands in -- it must still
    be rejected, so ``claim_text`` is always contained in ``claim_sentence`` whenever a
    proposition is used."""
    sentence = "In China, teachers differed e.g. in feedback practice (Mao, 2024)."
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["mao_2024"],
                "citation_text": "(Mao, 2024)",
                "proposition": "teachers differed e.g. in feedback practice",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"mao_2024": object()})
    assert len(claims) == 1
    claim_text, key, citation_text, claim_sentence = claims[0]
    assert key == "mao_2024"
    assert citation_text == "(Mao, 2024)"
    # The proposition spans the fragment split, so it is rejected: the sentence fragment
    # is sent instead, exactly as if no proposition had been given at all.
    assert claim_text == claim_sentence
    assert claim_text == "in feedback practice (Mao, 2024)."


def test_a_group_wide_links_proposition_is_shared_by_every_member_it_resolves():
    """A link naming a whole semicolon-separated group resolves one unit per member;
    its proposition, when valid, is given to
    every member the same way its keys are -- one link, one proposition, however many
    citations it covers."""
    sentence = "Evidence is mixed (Lee, 2020; Storch, 2018) in the field."
    proposition = "Evidence is mixed"
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["lee_2020", "storch_2018"],
                "citation_text": "(Lee, 2020; Storch, 2018)",
                "proposition": proposition,
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(
        doc, {"lee_2020": object(), "storch_2018": object()}
    )
    assert set(claims) == {
        (proposition, "lee_2020", "Lee, 2020", sentence),
        (proposition, "storch_2018", "Storch, 2018", sentence),
    }


def test_proposition_whitespace_is_normalised_for_the_occurrence_check_only():
    """The occurrence check tolerates a whitespace difference between the proposition and
    the paragraph's own rendering (mirrors how a link's ``sentence`` is matched), but the
    text actually used as ``claim_text`` is the proposition exactly as given, stripped of
    its own leading/trailing whitespace -- not a substring reconstructed from the
    paragraph."""
    sentence = "Tutoring  improves outcomes (Jones, 2021)."  # double space, as rendered
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["jones_2021"],
                "citation_text": "(Jones, 2021)",
                "proposition": "  Tutoring improves outcomes  ",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [("Tutoring improves outcomes", "jones_2021", "(Jones, 2021)", sentence)]


def test_two_links_naming_the_same_rendered_citation_with_different_propositions_both_yield_a_claim():
    """A sentence the citation-link step splits into two propositions against
    the same paper, rendered as ONE citation, sends two link entries whose own
    ``citation_text`` names the identical occurrence. Without a document-level unit
    model, the second link would match no unconsumed unit and be silently dropped
    (`links_ignored`) before the verifier ever saw it. Both propositions must
    reach the verifier as two separate claims, exactly as the writing loop's own
    ``claims_from_citation_links`` already does for the identical shape."""
    sentence = (
        "A comparison between the second draft and the first draft of Flora's essay "
        "showed that she responded to teacher feedback by carrying out a variety of "
        "revision operations, and 88% submitted to the AWE system more than five times "
        "(Zhang, 2018)."
    )
    proposition_one = (
        "she responded to teacher feedback by carrying out a variety of revision "
        "operations"
    )
    proposition_two = "88% submitted to the AWE system more than five times"
    doc = _doc(
        sentence,
        [
            {
                "sentence": sentence,
                "keys": ["zhang_2018"],
                "citation_text": "(Zhang, 2018)",
                "proposition": proposition_one,
            },
            {
                "sentence": sentence,
                "keys": ["zhang_2018"],
                "citation_text": "(Zhang, 2018)",
                "proposition": proposition_two,
            },
        ],
    )
    claims, coverage = extract_claims_from_document(doc, {"zhang_2018": object()})
    assert set(claims) == {
        (proposition_one, "zhang_2018", "(Zhang, 2018)", sentence),
        (proposition_two, "zhang_2018", "(Zhang, 2018)", sentence),
    }
    # The unit itself -- the one rendered citation -- still counts once, not twice.
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["sent_to_verifier"] == 2


def test_a_second_links_own_sentence_occurrence_is_preferred_over_another_sentences():
    """A real shape: "(Hyland, 2025)" rendered twice, in
    two different sentences. Three links name it: two for the first sentence (a
    fragment, then the whole clause), one for the second. The first link consumes the
    first sentence's own occurrence. The second link, whose own sentence is that same
    first sentence, must resolve to that SAME occurrence as an extra resolution --
    never fall through to the second sentence's occurrence and claim it as a primary
    resolution, which would let the second sentence's own duplicate claim through
    while the first sentence's whole-clause proposition gets no verdict at all and its
    sentence is wrongly deleted."""
    sentence_one = (
        "Teacher feedback has been found to address more error types (16) than AWE "
        "feedback (8), with many mechanical errors left undiagnosed (Hyland, 2025)."
    )
    sentence_two = (
        "A concern is that ChatGPT's more retiring stance may prevent it from "
        "offering accurate feedback on rhetorical and pragmatic aspects of "
        "argumentation (Hyland, 2025)."
    )
    fragment = "with many mechanical errors left undiagnosed"
    whole_clause = (
        "Teacher feedback has been found to address more error types (16) than AWE "
        "feedback (8), with many mechanical errors left undiagnosed"
    )
    proposition_two = (
        "ChatGPT's more retiring stance may prevent it from offering accurate "
        "feedback on rhetorical and pragmatic aspects of argumentation"
    )
    paragraph = f"{sentence_one} {sentence_two}"
    doc = _doc(
        paragraph,
        [
            {
                "sentence": sentence_one,
                "keys": ["hyland_2025"],
                "citation_text": "(Hyland, 2025)",
                "proposition": fragment,
            },
            {
                "sentence": sentence_one,
                "keys": ["hyland_2025"],
                "citation_text": "(Hyland, 2025)",
                "proposition": whole_clause,
            },
            {
                "sentence": sentence_two,
                "keys": ["hyland_2025"],
                "citation_text": "(Hyland, 2025)",
                "proposition": proposition_two,
            },
        ],
    )
    claims, coverage = extract_claims_from_document(doc, {"hyland_2025": object()})
    # Two rendered occurrences of "(Hyland, 2025)", one per sentence: both resolve, and
    # the first one carries both of the first sentence's propositions.
    assert coverage["found"] == 2
    assert coverage["sent_to_verifier"] == 3
    assert set(claims) == {
        (fragment, "hyland_2025", "(Hyland, 2025)", sentence_one),
        (whole_clause, "hyland_2025", "(Hyland, 2025)", sentence_one),
        (proposition_two, "hyland_2025", "(Hyland, 2025)", sentence_two),
    }
    # No claim is a whole other sentence: the second link's own proposition must never
    # be nulled by landing on the wrong sentence's unit.
    assert sentence_two not in {claim_text for claim_text, *_ in claims}


def test_a_proposition_on_a_link_the_regex_fallback_resolves_instead_is_never_used():
    """A ``proposition`` only ever narrows a unit `_assign_links_to_units` itself
    consumed. A link whose sentence the paragraph no longer contains is dropped before
    reaching that function at all, so its citation is resolved by
    the pre-existing author-year regex instead, and gets the sentence, never a stale
    link's proposition."""
    sentence = "The user rewrote this sentence entirely (Jones, 2021)."
    doc = _doc(
        sentence,
        [
            {
                "sentence": "Tutoring improves outcomes (Smith, 2020).",
                "keys": ["smith_2020"],
                "citation_text": "(Smith, 2020)",
                "proposition": "Tutoring improves outcomes",
            }
        ],
    )
    claims, _coverage = extract_claims_from_document(doc, {"jones_2021": object()})
    assert claims == [(sentence, "jones_2021", "(Jones, 2021)", sentence)]
