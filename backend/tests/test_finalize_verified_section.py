"""Finalize: remove every cited sentence whose final
status is not verified, remove every uncited sentence tagged "finding", drop a
citation of a paper without full text (and the sentence too when no verified citation
remains on it), and strip every ``[NEEDS CITATION]`` marker.
"""

from app.services.fulltext import (
    finalize_draft_document,
    finalize_generated_section,
    surviving_verifications,
)


def _link(sentence, keys, citation_text="(X, 2020)", paragraph_index=0, **extra):
    return {
        "paragraph_index": paragraph_index,
        "sentence": sentence,
        "keys": keys,
        "citation_text": citation_text,
        **extra,
    }


# --------------------------------------------------------------------------------------
# finalize_generated_section: rule 2, unverified sentences.
# --------------------------------------------------------------------------------------


def test_removes_a_sentence_whose_citation_is_unsupported():
    text = "Tutoring works well (Smith, 2020)."
    links = [_link(text, ["smith_2020"])]
    status = {(text, text, "smith_2020"): "unsupported"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == ""
    assert result.citation_links == []
    assert result.stats["sentences_removed_unverified"] == 1


def test_removes_a_sentence_whose_citation_needs_nuance():
    text = "Tutoring works well (Smith, 2020)."
    links = [_link(text, ["smith_2020"])]
    status = {(text, text, "smith_2020"): "needs_nuance"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == ""
    assert result.stats["sentences_removed_unverified"] == 1


def test_keeps_a_verified_sentence_unchanged():
    text = "Tutoring works well (Smith, 2020)."
    links = [_link(text, ["smith_2020"])]
    status = {(text, text, "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == text
    assert result.citation_links == [{**links[0], "paragraph_index": 0}]
    assert all(v == 0 for v in result.stats.values())


def test_a_sentence_with_no_verdict_at_all_is_removed_defensively():
    """A (sentence, key) pair missing from claim_status is treated exactly like an
    unsupported claim, never like a silently-passing one."""
    text = "Tutoring works well (Smith, 2020)."
    links = [_link(text, ["smith_2020"])]

    result = finalize_generated_section(text, links, [], {})

    assert result.text == ""
    assert result.stats["sentences_removed_unverified"] == 1


# --------------------------------------------------------------------------------------
# rule 1, uncited sentences.
# --------------------------------------------------------------------------------------


def test_removes_an_uncited_sentence_tagged_finding():
    text = "Tutoring is the single most effective intervention available."
    uncited = [{"paragraph_index": 0, "sentence": text, "tag": "finding"}]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == ""
    assert result.stats["sentences_removed_uncited_finding"] == 1


def test_keeps_an_uncited_sentence_tagged_framing():
    text = "This section reviews the literature on tutoring."
    uncited = [{"paragraph_index": 0, "sentence": text, "tag": "framing"}]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == text
    assert result.stats["sentences_removed_uncited_finding"] == 0


def test_an_uncited_sentence_absent_from_the_list_is_kept():
    text = "This is a plain sentence with no citation and no tag at all."

    result = finalize_generated_section(text, [], [], {})

    assert result.text == text


# --------------------------------------------------------------------------------------
# rule 3, no_full_text citations.
# --------------------------------------------------------------------------------------


def test_drops_a_no_full_text_citation_but_keeps_the_sentence_when_another_survives():
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    links = [
        _link(sentence, ["smith_2020"], citation_text="(Smith, 2020)"),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}

    result = finalize_generated_section(sentence, links, [], status)

    assert "(Jones, 2019)" not in result.text
    assert "(Smith, 2020)" in result.text
    assert result.stats["citations_dropped_no_full_text"] == 1
    assert result.stats["sentences_removed_no_full_text"] == 0
    assert [link["keys"] for link in result.citation_links] == [["smith_2020"]]


def test_a_no_full_text_drop_refreshes_the_surviving_links_own_sentence():
    """Once rule 3 rewrites the sentence, the
    surviving link's own `sentence` must name the rewritten text -- not the pre-heal
    wording that still included the dropped citation -- so a later pass over this
    paragraph still finds it (`extract_claims_from_document`'s own staleness check).

    The pre-heal wording is reported in `healed_sentences`, NOT stored on the link: a
    pairing key on the link is written into the document and into the job result with
    it, where the next heal reads it as if it described that pass. The map belongs to
    this pass alone.
    """
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    links = [
        _link(sentence, ["smith_2020"], citation_text="(Smith, 2020)"),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}

    result = finalize_generated_section(sentence, links, [], status)

    assert result.citation_links == [
        {
            "paragraph_index": 0,
            "sentence": result.text,
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
        }
    ]
    assert result.text != sentence
    assert result.healed_sentences == {sentence: result.text}


def test_a_no_full_text_drop_discards_a_proposition_the_rewritten_sentence_no_longer_holds():
    """The same refresh drops `proposition` when it no longer occurs in the rewritten
    sentence, rather than carrying forward a claim clause the new text cannot support.
    The proposition's own trailing "and" is what the rewrite's dangling-conjunction
    cleanup (`_strip_citation_substring`) removes; the proposition covers the whole
    surviving clause up to and including it (the coverage
    check requires the kept proposition to actually cover the sentence, so a bare
    "and" alone -- a realistic connector, not a claim -- would not reach this
    mechanic at all)."""
    sentence = "Findings replicate (Smith, 2020) and (Jones, 2019)."
    links = [
        _link(
            sentence, ["smith_2020"], citation_text="(Smith, 2020)",
            proposition="Findings replicate (Smith, 2020) and",
        ),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {
        (sentence, "Findings replicate (Smith, 2020) and", "smith_2020"): "verified",
        (sentence, sentence, "jones_2019"): "no_full_text",
    }

    result = finalize_generated_section(sentence, links, [], status)

    assert "and" not in result.text
    assert "proposition" not in result.citation_links[0]
    assert result.citation_links[0]["sentence"] == result.text


def test_a_no_full_text_drop_keeps_a_proposition_still_held_by_the_rewritten_sentence():
    """A proposition confined to prose the rewrite never touches survives the refresh,
    still narrowing the claim the same way it did before the heal."""
    sentence = "Costs fell sharply, researchers found (Smith, 2020) and (Jones, 2019)."
    links = [
        _link(
            sentence, ["smith_2020"], citation_text="(Smith, 2020)",
            proposition="Costs fell sharply",
        ),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {
        (sentence, "Costs fell sharply", "smith_2020"): "verified",
        (sentence, sentence, "jones_2019"): "no_full_text",
    }

    result = finalize_generated_section(sentence, links, [], status)

    assert result.citation_links[0]["proposition"] == "Costs fell sharply"
    assert result.citation_links[0]["sentence"] == result.text


def test_removes_the_sentence_when_its_only_citation_is_no_full_text():
    text = "This method was shown to work (Smith, 2020)."
    links = [_link(text, ["smith_2020"])]
    status = {(text, text, "smith_2020"): "no_full_text"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == ""
    assert result.stats["sentences_removed_no_full_text"] == 1
    assert result.stats["sentences_removed_unverified"] == 0


def test_mixed_co_citation_group_narrows_to_only_its_verified_keys():
    sentence = "Two papers agree (Jones, 2019; Lee, 2021)."
    links = [_link(sentence, ["jones_2019", "lee_2021"], citation_text="(Jones, 2019; Lee, 2021)")]
    status = {(sentence, sentence, "jones_2019"): "verified", (sentence, sentence, "lee_2021"): "no_full_text"}

    result = finalize_generated_section(sentence, links, [], status)

    assert result.text == sentence
    assert result.citation_links[0]["keys"] == ["jones_2019"]


# --------------------------------------------------------------------------------------
# [NEEDS CITATION] stripping and heading protection.
# --------------------------------------------------------------------------------------


def test_strips_the_needs_citation_marker():
    text = "This claim has no source. [NEEDS CITATION] It still reads fine."

    result = finalize_generated_section(text, [], [], {})

    assert "[NEEDS CITATION]" not in result.text
    assert result.stats["needs_citation_markers_removed"] == 1


def test_strips_a_needs_citation_marker_whose_two_words_are_split_by_a_doubled_space():
    """The writer prompt only tells the model to type the
    literal text "[NEEDS CITATION]" (writing.py's own base rules), so a doubled space
    typed inside it is an ordinary generation artefact, not a shape only a mark
    produces. It must be removed on the same pass, with the count reported as one, not
    left verbatim with the count reported as zero because the marker regex tolerates
    whitespace only before the marker, not inside it."""
    text = "Costs may fall [NEEDS  CITATION]. Tutoring works well (Smith, 2020)."
    links = [_link("Tutoring works well (Smith, 2020).", ["smith_2020"])]
    status = {("Tutoring works well (Smith, 2020).", "Tutoring works well (Smith, 2020).", "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert "[NEEDS" not in result.text
    assert "CITATION]" not in result.text
    assert result.stats["needs_citation_markers_removed"] == 1


# --------------------------------------------------------------------------------------
# One normalised sentence form for every rule. The fragment is
# marker-stripped, whitespace-collapsed and trimmed ONCE, before rule 1, rule 2 and
# rule 3 are checked, and every key each of those rules compares it against -- the
# per-sentence link map, the uncited-sentence set and the status lookup -- is built
# with that same normaliser, so no rule can see a marker or a whitespace difference
# the others do not.
# --------------------------------------------------------------------------------------


def test_removes_an_uncited_finding_recorded_with_its_own_marker():
    """The citation-link prompt tells the model to copy an
    uncited sentence exactly as it appears in the text, so a sentence the writer
    flagged with "[NEEDS CITATION]" comes back as an uncited "finding" entry with that
    marker inside it. The entry must still match the fragment, which is marker-stripped
    before rule 1 is checked, or the unsupported statement is shipped with the one
    visible sign that it was unsupported taken off it."""
    sentence = "Teachers reported gains in general [NEEDS CITATION]."
    text = f"Tutoring is effective (Smith, 2020). {sentence}"
    uncited = [{"paragraph_index": 0, "sentence": sentence, "tag": "finding"}]
    links = [
        _link(
            "Tutoring is effective (Smith, 2020).",
            ["smith_2020"],
            citation_text="(Smith, 2020)",
        )
    ]
    status = {("Tutoring is effective (Smith, 2020).", "Tutoring is effective (Smith, 2020).", "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, uncited, status)

    assert result.text == "Tutoring is effective (Smith, 2020)."
    assert result.stats["sentences_removed_uncited_finding"] == 1
    assert result.stats["needs_citation_markers_removed"] == 0


def test_removes_an_uncited_finding_whose_recorded_marker_has_a_doubled_space():
    """The same entry with the doubled space a formatting mark, or the model own
    typing, leaves inside the marker: recorded singly spaced here, against a fragment
    that carries the doubled space."""
    text = "Teachers reported gains in general [NEEDS  CITATION]."
    uncited = [
        {
            "paragraph_index": 0,
            "sentence": "Teachers reported gains in general [NEEDS CITATION].",
            "tag": "finding",
        }
    ]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == ""
    assert result.stats["sentences_removed_uncited_finding"] == 1


def test_removes_an_uncited_finding_recorded_with_a_doubled_space_marker():
    """The mirror of the test above: the doubled space is in the recorded entry and the
    text is singly spaced. Both sides go through the one normaliser, so neither
    direction depends on which side happens to carry the extra space."""
    text = "Teachers reported gains in general [NEEDS CITATION]."
    uncited = [
        {
            "paragraph_index": 0,
            "sentence": "Teachers reported gains in general [NEEDS  CITATION].",
            "tag": "finding",
        }
    ]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == ""
    assert result.stats["sentences_removed_uncited_finding"] == 1


def test_removes_an_uncited_finding_recorded_without_the_marker_its_fragment_carries():
    """On the write path, the splitter puts a marker written
    at the end of one sentence at the HEAD of the fragment for the sentence after it,
    so an entry recorded without the marker must match too. Both shapes are the same
    single normalisation."""
    text = (
        "No effect was observed in general. [NEEDS CITATION] "
        "Teachers reported gains in general."
    )
    uncited = [
        {
            "paragraph_index": 0,
            "sentence": "Teachers reported gains in general.",
            "tag": "finding",
        }
    ]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == "No effect was observed in general."
    assert result.stats["sentences_removed_uncited_finding"] == 1


def test_keeps_an_uncited_framing_sentence_recorded_with_its_own_marker():
    """The normalisation must not widen rule 1 beyond the "finding" tag: a "framing"
    entry is kept, and its marker is still stripped, exactly as design amendment A4
    asks."""
    sentence = "This section reviews the literature [NEEDS CITATION]."
    uncited = [{"paragraph_index": 0, "sentence": sentence, "tag": "framing"}]

    result = finalize_generated_section(sentence, [], uncited, {})

    assert result.text == "This section reviews the literature."
    assert result.stats["sentences_removed_uncited_finding"] == 0
    assert result.stats["needs_citation_markers_removed"] == 1


def test_removes_an_unsupported_sentence_whose_fragment_carries_the_previous_marker():
    """The linker records a cited sentence without the marker
    the writer left at the end of the sentence before it, which the splitter puts at
    the head of this fragment. Keyed on the raw fragment, the sentence matched no link
    at all, fell through rule 1 as an unlisted uncited sentence and was kept, leaving
    an unsupported citation in the text the write job saves, which is what rule 2
    exists to prevent."""
    sentence = "The replication failed (Park, 2018)."
    text = f"No effect was observed. [NEEDS CITATION] {sentence}"
    links = [_link(sentence, ["park_2018"], citation_text="(Park, 2018)")]
    status = {(sentence, sentence, "park_2018"): "unsupported"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "No effect was observed."
    assert result.stats["sentences_removed_unverified"] == 1
    assert result.citation_links == []


def test_removes_an_unsupported_sentence_whose_fragment_carries_a_doubled_space_marker():
    """The same shape with the doubled space a mark split leaves inside the marker."""
    sentence = "The replication failed (Park, 2018)."
    text = f"No effect was observed. [NEEDS  CITATION] {sentence}"
    links = [_link(sentence, ["park_2018"], citation_text="(Park, 2018)")]
    status = {(sentence, sentence, "park_2018"): "unsupported"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "No effect was observed."
    assert result.stats["sentences_removed_unverified"] == 1


def test_drops_a_no_full_text_citation_whose_fragment_carries_the_previous_marker():
    """Rule 3 through the same normalisation: the sentence keeps its verified citation
    and loses the uncheckable one own rendered text, rather than being kept whole
    because its fragment carried the previous sentence marker and so matched no link
    at all."""
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    text = f"No effect was observed. [NEEDS CITATION] {sentence}"
    links = [
        _link(sentence, ["smith_2020"], citation_text="(Smith, 2020)"),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "No effect was observed. Tutoring works (Smith, 2020)."
    assert result.stats["citations_dropped_no_full_text"] == 1
    assert [link["keys"] for link in result.citation_links] == [["smith_2020"]]


def test_strips_a_marker_whose_two_words_are_split_by_any_whitespace():
    """The marker pattern itself tolerates whitespace now, rather than relying on a
    collapse of runs of two or more spaces having already run over the text: a tab
    between its two words is the shape that collapse never reached."""
    text = "Costs may fall [NEEDS\tCITATION]."

    result = finalize_generated_section(text, [], [], {})

    assert result.text == "Costs may fall."
    assert result.stats["needs_citation_markers_removed"] == 1


def test_two_sentences_identical_once_their_marker_is_stripped_share_one_link_list():
    """The documented duplicate-sentence limitation: the
    per-sentence link map is keyed on the normalised sentence, so a paragraph that
    repeats a sentence applies one stored link to both occurrences and returns it once.
    What this pins is that it happens on the FIRST pass, so a second finalize is a
    byte-identical no-op. Without this, the two fragments would match different link
    lists on pass 1 (only one of them carrying the marker) and converge on pass 2,
    rewriting the paragraph's own stored links a whole heal after its text had
    settled."""
    sentence = "Lee (2021) reported that attendance rose steadily."
    text = f"{sentence} [NEEDS CITATION] {sentence}"
    links = [_link(sentence, ["lee_2021"], citation_text="Lee (2021)", evidence_ids=["e2"])]
    status = {(sentence, sentence, "lee_2021"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == f"{sentence} {sentence}"
    assert [link.get("evidence_ids") for link in result.citation_links] == [["e2"]]
    assert result.stats["needs_citation_markers_removed"] == 1

    again = finalize_generated_section(result.text, result.citation_links, [], status)

    assert again.text == result.text
    assert again.citation_links == result.citation_links
    assert all(value == 0 for value in again.stats.values())


def test_an_unsupported_citation_is_removed_from_every_repeat_of_its_sentence():
    """The other half of the duplicate-sentence limitation, and the half that matters
    for the exit invariant: one stored link is applied to every occurrence of its
    sentence, so an unsupported citation cannot survive in the second occurrence of a
    sentence the first occurrence was removed for."""
    sentence = "The replication failed (Park, 2018)."
    text = f"{sentence} {sentence}"
    links = [_link(sentence, ["park_2018"], citation_text="(Park, 2018)")]
    status = {(sentence, sentence, "park_2018"): "unsupported"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == ""
    assert result.stats["sentences_removed_unverified"] == 2


def test_the_status_lookup_sees_the_same_sentence_the_link_map_was_found_under():
    """The third key built with the one normaliser: a claim verified against the
    sentence as the model wrote it, marker and all, must still be found when the
    fragment is looked up marker-stripped. Keyed on two different forms, the verdict
    would be missing and rule 2 would remove the sentence as if it had never been
    verified at all."""
    marked = "Tutoring works (Smith, 2020) [NEEDS CITATION]."
    links = [_link(marked, ["smith_2020"], citation_text="(Smith, 2020)")]
    status = {(marked, marked, "smith_2020"): "verified"}

    result = finalize_generated_section(marked, links, [], status)

    assert result.text == "Tutoring works (Smith, 2020)."
    assert result.stats["sentences_removed_unverified"] == 0
    assert [link["keys"] for link in result.citation_links] == [["smith_2020"]]


def test_a_markdown_heading_is_dropped_whole_even_if_flagged_as_an_uncited_finding():
    """A heading block is dropped whole under ``headings_removed`` before the
    uncited-finding rule ever gets a chance to touch it -- it is never split into
    sentences, never counted as a finding removal, and never delivered either."""
    text = "## Background\n\nTutoring is effective (Smith, 2020)."
    uncited = [{"paragraph_index": 0, "sentence": "## Background", "tag": "finding"}]
    links = [_link("Tutoring is effective (Smith, 2020).", ["smith_2020"], paragraph_index=1)]
    status = {("Tutoring is effective (Smith, 2020).", "Tutoring is effective (Smith, 2020).", "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, uncited, status)

    assert "Background" not in result.text
    assert result.text == "Tutoring is effective (Smith, 2020)."
    assert result.stats["headings_removed"] == 1
    assert result.stats.get("sentences_removed_uncited_finding", 0) == 0


def test_finalize_drops_a_heading_left_as_the_sections_own_first_block():
    """`app.services.writing._strip_duplicate_leading_heading` only ever inspects the
    model's raw text before generation, so it cannot see a sub-heading finalize's OWN
    removals later leave standing as the section's own first surviving block. Here the
    section's genuine opening paragraph is entirely removed (its only sentence
    unsupported -- the opening-paragraph exemption only protects a paragraph that
    survives with no citation link, not one emptied out completely), leaving
    ``## Background`` as the raw text's second block -- dropped whole here too,
    whatever it says, the same as any other heading the writer wrote anywhere in
    the body. The body paragraph that follows it is unaffected."""
    text = (
        "Framing intro that will be entirely removed (Jones, 2019).\n\n"
        "## Background\n\n"
        "Body paragraph survives (Smith, 2020)."
    )
    links = [
        _link(
            "Framing intro that will be entirely removed (Jones, 2019).",
            ["jones_2019"], citation_text="(Jones, 2019)", paragraph_index=0,
        ),
        _link(
            "Body paragraph survives (Smith, 2020).",
            ["smith_2020"], paragraph_index=2,
        ),
    ]
    status = {
        ("Framing intro that will be entirely removed (Jones, 2019).", "Framing intro that will be entirely removed (Jones, 2019).", "jones_2019"): "unsupported",
        ("Body paragraph survives (Smith, 2020).", "Body paragraph survives (Smith, 2020).", "smith_2020"): "verified",
    }

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "Body paragraph survives (Smith, 2020)."
    assert result.stats["headings_removed"] == 1
    assert [link["keys"] for link in result.citation_links] == [["smith_2020"]]
    assert [link["paragraph_index"] for link in result.citation_links] == [0]


def test_finalize_drops_a_bold_heading_left_as_the_sections_own_first_block():
    text = (
        "Framing intro that will be entirely removed (Jones, 2019).\n\n"
        "**Background**\n\n"
        "Body paragraph survives (Smith, 2020)."
    )
    links = [
        _link(
            "Framing intro that will be entirely removed (Jones, 2019).",
            ["jones_2019"], citation_text="(Jones, 2019)", paragraph_index=0,
        ),
        _link(
            "Body paragraph survives (Smith, 2020).",
            ["smith_2020"], paragraph_index=2,
        ),
    ]
    status = {
        ("Framing intro that will be entirely removed (Jones, 2019).", "Framing intro that will be entirely removed (Jones, 2019).", "jones_2019"): "unsupported",
        ("Body paragraph survives (Smith, 2020).", "Body paragraph survives (Smith, 2020).", "smith_2020"): "verified",
    }

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "Body paragraph survives (Smith, 2020)."
    assert result.stats["headings_removed"] == 1


def test_a_whole_line_bold_heading_is_dropped_whole_even_if_flagged_as_an_uncited_finding():
    """A whole-line bold run that reads as a sub-heading (`_is_heading_block`) is
    dropped whole under ``headings_removed`` before the uncited-finding rule ever
    sees it -- never split into sentences, never counted as a finding removal, and
    never delivered either, exactly like a ``#``-style heading."""
    text = "**Key Findings**\n\nTutoring is effective (Smith, 2020)."
    uncited = [{"paragraph_index": 0, "sentence": "**Key Findings**", "tag": "finding"}]
    links = [_link("Tutoring is effective (Smith, 2020).", ["smith_2020"], paragraph_index=1)]
    status = {("Tutoring is effective (Smith, 2020).", "Tutoring is effective (Smith, 2020).", "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, uncited, status)

    assert "Key Findings" not in result.text
    assert result.text == "Tutoring is effective (Smith, 2020)."
    assert result.stats["headings_removed"] == 1
    assert result.stats.get("sentences_removed_uncited_finding", 0) == 0


def test_a_bold_block_carrying_a_citation_is_body_prose_not_a_heading():
    """A whole-line bold run that carries
    a citation and ends in sentence punctuation is a claim sentence the writer happened
    to bold, not a sub-heading -- `_is_heading_block` must not exempt it from the
    citation-status rules. Otherwise, the sentence below (status "unsupported") would
    survive finalize verbatim, its link dropped from ``surviving_links``, and
    `_build_section_tiptap_nodes` would save it as a level-3 heading node -- invisible
    to the verification gate, the final report and every later coverage-gap scan."""
    sentence = "**Direct corrections outperformed metalinguistic codes by 32% (Smith, 2020).**"
    text = f"Framing prose opens the paragraph.\n\n{sentence}"
    links = [_link(sentence, ["smith_2020"], paragraph_index=1)]
    status = {(sentence, sentence, "smith_2020"): "unsupported"}

    result = finalize_generated_section(text, links, [], status)

    assert "Direct corrections outperformed" not in result.text
    assert result.text == "Framing prose opens the paragraph."
    assert result.citation_links == []
    assert result.stats["sentences_removed_unverified"] == 1


def test_a_bold_block_ending_in_a_period_with_no_citation_is_also_body_prose():
    """The sentence-punctuation half of the same guard, with no citation at all: a
    whole-line bold run that reads as a full sentence is still not a genuine
    sub-heading, and an uncited "finding" verdict on it must be enforced, not skipped."""
    sentence = "**This reads like a finding, not a heading.**"
    text = sentence
    uncited = [{"paragraph_index": 0, "sentence": sentence, "tag": "finding"}]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == ""
    assert result.stats["sentences_removed_uncited_finding"] == 1


def test_paragraph_index_is_recomputed_after_a_paragraph_is_dropped():
    text = (
        "Unsupported claim here (Smith, 2020).\n\n"
        "This claim is verified (Jones, 2019)."
    )
    links = [
        _link("Unsupported claim here (Smith, 2020).", ["smith_2020"], paragraph_index=0),
        _link("This claim is verified (Jones, 2019).", ["jones_2019"], paragraph_index=1),
    ]
    status = {
        ("Unsupported claim here (Smith, 2020).", "Unsupported claim here (Smith, 2020).", "smith_2020"): "unsupported",
        ("This claim is verified (Jones, 2019).", "This claim is verified (Jones, 2019).", "jones_2019"): "verified",
    }

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "This claim is verified (Jones, 2019)."
    assert result.citation_links[0]["paragraph_index"] == 0


def test_a_link_naming_the_wrong_paragraph_index_is_repaired_against_its_own_sentence():
    """A link's own reported ``paragraph_index`` is untrusted
    model output. Without repairing against the link's own sentence, an unsupported
    sentence whose link named the WRONG paragraph would count as "covered" for the
    index-agnostic coverage check while finalize, keying its removal rule on the
    (wrong) reported index, never sees the citation on the paragraph that actually
    holds it -- so the unverified sentence would survive unremoved. Here the link
    claims paragraph_index 1 (the second paragraph) even though its own sentence is
    the first paragraph's; finalize must still remove it."""
    text = (
        "Unsupported claim here (Smith, 2020).\n\n"
        "This claim is verified (Jones, 2019)."
    )
    links = [
        _link("Unsupported claim here (Smith, 2020).", ["smith_2020"], paragraph_index=1),
        _link("This claim is verified (Jones, 2019).", ["jones_2019"], paragraph_index=0),
    ]
    status = {
        ("Unsupported claim here (Smith, 2020).", "Unsupported claim here (Smith, 2020).", "smith_2020"): "unsupported",
        ("This claim is verified (Jones, 2019).", "This claim is verified (Jones, 2019).", "jones_2019"): "verified",
    }

    result = finalize_generated_section(text, links, [], status)

    assert result.text == "This claim is verified (Jones, 2019)."
    assert result.stats["sentences_removed_unverified"] == 1


# --------------------------------------------------------------------------------------
# finalize_draft_document: the Tiptap-document-level twin used by verify_and_heal_claims.
# --------------------------------------------------------------------------------------


def test_finalize_draft_document_drops_an_unverified_paragraph_keeps_heading():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Lit review"}],
            },
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Unsupported claim (Smith, 2020).",
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": "Unsupported claim (Smith, 2020)."}],
            },
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Verified claim (Jones, 2019).",
                            "keys": ["jones_2019"],
                            "citation_text": "(Jones, 2019)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": "Verified claim (Jones, 2019)."}],
            },
        ],
    }
    status = {
        ("Unsupported claim (Smith, 2020).", "Unsupported claim (Smith, 2020).", "smith_2020"): "unsupported",
        ("Verified claim (Jones, 2019).", "Verified claim (Jones, 2019).", "jones_2019"): "verified",
    }
    claims = [
        ("Unsupported claim (Smith, 2020).", "smith_2020", "(Smith, 2020)",
         "Unsupported claim (Smith, 2020)."),
        ("Verified claim (Jones, 2019).", "jones_2019", "(Jones, 2019)",
         "Verified claim (Jones, 2019)."),
    ]

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph"]
    assert new_content["content"][1]["content"][0]["text"] == "Verified claim (Jones, 2019)."
    assert surviving_links == [
        {
            "sentence": "Verified claim (Jones, 2019).",
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
        }
    ]
    assert stats["sentences_removed_unverified"] == 1


def test_finalize_draft_document_drops_an_empty_list_container():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "attrs": {
                                    "citationLinks": [
                                        {
                                            "sentence": "Bad claim (Smith, 2020).",
                                            "keys": ["smith_2020"],
                                            "citation_text": "(Smith, 2020)",
                                        }
                                    ]
                                },
                                "content": [{"type": "text", "text": "Bad claim (Smith, 2020)."}],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    status = {("Bad claim (Smith, 2020).", "Bad claim (Smith, 2020).", "smith_2020"): "unsupported"}
    claims = [
        ("Bad claim (Smith, 2020).", "smith_2020", "(Smith, 2020)", "Bad claim (Smith, 2020)."),
    ]

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )

    assert new_content["content"] == []
    assert surviving_links == []
    assert stats["sentences_removed_unverified"] == 1


def test_finalize_draft_document_returns_a_paragraph_needing_no_healing_byte_identical():
    """A paragraph with marks, `origin` and `blockId` that needs no
    healing at all must come back as the exact same node object -- not rebuilt into a
    single plain-text node, which would silently drop every mark and relabel the
    user's own paragraph as AI-written (the editor's provenance indicator defaults
    `origin` to "ai" when the attr is missing)."""
    paragraph = {
        "type": "paragraph",
        "attrs": {"origin": "user", "blockId": "blk-1"},
        "content": [
            {"type": "text", "text": "My own paragraph, "},
            {"type": "text", "marks": [{"type": "bold"}], "text": "emphasised"},
            {"type": "text", "text": ", with no citation."},
        ],
    }
    doc = {"type": "doc", "content": [paragraph]}

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, [], {}
    )

    assert new_content["content"][0] is paragraph
    assert all(v == 0 for v in stats.values())
    assert surviving_links == []


def test_finalize_draft_document_keeps_a_verified_sentence_that_carries_a_mark():
    """A user-added mark (here, bold) splits the paragraph's
    text into several nodes; `_extract_text_from_tiptap` re-joins them with a literal
    space, so the paragraph's current text no longer matches the stored link's own
    ``sentence`` byte for byte. If `_claims_for_paragraph_text` carried that stale
    stored sentence into the rebuilt entry, `claim_status` (keyed on the claim's own,
    freshly-extracted sentence) would never match, and the whole sentence -- verified,
    marks and all -- would be removed as if it had no verdict at all. The rebuilt entry
    must carry the sentence the verdict is actually keyed on."""
    smith_sentence = "Tutoring works well  everywhere  (Smith, 2020)."
    jones_sentence = "Costs also fall (Jones, 2019)."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "citationLinks": [
                {
                    "sentence": "Tutoring works well everywhere (Smith, 2020).",
                    "keys": ["smith_2020"],
                    "citation_text": "(Smith, 2020)",
                    "proposition": "Tutoring works well everywhere",
                    "evidence_ids": ["e1"],
                },
                {
                    "sentence": jones_sentence,
                    "keys": ["jones_2019"],
                    "citation_text": "(Jones, 2019)",
                },
            ]
        },
        "content": [
            {"type": "text", "text": "Tutoring works well "},
            {"type": "text", "marks": [{"type": "bold"}], "text": "everywhere"},
            {"type": "text", "text": " (Smith, 2020). Costs also fall (Jones, 2019)."},
        ],
    }
    doc = {"type": "doc", "content": [paragraph]}
    claims = [
        # The claim's own text is the narrower proposition
        # `extract_claims_from_document` would really have matched against the stored
        # link's own `proposition` field, not the whole sentence -- keeping the two
        # consistent is what lets `_claims_for_paragraph_text` re-find this link by
        # (sentence, proposition, key).
        ("Tutoring works well everywhere", "smith_2020", "(Smith, 2020)", smith_sentence),
        (jones_sentence, "jones_2019", "(Jones, 2019)", jones_sentence),
    ]
    status = {
        (smith_sentence, "Tutoring works well everywhere", "smith_2020"): "verified",
        (jones_sentence, jones_sentence, "jones_2019"): "verified",
    }

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )

    assert all(v == 0 for v in stats.values())
    new_paragraph = new_content["content"][0]
    # The mark, and everything around it, survives untouched.
    assert new_paragraph["content"] == paragraph["content"]
    surviving_keys = {k for link in new_paragraph["attrs"]["citationLinks"] for k in link["keys"]}
    assert surviving_keys == {"smith_2020", "jones_2019"}
    smith_link = next(
        link for link in new_paragraph["attrs"]["citationLinks"] if "smith_2020" in link["keys"]
    )
    assert smith_link["proposition"] == "Tutoring works well everywhere"
    assert smith_link["evidence_ids"] == ["e1"]


def test_finalize_draft_document_heals_a_mark_split_needs_citation_marker_on_the_first_pass():
    """A bold mark ending in the middle of the literal text
    "[NEEDS CITATION]" leaves a doubled space between its two words in the text
    `_extract_text_from_tiptap` joins from the paragraph's own text nodes (it joins
    every adjacent pair with a literal space). The
    marker must be removed on the very first heal; the second heal must then be a
    byte-identical no-op reporting the count as zero, which is the repeated-heal
    invariant this fix exists to keep."""
    smith_sentence = "Tutoring works (Smith, 2020)."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "citationLinks": [
                {
                    "sentence": smith_sentence,
                    "keys": ["smith_2020"],
                    "citation_text": "(Smith, 2020)",
                }
            ]
        },
        "content": [
            {"type": "text", "text": f"{smith_sentence} Costs may fall"},
            {"type": "text", "marks": [{"type": "bold"}], "text": "[NEEDS"},
            {"type": "text", "text": " CITATION]."},
        ],
    }
    doc = {"type": "doc", "content": [paragraph]}
    claims = [(smith_sentence, "smith_2020", "(Smith, 2020)", smith_sentence)]
    status = {(smith_sentence, smith_sentence, "smith_2020"): "verified"}

    first_content, first_links, first_stats, _healed = finalize_draft_document(
        doc, claims, status
    )

    healed_text = first_content["content"][0]["content"][0]["text"]
    assert healed_text == f"{smith_sentence} Costs may fall."
    assert first_stats["needs_citation_markers_removed"] == 1

    second_content, second_links, second_stats, _healed2 = finalize_draft_document(
        first_content, claims, status
    )

    assert second_content == first_content
    assert second_links == first_links
    assert all(v == 0 for v in second_stats.values())


def test_finalize_draft_document_removes_an_uncited_finding_preceded_by_a_marker():
    """The sentence splitter puts a `[NEEDS CITATION]`
    marker the model wrote at the end of one sentence at the HEAD of the fragment for
    the sentence that follows it, so an uncited "finding" sentence recorded without the
    marker does not match the fragment as extracted unless the match is made against
    the marker-stripped, whitespace-collapsed fragment. The sentence is removed
    on the same pass its marker is, and the second pass is a no-op."""
    text = "No effect was observed in general. [NEEDS CITATION] Teachers reported gains in general."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "uncitedSentences": [
                {"sentence": "Teachers reported gains in general.", "tag": "finding"}
            ]
        },
        "content": [{"type": "text", "text": text}],
    }
    doc = {"type": "doc", "content": [paragraph]}

    first_content, first_links, first_stats, _healed = finalize_draft_document(doc, [], {})

    assert (
        first_content["content"][0]["content"][0]["text"]
        == "No effect was observed in general."
    )
    assert first_stats["sentences_removed_uncited_finding"] == 1
    assert first_stats["needs_citation_markers_removed"] == 0

    second_content, second_links, second_stats, _healed2 = finalize_draft_document(
        first_content, [], {}
    )

    assert second_content == first_content
    assert second_links == first_links
    assert all(v == 0 for v in second_stats.values())


def test_finalize_draft_document_healed_paragraph_keeps_its_non_citation_attrs():
    """When a paragraph IS healed (a no_full_text citation dropped here), its other
    attrs -- `origin`, `blockId`, and any extra field a stored citationLink already
    carried, like `evidence_ids` -- must survive the rebuild rather than being
    replaced wholesale with the finalize helper's own synthetic three-key link."""
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "origin": "ai",
            "blockId": "blk-2",
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
                    "evidence_ids": ["e2"],
                },
            ],
        },
        "content": [{"type": "text", "text": sentence}],
    }
    doc = {"type": "doc", "content": [paragraph]}
    claims = [
        (sentence, "smith_2020", "(Smith, 2020)", sentence),
        (sentence, "jones_2019", "(Jones, 2019)", sentence),
    ]
    status = {
        (sentence, sentence, "smith_2020"): "verified",
        (sentence, sentence, "jones_2019"): "no_full_text",
    }

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )

    new_paragraph = new_content["content"][0]
    assert new_paragraph is not paragraph
    assert new_paragraph["attrs"]["origin"] == "ai"
    assert new_paragraph["attrs"]["blockId"] == "blk-2"
    # The surviving link's own `sentence` is refreshed
    # to the rewritten text, not left naming the pre-heal wording that still included
    # the dropped citation -- a later pass over this paragraph must find it again.
    # Nothing on the saved link records the pre-heal wording: the document keeps only
    # what still describes itself, and the pre-to-post map goes back to this pass's
    # own caller instead.
    assert new_paragraph["attrs"]["citationLinks"] == [
        {
            "sentence": "Tutoring works (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
            "evidence_ids": ["e1"],
        }
    ]
    assert healed_sentences == {sentence: "Tutoring works (Smith, 2020)."}
    assert stats["citations_dropped_no_full_text"] == 1
    assert surviving_links == new_paragraph["attrs"]["citationLinks"]


def test_finalize_draft_document_mixed_co_citation_group_keeps_its_text_unsplit():
    """The saved-draft path shares the write path's own
    documented limitation for a co-cited group (see the comment in
    `_finalize_paragraph_text` on the mixed-status branch) -- a group such as
    "(Jones, 2019; Lee, 2021)" with one key no_full_text keeps that citation's own
    rendered text unsplit, rather than the text being cut and the sentence mangled,
    and the link survives narrowed to only its verified keys. Pins this choice on the
    saved-draft path (`finalize_draft_document`/`verify_and_heal_claims`), not just the
    write job's own (`test_mixed_co_citation_group_narrows_to_only_its_verified_keys`
    above)."""
    sentence = "Two papers agree (Jones, 2019; Lee, 2021)."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "citationLinks": [
                {
                    "sentence": sentence,
                    "keys": ["jones_2019", "lee_2021"],
                    "citation_text": "(Jones, 2019; Lee, 2021)",
                }
            ]
        },
        "content": [{"type": "text", "text": sentence}],
    }
    doc = {"type": "doc", "content": [paragraph]}
    claims = [
        (sentence, "jones_2019", "(Jones, 2019)", sentence),
        (sentence, "lee_2021", "(Lee, 2021)", sentence),
    ]
    status = {
        (sentence, sentence, "jones_2019"): "verified",
        (sentence, sentence, "lee_2021"): "no_full_text",
    }

    new_content, surviving_links, stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )

    assert new_content["content"][0]["content"][0]["text"] == sentence
    assert surviving_links == [
        {
            "sentence": sentence,
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019; Lee, 2021)",
        }
    ]
    assert stats["citations_dropped_no_full_text"] == 0


# --------------------------------------------------------------------------------------
# surviving_verifications: the "final claim report" must describe what actually
# survived finalize, not the raw verdict list.
# --------------------------------------------------------------------------------------


class _FakeVerification:
    def __init__(self, status, claim_sentence=None):
        self.status = status
        self.claim_sentence = claim_sentence

    def model_copy(self, *, update=None):
        # Mirrors the one pydantic `BaseModel.model_copy(update=...)` call
        # `surviving_verifications` makes: a shallow copy
        # with the given fields overridden, exactly like the real `ClaimVerification`.
        copied = _FakeVerification(self.status, self.claim_sentence)
        for key, value in (update or {}).items():
            setattr(copied, key, value)
        return copied


def test_surviving_verifications_excludes_a_verified_claim_whose_sentence_was_removed():
    """A sentence with two citations, one verified and one unsupported, is removed
    whole by finalize's rule 2 -- the verified citation is gone from the surviving
    links too, so the report must not list it as verified."""
    claims = [
        ("Tutoring boosts outcomes", "smith_2020", "(Smith, 2020)", "Sentence."),
        ("costs fall", "jones_2019", "(Jones, 2019)", "Sentence."),
    ]
    verifications = [_FakeVerification("verified"), _FakeVerification("unsupported")]

    result = surviving_verifications(
        claims, verifications, surviving_links=[], healed_sentences={}
    )

    assert result == []


def test_surviving_verifications_keeps_a_claim_whose_link_survived():
    v_smith = _FakeVerification("verified")
    claims = [("Tutoring boosts outcomes", "smith_2020", "(Smith, 2020)", "Sentence.")]
    verifications = [v_smith]
    # The surviving link's own `proposition` is what
    # `surviving_verifications` pairs the claim back on, alongside its sentence.
    surviving_links = [
        {
            "sentence": "Sentence.",
            "keys": ["smith_2020"],
            "proposition": "Tutoring boosts outcomes",
        }
    ]

    result = surviving_verifications(claims, verifications, surviving_links, {})

    assert result == [v_smith]


def test_surviving_verifications_keeps_a_refreshed_link_through_finalize_generated_section():
    """A no_full_text citation sharing a sentence with a verified
    one is dropped by rule 3, which refreshes the surviving link's own `sentence` to
    the rewritten text. The verified verdict was computed against
    the pre-heal sentence, so `surviving_verifications` must still find it -- the write
    job's own path (`finalize_generated_section`), used by `generate_section`."""
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    links = [
        _link(sentence, ["smith_2020"], citation_text="(Smith, 2020)"),
        _link(sentence, ["jones_2019"], citation_text="(Jones, 2019)"),
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}
    # The links above carry no explicit `proposition`, so their own claim_text (the
    # pairing key) is the whole sentence, exactly as
    # `claims_from_citation_links` would build it in production.
    claims = [
        (sentence, "smith_2020", "(Smith, 2020)", sentence),
        (sentence, "jones_2019", "(Jones, 2019)", sentence),
    ]
    verifications = [
        _FakeVerification("verified", claim_sentence=sentence),
        _FakeVerification("no_full_text", claim_sentence=sentence),
    ]

    result = finalize_generated_section(sentence, links, [], status)
    # The refresh actually happened -- otherwise this test would not exercise M1 at all.
    assert result.citation_links[0]["sentence"] != sentence

    final_verifications = surviving_verifications(
        claims, verifications, result.citation_links, result.healed_sentences
    )

    assert [v.status for v in final_verifications] == ["verified"]
    # The surviving verification's own `claim_sentence` is
    # refreshed to match the rewritten text too, not just the pairing lookup -- a
    # consumer of `final_report` (demo/check_delivered.py rule 1, in particular) must
    # never be shown a `claim_sentence` the delivered draft no longer contains
    # verbatim.
    assert final_verifications[0].claim_sentence == result.citation_links[0]["sentence"]
    assert final_verifications[0].claim_sentence != sentence


def test_surviving_verifications_keeps_a_refreshed_link_through_finalize_draft_document():
    """The same pairing must hold on the saved-draft path (`finalize_draft_document`,
    used by `verify_and_heal_claims`), not just the write job's own."""
    sentence = "Tutoring works (Smith, 2020) and (Jones, 2019)."
    paragraph = {
        "type": "paragraph",
        "attrs": {
            "citationLinks": [
                {"sentence": sentence, "keys": ["smith_2020"], "citation_text": "(Smith, 2020)"},
                {"sentence": sentence, "keys": ["jones_2019"], "citation_text": "(Jones, 2019)"},
            ]
        },
        "content": [{"type": "text", "text": sentence}],
    }
    doc = {"type": "doc", "content": [paragraph]}
    claims = [
        (sentence, "smith_2020", "(Smith, 2020)", sentence),
        (sentence, "jones_2019", "(Jones, 2019)", sentence),
    ]
    status = {(sentence, sentence, "smith_2020"): "verified", (sentence, sentence, "jones_2019"): "no_full_text"}
    verifications = [
        _FakeVerification("verified", claim_sentence=sentence),
        _FakeVerification("no_full_text", claim_sentence=sentence),
    ]

    _content, surviving_links, _stats, healed_sentences = finalize_draft_document(
        doc, claims, status
    )
    assert surviving_links[0]["sentence"] != sentence

    final_verifications = surviving_verifications(
        claims, verifications, surviving_links, healed_sentences
    )

    assert [v.status for v in final_verifications] == ["verified"]
    assert final_verifications[0].claim_sentence == surviving_links[0]["sentence"]


def test_finalize_draft_document_leaves_a_references_section_untouched():
    doc = {
        "type": "doc",
        "content": [
            {"type": "heading", "content": [{"type": "text", "text": "References"}]},
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "1. Smith, J. (2020). Title."}],
            },
        ],
    }

    new_content, _links, _stats, _healed = finalize_draft_document(doc, [], {})

    assert new_content["content"] == doc["content"]


# --------------------------------------------------------------------------------------
# The section title is the only heading: the node rebuild drops every other heading
# node whole, whatever it says. Covers the shape where the section's own title
# heading is directly followed by the writer's own leading sub-heading, with no
# body paragraph between them, once the section's real intro
# paragraph was removed.
# --------------------------------------------------------------------------------------


def _title_heading(text):
    return {
        "type": "heading",
        "attrs": {"level": 2, "sectionType": "literature_review"},
        "content": [{"type": "text", "text": text}],
    }


def _plain_heading(text, level=2):
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _cited_paragraph(text, key, citation_text="(X, 2020)"):
    return {
        "type": "paragraph",
        "attrs": {
            "citationLinks": [
                {"sentence": text, "keys": [key], "citation_text": citation_text}
            ]
        },
        "content": [{"type": "text", "text": text}],
    }


def test_node_rebuild_drops_run_8_section_3_second_heading_prose_intact():
    """Run 8's own delivered ``draft_content_3.json``, first three nodes: the
    section-title heading ("Learner engagement with written corrective feedback")
    directly followed by the writer's own leading sub-heading ("Effects of Feedback
    Types on Accuracy and Revision"), then the paragraph that used to be the second
    body block. The second heading is dropped whole; the paragraph, and its own
    citation link, are unchanged."""
    body_text = (
        "Immediate accuracy during revision improved under both direct and coded "
        "feedback, but only direct corrections showed a long-term advantage four "
        "weeks after feedback provision (Bonilla López et al., 2018)."
    )
    doc = {
        "type": "doc",
        "content": [
            _title_heading("Learner engagement with written corrective feedback"),
            _plain_heading("Effects of Feedback Types on Accuracy and Revision"),
            _cited_paragraph(body_text, "lopez_2018", "(Bonilla López et al., 2018)"),
        ],
    }
    claims = [(body_text, "lopez_2018", "(Bonilla López et al., 2018)", body_text)]
    status = {(body_text, body_text, "lopez_2018"): "verified"}

    new_content, surviving_links, stats, _healed = finalize_draft_document(
        doc, claims, status
    )

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph"]
    assert new_content["content"][0]["content"][0]["text"] == (
        "Learner engagement with written corrective feedback"
    )
    assert new_content["content"][1] == doc["content"][2]
    assert stats["headings_removed"] == 1
    assert surviving_links == [
        {
            "sentence": body_text,
            "keys": ["lopez_2018"],
            "citation_text": "(Bonilla López et al., 2018)",
        }
    ]


def test_node_rebuild_drops_run_8_section_5_second_heading_prose_intact():
    """Run 8's own delivered ``draft_content_5.json``, first three nodes: the
    section-title heading ("Metalinguistic codes as written corrective feedback")
    directly followed by the writer's own leading sub-heading ("Metalinguistic
    Coded Feedback in L2 Writing"), then the paragraph that used to be the second
    body block. Same shape, same fix, same result: second heading gone, paragraph
    untouched."""
    body_text = (
        "Bonilla López et al. (2018) randomly assigned 139 low-intermediate "
        "EFL writers to four experimental conditions."
    )
    doc = {
        "type": "doc",
        "content": [
            _title_heading("Metalinguistic codes as written corrective feedback"),
            _plain_heading("Metalinguistic Coded Feedback in L2 Writing"),
            _cited_paragraph(body_text, "lopez_2018", "(2018)"),
        ],
    }
    claims = [(body_text, "lopez_2018", "(2018)", body_text)]
    status = {(body_text, body_text, "lopez_2018"): "verified"}

    new_content, surviving_links, stats, _healed = finalize_draft_document(
        doc, claims, status
    )

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph"]
    assert new_content["content"][0]["content"][0]["text"] == (
        "Metalinguistic codes as written corrective feedback"
    )
    assert new_content["content"][1] == doc["content"][2]
    assert stats["headings_removed"] == 1
    assert surviving_links == [
        {"sentence": body_text, "keys": ["lopez_2018"], "citation_text": "(2018)"}
    ]


def test_node_rebuild_keeps_a_mid_section_sub_heading_gone_but_its_prose_kept():
    """A draft where the section's real intro survives: the
    section-title heading, an intro paragraph, a mid-section sub-heading, then more
    prose. The sub-heading is dropped whole regardless of an intro being present or
    not; both paragraphs, before and after it, are delivered unchanged."""
    intro = "Feedback research spans several theoretical traditions (Lee, 2021)."
    body = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    doc = {
        "type": "doc",
        "content": [
            _title_heading("Learner engagement with written corrective feedback"),
            _cited_paragraph(intro, "lee_2021"),
            _plain_heading("Effects of Feedback Types on Accuracy and Revision"),
            _cited_paragraph(body, "liu_2019"),
        ],
    }
    claims = [
        (intro, "lee_2021", "(X, 2020)", intro),
        (body, "liu_2019", "(X, 2020)", body),
    ]
    status = {
        (intro, intro, "lee_2021"): "verified",
        (body, body, "liu_2019"): "verified",
    }

    new_content, _surviving_links, stats, _healed = finalize_draft_document(
        doc, claims, status
    )

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph", "paragraph"]
    assert "Effects of Feedback Types" not in str(new_content)
    assert stats["headings_removed"] == 1


def test_finalize_generated_section_drops_a_references_heading_and_its_own_list():
    """A "## References" heading the writer wrote inside its own generated body,
    followed by a markdown list of citations with no citation link of its own: the
    heading is dropped whole (``headings_removed``); the list, carrying no
    surviving citation link and not the section's own opening paragraph, is
    dropped too, by the pre-existing dangling-paragraph rule -- both gone, neither
    by the same mechanism."""
    opening = "Feedback research spans several theoretical traditions (Lee, 2021)."
    text = (
        f"{opening}\n\n"
        "## References\n\n"
        "- Lee, S. (2021). Feedback in context.\n"
        "- Liu, Y. (2019). Correction types."
    )
    links = [_link(opening, ["lee_2021"], citation_text="(Lee, 2021)", paragraph_index=0)]
    status = {(opening, opening, "lee_2021"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert result.text == opening
    assert "References" not in result.text
    assert "Lee, S." not in result.text
    assert result.stats["headings_removed"] == 1
    assert result.stats.get("sentences_removed_dangling", 0) >= 1
