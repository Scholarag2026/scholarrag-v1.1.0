"""The deterministic coherence pass finalize runs after its own three ordinary removal
rules: a dangling or misattributed sentence -- an enumeration opener ("Three gaps
emerge.") with fewer following sentences than announced, or a framing sentence opening
with a discourse connective/ordinal or an unresolved deictic once something earlier in
its own paragraph was removed -- must not survive into the delivered text.

Three rules, in `app.services.fulltext`:

(a) a body paragraph left with no cited verified sentence is removed
    (`finalize_generated_section`, `_rebuild_finalized_node_list`), except the
    section's own opening paragraph;
(b) an uncited framing sentence whose own opening announces an enumeration
    (`_enumeration_announced_count`) is dropped when fewer surviving sentences than
    announced follow it in the paragraph (`_drop_dangling_framing_sentences`);
(c) an uncited framing sentence that opens with a discourse connective/ordinal or an
    unresolved deictic (`_opens_with_dangling_marker`) is dropped once some earlier
    fragment of the same paragraph has already been removed -- by the three ordinary
    rules or by (b) itself, so a cascade ("Three gaps emerge." dropped, then the
    ordinal-led "Third, ..." that pointed at it) is caught in one pass.

Both example-based tests and a property test over a small grammar of randomly ordered
fragments with a fixed seed, in the style of `test_citation_coverage_properties.py` and
`test_repeated_heal_properties.py`: no hypothesis library, plain `random`, reproducible
on failure.

All pure functions over plain strings and dicts: no network, no database, no model.
"""

import random

from app.services.fulltext import (
    _drop_dangling_framing_sentences,
    _enumeration_announced_count,
    _next_fragment_resolves_enumeration_as_a_list,
    _opens_with_dangling_marker,
    finalize_draft_document,
    finalize_generated_section,
)

# --------------------------------------------------------------------------------------
# _enumeration_announced_count
# --------------------------------------------------------------------------------------


def test_enumeration_announced_count_reads_a_number_word():
    assert _enumeration_announced_count("Three gaps emerge.") == 3


def test_enumeration_announced_count_reads_a_numeral():
    assert _enumeration_announced_count("4 themes follow.") == 4


def test_enumeration_announced_count_reads_other_verbs():
    assert _enumeration_announced_count("Two limitations stand out.") == 2


def test_enumeration_announced_count_ignores_a_plain_factual_verb():
    """"are", "were", "exist(s/ed)" and "remain(s/ed)" are
    generic existential verbs an ordinary factual sentence uses just as often as a real
    announcement -- these four are structurally identical to a genuine opener (number,
    plural noun, verb) but announce nothing, so they must never return a count."""
    assert _enumeration_announced_count("Three studies were conducted in naturalistic "
                                         "classrooms.") is None
    assert _enumeration_announced_count("Four papers are included in this review.") is None
    assert _enumeration_announced_count(
        "Two limitations remain unaddressed in the literature."
    ) is None
    assert _enumeration_announced_count("Three themes are evident in the library.") is None


def test_enumeration_announced_count_is_none_when_the_sentence_resolves_itself():
    """A colon-introduced, self-contained list resolves its own count in the same
    sentence -- "Three gaps remain: short designs; small corpora; and no engagement
    tracking." is never a shortfall, however many separate sentences follow it."""
    assert _enumeration_announced_count(
        "Three gaps remain: short designs; small corpora; and no engagement tracking."
    ) is None


def test_enumeration_announced_count_is_none_without_the_opening_shape():
    assert _enumeration_announced_count("These gaps motivate the present review.") is None
    assert _enumeration_announced_count("The three gaps are significant.") is None
    assert _enumeration_announced_count("Notably, three gaps emerge in the field.") is None
    assert _enumeration_announced_count("") is None


# --------------------------------------------------------------------------------------
# _next_fragment_resolves_enumeration_as_a_list
# --------------------------------------------------------------------------------------


def test_next_fragment_resolves_enumeration_as_a_list_true_in_order():
    """A colon-free list of the same three items, resolved one
    sentence later than the opener itself."""
    assert _next_fragment_resolves_enumeration_as_a_list(
        "First, designs are short; second, corpora are small; third, engagement is "
        "untracked (Smith, 2020).",
        3,
    )


def test_next_fragment_resolves_enumeration_as_a_list_false_out_of_order():
    assert not _next_fragment_resolves_enumeration_as_a_list(
        "Third, engagement is untracked; first, designs are short.", 3
    )


def test_next_fragment_resolves_enumeration_as_a_list_false_when_short():
    assert not _next_fragment_resolves_enumeration_as_a_list(
        "First, designs are short.", 3
    )


def test_next_fragment_resolves_enumeration_as_a_list_false_for_ordinary_prose():
    assert not _next_fragment_resolves_enumeration_as_a_list(
        "Tutoring boosts accuracy (Smith, 2020).", 3
    )


# --------------------------------------------------------------------------------------
# _opens_with_dangling_marker
# --------------------------------------------------------------------------------------


def test_opens_with_dangling_marker_true_for_a_connective():
    assert _opens_with_dangling_marker("However, the same authors caution against this.")
    assert _opens_with_dangling_marker("Third, theoretical grounding remains weak.")
    assert _opens_with_dangling_marker("Crucially, the authors caution that it is limited.")
    assert _opens_with_dangling_marker("First, designs are short.")


def test_opens_with_dangling_marker_true_for_a_deictic():
    assert _opens_with_dangling_marker("The study's own authors caution against this.")
    assert _opens_with_dangling_marker("The same authors caution against overuse.")
    assert _opens_with_dangling_marker("This finding is important for the field.")
    assert _opens_with_dangling_marker("These gaps motivate the present investigation.")
    assert _opens_with_dangling_marker("Such integration is not a panacea.")


def test_opens_with_dangling_marker_false_for_ordinary_prose():
    assert not _opens_with_dangling_marker("Tutoring works well for most learners.")
    assert not _opens_with_dangling_marker("Sources were selected for their relevance.")
    assert not _opens_with_dangling_marker("The comparative evidence remains mixed.")


# --------------------------------------------------------------------------------------
# _drop_dangling_framing_sentences directly, on fragment records
# --------------------------------------------------------------------------------------


def _record(text, has_citation=False, preceded_by_removal=False, unclassified=False):
    return {
        "text": text,
        "has_citation": has_citation,
        "preceded_by_removal": preceded_by_removal,
        "uncited_entries": [{"unclassified": True}] if unclassified else [],
    }


def test_enumeration_opener_dropped_when_short_of_its_count():
    """The shape: "Three gaps emerge." then
    only one sentence follows -- the opener is dropped, and the cascade then drops the
    ordinal-led follower too, since something before it (the opener) is now gone."""
    records = [
        _record("Three gaps emerge."),
        _record("Third, theoretical grounding remains weak across the field."),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert survivors == []
    assert stats["sentences_removed_dangling"] == 2


def test_enumeration_opener_kept_when_its_count_is_met():
    records = [
        _record("Three gaps emerge."),
        _record("First, theoretical grounding remains weak.", has_citation=True),
        _record("Second, sample sizes are small.", has_citation=True),
        _record("Third, replication is rare.", has_citation=True),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


def test_dangling_opener_dropped_only_after_a_removal():
    """`preceded_by_removal=False`: nothing before it was removed, so the connective
    opener is kept -- rule (c) never fires without a removal to react to."""
    records = [
        _record("Engagement research complicates any simple account."),
        _record("However, further research is needed in this area."),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


def test_dangling_opener_dropped_when_it_is_the_paragraphs_own_first_delivered_fragment():
    """``preceded_by_removal=True`` on the FIRST record: an ordinary rule removed
    something before this paragraph delivers anything at all, so this connective has no
    antecedent whatsoever, cited or not, anywhere in the delivered paragraph."""
    records = [
        _record("However, the same authors caution against overuse.", preceded_by_removal=True),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert survivors == []
    assert stats["sentences_removed_dangling"] == 1


def test_dangling_opener_survives_an_ordinary_removal_once_a_real_antecedent_precedes_it():
    """``preceded_by_removal=True`` is only ever read off the
    FIRST record. "Tutoring boosts accuracy (Smith, 2020). [an unrelated sentence
    removed] However, this finding rests on a single instructional setting." keeps the
    "However" sentence, because the Smith sentence is a real, present antecedent for it
    -- an ordinary rule's removal of something else in the paragraph does not, by
    itself, make a LATER connective dangling."""
    records = [
        _record("Tutoring boosts accuracy (Smith, 2020).", has_citation=True),
        _record(
            "However, this finding rests on a single instructional setting.",
            preceded_by_removal=True,
        ),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


def test_dangling_opener_still_cascades_from_an_enumeration_shortfall_past_a_survivor():
    """An enumeration shortfall (b) is a genuine, permanent absence, unlike an ordinary
    removal -- its own "trouble" keeps propagating even past a real, cited, surviving
    sentence in between, because that survivor is not what the deictic below actually
    refers to."""
    records = [
        _record("Three gaps emerge."),
        _record("Tutoring boosts accuracy (Smith, 2020).", has_citation=True),
        _record("These gaps motivate the present investigation."),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == ["Tutoring boosts accuracy (Smith, 2020)."]
    assert stats["sentences_removed_dangling"] == 2


def test_a_cited_sentence_is_never_dropped_by_either_rule():
    """Both (b) and (c) are scoped to an UNCITED framing sentence, exactly as
    authorised: a cited sentence that happens to share the same wording survives."""
    records = [
        _record("Three gaps emerge (Smith, 2020).", has_citation=True),
        _record("However, the same authors caution against overuse.", has_citation=True,
                 preceded_by_removal=True),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


def test_an_unclassified_enumeration_opener_survives_its_own_shortfall():
    """Rule (b) must not drop an enumeration opener short of
    its own count when the opener itself carries ``"unclassified": True`` --
    exactly the shape `_finalize_paragraph_text`'s own docstring and rule (a) both say
    is never removed. Nothing else in the paragraph was removed either, so the
    connective sentence after it is untouched."""
    records = [
        _record("Three gaps emerge.", unclassified=True),
        _record("However, the same authors caution that the effect may not persist."),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


def test_an_unclassified_connective_survives_a_preceding_ordinary_removal():
    """The same review finding's second shape: an ordinary rule already removed the
    fragment before this one (``preceded_by_removal=True``), which would ordinarily
    poison this paragraph's own first delivered fragment under rule (c) -- but this
    fragment itself carries ``"unclassified": True``, so it is exempt, the same as a
    cited fragment already is (`test_dangling_opener_dropped_when_it_is_the_
    paragraphs_own_first_delivered_fragment` is the same shape without the flag, and
    still gets removed)."""
    records = [
        _record(
            "This finding has not been replicated in later work.",
            preceded_by_removal=True,
            unclassified=True,
        ),
    ]
    stats = {"sentences_removed_dangling": 0}

    survivors = _drop_dangling_framing_sentences(records, stats)

    assert [r["text"] for r in survivors] == [r["text"] for r in records]
    assert stats["sentences_removed_dangling"] == 0


# --------------------------------------------------------------------------------------
# Integration: finalize_generated_section, reproducing the report's own shapes
# --------------------------------------------------------------------------------------


def _link(sentence, keys, citation_text="(Smith, 2020)", paragraph_index=0, **extra):
    return {
        "paragraph_index": paragraph_index,
        "sentence": sentence,
        "keys": keys,
        "citation_text": citation_text,
        **extra,
    }


def test_finalize_drops_the_extra_4_node_7_shape():
    text = (
        "Three gaps emerge. Third, theoretical grounding remains weak across the "
        "field (Mao et al., 2024)."
    )
    links = [_link("Third, theoretical grounding remains weak across the field (Mao et al., 2024).",
                    ["mao_2024"], citation_text="(Mao et al., 2024)")]
    status = {
        ("Third, theoretical grounding remains weak across the field (Mao et al., 2024).", "Third, theoretical grounding remains weak across the field (Mao et al., 2024).",
         "mao_2024"): "unsupported",
    }
    uncited = [{"paragraph_index": 0, "sentence": "Three gaps emerge.", "tag": "framing"}]

    result = finalize_generated_section(text, links, uncited, status)

    assert result.text == ""
    assert result.stats["sentences_removed_unverified"] == 1
    assert result.stats["sentences_removed_dangling"] == 1


def test_finalize_drops_the_extra_2_node_9_shape():
    sentence_2 = (
        "These gaps motivate the present review's contribution: a synthesis that "
        "treats feedback type as a design variable."
    )
    text = f"Three gaps emerge. {sentence_2}"
    uncited = [
        {"paragraph_index": 0, "sentence": "Three gaps emerge.", "tag": "framing"},
        {"paragraph_index": 0, "sentence": sentence_2, "tag": "framing"},
    ]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == ""
    assert result.stats["sentences_removed_dangling"] == 2


def test_finalize_keeps_an_enumeration_opener_whose_count_is_satisfied():
    text = (
        "Three gaps emerge. Tutoring improves accuracy (Smith, 2020). "
        "Engagement is under-studied (Jones, 2019). Automation is nascent (Lee, 2021)."
    )
    links = [
        _link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"]),
        _link("Engagement is under-studied (Jones, 2019).", ["jones_2019"],
              citation_text="(Jones, 2019)"),
        _link("Automation is nascent (Lee, 2021).", ["lee_2021"], citation_text="(Lee, 2021)"),
    ]
    status = {
        ("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified",
        ("Engagement is under-studied (Jones, 2019).", "Engagement is under-studied (Jones, 2019).", "jones_2019"): "verified",
        ("Automation is nascent (Lee, 2021).", "Automation is nascent (Lee, 2021).", "lee_2021"): "verified",
    }
    uncited = [{"paragraph_index": 0, "sentence": "Three gaps emerge.", "tag": "framing"}]

    result = finalize_generated_section(text, links, uncited, status)

    assert result.text.startswith("Three gaps emerge.")
    assert result.stats["sentences_removed_dangling"] == 0


def test_finalize_keeps_a_self_resolving_colon_enumeration_regardless_of_what_follows():
    text = (
        "Three gaps remain: short designs; small corpora; and no engagement "
        "tracking. These gaps motivate the present review's focus."
    )
    uncited = [
        {
            "paragraph_index": 0,
            "sentence": (
                "Three gaps remain: short designs; small corpora; and no engagement "
                "tracking."
            ),
            "tag": "framing",
        },
        {
            "paragraph_index": 0,
            "sentence": "These gaps motivate the present review's focus.",
            "tag": "framing",
        },
    ]

    result = finalize_generated_section(text, [], uncited, {})

    assert result.text == text
    assert result.stats["sentences_removed_dangling"] == 0


def test_finalize_drops_a_non_opening_paragraph_with_no_citation():
    text = (
        "This review synthesises the project library across three themes "
        "(Smith, 2020).\n\n"
        "Sources were selected for their relevance to the topic."
    )
    sentence = (
        "This review synthesises the project library across three themes (Smith, 2020)."
    )
    links = [_link(sentence, ["smith_2020"])]
    status = {(sentence, sentence, "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert "Sources were selected" not in result.text
    assert result.stats["sentences_removed_dangling"] == 1


def test_finalize_keeps_a_non_opening_paragraph_carrying_an_unclassified_sentence():
    """A sentence a citation-link retry
    could not classify is never silently dropped -- this outranks rule (a): a real
    regression rule (a) caused on
    `test_resolve_and_finalize_keep_orphaned_sentences_from_writing_result_on_retry_
    failure`, a genuine, coherent closing synthesis paragraph that carries no citation
    of its own but does carry one "unclassified" sentence."""
    text = (
        "Tutoring improves accuracy (Smith, 2020).\n\n"
        "Taken together, the evidence is mixed. No study compares all three "
        "sources directly. That gap motivates the present review."
    )
    links = [_link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"])]
    status = {("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified"}
    uncited = [
        {"paragraph_index": 1, "sentence": "Taken together, the evidence is mixed.",
         "tag": "framing"},
        {"paragraph_index": 1, "sentence": "No study compares all three sources directly.",
         "tag": "finding", "unclassified": True},
        {"paragraph_index": 1, "sentence": "That gap motivates the present review.",
         "tag": "framing"},
    ]

    result = finalize_generated_section(text, links, uncited, status)

    assert "No study compares all three sources directly." in result.text
    assert result.stats["sentences_removed_dangling"] == 0


def test_finalize_keeps_an_unclassified_sentence_through_an_enumeration_cascade():
    """"Three gaps emerge." is genuinely short
    of its own count (rule b removes it, as it should), but the connective sentence
    after it carries ``"unclassified": True`` and must survive rule (c)'s cascade --
    the paragraph has no citation of its own, so without the exemption this whole
    paragraph used to deliver nothing at all."""
    text = (
        "Tutoring improves accuracy (Smith, 2020).\n\n"
        "Three gaps emerge. However, the same authors caution that the effect may not "
        "persist."
    )
    links = [_link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"])]
    status = {("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified"}
    uncited = [
        {"paragraph_index": 1, "sentence": "Three gaps emerge.", "tag": "framing"},
        {
            "paragraph_index": 1,
            "sentence": "However, the same authors caution that the effect may not persist.",
            "tag": "finding",
            "unclassified": True,
        },
    ]

    result = finalize_generated_section(text, links, uncited, status)

    assert "However, the same authors caution that the effect may not persist." in result.text
    assert "Three gaps emerge." not in result.text
    assert result.stats["sentences_removed_dangling"] == 1
    assert result.stats["sentences_unclassified_kept"] == 1


def test_finalize_keeps_an_unclassified_sentence_after_an_ordinary_removal():
    """The paragraph's own citation is
    unsupported and removed by an ordinary rule, leaving its "unclassified" second
    sentence as the paragraph's own first delivered fragment with a real, unresolved
    deictic opener ("This finding...") -- rule (c) must not drop it either, and the
    whole paragraph used to deliver nothing at all."""
    text = (
        "Tutoring improves accuracy (Smith, 2020).\n\n"
        "Peer review is cheaper (Jones, 2019). This finding has not been replicated in "
        "later work."
    )
    links = [
        _link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"]),
        _link("Peer review is cheaper (Jones, 2019).", ["jones_2019"],
              citation_text="(Jones, 2019)"),
    ]
    status = {
        ("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified",
        ("Peer review is cheaper (Jones, 2019).", "Peer review is cheaper (Jones, 2019).", "jones_2019"): "unsupported",
    }
    uncited = [
        {
            "paragraph_index": 1,
            "sentence": "This finding has not been replicated in later work.",
            "tag": "finding",
            "unclassified": True,
        },
    ]

    result = finalize_generated_section(text, links, uncited, status)

    assert "This finding has not been replicated in later work." in result.text
    assert result.stats["sentences_removed_unverified"] == 1
    assert result.stats["sentences_removed_dangling"] == 0
    assert result.stats["sentences_unclassified_kept"] == 1


def test_finalize_keeps_the_opening_paragraph_even_with_no_citation():
    text = (
        "This review synthesises the project library across several themes.\n\n"
        "Tutoring improves accuracy (Smith, 2020)."
    )
    links = [_link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"])]
    status = {("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified"}

    result = finalize_generated_section(text, links, [], status)

    assert "This review synthesises" in result.text
    assert result.stats["sentences_removed_dangling"] == 0


def test_finalize_drops_every_heading_whole_including_a_now_bodiless_one():
    """Every heading is dropped whole now, whatever it says and wherever it sits --
    ``## Motivation`` just as much as ``## Synthesis and Gaps``. The second
    heading's own only paragraph is uncited, not the document's opening one, and
    gets dropped by rule (a) regardless; the heading's own removal is independent
    of that, not a downstream consequence of it."""
    text = (
        "## Motivation\n\n"
        "Tutoring improves accuracy (Smith, 2020).\n\n"
        "## Synthesis and Gaps\n\n"
        "These gaps motivate the present investigation."
    )
    links = [_link("Tutoring improves accuracy (Smith, 2020).", ["smith_2020"])]
    status = {("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified"}
    uncited = [
        {
            "paragraph_index": 3,
            "sentence": "These gaps motivate the present investigation.",
            "tag": "framing",
        }
    ]

    result = finalize_generated_section(text, links, uncited, status)

    assert result.text == "Tutoring improves accuracy (Smith, 2020)."
    assert "Motivation" not in result.text
    assert "Synthesis and Gaps" not in result.text
    assert result.stats["headings_removed"] == 2
    assert result.citation_links == [
        {**links[0], "paragraph_index": 0},
    ]


# --------------------------------------------------------------------------------------
# Integration: finalize_draft_document, the standalone heal's own path
# --------------------------------------------------------------------------------------


def test_finalize_draft_document_drops_a_non_opening_paragraph_with_no_surviving_link():
    """Dropping the sub-section's own only paragraph leaves its own
    heading with no body -- which is dropped too (`_drop_headings_with_no_body`),
    rather than shipping a heading over nothing."""
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
                            "sentence": "Verified claim (Jones, 2019).",
                            "keys": ["jones_2019"],
                            "citation_text": "(Jones, 2019)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": "Verified claim (Jones, 2019)."}],
            },
            {
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": "Sub-section"}],
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Sources were selected for their relevance."}
                ],
            },
        ],
    }
    status = {("Verified claim (Jones, 2019).", "Verified claim (Jones, 2019).", "jones_2019"): "verified"}
    claims = [
        ("Verified claim (Jones, 2019).", "jones_2019", "(Jones, 2019)",
         "Verified claim (Jones, 2019)."),
    ]

    new_content, _surviving_links, stats, _healed = finalize_draft_document(doc, claims, status)

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph"]
    assert stats["sentences_removed_dangling"] == 1


def test_finalize_draft_document_keeps_the_first_paragraph_even_with_no_citation():
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
                "content": [
                    {"type": "text", "text": "This review synthesises several themes."}
                ],
            },
        ],
    }

    new_content, _surviving_links, stats, _healed = finalize_draft_document(doc, [], {})

    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading", "paragraph"]
    assert stats["sentences_removed_dangling"] == 0


def test_finalize_draft_document_grants_only_one_opening_exemption_per_document():
    """A second level-2 heading must not re-arm rule (a)'s own
    opening-paragraph exemption for whatever paragraph follows it; `finalize_generated_
    section` and the checker's own rule 8a both grant exactly one exemption per
    document, so the heal must too. The second heading's own paragraph carries no
    citation and is not the document's own first paragraph, so it is dropped; the
    third, cited paragraph survives on its own citation."""
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Motivation"}],
            },
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Tutoring improves accuracy (Smith, 2020).",
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                        }
                    ]
                },
                "content": [
                    {"type": "text", "text": "Tutoring improves accuracy (Smith, 2020)."}
                ],
            },
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Design"}],
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Sources were selected for their relevance."}
                ],
            },
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Costs fell sharply (Jones, 2019).",
                            "keys": ["jones_2019"],
                            "citation_text": "(Jones, 2019)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": "Costs fell sharply (Jones, 2019)."}],
            },
        ],
    }
    status = {
        ("Tutoring improves accuracy (Smith, 2020).", "Tutoring improves accuracy (Smith, 2020).", "smith_2020"): "verified",
        ("Costs fell sharply (Jones, 2019).", "Costs fell sharply (Jones, 2019).", "jones_2019"): "verified",
    }
    claims = [
        ("Tutoring improves accuracy (Smith, 2020).", "smith_2020", "(Smith, 2020)",
         "Tutoring improves accuracy (Smith, 2020)."),
        ("Costs fell sharply (Jones, 2019).", "jones_2019", "(Jones, 2019)",
         "Costs fell sharply (Jones, 2019)."),
    ]

    new_content, _surviving_links, stats, _healed = finalize_draft_document(doc, claims, status)

    texts = [
        _extract_paragraph_text(n) for n in new_content["content"] if n["type"] == "paragraph"
    ]
    assert "Sources were selected for their relevance." not in texts
    assert stats["sentences_removed_dangling"] == 1


def _extract_paragraph_text(node):
    return "".join(
        child.get("text") or ""
        for child in node.get("content") or []
        if child.get("type") == "text"
    )


def test_finalize_draft_document_drops_a_dangling_opener_after_a_same_paragraph_removal():
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
                            "sentence": "Tutoring works well (Smith, 2020).",
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                        }
                    ]
                },
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Tutoring works well (Smith, 2020). However, the same authors "
                            "caution against overuse."
                        ),
                    }
                ],
            },
        ],
    }
    status = {("Tutoring works well (Smith, 2020).", "Tutoring works well (Smith, 2020).", "smith_2020"): "unsupported"}
    claims = [
        ("Tutoring works well (Smith, 2020).", "smith_2020", "(Smith, 2020)",
         "Tutoring works well (Smith, 2020)."),
    ]

    new_content, _surviving_links, stats, _healed = finalize_draft_document(doc, claims, status)

    # Both fragments of this one paragraph are gone -- the citation sentence to rule 2,
    # the dangling "However, the same authors ..." to rule (c), cascading from that
    # same removal -- so the paragraph rebuilds to empty text and is dropped exactly
    # like any other paragraph with nothing left in it, whether or not it is the
    # section's own opening paragraph (the opening exception only ever keeps NON-EMPTY
    # framing text that happens to carry no citation, never an empty paragraph).
    types = [n["type"] for n in new_content["content"]]
    assert types == ["heading"]
    assert stats["sentences_removed_unverified"] == 1
    assert stats["sentences_removed_dangling"] == 1


# --------------------------------------------------------------------------------------
# Property test: random paragraphs, random removals
# --------------------------------------------------------------------------------------

SEED = 20260912
CASES = 300

_CLAUSES = [
    "tutoring improved outcomes",
    "costs fell sharply",
    "engagement remained low",
    "the replication failed",
    "teachers reported gains",
    "revision behaviour shifted",
]
_SOURCES = [
    ("smith_2020", "Smith", "2020"),
    ("jones_2019", "Jones", "2019"),
    ("lee_2021", "Lee", "2021"),
]
_CONNECTIVES = ["However", "Moreover", "Crucially", "Notably", "Third", "Finally"]
_DEICTICS = [
    "The study's own authors caution against overuse",
    "The same authors caution against overuse",
    "This finding is important for the field",
    "These gaps motivate the present investigation",
]
_PLAIN_FRAMING = [
    "Sources were selected for their relevance to the topic",
    "The library is small and skewed toward qualitative studies",
    "Engagement research complicates any simple account",
]
_NUMBER_WORDS = ["One", "Two", "Three", "Four", "Five"]


def _make_fragment(rng, kind):
    """One (text, link_or_none, verdict_or_none, is_uncited_framing, announced) tuple
    for the property grammar below."""
    if kind == "verified":
        key, surname, year = rng.choice(_SOURCES)
        clause = rng.choice(_CLAUSES)
        text = f"{clause.capitalize()} ({surname}, {year})."
        return text, (key, f"({surname}, {year})"), "verified", False, None
    if kind == "removed":
        key, surname, year = rng.choice(_SOURCES)
        clause = rng.choice(_CLAUSES)
        text = f"{clause.capitalize()} ({surname}, {year})."
        return text, (key, f"({surname}, {year})"), "unsupported", False, None
    if kind == "plain_framing":
        text = f"{rng.choice(_PLAIN_FRAMING)}."
        return text, None, None, True, None
    if kind == "connective":
        text = f"{rng.choice(_CONNECTIVES)}, {rng.choice(_PLAIN_FRAMING).lower()}."
        return text, None, None, True, None
    if kind == "deictic":
        text = f"{rng.choice(_DEICTICS)}."
        return text, None, None, True, None
    if kind == "enum_opener":
        count = rng.randint(1, 4)
        word = _NUMBER_WORDS[count - 1] if count <= len(_NUMBER_WORDS) else str(count)
        text = f"{word} gaps emerge."
        return text, None, None, True, count
    raise AssertionError(kind)


_KINDS = ["verified", "removed", "plain_framing", "connective", "deictic", "enum_opener"]


def _build_case(rng):
    # Retried until every fragment's own text is unique in the case: two identical
    # sentences (e.g. the same deictic phrase drawn twice) would make the survival
    # check below -- ``fragment_text in result.text`` -- unable to tell one occurrence
    # from the other, which is a limitation of this test's own substring check, not of
    # the code under test.
    while True:
        n = rng.randint(2, 6)
        fragments = [_make_fragment(rng, rng.choice(_KINDS)) for _ in range(n)]
        texts = [f[0] for f in fragments]
        if len(set(texts)) == len(texts):
            break
    text = " ".join(texts)

    links = []
    status = {}
    uncited = []
    for fragment_text, link_info, verdict, is_framing, _announced in fragments:
        if link_info is not None:
            key, citation_text = link_info
            links.append(_link(fragment_text, [key], citation_text=citation_text))
            status[(fragment_text, fragment_text, key)] = verdict
        if is_framing:
            uncited.append({"paragraph_index": 0, "sentence": fragment_text, "tag": "framing"})
    return text, links, uncited, status, fragments


def _cases():
    rng = random.Random(SEED)
    return [_build_case(rng) for _ in range(CASES)]


def test_property_enumeration_opener_never_survives_short_of_its_count():
    for text, links, uncited, status, fragments in _cases():
        result = finalize_generated_section(text, links, uncited, status)
        for i, (fragment_text, _link_info, _verdict, _is_framing, announced) in enumerate(
            fragments
        ):
            if announced is None or fragment_text not in result.text:
                continue
            following_survivors = sum(
                1
                for later_text, *_rest in fragments[i + 1 :]
                if later_text in result.text
            )
            assert following_survivors >= announced, (text, result.text)


def test_property_no_dangling_connective_or_deictic_survives_without_a_real_antecedent():
    """Trouble for a later fragment is either (a) an earlier
    enumeration shortfall in this same paragraph (a genuine, permanent absence, so it
    keeps propagating past any number of later survivors), or (b) this fragment would
    otherwise be the very first one the paragraph delivers at all, and an ordinary rule
    removed something before it -- never a bare, unrelated removal once a real
    antecedent has already survived."""
    for text, links, uncited, status, fragments in _cases():
        result = finalize_generated_section(text, links, uncited, status)
        seen_survivor = False
        anything_removed = False
        enum_trouble = False
        for fragment_text, _link_info, _verdict, is_framing, announced in fragments:
            survives = fragment_text in result.text
            trouble = enum_trouble or (not seen_survivor and anything_removed)
            if is_framing and survives and trouble:
                assert not _opens_with_dangling_marker(fragment_text), (text, result.text)
            if survives:
                seen_survivor = True
            else:
                anything_removed = True
                if announced is not None:
                    enum_trouble = True


def test_property_a_non_opening_framing_only_paragraph_never_survives():
    """Rule (a): build a two-paragraph document, the second made entirely of framing
    sentences (no citation at all). Whatever the first paragraph is, the second must be
    entirely absent from the output unless it happens to carry a verified citation."""
    rng = random.Random(SEED + 1)
    for _ in range(CASES):
        # Retried until every fragment's own text is unique across both paragraphs, for
        # the same reason ``_build_case`` retries above.
        while True:
            opener_kind = rng.choice(["verified", "plain_framing"])
            opener_text, opener_link, opener_verdict, _is_framing, _announced = _make_fragment(
                rng, opener_kind
            )
            second_fragments = [
                _make_fragment(rng, rng.choice(["plain_framing", "connective", "deictic"]))
                for _ in range(rng.randint(1, 3))
            ]
            all_texts = [opener_text, *(f[0] for f in second_fragments)]
            if len(set(all_texts)) == len(all_texts):
                break
        second_text = " ".join(f[0] for f in second_fragments)
        full_text = f"{opener_text}\n\n{second_text}"

        links = []
        status = {}
        uncited = [
            {"paragraph_index": 1, "sentence": f[0], "tag": "framing"} for f in second_fragments
        ]
        if opener_link is not None:
            key, citation_text = opener_link
            links.append(_link(opener_text, [key], citation_text=citation_text,
                                paragraph_index=0))
            status[(opener_text, opener_text, key)] = opener_verdict

        result = finalize_generated_section(full_text, links, uncited, status)

        for fragment_text, *_rest in second_fragments:
            assert fragment_text not in result.text, (full_text, result.text)
