"""Deterministic renderer for the project's chosen citation style.

The writer and the claim verifier stay author-year; this module only renders the
project's chosen style at export and in paper chat, from the writer's own citation-link
map (draft path) or from `fulltext.citation_extents` + `fulltext.build_paper_lookup`
(chat path).
"""

import copy
from types import SimpleNamespace

from app.services.citation_render import (
    _format_entry,
    _is_balanced,
    render_answer,
    render_document,
)

ONE_PAPER = SimpleNamespace(
    authors=[{"name": "Jane Q. Smith"}, {"name": "Alice Jones"}],
    year=2020,
    title="Title of the paper",
    journal_name="Journal Name",
    doi="10.x",
)


def _doc_with_link(
    sentence: str, citation_text: str, keys: list[str], paragraph_text: str | None = None
):
    return {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": sentence, "citation_text": citation_text, "keys": keys}
                    ]
                },
                "content": [{"type": "text", "text": paragraph_text or sentence}],
            }
        ],
    }


# --------------------------------------------------------------------------------------
# Reference entry formatting, one style at a time
# --------------------------------------------------------------------------------------


def test_apa_entry():
    entry = _format_entry(None, ONE_PAPER, "APA")
    assert entry == (
        "Smith, J. Q., & Jones, A. (2020). Title of the paper. "
        "Journal Name. https://doi.org/10.x"
    )


def test_chicago_entry():
    paper = SimpleNamespace(
        authors=["Jane Q. Smith", "Alice Jones"],
        year=2020, title="Title of the paper", journal_name="Journal Name", doi="10.x",
    )
    entry = _format_entry(None, paper, "Chicago")
    assert entry == (
        'Smith, Jane Q., and Alice Jones. 2020. "Title of the paper." '
        "Journal Name. https://doi.org/10.x"
    )


def test_mla_entry_two_authors():
    paper = SimpleNamespace(
        authors=["Jane Q. Smith", "Alice Jones"],
        year=2020, title="Title of the paper", journal_name="Journal Name", doi="10.x",
    )
    entry = _format_entry(None, paper, "MLA")
    assert entry == (
        'Smith, Jane Q., and Alice Jones. "Title of the paper." '
        "Journal Name, 2020, https://doi.org/10.x."
    )


def test_mla_entry_three_authors_uses_et_al():
    paper = SimpleNamespace(
        authors=["Jane Q. Smith", "Alice Jones", "Bo Lee"],
        year=2020, title="Title of the paper", journal_name=None, doi=None,
    )
    entry = _format_entry(None, paper, "MLA")
    assert entry == 'Smith, Jane Q., et al. "Title of the paper." 2020.'


def test_harvard_entry():
    entry = _format_entry(None, ONE_PAPER, "Harvard")
    assert entry == (
        "Smith, J.Q. and Jones, A. (2020) 'Title of the paper', "
        "Journal Name. Available at: https://doi.org/10.x"
    )


def test_ieee_entry():
    entry = _format_entry(1, ONE_PAPER, "IEEE")
    assert entry == (
        '[1] J. Q. Smith and A. Jones, "Title of the paper," '
        "Journal Name, 2020. doi: 10.x."
    )


def test_vancouver_entry():
    entry = _format_entry(1, ONE_PAPER, "Vancouver")
    assert entry == "1. Smith JQ, Jones A. Title of the paper. Journal Name. 2020. doi:10.x."


def test_custom_style_behaves_as_apa():
    assert _format_entry(None, ONE_PAPER, "custom") == _format_entry(None, ONE_PAPER, "APA")


# --------------------------------------------------------------------------------------
# Missing metadata: a missing element is omitted, never
# invented; a missing year is "n.d." in author-year styles and dropped in numbered ones
# --------------------------------------------------------------------------------------


def test_missing_journal_omitted():
    paper = SimpleNamespace(
        authors=["Jane Smith"], year=2020, title="T", journal_name=None, doi="10.x"
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. (2020). T. https://doi.org/10.x"
    assert _format_entry(1, paper, "IEEE") == '[1] J. Smith, "T," 2020. doi: 10.x.'


def test_missing_doi_omitted():
    paper = SimpleNamespace(
        authors=["Jane Smith"], year=2020, title="T", journal_name="J", doi=None
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. (2020). T. J."
    assert _format_entry(1, paper, "Vancouver") == "1. Smith J. T. J. 2020."


def test_missing_year_is_nd_in_author_year_and_dropped_in_numbered():
    paper = SimpleNamespace(
        authors=["Jane Smith"], year=None, title="T", journal_name="J", doi="10.x"
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. (n.d.). T. J. https://doi.org/10.x"
    assert _format_entry(1, paper, "IEEE") == '[1] J. Smith, "T," J. doi: 10.x.'
    assert _format_entry(1, paper, "Vancouver") == "1. Smith J. T. J. doi:10.x."


# --------------------------------------------------------------------------------------
# Author counts and name shapes
# --------------------------------------------------------------------------------------


def test_one_author():
    paper = SimpleNamespace(
        authors=["Jane Smith"], year=2020, title="T", journal_name=None, doi=None
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. (2020). T."


def test_three_authors_apa_uses_oxford_ampersand():
    paper = SimpleNamespace(
        authors=["Jane Smith", "Alice Jones", "Bo Lee"],
        year=2020, title="T", journal_name=None, doi=None,
    )
    assert _format_entry(None, paper, "APA") == "Smith, J., Jones, A., & Lee, B. (2020). T."


def test_family_given_input():
    paper = SimpleNamespace(
        authors=["Smith, Jane Q."], year=2020, title="T", journal_name=None, doi=None
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. Q. (2020). T."


def test_given_family_input():
    paper = SimpleNamespace(
        authors=["Jane Q Smith"], year=2020, title="T", journal_name=None, doi=None
    )
    assert _format_entry(None, paper, "APA") == "Smith, J. Q. (2020). T."


def test_particle_surname():
    paper = SimpleNamespace(
        authors=["Teun van Dijk"], year=2015, title="Discourse and power",
        journal_name=None, doi=None,
    )
    assert _format_entry(None, paper, "APA") == "van Dijk, T. (2015). Discourse and power."
    assert _format_entry(1, paper, "Vancouver") == "1. van Dijk T. Discourse and power. 2015."


# --------------------------------------------------------------------------------------
# render_document: author-year styles never touch the text
# --------------------------------------------------------------------------------------


def test_author_year_style_leaves_text_byte_identical_and_appends_sorted_list():
    paper_b = SimpleNamespace(
        authors=["Bo Lee"], year=2019, title="B title", journal_name=None, doi=None
    )
    paper_a = SimpleNamespace(
        authors=["Amy Storch"], year=2018, title="A title", journal_name=None, doi=None
    )
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "Shown (Lee, 2019).", "citation_text": "(Lee, 2019)",
                         "keys": ["lee_2019"]},
                    ]
                },
                "content": [{"type": "text", "text": "Shown (Lee, 2019)."}],
            },
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "Also shown (Storch, 2018).",
                         "citation_text": "(Storch, 2018)", "keys": ["storch_2018"]},
                    ]
                },
                "content": [{"type": "text", "text": "Also shown (Storch, 2018)."}],
            },
        ],
    }
    result = render_document(doc, [paper_b, paper_a], "APA")
    assert result.numbered is False
    assert result.rendered == 0
    body = result.content["content"]
    assert body[0]["content"][0]["text"] == "Shown (Lee, 2019)."
    assert body[1]["content"][0]["text"] == "Also shown (Storch, 2018)."
    # sorted by family key: "Lee" before "Storch"
    assert result.references == [
        "Lee, B. (2019). B title.",
        "Storch, A. (2018). A title.",
    ]


def test_input_document_is_not_mutated():
    doc = _doc_with_link("Shown (Smith, 2020).", "(Smith, 2020)", ["smith_2020"])
    original = copy.deepcopy(doc)
    render_document(doc, [ONE_PAPER], "IEEE")
    assert doc == original


# --------------------------------------------------------------------------------------
# render_document: numbered styles
# --------------------------------------------------------------------------------------


def test_ieee_and_vancouver_number_in_first_appearance_order():
    paper_smith = SimpleNamespace(
        authors=["Jane Smith"], year=2020, title="T1", journal_name=None, doi=None
    )
    paper_jones = SimpleNamespace(
        authors=["Alice Jones"], year=2019, title="T2", journal_name=None, doi=None
    )
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "First (Jones, 2019) then (Smith, 2020).",
                         "citation_text": "(Jones, 2019)", "keys": ["jones_2019"]},
                        {"sentence": "First (Jones, 2019) then (Smith, 2020).",
                         "citation_text": "(Smith, 2020)", "keys": ["smith_2020"]},
                    ]
                },
                "content": [
                    {"type": "text", "text": "First (Jones, 2019) then (Smith, 2020)."}
                ],
            }
        ],
    }
    result = render_document(doc, [paper_smith, paper_jones], "IEEE")
    assert result.numbered is True
    assert result.content["content"][0]["content"][0]["text"] == "First [1] then [2]."
    assert result.rendered == 2
    assert result.found == 2


def test_multi_key_occurrence_renders_both_numbers():
    doc = _doc_with_link(
        "Shown (Storch, 2018; Lee, 2019).", "(Storch, 2018; Lee, 2019)",
        ["storch_2018", "lee_2019"],
    )
    storch = SimpleNamespace(
        authors=["Amy Storch"], year=2018, title="T1", journal_name=None, doi=None
    )
    lee = SimpleNamespace(authors=["Bo Lee"], year=2019, title="T2", journal_name=None, doi=None)
    result = render_document(doc, [storch, lee], "Vancouver")
    assert result.content["content"][0]["content"][0]["text"] == "Shown [1, 2]."
    assert result.rendered == 1
    assert result.found == 1


def test_semicolon_adjacent_occurrences_merge_before_replacement():
    """The chat path's `citation_extents` reconstructs "(Storch, 2018; Lee, 2019)" as two
    adjacent spans; they must merge into one rendered citation."""
    storch = SimpleNamespace(
        authors=["Amy Storch"], year=2018, title="T1", journal_name=None, doi=None
    )
    lee = SimpleNamespace(authors=["Bo Lee"], year=2019, title="T2", journal_name=None, doi=None)
    text = "This was shown (Storch, 2018; Lee, 2019) clearly."
    result = render_answer(text, [storch, lee], "IEEE")
    assert "[1], [2]" in result.text
    assert result.rendered == 1
    assert result.found == 1


def test_narrative_form_keeps_its_author():
    doc = _doc_with_link(
        "Smith et al. (2020) showed this.", "Smith et al. (2020)", ["smith_2020"]
    )
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == "Smith et al. [1] showed this."


def test_bare_parenthetical_replaced_whole():
    doc = _doc_with_link("Shown (Smith, 2020).", "(Smith, 2020)", ["smith_2020"])
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == "Shown [1]."


# --------------------------------------------------------------------------------------
# Draft-path edge cases
# --------------------------------------------------------------------------------------


def test_stale_link_is_ignored():
    """A link whose sentence no longer occurs in the paragraph is
    dropped outright: it is not counted in found, rendered or unresolved."""
    doc = _doc_with_link(
        sentence="A sentence that is no longer in the text.",
        citation_text="(Smith, 2020)",
        keys=["smith_2020"],
        paragraph_text="This paragraph was rewritten and no longer mentions the citation.",
    )
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 0
    assert result.rendered == 0
    assert result.unresolved == []
    assert result.content["content"][0]["content"][0]["text"] == (
        "This paragraph was rewritten and no longer mentions the citation."
    )


def test_unlinked_span_left_as_written_and_counted():
    """A surviving link with no keys ('any span with no map
    entry') is left exactly as written, counted in found and listed in unresolved."""
    doc = _doc_with_link("Shown (Smith, 2020).", "(Smith, 2020)", keys=[])
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Smith, 2020)"]
    assert result.content["content"][0]["content"][0]["text"] == "Shown (Smith, 2020)."


def test_citation_text_no_longer_verbatim_is_counted_unresolved():
    """The sentence still matches (normalised) but
    the citation_text itself no longer occurs verbatim."""
    doc = _doc_with_link(
        sentence="Shown (Smith,  2020) here.",
        citation_text="(Smith,  2020)",  # double space -- normalised-equal, not verbatim
        keys=["smith_2020"],
        paragraph_text="Shown (Smith, 2020) here.",
    )
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Smith,  2020)"]


def test_a_repeated_citation_is_counted_and_left_unresolved_when_only_one_is_linked():
    """`_draft_occurrences` dedupes a `citation_extents` span against every link's own
    position, not its normalised `citation_text`. A source cited twice in one paragraph
    with only the first occurrence linked would otherwise make the second occurrence's
    text match the first's and look "already accounted for", silently dropping it from
    both the rendering and the count. Deduping by position instead lets the second,
    unlinked occurrence surface on its own."""
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "A shown (Smith, 2020).", "citation_text": "(Smith, 2020)",
                         "keys": ["smith_2020"]}
                    ]
                },
                "content": [
                    {"type": "text", "text": "A shown (Smith, 2020). B shown (Smith, 2020)."}
                ],
            }
        ],
    }
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == (
        "A shown [1]. B shown (Smith, 2020)."
    )
    assert result.found == 2
    assert result.rendered == 1
    assert result.unresolved == ["(Smith, 2020)"]


def test_occurrence_straddling_two_text_nodes_is_unresolved():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "Shown (Smith, 2020).", "citation_text": "(Smith, 2020)",
                         "keys": ["smith_2020"]}
                    ]
                },
                "content": [
                    {"type": "text", "text": "Shown (Smith, "},
                    {"type": "text", "marks": [{"type": "italic"}], "text": "2020)."},
                ],
            }
        ],
    }
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Smith, 2020)"]
    assert result.content["content"][0]["content"] == [
        {"type": "text", "text": "Shown (Smith, "},
        {"type": "text", "marks": [{"type": "italic"}], "text": "2020)."},
    ]


def test_no_key_in_library_is_unresolved():
    doc = _doc_with_link("Shown (Nguyen, 2021).", "(Nguyen, 2021)", ["nguyen_2021"])
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Nguyen, 2021)"]
    assert result.references == []


def test_link_covered_occurrence_falls_back_to_the_audit_span_when_unbalanced():
    """The model's own
    `citation_text` runs one bracket past the citation's own close, into the enclosing
    "(as in ...)" parenthetical, so it is still found verbatim by `_find_all_spans` (the
    occurrence gets a real position) but the raw slice at that position is unbalanced.
    The key is absent from the library, so this reaches `unresolved`; what is reported is
    the audit's own span for the same citation, one bracket narrower and balanced,
    exactly what `fulltext.extract_claims_from_document` already reports for it."""
    sentence = "Findings (as in Smith (2020)) were mixed."
    doc = _doc_with_link(sentence, "Smith (2020))", ["nobody_1999"])
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.unresolved == ["Smith (2020)"]
    assert _is_balanced(result.unresolved[0])


def test_stale_links_own_text_is_trimmed_to_balanced():
    """Same paragraph, a link whose `citation_text` no longer occurs verbatim at all (a
    year that does not match what the draft says): stale, so it has no position to look
    an audit span up by. `_trim_to_balanced` is the only fallback available for that
    case, and it recovers the same shape (one bracket trimmed from the right). The
    paragraph's own citation, uncovered by this stale link, still surfaces separately
    through the ordinary unlinked-occurrence path and was already balanced before this
    fix; both entries in `unresolved` are bracket-balanced."""
    sentence = "Findings (as in Smith (2020)) were mixed."
    doc = _doc_with_link(sentence, "Smith (2019))", ["smith_2020"])
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.unresolved == ["Smith (2019)", "Smith (2020)"]
    assert all(_is_balanced(text) for text in result.unresolved)


def test_partially_resolvable_occurrence_left_as_written_and_unresolved():
    """A co-cited paper absent from the library
    must not be silently deleted from a grouped citation. When one key of a merged
    occurrence fails to resolve, the whole occurrence is left exactly as written, counted
    once, and reported in `unresolved` -- not rewritten to name only the source(s) that
    did resolve."""
    doc = _doc_with_link(
        "Shown (Smith, 2020; Nguyen, 2019).",
        "(Smith, 2020; Nguyen, 2019)",
        ["smith_2020", "nguyen_2019"],
    )
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == (
        "Shown (Smith, 2020; Nguyen, 2019)."
    )
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Smith, 2020; Nguyen, 2019)"]
    assert result.references == []


def test_render_answer_partially_resolvable_group_left_unresolved():
    """Same rule on the chat path, where the merge comes from adjacent semicolon spans
    rather than one stored link."""
    text = "This was shown (Smith, 2020; Nguyen, 2019) clearly."
    result = render_answer(text, [ONE_PAPER], "IEEE")
    assert "(Smith, 2020; Nguyen, 2019)" in result.text
    assert "[1]" not in result.text
    assert result.unresolved == ["(Smith, 2020; Nguyen, 2019)"]
    assert result.references == []


def test_unlinked_member_next_to_a_linked_one_in_the_same_group_is_not_deleted():
    """A keyless occurrence (a citation no
    link covers) sitting next to a linked one inside the same semicolon group could
    otherwise merge with it and vanish, because the merged occurrence's key count would
    happen to equal its resolved count. The unlinked member stays exactly as written,
    alongside the linked one's number."""
    jones = SimpleNamespace(
        authors=["Ann Jones"], year=2021, title="T2", journal_name=None, doi=None
    )
    doc = _doc_with_link(
        "Shown (Smith, 2020; Jones, 2021).", "Smith, 2020", ["smith_2020"],
        paragraph_text="Shown (Smith, 2020; Jones, 2021).",
    )
    result = render_document(doc, [ONE_PAPER, jones], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == "Shown ([1]; Jones, 2021)."
    assert result.found == 2
    assert result.rendered == 1
    assert len(result.unresolved) == 1
    assert "Jones, 2021" in result.unresolved[0]
    # Blocking the merge
    # leaves this member's own slice unbalanced ("; Jones, 2021)", no opening paren);
    # what is reported must still be bracket balanced.
    assert _is_balanced(result.unresolved[0])


def test_unlinked_third_member_of_a_group_is_not_deleted():
    jones = SimpleNamespace(
        authors=["Ann Jones"], year=2021, title="T2", journal_name=None, doi=None
    )
    lee = SimpleNamespace(authors=["Bo Lee"], year=2019, title="T3", journal_name=None, doi=None)
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": "Shown (Smith, 2020; Jones, 2021; Lee, 2019).",
                            "citation_text": "Smith, 2020; Jones, 2021",
                            "keys": ["smith_2020", "jones_2021"],
                        }
                    ]
                },
                "content": [
                    {"type": "text", "text": "Shown (Smith, 2020; Jones, 2021; Lee, 2019)."}
                ],
            }
        ],
    }
    result = render_document(doc, [ONE_PAPER, jones, lee], "IEEE")
    assert result.content["content"][0]["content"][0]["text"] == (
        "Shown ([1], [2]; Lee, 2019)."
    )
    assert len(result.unresolved) == 1
    assert "Lee, 2019" in result.unresolved[0]
    assert _is_balanced(result.unresolved[0])


def test_render_answer_two_keyless_group_members_are_each_reported_balanced():
    """Neither member of the group
    resolves (one is undated, the other names a paper absent from the library), so
    `_merge_semicolon_adjacent` blocks the merge on both sides and each member is
    reported on its own, rather than as unbalanced fragments like `['(Smith, n.d.',
    ' Nobody, 1999)']` or a deleted citation."""
    text = "Shown (Smith, n.d.; Nobody, 1999) here."
    result = render_answer(text, [ONE_PAPER], "IEEE")
    assert result.found == 2
    assert result.rendered == 0
    assert len(result.unresolved) == 2
    for entry in result.unresolved:
        assert _is_balanced(entry)
    assert any("Smith" in entry for entry in result.unresolved)
    assert any("Nobody, 1999" in entry for entry in result.unresolved)


def test_partial_resolution_with_no_existing_section_still_appends_the_list_in_apa():
    """Appending the generated list and dropping the
    document's own existing reference section are two different decisions. A draft with
    no reference section of its own -- every draft the writer produces -- has nothing
    for `unresolved` to protect, so the generated list must still be appended even
    though one citation stayed unresolved."""
    jones = SimpleNamespace(
        authors=["Ann Jones"], year=2021, title="U. K.", journal_name=None, doi=None
    )
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "A shown (Smith, 2020)."}]},
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "B shown (Jones, 2021).", "citation_text": "(Jones, 2021)",
                         "keys": ["jones_2021"]}
                    ]
                },
                "content": [{"type": "text", "text": "B shown (Jones, 2021)."}],
            },
        ],
    }
    result = render_document(doc, [ONE_PAPER, jones], "APA")
    assert result.unresolved == ["(Smith, 2020)"]
    assert result.references == [_format_entry(None, jones, "APA")]
    nodes = result.content["content"]
    assert nodes[0]["content"][0]["text"] == "A shown (Smith, 2020)."
    assert nodes[1]["content"][0]["text"] == "B shown (Jones, 2021)."
    headings = [n for n in nodes if n["type"] == "heading"]
    assert len(headings) == 1
    assert headings[0]["content"][0]["text"] == "References"
    assert nodes[-1]["content"][0]["text"] == result.references[0]


def test_partial_resolution_with_no_existing_section_still_appends_the_list_in_ieee():
    """Same as the APA test above, on the numbered path: the citation that did resolve
    is replaced with its number, and the generated list naming it is still appended even
    though the other citation in the draft is left unresolved."""
    jones = SimpleNamespace(
        authors=["Ann Jones"], year=2021, title="U. K.", journal_name=None, doi=None
    )
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "A shown (Smith, 2020)."}]},
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "B shown (Jones, 2021).", "citation_text": "(Jones, 2021)",
                         "keys": ["jones_2021"]}
                    ]
                },
                "content": [{"type": "text", "text": "B shown (Jones, 2021)."}],
            },
        ],
    }
    result = render_document(doc, [ONE_PAPER, jones], "IEEE")
    assert result.unresolved == ["(Smith, 2020)"]
    nodes = result.content["content"]
    assert nodes[0]["content"][0]["text"] == "A shown (Smith, 2020)."
    assert nodes[1]["content"][0]["text"] == "B shown [1]."
    assert result.references == ['[1] A. Jones, "U. K.," 2021.']
    headings = [n for n in nodes if n["type"] == "heading"]
    assert len(headings) == 1
    assert nodes[-1]["content"][0]["text"] == result.references[0]


def test_partial_resolution_keeps_the_existing_reference_section():
    """The generated list only ever covers the
    citations it resolved. Dropping the author's own section whenever *any* citation
    resolved, rather than only when *every* one did, would delete the reference entry
    for a citation that stayed unresolved, with nothing put in its place. The existing
    section survives, untouched, until nothing is left unresolved."""
    jones = SimpleNamespace(
        authors=["Ann Jones"], year=2021, title="U. K.", journal_name=None, doi=None
    )
    doc = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "A shown (Smith, 2020)."}]},
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "B shown (Jones, 2021).", "citation_text": "(Jones, 2021)",
                         "keys": ["jones_2021"]}
                    ]
                },
                "content": [{"type": "text", "text": "B shown (Jones, 2021)."}],
            },
            {"type": "heading", "attrs": {"level": 2},
             "content": [{"type": "text", "text": "References"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Smith, J. (2020). T. J."}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Jones, A. (2021). U. K."}]},
        ],
    }
    result = render_document(doc, [ONE_PAPER, jones], "APA")
    assert result.unresolved == ["(Smith, 2020)"]
    assert result.content == doc
    texts = [n["content"][0]["text"] for n in result.content["content"] if n["type"] != "heading"]
    assert "Smith, J. (2020). T. J." in texts
    assert "Jones, A. (2021). U. K." in texts


def test_no_resolvable_citations_restores_original_reference_section():
    """When nothing resolves, `references` comes
    back empty and the document's own existing reference section (and anything after it)
    must survive untouched."""
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Shown before (Smith, 2020)."}
                ],
            },
            {"type": "heading", "attrs": {"level": 2},
             "content": [{"type": "text", "text": "References"}]},
            {"type": "paragraph",
             "content": [{"type": "text", "text": "Smith, J. (2020). T. J."}]},
        ],
    }
    result = render_document(doc, [ONE_PAPER], "APA")
    assert result.references == []
    assert result.content == doc


def test_unlinked_citation_with_no_link_at_all_is_counted_unresolved():
    """A citation with no `citationLinks` entry
    at all (one from before citation links were added, or one typed into the editor
    after generation) is still found via `fulltext.citation_extents`, left untouched,
    and reported in `unresolved`."""
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Shown (Smith, 2020) here."}],
            }
        ],
    }
    result = render_document(doc, [ONE_PAPER], "IEEE")
    assert result.found == 1
    assert result.rendered == 0
    assert result.unresolved == ["(Smith, 2020)"]
    assert result.content["content"][0]["content"][0]["text"] == "Shown (Smith, 2020) here."


def test_existing_reference_section_replaced_not_duplicated():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {"sentence": "Shown (Smith, 2020).", "citation_text": "(Smith, 2020)",
                         "keys": ["smith_2020"]}
                    ]
                },
                "content": [{"type": "text", "text": "Shown (Smith, 2020)."}],
            },
            {"type": "heading", "attrs": {"level": 2},
             "content": [{"type": "text", "text": "References"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Old stale entry."}]},
            {"type": "heading", "attrs": {"level": 1},
             "content": [{"type": "text", "text": "Appendix"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Appendix text."}]},
        ],
    }
    result = render_document(doc, [ONE_PAPER], "APA")
    nodes = result.content["content"]
    headings = [n for n in nodes if n["type"] == "heading"]
    assert len(headings) == 2  # "Appendix" plus the one new "References"
    texts = [n["content"][0]["text"] for n in nodes]
    assert "Old stale entry." not in texts
    assert "Appendix text." in texts
    assert texts.count("References") == 1
    assert texts[-1] == result.references[0]


def test_no_papers_returns_document_unchanged():
    doc = _doc_with_link("Shown (Smith, 2020).", "(Smith, 2020)", ["smith_2020"])
    result = render_document(doc, [], "IEEE")
    assert result.content == doc
    assert result.references == []
    assert result.found == 0


def test_empty_content_returns_unchanged():
    result = render_document(None, [ONE_PAPER], "IEEE")
    assert result.content is None
    assert result.references == []


# --------------------------------------------------------------------------------------
# render_answer (chat path)
# --------------------------------------------------------------------------------------


def test_render_answer_author_year_appends_references_only():
    text = "This claim is supported (Smith, 2020) and by Jones's other work."
    result = render_answer(text, [ONE_PAPER], "APA")
    assert result.text.startswith(text)
    assert result.numbered is False
    assert result.rendered == 0
    assert "## References" in result.text
    assert result.references == [
        "Smith, J. Q., & Jones, A. (2020). Title of the paper. "
        "Journal Name. https://doi.org/10.x"
    ]


def test_render_answer_unresolved_citation_survives_verbatim():
    text = "An unsupported claim (Nguyen, 2021)."
    result = render_answer(text, [ONE_PAPER], "IEEE")
    assert "(Nguyen, 2021)" in result.text
    assert result.unresolved == ["(Nguyen, 2021)"]
    assert result.references == []


def test_render_answer_no_papers_returns_text_unchanged():
    result = render_answer("Some answer (Smith, 2020).", [], "IEEE")
    assert result.text == "Some answer (Smith, 2020)."
    assert result.references == []


def test_render_answer_strips_models_own_reference_section_before_appending():
    """Chat rule 5 tells the model not to write
    its own reference section; when it does anyway, the answer must end with one
    reference section (ours), not two."""
    text = (
        "This claim is supported (Smith, 2020).\n\n"
        "## References\n"
        "- Smith, J. Q. (2020) some other style entirely."
    )
    result = render_answer(text, [ONE_PAPER], "APA")
    assert result.text.count("## References") == 1


def test_render_answer_keeps_a_section_after_the_models_own_reference_heading():
    """Dropping everything to
    the end of the answer once the model's own reference heading is found would delete
    a genuine section (such as "## Limitations") that happens to follow it. Only the
    reference block itself -- up to the next heading -- is dropped."""
    text = (
        "This claim is supported (Smith, 2020).\n\n"
        "## References\n"
        "- Smith, J. Q. (2020) some other style entirely.\n\n"
        "## Limitations\n"
        "Only one study."
    )
    result = render_answer(text, [ONE_PAPER], "APA")
    assert result.text.count("## References") == 1
    assert "## Limitations" in result.text
    assert "Only one study." in result.text
    assert "some other style entirely" not in result.text
    assert "some other style entirely" not in result.text
    assert "This claim is supported (Smith, 2020)." in result.text


# --------------------------------------------------------------------------------------
# The three fulltext delegations must never drift from their private originals.
# --------------------------------------------------------------------------------------


def test_citation_extents_matches_audit_span_units():
    from app.services.fulltext import _audit_span_units, citation_extents

    text = "Shown (Smith, 2020) and Jones et al. (2019) too, plus (Lee, 2018; Bo, 2017)."
    assert citation_extents(text) == [
        (unit.start, unit.end, unit.rendered) for unit in _audit_span_units(text)
    ]


def test_build_paper_lookup_and_reference_heading_are_the_private_originals():
    from app.services.fulltext import (
        _REFERENCE_LIST_HEADING_RE,
        REFERENCE_LIST_HEADING_RE,
        _build_paper_lookup,
        build_paper_lookup,
    )

    assert build_paper_lookup is _build_paper_lookup
    assert REFERENCE_LIST_HEADING_RE is _REFERENCE_LIST_HEADING_RE
