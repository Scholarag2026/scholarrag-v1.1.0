"""Pure-function tests for the write job's finalize-time Tiptap construction and
section merge: _build_section_tiptap_nodes turns a
finalized section's text into Tiptap nodes with citationLinks repaired and a
sectionType-tagged heading; _merge_section_into_draft splices those nodes into an
existing draft, replacing an already-present section in place and leaving every other
section untouched. _with_verification_provenance folds a _verify_claims aggregate
into the section's own running provenance totals.
"""

from app.services.writing import (
    _build_section_tiptap_nodes,
    _merge_section_into_draft,
    _strip_locked_block_findings,
    _with_verification_provenance,
)


def test_build_section_tiptap_nodes_heading_is_tagged_with_section_type():
    nodes = _build_section_tiptap_nodes("methods", "Methods", "Body text.", [])

    assert nodes[0]["type"] == "heading"
    assert nodes[0]["attrs"]["sectionType"] == "methods"
    assert nodes[0]["content"][0]["text"] == "Methods"
    assert nodes[1] == {"type": "paragraph", "content": [{"type": "text", "text": "Body text."}]}


def test_build_section_tiptap_nodes_attaches_citation_links_by_paragraph_index():
    text = "First paragraph (Smith, 2020).\n\nSecond paragraph (Jones, 2019)."
    links = [
        {
            "paragraph_index": 1,
            "sentence": "Second paragraph (Jones, 2019).",
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
            "evidence_ids": ["e1"],
            "proposition": "Second paragraph",
        }
    ]

    nodes = _build_section_tiptap_nodes("methods", "Methods", text, links)

    paragraphs = [n for n in nodes if n["type"] == "paragraph"]
    assert "attrs" not in paragraphs[0]
    assert paragraphs[1]["attrs"]["citationLinks"] == [
        {
            "sentence": "Second paragraph (Jones, 2019).",
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
            "evidence_ids": ["e1"],
            "proposition": "Second paragraph",
        }
    ]


def test_build_section_tiptap_nodes_round_trips_proposition_for_extraction():
    """`extract_claims_from_document` narrows a claim's own text to
    `attrs.citationLinks[].proposition` (fulltext.py, `_assign_links_to_units`).
    `_build_section_tiptap_nodes` must persist that field on the saved draft, or a
    grouped-citation sentence is sent to the verifier whole, once per citation, which
    is a failure mode behind a large share of unsupported claims on grouped citations.
    This is the saved-draft path the standalone action, the demo runner and
    `claim_report.json` all read; the write job's own in-memory path already worked."""
    from app.services.fulltext import extract_claims_from_document

    text = "Tutoring boosts outcomes (Smith, 2020), while costs fall (Jones, 2019)."
    links = [
        {
            "paragraph_index": 0,
            "sentence": text,
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
            "proposition": "Tutoring boosts outcomes",
        },
        {
            "paragraph_index": 0,
            "sentence": text,
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
            "proposition": "costs fall",
        },
    ]

    nodes = _build_section_tiptap_nodes("methods", "Methods", text, links)
    doc = {"type": "doc", "content": nodes}

    claims, _coverage = extract_claims_from_document(doc, {})

    claim_text_by_key = {key: claim_text for claim_text, key, _cx, _sentence in claims}
    assert claim_text_by_key["smith_2020"] == "Tutoring boosts outcomes"
    assert claim_text_by_key["jones_2019"] == "costs fall"


def test_build_section_tiptap_nodes_renders_an_inline_markdown_heading():
    text = "## Sub-theme\n\nBody paragraph."

    nodes = _build_section_tiptap_nodes("literature_review", "Literature Review", text, [])

    sub_heading = nodes[1]
    assert sub_heading["type"] == "heading"
    assert sub_heading["attrs"]["level"] == 2
    assert sub_heading["content"][0]["text"] == "Sub-theme"
    assert "sectionType" not in sub_heading["attrs"]


def test_build_section_tiptap_nodes_renders_a_whole_line_bold_run_as_a_heading():
    """The writer sometimes marks a sub-heading with
    a whole line of ``**bold**`` instead of a markdown ``###`` line. That line must become
    a real heading node, not a paragraph whose only text is the literal asterisks."""
    text = "**Sub-theme**\n\nBody paragraph."

    nodes = _build_section_tiptap_nodes("literature_review", "Literature Review", text, [])

    sub_heading = nodes[1]
    assert sub_heading["type"] == "heading"
    assert sub_heading["attrs"]["level"] == 3
    assert sub_heading["content"][0]["text"] == "Sub-theme"
    assert "sectionType" not in sub_heading["attrs"]
    assert nodes[2] == {
        "type": "paragraph", "content": [{"type": "text", "text": "Body paragraph."}]
    }


def test_build_section_tiptap_nodes_leaves_a_bold_claim_with_a_citation_as_a_paragraph():
    """A whole-line bold run that carries a citation, or ends in
    sentence punctuation, is a claim sentence the writer happened to bold -- not a
    sub-heading -- and must render as an ordinary paragraph node, with its own
    citationLinks attached, so the citation-status rules and the coverage-gap scan both
    see it exactly as they see any other body sentence."""
    text = "**Direct corrections outperformed metalinguistic codes by 32% (Smith, 2020).**"
    links = [
        {
            "paragraph_index": 0,
            "sentence": text,
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
        }
    ]

    nodes = _build_section_tiptap_nodes("literature_review", "Literature Review", text, links)

    paragraph = nodes[1]
    assert paragraph["type"] == "paragraph"
    assert paragraph["content"][0]["text"] == text
    assert paragraph["attrs"]["citationLinks"][0]["keys"] == ["smith_2020"]


def test_build_section_tiptap_nodes_leaves_a_bold_sentence_ending_in_a_period_as_a_paragraph():
    """The sentence-punctuation half of the same guard, with no citation at all."""
    text = "**This reads like a finding, not a heading.**"

    nodes = _build_section_tiptap_nodes("literature_review", "Literature Review", text, [])

    assert nodes[1] == {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def test_build_section_tiptap_nodes_leaves_a_line_with_two_bold_runs_as_a_paragraph():
    """A line with more than one bold run is running prose, not a lone bold-line heading
    -- the narrow regex must not swallow the text between the two runs."""
    text = "**Term A** and **Term B** both matter."

    nodes = _build_section_tiptap_nodes("literature_review", "Literature Review", text, [])

    assert nodes[1] == {
        "type": "paragraph",
        "content": [{"type": "text", "text": "**Term A** and **Term B** both matter."}],
    }


def test_merge_section_into_draft_appends_to_an_empty_draft():
    nodes = [{"type": "heading", "attrs": {"sectionType": "methods"}, "content": []}]

    merged = _merge_section_into_draft(None, "methods", nodes)

    assert merged == {"type": "doc", "content": nodes}


def test_merge_section_into_draft_appends_a_new_section_after_existing_ones():
    existing = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"sectionType": "introduction"}, "content": []},
            {"type": "paragraph", "content": [{"type": "text", "text": "Intro body."}]},
        ],
    }
    new_nodes = [{"type": "heading", "attrs": {"sectionType": "methods"}, "content": []}]

    merged = _merge_section_into_draft(existing, "methods", new_nodes)

    assert merged["content"] == existing["content"] + new_nodes


def test_merge_section_into_draft_replaces_an_existing_section_in_place():
    existing = {
        "type": "doc",
        "content": [
            {"type": "heading", "attrs": {"sectionType": "introduction"}, "content": []},
            {"type": "paragraph", "content": [{"type": "text", "text": "Intro body."}]},
            {"type": "heading", "attrs": {"sectionType": "methods"}, "content": []},
            {"type": "paragraph", "content": [{"type": "text", "text": "Old methods body."}]},
            {"type": "heading", "attrs": {"sectionType": "results"}, "content": []},
            {"type": "paragraph", "content": [{"type": "text", "text": "Results body."}]},
        ],
    }
    new_nodes = [
        {"type": "heading", "attrs": {"sectionType": "methods"}, "content": []},
        {"type": "paragraph", "content": [{"type": "text", "text": "New methods body."}]},
    ]

    merged = _merge_section_into_draft(existing, "methods", new_nodes)

    assert merged["content"] == [
        existing["content"][0],
        existing["content"][1],
        new_nodes[0],
        new_nodes[1],
        existing["content"][4],
        existing["content"][5],
    ]


def test_merge_section_into_draft_keeps_untagged_content_after_the_replaced_section():
    """The scan that ends a replaced section must stop at the next
    heading of ANY kind -- a heading the user typed, a section written before section
    tagging existed, a hand-written References list -- not only at a heading this
    feature's own code tagged with `attrs.sectionType`. Only the tagged Literature
    Review section is replaced; the untagged Discussion heading, the user's own
    paragraph under it, and the untagged References list are all preserved."""
    existing = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2, "sectionType": "literature_review"},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "old paragraph"}]},
            {"type": "heading", "content": [{"type": "text", "text": "Discussion"}]},
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "My own hand-written discussion."}],
            },
            {"type": "heading", "content": [{"type": "text", "text": "References"}]},
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "Smith, J. (2020). A paper."}],
            },
        ],
    }
    new_nodes = [
        {
            "type": "heading",
            "attrs": {"level": 2, "sectionType": "literature_review"},
            "content": [{"type": "text", "text": "Literature Review"}],
        },
        {"type": "paragraph", "content": [{"type": "text", "text": "New AI text (Smith, 2020)."}]},
    ]

    merged = _merge_section_into_draft(existing, "literature_review", new_nodes)

    assert merged["content"] == new_nodes + existing["content"][2:]


def test_merge_section_into_draft_still_swallows_a_nested_subheading_of_the_same_section():
    """A deeper heading (level 3) written inside the section being replaced is still
    treated as part of that section's own content, not as a boundary -- only a heading
    at the replaced section's own level or higher (a smaller or equal level number)
    ends the scan."""
    existing = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2, "sectionType": "literature_review"},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": "Sub-theme"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "old sub-theme body"}]},
            {"type": "heading", "content": [{"type": "text", "text": "Discussion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "kept"}]},
        ],
    }
    new_nodes = [
        {
            "type": "heading",
            "attrs": {"level": 2, "sectionType": "literature_review"},
            "content": [{"type": "text", "text": "Literature Review"}],
        },
        {"type": "paragraph", "content": [{"type": "text", "text": "New text."}]},
    ]

    merged = _merge_section_into_draft(existing, "literature_review", new_nodes)

    assert merged["content"] == new_nodes + existing["content"][3:]


def test_merge_section_into_draft_drops_a_second_heading_of_the_same_section_type():
    """A document holding two headings tagged with
    the same `sectionType` -- a shape the backend never creates itself, but a
    hand-edited or imported draft could -- must not end up with the freshly generated
    section twice. Only the first heading is replaced; the second is dropped along
    with its own content."""
    existing = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2, "sectionType": "literature_review"},
                "content": [{"type": "text", "text": "Literature Review"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "old paragraph one"}]},
            {
                "type": "heading",
                "attrs": {"level": 2, "sectionType": "literature_review"},
                "content": [{"type": "text", "text": "Literature Review (duplicate)"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "old paragraph two"}]},
            {"type": "heading", "content": [{"type": "text", "text": "Discussion"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "kept"}]},
        ],
    }
    new_nodes = [
        {
            "type": "heading",
            "attrs": {"level": 2, "sectionType": "literature_review"},
            "content": [{"type": "text", "text": "Literature Review"}],
        },
        {"type": "paragraph", "content": [{"type": "text", "text": "New text."}]},
    ]

    merged = _merge_section_into_draft(existing, "literature_review", new_nodes)

    assert merged["content"] == new_nodes + existing["content"][4:]


def test_strip_locked_block_findings_drops_a_locked_sentence_flagged_as_finding():
    """A user's own locked paragraph, reproduced verbatim by
    the model inside the generated section, must never be removed by finalize's
    uncited-"finding" rule just because it carries no citation -- the user wrote it,
    not the model."""
    locked_sentence = "Tutoring is the single most effective intervention."
    locked_blocks = [{"position": 0, "text": locked_sentence}]
    uncited = [
        {"paragraph_index": 0, "sentence": locked_sentence, "tag": "finding"},
        {
            "paragraph_index": 1,
            "sentence": "This unrelated finding has no citation either.",
            "tag": "finding",
        },
    ]

    kept = _strip_locked_block_findings(uncited, locked_blocks)

    assert kept == [uncited[1]]


def test_strip_locked_block_findings_is_a_no_op_with_no_locked_blocks():
    uncited = [{"paragraph_index": 0, "sentence": "A finding.", "tag": "finding"}]

    assert _strip_locked_block_findings(uncited, None) == uncited
    assert _strip_locked_block_findings(uncited, []) == uncited


def test_with_verification_provenance_folds_calls_and_tokens():
    provenance = {"total_calls": 2, "total_input_tokens": 1000, "total_output_tokens": 200}
    verify_provenance = {"calls": 3, "input_tokens": 50, "output_tokens": 10}

    folded = _with_verification_provenance(provenance, verify_provenance)

    assert folded["total_calls"] == 5
    assert folded["total_input_tokens"] == 1050
    assert folded["total_output_tokens"] == 210
    assert folded["verification_calls"] == [verify_provenance]


def test_with_verification_provenance_appends_a_second_pass_without_double_counting():
    provenance = {"total_calls": 2, "total_input_tokens": 1000, "total_output_tokens": 200}
    first = {"calls": 2, "input_tokens": 40, "output_tokens": 8}
    second = {"calls": 1, "input_tokens": 10, "output_tokens": 2}

    folded = _with_verification_provenance(provenance, first)
    folded = _with_verification_provenance(folded, second)

    assert folded["total_calls"] == 2 + 2 + 1
    assert folded["total_input_tokens"] == 1000 + 40 + 10
    assert folded["total_output_tokens"] == 200 + 8 + 2
    assert folded["verification_calls"] == [first, second]


def test_with_verification_provenance_a_zero_call_pass_never_changes_totals():
    """The deterministic no_full_text path never calls the model, so its own aggregate
    provenance has calls=0 and no token fields -- folding it must be a pure no-op on
    the running totals, only appending the record itself."""
    provenance = {"total_calls": 1, "total_input_tokens": 500, "total_output_tokens": 100}
    empty = {"calls": 0, "input_tokens": None, "output_tokens": None}

    folded = _with_verification_provenance(provenance, empty)

    assert folded["total_calls"] == 1
    assert folded["total_input_tokens"] == 500
    assert folded["total_output_tokens"] == 100
