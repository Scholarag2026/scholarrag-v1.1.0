"""Evidence acceptance.

Code, not the model, decides which of the analysis agent's proposed evidence items are
accepted: verbatim, complete-sentence, at least 8 words. Pure-function tests here; the
DB-backed storage (``replace_paper_evidence``) and the end-to-end grounding pass are
covered in ``test_writing_grounding.py`` and ``test_deep_analysis_concurrency.py``.

What "verbatim" and "complete sentence" mean is deliberately wide, based on measuring on
the twelve demo seed papers that most of what the model proposed was a genuine copy from
the paper that a narrower acceptance rule could not recognise: the comparison folds
PDF-extraction typography on both sides, the quote is looked for in every chunk of the
paper rather than only the one the model named, and a span is accepted when it starts at
a sentence start and ends at a sentence end even where the extracted text glues a
heading, a table row or a line-number gutter onto the sentence in front of it.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.services.deep_analysis import (  # noqa: E402
    MIN_EVIDENCE_WORDS,
    analysis_prompt_version,
    fold_for_match,
    normalise_concepts,
    normalise_source_text,
    normalise_ws,
    quote_is_complete_sentences,
    validate_evidence_items,
)

CHUNKS = [
    {
        "text": (
            "Tutoring improved outcomes for most students in the sample. "
            "The effect was strongest for first-year undergraduates."
        ),
        "section": "Results",
    },
    {"text": "Some other chunk entirely, about method."},
]


def test_normalise_ws_collapses_all_whitespace():
    assert normalise_ws("Tutoring   improved\noutcomes.  ") == "Tutoring improved outcomes."
    assert normalise_ws(None) == ""


def test_normalise_concepts_lowercases_trims_and_dedupes():
    assert normalise_concepts([" Tutoring ", "tutoring", "Sample Size", ""]) == [
        "tutoring", "sample size",
    ]
    assert normalise_concepts(None) == []


def test_quote_is_complete_sentences_accepts_one_whole_sentence():
    chunk_text = CHUNKS[0]["text"]
    quote = "Tutoring improved outcomes for most students in the sample."
    assert quote_is_complete_sentences(quote, chunk_text) is True


def test_quote_is_complete_sentences_accepts_two_consecutive_sentences():
    chunk_text = CHUNKS[0]["text"]
    quote = (
        "Tutoring improved outcomes for most students in the sample. "
        "The effect was strongest for first-year undergraduates."
    )
    assert quote_is_complete_sentences(quote, chunk_text) is True


def test_quote_is_complete_sentences_rejects_a_fragment():
    chunk_text = CHUNKS[0]["text"]
    quote = "improved outcomes for most students in the sample."
    assert quote_is_complete_sentences(quote, chunk_text) is False


def test_quote_is_complete_sentences_rejects_a_quote_absent_from_the_chunk():
    assert quote_is_complete_sentences("Nothing like this is here.", CHUNKS[0]["text"]) is False


def test_quote_is_complete_sentences_rejects_empty():
    assert quote_is_complete_sentences("", CHUNKS[0]["text"]) is False


def test_validate_evidence_items_accepts_a_verbatim_complete_sentence():
    items = [{
        "quote": "Tutoring improved outcomes for most students in the sample.",
        "chunk_index": 0, "finding": "Tutoring helps.", "kind": "finding",
        "origin": "own", "concepts": ["Tutoring "],
    }]
    accepted, rejected = validate_evidence_items(items, CHUNKS)
    assert rejected == 0
    assert accepted == [{
        "quote": "Tutoring improved outcomes for most students in the sample.",
        "chunk_index": 0, "section": "Results", "finding": "Tutoring helps.",
        "kind": "finding", "origin": "own", "concepts": ["tutoring"],
    }]


def test_validate_evidence_items_rejects_a_quote_under_the_minimum_word_count():
    short_quote = " ".join(["word"] * (MIN_EVIDENCE_WORDS - 1))
    items = [{
        "quote": short_quote, "chunk_index": 0, "finding": "x", "kind": "finding",
        "origin": "own", "concepts": [],
    }]
    accepted, rejected = validate_evidence_items(items, CHUNKS)
    assert accepted == []
    assert rejected == 1


def test_validate_evidence_items_relocates_an_out_of_range_chunk_index():
    """On the twelve seed papers the model's ``chunk_index`` was wrong for 35 of
    78 proposed items, including every item of a paper stored as a single chunk, because
    a 76,000-character chunk gives it nothing to count. The index is a hint now: the
    quote is looked for in every chunk, and the chunk code found it in is what gets
    stored."""
    items = [{
        "quote": "Tutoring improved outcomes for most students in the sample.",
        "chunk_index": 5, "finding": "x", "kind": "finding",
        "origin": "own", "concepts": [],
    }]
    accepted, rejected = validate_evidence_items(items, CHUNKS)
    assert rejected == 0
    assert len(accepted) == 1
    assert accepted[0]["chunk_index"] == 0
    assert accepted[0]["section"] == "Results"


def test_validate_evidence_items_rejects_a_quote_not_verbatim_in_the_named_chunk():
    items = [{
        "quote": "This sentence never appears anywhere in the chunks at all today.",
        "chunk_index": 0, "finding": "x", "kind": "finding",
        "origin": "own", "concepts": [],
    }]
    accepted, rejected = validate_evidence_items(items, CHUNKS)
    assert accepted == []
    assert rejected == 1


def test_validate_evidence_items_tolerates_missing_or_none_items():
    assert validate_evidence_items(None, CHUNKS) == ([], 0)
    assert validate_evidence_items([], CHUNKS) == ([], 0)


def test_validate_evidence_items_counts_accepted_and_rejected_together():
    items = [
        {
            "quote": "Tutoring improved outcomes for most students in the sample.",
            "chunk_index": 0, "finding": "a", "kind": "finding",
            "origin": "own", "concepts": ["tutoring"],
        },
        {"quote": "Too short.", "chunk_index": 0, "finding": "b", "kind": "finding",
         "origin": "own", "concepts": []},
    ]
    accepted, rejected = validate_evidence_items(items, CHUNKS)
    assert len(accepted) == 1
    assert rejected == 1


# --------------------------------------------------------------------------------------
# The shared fold, and what the widened acceptance rules do and do not accept.
# Every chunk below is modelled on a real artefact measured in the twelve seed papers'
# stored `fulltext_chunks`.
# --------------------------------------------------------------------------------------

TYPOGRAPHY_CHUNK = [{
    "section": "Findings",
    "text": (
        "The teachers’ own ﬁndings about the drafts were clear to every reader. "
        "Students used the feed-\nback to revise their drafts and their scores rose."
    ),
}]

HEADING_CHUNK = [{
    "section": "Method",
    "text": (
        "Table 1 shows their background information. 3.2 Procedures "
        "Because data collection began in the middle of the semester, implementing the "
        "tool into those courses was not practical. "
        "Therefore, the participants were given a hypothetical scenario."
    ),
}]

TABLE_CHUNK = [{
    "section": "Findings",
    "text": (
        "Revision operations 2nd draft 3rd draft Zero correction 2.6% 2.1% "
        "Total 100% 100% "
        "We found that the students conducted all seven types of revision operations in "
        "their drafts. For instance, they focused more heavily on addressing errors."
    ),
}]

GUTTER_CHUNK = [{
    "section": "Discussion",
    "text": (
        "and incorporate it into their drafts. 1 2 3 4 5 6 7 8 9 25 "
        "More importantly, different feedback sources tend to elicit different "
        "engagement styles. This is particularly obvious in the case of one learner."
    ),
}]

ATTRIBUTION_CHUNK = [{
    "section": "Findings",
    "text": (
        "Figure 4, from the automated record, shows that Flora resubmitted her essay 13 "
        "times, and the score rose from 79 to 90 (100 as the full score)."
    ),
}]


def _one_item(quote: str, chunk_index: int = 0) -> list[dict]:
    return [{
        "quote": quote, "chunk_index": chunk_index, "finding": "f", "kind": "finding",
        "origin": "own", "concepts": [],
    }]


def test_fold_for_match_folds_pdf_typography_on_both_sides():
    """A curly apostrophe, an "fi" ligature, a hyphen and a hard line wrap are all
    PDF-extraction artefacts: the fold has to erase them identically whichever side
    carries them, or a genuine copy compares unequal to its own source."""
    source = "The teachers’ own ﬁndings were clear about feed-\nback."
    model = "The teachers' own findings were clear about feedback."
    assert fold_for_match(source) == fold_for_match(model)
    assert fold_for_match(None) == ""


def test_fold_for_match_agrees_with_the_fulltext_canonical_fold():
    """The same canonical form the claim-verification guards already compare with
    (`app.services.fulltext._normalise_for_match`), computed character by character so
    each kept character keeps the index it came from. Decomposed diacritics are the one
    deliberate difference: this fold drops the combining mark, so a composed and a
    decomposed rendering of the same word compare equal."""
    from app.services.fulltext import _normalise_for_match

    for text in (
        "The teachers’ own ﬁndings were clear.",
        "Effect size (r = -0.28; p < 0.05) for feed-\nback.",
        "CTLA-4 and 12-15 items",
    ):
        assert fold_for_match(text) == _normalise_for_match(text)
    assert fold_for_match("élan") == fold_for_match("élan")


def test_normalise_source_text_joins_line_break_hyphenation_and_collapses_space():
    assert normalise_source_text("Students used the feed-\nback to revise.") == (
        "Students used the feedback to revise."
    )
    assert normalise_source_text("soft­hyphen  and ﬁve") == "softhyphen and five"
    assert normalise_source_text(None) == ""


def test_validate_accepts_a_quote_the_model_retyped_without_the_source_typography():
    """The whole first cause measured on the seeds: 16 of 78 proposed items were exact
    copies whose only difference from the chunk was typography the extractor produced."""
    quote = (
        "The teachers' own findings about the drafts were clear to every reader. "
        "Students used the feedback to revise their drafts and their scores rose."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), TYPOGRAPHY_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == (
        "The teachers’ own findings about the drafts were clear to every reader. "
        "Students used the feedback to revise their drafts and their scores rose."
    )


def test_validate_stores_the_sources_own_wording_not_the_models_rendering():
    """What is stored is the span taken from the chunk, not the string the model typed,
    so an accepted quote is verbatim from the paper by construction even where the fold
    was what let the two compare equal."""
    quote = "The teachers' own findings about the drafts were clear to every reader."
    accepted, _ = validate_evidence_items(_one_item(quote), TYPOGRAPHY_CHUNK)
    assert accepted[0]["quote"] == (
        "The teachers’ own findings about the drafts were clear to every reader."
    )


def test_validate_accepts_a_sentence_that_follows_a_section_heading():
    """"3.2 Procedures" carries no full stop, so the old splitter glued it onto the
    sentence after it and every genuine copy of that sentence was rejected."""
    quote = (
        "Because data collection began in the middle of the semester, implementing the "
        "tool into those courses was not practical."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), HEADING_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_validate_accepts_a_sentence_that_follows_a_table_row():
    quote = (
        "We found that the students conducted all seven types of revision operations in "
        "their drafts."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), TABLE_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_validate_accepts_a_sentence_that_follows_a_line_number_gutter():
    quote = (
        "More importantly, different feedback sources tend to elicit different "
        "engagement styles."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), GUTTER_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


HEADING_LINE_CHUNK = [{
    "section": "Method",
    "text": (
        "They were briefed on the purposes of the research \n"
        "and were asked to complete the survey at their convenience. \n"
        "3.4 Data analysis \n"
        "Responses to the surveys were sorted into figures to illustrate preference \n"
        "patterns and trends between different proficiency levels. \n"
    ),
}]

LINE_WRAP_CHUNK = [{
    "section": "Findings",
    "text": (
        "Figure 4, from the automated record, shows that \n"
        "Flora resubmitted her essay 13 times, and the score rose from 79 to 90. \n"
    ),
}]

PERCENTAGE_CHUNK = [{
    "section": "Findings",
    "text": (
        "The comments were categorised by type. "
        "Most segments were visible revision comments (57.1%), followed by non-visible "
        "revision comments (22.2%), with ambiguous comments used the least (8.4%). "
        "The pattern held across all three rounds."
    ),
}]


def test_validate_accepts_a_sentence_that_follows_a_heading_on_its_own_line():
    """A numbered heading often ends in a lower-case word ("3.4 Data analysis",
    "6.3 Limitations and suggestions for future research"), so the lead-in test alone
    rejected the sentence after it. A span that begins a physical line whose previous
    line does not run into it is a sentence start too: on the twelve seed papers this
    recovered 12 more items, every one of them the first sentence under a section
    heading, and it accepted nothing that was not."""
    quote = (
        "Responses to the surveys were sorted into figures to illustrate preference "
        "patterns and trends between different proficiency levels."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), HEADING_LINE_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_validate_rejects_a_span_that_starts_after_a_mid_sentence_line_wrap():
    """The line-start rule must not reach a sentence cut open by a hard line wrap: the
    line in front of the span ends in "that", so it runs straight into it."""
    quote = "Flora resubmitted her essay 13 times, and the score rose from 79 to 90."
    accepted, rejected = validate_evidence_items(_one_item(quote), LINE_WRAP_CHUNK)
    assert accepted == []
    assert rejected == 1


def test_validate_accepts_a_sentence_that_ends_in_a_bracketed_percentage():
    """The fold keeps only letters and digits, so the last character it keeps in this
    sentence is the "4" of "8.4"; the full stop that closes the sentence is three
    characters further on, behind "%" and ")"."""
    quote = (
        "Most segments were visible revision comments (57.1%), followed by non-visible "
        "revision comments (22.2%), with ambiguous comments used the least (8.4%)."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), PERCENTAGE_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_validate_still_rejects_a_span_that_starts_inside_a_sentence():
    """The reason the whole-sentence rule exists (design section 6 amendment A2): the
    dropped lead-in is an attribution here, and dropping it turns a screenshot's
    description into the paper's own finding. A lower-case word in front of the span is
    what tells this apart from a heading or a table row."""
    quote = (
        "Flora resubmitted her essay 13 times, and the score rose from 79 to 90 "
        "(100 as the full score)."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), ATTRIBUTION_CHUNK)
    assert accepted == []
    assert rejected == 1


def test_validate_rejects_a_span_that_stops_inside_a_sentence():
    quote = "Because data collection began in the middle of the semester, implementing"
    accepted, rejected = validate_evidence_items(_one_item(quote), HEADING_CHUNK)
    assert accepted == []
    assert rejected == 1


def test_validate_accepts_a_quote_that_starts_with_a_number():
    chunks = [{
        "section": "Findings",
        "text": (
            "Attitudes were positive overall. 91% of the students in the interviews "
            "expressed positive attitudes toward the integration of the feedback."
        ),
    }]
    quote = (
        "91% of the students in the interviews expressed positive attitudes toward the "
        "integration of the feedback."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), chunks)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_validate_finds_a_quote_in_a_later_chunk_than_the_one_named():
    chunks = [
        {"section": "intro", "text": "An opening paragraph about nothing in particular."},
        {
            "section": "results",
            "text": (
                "Scores rose across all four drafts of the assignment. "
                "The gain was largest between the second and the third draft."
            ),
        },
    ]
    quote = "The gain was largest between the second and the third draft."
    accepted, rejected = validate_evidence_items(_one_item(quote, chunk_index=0), chunks)
    assert rejected == 0
    assert accepted[0]["chunk_index"] == 1
    assert accepted[0]["section"] == "results"


def test_validate_rejects_a_paraphrase_of_a_real_sentence():
    """The one cause no acceptance rule can fix, and the reason the prompt now demands a
    character-for-character copy: 10 of 78 proposed items were the model's own wording
    of something the paper says, present nowhere in its text."""
    quote = "Of the six teachers, four were positive about the tool, while two were not."
    chunks = [{
        "section": "Findings",
        "text": (
            "While four participants were positive about the tool, two were pessimistic "
            "about using it in their own writing classroom."
        ),
    }]
    accepted, rejected = validate_evidence_items(_one_item(quote), chunks)
    assert accepted == []
    assert rejected == 1


# --------------------------------------------------------------------------------------
# The terminal-punctuation boundary must require whitespace after it,
# the same shape ``_SENTENCE_BOUNDARY_RE`` already requires two functions further up, so
# a decimal point or the last period of an author initial is never a sentence boundary.
# --------------------------------------------------------------------------------------

DECIMAL_CHUNK = [{
    "section": "Results",
    "text": (
        "The correlation between feedback and revision quality was reported as "
        "r = 0.28 for the whole sample and remained stable across the three cohorts."
    ),
}]


def test_validate_rejects_a_span_that_starts_mid_decimal():
    """A decimal point is terminal punctuation with no whitespace after it, so a span
    that begins one character into "0.28" is cutting the integer part off a real number,
    not starting a sentence."""
    quote = "28 for the whole sample and remained stable across the three cohorts."
    accepted, rejected = validate_evidence_items(_one_item(quote), DECIMAL_CHUNK)
    assert accepted == []
    assert rejected == 1


INITIAL_CHUNK = [{
    "section": "Method",
    "text": (
        "The study was designed by J.R. Smith and colleagues to test whether written "
        "corrective feedback improves accuracy over a full semester of instruction."
    ),
}]


def test_validate_rejects_a_span_that_starts_after_an_author_initial():
    """The same rule: the last period of "J.R." is followed immediately by "Smith", not
    whitespace, so it is not a sentence boundary either."""
    quote = (
        "R. Smith and colleagues to test whether written corrective feedback improves "
        "accuracy over a full semester of instruction."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), INITIAL_CHUNK)
    assert accepted == []
    assert rejected == 1


DECIMAL_THEN_SENTENCE_CHUNK = [{
    "section": "Results",
    "text": (
        "The pilot data showed an effect of 0.28. The result held across the whole "
        "sample and remained stable across the three cohorts."
    ),
}]


def test_validate_accepts_a_sentence_that_follows_a_terminal_with_whitespace_after_it():
    """The positive control for the same rule: a real sentence boundary, terminal
    punctuation followed by whitespace, is still accepted even when an unrelated decimal
    point sits earlier in the same chunk."""
    quote = (
        "The result held across the whole sample and remained stable across the three "
        "cohorts."
    )
    accepted, rejected = validate_evidence_items(_one_item(quote), DECIMAL_THEN_SENTENCE_CHUNK)
    assert rejected == 0
    assert accepted[0]["quote"] == quote


def test_analysis_prompt_version_is_a_sha256_string():
    version = analysis_prompt_version(None)
    assert version.startswith("sha256:")


def test_analysis_prompt_version_differs_by_expertise_level():
    assert analysis_prompt_version("student") != analysis_prompt_version(None)


@pytest.mark.asyncio
async def test_replace_paper_evidence_deletes_then_inserts(db_session):
    from sqlalchemy import select

    from app.models.evidence import Evidence
    from app.models.paper import Paper, SourceApi
    from app.services.deep_analysis import replace_paper_evidence

    paper = Paper(title="P", authors=[], year=2020, source_api=SourceApi.manual)
    db_session.add(paper)
    await db_session.flush()
    db_session.add(Evidence(
        paper_id=paper.id, quote="old", chunk_index=0, finding="old",
        kind="finding", origin="own", concepts=[], prompt_version="sha256:old",
    ))
    await db_session.commit()

    await replace_paper_evidence(
        db_session, paper.id,
        [{
            "quote": "Tutoring improved outcomes for most students in the sample.",
            "chunk_index": 0, "section": "Results", "finding": "new",
            "kind": "finding", "origin": "own", "concepts": ["tutoring"],
        }],
        "sha256:new",
    )
    await db_session.commit()

    rows = (
        await db_session.execute(select(Evidence).where(Evidence.paper_id == paper.id))
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].quote == "Tutoring improved outcomes for most students in the sample."
    assert rows[0].prompt_version == "sha256:new"
