"""The delivered-evidence record: one row per claim
that survived into the final, healed report, each carrying the verbatim source passage
a human judge checks it against.

``build_delivered_evidence_rows``/``build_delivered_evidence`` never make an HTTP call
themselves -- ``fetch_chunk`` is injected -- so these tests run against a plain fixture,
with no ``FakeServer``/``ApiClient`` involved (that end-to-end wiring is exercised by
``test_run_demo.py``'s ``test_full_run_calls_every_endpoint_in_order``).
"""

from __future__ import annotations

import pytest
from run_demo import (
    ABSTRACT_CHUNK_INDEX,
    ApiError,
    DemoError,
    _find_quote_in_text,
    _keep_letters_and_digits_with_offsets,
    _locate_quote_excerpt,
    build_delivered_evidence,
    build_delivered_evidence_rows,
    merge_delivered_evidence_records,
    parse_evidence_location,
    trim_source_passage,
)

PAPER_ID = "44444444-4444-4444-4444-444444444444"
CHUNK_TEXT = "The intervention produced a gain in vocabulary retention among learners."


def _verification(**overrides):
    base = {
        "claim_text": "Learners showed a gain in vocabulary retention.",
        "claim_sentence": "Learners showed a gain in vocabulary retention (Beta, 2016).",
        "citation": "(Beta, 2016)",
        "paper_id": PAPER_ID,
        "paper_doi": "10.5555/demo-001",
        "paper_title": "Fixture paper one",
        "status": "verified",
        "evidence_quote": "a gain in vocabulary retention",
        "evidence_quotes": ["a gain in vocabulary retention"],
        "evidence_location": "chunk 1 (results)",
    }
    base.update(overrides)
    return base


def _fetch_chunk(paper_id: str, chunk_index: int) -> dict:
    assert paper_id == PAPER_ID
    assert chunk_index == 0
    return {
        "paper_id": paper_id, "chunk_index": chunk_index, "section": "results", "text": CHUNK_TEXT,
    }


# --------------------------------------------------------------------------------------
# parse_evidence_location
# --------------------------------------------------------------------------------------


def test_parse_evidence_location_with_section():
    assert parse_evidence_location("chunk 3 (results)") == (2, "results")


def test_parse_evidence_location_without_section():
    assert parse_evidence_location("chunk 1") == (0, None)


def test_parse_evidence_location_none_or_unparseable():
    assert parse_evidence_location(None) is None
    assert parse_evidence_location("") is None
    assert parse_evidence_location("not a location") is None


# --------------------------------------------------------------------------------------
# trim_source_passage
# --------------------------------------------------------------------------------------


def test_trim_source_passage_returns_short_chunk_whole():
    assert trim_source_passage(CHUNK_TEXT, "a gain in vocabulary retention") == CHUNK_TEXT


def test_trim_source_passage_centres_on_an_exact_match():
    long_text = ("padding " * 500) + "THE EXACT QUOTE" + (" padding" * 500)
    result = trim_source_passage(long_text, "THE EXACT QUOTE", max_chars=100)
    assert "THE EXACT QUOTE" in result
    assert len(result) <= 100


def test_trim_source_passage_is_verbatim_from_the_chunk():
    """The excerpt is a real substring of the chunk, never rewritten."""
    long_text = ("padding " * 500) + "THE EXACT QUOTE" + (" padding" * 500)
    result = trim_source_passage(long_text, "THE EXACT QUOTE", max_chars=100)
    assert result in long_text


def test_trim_source_passage_tolerates_a_whitespace_difference():
    """A PDF line-wrap can put a newline where the model's own quote has a plain space;
    the whitespace-tolerant fallback still locates it."""
    long_text = ("padding " * 500) + "THE EXACT\nQUOTE" + (" padding" * 500)
    result = trim_source_passage(long_text, "THE EXACT QUOTE", max_chars=100)
    assert "THE EXACT\nQUOTE" in result


def test_trim_source_passage_falls_back_to_the_start_when_the_quote_is_not_found():
    long_text = "x" * 5000
    result = trim_source_passage(long_text, "not present anywhere", max_chars=100)
    assert result == "x" * 100


def test_trim_source_passage_handles_a_missing_quote():
    long_text = "x" * 5000
    result = trim_source_passage(long_text, None, max_chars=100)
    assert result == "x" * 100


# --------------------------------------------------------------------------------------
# build_delivered_evidence_rows
# --------------------------------------------------------------------------------------


def test_build_delivered_evidence_rows_builds_one_row_per_verification():
    rows = build_delivered_evidence_rows([_verification()], _fetch_chunk)
    assert len(rows) == 1
    row = rows[0]
    assert row["row"] == 1
    assert row["sentence"] == "Learners showed a gain in vocabulary retention (Beta, 2016)."
    assert row["citation_text"] == "(Beta, 2016)"
    assert row["paper_title"] == "Fixture paper one"
    assert row["paper_doi"] == "10.5555/demo-001"
    assert row["claim_text"] == "Learners showed a gain in vocabulary retention."
    assert row["status"] == "verified"
    assert row["evidence_quotes"] == ["a gain in vocabulary retention"]
    assert "a gain in vocabulary retention" in row["source_passage"]
    assert row["source_passage"] in CHUNK_TEXT
    assert row["source_locator"] == {"chunk_index": 0, "section": "results"}
    assert row["excerpt_locations"] == [{"chunk_index": 0, "section": "results"}]
    assert row["source_located"] is True
    assert row["passage_located"] is True
    assert row["unlocated_quotes"] == []
    assert row["location_section_mismatch"] is False
    assert row["sentence_in_draft"] is None  # no draft_content given
    assert row["fetch_error"] is None


def test_build_delivered_evidence_rows_numbers_rows_in_order_starting_at_one():
    rows = build_delivered_evidence_rows(
        [_verification(claim_text="First."), _verification(claim_text="Second.")],
        _fetch_chunk,
    )
    assert [r["row"] for r in rows] == [1, 2]
    assert [r["claim_text"] for r in rows] == ["First.", "Second."]


def test_build_delivered_evidence_rows_falls_back_to_claim_text_without_a_claim_sentence():
    rows = build_delivered_evidence_rows(
        [_verification(claim_sentence=None)], _fetch_chunk
    )
    assert rows[0]["sentence"] == rows[0]["claim_text"]


def test_build_delivered_evidence_rows_falls_back_to_evidence_quote_without_a_list():
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=[])], _fetch_chunk
    )
    assert rows[0]["evidence_quotes"] == ["a gain in vocabulary retention"]


def test_build_delivered_evidence_rows_raises_on_a_non_verified_row():
    with pytest.raises(DemoError, match="not \"verified\""):
        build_delivered_evidence_rows(
            [_verification(status="needs_nuance")], _fetch_chunk
        )


# --------------------------------------------------------------------------------------
# A passage that contains every quote, fetching a quote's own chunk
# when it differs from the row's primary (``evidence_location``) chunk.
# --------------------------------------------------------------------------------------

OTHER_CHUNK_TEXT = "A later section reports durable transfer to novel vocabulary items."


def _fetch_two_chunks(paper_id: str, chunk_index: int) -> dict:
    assert paper_id == PAPER_ID
    texts = {0: CHUNK_TEXT, 1: OTHER_CHUNK_TEXT}
    sections = {0: "results", 1: "discussion"}
    return {
        "paper_id": paper_id, "chunk_index": chunk_index, "chunk_count": 2,
        "section": sections[chunk_index], "text": texts[chunk_index],
    }


def test_build_delivered_evidence_rows_finds_a_second_quote_in_a_different_chunk():
    """Two evidence_quotes, one in the primary (evidence_location) chunk, one only in
    another chunk of the same paper -- both must appear in source_passage, and the
    second quote's own chunk is fetched to find it."""
    rows = build_delivered_evidence_rows(
        [
            _verification(
                evidence_quotes=[
                    "a gain in vocabulary retention",
                    "durable transfer to novel vocabulary items",
                ]
            )
        ],
        _fetch_two_chunks,
    )
    row = rows[0]
    assert "a gain in vocabulary retention" in row["source_passage"]
    assert "durable transfer to novel vocabulary items" in row["source_passage"]
    assert row["unlocated_quotes"] == []
    assert row["passage_located"] is True
    # The row's own source_locator still names the primary chunk (where evidence_location
    # pointed); the second quote's own chunk is used only to build its own excerpt.
    assert row["source_locator"] == {"chunk_index": 0, "section": "results"}
    # excerpt_locations records, per excerpt joined
    # into source_passage, exactly which chunk it was found in -- here the first excerpt
    # is the primary chunk (0, results) and the second is the other chunk (1, discussion),
    # not both silently reported under the row's single primary locator.
    assert row["excerpt_locations"] == [
        {"chunk_index": 0, "section": "results"},
        {"chunk_index": 1, "section": "discussion"},
    ]


def test_build_delivered_evidence_rows_marks_a_quote_unlocatable_anywhere():
    """A quote present in neither the primary chunk nor any other chunk of the paper is
    named in unlocated_quotes, not silently replaced by the primary chunk's own opening
    characters. This is the "partly located" case:
    source_located is true and source_passage is non-empty (it shows the quote that WAS
    found), but passage_located is false, since not every evidence quote was located."""
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["a gain in vocabulary retention", "not present anywhere"])],
        _fetch_two_chunks,
    )
    row = rows[0]
    assert row["unlocated_quotes"] == ["not present anywhere"]
    assert "a gain in vocabulary retention" in row["source_passage"]
    assert "not present anywhere" not in row["source_passage"]
    assert row["source_located"] is True
    assert row["passage_located"] is False


def test_build_delivered_evidence_rows_passage_located_false_when_no_quote_is_found():
    """The chunk fetch succeeds (source_located true) but none of the row's evidence
    quotes could be found in it, so source_passage is None -- passage_located must be
    false here too: a row with source_located true and source_passage empty must not
    be counted as a located passage."""
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["not present anywhere"])], _fetch_chunk,
    )
    row = rows[0]
    assert row["source_located"] is True
    assert row["source_passage"] is None
    assert row["passage_located"] is False
    assert row["unlocated_quotes"] == ["not present anywhere"]


def test_build_delivered_evidence_rows_passage_located_false_with_no_evidence_quotes():
    """A row with no evidence_quotes at all falls back
    to a passage taken from the start of the chunk (the "elif not evidence_quotes"
    branch) with an empty unlocated_quotes list -- "not unlocated_quotes" alone is
    vacuously true here, so passage_located must also require evidence_quotes to be
    non-empty, or a human is asked to judge an arbitrary excerpt against no quote at
    all."""
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=[], evidence_quote="")], _fetch_chunk,
    )
    row = rows[0]
    assert row["evidence_quotes"] == []
    assert row["source_located"] is True
    assert row["source_passage"] == CHUNK_TEXT
    assert row["unlocated_quotes"] == []
    assert row["passage_located"] is False


# --------------------------------------------------------------------------------------
# The verifier's own prompt gives the model the paper's abstract as context alongside
# its numbered full-text chunks, so a quote can be genuinely verbatim only in the
# abstract, not in any `fulltext_chunks` entry -- but `evidence_location` never names
# the abstract as a "chunk N". `GET /papers/{id}/fulltext-chunks/{n}` also returns
# `abstract` (a fixed, small string, present on every chunk of the same paper), and
# the locator's own fallback loop tries it last, after every numbered chunk.
# --------------------------------------------------------------------------------------

ABSTRACT_TEXT = "This survey found broad support for scaffolded feedback across cohorts."


def _fetch_chunk_with_abstract(paper_id: str, chunk_index: int) -> dict:
    assert paper_id == PAPER_ID
    assert chunk_index == 0
    return {
        "paper_id": paper_id, "chunk_index": chunk_index, "section": "results",
        "text": CHUNK_TEXT, "abstract": ABSTRACT_TEXT,
    }


def test_locate_quote_excerpt_falls_back_to_the_abstract_when_no_chunk_has_it():
    result = _locate_quote_excerpt(
        "broad support for scaffolded feedback",
        CHUNK_TEXT,
        primary_index=0,
        chunk_count=1,
        fetch_chunk_text=lambda idx: CHUNK_TEXT,
        abstract_text=ABSTRACT_TEXT,
    )
    assert result is not None
    excerpt, found_index = result
    assert "broad support for scaffolded feedback" in excerpt
    assert found_index == ABSTRACT_CHUNK_INDEX


def test_locate_quote_excerpt_never_reaches_the_abstract_when_a_chunk_already_has_it():
    result = _locate_quote_excerpt(
        "a gain in vocabulary retention",
        CHUNK_TEXT,
        primary_index=0,
        chunk_count=1,
        fetch_chunk_text=lambda idx: CHUNK_TEXT,
        abstract_text=ABSTRACT_TEXT,
    )
    assert result is not None
    _excerpt, found_index = result
    assert found_index == 0


def test_locate_quote_excerpt_returns_none_when_neither_chunks_nor_the_abstract_have_it():
    result = _locate_quote_excerpt(
        "not present anywhere",
        CHUNK_TEXT,
        primary_index=0,
        chunk_count=1,
        fetch_chunk_text=lambda idx: CHUNK_TEXT,
        abstract_text=ABSTRACT_TEXT,
    )
    assert result is None


def test_build_delivered_evidence_rows_locates_a_quote_only_in_the_abstract():
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["broad support for scaffolded feedback"])],
        _fetch_chunk_with_abstract,
    )
    row = rows[0]
    assert row["unlocated_quotes"] == []
    assert row["passage_located"] is True
    assert "broad support for scaffolded feedback" in row["source_passage"]
    assert row["excerpt_locations"] == [{"chunk_index": None, "section": "abstract"}]


def test_build_delivered_evidence_rows_still_reports_unlocated_when_the_abstract_lacks_it():
    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["not present anywhere"])], _fetch_chunk_with_abstract,
    )
    row = rows[0]
    assert row["unlocated_quotes"] == ["not present anywhere"]
    assert row["passage_located"] is False


# --------------------------------------------------------------------------------------
# The locator must find a span the verifier's own guard
# already accepted as verbatim under its letters-and-digits fold, even when none of
# _find_quote_in_text's three narrower matches can.
# --------------------------------------------------------------------------------------


def test_find_quote_in_text_locates_an_en_dash_against_a_hyphen():
    """The guard's own fold (NFKC, letters and digits only, case-folded) discards dashes
    and hyphens identically, so it accepts a hyphenated model quote against an en-dashed
    chunk as verbatim. This locator must find the same real, contiguous span."""
    chunk_text = "Participants reported gains of 12–14 percent on the post-test."
    quote = "gains of 12-14 percent"
    span = _find_quote_in_text(chunk_text, quote)
    assert span is not None
    start, end = span
    assert chunk_text[start:end] == "gains of 12–14 percent"


def test_build_delivered_evidence_rows_locates_a_quote_across_an_en_dash_hyphen_difference():
    """End to end: a chunk with an en dash and a model quote with a plain hyphen is
    fully located, not left in unlocated_quotes."""

    def _fetch_en_dash_chunk(paper_id: str, chunk_index: int) -> dict:
        return {
            "paper_id": paper_id, "chunk_index": chunk_index, "chunk_count": 1,
            "section": "results",
            "text": "Participants reported gains of 12–14 percent on the post-test.",
        }

    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["gains of 12-14 percent"])], _fetch_en_dash_chunk,
    )
    row = rows[0]
    assert row["unlocated_quotes"] == []
    assert row["passage_located"] is True
    assert "12–14" in row["source_passage"]


# --------------------------------------------------------------------------------------
# The guard-fold locator's offset map must stay in step
# with its own folded string even when a kept character's case fold expands (German
# sharp s -> "ss", Turkish dotted capital I -> two characters) -- one offset per
# character of the folded OUTPUT, not per raw character.
# --------------------------------------------------------------------------------------


def test_keep_letters_and_digits_with_offsets_stays_in_step_when_casefold_expands():
    """"ß" (German sharp s) folds to two characters ("ss"); every kept character of the
    folded string must still have its own offset, or a lookup past the expansion point
    reads the wrong raw index (or none at all)."""
    text = "Straße et al."
    folded, offsets = _keep_letters_and_digits_with_offsets(text)
    assert folded == "strasseetal"
    assert len(folded) == len(offsets)


def test_find_quote_in_text_locates_the_right_span_past_a_casefold_expanding_character():
    """The reviewer's own reproduction: a German sharp s
    earlier in the chunk must not shift the span the guard-fold branch returns for a
    later quote that only differs from the chunk by an en dash versus a hyphen."""
    chunk_text = (
        "Straße et al. reported gains of 12–14 percent in the treated group, overall."
    )
    quote = "gains of 12-14 percent"
    span = _find_quote_in_text(chunk_text, quote)
    assert span is not None
    start, end = span
    assert chunk_text[start:end] == "gains of 12–14 percent"


def test_find_quote_in_text_does_not_raise_when_the_quote_ends_the_chunk_past_an_expansion():
    """The reviewer's own IndexError reproduction: the quote's own span reaches the very
    end of the chunk, past the German sharp s earlier in it."""
    chunk_text = "Straße et al. reported gains of 12–14 percent in the treated group."
    quote = "gains of 12-14 percent in the treated group."
    span = _find_quote_in_text(chunk_text, quote)
    assert span is not None
    start, end = span
    # The fold keeps only letters and digits, so the trailing "." is not part of the
    # located span -- what matters here is that this does not raise IndexError, and
    # that the span still ends at the right word.
    assert chunk_text[start:end] == "gains of 12–14 percent in the treated group"


def test_build_delivered_evidence_rows_locates_a_quote_past_a_casefold_expanding_character():
    """End to end: the run must not raise and must locate the quote correctly when an
    earlier German sharp s expands under the guard fold."""

    def _fetch_strasse_chunk(paper_id: str, chunk_index: int) -> dict:
        return {
            "paper_id": paper_id, "chunk_index": chunk_index, "chunk_count": 1,
            "section": "results",
            "text": (
                "Straße et al. reported gains of 12–14 percent in the treated group, "
                "overall."
            ),
        }

    rows = build_delivered_evidence_rows(
        [_verification(evidence_quotes=["gains of 12-14 percent"])], _fetch_strasse_chunk,
    )
    row = rows[0]
    assert row["unlocated_quotes"] == []
    assert row["passage_located"] is True
    assert "gains of 12–14 percent" in row["source_passage"]


def test_find_quote_in_text_case_insensitive_match_past_a_length_expanding_character():
    """The case-insensitive branch (one match
    earlier than the guard fold) has the same length-preservation bug -- "İ".lower() is
    two characters -- and must map back to the right raw span too."""
    text = "İstanbul: gains of 12-14 percent were reported."
    quote = "GAINS OF 12-14 PERCENT"
    span = _find_quote_in_text(text, quote)
    assert span is not None
    start, end = span
    assert text[start:end] == "gains of 12-14 percent"


# --------------------------------------------------------------------------------------
# The parsed section is compared with the fetched chunk's own,
# instead of discarded.
# --------------------------------------------------------------------------------------


def test_build_delivered_evidence_rows_flags_a_section_mismatch():
    def _fetch_chunk_wrong_section(paper_id: str, chunk_index: int) -> dict:
        return {
            "paper_id": paper_id, "chunk_index": chunk_index, "chunk_count": 1,
            "section": "discussion", "text": CHUNK_TEXT,
        }

    rows = build_delivered_evidence_rows(
        [_verification(evidence_location="chunk 1 (results)")], _fetch_chunk_wrong_section
    )
    assert rows[0]["location_section_mismatch"] is True


def test_build_delivered_evidence_rows_no_mismatch_when_sections_agree():
    rows = build_delivered_evidence_rows([_verification()], _fetch_chunk)
    assert rows[0]["location_section_mismatch"] is False


def test_build_delivered_evidence_rows_no_mismatch_when_location_carries_no_section():
    rows = build_delivered_evidence_rows(
        [_verification(evidence_location="chunk 1")], _fetch_chunk
    )
    assert rows[0]["location_section_mismatch"] is False


# --------------------------------------------------------------------------------------
# sentence_in_draft, checked against the saved draft content the
# runner already fetched.
# --------------------------------------------------------------------------------------

DRAFT_CONTENT = {
    "type": "doc",
    "content": [
        {
            "type": "paragraph",
            "content": [
                {
                    "type": "text",
                    "text": "Learners showed a gain in vocabulary retention (Beta, 2016).",
                }
            ],
        }
    ],
}


def test_build_delivered_evidence_rows_sentence_in_draft_true():
    rows = build_delivered_evidence_rows(
        [_verification()], _fetch_chunk, draft_content=DRAFT_CONTENT
    )
    assert rows[0]["sentence_in_draft"] is True


def test_build_delivered_evidence_rows_sentence_in_draft_false():
    """Finalize rule 3 can rewrite a sentence in place after dropping a co-cited
    no_full_text citation; the stored claim_sentence then keeps the pre-heal wording,
    which no longer occurs in the saved draft. Flagged, not silently shown as if it
    still matched."""
    rows = build_delivered_evidence_rows(
        [_verification(claim_sentence="A sentence finalize has since rewritten.")],
        _fetch_chunk,
        draft_content=DRAFT_CONTENT,
    )
    assert rows[0]["sentence_in_draft"] is False


def test_build_delivered_evidence_rows_does_not_raise_when_evidence_location_is_missing():
    """A verified row with a real but short (under
    QUOTE_SEGMENT_MIN_CHARS) quote legitimately has no locatable evidence_location --
    `_guard_quote_fidelity` passes it as "no evidence either way", not a defect. The row
    is still emitted, marked, rather than aborting the whole cold demo run."""
    rows = build_delivered_evidence_rows(
        [_verification(evidence_location=None)], _fetch_chunk
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["source_located"] is False
    assert row["source_passage"] is None
    assert row["source_locator"] is None
    assert row["excerpt_locations"] == []
    assert row["passage_located"] is False
    assert row["unlocated_quotes"] == ["a gain in vocabulary retention"]


def test_build_delivered_evidence_rows_does_not_raise_when_evidence_location_is_unparseable():
    rows = build_delivered_evidence_rows(
        [_verification(evidence_location="somewhere")], _fetch_chunk
    )
    assert rows[0]["source_located"] is False
    assert rows[0]["source_passage"] is None
    assert rows[0]["source_locator"] is None
    assert rows[0]["excerpt_locations"] == []
    assert rows[0]["passage_located"] is False


# --------------------------------------------------------------------------------------
# A chunk-fetch failure marks the row instead of
# aborting the whole run -- `fetch_fulltext_chunk` raises `ApiError` on a 404, a timeout,
# or a dropped connection, and this is the run's last step, after screening, writing and
# verification have already spent their budget.
# --------------------------------------------------------------------------------------


def test_build_delivered_evidence_rows_marks_the_row_when_the_chunk_fetch_fails():
    def _raising_fetch(paper_id: str, chunk_index: int) -> dict:
        raise ApiError(f"GET /papers/{paper_id}/fulltext-chunks/{chunk_index}: HTTP 404")

    rows = build_delivered_evidence_rows([_verification()], _raising_fetch)

    assert len(rows) == 1
    row = rows[0]
    assert row["source_located"] is False
    assert row["source_passage"] is None
    assert row["source_locator"] is None
    assert row["excerpt_locations"] == []
    assert row["passage_located"] is False
    assert row["unlocated_quotes"] == ["a gain in vocabulary retention"]
    assert row["location_section_mismatch"] is False
    assert "404" in row["fetch_error"]


def test_build_delivered_evidence_rows_a_fetch_failure_does_not_abort_later_rows():
    """The row that fails is marked; every row after it is still built normally, and no
    exception reaches the caller."""
    calls = {"n": 0}

    def _fails_on_first_call_only(paper_id: str, chunk_index: int) -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApiError("could not reach the API")
        return _fetch_chunk(paper_id, chunk_index)

    rows = build_delivered_evidence_rows(
        [_verification(claim_text="First."), _verification(claim_text="Second.")],
        _fails_on_first_call_only,
    )

    assert len(rows) == 2
    assert rows[0]["source_located"] is False
    assert rows[0]["fetch_error"] == "could not reach the API"
    assert rows[1]["source_located"] is True
    assert rows[1]["fetch_error"] is None


def test_build_delivered_evidence_counts_a_fetch_failure_as_unlocated():
    def _raising_fetch(paper_id: str, chunk_index: int) -> dict:
        raise ApiError("boom")

    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="s",
        verifications=[_verification()], fetch_chunk=_raising_fetch,
    )
    assert record["unlocated_rows"] == 1


# --------------------------------------------------------------------------------------
# The chunk cache is shared across rows, keyed by
# (paper_id, chunk_index), so two rows citing the same paper's chunk fetch it once.
# --------------------------------------------------------------------------------------


def test_build_delivered_evidence_rows_fetches_each_primary_chunk_at_most_once():
    calls: list[tuple[str, int]] = []

    def _counting_fetch(paper_id: str, chunk_index: int) -> dict:
        calls.append((paper_id, chunk_index))
        return _fetch_chunk(paper_id, chunk_index)

    rows = build_delivered_evidence_rows(
        [_verification(claim_text="First."), _verification(claim_text="Second.")],
        _counting_fetch,
    )

    assert len(rows) == 2
    assert calls == [(PAPER_ID, 0)]


def test_build_delivered_evidence_rows_reuses_a_second_quotes_chunk_fetch_across_rows():
    """Row 1's quote lookup fetches chunk 1 (the second quote's own chunk); row 2 cites
    the same paper and quotes, so its own chunk 1 lookup must be served from the cache,
    not fetched again."""
    calls: list[tuple[str, int]] = []

    def _counting_two_chunk_fetch(paper_id: str, chunk_index: int) -> dict:
        calls.append((paper_id, chunk_index))
        return _fetch_two_chunks(paper_id, chunk_index)

    verification_with_second_quote = _verification(
        evidence_quotes=[
            "a gain in vocabulary retention",
            "durable transfer to novel vocabulary items",
        ]
    )
    rows = build_delivered_evidence_rows(
        [verification_with_second_quote, verification_with_second_quote],
        _counting_two_chunk_fetch,
    )

    assert len(rows) == 2
    assert calls == [(PAPER_ID, 0), (PAPER_ID, 1)]


# --------------------------------------------------------------------------------------
# build_delivered_evidence
# --------------------------------------------------------------------------------------


def test_build_delivered_evidence_top_level_shape():
    record = build_delivered_evidence(
        run_id="20260911-000000",
        draft_id="33333333-3333-3333-3333-333333333333",
        section_title="Literature Review",
        verifications=[_verification()],
        fetch_chunk=_fetch_chunk,
    )
    assert record["run_id"] == "20260911-000000"
    assert record["draft_id"] == "33333333-3333-3333-3333-333333333333"
    assert record["section_title"] == "Literature Review"
    assert len(record["rows"]) == 1
    assert record["unlocated_rows"] == 0


def test_build_delivered_evidence_with_no_verifications_is_an_empty_row_list():
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="s", verifications=[], fetch_chunk=_fetch_chunk,
    )
    assert record["rows"] == []
    assert record["unlocated_rows"] == 0


def test_build_delivered_evidence_counts_unlocated_rows():
    """A row with no locatable evidence_location is
    emitted, not raised on, and counted at the top level so a reader of the record does
    not have to scan every row to see how many carry no located passage."""
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="s",
        verifications=[_verification(), _verification(evidence_location=None)],
        fetch_chunk=_fetch_chunk,
    )
    assert len(record["rows"]) == 2
    assert record["unlocated_rows"] == 1


def test_build_delivered_evidence_counts_a_partly_located_row_as_unlocated():
    """unlocated_rows is counted against
    passage_located, not source_located -- a row whose chunk fetch succeeded but whose
    quote could not be found (source_located true, passage_located false) must still be
    counted here, not silently treated as located."""
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="s",
        verifications=[_verification(evidence_quotes=["not present anywhere"])],
        fetch_chunk=_fetch_chunk,
    )
    row = record["rows"][0]
    assert row["source_located"] is True
    assert row["passage_located"] is False
    assert record["unlocated_rows"] == 1


def test_build_delivered_evidence_passes_draft_content_through():
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="s",
        verifications=[_verification()],
        fetch_chunk=_fetch_chunk,
        draft_content=DRAFT_CONTENT,
    )
    assert record["rows"][0]["sentence_in_draft"] is True


# --------------------------------------------------------------------------------------
# merge_delivered_evidence_records: one delivered_evidence.json for a
# multi-section run, built from one build_delivered_evidence record per section.
# --------------------------------------------------------------------------------------


def test_merge_delivered_evidence_records_tags_each_row_with_its_own_section():
    protocol_record = build_delivered_evidence(
        run_id="r", draft_id="draft-1", section_title="Protocol Section",
        verifications=[_verification(claim_text="First.")], fetch_chunk=_fetch_chunk,
    )
    extra_record = build_delivered_evidence(
        run_id="r", draft_id="draft-2", section_title="Extra Section",
        verifications=[_verification(claim_text="Second.")], fetch_chunk=_fetch_chunk,
    )
    merged = merge_delivered_evidence_records([protocol_record, extra_record])

    assert merged["run_id"] == "r"
    assert len(merged["rows"]) == 2
    assert merged["rows"][0]["claim_text"] == "First."
    assert merged["rows"][0]["draft_id"] == "draft-1"
    assert merged["rows"][0]["section_title"] == "Protocol Section"
    assert merged["rows"][1]["claim_text"] == "Second."
    assert merged["rows"][1]["draft_id"] == "draft-2"
    assert merged["rows"][1]["section_title"] == "Extra Section"


def test_merge_delivered_evidence_records_sums_unlocated_rows():
    record_a = build_delivered_evidence(
        run_id="r", draft_id="a", section_title="A",
        verifications=[_verification(evidence_location=None)], fetch_chunk=_fetch_chunk,
    )
    record_b = build_delivered_evidence(
        run_id="r", draft_id="b", section_title="B",
        verifications=[_verification()], fetch_chunk=_fetch_chunk,
    )
    merged = merge_delivered_evidence_records([record_a, record_b])
    assert merged["unlocated_rows"] == 1


def test_merge_delivered_evidence_records_preserves_row_order_within_each_section():
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="S",
        verifications=[_verification(claim_text="First."), _verification(claim_text="Second.")],
        fetch_chunk=_fetch_chunk,
    )
    merged = merge_delivered_evidence_records([record])
    assert [r["claim_text"] for r in merged["rows"]] == ["First.", "Second."]
    assert [r["row"] for r in merged["rows"]] == [1, 2]


def test_merge_delivered_evidence_records_of_a_single_section_is_that_records_own_rows():
    record = build_delivered_evidence(
        run_id="r", draft_id="d", section_title="Only Section",
        verifications=[_verification()], fetch_chunk=_fetch_chunk,
    )
    merged = merge_delivered_evidence_records([record])
    assert merged["run_id"] == "r"
    assert merged["unlocated_rows"] == record["unlocated_rows"]
    assert len(merged["rows"]) == 1
    assert merged["rows"][0]["draft_id"] == "d"
    assert merged["rows"][0]["section_title"] == "Only Section"


def test_merge_delivered_evidence_records_of_an_empty_list_is_well_shaped():
    merged = merge_delivered_evidence_records([])
    assert merged == {"run_id": None, "rows": [], "unlocated_rows": 0}
