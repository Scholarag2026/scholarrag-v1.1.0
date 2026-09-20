"""Quote relocation: ``app.services.fulltext.relocate_evidence_quotes``.

A separate, versioned step that runs immediately before ``apply_verification_guards`` inside
the shared verification policy (``app.services.fulltext.verify_claim_with_policy``, see
``test_verification_policy.py``), never inside the guards themselves: the frozen guard set
(``apply_verification_guards`` and its seven guards, digest ``8a6c833ffc329f89`` since guard
7) is untouched by this file, and its own tests
(``test_verification_guards.py``, ``test_evidence_chunk_ordering.py``) are not edited here.

Word-level tolerance is abandoned entirely: the acceptance rule is not "small enough
and not a digit/number/negation", nor "small enough and every touched token in one
closed-class allowlist" -- every allowlist of interchangeable words, however
narrowly drawn, turns up a meaning-flipping pair somewhere. A British/American
spelling fold, tried as a second neutral class, also turned out to merge distinct
English words (the doubled-l fold's ``filled``/``filed``, the digraph fold's
``shoe``/``she``), so it is not included either. The rule is closed to
exactly two neutral classes, stated once here and in ``relocate_evidence_quotes``'s own
docstring: **at least 8 tokens, any number of neutral-class edits, nothing else.**

1. Articles: a substitution within {a, an, the}, or an insertion or deletion of one of them.
2. Deletion, never insertion, of one whole parenthetical citation (``_is_dropped_
   parenthetical_citation``) -- counted as one edit however many tokens it spans, and
   requiring a real citation shape, not merely a year.

The authorised motivating case ("we" typed for "and") is deliberately *not* handled here any
more: it is not an article or a citation. Neither is a spelling difference any longer. Both
are resolved by Part B, the shared verification policy's bounded second pass
(``test_verification_policy.py``).

Pure-function tests (no database) live in the first half; the last section is one end-to-end
test through ``verify_and_heal_claims`` proving the production wiring (the glue code inside
``_verify_claims``, via ``verify_claim_with_policy``) actually relocates before guarding,
stores the relocated text, and threads the relocation diagnostic through without losing the
guards' own ``diagnostics`` list.
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------------------
# _tokens_with_offsets / _is_negation_token / _token_contains_digit
# --------------------------------------------------------------------------------------


def test_tokens_with_offsets_keeps_a_contraction_as_one_token():
    from app.services.fulltext import _tokens_with_offsets

    tokens = _tokens_with_offsets("The authors didn't find an effect.")
    raw = [t for t, _s, _e in tokens]
    assert "didn't" in raw
    assert "didn" not in raw and "t" not in raw


def test_tokens_with_offsets_recovers_raw_offsets():
    from app.services.fulltext import _tokens_with_offsets

    text = "Scores rose 12% overall."
    tokens = _tokens_with_offsets(text)
    for raw, start, end in tokens:
        assert text[start:end] == raw


def test_is_negation_token_recognises_words_and_contractions():
    from app.services.fulltext import _is_negation_token

    for word in ("no", "not", "never", "none", "nor", "without", "cannot", "Not", "NEVER"):
        assert _is_negation_token(word), word
    for word in ("didn't", "isn't", "wasn't", "aren't", "won't", "can't", "shouldn't"):
        assert _is_negation_token(word), word
    for word in ("known", "return", "did", "is", "not-really-a-negation-just-hyphenated"):
        assert not _is_negation_token(word), word


def test_token_contains_digit():
    from app.services.fulltext import _token_contains_digit

    assert _token_contains_digit("2021")
    assert _token_contains_digit("L2")
    assert not _token_contains_digit("effect")


def test_is_negation_token_recognises_the_extended_word_list():
    from app.services.fulltext import _is_negation_token

    for word in ("neither", "non", "nothing", "nobody", "nowhere", "unable", "rarely",
                 "hardly", "barely", "unless", "except", "Neither", "NON"):
        assert _is_negation_token(word), word


def test_is_negation_token_recognises_lack_fail_and_absen_prefixed_words():
    from app.services.fulltext import _is_negation_token

    for word in ("lack", "lacks", "lacked", "lacking", "fail", "fails", "failed", "failing",
                 "absence", "absent", "absently", "Failed"):
        assert _is_negation_token(word), word
    for word in ("lackluster",):  # starts with "lack" -- an accepted, documented over-reach
        assert _is_negation_token(word)
    for word in ("failure",):  # starts with "fail" too
        assert _is_negation_token(word)


def test_tokens_differ_only_by_negation_prefix_recognises_common_pairs():
    from app.services.fulltext import _tokens_differ_only_by_negation_prefix

    for shorter, longer in (
        ("changed", "unchanged"),
        ("significant", "insignificant"),
        ("effective", "ineffective"),
        ("possible", "impossible"),
    ):
        assert _tokens_differ_only_by_negation_prefix(shorter, longer)
        assert _tokens_differ_only_by_negation_prefix(longer, shorter)
        assert _tokens_differ_only_by_negation_prefix(shorter.upper(), longer.upper())
    assert not _tokens_differ_only_by_negation_prefix("changed", "changed")
    assert not _tokens_differ_only_by_negation_prefix("changed", "moved")


def test_is_protected_number_word_recognises_cardinals_and_ordinals():
    from app.services.fulltext import _is_protected_number_word

    for word in ("one", "three", "twelve", "twenty", "thirty", "hundred", "thousand",
                 "million", "half", "third", "quarter", "twice", "double", "first", "tenth",
                 "Three", "FIVE"):
        assert _is_protected_number_word(word), word
    for word in ("effect", "improved", "scores"):
        assert not _is_protected_number_word(word), word


# --------------------------------------------------------------------------------------
# Neutral class 1: articles (`_is_relocation_article`, `_RELOCATION_ARTICLES`)
# --------------------------------------------------------------------------------------


def test_is_relocation_article_recognises_only_a_an_the():
    from app.services.fulltext import _is_relocation_article

    for word in ("a", "an", "the", "A", "AN", "The"):
        assert _is_relocation_article(word), word
    for word in ("this", "that", "some", "any", "we", "is", "of", "and"):
        assert not _is_relocation_article(word), word


def test_relocation_articles_are_exactly_a_an_the():
    """With the spelling-fold class removed, the shipped neutral classes contain no pair
    of distinct tokens outside {a, an, the} -- i.e. there is no *other* word-membership
    allowlist left to reopen an antonym-pair failure mode. `_RELOCATION_ARTICLES` is the
    only closed word list either neutral class's acceptance depends on (class 2, the
    dropped citation, is a structural
    shape check -- a bracket, a year, a surname-like token, an et al pair -- not a word
    list)."""
    from app.services.fulltext import _RELOCATION_ARTICLES

    assert _RELOCATION_ARTICLES == frozenset({"a", "an", "the"})


# --------------------------------------------------------------------------------------
# The British/American spelling class is deleted outright, not narrowed. `_spelling_fold`,
# `_is_spelling_variant_pair` and their four fold-table constants no longer exist; a
# spelling difference refuses relocation the same way any other single-word difference
# does, and falls to `verify_claim_with_policy`'s bounded second pass instead
# (`test_verification_policy.py`).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new,sentence",
    [
        (
            "filed", "filled",
            "Eligible households filed for the benefit before the deadline closed for "
            "every participating region.",
        ),
        (
            "filing", "filling",
            "The filing of the standard form took less time than expected across every "
            "participating office this year.",
        ),
        (
            "polled", "poled",
            "The households polled in the second wave completed every scheduled follow-up "
            "session of the survey.",
        ),
        (
            "she", "shoe",
            "The interviewer noted that she declined to answer several questions during "
            "the second recorded session.",
        ),
        (
            "colour", "color",
            "The study found that participants who colour code their notes performed "
            "better on every measured outcome overall.",
        ),
        (
            "organisation", "organization",
            "The collaboration between the funding organisation and the university "
            "produced a shared dataset for every participating site.",
        ),
    ],
)
def test_deleted_spelling_class_never_relocates(old, new, sentence):
    """A doubled-l fold merged
    `filled`/`filed`, `filling`/`filing` and `polled`/`poled`; a digraph fold with no length
    or position guard merged `shoe`/`she`. Both folds, and the class itself, are deleted; a
    same-length substitution between any of these pairs -- content word or genuine spelling
    variant alike -- must now refuse relocation, whether the sentence uses the American or
    the British/short form."""
    from app.services.fulltext import relocate_evidence_quotes

    quote = sentence.replace(old, new, 1)
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [sentence])
    assert evidence_quote == quote
    assert diagnostics == []


# --------------------------------------------------------------------------------------
# Neutral class 2: a dropped parenthetical citation, requiring a citation *shape*, not
# merely a year.
# --------------------------------------------------------------------------------------


def test_token_is_citation_year_accepts_1900_to_2099_only():
    from app.services.fulltext import _token_is_citation_year

    for year in ("1900", "1987", "2020", "2099"):
        assert _token_is_citation_year(year), year
    for not_year in ("1899", "2100", "202", "20200", "20a0"):
        assert not _token_is_citation_year(not_year), not_year


def test_looks_like_citation_surname_requires_a_capitalised_letters_only_token():
    from app.services.fulltext import _looks_like_citation_surname

    for word in ("Smith", "Truscott", "Sheen", "Mao", "Lee", "Li"):
        assert _looks_like_citation_surname(word), word
    for word in ("smith", "SMITH", "2020", "O'Brien", "Al-Sayed", ""):
        assert not _looks_like_citation_surname(word), word


def test_run_contains_et_al_pair_requires_adjacent_et_al():
    from app.services.fulltext import _run_contains_et_al_pair

    assert _run_contains_et_al_pair(["Smith", "et", "al", "2020"])
    assert _run_contains_et_al_pair(["et", "al"])
    assert not _run_contains_et_al_pair(["Smith", "Jones"])
    assert not _run_contains_et_al_pair(["et", "cetera"])  # "et" not followed by "al"


def test_run_is_numeric_only_requires_every_token_to_be_digits():
    from app.services.fulltext import _run_is_numeric_only

    assert _run_is_numeric_only(["12"])
    assert _run_is_numeric_only(["3", "4"])
    assert not _run_is_numeric_only(["12", "to", "14"])
    assert not _run_is_numeric_only([])


def test_run_has_citation_shape_requires_year_and_surname_or_et_al_or_numeric_only():
    """A year alone is not a citation shape. `"(Mao, Lee and Li, 2024)"` relocates via year+surname even though
    `"and"` sits in the run beside the names."""
    from app.services.fulltext import _run_has_citation_shape

    assert _run_has_citation_shape(["Truscott", "1996"])
    assert _run_has_citation_shape(["Sheen", "et", "al", "2009"])
    assert _run_has_citation_shape(["Mao", "Lee", "and", "Li", "2024"])
    assert _run_has_citation_shape(["12"])
    assert _run_has_citation_shape(["3", "4"])
    for run in (
        ["no", "effect", "was", "seen", "after", "2019"],
        ["this", "reversed", "after", "2021", "in", "transport"],
        ["from", "42", "percent", "in", "2011", "to", "38", "percent"],
        ["1950", "to", "1990"],
    ):
        assert not _run_has_citation_shape(run), run


def test_run_has_disqualifying_content_ignores_the_runs_own_citation_year():
    """A recognised citation year is never itself disqualifying content, or no year-bearing
    citation could ever be accepted; every *other* digit, spelled-out number word or
    negation in the run still is, and a numeric-only run is exempt
    from this check entirely."""
    from app.services.fulltext import _run_has_disqualifying_content

    assert not _run_has_disqualifying_content(["Truscott", "1996"])
    assert not _run_has_disqualifying_content(["Sheen", "et", "al", "2009"])
    assert not _run_has_disqualifying_content(["Mao", "Lee", "and", "Li", "2024"])
    assert not _run_has_disqualifying_content(["12"])  # numeric-only: exempt
    assert _run_has_disqualifying_content(["Smith", "2020", "but", "not", "confirmed"])
    assert _run_has_disqualifying_content(["from", "42", "percent", "in", "2011"])


def test_run_sits_in_brackets_requires_the_matching_close():
    from app.services.fulltext import _run_sits_in_brackets

    text = "the effect was significant (Smith et al., 2020) across every site"
    start = text.index("Smith")
    end = start + len("Smith et al., 2020")
    assert _run_sits_in_brackets(text, start, end)

    square = "the effect was significant [Smith et al., 2020] across every site"
    start2 = square.index("Smith")
    end2 = start2 + len("Smith et al., 2020")
    assert _run_sits_in_brackets(square, start2, end2)

    mismatched = "the effect was significant (Smith et al., 2020] across every site"
    assert not _run_sits_in_brackets(mismatched, start, end)

    unbracketed = "the effect was significant Smith et al., 2020 across every site"
    start3 = unbracketed.index("Smith")
    end3 = start3 + len("Smith et al., 2020")
    assert not _run_sits_in_brackets(unbracketed, start3, end3)


def test_is_dropped_parenthetical_citation_end_to_end():
    from app.services.fulltext import _is_dropped_parenthetical_citation, _tokens_with_offsets

    chunk = "the effect was significant (Smith et al., 2020) across every site"
    tokens = _tokens_with_offsets(chunk)
    raw = [t for t, _s, _e in tokens]
    lo = raw.index("Smith")
    hi = raw.index("2020")
    assert _is_dropped_parenthetical_citation(raw, tokens, chunk, lo, hi)

    # Not bracketed: the same token run, unbracketed in the raw source.
    unbracketed = "the effect was significant Smith et al., 2020 across every site"
    tokens2 = _tokens_with_offsets(unbracketed)
    raw2 = [t for t, _s, _e in tokens2]
    lo2 = raw2.index("Smith")
    hi2 = raw2.index("2020")
    assert not _is_dropped_parenthetical_citation(raw2, tokens2, unbracketed, lo2, hi2)


# --------------------------------------------------------------------------------------
# relocate_evidence_quotes: class 1 (articles) relocates, alone and combined.
# --------------------------------------------------------------------------------------


def test_article_substitution_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("the entire", "a entire")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == source
    assert len(diagnostics) == 1 and diagnostics[0]["edit_distance"] == 1

    # Idempotent: relocating the already-relocated text again changes nothing.
    evidence_quote2, _q2, _a2, diagnostics2 = relocate_evidence_quotes(
        evidence_quote, None, [], [source]
    )
    assert evidence_quote2 == evidence_quote
    assert diagnostics2 == []


def test_article_insertion_relocates():
    """The source has an article the model's quote dropped."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("the entire sample", "entire sample")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == source
    assert len(diagnostics) == 1 and diagnostics[0]["edit_distance"] == 1


def test_article_deletion_relocates():
    """The model's quote has an article the source does not carry at that position."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across entire "
        "sample cohort"
    )
    quote = source.replace("across entire", "across the entire")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == source
    assert len(diagnostics) == 1 and diagnostics[0]["edit_distance"] == 1


def test_multiple_article_edits_in_one_quote_all_relocate_together():
    """Round 4's own effective rule: any *number* of neutral-class edits, not just one."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers noted that a control group and the treatment group both "
        "completed the entire assigned protocol during the trial"
    )
    quote = (
        source.replace("the researchers", "a researchers")
        .replace("a control", "the control")
        .replace("the entire", "an entire")
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == source
    assert len(diagnostics) == 1 and diagnostics[0]["edit_distance"] == 3



# --------------------------------------------------------------------------------------
# relocate_evidence_quotes: class 2 (a dropped parenthetical citation) relocates; an
# inserted one refuses.
# --------------------------------------------------------------------------------------


def test_dropped_parenthetical_citation_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "The intervention improved reading outcomes for every enrolled participant "
        "(Smith et al., 2020) across the full trial period"
    )
    quote = (
        "The intervention improved reading outcomes for every enrolled participant "
        "across the full trial period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert "(Smith et al., 2020)" in evidence_quote or "(Smith et al., 2020" in evidence_quote
    assert len(diagnostics) == 1 and diagnostics[0]["edit_distance"] == 1


def test_inserted_citation_never_relocates():
    """A citation the model's quote carries that the source does not have at all must never
    be waved through -- only "deletion, never insertion" of a citation is licensed."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "The intervention improved reading outcomes for every enrolled participant "
        "across the full trial period"
    )
    quote = (
        "The intervention improved reading outcomes for every enrolled participant "
        "(Smith et al., 2020) across the full trial period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_square_bracket_citation_also_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "The intervention improved reading outcomes for every enrolled participant "
        "[12, 2020] across the full trial period"
    )
    quote = (
        "The intervention improved reading outcomes for every enrolled participant "
        "across the full trial period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert len(diagnostics) == 1


def test_unbracketed_citation_like_text_never_relocates():
    """The digit inside the year is still protected as defence in depth when the run is not
    actually bracketed in the raw source (it is then just an ordinary dropped digit)."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "The intervention improved reading outcomes for every enrolled participant "
        "Smith et al 2020 across the full trial period"
    )
    quote = (
        "The intervention improved reading outcomes for every enrolled participant "
        "across the full trial period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


@pytest.mark.parametrize(
    "citation",
    [
        "(Truscott, 1996)",
        "(Sheen et al., 2009)",
        "(Mao, Lee and Li, 2024)",
        "[12]",
    ],
)
def test_citation_shape_relocates(citation):
    """A year with a
    surname beside it, an et al pair, an author list joined by "and", and a bare numbered
    reference all relocate under the citation-shape check."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        f"The intervention improved outcomes for every enrolled participant {citation} "
        "across the whole trial period"
    )
    quote = (
        "The intervention improved outcomes for every enrolled participant "
        "across the whole trial period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert citation in evidence_quote or citation[:-1] in evidence_quote
    assert len(diagnostics) == 1


@pytest.mark.parametrize(
    "aside",
    [
        "(no effect was seen after 2019)",
        "[this reversed after 2021 in transport]",
        "(from 42 percent in 2011 to 38 percent)",
        "(1950 to 1990)",
    ],
)
def test_non_citation_bracketed_aside_never_relocates(aside):
    """A bracketed run
    that merely contains a year, with no surname beside it, no et al pair and no all-digit
    shape, is not a citation -- and, since none of the four is bracketed with a genuine
    citation shape, each one refuses regardless of whether it also carries a negation or a
    non-year digit."""
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        f"The pooled results held for every enrolled participant {aside} across the "
        "whole trial period"
    )
    quote = (
        "The pooled results held for every enrolled participant across the whole trial "
        "period"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


# --------------------------------------------------------------------------------------
# The authorised motivating case no longer relocates via Part A (Part B resolves it now,
# see test_verification_policy.py), and every fabricated/meaning-changing edit still refuses.
# --------------------------------------------------------------------------------------

_WE_AND_CHUNK = (
    "We did not find any evidence of processing comprehensive CF forms being "
    "overloading and unwelcome so much so that corrections could not be processed and "
    "L2 learning could not take place."
)
_WE_AND_MODEL_QUOTE = (
    "We did not find any evidence of processing comprehensive CF forms being "
    "overloading we unwelcome so much so that corrections could not be processed and "
    "L2 learning could not take place"
)


def test_we_and_case_no_longer_relocates_via_part_a():
    """Neither "we" nor "and" is an article, a spelling variant or a citation, so
    Part A refuses this edit -- it is Part B's bounded second pass that resolves it
    (`test_verification_policy.py`)."""
    from app.services.fulltext import relocate_evidence_quotes

    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(
        _WE_AND_MODEL_QUOTE, [_WE_AND_MODEL_QUOTE], [], [_WE_AND_CHUNK]
    )
    assert evidence_quote == _WE_AND_MODEL_QUOTE
    assert diagnostics == []


def test_fabricated_quote_relocates_nowhere():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "The study reported that the intervention group scored higher than the "
        "control group on every measured outcome across all three sessions of the "
        "experiment."
    )
    fabricated = (
        "The purple elephants danced gracefully across the moonlit savanna while "
        "singing ancient songs of forgotten kingdoms long ago"
    )
    evidence_quote, _quotes, _assertions, diagnostics = relocate_evidence_quotes(
        fabricated, None, [], [chunk]
    )
    assert evidence_quote == fabricated
    assert diagnostics == []


def test_numeric_difference_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "The study reported that the intervention group scored higher than the "
        "control group on 3 measured outcomes across all recorded sessions of the "
        "experiment."
    )
    segment = (
        "the intervention group scored higher than the control group on 5 measured "
        "outcomes across all recorded sessions"
    )
    evidence_quote, _quotes, _assertions, diagnostics = relocate_evidence_quotes(
        segment, None, [], [chunk]
    )
    assert evidence_quote == segment
    assert diagnostics == []


def test_negation_difference_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "The authors did not find any significant improvement in reading "
        "comprehension across the treatment groups over the semester."
    )
    segment = (
        "the authors did find any significant improvement in reading comprehension "
        "across the treatment groups"
    )
    evidence_quote, _quotes, _assertions, diagnostics = relocate_evidence_quotes(
        segment, None, [], [chunk]
    )
    assert evidence_quote == segment
    assert diagnostics == []


def test_neither_nor_case_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "Neither the intervention group nor the control group showed any measurable "
        "improvement over the semester."
    )
    segment = (
        "either the intervention group nor the control group showed any measurable "
        "improvement"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_inserted_non_case_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "The difference between the two groups was non-significant after adjusting for "
        "baseline reading ability."
    )
    segment = (
        "the difference between the two groups was significant after adjusting for "
        "baseline reading ability"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_negation_prefix_substitution_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "Reading comprehension scores remained unchanged across the semester for every "
        "cohort in the study."
    )
    segment = (
        "reading comprehension scores remained changed across the semester for every "
        "cohort in the study"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_lack_and_fail_prefixed_negations_never_relocate():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "The intervention group failed to produce any measurable improvement over the "
        "semester in reading fluency."
    )
    segment = (
        "the intervention group managed to produce any measurable improvement over the "
        "semester in reading fluency"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_spelled_out_number_difference_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    chunk = (
        "Participants completed three follow-up sessions during the intervention period "
        "of the trial."
    )
    segment = (
        "participants completed five follow-up sessions during the intervention period "
        "of the trial"
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_sixty_ninety_decade_pair_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "In the pooled analysis sixty of the recruited participants completed every "
        "follow-up session of the trial"
    )
    quote = source.replace("sixty", "ninety")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_homoglyph_negation_never_relocates():
    """A Cyrillic "o" (U+043E) in place of the Latin "o" inside "not" is not recognised by
    `_is_negation_token`, but it is also simply not an article, a spelling variant or a
    citation, so the closed neutral classes reject it regardless of script."""
    from app.services.fulltext import _is_negation_token, relocate_evidence_quotes

    homoglyph_not = "n" + "о" + "t"
    assert not _is_negation_token(homoglyph_not)  # the gap the closed classes cover anyway

    source = (
        "the association was clearly significant in the fully adjusted regression model "
        "overall"
    )
    quote = source.replace("clearly", homoglyph_not)
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []

    ascii_quote = source.replace("clearly", "not")
    evidence_quote2, _q2, _a2, diagnostics2 = relocate_evidence_quotes(
        ascii_quote, None, [], [source]
    )
    assert evidence_quote2 == ascii_quote
    assert diagnostics2 == []


def test_content_word_insertion_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("consistent across", "consistent significantly across")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_content_word_deletion_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("entire sample cohort", "sample cohort")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_modal_substitution_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the treatment is effective for reducing symptoms across the entire patient "
        "cohort"
    )
    quote = source.replace(" is ", " must ")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_modal_insertion_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the treatment is effective for reducing symptoms across the entire patient "
        "cohort"
    )
    quote = source.replace("is effective", "should be effective")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_increased_reduced_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the intervention increased the primary outcome across every session of the "
        "trial period"
    )
    quote = source.replace("increased", "reduced")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_higher_lower_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the treatment group reported higher satisfaction than the control group "
        "after the trial"
    )
    quote = source.replace("higher", "lower")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_may_does_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the association may reflect residual confounding across the entire sampled "
        "population overall"
    )
    quote = source.replace("may", "does")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_dropped_only_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the improvement was only marginally larger than the baseline measurement "
        "recorded overall"
    )
    quote = source.replace("only ", "")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_rose_fell_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "attendance rose sharply in the treatment arm throughout the entire semester "
        "period"
    )
    quote = source.replace("rose", "fell")
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


def test_eighty_eight_token_triple_flip_never_relocates():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "In the pooled analysis across all recruited sites the intervention increased the "
        "primary outcome relative to the comparison group, and participants in the "
        "treatment arm reported higher satisfaction than those assigned to the control "
        "condition throughout the full follow-up period. The authors note that the "
        "observed association may reflect a genuine causal pathway rather than any "
        "residual confounding left over from the original study design."
    )
    quote = (
        source.replace("increased", "reduced").replace("higher", "lower").replace("may", "does")
    )
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [source])
    assert evidence_quote == quote
    assert diagnostics == []


# --------------------------------------------------------------------------------------
# Measured antonym-pair table: every one of these must still refuse
# under the closed three-class rule (none is an article, a spelling variant or a citation).
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new,sentence",
    [
        (
            "over", "under",
            "Participation rates were measured over 30 days and the reported gains "
            "persisted at follow up for every enrolled cohort.",
        ),
        (
            "before", "after",
            "Symptoms were assessed before the intervention began for every enrolled "
            "participant across all recruited sites.",
        ),
        (
            "we", "they",
            "We report that the intervention improved outcomes for every enrolled "
            "participant across the full trial period.",
        ),
        (
            "is", "was",
            "The treatment is effective for reducing symptoms across the entire patient "
            "cohort throughout the whole study period.",
        ),
        (
            "up", "down",
            "Attendance moved up sharply in the treatment arm throughout the entire "
            "semester period for every cohort tested.",
        ),
        (
            "to", "from",
            "Participants were referred to the clinic for every scheduled appointment "
            "across the entire study period overall.",
        ),
    ],
)
def test_round3_antonym_pairs_never_relocate(old, new, sentence):
    from app.services.fulltext import relocate_evidence_quotes

    quote = sentence.replace(old, new, 1)
    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(quote, None, [], [sentence])
    assert evidence_quote == quote
    assert diagnostics == []


def test_segment_under_eight_tokens_never_relocates():
    from app.services.fulltext import (
        QUOTE_RELOCATION_MIN_SEGMENT_TOKENS,
        _tokens_with_offsets,
        relocate_evidence_quotes,
    )

    chunk = (
        "The extraordinarily comprehensively documented outcomes demonstrated "
        "substantial improvement overall."
    )
    segment = "extraordinarily comprehensively documented outcomes revealed"
    assert len(_tokens_with_offsets(segment)) < QUOTE_RELOCATION_MIN_SEGMENT_TOKENS
    assert len(segment) >= 30  # clears `_quote_segments`' own floor; only the token floor bites

    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_relocation_never_shrinks_the_models_own_window():
    """A window may only stay the same length as the segment or grow, never shrink -- an
    accepted window one token shorter than the model's own segment would silently truncate
    the stored quote."""
    from app.services.fulltext import _tokens_with_offsets, relocate_evidence_quotes

    segment = "the researchers reported that the effect was consistent across a entire today"
    chunk = "The researchers reported that the effect was consistent across the entire."
    assert len(_tokens_with_offsets(segment)) == 12
    assert len(_tokens_with_offsets(chunk)) == 11

    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert evidence_quote == segment
    assert diagnostics == []


def test_relocation_is_idempotent():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("the entire", "a entire")
    assertion = {"quote": quote, "quotes": [quote]}
    first = relocate_evidence_quotes(quote, [quote], [assertion], [source])
    evidence_quote, evidence_quotes, assertions, diagnostics = first
    assert diagnostics  # the first pass actually changed something

    second = relocate_evidence_quotes(evidence_quote, evidence_quotes, assertions, [source])
    evidence_quote2, evidence_quotes2, assertions2, diagnostics2 = second
    assert evidence_quote2 == evidence_quote
    assert evidence_quotes2 == evidence_quotes
    assert assertions2 == assertions
    assert diagnostics2 == []


def test_relocation_absorbs_a_trailing_combining_mark():
    """A combining mark is neither a letter nor a digit, so it ends a token run the same way
    whitespace does and can sit immediately after a window's last token. Without absorbing
    it, the recovered raw slice ends with the base letter while the chunk's own
    `_normalise_for_match` fold has already composed base plus mark into one letter under
    NFKC, so "verbatim by construction" would not actually be verbatim, and idempotence
    would break (a second pass would "relocate" again)."""
    import unicodedata

    from app.services.fulltext import _normalise_for_match, relocate_evidence_quotes

    decomposed_e_acute = unicodedata.normalize("NFD", "é")
    chunk = (
        "The interviews for this study took place in a small local caf"
        + decomposed_e_acute
        + " near the campus."
    )
    segment = "the interviews for this study took place in a small local cafe"

    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(segment, None, [], [chunk])
    assert diagnostics and diagnostics[0]["replacement"].endswith("caf" + decomposed_e_acute)
    assert _normalise_for_match(evidence_quote) in _normalise_for_match(chunk)

    evidence_quote2, _q2, _a2, diagnostics2 = relocate_evidence_quotes(
        evidence_quote, None, [], [chunk]
    )
    assert evidence_quote2 == evidence_quote
    assert diagnostics2 == []


def test_relocation_covers_assertion_quote_and_quotes_fields():
    from app.services.fulltext import relocate_evidence_quotes

    source = (
        "the researchers reported that the effect was consistent across the entire "
        "sample cohort"
    )
    quote = source.replace("the entire", "a entire")
    assertions_in = [
        {
            "text": "t",
            "kind": "other",
            "verdict": "supported",
            "quote": quote,
            "quotes": [quote],
        }
    ]
    _eq, _eqs, assertions_out, diagnostics = relocate_evidence_quotes(
        None, None, assertions_in, [source]
    )
    assert assertions_out[0]["quote"] == source
    assert assertions_out[0]["quotes"][0] == assertions_out[0]["quote"]
    assert assertions_out[0]["text"] == "t" and assertions_out[0]["kind"] == "other"
    assert len(diagnostics) == 2  # one for "quote", one for "quotes"[0]
    assert assertions_in[0]["quote"] == quote  # nothing is mutated in place


def test_relocation_leaves_none_and_empty_inputs_alone():
    from app.services.fulltext import relocate_evidence_quotes

    evidence_quote, evidence_quotes, assertions, diagnostics = relocate_evidence_quotes(
        None, None, None, [_WE_AND_CHUNK]
    )
    assert evidence_quote is None and evidence_quotes is None
    assert assertions == [] and diagnostics == []

    evidence_quote, evidence_quotes, assertions, diagnostics = relocate_evidence_quotes(
        None, [], [], []
    )
    assert evidence_quote is None and evidence_quotes == [] and assertions == []
    assert diagnostics == []


# --------------------------------------------------------------------------------------
# A splice must never change how `_quote_segments` re-parses some OTHER,
# untouched span of the same quote (e.g. by shifting the count of quote-delimiter
# characters).
# --------------------------------------------------------------------------------------


def test_delimiter_crossing_window_is_refused_so_other_fragments_survive_unchanged():
    """A measured case: the recovered raw slice for the first
    fragment would cross a curly quote inside the source (an article typo, "the" for "a",
    right where the source happens to open a quoted aside), which would re-pair every
    quotation mark after it under `_QUOTED_FRAGMENT_RE`. The window must be refused, leaving
    the whole quote -- both fragments -- untouched."""
    from app.services.fulltext import _quote_segments, relocate_evidence_quotes

    chunk = (
        "The authors describe the programme as a “whole school approach that "
        "reaches every classroom” and they report sustained gains at each of the "
        "sites."
    )
    model_quote = (
        'The paper states "The authors describe a programme as a whole school approach '
        'that reaches every classroom" and also "The programme eliminated every reported '
        'barrier to participation across all sites nationwide."'
    )
    segments_before = _quote_segments(model_quote)
    assert len(segments_before) == 2  # both fragments cross the 30-character floor

    evidence_quote, _q, _a, diagnostics = relocate_evidence_quotes(
        model_quote, None, [], [chunk]
    )
    assert evidence_quote == model_quote
    assert diagnostics == []
    # The post-condition holds trivially here (nothing changed), but restated directly:
    # every original segment is still exactly recoverable.
    assert _quote_segments(evidence_quote) == segments_before


def test_relocation_never_shrinks_a_windows_delimiter_count_below_the_segments():
    """A synthetic, more direct check of `_best_relocation_window`'s own delimiter-count
    refusal: a window whose raw slice would carry a quote-delimiter character the segment
    itself does not have is never accepted, even when the edit itself (an article
    substitution) would otherwise be licensed.

    The check used to be one-sided
    (``if replacement_delimiters and replacement_delimiters != segment_delimiters``), so the
    mirror image -- a window whose raw slice would carry *fewer* delimiters than the segment
    itself -- was not refused (``replacement_delimiters`` empty, hence falsy, short-circuited
    the ``and`` before the counts were ever compared). Both directions are exercised
    directly, since `_QUOTED_FRAGMENT_RE`'s own capture group cannot itself contain a
    delimiter character in real usage."""
    from app.services.fulltext import _best_relocation_window

    chunk = 'the effect was a “whole school approach that reached every classroom today'
    segment = "the effect was the whole school approach that reached every classroom"
    # Would otherwise relocate (a single article substitution, "the" for "a"), but the
    # recovered window's raw slice would include the un-matched opening curly quote.
    assert _best_relocation_window(segment, [chunk]) is None

    # Mirror image: the segment itself carries the curly quote the chunk's own matching
    # window would not recover: the window's raw slice would carry *fewer* delimiters than
    # the segment, which a one-sided check would otherwise let through.
    mirror_chunk = "the effect was the whole school approach that reached every classroom today"
    mirror_segment = "the effect was a “whole school approach that reached every classroom"
    assert _best_relocation_window(mirror_segment, [mirror_chunk]) is None


# --------------------------------------------------------------------------------------
# apply_verification_guards is untouched by this feature: same inputs, same outputs, and
# the frozen digest still pins the same literal `test_verification_guard_functions_are_
# unchanged` / `test_verification_guards_are_still_frozen` compute independently.
# --------------------------------------------------------------------------------------


def test_apply_verification_guards_behaviour_is_unchanged_for_a_quote_relocation_never_touches():
    from app.services.fulltext import apply_verification_guards

    chunk = "The intervention improved reading fluency for every participant in the study."
    for quote in (
        "The intervention improved reading fluency for every participant in the study.",
        "A quote that never appears anywhere in the source text at all, invented whole.",
    ):
        result = apply_verification_guards(
            "verified",
            claim_text="x",
            evidence_quote=quote,
            evidence_quotes=None,
            assertions=[],
            chunk_texts=[chunk],
            chunks=[{"text": chunk}],
        )
        expected_status = "verified" if quote.startswith("The intervention") else "needs_nuance"
        assert result[0] == expected_status


def test_guard_digest_constant_matches_the_frozen_literal():
    from app.services.fulltext import GUARD_DIGEST

    assert GUARD_DIGEST == "8a6c833ffc329f89"


def test_quote_relocation_version_is_a_stable_16_character_hex_digest():
    from app.services.fulltext import QUOTE_RELOCATION_VERSION

    assert len(QUOTE_RELOCATION_VERSION) == 16
    int(QUOTE_RELOCATION_VERSION, 16)  # raises ValueError if not hex


# --------------------------------------------------------------------------------------
# QUOTE_RELOCATION_VERSION hashes the whole relocation set -- every helper, every
# threshold, every word list and structural constant a decision passes through, not just
# `relocate_evidence_quotes`'s own wrapper source.
# --------------------------------------------------------------------------------------


def test_quote_relocation_version_changes_when_the_articles_change(monkeypatch):
    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "_RELOCATION_ARTICLES", frozenset(ft._RELOCATION_ARTICLES - {"an"}))
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_quote_relocation_version_changes_when_a_threshold_changes(monkeypatch):
    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "QUOTE_RELOCATION_MAX_CITATION_TOKENS", 20)
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_quote_relocation_version_changes_when_number_words_changes(monkeypatch):
    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "_NUMBER_WORDS", frozenset(ft._NUMBER_WORDS - {"sixty"}))
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_quote_relocation_version_changes_when_the_citation_year_pattern_changes(monkeypatch):
    """`_CITATION_YEAR_RE`'s pattern is
    referenced by name inside `_token_is_citation_year`'s body, so its *value* is not on that
    function's own source line and must be hashed by `repr`, in the shape of `test_quote_
    relocation_version_changes_when_the_token_apostrophes_change` below. Rebinding it to a
    wider pattern (e.g. any four digits) would otherwise leave the digest unchanged."""
    import re

    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "_CITATION_YEAR_RE", re.compile(r"^\d{4}$"))
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_quote_relocation_version_changes_when_the_citation_surname_pattern_changes(monkeypatch):
    """A constant hashed from the start (unlike `_CITATION_YEAR_RE`, which shipped
    unhashed)."""
    import re

    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "_CITATION_SURNAME_RE", re.compile(r"^[A-Z][A-Za-z]*$"))
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_quote_relocation_version_changes_when_the_token_apostrophes_change(monkeypatch):
    """`_TOKEN_APOSTROPHES` decides where `_tokens_with_offsets` breaks a
    token (and, through `_is_negation_token`, which "n't" contractions are protected) but is
    not reachable through any hashed function's own source line."""
    import app.services.fulltext as ft

    original = ft._compute_quote_relocation_version()
    monkeypatch.setattr(ft, "_TOKEN_APOSTROPHES", ("'",))
    changed = ft._compute_quote_relocation_version()
    assert changed != original


def test_compute_guard_digest_falls_back_to_none_when_source_is_unavailable(monkeypatch, caplog):
    import app.services.fulltext as ft

    def _raise(*_args, **_kwargs):
        raise OSError("source not available")

    monkeypatch.setattr(ft.inspect, "getsource", _raise)
    with caplog.at_level("WARNING"):
        result = ft._compute_guard_digest()
    assert result is None
    assert "GUARD_DIGEST" in caplog.text


def test_compute_quote_relocation_version_falls_back_to_none_when_source_is_unavailable(
    monkeypatch, caplog
):
    import app.services.fulltext as ft

    def _raise(*_args, **_kwargs):
        raise OSError("source not available")

    monkeypatch.setattr(ft.inspect, "getsource", _raise)
    with caplog.at_level("WARNING"):
        result = ft._compute_quote_relocation_version()
    assert result is None
    assert "QUOTE_RELOCATION_VERSION" in caplog.text


# --------------------------------------------------------------------------------------
# `attribution_guard_fires_before_relocation`: relocation must not manufacture guard 1's own
# located-quote exemption. A caller checks this, on the pre-relocation quote, before calling
# `relocate_evidence_quotes` at all (now: inside `verify_claim_with_policy`).
# --------------------------------------------------------------------------------------


def test_attribution_guard_fires_before_relocation_matches_the_measured_case():
    from app.services.fulltext import attribution_guard_fires_before_relocation

    assert attribution_guard_fires_before_relocation(
        "Researchers reported this pattern (Smith, 2020).",
        [_WE_AND_CHUNK],
        _WE_AND_MODEL_QUOTE,
    )


def test_attribution_guard_fires_before_relocation_is_false_when_terms_are_covered():
    from app.services.fulltext import attribution_guard_fires_before_relocation

    assert not attribution_guard_fires_before_relocation(
        "Corrections could not be processed and L2 learning could not take place "
        "(Smith, 2020).",
        [_WE_AND_CHUNK],
        _WE_AND_MODEL_QUOTE,
    )


# --------------------------------------------------------------------------------------
# End to end: the production caller (`_verify_claims`, via `verify_and_heal_claims`)
# actually relocates before guarding, stores the relocated text, and appends the
# "quote_relocated" slug to `diagnostics` without losing the guards' own entries.
# --------------------------------------------------------------------------------------

import os  # noqa: E402

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.models.analysis_job import JobType  # noqa: E402
from app.schemas.fulltext import ClaimVerification  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    FakeAgent,
    make_session_factory,
    seed_draft,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
    tiptap_paragraph,
)

_RELOCATION_CHUNK_TEXT = (
    "Corrections could not be processed and the L2 learning could not take place for "
    "every enrolled participant across the whole trial period at every recruited site."
)
_RELOCATION_MODEL_QUOTE = _RELOCATION_CHUNK_TEXT.replace(
    "and the L2 learning", "and a L2 learning"
)
_CHUNKS = [{"section": "results", "text": _RELOCATION_CHUNK_TEXT}]

# The claim text must share enough vocabulary with the chunk that guard 1 (attribution)
# does not fire on it -- otherwise this end-to-end test would be exercising the attribution
# pre-check, not the relocation-rescue path it is meant to isolate. Every content word here
# is drawn straight from `_RELOCATION_CHUNK_TEXT`, so coverage is 1.0 regardless of whether
# the (still not located, pre-relocation) quote is located.
_RELOCATION_CLAIM_TEXT = (
    "Corrections could not be processed and the L2 learning could not take place "
    "(Smith, 2020)."
)


async def _seed_relocation_world(db_session):
    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": _CHUNKS},
    )
    draft = await seed_draft(
        db_session,
        project,
        user,
        tiptap_paragraph(_RELOCATION_CLAIM_TEXT),
    )
    job = await seed_job(db_session, project, JobType.claim_verify)
    return project, paper, draft, job


@pytest.mark.asyncio
async def test_verify_and_heal_claims_relocates_before_guarding(db_session):
    from app.services.fulltext import verify_and_heal_claims

    project, paper, draft, job = await _seed_relocation_world(db_session)
    agent = FakeAgent(
        output=ClaimVerification(
            claim_text=_RELOCATION_CLAIM_TEXT,
            paper_id=paper.id,
            status="verified",
            evidence_quote=_RELOCATION_MODEL_QUOTE,
            explanation="Matches the source's own conclusion.",
        )
    )

    factory, engine = make_session_factory()
    from unittest.mock import patch

    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.error is None, job.error
    (verification,) = job.result["verifications"]
    # The guard would have demoted the model's raw quote (`quote_not_verbatim`); relocation
    # ran first (an article, "a" for "the"), so the stored, final status is `verified` with
    # no guard reason fired -- and, since the final status is `verified`, not `needs_nuance`,
    # Part B's second pass never fires either (only one model call is made).
    assert len(agent.calls) == 1
    assert verification["model_status"] == "verified"
    assert verification["status"] == "verified"
    assert verification["machine_reasons"] == []
    assert verification["evidence_quote"] != _RELOCATION_MODEL_QUOTE
    # The window's raw slice ends at the last token's own end offset, not past the source's
    # trailing full stop (no token, so nothing to extend to) -- verbatim under the same fold
    # `_guard_quote_fidelity` itself uses, not necessarily byte-identical to the whole chunk.
    from app.services.fulltext import _normalise_for_match

    assert _normalise_for_match(verification["evidence_quote"]) == _normalise_for_match(
        _RELOCATION_CHUNK_TEXT
    )
    assert "quote_relocated" in verification["diagnostics"]
    assert len(verification["quote_relocations"]) == 1
    record = verification["quote_relocations"][0]
    assert record["type"] == "quote_relocated" and record["chunk_index"] == 0
    assert record["original"] == _RELOCATION_MODEL_QUOTE
    assert verification["evidence_location"] == "chunk 1 (results)"
    assert len(verification["passes"]) == 1
    assert verification["passes"][0]["machine_reasons"] == []

    provenance = job.result["provenance"]
    assert provenance["guard_digest"] == "8a6c833ffc329f89"
    assert len(provenance["quote_relocation_version"]) == 16


@pytest.mark.asyncio
async def test_verify_and_heal_claims_skips_relocation_when_attribution_already_fires(
    db_session,
):
    """A claim whose content has essentially no overlap with the cited chunk (a wrong-paper
    citation), and a quote that is the chunk's own sentence with one word changed.
    Relocating that quote would locate it and stand down guard 1's located-quote exemption
    for exactly the class of claim the guard exists to catch, so the pre-check must skip
    relocation for this claim, leaving guard 1 to fire on the model's own, unrelocated
    quote. Guard 1 caps the status to `unsupported`, not `needs_nuance`, so Part B's second
    pass never fires either (only one model call is made)."""
    from app.services.fulltext import verify_and_heal_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": _CHUNKS},
    )
    wrong_paper_claim = "Researchers reported this pattern (Smith, 2020)."
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(wrong_paper_claim))
    job = await seed_job(db_session, project, JobType.claim_verify)

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text=wrong_paper_claim,
            paper_id=paper.id,
            status="verified",
            evidence_quote=_RELOCATION_MODEL_QUOTE,
            explanation="Matches the source's own conclusion.",
        )
    )

    factory, engine = make_session_factory()
    from unittest.mock import patch

    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.error is None, job.error
    (verification,) = job.result["verifications"]
    assert len(agent.calls) == 1
    assert verification["model_status"] == "verified"
    assert verification["status"] == "unsupported"
    assert verification["machine_reasons"] == ["attribution_mismatch"]
    assert verification["evidence_quote"] == _RELOCATION_MODEL_QUOTE
    assert verification["quote_relocations"] == []
    assert "quote_relocated" not in verification["diagnostics"]
    assert len(verification["passes"]) == 1
