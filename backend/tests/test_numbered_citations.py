"""Numbered-citation resolution inside `extract_claims_from_document`.

All pure-function tests against plain Tiptap-document dicts; no network, no database.
"""

from types import SimpleNamespace

import pytest

from app.services.fulltext import extract_claims_from_document


def _paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def _heading(text: str) -> dict:
    return {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [{"type": "text", "text": text}],
    }


def _doc(body: list[str], reference_heading: str, entries: list[str]) -> dict:
    nodes = [_paragraph(t) for t in body]
    nodes.append(_heading(reference_heading))
    nodes.extend(_paragraph(e) for e in entries)
    return {"type": "doc", "content": nodes}


def paper(authors, year, doi=None):
    return SimpleNamespace(authors=authors, year=year, doi=doi)


# --------------------------------------------------------------------------------------
# The forms
# --------------------------------------------------------------------------------------


def test_bracket_single_number():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        ["Tutoring improves outcomes [1]."],
        "References",
        ["1. Smith, J. (2020). A study of tutoring."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Tutoring improves outcomes [1].",
            "smith_2020",
            "[1]",
            "Tutoring improves outcomes [1].",
        )
    ]
    assert coverage["by_source"]["numbered"] == 1
    assert coverage["linked"] == 1
    assert coverage["unresolved"] == 0


def test_bracket_comma_list_cites_every_number():
    lookup = {
        "smith_2020": paper([{"name": "Jane Smith"}], 2020),
        "jones_2021": paper([{"name": "Ken Jones"}], 2021),
    }
    doc = _doc(
        ["Two studies agree [1, 2]."],
        "References",
        ["1. Smith, J. (2020). A study.", "2. Jones, K. (2021). Another study."],
    )
    claims, _coverage = extract_claims_from_document(doc, lookup)
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("smith_2020", "[1, 2]"),
        ("jones_2021", "[1, 2]"),
    }


def test_bracket_range_and_en_dash_variant_expand_every_number():
    lookup = {
        "one_2018": paper([{"name": "A. One"}], 2018),
        "two_2019": paper([{"name": "B. Two"}], 2019),
        "three_2020": paper([{"name": "C. Three"}], 2020),
    }
    entries = [
        "1. One, A. (2018). Study one.",
        "2. Two, B. (2019). Study two.",
        "3. Three, C. (2020). Study three.",
    ]
    doc = _doc(["Range hyphen [1-3]."], "References", entries)
    claims, _c = extract_claims_from_document(doc, lookup)
    assert {k for _s, k, _c, _cs in claims} == {"one_2018", "two_2019", "three_2020"}

    doc_en_dash = _doc(["Range en dash [1–3]."], "References", entries)
    claims2, _c2 = extract_claims_from_document(doc_en_dash, lookup)
    assert {k for _s, k, _c, _cs in claims2} == {"one_2018", "two_2019", "three_2020"}


def test_superscript_digits():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    entries = [f"{i}. Other{i}, X. (2019). Unrelated." for i in range(1, 12)]
    entries.append("12. Smith, J. (2020). A study.")
    doc = _doc(["Effects were shown¹²."], "References", entries)
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert ("Effects were shown¹².", "smith_2020", "¹²", "Effects were shown¹².") in claims
    assert coverage["by_source"]["numbered"] == 1


# --------------------------------------------------------------------------------------
# An ordinary mathematical exponent ("R²",
# "chi²", "km²") must never be counted as a numbered citation, and must never disable a
# working resolver for a genuine numbered citation elsewhere in the same document.
# --------------------------------------------------------------------------------------


def test_ordinary_superscript_exponent_with_no_reference_list_is_not_a_citation():
    """Reproduces the review's first draft: no reference list at all, and the only
    superscript in the text is an R-squared value, not a citation. Before the fix this
    reported `found: 1, unresolved: 1, unresolved_citations: ["²"]`."""
    lookup: dict = {}
    doc = {
        "type": "doc",
        "content": [
            _paragraph("The model explained most of the variance (R² = 0.45) in the outcome.")
        ],
    }
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0
    assert coverage["unresolved_citations"] == []


def test_ordinary_superscript_exponent_does_not_disable_a_working_numbered_resolver():
    """Reproduces the review's second draft: a genuine `[1]` alongside an unrelated R²
    value, with a one-entry, contiguous reference list. Before the fix, the bogus
    superscript "citation" number 2 exceeded the one-entry list's `max_entry`, switching
    the whole resolver off and reporting the genuine `[1]` unresolved too."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        [
            "Prior work reported the effect [1].",
            "Variance explained was high (R² = 0.45).",
        ],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Prior work reported the effect [1].",
            "smith_2020",
            "[1]",
            "Prior work reported the effect [1].",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["unresolved"] == 0


# --------------------------------------------------------------------------------------
# A NEGATIVE mathematical exponent, written with a real superscript minus sign
# ("10⁻³"), must not be counted as a citation: `_SUPERSCRIPT_DIGITS_RE` matches
# superscript digits only, so the text immediately before the digit run ends in the
# superscript minus, which `[A-Za-z0-9]+$` alone cannot match.
# --------------------------------------------------------------------------------------


def test_negative_exponent_with_a_superscript_minus_sign_is_not_a_citation():
    """"p < 10⁻³" used to report `found: 1, unresolved: 1,
    unresolved_citations: ["³"]` -- a draft with no citation at all was told it had one
    unresolved reference."""
    lookup: dict = {}
    doc = {
        "type": "doc",
        "content": [_paragraph("Significance was p < 10⁻³ across runs.")],
    }
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0
    assert coverage["unresolved_citations"] == []


def test_negative_exponent_after_a_multiplication_sign_is_not_a_citation():
    lookup: dict = {}
    doc = {
        "type": "doc",
        "content": [_paragraph("Concentration fell to 5 x 10⁻⁶ mol.")],
    }
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


def test_negative_exponent_does_not_disable_a_genuine_bracket_citation_alongside_it():
    """A genuine `[1]`, resolved by a one-entry reference list, used to be reported
    unresolved too: the bogus "citation" number 3 from "10⁻³" exceeded the one-entry
    list's `max_entry` and switched the whole resolver off."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        [
            "Prior work reported the effect [1].",
            "Significance was p < 10⁻³ across runs.",
        ],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Prior work reported the effect [1].",
            "smith_2020",
            "[1]",
            "Prior work reported the effect [1].",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["unresolved"] == 0


# --------------------------------------------------------------------------------------
# The exponent-base test must be shape based, not
# length based -- a length threshold cannot tell "chi" (an exponent base) apart from
# "men" (an ordinary word), both three characters. A genuine Vancouver superscript
# directly after a short word must still be reported as a citation.
# --------------------------------------------------------------------------------------


def test_short_ordinary_word_before_a_superscript_is_still_a_citation():
    """"men³" used to be swallowed by the old length-based rule (any alnum token of
    three characters or fewer read as an exponent base); "men" is an ordinary clause
    ending, not an exponent's base, so the superscript is a real citation."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    entries = [f"{i}. Other{i}, X. (2019). Unrelated." for i in range(1, 3)]
    entries.append("3. Smith, J. (2020). A study.")
    doc = _doc(["This was shown in men³."], "References", entries)
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [("This was shown in men³.", "smith_2020", "³", "This was shown in men³.")]
    assert coverage["by_source"]["numbered"] == 1
    assert coverage["unresolved"] == 0


@pytest.mark.parametrize(
    "sentence",
    [
        "The model explained most of the variance (R² = 0.45) in the outcome.",
        "The measured area was several km² in total.",
        "The chi² statistic was significant at p < 0.05.",
    ],
)
def test_single_letter_and_unit_bases_are_kept_as_exponents(sentence):
    """A single variable letter ("R²") and a short unit or symbol abbreviation ("km²",
    "chi²") are all exponent bases, not citations -- shape based, not length based
    (minor finding 1), must not regress any of them. The numeral base case ("10⁻³") is
    covered above, by the sign-based half of major finding 1's fix."""
    lookup: dict = {}
    doc = {"type": "doc", "content": [_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


# --------------------------------------------------------------------------------------
# An early allowlist covered only "km" and "cm";
# every other common two-letter metric unit must not be misread as a citation superscript.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The cross-section was 5 mm² in area.",
        "Density was 3 nm³ per cell.",
        "The dose was 5 ml² equivalent.",
        "Mass scaled with kg² units.",
        "Terrain covered 2 ha² zones.",
        "Length measured 4 ft² planks.",
        "Distance was 6 in² blocks.",
        "Size was 1 µm² wide.",
    ],
)
def test_common_two_letter_metric_units_are_kept_as_exponents(sentence):
    """"mm²", "nm³", "ml²", "kg²", "ha²", "ft²", "in²" and "µm²" are all common metric
    units, not citations -- the allowlist names only "km" and "cm" as examples of
    the shape, not an exhaustive list, and every other short unit abbreviation fell
    through to being read as a citation superscript."""
    lookup: dict = {}
    doc = {"type": "doc", "content": [_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


# --------------------------------------------------------------------------------------
# `_build_paper_lookup` keeps a first author's
# diacritics ("García, M." -> "garcía_2020"), while the reference entry's own key folds
# them away ("garcia_2020") -- the resolver must try both forms.
# --------------------------------------------------------------------------------------


def test_numbered_citation_resolves_an_accented_first_author_surname():
    lookup = {"garcía_2020": paper([{"name": "García, M."}], 2020)}
    doc = _doc(
        ["Prior work supports this claim [1]."],
        "References",
        ["1. García, M. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Prior work supports this claim [1].",
            "garcía_2020",
            "[1]",
            "Prior work supports this claim [1].",
        )
    ]
    assert coverage["linked"] == 1
    assert coverage["unresolved"] == 0


def test_bare_paren_form_resolves_at_or_below_the_highest_entry():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        ["Tutoring improves outcomes (1)."],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Tutoring improves outcomes (1).",
            "smith_2020",
            "(1)",
            "Tutoring improves outcomes (1).",
        )
    ]
    assert coverage["by_source"]["numbered"] == 1


def test_bare_paren_form_never_fires_above_the_highest_entry_number():
    """A `(12)` whose number exceeds every reference entry is never treated as a
    citation at all -- not even counted as unresolved, since nothing here suggests it
    is a citation rather than ordinary parenthetical prose."""
    lookup: dict = {}
    doc = _doc(
        ["The sample size was large (12)."],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


def test_bare_paren_form_never_matches_a_four_digit_year():
    """`(2020)` reads as a bracketed year, not entry #2020; the pattern itself caps at
    3 digits, so this is never even a numbered-citation candidate."""
    lookup: dict = {}
    doc = _doc(
        ["This was already known (2020)."],
        "References",
        [f"{i}. Author{i}, A. ({2010 + i}). Study." for i in range(1, 6)],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


# --------------------------------------------------------------------------------------
# Resolution: DOI first, then surname/year
# --------------------------------------------------------------------------------------


def test_doi_first_resolution():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020, doi="10.1/xyz")}
    doc = _doc(
        ["Tutoring improves outcomes [1]."],
        "References",
        ["1. Smith, J. (2020). A study. https://doi.org/10.1/XYZ"],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Tutoring improves outcomes [1].",
            "smith_2020",
            "[1]",
            "Tutoring improves outcomes [1].",
        )
    ]
    assert coverage["linked"] == 1


def test_surname_year_resolution_when_no_doi_present():
    lookup = {"dijk_2015": paper([{"name": "Teun van Dijk"}], 2015)}
    doc = _doc(
        ["Discourse matters [1]."],
        "References",
        ["1. van Dijk, T. (2015). Discourse studies."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [("Discourse matters [1].", "dijk_2015", "[1]", "Discourse matters [1].")]
    assert coverage["linked"] == 1


def test_entry_with_no_recoverable_author_or_year_is_unresolved_not_dropped_silently():
    lookup: dict = {}
    doc = _doc(
        ["Tutoring improves outcomes [1]."],
        "References",
        ["1. (no parseable author or year at all)"],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 1
    assert coverage["unresolved"] == 1
    assert coverage["unresolved_citations"] == ["[1]"]


# --------------------------------------------------------------------------------------
# The off-switch
# --------------------------------------------------------------------------------------


def test_resolver_off_when_numbering_is_not_contiguous_from_one():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        ["Tutoring improves outcomes [1]."],
        "References",
        ["2. Smith, J. (2020). A study."],  # starts at 2, not 1
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 1
    assert coverage["unresolved"] == 1
    assert coverage["unresolved_citations"] == ["[1]"]
    assert coverage["by_source"]["numbered"] == 1


def test_resolver_off_when_body_cites_a_number_outside_the_reference_list():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        ["Tutoring improves outcomes [1]. Something else [5]."],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    # Neither [1] nor [5] resolves: the whole document's numbering switches off, not
    # just the offending citation.
    assert claims == []
    assert coverage["found"] == 2
    assert coverage["unresolved"] == 2


def test_resolver_off_when_there_is_no_reference_list_at_all():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = {"type": "doc", "content": [_paragraph("Tutoring improves outcomes [1].")]}
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 1
    assert coverage["unresolved"] == 1
    assert coverage["by_source"]["numbered"] == 1


def test_reference_list_heading_matches_the_chinese_equivalent():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        ["Tutoring improves outcomes [1]."],
        "参考文献",  # 参考文献
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Tutoring improves outcomes [1].",
            "smith_2020",
            "[1]",
            "Tutoring improves outcomes [1].",
        )
    ]
    assert coverage["linked"] == 1


def test_reference_section_stops_at_the_next_heading():
    """Entries only run to the next heading; content after that heading is not part of
    the reference list (and, since it is a body heading, not scanned for entries)."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = {
        "type": "doc",
        "content": [
            _paragraph("Tutoring improves outcomes [1]."),
            _heading("References"),
            _paragraph("1. Smith, J. (2020). A study."),
            _heading("Appendix"),
            _paragraph("2. Not really a reference, just appendix text."),
        ],
    }
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Tutoring improves outcomes [1].",
            "smith_2020",
            "[1]",
            "Tutoring improves outcomes [1].",
        )
    ]
    assert coverage["linked"] == 1


# --------------------------------------------------------------------------------------
# An earlier allowlist named eight metric units and every two-letter unit outside it
# was still read as a citation superscript, so a draft carrying one of them plus a
# one-entry
# reference list still collected a bogus cited number that switched the whole numbered
# resolver off. The base test is now any one- or two-letter token, a digit run, or "chi".
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "The dose was 5 mg² per litre.",
        "The signal was 5 Hz² wide.",
        "The genome span was 5 kb² long.",
        "Noise rose by 5 dB² overall.",
        "The particle was 5 um² across.",
        "Size was 1 µm² wide.",
        "The measured area was several km² in total.",
        "The chi² statistic was significant at p < 0.05.",
    ],
)
def test_two_letter_unit_abbreviations_are_all_kept_as_exponents(sentence):
    """"mg", "Hz", "kb", "dB" and "um" were previously outside the allowlist and were each read as
    a citation superscript; the units the list did name, and "chi", must not regress."""
    lookup: dict = {}
    doc = {"type": "doc", "content": [_paragraph(sentence)]}
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == []
    assert coverage["found"] == 0


def test_a_two_letter_unit_does_not_disable_a_working_numbered_resolver():
    """The harm the shape rule closes: "mg²" used to be cited number 2, which exceeds a
    one-entry reference list's highest entry, which switched the resolver off for the
    whole document and reported the genuine "[1]" unresolved."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    doc = _doc(
        [
            "Prior work reported the effect [1].",
            "The dose was 5 mg² per litre.",
        ],
        "References",
        ["1. Smith, J. (2020). A study."],
    )
    claims, coverage = extract_claims_from_document(doc, lookup)
    assert claims == [
        (
            "Prior work reported the effect [1].",
            "smith_2020",
            "[1]",
            "Prior work reported the effect [1].",
        )
    ]
    assert coverage["found"] == 1
    assert coverage["linked"] == 1
    assert coverage["unresolved"] == 0


def test_a_superscript_after_a_two_letter_word_is_the_accepted_cost_of_the_shape_rule():
    """The accepted tradeoff, stated in terms: accepting any one- or
    two-letter token as an exponent base loses a genuine Vancouver superscript written
    directly after a two-letter word ("US¹"). A three-letter word ("men³") is still a
    citation."""
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    entries = ["1. Smith, J. (2020). A study."]
    two_letter = _doc(["Rates in the US¹ were higher."], "References", entries)
    claims, coverage = extract_claims_from_document(two_letter, lookup)
    assert claims == []
    assert coverage["found"] == 0

    three_letter_entries = [f"{i}. Other{i}, X. (2019). Unrelated." for i in range(1, 3)]
    three_letter_entries.append("3. Smith, J. (2020). A study.")
    three_letter = _doc(["This was shown in men³."], "References", three_letter_entries)
    claims, coverage = extract_claims_from_document(three_letter, lookup)
    assert claims == [("This was shown in men³.", "smith_2020", "³", "This was shown in men³.")]
    assert coverage["unresolved"] == 0


def test_every_exponent_base_allowlist_entry_is_reachable():
    """Round 4's minor finding 1, second half: the list used to carry a micro-sign entry
    the look-back could never produce, because `_PRECEDING_ALNUM_TOKEN_RE` is
    `[A-Za-z0-9]+$` and matches neither the micro sign nor a Greek mu. Every entry left
    in the list must be a token that pattern can actually hand to the base test."""
    from app.services.fulltext import _EXPONENT_BASE_ALLOWLIST, _PRECEDING_ALNUM_TOKEN_RE

    for entry in _EXPONENT_BASE_ALLOWLIST:
        assert _PRECEDING_ALNUM_TOKEN_RE.fullmatch(entry), entry


# --------------------------------------------------------------------------------------
# A link's own sentence, once the claim splitter breaks it at an abbreviation such as
# "U.S.", must be compared only against the fragment holding the link's own citation,
# not every fragment of the paragraph -- otherwise it would suppress a numbered
# citation in a fragment the link never touches.
# --------------------------------------------------------------------------------------


def test_a_link_spanning_a_fragment_split_does_not_suppress_a_numbered_citation_next_door():
    lookup = {"smith_2020": paper([{"name": "Jane Smith"}], 2020)}
    sentence = "Gains were reported (Smith, 2020) in the U.S. Others agreed [1]."
    entries = ["1. Smith, J. (2020). A study of tutoring."]

    unmapped_doc = _doc([sentence], "References", entries)
    _claims, unmapped_coverage = extract_claims_from_document(unmapped_doc, lookup)
    assert unmapped_coverage["found"] == 2
    assert unmapped_coverage["by_source"]["numbered"] == 1

    mapped_doc = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "attrs": {
                    "citationLinks": [
                        {
                            "sentence": sentence,
                            "keys": ["smith_2020"],
                            "citation_text": "(Smith, 2020)",
                        }
                    ]
                },
                "content": [{"type": "text", "text": sentence}],
            },
            _heading("References"),
            _paragraph(entries[0]),
        ],
    }
    claims, mapped_coverage = extract_claims_from_document(mapped_doc, lookup)
    assert mapped_coverage["found"] == 2
    assert mapped_coverage["by_source"]["numbered"] == 1
    assert set((k, c) for _s, k, c, _cs in claims) == {
        ("smith_2020", "(Smith, 2020)"),
        ("smith_2020", "[1]"),
    }
