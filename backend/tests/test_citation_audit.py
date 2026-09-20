"""Code-level audit of APA in-text citations against the project library.

``[NEEDS CITATION]`` is a prompt instruction to the writing model; the audit counts how
often the model actually emitted it and checks every ``(Author, Year)`` /
``Author (Year)`` citation against the first authors of the library.
"""

from pathlib import Path
from types import SimpleNamespace

from app.services.citation_audit import (
    CitationAudit,
    _find_citations,
    audit_citations,
    citation_spans,
)
from tests.conftest import skip_unless_run_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

LIBRARY = [
    {"authors": [{"name": "Jane Smith"}, {"name": "Ken Jones"}], "year": 2020},
    {"authors": ["Amy Brown", "Bo Li"], "year": 2018},
    SimpleNamespace(authors=[{"name": "Teun van Dijk"}], year=2015),
    SimpleNamespace(authors=["García, María"], year=2019),
]


def test_paragraph_with_two_library_one_foreign_and_one_flag():
    text = (
        "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
        "Some claim otherwise (Nguyen, 2021). Costs are rising [NEEDS CITATION]."
    )
    audit = audit_citations(text, LIBRARY)
    assert isinstance(audit, CitationAudit)
    assert audit.matched == ["Smith, 2020", "Brown, 2018"]
    assert audit.unmatched == ["Nguyen, 2021"]
    assert audit.needs_citation_flags == 1
    assert audit.total == 3


def test_all_apa_forms_are_recognised():
    text = (
        "(Smith, 2020) (Smith & Jones, 2020) (Smith and Jones, 2020) (Smith et al., 2020) "
        "Smith (2020) Smith et al. (2020) Smith and Jones (2020) Smith & Jones (2020) "
        "(Smith, 2020a) (Smith, 2020, p. 12) (see Smith, 2020; Brown, 2018) "
        "(e.g., Brown, 2018, pp. 3-4)"
    )
    audit = audit_citations(text, LIBRARY)
    assert audit.matched == ["Smith, 2020", "Brown, 2018"]
    assert audit.unmatched == []
    assert audit.total == 2


def test_secondary_citation_checks_only_the_cited_source():
    audit = audit_citations("(Foo, 1999, as cited in Smith, 2020)", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.unmatched == []


def test_year_mismatch_is_unmatched():
    audit = audit_citations("Findings hold (Smith, 2019).", LIBRARY)
    assert audit.matched == []
    assert audit.unmatched == ["Smith, 2019"]


def test_particles_and_diacritics_match_first_author_surnames():
    audit = audit_citations("Discourse matters (van Dijk, 2015; García, 2019).", LIBRARY)
    assert audit.matched == ["van Dijk, 2015", "García, 2019"]
    assert audit.unmatched == []


def test_capitalised_prose_before_a_narrative_citation_is_not_the_surname():
    audit = audit_citations("In contrast, Smith (2020) argued the opposite.", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.unmatched == []


# --------------------------------------------------------------------------------------
# A capitalised lead-in word directly
# followed by a comma and a two- or three-author narrative citation ("Interestingly, Liu
# and Wu (2019)") can make `_NARRATIVE_CITE`'s own `surname` group resolve to the lead-in
# word, not the real first author, the same ambiguity extraction handles
# (`app.services.fulltext._resolve_narrative_citation_key`). Without that same handling,
# the audit would report "Interestingly, 2019" as
# unmatched even when the project library holds `liu_2019` and the extractor verifies it.
# --------------------------------------------------------------------------------------


def test_capitalised_lead_in_word_before_a_two_author_narrative_citation_is_not_the_surname():
    papers = [{"authors": ["Amy Liu"], "year": 2019}]
    audit = audit_citations(
        "Interestingly, Liu and Wu (2019) also found higher engagement.", papers
    )
    assert audit.matched == ["Liu, 2019"]
    assert audit.unmatched == []
    assert audit.total == 1


def test_lead_in_word_disambiguation_falls_back_when_no_candidate_is_in_the_library():
    """No candidate token in the whole match is in the library at all: the original
    (wrong, but unavoidable without a library) surname is kept -- this is a decision,
    not a defect, the same one extraction's own fallback makes with no library."""
    audit = audit_citations("Interestingly, Liu and Wu (2019) also found this.", [])
    assert audit.matched == []
    assert audit.unmatched == ["Interestingly, 2019"]


def test_an_oxford_comma_three_author_list_still_credits_the_real_first_author():
    """"Zhang, Li and Wang (2021)" is structurally identical to a lead-in-word case --
    a capitalised word immediately followed by a comma, then more names -- but here the
    first word genuinely is the first author. Tried first against the library, it wins
    immediately, so a real Oxford-comma author list is never misattributed to the second
    author the way a lead-in word is."""
    papers = [{"authors": ["Wei Zhang"], "year": 2021}]
    audit = audit_citations("Zhang, Li and Wang (2021) replicated the study.", papers)
    assert audit.matched == ["Zhang, 2021"]
    assert audit.unmatched == []


def test_extractor_and_audit_agree_on_the_demo_drafts_lead_in_word_citation():
    """End-to-end agreement check on a real demo draft
    (`demo/output/20260908-213423/writing_result.json`): the audit and the extractor
    must key "Interestingly, Liu and Wu (2019)" to the same surname, whether or not a
    library is available, since both share the identical "try every candidate token in
    order" strategy for this ambiguity."""
    import json

    from app.services.fulltext import _extract_claims

    run_dir = skip_unless_run_dir(REPO_ROOT / "demo" / "output" / "20260908-213423")
    demo_path = run_dir / "writing_result.json"
    content = json.loads(demo_path.read_text(encoding="utf-8"))["content"]
    assert "Interestingly, Liu and Wu (2019)" in content

    paper_lookup = {"liu_2019": object()}
    papers = [{"authors": ["Amy Liu"], "year": 2019}]

    audit = audit_citations(content, papers)
    assert "Liu, 2019" in audit.matched
    assert not any(label.startswith("Interestingly") for label in audit.unmatched)

    claims = _extract_claims(content, paper_lookup)
    lead_in_claims = [c for c in claims if "Interestingly, Liu and Wu (2019)" in c[2]]
    assert lead_in_claims, "the extractor must still emit a claim for this citation"
    assert all(key == "liu_2019" for _sentence, key, _text in lead_in_claims)


def test_author_name_formats():
    papers = [
        {"authors": ["Smith, J."], "year": 2001},
        {"authors": ["J. Smith"], "year": 2002},
        {"authors": ["Smith J"], "year": 2003},
        {"authors": [{"name": "Jane Smith"}], "year": 2004},
    ]
    audit = audit_citations("(Smith, 2001; Smith, 2002; Smith, 2003; Smith, 2004)", papers)
    assert audit.matched == ["Smith, 2001", "Smith, 2002", "Smith, 2003", "Smith, 2004"]
    assert audit.unmatched == []


def test_repeated_citations_are_counted_once():
    audit = audit_citations("(Smith, 2020) and again (Smith, 2020) and Smith (2020).", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.total == 1


def test_empty_text_and_empty_library():
    assert audit_citations("", LIBRARY) == CitationAudit(
        matched=[], unmatched=[], needs_citation_flags=0, total=0
    )
    audit = audit_citations("Claim (Smith, 2020). [NEEDS CITATION] [NEEDS CITATION]", [])
    assert audit.matched == []
    assert audit.unmatched == ["Smith, 2020"]
    assert audit.needs_citation_flags == 2


def test_papers_without_authors_or_year_are_tolerated():
    papers = [
        {"authors": [], "year": 2020},
        {"authors": None, "year": 2020},
        {"authors": [{"name": ""}], "year": 2020},
        {"authors": ["Jane Smith"], "year": None},
    ]
    audit = audit_citations("(Smith, 2020) (Smith, n.d.)", papers)
    # A library entry without a year matches on surname alone.
    assert audit.matched == ["Smith, 2020", "Smith, n.d."]
    assert audit.unmatched == []


def test_non_citation_parentheses_are_ignored():
    audit = audit_citations("The sample (N = 200) was drawn in 2020 (see Table 1).", LIBRARY)
    assert audit.total == 0
    assert audit.needs_citation_flags == 0


def test_prose_word_ending_in_a_particle_is_not_glued_onto_the_surname():
    """"laTER Smith" must not become the surname "ter Smith"."""
    audit = audit_citations("later Smith, Jones, and Brown (2021) disagreed.", [
        {"authors": ["Jane Smith"], "year": 2021}
    ])
    assert audit.matched == ["Smith, 2021"]
    assert audit.unmatched == []

    audit = audit_citations("under Smith (2020); these include Smith (2020).", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.unmatched == []


def test_possessive_narrative_citations():
    assert audit_citations("Smith's (2020) study found X.", LIBRARY).matched == ["Smith, 2020"]
    assert audit_citations("Smith’s (2020) study found X.", LIBRARY).matched == ["Smith, 2020"]

    audit = audit_citations("Smith et al.'s (2020) meta-analysis", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.total == 1
    audit = audit_citations("Smith and Jones’s (2020) meta-analysis", LIBRARY)
    assert audit.matched == ["Smith, 2020"]
    assert audit.total == 1


def test_library_years_are_normalised_to_four_digits():
    papers = [
        {"authors": ["Jane Smith"], "year": 2020.0},
        {"authors": ["Amy Brown"], "year": "2018-01-01"},
        {"authors": ["Bo Li"], "year": "unknown"},
    ]
    audit = audit_citations("(Smith, 2020; Brown, 2018; Li, 2019; Smith, 2021)", papers)
    # An unparseable library year counts as unknown: surname-only match (Li).
    assert audit.matched == ["Smith, 2020", "Brown, 2018", "Li, 2019"]
    assert audit.unmatched == ["Smith, 2021"]


def test_odd_author_shapes_never_raise():
    papers = [
        {"authors": {"name": "Jane Smith"}, "year": 2020},
        {"authors": "Jane Smith", "year": 2020},
        {"authors": 42, "year": 2020},
    ]
    audit = audit_citations("(Smith, 2020)", papers)
    assert audit.total == 1


def test_in_press_is_accepted_as_an_apa_year_placeholder():
    """``in press`` behaves like ``n.d.``: no year key, matched on surname alone."""
    from app.services.citation_audit import _year_key

    assert _year_key("n.d.") is None
    assert _year_key("in press") is None
    assert _year_key("2020a") == "2020"

    papers = [{"authors": ["Jane Smith"], "year": None}]
    audit = audit_citations(
        "Tutoring helps (Smith, in press). Smith (in press) also notes costs.", papers
    )
    assert audit.matched == ["Smith, in press"]
    assert audit.unmatched == []
    assert audit.total == 1

    # A placeholder year never matches a dated-only library entry.
    audit = audit_citations("(Jones, in press)", [{"authors": ["Ken Jones"], "year": 2020}])
    assert audit.matched == []
    assert audit.unmatched == ["Jones, in press"]


# --------------------------------------------------------------------------------------
# `citation_spans` is a public wrapper over the private `_find_citations`. No behaviour
# change: `audit_citations` still calls `_find_citations` directly and is untouched by
# this addition.
# --------------------------------------------------------------------------------------


def test_citation_spans_is_find_citations_unchanged():
    """`citation_spans` is a pure pass-through: identical output to `_find_citations` on
    every citation shape the audit recognises, so wrapping it publicly changes nothing."""
    text = (
        "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
        "Some claim otherwise (Nguyen, 2021). In contrast, Smith (2020) argued the opposite."
    )
    assert citation_spans(text) == _find_citations(text)


def test_citation_spans_returns_positions_for_every_citation_audit_citations_counts():
    """`citation_spans` is authoritative about how many citations exist in a text:
    every (surname, year) pair `audit_citations` counts
    into `matched`/`unmatched` must appear, with a real position into the text, among the
    spans this function returns."""
    text = (
        "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur. "
        "Some claim otherwise (Nguyen, 2021). Costs are rising [NEEDS CITATION]."
    )
    spans = citation_spans(text)
    audit = audit_citations(text, LIBRARY)

    distinct_pairs = {(surname.split()[-1].lower(), year) for _pos, surname, year, _label in spans}
    assert len(distinct_pairs) == audit.total
    assert audit.matched == ["Smith, 2020", "Brown, 2018"]
    assert audit.unmatched == ["Nguyen, 2021"]

    for position, surname, _year, _label in spans:
        assert text[position : position + len(surname)] == surname


def test_citation_spans_on_empty_text_is_empty():
    assert citation_spans("") == []
