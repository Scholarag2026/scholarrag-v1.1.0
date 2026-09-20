"""Tests for the citation author string sourced from the acquired full text's own
byline.

The Razmi-and-Ghane fixture chunk is the design's own motivating case: OpenAlex records
two authors in an order the article itself does not use, while the article's own
running head names four authors, Ghane first.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.fulltext import (  # noqa: E402
    _accept_byline,
    _build_paper_lookup,
    _parse_byline_surnames,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/evidence/seeds12/53ae2131-383c-4642-ae68-6e14983b73f9.json"
)
#: The same
#: paper's chunk 0 exactly as the LIVE PDF acquisition extracted it, unredacted -- the
#: running-header contact block that sits ahead of the title and the real byline on the
#: live text, which the fixture above replaces with a "[TEXT OMITTED FROM FIXTURE]"
#: placeholder and so never exercised.
LIVE_HEADER_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/byline/53ae2131-383c-4642-ae68-6e14983b73f9-live-header.json"
)

OPENALEX_RAZMI_GHANE = ["Mohammad Hasan Razmi", "Mohammad Hossein Ghane"]


def _fixture_chunk_text() -> str:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["chunks"][0]["text"]


def _live_header_fixture_chunk_text() -> str:
    data = json.loads(LIVE_HEADER_FIXTURE.read_text(encoding="utf-8"))
    return data["chunks"][0]["text"]


def test_parses_all_four_byline_surnames_ghane_first():
    surnames = _parse_byline_surnames(_fixture_chunk_text())
    assert surnames == ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"]


def test_affiliation_footnote_does_not_leak_into_the_last_surname():
    """The last author's own segment runs directly into the affiliation list with no
    comma between them ("Nematollahi2 \\n1 Yazd University | Iran"); the affiliation
    marker cut must stop the parse before "Iran" is read as a surname."""
    surnames = _parse_byline_surnames(_fixture_chunk_text())
    assert "Iran" not in surnames
    assert "University" not in surnames


def test_title_and_copyright_prose_are_not_read_as_a_name():
    surnames = _parse_byline_surnames(_fixture_chunk_text())
    assert "The" not in surnames
    assert "Copyright" not in surnames


def test_accept_byline_true_when_every_openalex_surname_is_found():
    parsed = ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"]
    openalex = ["Mohammad Hasan Razmi", "Mohammad Hossein Ghane"]
    assert _accept_byline(parsed, openalex) is True


def test_accept_byline_false_when_an_openalex_surname_is_missing():
    parsed = ["Ghane", "Dehghanpoor", "Nematollahi"]  # Razmi missing
    openalex = ["Mohammad Hasan Razmi", "Mohammad Hossein Ghane"]
    assert _accept_byline(parsed, openalex) is False


def test_accept_byline_false_on_empty_parse_or_empty_openalex_list():
    assert _accept_byline([], ["Smith"]) is False
    assert _accept_byline(["Smith"], []) is False


def test_accept_byline_is_diacritic_and_case_insensitive():
    assert _accept_byline(["Hebert"], ["L. Hébert"]) is True


def test_accept_byline_rejects_the_noisy_live_pdf_contact_block_parse():
    """The exact parse the live PDF's own running-header
    contact block produces -- "Razmi" once from the contact block ahead of the title,
    the real byline's own four surnames after it, and "Razmi" a second time as the real
    byline's own second author. Membership alone accepted this (every OpenAlex surname
    is present somewhere), which is what let a running header decide the first author.
    A parse where an OpenAlex surname repeats, and the two OpenAlex surnames are not a
    contiguous run with nothing else between them, must be rejected."""
    noisy_parse = [
        "Research",
        "Razmi",
        "Languages",
        "Kerman",
        "Ghane",
        "Razmi",
        "Dehghanpoor",
        "Nematollahi",
    ]
    assert _accept_byline(noisy_parse, OPENALEX_RAZMI_GHANE) is False


def test_accept_byline_rejects_a_repeated_openalex_surname_even_when_contiguous():
    assert _accept_byline(["Razmi", "Razmi", "Ghane"], OPENALEX_RAZMI_GHANE) is False


def test_accept_byline_rejects_a_non_name_token_interleaved_between_the_two_surnames():
    assert _accept_byline(["Razmi", "Languages", "Ghane"], OPENALEX_RAZMI_GHANE) is False


def test_accept_byline_rejects_a_clean_contiguous_run_that_does_not_start_the_parse():
    """The repeated-surname-plus-contiguity rule alone does not
    require the OpenAlex run to START the parse, so a noise token OpenAlex does not
    name at all -- one that never collides with an OpenAlex surname, unlike the live
    contact block's own repeated "Razmi" -- is still accepted ahead of an otherwise
    clean, contiguous run and still becomes the first surname handed to the prompt."""
    assert (
        _accept_byline(
            [
                "Research",
                "Karimi",
                "Languages",
                "Kerman",
                "Ghane",
                "Razmi",
                "Dehghanpoor",
                "Nematollahi",
            ],
            OPENALEX_RAZMI_GHANE,
        )
        is False
    )
    assert (
        _accept_byline(
            ["Research", "Languages", "Kerman", "Ghane", "Razmi", "Dehghanpoor", "Nematollahi"],
            OPENALEX_RAZMI_GHANE,
        )
        is False
    )
    assert _accept_byline(["Journal", "Ghane", "Razmi"], OPENALEX_RAZMI_GHANE) is False


def test_accept_byline_still_accepts_the_true_byline_ghane_first():
    """The true byline, clean of any running-header noise: must still accept, and in
    an order that renders "Ghane et al. (2024)"."""
    true_byline = "Mohammad Hossein Ghane, Mohammad Hasan Razmi, Farzaneh Dehghanpoor and Rouhollah Nematollahi"
    surnames = _parse_byline_surnames(true_byline)
    assert surnames == ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"]
    assert _accept_byline(surnames, OPENALEX_RAZMI_GHANE) is True
    assert surnames[0] == "Ghane"


def test_the_live_pdf_header_fixtures_own_noisy_parse_is_rejected_by_acceptance():
    """The fixture whose first chunk is the live PDF's own, unredacted text: the parser
    reproduces the exact noisy surname list, and acceptance rejects it, so the caller
    falls back to OpenAlex's own
    string rather than rendering the contact block's "Razmi" as the first author."""
    surnames = _parse_byline_surnames(_live_header_fixture_chunk_text())
    assert surnames == [
        "Research",
        "Razmi",
        "Languages",
        "Kerman",
        "Ghane",
        "Razmi",
        "Dehghanpoor",
        "Nematollahi",
    ]
    assert _accept_byline(surnames, OPENALEX_RAZMI_GHANE) is False


def test_the_altered_live_chunk_with_a_non_openalex_contact_name_is_still_rejected():
    """The live chunk's own noisy parse is caught only
    because its contact block happens to name "Razmi" a second time, which the
    repeated-surname check rejects. Replacing that one contact name with a surname
    OpenAlex does not list at all -- "Razmi" -> "Karimi", contact block only, the real
    byline's own "Razmi" untouched -- removes the repetition: the OpenAlex run (Ghane,
    Razmi) is still one clean, contiguous match, but it no longer opens the parse.
    Acceptance must still reject it, and the caller must still fall back to OpenAlex's
    own author order rather than rendering "Research" as the first surname."""
    altered = _live_header_fixture_chunk_text().replace(
        "Contact: Mohammad Hasan Razmi", "Contact: Mohammad Hasan Karimi", 1
    )
    surnames = _parse_byline_surnames(altered)
    assert surnames == [
        "Research",
        "Karimi",
        "Languages",
        "Kerman",
        "Ghane",
        "Razmi",
        "Dehghanpoor",
        "Nematollahi",
    ]
    assert _accept_byline(surnames, OPENALEX_RAZMI_GHANE) is False


def test_build_paper_lookup_registers_the_byline_alias_beside_the_canonical_key():
    paper = SimpleNamespace(
        authors=[{"name": "Mohammad Hasan Razmi"}, {"name": "Mohammad Hossein Ghane"}],
        year=2024,
        metadata_={
            "fulltext_byline": {
                "surnames": ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"],
                "source": "fulltext",
            }
        },
    )
    lookup = _build_paper_lookup([paper])
    assert lookup["razmi_2024"] is paper
    assert lookup["ghane_2024"] is paper


def test_build_paper_lookup_registers_no_alias_when_byline_was_not_accepted():
    paper = SimpleNamespace(
        authors=[{"name": "Mohammad Hasan Razmi"}],
        year=2024,
        metadata_={"fulltext_byline": {"source": "openalex"}},
    )
    lookup = _build_paper_lookup([paper])
    assert lookup == {"razmi_2024": paper}


def test_build_paper_lookup_registers_no_alias_when_paper_has_no_metadata_at_all():
    paper = SimpleNamespace(
        authors=[{"name": "Mohammad Hasan Razmi"}], year=2024, metadata_=None
    )
    lookup = _build_paper_lookup([paper])
    assert lookup == {"razmi_2024": paper}
