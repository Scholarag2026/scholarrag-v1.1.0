"""harvest_real_claims.py: naturalistic claim-to-source items from published review articles.

No network, no LLM, no backend import: every function tested here is pure (regex matching,
reference-list parsing, sampling, item-shape assembly) or works from local fixture caches
(``--skip-fetch``-style rebuild) or a fake in-process client (discovery pagination). The
acquisition step itself (OpenAlex/Crossref/Unpaywall/PDF) is never exercised here, mirroring
``test_claims_hss.py``'s treatment of ``build_hss_set.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import harvest_real_claims as hc

import run_hss as rh
from common import write_json

# --------------------------------------------------------------------------------------
# Citation detection (design item 3): the recognised APA forms.
# --------------------------------------------------------------------------------------


def test_find_single_citation_recognises_the_parenthetical_forms():
    s = ("Recent classroom-based work has repeatedly linked structured feedback to gains in "
         "learner accuracy over an academic term (Smith, 2019).")
    m = hc.find_single_citation(s)
    assert m is not None and m.surname == "Smith" and m.year == "2019"
    assert m.connector == "single" and m.text == "(Smith, 2019)"

    s2 = ("Recent classroom-based work has repeatedly linked structured feedback to gains in "
          "learner accuracy over an academic term (Smith and Jones, 2019).")
    m2 = hc.find_single_citation(s2)
    assert m2 is not None and m2.surname == "Smith" and m2.surname2 == "Jones"
    assert m2.connector == "and" and m2.year == "2019"

    s3 = ("Recent classroom-based work has repeatedly linked structured feedback to gains in "
          "learner accuracy over an academic term (Smith et al., 2019).")
    m3 = hc.find_single_citation(s3)
    assert m3 is not None and m3.surname == "Smith" and m3.connector == "et_al"


def test_find_single_citation_recognises_the_narrative_forms():
    s4 = ("Smith (2019) linked structured feedback to gains in learner accuracy across an "
          "entire academic term for every cohort observed in the study.")
    m4 = hc.find_single_citation(s4)
    assert m4 is not None and m4.connector == "single" and m4.text == "Smith (2019)"

    s5 = ("Aryadoust et al. (2019) linked structured feedback to gains in learner accuracy "
          "across an entire academic term for every cohort observed in the study.")
    m5 = hc.find_single_citation(s5)
    assert m5 is not None and m5.connector == "et_al" and m5.surname == "Aryadoust"
    assert m5.text == "Aryadoust et al. (2019)"


def test_find_single_citation_recognises_narrative_ampersand_form():
    """Design item 3: a narrative two-author citation joined by '&' (not only 'and') must
    still capture the full author group with the FIRST surname, the same anchoring as the
    'and' form."""
    s = ("Crosthwaite & Baisa (2023) compare corpora and generative AI within data-driven "
         "learning approaches used across many different classroom contexts and cohorts.")
    m = hc.find_single_citation(s)
    assert m is not None
    assert m.surname == "Crosthwaite" and m.surname2 == "Baisa"
    assert m.connector == "and" and m.text == "Crosthwaite & Baisa (2023)"


def test_find_single_citation_rejects_ampersand_parenthetical_and_multi_citation_sentences():
    # a PARENTHETICAL "&" is not one of the recognised forms (design choice, ampersand is
    # only recognised in the narrative form above, which is unambiguous about which surname
    # is first because it appears before the connector, not inside a parenthesis).
    s = ("Recent classroom-based work has repeatedly linked structured feedback to gains in "
         "learner accuracy over an academic term (Smith & Jones, 2019).")
    assert hc.find_single_citation(s) is None
    # two citations, even though one is a recognised form
    s2 = ("Smith (2019) linked feedback to accuracy gains, and separately Jones (2020) agreed "
          "with this conclusion after reviewing a comparable cohort of learners.")
    assert hc.find_single_citation(s2) is None
    # a recognised form plus an unrecognised ampersand citation elsewhere in the sentence
    s3 = ("Smith (2019) linked feedback to accuracy gains, in line with related findings "
          "(Jones & Brown, 2020) reported for a comparable cohort of learners overall.")
    assert hc.find_single_citation(s3) is None


def test_find_single_citation_rejects_word_count_out_of_range():
    short = "Smith (2019) agreed."
    assert hc.find_single_citation(short) is None
    long_words = " ".join(["word"] * 58)
    long_sentence = f"Smith (2019) {long_words} across every cohort in the sample studied here."
    assert hc.find_single_citation(long_sentence) is None


def test_find_single_citation_rejects_leading_see_cf_for_example():
    base = ("classroom-based work has repeatedly linked structured feedback to gains in "
            "learner accuracy over an academic term (Smith, 2019) for every cohort studied.")
    assert hc.find_single_citation(f"See {base}") is None
    assert hc.find_single_citation(f"Cf. {base}") is None
    assert hc.find_single_citation(f"For example {base}") is None
    assert hc.find_single_citation(base[0].upper() + base[1:]) is not None


def test_find_single_citation_rejects_list_items_and_headings():
    body = ("Smith (2019) linked structured feedback to gains in learner accuracy across an "
            "entire academic term for every cohort in the sample studied here.")
    assert hc.find_single_citation(f"1. {body}") is None
    assert hc.find_single_citation(f"(a) {body}") is None
    assert hc.find_single_citation(f"- {body}") is None
    assert hc.find_single_citation(f"Introduction {body}") is None
    assert hc.find_single_citation(f"Discussion {body}") is None
    assert hc.find_single_citation(body) is not None


def test_find_single_citation_recognises_narrative_two_and_three_author_forms():
    s = ("Smith and Jones (2019) linked structured feedback to gains in learner accuracy "
         "across an entire academic term for every cohort observed in the study.")
    m = hc.find_single_citation(s)
    assert m is not None and m.surname == "Smith" and m.surname2 == "Jones"
    assert m.connector == "and" and m.text == "Smith and Jones (2019)"

    s2 = ("Smith, Jones, and Brown (2019) linked structured feedback to gains in learner "
          "accuracy across an entire academic term for every cohort in the sample studied.")
    m2 = hc.find_single_citation(s2)
    assert m2 is not None and m2.surname == "Smith" and m2.surname2 == "Jones"
    assert m2.connector == "and" and m2.year == "2019"


def test_find_single_citation_recognises_particle_surnames():
    s = ("van Lier (2004) argued that language learning is best understood through an "
         "ecological, sociocultural lens across many different instructional contexts.")
    m = hc.find_single_citation(s)
    assert m is not None and m.surname == "van Lier" and m.connector == "single"


def test_find_single_citation_narrative_and_form_does_not_capture_the_last_author():
    """The un-anchored narrative branch used to match the
    LAST surname of a two-author narrative citation ('Crosthwaite and Baisa (2023)' matched
    'Baisa' alone) -- and, when the second author also has an unrelated same-year entry of
    their own in the reference list, silently resolved the claim to the wrong paper."""
    clean_cases = [
        ("Al Nafjan and Mohammed (2024) conducted a systematic and bibliometric analysis of "
         "the top fifty linguistics publications on AI, ChatGPT and large language models.",
         "Al Nafjan"),
        ("Curry and McEnery (2025) propose corpus methods as a bridge between AI-enhanced "
         "digital pedagogy and classroom practice.", "Curry"),
        ("It also reflects Curry and McEnery (2025) view of corpus methods as a bridge "
         "between digital pedagogy and classroom practice.", "Curry"),
        ("Lee and Bozeman (2005) further argue that collaboration is a strong predictor of "
         "research performance when assessed using normal counts of publications.", "Lee"),
    ]
    for sentence, expected_surname in clean_cases:
        m = hc.find_single_citation(sentence)
        assert m is not None, sentence
        assert m.surname == expected_surname, (sentence, m.surname)


def test_find_single_citation_rejects_pdf_hyphen_break_artefacts():
    """A PDF line-break-hyphenated sentence must not be
    shipped verbatim into the claim field. Reused from ``build_hss_set.HYPHEN_BREAK_RE``, as
    defence in depth: ``clean_extracted_text`` is expected to repair this shape before a
    sentence ever reaches ``find_single_citation`` (see the pipeline-level tests below), so
    this only fires when the repair itself is skipped, as it is here."""
    s = ("Crosthwaite and Baisa (2023) com- pare corpora and generative AI within "
         "data-driven learning, treated here as a pedagogi- cal construct where learners "
         "notice and generalise patterns from authentic language data.")
    assert hc.find_single_citation(s) is None
    repaired = s.replace("com- pare", "compare").replace("pedagogi- cal", "pedagogical")
    assert hc.find_single_citation(repaired) is not None


def test_find_single_citation_rejects_orphan_year_fragment_and_non_terminal_ending():
    """Belt-and-braces guard: even if a splitting
    defect elsewhere produced a fragment starting with an orphaned '(YYYY)' or not ending in
    terminal punctuation, this must not be accepted as a claim on its own."""
    orphan = ("(2022) demonstrated that cognitive load measures correlate strongly with "
              "reading comprehension outcomes across many different proficiency levels.")
    assert hc.find_single_citation(orphan) is None
    truncated = ("Philp and Duchesne (2016) introduced social engagement to enrich the "
                 "tripartite model of L2 engagement, and Wang et al.")
    assert hc.find_single_citation(truncated) is None


def test_find_single_citation_rejects_review_own_procedure_sentences():
    """A sentence about the CITING review's own procedure or corpus
    (an intercoder-reliability statement, a first-person-plural methods statement) is not a
    checkable claim about the cited work, even though it carries exactly one citation."""
    s1 = ("Intercoder reliability checks were conducted throughout the coding process, and "
          "disagreements were resolved through discussion following Cofie et al. (2021).")
    assert hc.find_single_citation(s1) is None
    s2 = ("In our corpus, we conducted a systematic search across five databases following "
          "the protocol described by Page et al. (2021) for every included study.")
    assert hc.find_single_citation(s2) is None
    s3 = ("The resulting kappa of 0.91 indicated excellent agreement between the two coders, "
          "consistent with the benchmark reported by McHugh (2012) for interrater studies.")
    assert hc.find_single_citation(s3) is None


def test_find_single_citation_rejects_page_furniture_inside_the_sentence():
    """A running header or 'Page N of M' / bare
    'Downloaded from' fragment glued inside an otherwise plausible sentence must be rejected,
    not shipped as part of the claim."""
    s1 = ("During the revision Page 5 of 25 Teng Asian-Pacific Journal of Second Language "
          "Education (2026) 11:70 and reviewing phase, GenAI provides immediate feedback.")
    assert hc.find_single_citation(s1) is None
    s2 = ("Downloaded from a shared institutional repository, the dataset described by Smith "
          "(2019) contains transcripts from over two hundred classroom observation sessions.")
    assert hc.find_single_citation(s2) is None


# --------------------------------------------------------------------------------------
# Text hygiene (design item 4): soft-hyphen and hyphen-space line breaks are repaired, not
# merely detected, before a chunk's text is ever split into sentences or parsed for
# references.
# --------------------------------------------------------------------------------------


def test_clean_extracted_text_strips_zero_width_and_undoes_ascii_hyphen_breaks():
    """A real PDF line wrap is a hyphen immediately followed
    by a NEWLINE, not a space (a hyphen-space is either an ordinary hyphenated compound or an
    APA suspended hyphenation, see the dedicated tests below); the fixture below uses a
    hyphen-newline break for that reason."""
    dirty = "10.​1016/j.​joi.​2017 and a joi-\nnal name wrapped mid line"
    cleaned = hc.clean_extracted_text(dirty)
    assert "​" not in cleaned
    assert "10.1016/j.joi.2017" in cleaned
    assert "joi-\nnal" not in cleaned and "joinal" in cleaned


def test_clean_extracted_text_undoes_soft_hyphen_line_breaks():
    """Many publishers emit
    U+00AD (soft hyphen) instead of an ASCII hyphen at a PDF line break; it must be joined
    away, not merely stripped (stripping alone still leaves the separating space behind)."""
    dirty = "This is cor­ roborated by Wang (2022), who confirms the same trend."
    cleaned = hc.clean_extracted_text(dirty)
    assert "­" not in cleaned
    assert "corroborated" in cleaned
    dirty2 = "reaching lasting pro­ficiency across every cohort studied over time here."
    cleaned2 = hc.clean_extracted_text(dirty2)
    assert "proficiency" in cleaned2


def test_parse_reference_entry_strips_zero_width_characters_from_the_doi():
    """Springer/Nature-style PDFs render a reference's
    printed DOI with U+200B zero-width spaces inside it; unstripped, ``_DOI_RE`` never
    matches and the printed-DOI path is dead for that publisher."""
    text = (
        "References\n"
        "Barrot, J. S. (2021). Research on education and learning technologies during the "
        "pandemic: A retrospective. Journal of Information Science, 74(1), 59-109. "
        "https://​doi.​org/​10.​1016/j.​joi.​2017.​09.​007\n"
    )
    entries = hc.parse_reference_entries(text)
    assert len(entries) == 1
    assert entries[0]["doi"] == "10.1016/j.joi.2017.09.007"


# --------------------------------------------------------------------------------------
# Sentence splitting (design item 3): never breaks at "al.", "e.g.", "i.e.", "cf.", "vs." or
# a single-letter initial. The shared
# ``common.split_sentences`` breaks a narrative "et al. (YYYY)" citation into an orphaned
# author fragment and an orphaned year fragment, so this harvester owns its own splitter.
# --------------------------------------------------------------------------------------


def test_split_sentences_real_does_not_break_between_et_al_and_the_year():
    text = (
        "Building on this foundation, Aryadoust et al. (2022) demonstrated that cognitive "
        "load measures correlate strongly with reading comprehension outcomes across diverse "
        "instructional settings and proficiency levels. A separate strand of work has since "
        "extended this finding to listening comprehension tasks as well."
    )
    sentences = hc.split_sentences_real(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Building on this foundation, Aryadoust et al. (2022)")
    assert sentences[0].endswith("proficiency levels.")
    citation = hc.find_single_citation(sentences[0])
    assert citation is not None and citation.surname == "Aryadoust" and citation.year == "2022"


def test_split_sentences_real_reproduces_the_six_observed_truncation_cases_as_one_sentence():
    """The six sentences an earlier review captured as *already truncated* fragments (the
    old splitter's output). Reconstructed here with their citation's year restored, feeding
    the ORIGINAL (untruncated) text through the new splitter: each must survive as one
    sentence, not be cut before the year, and (since every one of these six actually carries
    TWO citations once whole) must then be correctly rejected by ``find_single_citation`` for
    having more than one citation -- not wrongly accepted as a one-citation fragment
    attributed to the wrong (first) author, which is what the old splitter did."""
    cases = [
        "As indicated by data on fixation rate, fixation duration, and neuroimaging, the "
        "research found that, consistent with both the cognitive load theory (Sweller, 2011) "
        "and the study by Aryadoust et al. (2022), cognitive load increases with complexity.",
        "Philp and Duchesne (2016) introduced social engagement to enrich the tripartite "
        "model of L2 engagement, and Wang et al. (2020) focused on willingness to speak.",
        "The prevailing consensus, as seen in the policies analysed by Yoo (2025) and "
        "Bombier et al. (2023), is that plurilingual policy adoption remains uneven overall.",
        "The ease of text generation has profound implications for academic integrity, "
        "giving rise to what Al Hosni (2025) and Goyibova et al. (2024) call a crisis point.",
        "Readers interested in a more comprehensive review are encouraged to see a research "
        "timeline by Lim and Kessler (2024), along with a review study by Zhang et al. (2021).",
        "Those readers interested in developing connections between theory and methods are "
        "encouraged to see Paltridge and Phakiti (2015) and Phakiti et al. (2018) as well.",
    ]
    for text in cases:
        sentences = hc.split_sentences_real(text)
        assert len(sentences) == 1, (text, sentences)
        assert sentences[0] == text
        assert hc.find_single_citation(sentences[0]) is None, text


def test_split_sentences_real_does_not_break_after_e_g_i_e_cf_vs():
    text = (
        "Several instructional variables (e.g. task complexity, feedback timing) were shown "
        "by Chen (2020) to moderate outcomes across a wide range of classroom settings. A "
        "different strand of work, cf. the meta-analysis by Diaz (2019), reached a similar "
        "conclusion using a substantially larger and more diverse sample of learners."
    )
    sentences = hc.split_sentences_real(text)
    assert len(sentences) == 2
    assert sentences[0].endswith("classroom settings.")
    assert sentences[1].startswith("A different strand")


def test_split_sentences_real_does_not_break_after_a_single_letter_initial():
    text = (
        "Smith, J. argued that motivation drives engagement across a wide range of "
        "instructional contexts and learner populations studied over an extended period. "
        "This claim was later revisited by several other researchers in the field."
    )
    sentences = hc.split_sentences_real(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("Smith, J. argued")
    assert sentences[1].startswith("This claim was later revisited")


def test_split_sentences_real_still_splits_ordinary_sentences():
    text = "First sentence ends here. Second sentence starts and ends here too."
    assert hc.split_sentences_real(text) == [
        "First sentence ends here.", "Second sentence starts and ends here too.",
    ]
    assert hc.split_sentences_real("") == []
    assert hc.split_sentences_real("   ") == []


# --------------------------------------------------------------------------------------
# Reference-list parsing and matching (design item 3)
# --------------------------------------------------------------------------------------

REFERENCES_TEXT = (
    "References\n"
    "Smith, J. A., & Jones, B. (2019). Feedback and second language writing development. "
    "Journal of Second Language Writing, 45, 12-29. "
    "https://doi.org/10.1016/j.jslw.2019.03.002\n"
    "Brown, C. (2020). Motivation in language learning contexts. Applied Linguistics, "
    "41(2), 200-225.\n"
    "Lee, S., Kim, H., & Park, J. (2018). Exploring learner autonomy in EFL classrooms.\n"
    "Language Teaching Research, 22(4), 455-478. doi:10.1177/1362168817718693\n"
)


def test_find_references_chunk_returns_the_references_section_only():
    chunks = [
        {"section": "introduction", "text": "Some intro text here."},
        {"section": "references", "text": REFERENCES_TEXT},
    ]
    assert hc.find_references_chunk(chunks) == REFERENCES_TEXT
    assert hc.find_references_chunk([{"section": "intro", "text": "x"}]) is None
    assert hc.find_references_chunk([]) is None


def test_split_reference_lines_merges_wrapped_entries_and_drops_the_heading():
    entries = hc.split_reference_lines(REFERENCES_TEXT)
    assert len(entries) == 3
    assert entries[0].startswith("Smith, J. A.,")
    assert entries[2].startswith("Lee, S.,") and "Language Teaching Research" in entries[2]


def test_parse_reference_entry_extracts_surname_year_title_doi():
    entries = hc.parse_reference_entries(REFERENCES_TEXT)
    assert len(entries) == 3
    smith = entries[0]
    assert smith["surname"] == "Smith" and smith["year"] == "2019"
    assert smith["title"] == "Feedback and second language writing development"
    assert smith["doi"] == "10.1016/j.jslw.2019.03.002"
    brown = entries[1]
    assert brown["surname"] == "Brown" and brown["year"] == "2020" and brown["doi"] is None
    assert brown["title"] == "Motivation in language learning contexts"
    lee = entries[2]
    assert lee["surname"] == "Lee" and lee["year"] == "2018"
    assert lee["title"] == "Exploring learner autonomy in EFL classrooms"
    assert lee["doi"] == "10.1177/1362168817718693"


def test_parse_reference_entry_returns_none_without_surname_or_year():
    assert hc.parse_reference_entry("Not a reference line at all here") is None


def test_match_reference_entry_finds_exact_surname_and_year():
    entries = hc.parse_reference_entries(REFERENCES_TEXT)
    m = hc.match_reference_entry("Smith", None, "2019", entries)
    assert m is not None and m["doi"] == "10.1016/j.jslw.2019.03.002"
    assert hc.match_reference_entry("Nobody", None, "2019", entries) is None
    assert hc.match_reference_entry("Smith", None, "2020", entries) is None


def test_match_reference_entry_is_diacritic_and_case_insensitive():
    entries = [{"surname": "García", "year": "2021", "title": "T", "doi": None, "raw": "x"}]
    assert hc.match_reference_entry("Garcia", None, "2021", entries) is not None
    assert hc.match_reference_entry("GARCIA", None, "2021", entries) is not None


def test_match_reference_entry_declines_on_ambiguous_duplicates():
    entries = [
        {"surname": "Smith", "year": "2019", "title": "A", "doi": None, "raw": "x"},
        {"surname": "Smith", "year": "2019", "title": "B", "doi": None, "raw": "y"},
    ]
    assert hc.match_reference_entry("Smith", None, "2019", entries) is None


def test_match_reference_entry_requires_the_second_surname_in_the_entry_when_given():
    """``CitationMatch`` carries the second
    surname of a two-author citation; matching must use it, not just the first surname and
    year, so a wrapped or non-APA entry that happens to share first-surname-plus-year with an
    unrelated entry is not silently accepted."""
    entries = [
        {"surname": "Smith", "year": "2019", "title": "A", "doi": None,
         "raw": "Smith, J., & Jones, B. (2019). A. Journal."},
    ]
    assert hc.match_reference_entry("Smith", "Jones", "2019", entries) is not None
    assert hc.match_reference_entry("Smith", "Brown", "2019", entries) is None
    # no second surname on the citation itself (an "et al." form): unchanged behaviour.
    assert hc.match_reference_entry("Smith", None, "2019", entries) is not None


def test_split_reference_lines_recognises_particle_surnames_as_entry_starts():
    """A lowercase-particle surname ('van Lier') must be
    recognised as an entry start; otherwise it is merged into the previous entry as if it were a
    wrapped continuation, so an entry with no printed DOI would silently adopt its neighbour's."""
    text = (
        "References\n"
        "Brown, C. (2020). Motivation in language learning contexts. Applied Linguistics, "
        "41(2), 200-225.\n"
        "van Lier, L. (2004). The ecology and semiotics of language learning: A "
        "sociocultural perspective. Kluwer Academic. "
        "https://doi.org/10.1007/1-4020-7912-5\n"
    )
    entries = hc.parse_reference_entries(text)
    assert len(entries) == 2
    brown, van_lier = entries
    assert brown["surname"] == "Brown" and brown["doi"] is None
    assert van_lier["surname"] == "van Lier"
    assert van_lier["doi"] == "10.1007/1-4020-7912-5"


# --------------------------------------------------------------------------------------
# Crossref similarity / URL helpers
# --------------------------------------------------------------------------------------


def test_best_crossref_match_picks_the_most_similar_title_above_threshold():
    candidates = [
        {"doi": "10.1/aaa", "title": "Totally unrelated topic about gardening techniques"},
        {"doi": "10.1/bbb", "title": "Feedback and second language writing development"},
    ]
    match = hc.best_crossref_match("Feedback and second language writing development", candidates)
    assert match is not None and match["doi"] == "10.1/bbb" and match["similarity"] > 0.9


def test_best_crossref_match_returns_none_below_threshold():
    candidates = [{"doi": "10.1/aaa", "title": "Completely different subject matter entirely"}]
    assert hc.best_crossref_match("Feedback and writing development", candidates) is None
    assert hc.best_crossref_match("Anything", []) is None


def test_crossref_search_to_candidates_shape():
    payload = {"message": {"items": [
        {"DOI": "10.1/x", "title": ["A Title"], "type": "journal-article",
         "author": [{"given": "A.", "family": "Author"}]},
        {"title": ["no doi"]},
    ]}}
    cands = hc.crossref_search_to_candidates(payload)
    assert cands == [{
        "doi": "10.1/x", "title": "A Title", "type": "journal-article", "authors": ["A. Author"],
        "year": None,
    }]


def test_crossref_authors_joins_given_and_family_and_drops_blanks():
    message = {"author": [
        {"given": "A.", "family": "Author"}, {"family": "OnlyFamily"}, {"given": "OnlyGiven"},
        {},
    ]}
    assert hc._crossref_authors(message) == ["A. Author", "OnlyFamily", "OnlyGiven"]
    assert hc._crossref_authors({}) == []


def test_crossref_url_helpers_never_carry_an_email_in_the_public_form():
    url = hc.crossref_title_query_url("Some Title Here", "x@y.z")
    assert "api.crossref.org/works" in url and "x@y.z" in url  # mailto stays on the wire
    doi_url = hc.crossref_doi_url("10.1/x", "x@y.z")
    assert doi_url.endswith("/works/10.1/x?mailto=x@y.z")


def test_openalex_work_by_doi_url_shape():
    url = hc.openalex_work_by_doi_url("10.1/x", "x@y.z")
    assert url == "https://api.openalex.org/works/https://doi.org/10.1/x?mailto=x@y.z"


# --------------------------------------------------------------------------------------
# resolve_citation: printed-DOI path is verified against the entry's own title, not accepted
# unchecked; the cited work's authors come from OpenAlex
# authorships, not Crossref (design item 3).
# --------------------------------------------------------------------------------------


def test_resolve_citation_rejects_printed_doi_when_crossref_title_mismatches(monkeypatch):
    """Mirrors the finding's own reproduction: a captured surname's printed DOI actually
    belongs to an unrelated paper (a bled or mis-OCRed DOI); the mismatch must be caught here
    rather than shipped silently into the harvested set."""
    async def fake_lookup(doi, mailto):
        return {
            "doi": doi, "title": "An unrelated paper about lexicography tooling",
            "type": "journal-article",
        }

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    entry = {
        "doi": "10.1234/lex.2023.0099", "year": "2023",
        "title": "Corpus-driven grammar acquisition in adult second language learners",
    }
    assert asyncio.run(hc.resolve_citation(entry, mailto="x@y.z")) is None


def test_resolve_citation_accepts_printed_doi_when_titles_agree(monkeypatch):
    async def fake_lookup(doi, mailto):
        return {
            "doi": doi,
            "title": "Corpus-driven grammar acquisition in adult second language learners",
            "type": "journal-article",
        }

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    entry = {
        "doi": "10.1016/j.acorp.2023.100121", "year": "2023",
        "title": "Corpus-driven grammar acquisition in adult second language learners",
    }
    result = asyncio.run(hc.resolve_citation(entry, mailto="x@y.z"))
    assert result is not None
    assert result["doi"] == "10.1016/j.acorp.2023.100121"
    assert result["resolution"] == "printed_doi" and result["similarity"] >= 0.6


def test_resolve_citation_accepts_printed_doi_when_no_title_comparison_is_possible(monkeypatch):
    """A Crossref lookup failure (network, or the DOI not found) leaves no title to compare
    against, so the printed DOI is accepted as before -- not a regression from prior
    behaviour, since no comparison is possible either way."""
    async def fake_lookup(doi, mailto):
        return None

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    entry = {"doi": "10.1/x", "year": "2020", "title": "Some Title Here"}
    result = asyncio.run(hc.resolve_citation(entry, mailto="x@y.z"))
    assert result is not None
    assert result["title"] == "Some Title Here" and result["similarity"] is None


def test_resolve_citation_carries_the_crossref_title_match_through(monkeypatch):
    async def fake_title_lookup(title, mailto):
        return [{"doi": "10.1/y", "title": "Some Title Here", "type": "journal-article"}]

    monkeypatch.setattr(hc, "crossref_lookup_by_title", fake_title_lookup)
    entry = {"doi": None, "year": "2020", "title": "Some Title Here"}
    result = asyncio.run(hc.resolve_citation(entry, mailto="x@y.z"))
    assert result is not None
    assert result["resolution"] == "crossref_title" and result["doi"] == "10.1/y"


def test_resolve_citation_returns_none_without_a_doi_or_title():
    entry = {"doi": None, "year": "2020", "title": None}
    assert asyncio.run(hc.resolve_citation(entry, mailto="x@y.z")) is None


def test_resolve_citation_carries_crossref_authors_as_a_fallback(monkeypatch):
    """Shipping an empty author list for every real-
    claims item is a defect. Crossref's own ``author`` list (already fetched for the title
    check, free) is carried through as a fallback for when OpenAlex has nothing."""
    async def fake_lookup(doi, mailto):
        return {
            "doi": doi, "title": "Corpus-driven grammar acquisition",
            "type": "journal-article", "authors": ["A. Author", "B. Coauthor"],
        }

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    entry = {"doi": "10.1/x", "year": "2020", "title": "Corpus-driven grammar acquisition"}
    result = asyncio.run(hc.resolve_citation(entry, mailto="x@y.z"))
    assert result is not None and result["authors"] == ["A. Author", "B. Coauthor"]


def test_resolve_citation_rejects_a_title_with_a_unicode_replacement_character(monkeypatch):
    """Real-run finding (2026-09-07): Crossref returned one title with a genuinely malformed
    byte sequence, decoded as U+FFFD; a corrupted title must never be shipped."""
    async def fake_lookup(doi, mailto):
        return {"doi": doi, "title": "Indian�CIranian transnational family",
                "type": "journal-article", "authors": []}

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    entry = {"doi": "10.1/x", "year": "2020", "title": None}
    assert asyncio.run(hc.resolve_citation(entry, mailto="x@y.z")) is None


# --------------------------------------------------------------------------------------
# Per-candidate wall-clock budget (design item 6): a hang inside the reused acquisition
# pipeline must become a recorded skip, not an indefinite block (the P7 real run hung on one
# candidate under httpx on Windows).
# --------------------------------------------------------------------------------------


def test_acquire_one_with_budget_records_a_skip_reason_instead_of_hanging(monkeypatch):
    async def hangs(work, pipeline, *, email, min_chars, min_chunks):
        await asyncio.sleep(10)
        return None, None, None

    monkeypatch.setattr(hc, "acquire_one", hangs)
    source, payload, reason = asyncio.run(
        hc.acquire_one_with_budget(
            {}, {}, email="x@y.z", min_chars=1, min_chunks=1, budget_s=0.05,
        )
    )
    assert source is None and payload is None and reason == "timeout_budget_exceeded"


def test_acquire_one_with_budget_records_a_reason_for_any_other_exception(monkeypatch):
    """Real-run finding (2026-09-07): a print-encoding crash deep inside the reused pipeline
    (a non-ASCII character hitting the Windows console's non-UTF-8 codepage) is not a
    ``TimeoutError`` but must still be turned into a recorded skip for this one candidate --
    otherwise it propagates through ``asyncio.gather`` and aborts the entire harvest, losing
    every review processed so far in the same run."""
    async def raises(work, pipeline, *, email, min_chars, min_chunks):
        raise UnicodeEncodeError("gbk", "x", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr(hc, "acquire_one", raises)
    source, payload, reason = asyncio.run(
        hc.acquire_one_with_budget({}, {}, email="x@y.z", min_chars=1, min_chunks=1)
    )
    assert source is None and payload is None
    assert reason is not None and "UnicodeEncodeError" in reason


def test_acquire_one_with_budget_passes_through_a_fast_result(monkeypatch):
    async def fast(work, pipeline, *, email, min_chars, min_chunks):
        return {"doi": "x"}, {"doi": "x", "chunks": []}, None

    monkeypatch.setattr(hc, "acquire_one", fast)
    source, payload, reason = asyncio.run(
        hc.acquire_one_with_budget({}, {}, email="x@y.z", min_chars=1, min_chunks=1, budget_s=5.0)
    )
    assert source == {"doi": "x"} and payload == {"doi": "x", "chunks": []} and reason is None


# --------------------------------------------------------------------------------------
# Acquisition-outcome bucketing (reuses build_hss_set.acquire_one's reason vocabulary)
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# extract_candidates: OpenAlex authorship enrichment is fetched LAZILY, only for a citation
# that actually gets kept (real-run finding, 2026-09-07): a live run resolved 467 citations
# but kept only 28, and fetching OpenAlex authorships for every RESOLVED citation (rather than
# only the KEPT ones) exhausted OpenAlex's per-request daily budget in one run, well before
# reaching anywhere near the --max-reviews target. Fetching authors only after a candidate's
# full text is actually acquired cuts OpenAlex authorship calls by an order of magnitude.
# --------------------------------------------------------------------------------------

REFERENCES_ONE_ENTRY = (
    "References\n"
    "Smith, J. A. (2019). Feedback and second language writing development. Journal of "
    "Second Language Writing, 45, 12-29. https://doi.org/10.1016/j.jslw.2019.03.002\n"
)


def _review_payload_with_one_citation(sentence: str) -> dict:
    return {
        "doi": "10.1/review0", "title": "A Review",
        "chunks": [
            {"section": "body", "text": sentence},
            {"section": "references", "text": REFERENCES_ONE_ENTRY},
        ],
    }


def test_extract_candidates_fetches_openalex_authors_only_for_a_kept_candidate(
    tmp_path: Path, monkeypatch,
):
    calls = {"openalex_authors": 0}
    sentence = (
        "Smith (2019) found that structured feedback improved learner accuracy across an "
        "entire academic term for every cohort studied in the sample overall."
    )
    payload = _review_payload_with_one_citation(sentence)

    async def fake_crossref_doi(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing development",
            "type": "journal-article",
        }

    async def fake_openalex_work(doi, mailto):
        calls["openalex_authors"] += 1
        return {"authorships": [{"author": {"display_name": "J. A. Smith"}}]}

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return (
            {"doi": work["doi"]},
            {"doi": work["doi"], "title": work["title"], "authors": [],
             "source_url": "https://x/y.pdf", "fetched_at": "2026-09-07T00:00:00+00:00",
             "n_chars": 20000, "chunks": [{"section": "results", "text": "It helped."}]},
            None,
        )

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert len(kept) == 1 and stats["n_kept"] == 1
    assert calls["openalex_authors"] == 1  # fetched exactly once, for the one kept candidate
    cached = json.loads((fulltext_dir / f"{hc.doi_slug('10.1016/j.jslw.2019.03.002')}.json")
                         .read_text(encoding="utf-8"))
    assert cached["authors"] == ["J. A. Smith"]


def test_extract_candidates_falls_back_to_crossref_authors_when_openalex_has_nothing(
    tmp_path: Path, monkeypatch,
):
    sentence = (
        "Smith (2019) found that structured feedback improved learner accuracy across an "
        "entire academic term for every cohort studied in the sample overall."
    )
    payload = _review_payload_with_one_citation(sentence)

    async def fake_crossref_doi(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing development",
            "type": "journal-article", "authors": ["Crossref J. Smith"],
        }

    async def fake_openalex_work_empty(doi, mailto):
        return None  # over budget / not indexed -- OpenAlex has nothing to offer

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return (
            {"doi": work["doi"]},
            {"doi": work["doi"], "title": work["title"], "authors": [],
             "source_url": "https://x/y.pdf", "fetched_at": "2026-09-07T00:00:00+00:00",
             "n_chars": 20000, "chunks": [{"section": "results", "text": "It helped."}]},
            None,
        )

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work_empty)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert len(kept) == 1
    cached = json.loads((fulltext_dir / f"{hc.doi_slug('10.1016/j.jslw.2019.03.002')}.json")
                         .read_text(encoding="utf-8"))
    assert cached["authors"] == ["Crossref J. Smith"]


def test_extract_candidates_never_calls_openalex_authors_when_acquisition_fails(
    tmp_path: Path, monkeypatch,
):
    """The optimisation's whole point: a candidate that resolves fine but never acquires (not
    OA, or below the min-chars/min-chunks floor) must cost zero OpenAlex authorship calls."""
    calls = {"openalex_authors": 0}
    sentence = (
        "Smith (2019) found that structured feedback improved learner accuracy across an "
        "entire academic term for every cohort studied in the sample overall."
    )
    payload = _review_payload_with_one_citation(sentence)

    async def fake_crossref_doi(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing development",
            "type": "journal-article",
        }

    async def fake_openalex_work(doi, mailto):
        calls["openalex_authors"] += 1
        return {"authorships": [{"author": {"display_name": "J. A. Smith"}}]}

    async def fake_acquire_one_fails(work, pipeline, *, email, min_chars, min_chunks):
        return None, None, "not_oa"

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one_fails)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert kept == [] and stats["n_kept"] == 0
    assert calls["openalex_authors"] == 0


def test_extract_candidates_reuses_a_cached_cited_work_without_any_openalex_call(
    tmp_path: Path, monkeypatch,
):
    """Resumability (design item 6): a cited work already in the fulltext cache is never
    re-fetched, and its authors are trusted as already-correct from when it was first
    fetched -- no OpenAlex call at all on this path."""
    calls = {"openalex_authors": 0, "acquire_one": 0}
    sentence = (
        "Smith (2019) found that structured feedback improved learner accuracy across an "
        "entire academic term for every cohort studied in the sample overall."
    )
    payload = _review_payload_with_one_citation(sentence)

    async def fake_crossref_doi(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing development",
            "type": "journal-article",
        }

    async def fake_openalex_work(doi, mailto):
        calls["openalex_authors"] += 1
        return {"authorships": []}

    async def fake_acquire_one(*a, **kw):
        calls["acquire_one"] += 1
        raise AssertionError("must not be called for an already-cached cited work")

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    cited_doi = "10.1016/j.jslw.2019.03.002"
    write_json(fulltext_dir / f"{hc.doi_slug(cited_doi)}.json", {
        "doi": cited_doi, "title": "Feedback and second language writing development",
        "authors": ["Pre-existing Author"], "source_url": "https://x/y.pdf",
        "fetched_at": "2026-09-06T00:00:00+00:00", "n_chars": 20000,
        "chunks": [{"section": "results", "text": "It helped."}],
    })
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert len(kept) == 1 and stats["n_kept"] == 1
    assert calls["openalex_authors"] == 0 and calls["acquire_one"] == 0


def test_fetch_reviews_reprocesses_a_cached_review_this_runs_discovery_did_not_return(
    tmp_path: Path, monkeypatch,
):
    """A review already fetched in a previous run must be recomputed even when this run's own
    discovery call returns a narrower (or empty) result -- e.g. an OpenAlex channel over its
    request budget -- so a fix to this module's own extraction logic still reaches it."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    cached_doi = "10.1/already-cached"
    _write_review_cache(reviews_dir, cached_doi, "Cached Review", [], {})

    processed: list[str] = []

    async def fake_discover(mailto, *, max_reviews):
        return []  # this run's discovery found nothing (e.g. over budget)

    async def fake_process_review(work, **kwargs):
        processed.append(work["doi"])

    monkeypatch.setattr(hc, "load_pipeline", lambda: {})
    monkeypatch.setattr(hc, "discover_reviews", fake_discover)
    monkeypatch.setattr(hc, "_process_review", fake_process_review)
    n = asyncio.run(hc.fetch_reviews(
        mailto="x@y.z", excluded=set(), reviews_cache_dir=reviews_dir,
        fulltext_cache_dir=fulltext_dir, max_reviews=300, max_per_review=4, min_chars=1,
        min_chunks=1,
    ))
    assert n == 0  # what THIS run's discovery returned
    assert processed == [cached_doi]  # but the cached review was still reprocessed


def test_fetch_reviews_does_not_overwrite_a_cached_readings_payload(
    tmp_path: Path, monkeypatch,
):
    """Both --from-readings and the default
    regex-extraction path write into the same DEFAULT_REVIEWS_CACHE_DIR, and fetch_reviews
    deliberately reprocesses every cached review this run's own discovery did not return
    (test above). Left unguarded, that resumability design would re-run regex extraction over
    a readings payload's placeholder chunks, yielding zero candidates and silently flipping the
    payload's extraction back to "regex" -- destroying reader-extracted evidence with a plain
    ``python harvest_real_claims.py`` invocation. extract_candidates must never even be called
    for such a payload, and the cache file on disk must be byte-for-byte unchanged."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/readings-review"
    candidates = [_candidate(0, "10.1/cited0", extraction="readings")]
    _write_review_cache(
        reviews_dir, review_doi, "Readings Review", candidates, {"n_kept": 1},
        extraction="readings",
    )
    cache_path = reviews_dir / f"{hc.doi_slug(review_doi)}.json"
    before = cache_path.read_text(encoding="utf-8")

    async def fake_discover(mailto, *, max_reviews):
        return []

    calls: list[str] = []

    async def tracking_extract_candidates(*a, **kw):
        calls.append("called")
        return [], {}

    monkeypatch.setattr(hc, "load_pipeline", lambda: {})
    monkeypatch.setattr(hc, "discover_reviews", fake_discover)
    monkeypatch.setattr(hc, "extract_candidates", tracking_extract_candidates)
    asyncio.run(hc.fetch_reviews(
        mailto="x@y.z", excluded=set(), reviews_cache_dir=reviews_dir,
        fulltext_cache_dir=fulltext_dir, max_reviews=300, max_per_review=4, min_chars=1,
        min_chunks=1,
    ))
    assert calls == []  # extract_candidates (regex extraction) was never invoked
    assert cache_path.read_text(encoding="utf-8") == before
    after = json.loads(cache_path.read_text(encoding="utf-8"))
    assert after["extraction"] == "readings"
    assert after["candidates"] == candidates


def test_process_review_does_not_propagate_an_unexpected_extraction_failure(
    tmp_path: Path, monkeypatch,
):
    """A per-review failure (a bug in this module's own extraction logic, or anything else
    unexpected) must not abort the whole ``asyncio.gather`` over up to --max-reviews reviews;
    it is logged and the review is simply skipped for this run."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/review0"
    _write_review_cache(reviews_dir, review_doi, "Review 0", [], {})

    async def raises(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(hc, "extract_candidates", raises)
    asyncio.run(hc._process_review(
        {"doi": review_doi}, mailto="x@y.z", pipeline={}, excluded=set(),
        reviews_cache_dir=reviews_dir, fulltext_cache_dir=fulltext_dir, min_chars=1,
        min_chunks=1, max_per_review=4, semaphore=asyncio.Semaphore(1),
    ))  # must return normally, not raise


def test_acquire_outcome_buckets_cover_every_reason():
    assert hc.acquire_outcome_buckets("not_oa") == {"oa": 0, "fetched": 0, "kept": 0}
    assert hc.acquire_outcome_buckets("html_not_pdf") == {"oa": 1, "fetched": 0, "kept": 0}
    assert hc.acquire_outcome_buckets("min_chars") == {"oa": 1, "fetched": 1, "kept": 0}
    assert hc.acquire_outcome_buckets("min_chunks") == {"oa": 1, "fetched": 1, "kept": 0}
    assert hc.acquire_outcome_buckets(None) == {"oa": 1, "fetched": 1, "kept": 1}
    assert hc.acquire_outcome_buckets("http_403") == {"oa": 1, "fetched": 0, "kept": 0}


# --------------------------------------------------------------------------------------
# Review-work discovery filters (design item 1)
# --------------------------------------------------------------------------------------


def test_is_review_title_matches_the_markers():
    assert hc.is_review_title("A systematic review of corrective feedback studies")
    assert hc.is_review_title("A meta-analysis of task-based interventions")
    assert hc.is_review_title("Second language motivation: state of the art")
    assert hc.is_review_title("Toward a synthesis of willingness to communicate research")
    assert hc.is_review_title("Setting a research agenda for L2 pragmatics")
    assert not hc.is_review_title("Learners' motivation in an online EFL classroom")
    assert not hc.is_review_title(None)


def test_is_front_matter_title_excludes_editorials_and_erratum_notices():
    """'10.18806/tesl.v38i2.1363' is a guest-editorial
    front-matter document, not a research article, and must never enter the discovery pool."""
    assert hc.is_front_matter_title("A Word From The Guest Editors")
    assert hc.is_front_matter_title("Editorial: introducing this special issue")
    assert hc.is_front_matter_title("Erratum to: A study of feedback")
    assert hc.is_front_matter_title("Corrigendum to a previous article")
    assert not hc.is_front_matter_title("A review of corrective feedback studies")
    assert not hc.is_front_matter_title(None)


def test_is_review_work_checks_title_or_type_and_excludes_front_matter():
    assert hc.is_review_work({"title": "A review of feedback studies", "type": "article"})
    assert hc.is_review_work({"title": "Something else entirely", "type": "review"})
    assert not hc.is_review_work({"title": "An empirical study of feedback", "type": "article"})
    assert not hc.is_review_work({"title": "A Word From The Guest Editors", "type": "article"})


def test_has_pdf_location_requires_a_pdf_url_anywhere_on_the_work():
    assert hc.has_pdf_location({"best_oa_location": {"pdf_url": "https://x/y.pdf"}})
    assert not hc.has_pdf_location({"best_oa_location": {"pdf_url": None}})
    assert not hc.has_pdf_location({"best_oa_location": {}})
    assert not hc.has_pdf_location({})
    # design item 1: "best_oa_location or any location with pdf_url"
    assert hc.has_pdf_location({
        "best_oa_location": {}, "locations": [{"pdf_url": None}, {"pdf_url": "https://x/z.pdf"}],
    })
    assert not hc.has_pdf_location({"best_oa_location": {}, "locations": [{"pdf_url": None}]})


def test_openalex_journal_query_url_uses_2015_lower_bound_and_source_id_filter():
    url = hc.openalex_journal_query_url("S123456", "x@y.z", cursor="*")
    assert "primary_location.source.id:S123456" in url and "publication_year:>2014" in url
    assert "type:article|review" in url and "x@y.z" in url and "cursor=*" in url


def test_topic_search_url_and_openalex_topic_query_url():
    url = hc.topic_search_url("Applied Linguistics", "x@y.z")
    assert "api.openalex.org/topics" in url and "Applied" in url
    turl = hc.openalex_topic_query_url("T123", "x@y.z", cursor="*")
    assert "primary_topic.id:T123" in turl and "publication_year:>2014" in turl
    assert "cursor=*" in turl


def test_subfield_search_url_and_openalex_subfield_query_url():
    url = hc.subfield_search_url("Linguistics and Language", "x@y.z")
    assert "api.openalex.org/subfields" in url and "Linguistics" in url
    surl = hc.openalex_subfield_query_url("SF123", "x@y.z", cursor="*")
    assert "primary_topic.subfield.id:SF123" in surl and "publication_year:>2014" in surl


def test_openalex_title_keyword_query_url_encodes_multiword_keywords_and_scopes_to_a_subfield():
    """A title-keyword channel with no subfield scope matches every OA review in OpenAlex
    regardless of discipline; live discovery measured this crowding out every applied-
    linguistics candidate with JAMA/Cochrane medical reviews once the pool was capped."""
    url = hc.openalex_title_keyword_query_url("state of the art", "SF123", "x@y.z", cursor="*")
    assert "title.search:state" in url and "publication_year:>2014" in url
    assert "primary_topic.subfield.id:SF123" in url
    assert "x@y.z" in url


def test_source_lookup_url_shape():
    url = hc.source_lookup_url("1234-5678", "x@y.z")
    assert "api.openalex.org/sources" in url and "issn:1234-5678" in url and "x@y.z" in url


# --------------------------------------------------------------------------------------
# resolve_source_ids: a journal that already carries a known OpenAlex source id (JOURNALS,
# from build_hss_set) is reused without a network call.
# --------------------------------------------------------------------------------------


def test_resolve_source_ids_reuses_known_ids_without_a_network_call():
    journals = [{"name": "A", "issn": "1-2", "openalex": "S123"}]
    out = asyncio.run(hc.resolve_source_ids(journals, "x@y.z"))
    assert out == {"A": "S123"}


# --------------------------------------------------------------------------------------
# Discovery pagination on a mocked client (design item 1): cursor pagination over an
# OpenAlex-shaped ``{"results": [...], "meta": {"next_cursor": ...}}`` sequence, deterministic
# and bounded, with no real network client involved.
# --------------------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeClient:
    """A minimal stand-in for ``httpx.AsyncClient``: ``get(url)`` returns the next queued
    page regardless of *url*, so the caller's url-building logic is exercised only through
    its cursor-carrying callback, not through this fake's own request matching."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = 0

    async def get(self, url):
        self.calls += 1
        if self.calls > len(self._pages):
            return _FakeResponse({"results": [], "meta": {"next_cursor": None}})
        return self._pages[self.calls - 1]


def test_openalex_cursor_pages_follows_next_cursor_until_none():
    pages = [
        _FakeResponse({"results": [{"id": 1}, {"id": 2}], "meta": {"next_cursor": "abc"}}),
        _FakeResponse({"results": [{"id": 3}], "meta": {"next_cursor": None}}),
    ]
    client = _FakeClient(pages)
    out = asyncio.run(
        hc.openalex_cursor_pages(client, lambda cursor: f"http://x/?cursor={cursor}")
    )
    assert [r["id"] for r in out] == [1, 2, 3]
    assert client.calls == 2


def test_openalex_cursor_pages_respects_max_pages():
    pages = [
        _FakeResponse({"results": [{"id": i}], "meta": {"next_cursor": "more"}})
        for i in range(10)
    ]
    client = _FakeClient(pages)
    out = asyncio.run(
        hc.openalex_cursor_pages(
            client, lambda cursor: f"http://x/?cursor={cursor}", max_pages=3,
        )
    )
    assert len(out) == 3
    assert client.calls == 3


def test_openalex_cursor_pages_stops_on_a_non_200_response():
    pages = [_FakeResponse({}, status_code=500)]
    client = _FakeClient(pages)
    out = asyncio.run(
        hc.openalex_cursor_pages(client, lambda cursor: f"http://x/?cursor={cursor}")
    )
    assert out == []
    assert client.calls == 1


def test_openalex_cursor_pages_passes_the_cursor_value_through_the_callback():
    seen_cursors: list[str] = []

    def url_for(cursor):
        seen_cursors.append(cursor)
        return f"http://x/?cursor={cursor}"

    pages = [
        _FakeResponse({"results": [{"id": 1}], "meta": {"next_cursor": "page2"}}),
        _FakeResponse({"results": [{"id": 2}], "meta": {"next_cursor": None}}),
    ]
    client = _FakeClient(pages)
    asyncio.run(hc.openalex_cursor_pages(client, url_for))
    assert seen_cursors == ["*", "page2"]


# --------------------------------------------------------------------------------------
# Deterministic sampling: shuffled review order + test-FIRST, then dev split (design item 5),
# because a dev-first split would starve the held-out test set
# whenever total yield is below the dev target.
# --------------------------------------------------------------------------------------


def test_shuffled_review_order_is_deterministic_and_seed_sensitive():
    dois = [f"10.1/r{i}" for i in range(10)]
    a = hc.shuffled_review_order(dois, hc.DEFAULT_SEED)
    b = hc.shuffled_review_order(list(reversed(dois)), hc.DEFAULT_SEED)
    assert a == b  # order-of-input independent (sorted before shuffling)
    c = hc.shuffled_review_order(dois, hc.DEFAULT_SEED + 1)
    assert a != c
    assert sorted(a) == sorted(dois)


def test_split_items_by_review_keeps_dev_and_test_reviews_disjoint():
    items_by_review = {
        f"10.1/r{i}": [{"claim": f"claim-{i}-{j}"} for j in range(3)] for i in range(6)
    }
    order = [f"10.1/r{i}" for i in range(6)]
    dev, test, split = hc.split_items_by_review(
        order, items_by_review, n_dev=6, n_test=6, max_per_review=3,
    )
    assert len(dev) == 6 and len(test) == 6
    dev_reviews = {row["claim"].split("-")[1] for row in dev}
    test_reviews = {row["claim"].split("-")[1] for row in test}
    assert not dev_reviews & test_reviews
    assert set(split.values()) <= {"dev", "test"}


def test_split_items_by_review_fills_test_before_dev():
    """The fix itself: a single review's items go to TEST first, not dev, once yield exists."""
    items_by_review = {"10.1/r0": [{"i": j} for j in range(10)]}
    dev, test, split = hc.split_items_by_review(
        ["10.1/r0"], items_by_review, n_dev=20, n_test=20, max_per_review=4,
    )
    assert len(test) == 4 and len(dev) == 0
    assert split == {"10.1/r0": "test"}


def test_split_items_by_review_fills_test_before_dev_even_when_total_yield_is_low():
    """Design item 5: with a low total yield, test must never be starved
    just because dev would have been filled first under the old (dev-first) ordering."""
    items_by_review = {f"10.1/r{i}": [{"i": i}] for i in range(3)}
    order = list(items_by_review)
    dev, test, split = hc.split_items_by_review(
        order, items_by_review, n_dev=20, n_test=60, max_per_review=4,
    )
    assert len(test) == 3 and len(dev) == 0
    assert set(split.values()) == {"test"}


def test_split_items_by_review_skips_reviews_once_both_targets_are_met():
    items_by_review = {f"10.1/r{i}": [{"i": i}] for i in range(4)}
    dev, test, split = hc.split_items_by_review(
        list(items_by_review), items_by_review, n_dev=1, n_test=1, max_per_review=4,
    )
    assert len(dev) == 1 and len(test) == 1
    assert len(split) == 2


# --------------------------------------------------------------------------------------
# Item assembly and the run_hss.py contract
# --------------------------------------------------------------------------------------


def test_build_item_row_matches_run_hss_item_keys_contract():
    review = {"doi": "10.1/review", "title": "A Review of Feedback Studies"}
    cited = {"doi": "10.1/cited", "title": "Feedback in L2 Writing", "authors": ["A. Author"]}
    row = hc.build_item_row(
        sentence_info={
            "sentence": "Smith (2019) found that feedback improved accuracy across cohorts.",
            "context_before": "Prior sentence here.",
            "citation_text": "Smith (2019)",
            "cited_doi": "10.1/cited",
            "cited_title": "Feedback in L2 Writing",
            "cited_year": "2019",
            "resolution": "printed_doi",
            "similarity": None,
        },
        review=review,
        cited=cited,
    )
    for key in rh.ITEM_KEYS:
        if key == "item_id":
            continue  # assigned by the caller once dev/test sampling order is known
        assert key in row, key
    assert row["source_doi"] == "10.1/review" and row["chunk_doi"] == "10.1/cited"
    assert row["title"] == "Feedback in L2 Writing" and row["authors"] == ["A. Author"]
    assert row["rule"] == "real" and row["expected"] is None and row["alteration"] is None


def test_run_hss_loads_real_claims_rows_unchanged(tmp_path: Path):
    """brief: run_hss.py --claims real_claims_test.jsonl --cache-dir ... must work unchanged."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cited_doi = "10.1/cited"
    write_json(cache_dir / f"{hc.doi_slug(cited_doi)}.json", {
        "doi": cited_doi, "title": "Feedback in L2 Writing", "authors": ["A. Author"],
        "source_url": "https://x/y.pdf", "fetched_at": "2026-09-06T00:00:00+00:00",
        "n_chars": 20000, "chunks": [{"section": "results", "text": "Feedback helped."}],
    })
    row = hc.build_item_row(
        sentence_info={
            "sentence": "Smith (2019) found that feedback improved accuracy across cohorts.",
            "context_before": "", "citation_text": "Smith (2019)", "cited_doi": cited_doi,
            "cited_title": "Feedback in L2 Writing", "cited_year": "2019",
            "resolution": "printed_doi", "similarity": None,
        },
        review={"doi": "10.1/review", "title": "A Review"},
        cited={"doi": cited_doi, "title": "Feedback in L2 Writing", "authors": ["A. Author"]},
    )
    row["item_id"] = "real-test-01"
    claims_path = tmp_path / "real_claims_test.jsonl"
    claims_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = rh.load_items(claims_path, cache_dir)
    assert len(loaded) == 1
    item = loaded[0]
    assert item["chunks"] == ["Feedback helped."]
    assert item["title"] == "Feedback in L2 Writing" and item["authors"] == ["A. Author"]
    assert item["claim"].startswith("Smith (2019)")


# --------------------------------------------------------------------------------------
# build_outputs: offline rebuild from review + fulltext caches (the --skip-fetch path)
# --------------------------------------------------------------------------------------


def _write_review_cache(
    cache_dir: Path, doi: str, title: str, candidates: list, stats: dict,
    *, extraction: str | None = None,
):
    payload = {
        "doi": doi, "title": title, "authors": ["Reviewer A"],
        "source_url": "https://x/review.pdf", "fetched_at": "2026-09-06T00:00:00+00:00",
        "n_chars": 30000, "chunks": [{"section": "body", "text": "Body text."}],
        "candidates": candidates, "stats": stats,
    }
    if extraction is not None:
        payload["extraction"] = extraction
    write_json(cache_dir / f"{hc.doi_slug(doi)}.json", payload)


def _write_cited_cache(cache_dir: Path, doi: str, title: str):
    # build_outputs now runs check_cache_identity over the
    # fulltext cache, so a fixture's own "text" must actually describe its own "title"/"authors"
    # (title tokens and the author surname both present) or it is dropped as a wrong-work
    # mis-fetch before these tests' own assertions ever run.
    write_json(cache_dir / f"{hc.doi_slug(doi)}.json", {
        "doi": doi, "title": title, "authors": ["Cited Author"],
        "source_url": "https://x/cited.pdf", "fetched_at": "2026-09-06T00:00:00+00:00",
        "n_chars": 20000,
        "chunks": [{"section": "results", "text": f"{title}. Cited Author reported the effect."}],
    })


def _candidate(i: int, cited_doi: str, *, extraction: str | None = None) -> dict:
    row = {
        "sentence": f"Smith (2019) found effect number {i} across every cohort studied here.",
        "context_before": "", "citation_text": "Smith (2019)", "cited_doi": cited_doi,
        "cited_title": f"Cited work {i}", "cited_year": "2019", "resolution": "printed_doi",
        "similarity": None,
    }
    if extraction is not None:
        row["extraction"] = extraction
    return row


def test_build_outputs_rebuilds_deterministically_from_caches(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    stats = {"n_sentences_seen": 40, "n_single_citation": 3, "n_resolved": 2, "n_oa": 2,
              "n_fetched": 2, "n_kept": 1}
    for i in range(4):
        review_doi = f"10.1/review{i}"
        cited_doi = f"10.1/cited{i}"
        _write_cited_cache(fulltext_dir, cited_doi, f"Cited work {i}")
        _write_review_cache(
            reviews_dir, review_doi, f"Review {i}", [_candidate(i, cited_doi)], stats,
        )
    dev1, test1, info1 = hc.build_outputs(
        reviews_dir, fulltext_dir, seed=hc.DEFAULT_SEED, n_dev=2, n_test=2, max_per_review=4,
    )
    dev2, test2, info2 = hc.build_outputs(
        reviews_dir, fulltext_dir, seed=hc.DEFAULT_SEED, n_dev=2, n_test=2, max_per_review=4,
    )
    assert dev1 == dev2 and test1 == test2 and info1["review_split"] == info2["review_split"]
    assert len(dev1) == 2 and len(test1) == 2
    assert info1["n_sentences_seen"] == 4 * 40
    assert info1["n_kept"] == 4 * 1
    dev_reviews = {row["review_doi"] for row in dev1}
    test_reviews = {row["review_doi"] for row in test1}
    assert not dev_reviews & test_reviews
    ids = [row["item_id"] for row in dev1]
    assert ids == [f"real-dev-{i:02d}" for i in range(1, len(dev1) + 1)]


def test_build_outputs_excludes_hss_source_dois_and_self_citations(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/review0"
    excluded_doi = "10.1/excluded"
    good_doi = "10.1/good"
    _write_cited_cache(fulltext_dir, excluded_doi, "Excluded work")
    _write_cited_cache(fulltext_dir, good_doi, "Good work")
    _write_cited_cache(fulltext_dir, review_doi, "Self citation target")
    candidates = [
        _candidate(0, excluded_doi), _candidate(1, good_doi), _candidate(2, review_doi),
    ]
    _write_review_cache(reviews_dir, review_doi, "Review 0", candidates, {})
    dev, test, info = hc.build_outputs(
        reviews_dir, fulltext_dir, n_dev=5, n_test=5, excluded={excluded_doi},
    )
    all_items = dev + test
    assert len(all_items) == 1 and all_items[0]["cited_doi"] == good_doi
    assert info["excluded_dois"] == [excluded_doi]


def test_build_outputs_requires_cited_doi_to_have_a_fulltext_cache_entry(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/review0"
    # cited_doi has no corresponding fulltext cache file at all
    _write_review_cache(reviews_dir, review_doi, "Review 0", [_candidate(0, "10.1/missing")], {})
    dev, test, info = hc.build_outputs(reviews_dir, fulltext_dir, n_dev=5, n_test=5)
    assert dev == [] and test == []


def test_build_outputs_drops_a_wrong_work_fulltext_cache_entry_via_identity_check(
    tmp_path: Path,
):
    """build_outputs runs check_cache_identity over the
    fulltext cache before using it at all, so a cited-work cache entry whose own recorded
    title/authors do not match its own fetched text is dropped regardless of which acquisition
    path (or which prior run's was_cached reuse) put it there -- closing the same hole on the
    regex path and on --skip-fetch rebuilds that finding 2's own targeted was_cached fix does
    not reach by itself."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/review0"
    wrong_work_doi = "10.1/wrong-work"
    good_doi = "10.1/good"
    write_json(fulltext_dir / f"{hc.doi_slug(wrong_work_doi)}.json", {
        "doi": wrong_work_doi, "title": "A Completely Unrelated Title About Phonology",
        "authors": ["Some Author"], "source_url": "https://x/y.pdf",
        "fetched_at": "2026-09-06T00:00:00+00:00", "n_chars": 20000,
        "chunks": [{"section": "results", "text": "Nothing here matches the title above."}],
    })
    _write_cited_cache(fulltext_dir, good_doi, "Good work")
    _write_review_cache(
        reviews_dir, review_doi, "Review 0",
        [_candidate(0, wrong_work_doi), _candidate(1, good_doi)], {},
    )
    dev, test, info = hc.build_outputs(reviews_dir, fulltext_dir, n_dev=5, n_test=5)
    all_items = dev + test
    assert len(all_items) == 1 and all_items[0]["cited_doi"] == good_doi
    assert info["n_wrong_work_sources"] == 1
    assert info["wrong_work_sources"][0]["doi"] == wrong_work_doi


def test_build_outputs_extraction_filter_isolates_readings_items_and_reports_both_counts(
    tmp_path: Path,
):
    """A --from-readings run must not silently ship items a
    regex-extraction cache already sitting in the same directory produced. A cache dir with one
    un-stamped ("no extraction key" -- predates this field, must default to "regex") regex
    review and one "readings"-stamped review, built with extraction="readings", yields only the
    readings item, while info["n_items_by_extraction"] still discloses both counts."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    regex_review_doi = "10.1/regex-review"
    readings_review_doi = "10.1/readings-review"
    regex_cited_doi = "10.1/regex-cited"
    readings_cited_doi = "10.1/readings-cited"
    _write_cited_cache(fulltext_dir, regex_cited_doi, "Regex cited work")
    _write_cited_cache(fulltext_dir, readings_cited_doi, "Readings cited work")
    _write_review_cache(  # no extraction key at all -- must default to "regex"
        reviews_dir, regex_review_doi, "Regex Review", [_candidate(0, regex_cited_doi)], {},
    )
    _write_review_cache(
        reviews_dir, readings_review_doi, "Readings Review",
        [_candidate(1, readings_cited_doi, extraction="readings")], {}, extraction="readings",
    )
    dev, test, info = hc.build_outputs(
        reviews_dir, fulltext_dir, n_dev=5, n_test=5, extraction="readings",
    )
    all_items = dev + test
    assert len(all_items) == 1
    assert all_items[0]["cited_doi"] == readings_cited_doi
    assert all_items[0]["extraction"] == "readings"
    assert info["extraction_filter"] == "readings"
    assert info["n_items_by_extraction"] == {"regex": 1, "readings": 1}


def test_build_outputs_extraction_none_keeps_both_paths_unchanged(tmp_path: Path):
    """The default (extraction=None) must reproduce the unfiltered behaviour byte for byte:
    both a "regex" and a "readings" review contribute items."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    _write_cited_cache(fulltext_dir, "10.1/regex-cited", "Regex cited work")
    _write_cited_cache(fulltext_dir, "10.1/readings-cited", "Readings cited work")
    _write_review_cache(
        reviews_dir, "10.1/regex-review", "Regex Review",
        [_candidate(0, "10.1/regex-cited")], {},
    )
    _write_review_cache(
        reviews_dir, "10.1/readings-review", "Readings Review",
        [_candidate(1, "10.1/readings-cited", extraction="readings")], {},
        extraction="readings",
    )
    dev, test, info = hc.build_outputs(reviews_dir, fulltext_dir, n_dev=5, n_test=5)
    all_items = dev + test
    assert {row["cited_doi"] for row in all_items} == {"10.1/regex-cited", "10.1/readings-cited"}
    assert info["extraction_filter"] is None
    assert info["n_items_by_extraction"] == {"regex": 1, "readings": 1}


def test_main_skip_fetch_with_zero_items_still_writes_readable_empty_output_files(
    tmp_path: Path,
):
    """Real-network smoke test (2026-09-06) found this: a run where every discovered review
    fails resolution/OA yields zero dev and zero test items; ``append_jsonl`` never creates a
    file for an empty row list, so ``sha256_file`` on the (nonexistent) output previously
    crashed with ``FileNotFoundError`` instead of completing the (valid, if empty) build."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    rc = hc.main([
        "--skip-fetch", "--reviews-cache-dir", str(reviews_dir),
        "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
        "--out-test", str(out_test), "--build-out", str(build_out),
        "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
        "--hss-test-v3-sources", str(empty_sources),
    ])
    assert rc == 0
    assert out_dev.exists() and out_dev.read_text(encoding="utf-8") == ""
    assert out_test.exists() and out_test.read_text(encoding="utf-8") == ""
    build = json.loads(build_out.read_text(encoding="utf-8"))
    assert build["n_dev"] == 0 and build["n_test"] == 0
    assert build["out_dev_sha256"] == hc.sha256_file(out_dev)


# --------------------------------------------------------------------------------------
# Exclusion of the constructed sets' sources (design item 3 "keep the exclusions"): the two
# HSS source lists plus the frozen v3 test set's own sources file.
# --------------------------------------------------------------------------------------


def test_main_skip_fetch_excludes_dois_from_all_three_source_files(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    review_doi = "10.1/review0"
    excluded_doi = "10.1/excluded-v3"
    good_doi = "10.1/good"
    _write_cited_cache(fulltext_dir, excluded_doi, "Excluded v3 work")
    _write_cited_cache(fulltext_dir, good_doi, "Good work")
    _write_review_cache(
        reviews_dir, review_doi, "Review 0",
        [_candidate(0, excluded_doi), _candidate(1, good_doi)], {},
    )
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    v3_sources = tmp_path / "hss_test_v3_sources.json"
    write_json(empty_sources, [])
    write_json(v3_sources, [{"doi": excluded_doi}])
    rc = hc.main([
        "--skip-fetch", "--reviews-cache-dir", str(reviews_dir),
        "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
        "--out-test", str(out_test), "--build-out", str(build_out),
        "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
        "--hss-test-v3-sources", str(v3_sources), "--n-dev", "5", "--n-test", "5",
    ])
    assert rc == 0
    dev_rows = [json.loads(x) for x in out_dev.read_text(encoding="utf-8").splitlines()]
    test_rows = [json.loads(x) for x in out_test.read_text(encoding="utf-8").splitlines()]
    all_items = dev_rows + test_rows
    assert len(all_items) == 1 and all_items[0]["cited_doi"] == good_doi
    build = json.loads(build_out.read_text(encoding="utf-8"))
    assert excluded_doi in build["excluded_dois"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_main_help_and_dry_run(tmp_path: Path, capsys):
    import pytest

    with pytest.raises(SystemExit) as exc:
        hc.main(["--help"])
    assert exc.value.code == 0
    rc = hc.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry" in out.lower() or "plan" in out.lower()


def test_main_skip_fetch_writes_dev_test_and_build_json(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    stats = {"n_sentences_seen": 10, "n_single_citation": 2, "n_resolved": 1, "n_oa": 1,
              "n_fetched": 1, "n_kept": 1}
    for i in range(3):
        review_doi = f"10.1/review{i}"
        cited_doi = f"10.1/cited{i}"
        _write_cited_cache(fulltext_dir, cited_doi, f"Cited work {i}")
        _write_review_cache(
            reviews_dir, review_doi, f"Review {i}", [_candidate(i, cited_doi)], stats,
        )
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    rc = hc.main([
        "--skip-fetch", "--reviews-cache-dir", str(reviews_dir),
        "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
        "--out-test", str(out_test), "--build-out", str(build_out),
        "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
        "--hss-test-v3-sources", str(empty_sources),
        "--n-dev", "2", "--n-test", "1", "--seed", str(hc.DEFAULT_SEED),
    ])
    assert rc == 0
    dev_rows = [json.loads(x) for x in out_dev.read_text(encoding="utf-8").splitlines()]
    test_rows = [json.loads(x) for x in out_test.read_text(encoding="utf-8").splitlines()]
    assert len(dev_rows) == 2 and len(test_rows) == 1
    build = json.loads(build_out.read_text(encoding="utf-8"))
    assert build["seed"] == hc.DEFAULT_SEED
    assert build["n_dev_target"] == 2 and build["n_test_target"] == 1
    assert build["out_dev_sha256"] == hc.sha256_file(out_dev)
    assert build["out_test_sha256"] == hc.sha256_file(out_test)
    assert build["n_sentences_seen"] == 3 * 10


def test_sha256_file_matches_hashlib(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    import hashlib

    assert hc.sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


# --------------------------------------------------------------------------------------
# --exclude-item-ids: translate a previous run's item_id into its cited_doi (neither a
# --veto-ids flag nor an item-id-keyed build-json field exists, so this is the smallest
# addition that lets a vetoed item's cited work be excluded from a rebuild).
# --------------------------------------------------------------------------------------


def _write_items_jsonl(path: Path, rows: list) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8",
    )


def test_load_item_id_cited_dois_finds_ids_across_dev_and_test_files(tmp_path: Path):
    dev = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    _write_items_jsonl(dev, [
        {"item_id": "real-dev-01", "cited_doi": "10.1/aaa"},
        {"item_id": "real-dev-02", "cited_doi": "10.1/bbb"},
    ])
    _write_items_jsonl(test, [{"item_id": "real-test-01", "cited_doi": "10.1/ccc"}])
    found = hc.load_item_id_cited_dois(
        ["real-dev-02", "real-test-01", "real-dev-99"], dev, test,
    )
    assert found == {"real-dev-02": "10.1/bbb", "real-test-01": "10.1/ccc"}
    assert "real-dev-99" not in found  # absent id is simply missing, not an error here


def test_load_item_id_cited_dois_skips_a_nonexistent_path(tmp_path: Path):
    dev = tmp_path / "dev.jsonl"
    _write_items_jsonl(dev, [{"item_id": "real-dev-01", "cited_doi": "10.1/aaa"}])
    missing_path = tmp_path / "does-not-exist.jsonl"
    found = hc.load_item_id_cited_dois(["real-dev-01"], dev, missing_path)
    assert found == {"real-dev-01": "10.1/aaa"}


def test_main_exclude_item_ids_removes_the_vetoed_cited_doi_from_a_skip_fetch_rebuild(
    tmp_path: Path,
):
    """A vetoed item from a *previous* run's output must not reappear after its cited_doi is
    passed via --exclude-item-ids, even though a fresh --skip-fetch rebuild reassigns every
    item_id from scratch. Targets --n-test 0 so every item lands in dev (test-first fill
    order would otherwise send all three single-item reviews to test at --n-test 3)."""
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    stats = {"n_sentences_seen": 10, "n_single_citation": 2, "n_resolved": 1, "n_oa": 1,
              "n_fetched": 1, "n_kept": 1}
    for i in range(3):
        review_doi = f"10.1/review{i}"
        cited_doi = f"10.1/cited{i}"
        _write_cited_cache(fulltext_dir, cited_doi, f"Cited work {i}")
        _write_review_cache(
            reviews_dir, review_doi, f"Review {i}", [_candidate(i, cited_doi)], stats,
        )
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    base_args = [
        "--skip-fetch", "--reviews-cache-dir", str(reviews_dir),
        "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
        "--out-test", str(out_test), "--build-out", str(build_out),
        "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
        "--hss-test-v3-sources", str(empty_sources),
        "--n-dev", "3", "--n-test", "0", "--seed", str(hc.DEFAULT_SEED),
    ]
    assert hc.main(base_args) == 0
    dev_rows = [json.loads(x) for x in out_dev.read_text(encoding="utf-8").splitlines()]
    assert len(dev_rows) == 3
    vetoed_id = dev_rows[0]["item_id"]
    vetoed_doi = dev_rows[0]["cited_doi"]

    rc = hc.main([*base_args, "--exclude-item-ids", vetoed_id])
    assert rc == 0
    dev_rows_2 = [json.loads(x) for x in out_dev.read_text(encoding="utf-8").splitlines()]
    assert len(dev_rows_2) == 2
    assert all(row["cited_doi"] != vetoed_doi for row in dev_rows_2)
    build = json.loads(build_out.read_text(encoding="utf-8"))
    assert vetoed_doi in build["excluded_dois"]


def test_main_exclude_item_ids_raises_when_the_id_is_not_found_anywhere(tmp_path: Path):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    _write_items_jsonl(out_dev, [])
    _write_items_jsonl(out_test, [])
    import pytest

    with pytest.raises(SystemExit):
        hc.main([
            "--skip-fetch", "--reviews-cache-dir", str(reviews_dir),
            "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
            "--out-test", str(out_test), "--build-out", str(build_out),
            "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
            "--hss-test-v3-sources", str(empty_sources),
            "--exclude-item-ids", "real-dev-does-not-exist",
        ])


# --------------------------------------------------------------------------------------
# resolve_citation's Crossref TITLE path must compare the
# citation's own year against a candidate's; without that check the similarity threshold (0.6)
# is low enough that a wrong-but-plausible same-subject paper by the same two authors is accepted.
# Reproduced from the exact shipped defect (real-test-06, 10.1016/j.compedu.2020.103819):
# the review cites "Teo, W.C., & Sathappan, R. (2018). The effectiveness of using Flipped
# Classroom Approach to teach adjectives to Malaysian Year 4 Chinese ESL learner.", but the
# Crossref title search's best (and only) hit was a DIFFERENT 2022 paper by the same two
# authors, "Using Flipped Classroom Approach to Teach Adjectives to Malaysian Year 4 Chinese
# Intermediate ESL Learners" (10.6007/ijarbss/v12-i10/15121), token-Jaccard 0.667 -- above the
# old 0.6 threshold and with no year check to catch the 4-year gap.
# --------------------------------------------------------------------------------------


def test_resolve_citation_rejects_the_real_test_06_wrong_year_wrong_paper_match(monkeypatch):
    """Must return None, not the wrong 2022 paper, once the
    year guard and the raised similarity threshold are both in place."""
    entry = {
        "doi": None,
        "year": "2018",
        "title": (
            "The effectiveness of using Flipped Classroom Approach to teach adjectives to "
            "Malaysian Year 4 Chinese ESL learner"
        ),
    }

    async def fake_title_lookup(title, mailto):
        return [{
            "doi": "10.6007/ijarbss/v12-i10/15121",
            "title": (
                "Using Flipped Classroom Approach to Teach Adjectives to Malaysian Year 4 "
                "Chinese Intermediate ESL Learners"
            ),
            "type": "journal-article",
            "authors": ["W. C. Teo", "R. Sathappan"],
            "year": "2022",
        }]

    monkeypatch.setattr(hc, "crossref_lookup_by_title", fake_title_lookup)
    assert asyncio.run(hc.resolve_citation(entry, mailto="x@y.z")) is None


def test_best_crossref_match_rejects_a_candidate_more_than_one_year_from_the_entry_year():
    """The year guard alone (independent of the threshold raise): a candidate whose Crossref
    ``issued`` year is more than one year away from the reference entry's own printed year is
    never picked, even at a similarity that would otherwise clear the threshold."""
    candidates = [
        {"doi": "10.1/wrong-year", "title": "Feedback and second language writing development",
         "year": "2022"},
    ]
    assert hc.best_crossref_match(
        "Feedback and second language writing development", candidates, entry_year="2018",
    ) is None


def test_best_crossref_match_accepts_a_candidate_within_one_year_of_the_entry_year():
    candidates = [
        {"doi": "10.1/right-year", "title": "Feedback and second language writing development",
         "year": "2019"},
    ]
    match = hc.best_crossref_match(
        "Feedback and second language writing development", candidates, entry_year="2018",
    )
    assert match is not None and match["doi"] == "10.1/right-year"


def test_best_crossref_match_prefers_the_year_matching_candidate_over_a_wrong_year_one():
    candidates = [
        {"doi": "10.1/wrong-year", "title": "Feedback and second language writing development",
         "year": "2022"},
        {"doi": "10.1/right-year", "title": "Feedback and second language writing development",
         "year": "2018"},
    ]
    match = hc.best_crossref_match(
        "Feedback and second language writing development", candidates, entry_year="2018",
    )
    assert match is not None and match["doi"] == "10.1/right-year"


def test_best_crossref_match_skips_the_year_guard_when_no_entry_year_is_given():
    """Backward compatible: existing callers that never pass ``entry_year`` keep matching on
    title similarity alone (unchanged behaviour)."""
    candidates = [{"doi": "10.1/x", "title": "Feedback and second language writing development",
                   "year": "2022"}]
    match = hc.best_crossref_match("Feedback and second language writing development", candidates)
    assert match is not None and match["doi"] == "10.1/x"


def test_crossref_search_to_candidates_carries_the_issued_year_through():
    payload = {"message": {"items": [
        {"DOI": "10.1/x", "title": ["A Title"], "type": "journal-article",
         "issued": {"date-parts": [[2022, 10, 1]]}},
        {"DOI": "10.1/y", "title": ["No Issued Date"], "type": "journal-article"},
    ]}}
    cands = hc.crossref_search_to_candidates(payload)
    assert cands[0]["year"] == "2022"
    assert cands[1]["year"] is None


def test_crossref_title_query_url_selects_author_and_issued():
    """The title-search path's Crossref ``select`` must include both ``author``
    (otherwise every title-resolved item structurally carries an empty author list) and
    ``issued`` (otherwise there is no year to guard on)."""
    url = hc.crossref_title_query_url("Some Title Here", "x@y.z")
    assert "select=" in url
    select = url.split("select=", 1)[1].split("&", 1)[0]
    fields = set(select.split(","))
    assert {"DOI", "title", "type", "author", "issued"} <= fields


# --------------------------------------------------------------------------------------
# The ASCII hyphen line-break repair must key on the real newline the PDF extractor left
# behind, not a literal space: only then does it fix a real PDF line break (hyphen immediately
# followed by a newline) without wrongly joining an APA suspended hyphenation
# ("pre- and post-implementation") into a non-word ("preand post-implementation").
# --------------------------------------------------------------------------------------


def test_clean_extracted_text_leaves_suspended_hyphenation_before_and_unchanged():
    """Red-first: the real real-test-14 substring (Jaiswal review, methodology section) --
    a genuine APA suspended hyphenation on ONE physical line (hyphen-SPACE, no line break at
    all) -- must survive completely unchanged."""
    real_substring = (
        "involved a measure of student pass/fail rates pre- and post-implementation. Another"
    )
    cleaned = hc.clean_extracted_text(real_substring)
    assert cleaned == real_substring
    assert "preand" not in cleaned


def test_clean_extracted_text_repairs_a_real_ascii_hyphen_newline_break():
    """Red-first: a genuine PDF line-break-hyphenated word (hyphen immediately followed by a
    newline, no space) -- the real substring from 10.1007/s00787-022-02012-8 -- must be
    joined into one word, which the old space-keyed regex never touched."""
    real_substring = (
        "Alongside increased owner-\nship rates, multifunctionality has expanded"
    )
    cleaned = hc.clean_extracted_text(real_substring)
    assert "ownership rates" in cleaned
    assert "owner-\nship" not in cleaned and "owner-ship" not in cleaned


def test_clean_extracted_text_does_not_join_a_hyphen_newline_break_before_and_or_to_but_nor():
    """Belt-and-braces: even when the hyphen IS immediately followed by a newline, the repair
    must still refuse to join it into the following word when that word is one of the
    suspended-hyphenation connectors (and/or/to/but/nor)."""
    dirty = "the study measured pre-\nand post-treatment outcomes across every cohort."
    cleaned = hc.clean_extracted_text(dirty)
    assert "preand" not in cleaned
    assert "pre-" in cleaned


# --------------------------------------------------------------------------------------
# The citation-safe splitter must not cut a sentence between a
# page-locator abbreviation ("(p." / "(pp.") and its digit, since the boundary regex's
# lookahead class includes digits and "(p."/"(pp." is not one of the never-split
# abbreviations otherwise. Reproduced pipeline-level from the exact real-test-09 paragraph
# (10.1007/s10763-026-10681-z, introduction section).
# --------------------------------------------------------------------------------------

_REAL_TEST_09_PARAGRAPH = (
    "To distinguish FA-T from \nthe overarching process of teaching and learning, Black and "
    "Wiliam (2009) indicate \nthat “formative assessment is concerned with the creation "
    "of, and capitalization upon, \n‘moments of contingency’ in instruction for the "
    "purpose of the regulation of learn­\ning processes” (p. 10). Such moments of "
    "contingency may occur during classroom \nteaching; for instance, when a teacher makes "
    "on-the-spot contingent decisions about \nhow to respond to unexpected students’ "
    "mathematical ideas during a class discussion that continues for the rest of the period."
)


def test_split_sentences_real_does_not_truncate_before_a_page_locator():
    """Red-first: the real real-test-09 paragraph must split into
    exactly two sentences, the first ending in the full '(p. 10).' locator, not a bare
    '(p.' fragment."""
    cleaned = hc.clean_extracted_text(_REAL_TEST_09_PARAGRAPH)
    sentences = hc.split_sentences_real(cleaned)
    assert len(sentences) == 2, sentences
    assert sentences[0].endswith("(p. 10).")
    assert not sentences[0].endswith("(p.")
    assert sentences[1].startswith("Such moments of contingency")
    citation = hc.find_single_citation(sentences[0])
    assert citation is not None
    assert citation.surname == "Black" and citation.surname2 == "Wiliam" and citation.year == "2009"


def test_find_single_citation_rejects_a_fragment_ending_in_a_page_locator_abbreviation():
    """Belt-and-braces: even a candidate that somehow still ends in '(p.'/'(pp.' (a splitting
    defect elsewhere) is never accepted as a complete claim."""
    truncated = (
        "Black and Wiliam (2009) indicate that formative assessment concerns the creation of "
        "moments of contingency in instruction for the purpose of regulation (p."
    )
    assert hc.find_single_citation(truncated) is None
    truncated_pp = truncated[:-4] + "(pp."
    assert hc.find_single_citation(truncated_pp) is None


# --------------------------------------------------------------------------------------
# A section heading glued (via the extraction's own line
# break, not a sentence boundary) to the start of the next sentence must not ship as part of the
# claim. Fixed structurally (a short, unpunctuated physical line followed by a
# capital-starting line is dropped), not by extending a fixed heading-word vocabulary.
# Reproduced verbatim (with real newlines) from all four shipped defects.
# --------------------------------------------------------------------------------------


def test_drop_heading_lines_guard_is_removed():
    """The physical-line heading guard once deleted 19.5 percent of
    body lines corpus-wide (16,356 of them mid-sentence) and fabricated claim text by gluing
    two non-adjacent wrapped fragments together whenever an ordinary body line happened to be
    short, unpunctuated and followed by a line starting with a capital (a proper noun, not a
    heading) -- see the regression test below for the exact shipped example. The guard is
    removed outright rather than narrowed: this module's regex-extraction path no longer
    feeds the shipped build (build_review_cache_from_readings does, see the module
    docstring), so the same-line heading vocabulary (HEADING_RE, inside
    find_single_citation) is the only heading guard left, and it is safe because it only ever
    drops a heading fused to the START of the very sentence being evaluated, never a
    physically separate body line two lines away."""
    assert not hasattr(hc, "_drop_heading_lines")
    assert not hasattr(hc, "_looks_like_heading_line")


NEUMANN_MERCHANT_REFERENCES = (
    "References\n"
    "Neumann, M. M., & Merchant, S. (2022). Digital storytelling apps in early childhood "
    "classrooms. Journal of Digital Learning, 5, 1-9. https://doi.org/10.1234/neumann.2022\n"
)


def test_extract_candidates_does_not_fabricate_text_from_a_hard_wrapped_paragraph(
    tmp_path: Path, monkeypatch,
):
    """The heading guard's exact shipped fabrication, replayed against the real
    review 10.1007/s10643-024-01804-8's source lines: with the heading guard gone, this
    four-line hard-wrapped paragraph must survive whole -- the accepted claim must be a
    substring of the whitespace-normalised source chunk, not a splice that skips the third
    line just because the fourth happens to start with a capitalised proper noun ("Three")."""
    body = (
        "However, in contrast to these two studies, Neumann and Merchant (2022) \n"
        "used a naturalistic observational approach to examine how \n"
        "early childhood teachers use digital storytelling apps (e.g., \n"
        "Three Little Pigs and Peppa Pig's Party) for shared reading with young children."
    )
    payload = {
        "doi": "10.1/review0", "title": "A Review",
        "chunks": [
            {"section": "body", "text": body},
            {"section": "references", "text": NEUMANN_MERCHANT_REFERENCES},
        ],
    }

    async def fake_crossref_doi(doi, mailto):
        return None  # no title comparison possible -- printed doi accepted as given

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return (
            {"doi": work["doi"]},
            {"doi": work["doi"], "title": work["title"], "authors": [],
             "source_url": "https://x/y.pdf", "fetched_at": "2026-09-07T00:00:00+00:00",
             "n_chars": 20000, "chunks": [{"section": "results", "text": "It helped."}]},
            None,
        )

    async def fake_openalex_work(doi, mailto):
        return {"authorships": []}

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert len(kept) == 1
    claim = kept[0]["sentence"]
    normalised_chunk = " ".join(hc.clean_extracted_text(body).split())
    assert claim in normalised_chunk, (claim, normalised_chunk)
    assert "early childhood teachers use digital storytelling apps" in claim


def test_list_item_re_matches_multi_level_numbered_headings():
    """LIST_ITEM_RE must recognise '4.3 ' (a period followed by another digit, not
    whitespace), not only a single-level number."""
    assert hc.LIST_ITEM_RE.match("4.3 General characteristics of the sample")
    assert hc.LIST_ITEM_RE.match("5.1. Provision and programme design")
    assert hc.LIST_ITEM_RE.match("1. A plain single-level list item")
    assert hc.LIST_ITEM_RE.match("(a) a lettered list item")


# --------------------------------------------------------------------------------------
# REVIEW_OWN_WORK_RE must match the passive voice and common
# methods verbs, not only first-person-plural active voice. Reproduced verbatim from the two
# shipped defects (real-test-01, real-test-05).
# --------------------------------------------------------------------------------------


def test_find_single_citation_rejects_the_real_test_01_passive_methods_sentence():
    s = (
        "The methodological quality of the included studies was assessed using the "
        "eight-item tool developed by Nudelman and Otto (2020), which is specifically "
        "tailored for use with non-experimental, observational research studies overall."
    )
    assert hc.find_single_citation(s) is None


def test_find_single_citation_rejects_the_real_test_05_we_applied_sentence():
    s = (
        "As one example, we applied a slightly modified version of an established category "
        "system by Cevikbas et al. (2022) to describe every study included in the review."
    )
    assert hc.find_single_citation(s) is None


# --------------------------------------------------------------------------------------
# crossref_title_query_url's ``author`` select field is
# covered above; this covers the second half: a cached cited work with an empty author list
# is enriched (OpenAlex, then the resolved Crossref fallback) and the cache file rewritten,
# instead of being frozen empty forever.
# --------------------------------------------------------------------------------------


def test_extract_candidates_enriches_a_cached_cited_work_that_has_empty_authors(
    tmp_path: Path, monkeypatch,
):
    sentence = (
        "Smith (2019) found that structured feedback improved learner accuracy across an "
        "entire academic term for every cohort studied in the sample overall."
    )
    payload = _review_payload_with_one_citation(sentence)

    async def fake_crossref_doi(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing development",
            "type": "journal-article",
        }

    calls = {"openalex_authors": 0}

    async def fake_openalex_work(doi, mailto):
        calls["openalex_authors"] += 1
        return {"authorships": [{"author": {"display_name": "J. A. Smith"}}]}

    async def fake_acquire_one_never_called(*a, **kw):
        raise AssertionError("must not re-fetch an already-cached cited work")

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "openalex_work_by_doi", fake_openalex_work)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one_never_called)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    cited_doi = "10.1016/j.jslw.2019.03.002"
    cache_path = fulltext_dir / f"{hc.doi_slug(cited_doi)}.json"
    write_json(cache_path, {
        "doi": cited_doi, "title": "Feedback and second language writing development",
        "authors": [], "source_url": "https://x/y.pdf",
        "fetched_at": "2026-09-06T00:00:00+00:00", "n_chars": 20000,
        "chunks": [{"section": "results", "text": "It helped."}],
    })
    kept, stats = asyncio.run(hc.extract_candidates(
        payload, mailto="x@y.z", pipeline={}, excluded=set(), fulltext_cache_dir=fulltext_dir,
        min_chars=1, min_chunks=1, review_doi="10.1/review0", max_per_review=4,
    ))
    assert len(kept) == 1 and stats["n_kept"] == 1
    assert calls["openalex_authors"] == 1
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["authors"] == ["J. A. Smith"]


# --------------------------------------------------------------------------------------
# --from-readings builds review caches from a reader-extracted-citations readings jsonl
# file (one JSON object per line: review_doi, review_title, claim,
# citation_text, reference_entry, cited_title, cited_year, cited_authors, cited_doi,
# context_before, reader_notes) instead of the regex-extraction discovery pipeline. Every
# review-cache payload it writes has the SAME shape extract_candidates produces (doi/title/
# candidates/stats), so build_outputs/build_from_cache need no readings-specific code.
# --------------------------------------------------------------------------------------


def _reading_row(**overrides) -> dict:
    row = {
        "review_doi": "10.1/review0",
        "review_title": "A Review of Feedback Studies",
        "claim": (
            "Smith (2019) found that structured feedback improved learner accuracy across "
            "an entire academic term for every cohort studied in the sample overall."
        ),
        "citation_text": "Smith (2019)",
        "reference_entry": (
            "Smith, J. A. (2019). Feedback and second language writing development. "
            "Journal of Second Language Writing, 45, 12-29."
        ),
        "cited_title": "Feedback and second language writing development",
        "cited_year": "2019",
        "cited_authors": ["J. A. Smith"],
        "cited_doi": None,
        "context_before": "Prior sentence here.",
        "reader_notes": "",
    }
    row.update(overrides)
    return row


# ---- group_readings_by_review -------------------------------------------------------


def test_group_readings_by_review_groups_rows_and_drops_rows_without_a_review_doi():
    rows = [
        _reading_row(review_doi="10.1/a"),
        _reading_row(review_doi="10.1/a"),
        _reading_row(review_doi="10.1/b"),
        _reading_row(review_doi=None),
        _reading_row(review_doi=""),
    ]
    grouped = hc.group_readings_by_review(rows)
    assert set(grouped) == {"10.1/a", "10.1/b"}
    assert len(grouped["10.1/a"]) == 2
    assert len(grouped["10.1/b"]) == 1


# ---- validate_reading_row -------------------------------------------------------------


def test_validate_reading_row_accepts_a_well_formed_row():
    assert hc.validate_reading_row(_reading_row()) == []


def test_validate_reading_row_rejects_a_claim_under_the_word_floor():
    row = _reading_row(claim="Smith (2019) found an effect.")
    errors = hc.validate_reading_row(row)
    assert any("word" in e for e in errors)


def test_validate_reading_row_rejects_a_claim_over_the_word_ceiling():
    long_claim = (
        "Smith (2019) found that " + ("structured feedback improved accuracy " * 15) + "overall."
    )
    row = _reading_row(claim=long_claim)
    errors = hc.validate_reading_row(row)
    assert any("word" in e for e in errors)


def test_validate_reading_row_rejects_a_claim_with_two_citations():
    row = _reading_row(
        claim=(
            "Smith (2019) found that feedback improved accuracy while Jones (2020) reported "
            "similar gains across an entire academic term for every cohort studied overall."
        ),
    )
    errors = hc.validate_reading_row(row)
    assert any("citation" in e for e in errors)


def test_validate_reading_row_rejects_a_claim_with_no_recognised_citation():
    row = _reading_row(
        claim=(
            "Structured feedback improved learner accuracy across an entire academic term "
            "for every cohort studied in the sample overall without any citation at all here."
        ),
    )
    errors = hc.validate_reading_row(row)
    assert any("citation" in e for e in errors)


def test_validate_reading_row_rejects_a_missing_reference_entry():
    row = _reading_row(reference_entry="")
    errors = hc.validate_reading_row(row)
    assert any("reference_entry" in e for e in errors)


# ---- The claim's OWN in-text citation must match what the
# reader recorded as its reference entry/cited work -- a live CLI run paired a claim about
# "Kalyuga (2011)" with a reference entry/cited_year/cited_authors for Paas and Van
# Merrienboer (1994) and nothing caught it. ----------------------------------------------------


def _mismatched_kalyuga_row() -> dict:
    """The exact live-run defect: the claim's own citation (Kalyuga, 2011) has nothing to do
    with the reference entry/cited work the reader recorded (Paas & Van Merrienboer, 1994)."""
    return _reading_row(
        claim=(
            "This mirrors a pattern Kalyuga (2011) also observed across several distinct "
            "experimental training conditions previously studied here."
        ),
        citation_text="Kalyuga (2011)",
        reference_entry=(
            "Paas, F., & Van Merrienboer, J. J. G. (1994). Instructional control of cognitive "
            "load in the training of complex cognitive tasks. Educational Psychology Review, "
            "6(4), 351-371."
        ),
        cited_title=(
            "Instructional control of cognitive load in the training of complex cognitive tasks"
        ),
        cited_year="1994",
        cited_authors=["F. Paas", "J. J. G. Van Merrienboer"],
        cited_doi="10.1007/bf02213420",
    )


def _matching_paas_row() -> dict:
    """Same reference entry/cited work as above, but the claim's own citation now actually
    names Paas and Van Merrienboer (1994) -- the correctly-paired version of the same row."""
    return _reading_row(
        claim=(
            "This mirrors a pattern that Paas and Van Merrienboer (1994) also observed across "
            "several distinct experimental training conditions previously studied."
        ),
        citation_text="Paas and Van Merrienboer (1994)",
        reference_entry=(
            "Paas, F., & Van Merrienboer, J. J. G. (1994). Instructional control of cognitive "
            "load in the training of complex cognitive tasks. Educational Psychology Review, "
            "6(4), 351-371."
        ),
        cited_title=(
            "Instructional control of cognitive load in the training of complex cognitive tasks"
        ),
        cited_year="1994",
        cited_authors=["F. Paas", "J. J. G. Van Merrienboer"],
        cited_doi="10.1007/bf02213420",
    )


def test_validate_reading_row_rejects_a_claim_whose_own_citation_mismatches_the_row():
    errors = hc.validate_reading_row(_mismatched_kalyuga_row())
    assert errors  # non-empty: the year and the surname both disagree with the reader's own row
    assert any("year" in e for e in errors)
    assert any("surname" in e for e in errors)


def test_validate_reading_row_accepts_a_claim_whose_own_citation_matches_the_row():
    assert hc.validate_reading_row(_matching_paas_row()) == []


# ---- find_single_citation is a SELECTION filter for
# regex-extracted sentences (word count, PDF-artefact guards, REVIEW_OWN_WORK_RE, heading/
# list-item shape, terminal punctuation...), not an integrity check -- it returns None for any
# of those reasons even when the claim carries a perfectly identifiable citation, which used to
# silently disable the entire cross-check above. --------------------------------------------


def test_validate_reading_row_rejects_mismatch_despite_hyphenation_blocking_find_single_citation():
    """Red-first: an everyday suspended hyphenation ('pre- and post-test') trips build_hss_set's
    HYPHEN_BREAK_RE ([A-Za-z]- [A-Za-z], matching 'e- a'), so find_single_citation(claim) is
    None even though the claim carries exactly one recognised citation. validate_reading_row
    must not route its cross-check through find_single_citation, since doing so would
    report no error at all for this exact mis-pairing (Kalyuga (2011) claim paired
    with a Paas and Van Merrienboer (1994) reference entry/cited_year/cited_authors)."""
    claim = (
        "This mirrors a pattern Kalyuga (2011) also observed across pre- and post-test "
        "scores in several distinct training conditions previously studied here."
    )
    assert hc.find_single_citation(claim) is None  # confirms the artefact really blocks it
    row = _mismatched_kalyuga_row()
    row["claim"] = claim
    errors = hc.validate_reading_row(row)
    assert any("year" in e for e in errors)
    assert any("surname" in e for e in errors)


def test_validate_reading_row_rejects_mismatch_despite_adjacent_numbers_artefact():
    """A second independently-reproduced find_single_citation-blocking artefact
    (ADJACENT_NUMBERS_RE, 'ranged from 0.21 0.69') must not disable the cross-check either."""
    claim = (
        "This mirrors a pattern Kalyuga (2011) also observed, with effect sizes that ranged "
        "from 0.21 0.69 across several distinct training conditions studied previously."
    )
    assert hc.find_single_citation(claim) is None
    row = _mismatched_kalyuga_row()
    row["claim"] = claim
    errors = hc.validate_reading_row(row)
    assert any("year" in e for e in errors)
    assert any("surname" in e for e in errors)


def test_validate_reading_row_rejects_mismatch_despite_review_own_work_artefact():
    """A third independently-reproduced find_single_citation-blocking filter
    (REVIEW_OWN_WORK_RE, 'these data were coded using an established taxonomy') must not disable
    the cross-check either."""
    claim = (
        "This mirrors a pattern Kalyuga (2011) also observed; these data were coded using an "
        "established taxonomy across several distinct training conditions studied previously."
    )
    assert hc.find_single_citation(claim) is None
    row = _mismatched_kalyuga_row()
    row["claim"] = claim
    errors = hc.validate_reading_row(row)
    assert any("year" in e for e in errors)
    assert any("surname" in e for e in errors)


# ---- The surname check must not accept a bare substring
# match -- a short surname (Li, He, Wu, Xu, Yu, Ma, Ng, An, Lee: dense in this corpus) is
# otherwise a substring of an unrelated word in the reference entry's own title/journal name. --


def test_validate_reading_row_rejects_a_surname_matched_only_as_a_substring_of_another_word():
    """Red-first: normalize_surname('Li') == 'li' is a bare substring of 'applied linguistics',
    so the pre-fix raw-containment check silently accepted a claim about Li (2019) paired with
    an unrelated Smith (2019) reference entry of the same year -- the only remaining identity
    guard once major finding 1 above is fixed, and same-year is the common mis-pairing case."""
    row = _reading_row(
        claim=(
            "This builds on the classroom studies that Li (2019) reviewed across several "
            "distinct instructional settings and learner populations studied previously here."
        ),
        citation_text="Li (2019)",
        reference_entry=(
            "Smith, J. (2019). Feedback and second language writing. Applied Linguistics, "
            "40(2), 1-20."
        ),
        cited_title="Feedback and second language writing",
        cited_year="2019",
        cited_authors=["J. Smith"],
        cited_doi=None,
    )
    errors = hc.validate_reading_row(row)
    assert any("surname" in e for e in errors)


def test_validate_reading_row_accepts_a_second_surname_found_only_in_cited_authors():
    """The second-surname branch still passes on a well-formed two-author row (regression guard
    for the word-boundary rewrite of that branch): _matching_paas_row's own second surname,
    'Van Merrienboer', is present in both reference_entry and cited_authors."""
    assert hc.validate_reading_row(_matching_paas_row()) == []


def test_validate_reading_row_falls_back_to_word_boundary_containment_when_entry_does_not_parse():
    """When reference_entry does not even parse (no leading 'Surname, I.' plus a parenthesised
    year -- e.g. a wrapped/garbled entry), the surname check falls back to word-boundary
    containment over reference_entry + cited_authors rather than exact parsed-surname
    comparison, and still rejects an unrelated short surname matched only as a substring."""
    row = _reading_row(
        claim=(
            "This builds on the classroom studies that Li (2019) reviewed across several "
            "distinct instructional settings and learner populations studied previously here."
        ),
        citation_text="Li (2019)",
        reference_entry="a garbled fragment with no leading surname or year at all here",
        cited_title="Feedback and second language writing",
        cited_year="2019",
        cited_authors=["J. Smith"],
        cited_doi=None,
    )
    assert hc.parse_reference_entry(row["reference_entry"]) is None  # confirms the fallback path
    errors = hc.validate_reading_row(row)
    assert any("surname" in e for e in errors)


# ---- resolve_reading_citation: printed-DOI sanity-checks at 0.5, title-search at 0.6, both
# looser than resolve_citation's own regex-extraction-path thresholds (0.8 title-search, no
# sanity check at all on its printed-DOI path) -- a reader already verified the reference-list
# entry itself, so the DOI needs only a sanity check, not the regex path's full re-derivation.
# --------------------------------------------------------------------------------------


def test_resolve_reading_citation_accepts_a_printed_doi_when_titles_loosely_agree(monkeypatch):
    async def fake_lookup(doi, mailto):
        return {
            "doi": doi, "title": "Feedback and second language writing progress",
            "type": "journal-article",
        }

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    row = _reading_row(cited_doi="10.1016/j.jslw.2019.03.002")
    result = asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z"))
    assert result is not None
    assert result["doi"] == "10.1016/j.jslw.2019.03.002"
    assert result["resolution"] == "printed_doi"
    # below the regex-extraction path's own 0.8 threshold, at or above the readings path's 0.5
    assert hc.READINGS_DOI_SANITY_THRESHOLD <= result["similarity"] < 0.8


def test_resolve_reading_citation_rejects_a_printed_doi_below_the_sanity_threshold(monkeypatch):
    async def fake_lookup(doi, mailto):
        return {
            "doi": doi, "title": "An unrelated paper about lexicography tooling",
            "type": "journal-article",
        }

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    row = _reading_row(cited_doi="10.1/x")
    assert asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z")) is None


def test_resolve_reading_citation_accepts_a_printed_doi_when_no_title_comparison_is_possible(
    monkeypatch,
):
    """A Crossref lookup failure leaves no title to compare against, so the printed cited_doi
    is accepted as given -- the same fallback resolve_citation applies on its own path."""
    async def fake_lookup(doi, mailto):
        return None

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_lookup)
    row = _reading_row(cited_doi="10.1/x")
    result = asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z"))
    assert result is not None
    assert result["similarity"] is None and result["title"] == row["cited_title"]


def test_resolve_reading_citation_resolves_by_title_search_when_no_doi_is_printed(monkeypatch):
    async def fake_title_lookup(title, mailto):
        return [{"doi": "10.1/y", "title": title, "type": "journal-article", "year": "2019"}]

    monkeypatch.setattr(hc, "crossref_lookup_by_title", fake_title_lookup)
    row = _reading_row(cited_doi=None)
    result = asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z"))
    assert result is not None
    assert result["resolution"] == "crossref_title" and result["doi"] == "10.1/y"


def test_resolve_reading_citation_rejects_a_title_search_below_its_own_threshold(monkeypatch):
    async def fake_title_lookup(title, mailto):
        return [{
            "doi": "10.1/y", "title": "A wholly different paper about phonology acquisition",
            "type": "journal-article", "year": "2019",
        }]

    monkeypatch.setattr(hc, "crossref_lookup_by_title", fake_title_lookup)
    row = _reading_row(cited_doi=None)
    assert asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z")) is None


def test_resolve_reading_citation_returns_none_without_a_doi_or_title():
    row = _reading_row(cited_doi=None, cited_title=None)
    assert asyncio.run(hc.resolve_reading_citation(row, mailto="x@y.z")) is None


# ---- build_review_cache_from_readings: validation, resolution, OA+fetch+identity-check and
# exclusions, all producing extract_candidates' own cache/stats shape. ---------------------


def _fake_reading_acquire_one_ok(work, pipeline, *, email, min_chars, min_chunks):
    # build_outputs now runs check_cache_identity over the
    # fulltext cache, so the fetched text must actually describe the work's own title (and, for
    # the tests below, the default _reading_row()'s cited_authors surname "Smith") or a
    # downstream build_outputs call would drop it as a wrong-work mis-fetch.
    return (
        {"doi": work["doi"]},
        {"doi": work["doi"], "title": work["title"], "authors": [],
         "source_url": "https://x/y.pdf", "fetched_at": "2026-09-07T00:00:00+00:00",
         "n_chars": 20000,
         "chunks": [{"section": "results",
                     "text": f"{work['title']}. Smith reported that it helped a lot."}]},
        None,
    )


def test_build_review_cache_from_readings_builds_one_kept_item_and_funnel_counts(
    tmp_path: Path, monkeypatch,
):
    async def fake_crossref_doi(doi, mailto):
        return None  # no title comparison possible -- printed doi accepted as given

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return _fake_reading_acquire_one_ok(
            work, pipeline, email=email, min_chars=min_chars, min_chunks=min_chunks,
        )

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    row = _reading_row(cited_doi="10.1016/j.jslw.2019.03.002")
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert len(payload["candidates"]) == 1
    assert payload["doi"] == "10.1/review0" and payload["title"] == "A Review of Feedback Studies"
    assert payload["stats"] == {
        "n_sentences_seen": 1, "n_single_citation": 1, "n_resolved": 1, "n_oa": 1,
        "n_fetched": 1, "n_kept": 1,
    }
    cached = json.loads(
        (fulltext_dir / f"{hc.doi_slug('10.1016/j.jslw.2019.03.002')}.json")
        .read_text(encoding="utf-8")
    )
    assert cached["authors"] == ["J. A. Smith"]


def test_build_review_cache_from_readings_skips_an_invalid_row(tmp_path: Path):
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    row = _reading_row(reference_entry="")  # invalid: no reference entry to resolve from
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert payload["candidates"] == []
    assert payload["stats"]["n_sentences_seen"] == 1
    assert payload["stats"]["n_single_citation"] == 0


def test_build_review_cache_from_readings_drops_a_claim_citation_reference_entry_mismatch(
    tmp_path: Path, monkeypatch,
):
    """the Kalyuga/Paas mismatch never reaches
    resolution at all (validate_reading_row rejects it before n_single_citation is even
    counted), while the correctly-paired version of the same row is still kept."""
    async def fake_crossref_doi(doi, mailto):
        return None

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return _fake_reading_acquire_one_ok(
            work, pipeline, email=email, min_chars=min_chars, min_chunks=min_chunks,
        )

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    mismatched = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [_mismatched_kalyuga_row()], mailto="x@y.z", pipeline={},
        excluded=set(), fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1,
        max_per_review=4,
    ))
    assert mismatched["candidates"] == []
    assert mismatched["stats"]["n_single_citation"] == 0

    matching = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review1", [_matching_paas_row()], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert len(matching["candidates"]) == 1
    assert matching["stats"]["n_kept"] == 1


def test_build_review_cache_from_readings_drops_an_identity_check_rejection(
    tmp_path: Path, monkeypatch,
):
    """The wrong-work identity check (title + first author against the fetched PDF text,
    inside acquire_one, reused unchanged here) rejects a mismatched fetch: the candidate must
    not be kept, and nothing is written to the fulltext cache."""
    async def fake_crossref_doi(doi, mailto):
        return None

    async def fake_acquire_one_wrong_work(work, pipeline, *, email, min_chars, min_chunks):
        return None, None, "wrong_work: some unrelated fetched text fragment"

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one_wrong_work)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    row = _reading_row(cited_doi="10.1016/j.jslw.2019.03.002")
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert payload["candidates"] == []
    stats = payload["stats"]
    assert stats["n_resolved"] == 1
    assert stats["n_oa"] == 1 and stats["n_fetched"] == 0 and stats["n_kept"] == 0
    assert not (fulltext_dir / f"{hc.doi_slug('10.1016/j.jslw.2019.03.002')}.json").exists()


def test_build_review_cache_from_readings_drops_an_excluded_cited_doi(
    tmp_path: Path, monkeypatch,
):
    async def fake_crossref_doi(doi, mailto):
        return None

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    row = _reading_row(cited_doi="10.1/excluded")
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded={"10.1/excluded"},
        fulltext_cache_dir=tmp_path, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert payload["candidates"] == [] and payload["stats"]["n_resolved"] == 0


def test_build_review_cache_from_readings_drops_a_self_citation(tmp_path: Path, monkeypatch):
    async def fake_crossref_doi(doi, mailto):
        return None

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    row = _reading_row(cited_doi="10.1/review0")
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=tmp_path, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert payload["candidates"] == []


def test_build_review_cache_from_readings_reuses_a_cached_cited_work_without_refetch(
    tmp_path: Path, monkeypatch,
):
    async def fake_crossref_doi(doi, mailto):
        return None

    async def fake_acquire_one_must_not_be_called(*a, **kw):
        raise AssertionError("must not re-fetch an already-cached cited work")

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one_must_not_be_called)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    cited_doi = "10.1016/j.jslw.2019.03.002"
    write_json(fulltext_dir / f"{hc.doi_slug(cited_doi)}.json", {
        "doi": cited_doi, "title": "Feedback and second language writing development",
        "authors": ["Pre-existing Author"], "source_url": "https://x/y.pdf",
        "fetched_at": "2026-09-06T00:00:00+00:00", "n_chars": 20000,
        # The was_cached branch re-checks identity
        # against the row's own cited_authors ("J. A. Smith" for the default _reading_row()),
        # so the text must contain the title's own tokens and the surname "Smith".
        "chunks": [{"section": "results",
                    "text": "Feedback and second language writing development. "
                    "Smith found this to be true."}],
    })
    row = _reading_row(cited_doi=cited_doi)
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert len(payload["candidates"]) == 1 and payload["stats"]["n_kept"] == 1
    cached = json.loads(
        (fulltext_dir / f"{hc.doi_slug(cited_doi)}.json").read_text(encoding="utf-8")
    )
    assert cached["authors"] == ["Pre-existing Author"]  # not overwritten, already non-empty


def test_build_review_cache_from_readings_rejects_an_already_cached_wrong_work(
    tmp_path: Path, monkeypatch,
):
    """The was_cached branch must not accept a pre-existing
    fulltext cache entry on the strength of its filename alone. Here the cache entry's own
    recorded title/DOI belong to one work (planted for a DOI that resolves to Paas & Van
    Merrienboer, 1994) while its chunk text is unmistakably a different work's -- the exact
    live-run defect (a cache entry carrying one work's metadata but another's chunks). The
    candidate must not be kept, and the mis-fetched cache entry itself must be left untouched
    (never rewritten, never deleted -- this check only decides whether to REUSE it here)."""
    async def fake_crossref_doi(doi, mailto):
        return None

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    cited_doi = "10.1007/bf02213420"
    cached_path = fulltext_dir / f"{hc.doi_slug(cited_doi)}.json"
    write_json(cached_path, {
        "doi": cited_doi,
        "title": (
            "Instructional control of cognitive load in the training of complex cognitive "
            "tasks"
        ),
        "authors": ["F. Paas", "J. J. G. Van Merrienboer"], "source_url": "https://x/y.pdf",
        "fetched_at": "2026-09-06T00:00:00+00:00", "n_chars": 20000,
        # An unrelated work's text entirely -- no overlap with the title/authors above.
        "chunks": [{"section": "results", "text": "Verhoeven reported unrelated findings here."}],
    })
    row = _matching_paas_row()
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", [row], mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=4,
    ))
    assert payload["candidates"] == []
    stats = payload["stats"]
    assert stats["n_resolved"] == 1
    assert stats["n_oa"] == 1 and stats["n_fetched"] == 0 and stats["n_kept"] == 0
    unchanged = json.loads(cached_path.read_text(encoding="utf-8"))
    assert unchanged["authors"] == ["F. Paas", "J. J. G. Van Merrienboer"]  # untouched


def test_build_review_cache_from_readings_caps_at_max_per_review_and_dedupes_cited_doi(
    tmp_path: Path, monkeypatch,
):
    async def fake_crossref_doi(doi, mailto):
        return None

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return _fake_reading_acquire_one_ok(
            work, pipeline, email=email, min_chars=min_chars, min_chunks=min_chunks,
        )

    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    fulltext_dir = tmp_path / "fulltext"
    fulltext_dir.mkdir()
    rows = [
        _reading_row(cited_doi="10.1/dup"),
        _reading_row(cited_doi="10.1/dup"),  # duplicate cited work, same review -- skipped
        _reading_row(cited_doi="10.1/second"),
        _reading_row(cited_doi="10.1/third"),  # never attempted, cap is 2
    ]
    payload = asyncio.run(hc.build_review_cache_from_readings(
        "10.1/review0", rows, mailto="x@y.z", pipeline={}, excluded=set(),
        fulltext_cache_dir=fulltext_dir, min_chars=1, min_chunks=1, max_per_review=2,
    ))
    cited_dois = [c["cited_doi"] for c in payload["candidates"]]
    assert cited_dois == ["10.1/dup", "10.1/second"]


# ---- fetch_from_readings + build_outputs: end-to-end (still monkeypatched network) proof
# that the readings path feeds the same test-first-then-dev split, per-review cap and DOI
# exclusions as the regex-extraction path, because it writes the identical review-cache shape.
# --------------------------------------------------------------------------------------


def test_fetch_from_readings_then_build_outputs_splits_test_first_and_applies_exclusions(
    tmp_path: Path, monkeypatch,
):
    async def fake_crossref_doi(doi, mailto):
        return None

    async def fake_acquire_one(work, pipeline, *, email, min_chars, min_chunks):
        return _fake_reading_acquire_one_ok(
            work, pipeline, email=email, min_chars=min_chars, min_chunks=min_chunks,
        )

    monkeypatch.setattr(hc, "load_pipeline", lambda: {})
    monkeypatch.setattr(hc, "crossref_lookup_by_doi", fake_crossref_doi)
    monkeypatch.setattr(hc, "acquire_one", fake_acquire_one)
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    readings_path = tmp_path / "readings.jsonl"
    rows = [
        _reading_row(review_doi=f"10.1/review{i}", review_title=f"Review {i}",
                     cited_doi=f"10.1/cited{i}")
        for i in range(4)
    ]
    rows.append(_reading_row(
        review_doi="10.1/review4", review_title="Review 4", cited_doi="10.1/excluded",
    ))
    readings_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    n = asyncio.run(hc.fetch_from_readings(
        readings_path=readings_path, mailto="x@y.z", excluded={"10.1/excluded"},
        reviews_cache_dir=reviews_dir, fulltext_cache_dir=fulltext_dir, min_chars=1,
        min_chunks=1, max_per_review=4,
    ))
    assert n == 5  # five distinct review_doi values in the readings file
    dev, test, info = hc.build_outputs(
        reviews_dir, fulltext_dir, seed=hc.DEFAULT_SEED, n_dev=2, n_test=2, max_per_review=4,
        excluded={"10.1/excluded"},
    )
    all_items = dev + test
    assert len(test) == 2 and len(dev) == 2  # test-first, both targets met from 4 good reviews
    assert "10.1/excluded" not in {row["cited_doi"] for row in all_items}
    dev_reviews = {row["review_doi"] for row in dev}
    test_reviews = {row["review_doi"] for row in test}
    assert not dev_reviews & test_reviews


# ---- CLI: --from-readings ------------------------------------------------------------


def test_main_from_readings_calls_fetch_from_readings_and_builds_outputs(
    tmp_path: Path, monkeypatch,
):
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    readings_path = tmp_path / "readings.jsonl"
    readings_path.write_text("", encoding="utf-8")
    calls = {}

    async def fake_fetch_from_readings(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(hc, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    monkeypatch.setattr(hc, "export_env_from_dotenv", lambda names: None)
    monkeypatch.setattr(hc, "fetch_from_readings", fake_fetch_from_readings)
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    rc = hc.main([
        "--from-readings", str(readings_path), "--reviews-cache-dir", str(reviews_dir),
        "--fulltext-cache-dir", str(fulltext_dir), "--out-dev", str(out_dev),
        "--out-test", str(out_test), "--build-out", str(build_out),
        "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
        "--hss-test-v3-sources", str(empty_sources),
    ])
    assert rc == 0
    assert calls["readings_path"] == readings_path.resolve()
    assert calls["mailto"] == "x@y.z"
    assert out_dev.exists() and out_test.exists()
    # --from-readings isolates the build to readings-path
    # items, so a --from-readings run reports its own extraction filter in the build json.
    build_info = json.loads(build_out.read_text(encoding="utf-8"))
    assert build_info["extraction_filter"] == "readings"


# ---- --from-readings input guards -----------


def test_fetch_from_readings_raises_systemexit_on_a_missing_file(tmp_path: Path):
    import pytest

    with pytest.raises(SystemExit):
        asyncio.run(hc.fetch_from_readings(
            readings_path=tmp_path / "does-not-exist.jsonl", mailto="x@y.z", excluded=set(),
            reviews_cache_dir=tmp_path / "reviews", fulltext_cache_dir=tmp_path / "fulltext",
            min_chars=1, min_chunks=1, max_per_review=4,
        ))
    # No cache directory was created: the guard fires before any output side effect.
    assert not (tmp_path / "reviews").exists()


def test_fetch_from_readings_raises_systemexit_on_an_empty_file(tmp_path: Path):
    import pytest

    readings_path = tmp_path / "readings.jsonl"
    readings_path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        asyncio.run(hc.fetch_from_readings(
            readings_path=readings_path, mailto="x@y.z", excluded=set(),
            reviews_cache_dir=tmp_path / "reviews", fulltext_cache_dir=tmp_path / "fulltext",
            min_chars=1, min_chunks=1, max_per_review=4,
        ))


def test_fetch_from_readings_raises_systemexit_when_every_row_lacks_a_review_doi(
    tmp_path: Path,
):
    import pytest

    readings_path = tmp_path / "readings.jsonl"
    rows = [_reading_row(review_doi=None), _reading_row(review_doi="")]
    readings_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        asyncio.run(hc.fetch_from_readings(
            readings_path=readings_path, mailto="x@y.z", excluded=set(),
            reviews_cache_dir=tmp_path / "reviews", fulltext_cache_dir=tmp_path / "fulltext",
            min_chars=1, min_chunks=1, max_per_review=4,
        ))


def test_main_from_readings_raises_systemexit_on_a_missing_path_and_never_touches_output(
    tmp_path: Path, monkeypatch,
):
    """Reproduced end to end: before this fix, a
    mistyped --from-readings path reached build_from_cache with zero items, and
    _write_jsonl_file truncates the output file before appending nothing, so a single typo
    silently destroyed the committed real_claims_test.jsonl with exit code 0. main must instead
    raise SystemExit before --out-test (or --build-out) is written at all."""
    import pytest

    monkeypatch.setattr(hc, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    monkeypatch.setattr(hc, "export_env_from_dotenv", lambda names: None)
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    original_test_content = '{"item_id": "real-test-01"}\n'
    out_test.write_text(original_test_content, encoding="utf-8")
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    with pytest.raises(SystemExit):
        hc.main([
            "--from-readings", str(tmp_path / "missing.jsonl"),
            "--reviews-cache-dir", str(reviews_dir), "--fulltext-cache-dir", str(fulltext_dir),
            "--out-dev", str(out_dev), "--out-test", str(out_test),
            "--build-out", str(build_out),
            "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
            "--hss-test-v3-sources", str(empty_sources),
        ])
    assert out_test.read_text(encoding="utf-8") == original_test_content
    assert not build_out.exists()


def test_main_from_readings_raises_systemexit_on_an_empty_readings_file(
    tmp_path: Path, monkeypatch,
):
    """The distinct empty-but-present-file case, guarded inside fetch_from_readings rather
    than main's own existence check."""
    import pytest

    monkeypatch.setattr(hc, "load_dotenv_values", lambda path: {"OPENALEX_EMAIL": "x@y.z"})
    monkeypatch.setattr(hc, "export_env_from_dotenv", lambda names: None)
    reviews_dir = tmp_path / "reviews"
    fulltext_dir = tmp_path / "fulltext"
    reviews_dir.mkdir()
    fulltext_dir.mkdir()
    readings_path = tmp_path / "readings.jsonl"
    readings_path.write_text("", encoding="utf-8")
    out_dev = tmp_path / "real_claims_dev.jsonl"
    out_test = tmp_path / "real_claims_test.jsonl"
    original_test_content = '{"item_id": "real-test-01"}\n'
    out_test.write_text(original_test_content, encoding="utf-8")
    build_out = tmp_path / "real_claims.build.json"
    empty_sources = tmp_path / "empty_sources.json"
    write_json(empty_sources, [])
    with pytest.raises(SystemExit):
        hc.main([
            "--from-readings", str(readings_path),
            "--reviews-cache-dir", str(reviews_dir), "--fulltext-cache-dir", str(fulltext_dir),
            "--out-dev", str(out_dev), "--out-test", str(out_test),
            "--build-out", str(build_out),
            "--hss-sources", str(empty_sources), "--hss-dev-sources", str(empty_sources),
            "--hss-test-v3-sources", str(empty_sources),
        ])
    assert out_test.read_text(encoding="utf-8") == original_test_content
    assert not build_out.exists()
