"""Full-text pipeline guards.

Two guards run inside ``acquire_full_texts``, after extraction and before storage, so the
claim verifier can never be shown the wrong paper's text or a reference list mislabelled as
prose:

1. ``check_full_text_identity`` -- the paper's own title / first-author surname must be
   findable, by normalised token overlap, in the first 3,000 characters of the freshly
   extracted text. On a mismatch the full text is marked not acquired with reason
   ``wrong_work`` and is never chunked or stored.
2. ``drop_reference_and_backmatter_chunks`` -- a chunk whose reference-list signal density
   (years in parentheses, DOIs, "Retrieved from", "pp.", author-initial patterns) exceeds a
   fixed threshold, or that falls at or after the first bare References/Bibliography
   heading, is dropped before the chunk list is stored, regardless of its own section label.
   For a chunk at or above ``REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH``, density alone is not
   enough: the chunk must also show at least one of two independent bibliography-structure
   signals (APA-style author-initial matches, or Vancouver/IEEE-style numbered-entry
   matches), because an unconditional, single-signal version of this floor wrongly keeps
   short front-matter/back-matter chunks and long non-APA bibliographies as if they were
   prose.

The fixtures below mirror two real failures: ``10.55593/ej.27108a7``'s Unpaywall link
served Ambele 2022, in Issues in Educational Research, instead of the named Thongwichit and
Ulla TESL-EJ article; and the Apples paper's chunk 2, 2,999 characters of pure
reference list, carried the section label "discussion".

This logic mirrors, but does not import, ``evaluation/claims/build_hss_set.py``'s
``check_fetched_text_identity`` / ``is_reference_or_frontmatter_chunk`` -- the backend never
depends on the evaluation package.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobType  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    make_session_factory,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
)

# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------

WRONG_WORK_TITLE = (
    "Translanguaging Pedagogy in Thailand's English Medium of Instruction Classrooms: "
    "Teachers' Perspectives and Practices"
)
WRONG_WORK_AUTHOR_NAMES = ["Napapat Thongwichit", "Mark B. Ulla"]
WRONG_WORK_AUTHORS = [{"name": n} for n in WRONG_WORK_AUTHOR_NAMES]

# What the OA link actually served for this DOI: a different
# article, by a different author, in a different journal, in a different year.
WRONG_WORK_EXTRACTED_TEXT = (
    "Issues in Educational Research, 32(3), 2022 871\n"
    "Supporting English teaching in Thailand by accepting translanguaging: "
    "Views from Thai university teachers\n"
    "Eric A. Ambele\n"
    "Mahasarakham University, Thailand\n\n"
    "This study explores the perceptions of Thai university English teachers "
    "toward translanguaging as a pedagogical resource in higher education classrooms. "
    "Ambele 879\n"
) * 3

# The text the DOI's own record actually describes.
MATCHING_EXTRACTED_TEXT = (
    "Translanguaging Pedagogy in Thailand's English Medium of Instruction Classrooms: "
    "Teachers' Perspectives and Practices\n"
    "Napapat Thongwichit and Mark B. Ulla\n\n"
    "Abstract\n"
    "This study explores translanguaging pedagogy among Thai teachers in English "
    "Medium of Instruction classrooms.\n\n"
    "Introduction\n"
    "Translanguaging has become an important pedagogical resource in Thailand's "
    "English medium classrooms, and teachers report varied practices.\n"
)

# A reference-list chunk mislabelled "discussion".
REFERENCE_LIST_CHUNK_TEXT = (
    "Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och flerspraakighet. "
    "Apples, 9(1), 1-18. Retrieved from https://apples.jyu.fi\n"
    "Sinclair, J. M., & Coulthard, R. M. (1975). Towards an analysis of discourse. "
    "Oxford, UK: Oxford University Press. pp. 12-34.\n"
    "Snell, J., Shaw, S., & Copland, F. (2015). Linguistic ethnography: Interdisciplinary "
    "explorations. London, UK: Palgrave Macmillan. 10.1057/9781137035035\n"
    "Swain, M. (1985). Communicative competence: Some roles of comprehensible input and "
    "comprehensible output in its development. Rowley, MA: Newbury House. pp. 235-253.\n"
    "Swain, M. (1995). Three functions of output in second language learning. Oxford, "
    "UK: Oxford University Press. pp. 125-144.\n"
)

GOOD_CHUNK_TEXT = (
    "The results indicate that students who received corrective feedback showed higher "
    "post-test scores than those in the control group, suggesting a durable effect of "
    "feedback interventions over time. Teachers reported that flexible deployment of "
    "feedback, rather than a single fixed method, best matched learners' needs."
)

# A real cached-corpus reproduction (evaluation's own
# `10-55593_ej-26103a4.json`, Weng, Zhu and Kim, TESOL International Journal), re-chunked
# with the backend's own `chunk_text`, produces a 5,317-character "Data Analysis" narrative
# chunk that is pure prose -- every sentence is a citation-supported claim about how the
# reviewed studies analysed their data -- but that scores 7.44 hits per 1,000 characters on
# `is_reference_or_frontmatter_chunk`'s five-pattern density test (24 narrative "(YYYY)"
# citations and 12 "p. NN" page cites), comfortably over the 6.0 threshold, while carrying
# zero `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE` ("Smith, J. A."-style) matches: every citation
# here is narrative ("Feryok (2012)", "Miller and Gkonou (2018)"), never a bibliography
# entry. Reproduced (ASCII-cleaned of the source PDF's extraction artefacts -- curly quotes
# and an accented character that are immaterial to the density/author-match counts) rather
# than invented, so this is the same false positive this fixture is meant to catch, not a
# synthetic stand-in.
DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT = (
    "Data Analysis\n\n"
    "Regarding data analysis in those selected studies, a large portion of the studies "
    "emphasizes the iterative and comparative nature of data analysis in a qualitative "
    "fashion. To name a few, Feryok (2012) constantly compared the collected data related "
    "to the research topic. The author specifies that salient information was identified "
    "and reorganized (p. 99). Nguyen and Bui (2016) emphasize a recursive process that "
    "sought patterns, themes, and categories that emerged from data. Similarly, Liyanage "
    "et al. (2015) united recurring ideas and experiences (p. 256) and repeatedly examined "
    "the interview components. Newcomer and Collier (2015) dealt with their data sets both "
    "individually and collaboratively. Likewise, Huyen Phan and Hamid (2017) state that the "
    "data analysis focused on the reinterpretation and practice of learner autonomy in the "
    "English Language classrooms of the four English lectures (p. 46). Babino and Stewart "
    "(2018) utilized a constant comparative method to examine each concept which led to "
    "emerging themes. Also, Zhang (2018) dealt with the transcribed texts repeatedly until "
    "preliminary codes were ready.\n\n"
    "With regards to the procedures of data analysis, most studies conducted data "
    "collection and analysis simultaneously and followed an inductive and interpretive "
    "process for analyzing data while some mentioned the inclusion of deductive components "
    "(e.g., Venegas, 2018). Thematic analysis with coding and memos is a prominent "
    "technique used by the selected studies for data analysis (e.g., Ilieva and Ravindran, "
    "2018; Ishihara et al., 2018; Li and De Costa, 2017; Liyanage et al., 2015; Mifsud and "
    "Vella, 2018; Newcomer and Collier, 2015; Palmer et al., 2016; Tao and Gao, 2017; "
    "Varghese and Snyder, 2018; Wong et al., 2017). In particular, Newcomer and Collier "
    "(2015) identified and refined patterns and themes as local theory that revealed the "
    "constraints faced by teachers while exercising agency (p. 166). Palmer et al. (2016) "
    "and Liyanage et al. (2015) specify that their studies included a deductive process. "
    "Palmer et al. (2016) analyzed each school's data set thematically and explored a "
    "disconfirming evidence (p. 398) during the identification of possible themes. "
    "Liyanage et al. (2015) uncovered recurring themes and meanwhile checked if the "
    "evidence fits the themes well.\n\n"
    "Notably, conceptual or theoretical frameworks were used in several studies to assist "
    "the process of data analysis. For instance, Babino and Stewart (2018) employed "
    "theoretical propositions and grounded theory for guiding their analytic procedures. "
    "For another, using activity theory as theoretical framework, Yang and Clark (2018) and "
    "Yang (2018) scrutinized the interactions of different activity systems. Similarly, Ray "
    "(2009) employed template analysis with a priori themes drawing from a notion of human "
    "agency (p. 126), which allowed the researcher to compare data under a framework.\n\n"
    "Discourse analysis is also another analytic tool that appears across several studies "
    "(e.g., Christiansen et al., 2018; Colegrove and Zuniga, 2018; Glas, 2016; Yang and "
    "Clark, 2018). Taking Christiansen et al. (2018) as an example, the study utilized "
    "narrative inquiry to uncover participants' nuanced reflections on their agentive "
    "work. For another, in White (2018), data was analyzed relying upon the notion of "
    "narrative accounts and stance as an emergent product through social interactions "
    "(p. 582). Similarly, Leal and Crookes (2018) employed a model of teacher agency for "
    "social justice for guiding the analytic procedures.\n\n"
    "It is worth noting that a few studies used data analysis programs to assist their "
    "analysis procedures. For instance, Ray (2009) states that a qualitative software tool "
    "was utilized to analyze a clean data set (p. 120) for a systematic analysis based on "
    "the teacher agency template drawn from its theoretical framework. Likewise, Kang "
    "(2017) indicates that the statistical procedure of the repeated measurement design "
    "was used to determine whether the students' learning outcomes in classes where the LP "
    "was constructed were significantly better than those in classes where LP was not "
    "constructed (p. 88). Miller and Gkonou (2018) involved qualitative data management "
    "software Atlas.ti to inspect the data. Additionally, Tutunis and Hacifazliogulu (2018) "
    "conducted analysis of each questionnaire partly from a quantitative analytic tool "
    "named Statistical Package for Social Sciences (p. 111)."
)

# An unconditional author-match floor reopens the guard for short front-matter and
# back-matter chunks. Reproduced (not invented) from a real cached-corpus paper,
# `evaluation/claims/data/hss_fulltext/10-32601_ejal-710194.json` chunk 0, re-chunked with
# this module's own `chunk_text`: 779 characters of journal masthead, DOI, submission dates
# and the paper's own APA self-citation, scoring 7.70 hits/1,000 chars (over the 6.0
# threshold) with only 2 author-initial matches -- one short of the floor of 3, so an
# unconditional floor would wrongly keep it. The source PDF's
# extraction mangled an en dash into a mojibake byte pair; replaced here with a plain
# hyphen, immaterial to the density/author-match counts.
JOURNAL_FRONT_MATTER_CHUNK_TEXT = (
    "Available online at www.ejal.info \n"
    "http://dx.doi.org/10.32601/ejal.710194  \n"
    "Eurasian Journal of Applied Linguistics, 6(1) (2020) 23-44 \n"
    "EJAL \n"
    "Eurasian Journal of \n"
    "Applied Linguistics \n"
    " \n"
    "Tracing the signature dynamics of foreign language \n"
    "classroom anxiety and foreign language enjoyment: \n"
    "A retrodictive qualitative modeling \n"
    "Majid Elahi Shirvana* \n"
    ", Nahid Talebzadeha \n"
    " \n"
    "aUniversity of Bojnord, Bojnord, Iran \n"
    "Received 27 February 2019 \n"
    "Received in revised form 14 April 2019 \n"
    "Accepted 2 May 2019 \n"
    "APA Citation: \n"
    "Elahi Shirvan, M., & Talebzadeh, N. (2020). Tracing the signature dynamics of foreign "
    "language classroom anxiety \n"
    "and foreign language enjoyment: A retrodictive qualitative modeling. Eurasian Journal "
    "of Applied Linguistics, 6(1), \n"
    "23-44.  \n"
    "Doi: 10.32601/ejal.710194"
)

# Reproduced from the same corpus, `10-55593_ej-27105a9.json` chunk 24: a bibliography tail
# plus copyright notice, 243 characters, scoring 12.35 hits/1,000 chars with only 2
# author-initial matches -- the same one-short-of-the-floor regression as above.
BIBLIOGRAPHY_TAIL_CHUNK_TEXT = (
    "TESL-EJ 27.1, May 2023  \n"
    "Hiratsuka et al. \n"
    "20 \n"
    "Wadden, P., & Hale, C. C. (Eds.). (2019). Teaching English at Japanese universities: "
    "A new \n"
    "handbook. Routledge. \n"
    " \n"
    "Copyright of articles rests with the authors. Please cite TESL-EJ appropriately."
)

# A short (well under the length floor) non-APA, Vancouver-style bibliography: numbered
# entries of the form "1. Surname AB, Surname CD. Title...", never "Surname, A. B." --
# carries zero `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE` matches regardless of length, since that
# pattern is APA-specific. Round 4's unconditional floor exempted this from the density
# check entirely; it must still be dropped.
SHORT_VANCOUVER_BIBLIOGRAPHY_TEXT = (
    "1. Smith JA, Doe RB, Lee KH. Effects of feedback timing on second language "
    "acquisition outcomes. Lang Learn J. 2018;45(3):210-225. "
    "doi:10.1016/j.langlearn.2018.03.004\n"
    "2. Nguyen TT, Park SY. Corrective feedback and learner uptake in EFL classrooms. "
    "Appl Linguist Rev. 2019;12(1):45-60. doi:10.1075/alr.19004.ngu\n"
    "3. Chen WL. Teacher cognition in feedback delivery. TESOL Q. 2020;54(2):301-310. "
    "doi:10.1002/tesq.3021"
)

# A short, non-APA, IEEE-style bibliography: bracketed entries of the form "[1] A. Surname
# and B. Surname, ...", also zero author-initial matches for the same reason.
SHORT_IEEE_BIBLIOGRAPHY_TEXT = (
    "[1] J. Smith and R. Doe, feedback timing effects on language acquisition, "
    "Lang. Learn. J., vol. 45, no. 3, pp. 210-225, 2018, "
    "doi: 10.1016/j.langlearn.2018.03.004.\n"
    "[2] T. Nguyen, corrective feedback and uptake, Appl. Linguist. Rev., vol. 12, "
    "pp. 45-60, 2019, doi: 10.1075/alr.19004.ngu."
)


def _build_long_vancouver_bibliography_text() -> str:
    """A 25-entry, Vancouver-style bibliography at or above
    `REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH`: zero author-initial matches (same reason as
    the short fixtures above), but 25 `_REFERENCE_CHUNK_NUMBERED_ENTRY_RE` matches -- the
    second structural signal that must, on its own, keep this chunk eligible for the density
    check even though it clears the length floor and has no APA-style author-initial
    commas."""
    authors_pool = [
        ("Smith JA", "Doe RB"), ("Nguyen TT", "Park SY"), ("Chen WL", "Lopez MA"),
        ("Ahmed KH", "Fischer TN"), ("Okafor CN", "Brown LJ"), ("Tanaka H", "Silva RP"),
        ("Kim JY", "Rossi GB"), ("Novak PK", "Duarte FM"), ("Ivanov DS", "Weber AK"),
        ("Hassan MI", "Costa VL"),
    ]
    journals = ["Lang Learn J", "Appl Linguist Rev", "TESOL Q", "System", "ELT J", "RELC J"]
    lines = []
    for i in range(1, 26):
        a1, a2 = authors_pool[(i - 1) % len(authors_pool)]
        journal = journals[(i - 1) % len(journals)]
        volume = 40 + i
        lines.append(
            f"{i}. {a1}, {a2}. Study {i} on language learning strategies. {journal}. "
            f"2019;{volume}(2):{100 + i}-{120 + i}. doi:10.1016/j.ll.2019.{i:03d}"
        )
    return "\n".join(lines)


LONG_VANCOUVER_BIBLIOGRAPHY_TEXT = _build_long_vancouver_bibliography_text()


# ---------------------------------------------------------------------------------------
# Unit tests: check_full_text_identity
# ---------------------------------------------------------------------------------------


def test_check_full_text_identity_accepts_a_matching_paper():
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(
        expected_title=WRONG_WORK_TITLE,
        extracted_text=MATCHING_EXTRACTED_TEXT,
        expected_authors=WRONG_WORK_AUTHOR_NAMES,
    )
    assert ok is True
    assert fragment is None


def test_check_full_text_identity_rejects_the_ambele_wrong_work_fixture():
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(
        expected_title=WRONG_WORK_TITLE,
        extracted_text=WRONG_WORK_EXTRACTED_TEXT,
        expected_authors=WRONG_WORK_AUTHOR_NAMES,
    )
    assert ok is False
    assert fragment is not None
    assert "ambele" in fragment.lower()


def test_check_full_text_identity_has_nothing_to_check_when_title_is_empty():
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(expected_title="", extracted_text="anything at all")
    assert ok is True
    assert fragment is None


def test_check_full_text_identity_skips_the_author_check_when_no_authors_given():
    from app.services.fulltext import check_full_text_identity

    ok, _fragment = check_full_text_identity(
        expected_title=WRONG_WORK_TITLE,
        extracted_text=MATCHING_EXTRACTED_TEXT,
        expected_authors=None,
    )
    assert ok is True


def test_check_full_text_identity_rejects_a_same_subfield_decoy_despite_high_title_overlap():
    """The Ambele fixture above is exactly this case in the wild: two papers in the same
    narrow subfield share enough title vocabulary that title overlap alone clears the
    threshold, so the author check is what actually catches the decoy -- title overlap
    alone is deliberately not sufficient. Restated here with a smaller, simpler fixture so
    the requirement is pinned independently of the Ambele text's exact wording."""
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(
        expected_title="Corrective Feedback and Learner Uptake in EFL Classrooms",
        extracted_text=(
            "Corrective Feedback and Learner Motivation in EFL Classrooms\n"
            "Somchai Rattana\n\n"
            "Abstract\nThis study examines corrective feedback and learner motivation in "
            "EFL classrooms across three schools."
        ),
        expected_authors=["Amara Vidal"],
    )
    assert ok is False
    assert fragment is not None


def test_check_full_text_identity_rejects_a_wrong_author_even_with_perfect_title_overlap():
    """Symmetric case: an exact title match alone is also not sufficient when the named
    author is demonstrably someone else -- both signals must agree."""
    from app.services.fulltext import check_full_text_identity

    ok, _fragment = check_full_text_identity(
        expected_title="A Study of Feedback",
        extracted_text="A Study of Feedback\nSomeone Else Entirely\n\nAbstract...",
        expected_authors=["Jane Smith"],
    )
    assert ok is False


def test_check_full_text_identity_accepts_a_paper_with_the_openalex_untitled_fallback():
    """`app/clients/openalex.py`'s `_parse_work` stores the literal title "Untitled" when
    OpenAlex itself has no title for a work. That single non-stopword token can never
    overlap a real PDF's own text, so an untitled paper's perfectly correct full text
    would otherwise be rejected as wrong_work. A title that folds to exactly "untitled"
    must be treated as absent, the same as an empty title."""
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(
        expected_title="Untitled",
        extracted_text=(
            "Some Real Paper About Feedback\nSome Real Author\n\n"
            "Abstract\nThis paper studies feedback in the classroom."
        ),
        expected_authors=["Some Real Author"],
    )
    assert ok is True
    assert fragment is None


def test_check_full_text_identity_accepts_a_family_comma_given_author_against_initials():
    """`first.strip().split()[-1]` returns the
    given name, not the surname, when an author is stored "Family, Given" -- the
    bulk-upload path takes author names from an LLM extraction agent, so this format is not
    guaranteed. `_build_author_year_lookup` already handles the same format defensively in
    this module (`first_author.split(",")[0].split()[-1]`); `check_full_text_identity` must
    reuse that extraction rather than its own comma-naive one."""
    from app.services.fulltext import check_full_text_identity

    ok, fragment = check_full_text_identity(
        expected_title="A Study of Pedagogical Translanguaging",
        extracted_text=(
            "A Study of Pedagogical Translanguaging\nN. Thongwichit\n\n"
            "Abstract\nThis study examines pedagogical translanguaging practices."
        ),
        expected_authors=["Thongwichit, Napapat"],
    )
    assert ok is True
    assert fragment is None


# ---------------------------------------------------------------------------------------
# Unit tests: is_reference_or_frontmatter_chunk / drop_reference_and_backmatter_chunks
# ---------------------------------------------------------------------------------------


def test_is_reference_or_frontmatter_chunk_flags_a_dense_bibliography():
    from app.services.fulltext import is_reference_or_frontmatter_chunk

    assert is_reference_or_frontmatter_chunk(REFERENCE_LIST_CHUNK_TEXT) is True


def test_is_reference_or_frontmatter_chunk_leaves_ordinary_prose_alone():
    from app.services.fulltext import is_reference_or_frontmatter_chunk

    assert is_reference_or_frontmatter_chunk(GOOD_CHUNK_TEXT) is False


def test_is_reference_or_frontmatter_chunk_ignores_a_short_fragment():
    """A chunk shorter than the minimum length is never flagged: one citation would look
    arbitrarily dense at that scale."""
    from app.services.fulltext import is_reference_or_frontmatter_chunk

    assert is_reference_or_frontmatter_chunk("(2020) pp. 1") is False


def test_is_reference_or_frontmatter_chunk_keeps_a_citation_dense_discussion_paragraph():
    """Density alone is not enough. The real cached-corpus
    Data Analysis narrative (see the fixture's own comment) clears the density threshold on
    narrative "(YYYY)" citations and "p. NN" page cites alone, with zero author-initial
    ("Smith, J. A."-style) matches -- the structural signal that actually marks a
    bibliography. This must be kept, not dropped as a reference list."""
    from app.services.fulltext import (
        REFERENCE_CHUNK_DENSITY_THRESHOLD,
        _reference_chunk_pattern_count,
        is_reference_or_frontmatter_chunk,
    )

    text = DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT
    # Sanity check on the fixture itself: it must actually clear the density threshold,
    # otherwise this test would pass for the wrong reason (never reaching the author-match
    # requirement at all).
    density = _reference_chunk_pattern_count(text) / (len(text) / 1000.0)
    assert density > REFERENCE_CHUNK_DENSITY_THRESHOLD

    assert is_reference_or_frontmatter_chunk(text) is False


def test_is_reference_or_frontmatter_chunk_still_flags_a_true_bibliography_by_author_count():
    """Regression guard for the fix above: the true reference-list fixture has well over
    three author-initial entries (Rosen, J.; Wedin, A.; Sinclair, J. M.; ...), so the new
    author-match requirement must not accidentally exempt real bibliographies."""
    from app.services.fulltext import (
        _REFERENCE_CHUNK_INITIAL_AUTHOR_RE,
        is_reference_or_frontmatter_chunk,
    )

    assert len(_REFERENCE_CHUNK_INITIAL_AUTHOR_RE.findall(REFERENCE_LIST_CHUNK_TEXT)) >= 3
    assert is_reference_or_frontmatter_chunk(REFERENCE_LIST_CHUNK_TEXT) is True


def test_is_reference_or_frontmatter_chunk_still_flags_short_journal_front_matter():
    """An author-match floor, applied
    unconditionally, would wrongly keep this real 779-character front-matter chunk (masthead,
    DOI, submission dates, the paper's own APA self-citation) because it carries only 2
    author-initial matches, one short of the floor of 3. Below
    `REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH` the floor must not apply at all, so density
    alone -- which this chunk clears -- still flags it."""
    from app.services.fulltext import (
        REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH,
        REFERENCE_CHUNK_DENSITY_THRESHOLD,
        _reference_chunk_pattern_count,
        is_reference_or_frontmatter_chunk,
    )

    text = JOURNAL_FRONT_MATTER_CHUNK_TEXT
    assert len(text) < REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
    density = _reference_chunk_pattern_count(text) / (len(text) / 1000.0)
    assert density > REFERENCE_CHUNK_DENSITY_THRESHOLD

    assert is_reference_or_frontmatter_chunk(text) is True


def test_is_reference_or_frontmatter_chunk_still_flags_short_bibliography_tail():
    """Same regression as above, on the real 243-character bibliography-tail-plus-
    copyright-notice chunk: also 2 author-initial matches, also wrongly kept once the floor
    applied unconditionally."""
    from app.services.fulltext import (
        REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH,
        is_reference_or_frontmatter_chunk,
    )

    assert len(BIBLIOGRAPHY_TAIL_CHUNK_TEXT) < REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
    assert is_reference_or_frontmatter_chunk(BIBLIOGRAPHY_TAIL_CHUNK_TEXT) is True


@pytest.mark.parametrize(
    "text",
    [SHORT_VANCOUVER_BIBLIOGRAPHY_TEXT, SHORT_IEEE_BIBLIOGRAPHY_TEXT],
    ids=["vancouver", "ieee"],
)
def test_is_reference_or_frontmatter_chunk_flags_short_non_apa_bibliographies(text):
    """Numbered (Vancouver/IEEE-style)
    bibliographies carry zero `_REFERENCE_CHUNK_INITIAL_AUTHOR_RE` matches regardless of
    length, since that pattern is APA-specific ("Surname, A. B."). Below the length floor
    this must not matter -- density alone still flags them, exactly as if no
    author-match floor existed."""
    from app.services.fulltext import (
        _REFERENCE_CHUNK_INITIAL_AUTHOR_RE,
        REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH,
        is_reference_or_frontmatter_chunk,
    )

    assert len(text) < REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
    assert _REFERENCE_CHUNK_INITIAL_AUTHOR_RE.findall(text) == []

    assert is_reference_or_frontmatter_chunk(text) is True


def test_is_reference_or_frontmatter_chunk_flags_a_long_non_apa_bibliography_by_numbered_entries():
    """A long (>= the length floor) numbered
    bibliography has zero APA-style author-initial matches, so the author-match signal alone
    would wrongly exempt it once the chunk clears the length floor. The numbered-entry
    signal must independently keep it eligible for the density check."""
    from app.services.fulltext import (
        _REFERENCE_CHUNK_INITIAL_AUTHOR_RE,
        _REFERENCE_CHUNK_NUMBERED_ENTRY_RE,
        REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH,
        REFERENCE_CHUNK_DENSITY_THRESHOLD,
        REFERENCE_CHUNK_MIN_NUMBERED_ENTRIES,
        _reference_chunk_pattern_count,
        is_reference_or_frontmatter_chunk,
    )

    text = LONG_VANCOUVER_BIBLIOGRAPHY_TEXT
    assert len(text) >= REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
    assert _REFERENCE_CHUNK_INITIAL_AUTHOR_RE.findall(text) == []
    numbered_matches = _REFERENCE_CHUNK_NUMBERED_ENTRY_RE.findall(text)
    assert len(numbered_matches) >= REFERENCE_CHUNK_MIN_NUMBERED_ENTRIES
    density = _reference_chunk_pattern_count(text) / (len(text) / 1000.0)
    assert density > REFERENCE_CHUNK_DENSITY_THRESHOLD

    assert is_reference_or_frontmatter_chunk(text) is True


def test_is_reference_or_frontmatter_chunk_numbered_entry_regex_does_not_fire_on_narrative_prose():
    """The numbered-entry signal must stay 0 on ordinary narrative prose, including the
    dense-discussion fixture, so it cannot itself cause a false positive on prose
    that happens to be long and citation-dense."""
    from app.services.fulltext import _REFERENCE_CHUNK_NUMBERED_ENTRY_RE

    assert _REFERENCE_CHUNK_NUMBERED_ENTRY_RE.findall(
        DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT
    ) == []
    assert _REFERENCE_CHUNK_NUMBERED_ENTRY_RE.findall(GOOD_CHUNK_TEXT) == []


# Counting any line shaped like "[12] ", "12. " or "12) " toward the numbered-entry
# signal, with no requirement that the line actually look like a bibliography entry,
# would misfire on the same shape as an ordinary numbered list in running prose
# (enumerated research questions, findings, steps). A normal 3-item numbered
# research-question list carries none of a DOI, a "pp." page range, "Retrieved from/on",
# a parenthesised year, or a bare 4-digit year on any of its lines, while a Vancouver or
# IEEE bibliography entry carries at least one on every line, which is why the count is
# gated per line below.
NUMBERED_RESEARCH_QUESTION_LIST_TEXT = (
    "1. What challenges do teachers report when adopting the new curriculum?\n"
    "2. How do students respond to more frequent formative feedback?\n"
    "3. What role does timing play in the effectiveness of that feedback?\n"
)


def test_reference_chunk_numbered_entry_count_ignores_a_plain_numbered_list():
    """None of the three research-question lines carries a reference signal, so the
    signal-gated count must stay at 0 even though the raw shape-only regex fires 3 times."""
    from app.services.fulltext import (
        _REFERENCE_CHUNK_NUMBERED_ENTRY_RE,
        _reference_chunk_numbered_entry_count,
    )

    assert len(_REFERENCE_CHUNK_NUMBERED_ENTRY_RE.findall(
        NUMBERED_RESEARCH_QUESTION_LIST_TEXT
    )) == 3
    assert _reference_chunk_numbered_entry_count(NUMBERED_RESEARCH_QUESTION_LIST_TEXT) == 0


def test_reference_chunk_numbered_entry_count_counts_lines_carrying_a_reference_signal():
    """Every line of the Vancouver/IEEE fixtures below carries a DOI, so the signal-gated
    count must equal the raw shape-only count on all three."""
    from app.services.fulltext import _reference_chunk_numbered_entry_count

    assert _reference_chunk_numbered_entry_count(LONG_VANCOUVER_BIBLIOGRAPHY_TEXT) == 25
    assert _reference_chunk_numbered_entry_count(SHORT_VANCOUVER_BIBLIOGRAPHY_TEXT) == 3
    assert _reference_chunk_numbered_entry_count(SHORT_IEEE_BIBLIOGRAPHY_TEXT) == 2


def test_reference_chunk_numbered_entry_count_is_gated_per_line_not_globally():
    """A numbered list with a mix of reference-shaped and plain lines must only count the
    lines that individually carry a reference signal, not the chunk as a whole."""
    from app.services.fulltext import _reference_chunk_numbered_entry_count

    mixed = (
        "1. Smith JA. A study of language learning. J Ex. 2019;1(1):1-2. "
        "doi:10.1000/xyz\n"
        "2. What role does motivation play in learning outcomes?\n"
        "3. Doe RB. Another study of feedback. Oxford, UK: OUP. pp. 12-14\n"
    )
    assert _reference_chunk_numbered_entry_count(mixed) == 2


def test_is_reference_or_frontmatter_chunk_keeps_a_dense_discussion_with_a_plain_numbered_list():
    """Inserting a normal 3-item numbered research-question list into the dense-discussion
    fixture could reopen the false positive above if a numbered-entry count counted the
    list's 3 shape-only matches and exempted the chunk from the author-match floor,
    routing it back through density alone (where it would be wrongly dropped). None of
    the list's lines carries a reference signal, so the signal-gated count must stay at 0
    and the chunk must be kept, exactly like the base fixture without the list."""
    from app.services.fulltext import (
        _REFERENCE_CHUNK_NUMBERED_ENTRY_RE,
        REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH,
        is_reference_or_frontmatter_chunk,
    )

    text = DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT.replace(
        "Data Analysis\n\n",
        "Data Analysis\n\n" + NUMBERED_RESEARCH_QUESTION_LIST_TEXT + "\n",
        1,
    )
    # Sanity checks on the fixture itself, so this test fails for the right reason: the
    # chunk is still above the length floor, and the raw shape-only regex still fires 3
    # times (reproducing the exact condition that reopened the false positive).
    assert len(text) >= REFERENCE_CHUNK_AUTHOR_FLOOR_MIN_LENGTH
    assert len(_REFERENCE_CHUNK_NUMBERED_ENTRY_RE.findall(text)) == 3

    assert is_reference_or_frontmatter_chunk(text) is False


def test_drop_reference_and_backmatter_chunks_drops_a_mislabelled_reference_chunk():
    """The Apples fixture: dropped by density, regardless of its own section label."""
    from app.services.fulltext import drop_reference_and_backmatter_chunks

    chunks = [
        {"section": "results", "text": GOOD_CHUNK_TEXT},
        {"section": "discussion", "text": REFERENCE_LIST_CHUNK_TEXT},
    ]
    kept, dropped = drop_reference_and_backmatter_chunks(chunks)
    assert [c["section"] for c in kept] == ["results"]
    assert dropped == 1


def test_drop_reference_and_backmatter_chunks_drops_everything_from_a_references_heading_onward():
    from app.services.fulltext import drop_reference_and_backmatter_chunks

    chunks = [
        {"section": "discussion", "text": GOOD_CHUNK_TEXT},
        {"section": "references", "text": "References\n" + REFERENCE_LIST_CHUNK_TEXT},
        {"section": "appendix", "text": "Appendix A. A short, non-dense note about materials."},
    ]
    kept, dropped = drop_reference_and_backmatter_chunks(chunks)
    assert [c["section"] for c in kept] == ["discussion"]
    assert dropped == 2


def test_drop_reference_and_backmatter_chunks_keeps_a_citation_dense_discussion_chunk():
    """End-to-end version of the dense-discussion-with-a-numbered-list fixture above,
    through the public drop function:
    a citation-dense discussion chunk with no bibliography structure must survive, sitting
    next to a true reference-list chunk that is still dropped."""
    from app.services.fulltext import drop_reference_and_backmatter_chunks

    chunks = [
        {"section": "discussion", "text": DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT},
        {"section": "references", "text": REFERENCE_LIST_CHUNK_TEXT},
    ]
    kept, dropped = drop_reference_and_backmatter_chunks(chunks)
    assert [c["section"] for c in kept] == ["discussion"]
    assert dropped == 1


def test_drop_reference_and_backmatter_chunks_keeps_normal_chunks_untouched():
    from app.services.fulltext import drop_reference_and_backmatter_chunks

    chunks = [
        {"section": "introduction", "text": GOOD_CHUNK_TEXT},
        {"section": "results", "text": GOOD_CHUNK_TEXT},
    ]
    kept, dropped = drop_reference_and_backmatter_chunks(chunks)
    assert kept == chunks
    assert dropped == 0


# ---------------------------------------------------------------------------------------
# Integration tests: acquire_full_texts wires both guards into the acquisition pipeline
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_work_full_text_is_rejected_and_never_chunked(db_session):
    """A wrong-work full text is marked not acquired with reason ``wrong_work``, the job
    records the reason, and the text is never passed to ``chunk_text``."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=WRONG_WORK_TITLE,
        authors=WRONG_WORK_AUTHORS,
        doi="10.55593/ej.27108a7",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://www.iier.org.au/iier32/ambele.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=WRONG_WORK_EXTRACTED_TEXT,
        ),
        patch("app.services.fulltext.chunk_text") as mock_chunk_text,
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_chunk_text.assert_not_called()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] != "acquired"
    assert metadata["fulltext_reason"] == "wrong_work"
    assert "fulltext_chunks" not in metadata

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["acquired"] == 0
    assert job.result["wrong_work"] == 1


@pytest.mark.asyncio
async def test_reference_list_chunk_is_dropped_before_storage(db_session):
    """A reference-list chunk mislabelled with a body section name is dropped before the
    chunk list is stored, and the dropped count is recorded in the paper's own acquisition
    provenance and in the job result."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="A Study of Feedback",
        authors=[{"name": "Jane Smith"}],
        doi="10.1/feedback-study",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    crafted_chunks = [
        {"section": "results", "text": GOOD_CHUNK_TEXT},
        {"section": "discussion", "text": REFERENCE_LIST_CHUNK_TEXT},
    ]

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/feedback-study.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value="A Study of Feedback by Jane Smith. " + GOOD_CHUNK_TEXT,
        ),
        patch("app.services.fulltext.chunk_text", return_value=crafted_chunks),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert [c["section"] for c in metadata["fulltext_chunks"]] == ["results"]
    assert metadata["fulltext_dropped_reference_chunks"] == 1

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["acquired"] == 1
    assert job.result["reference_chunks_dropped"] == 1


@pytest.mark.asyncio
async def test_real_chunker_keeps_a_dense_discussion_and_drops_only_the_references_section(
    db_session,
):
    """Every integration test above patches ``chunk_text``,
    so none of them exercises the real chunker's actual output shape -- whole,
    arbitrarily long sections rather than small, hand-crafted fragments. This test lets
    ``chunk_text`` run for real against a document with genuine section headers (so
    ``detect_sections`` produces one chunk per section, each as large as the section
    itself), so a citation-dense discussion section that is not a bibliography survives
    acquisition, and only the true references section is dropped."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="A Study of Feedback",
        authors=[{"name": "Jane Smith"}],
        doi="10.1/feedback-study-real-chunker",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    extracted_text = (
        "A Study of Feedback\nJane Smith\n\n"
        "Abstract\n" + GOOD_CHUNK_TEXT + "\n\n"
        "Introduction\n" + GOOD_CHUNK_TEXT + "\n\n"
        "Discussion\n" + DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT + "\n\n"
        "References\n" + REFERENCE_LIST_CHUNK_TEXT
    )

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/feedback-study-real-chunker.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=extracted_text,
        ),
        # `chunk_text` is deliberately NOT patched here: this is the point of the test.
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    sections = [c["section"] for c in metadata["fulltext_chunks"]]
    assert "references" not in sections
    assert "discussion" in sections
    discussion_text = next(
        c["text"] for c in metadata["fulltext_chunks"] if c["section"] == "discussion"
    )
    assert discussion_text.strip() == DENSE_DISCUSSION_WITHOUT_AUTHOR_INITIALS_TEXT.strip()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["reference_chunks_dropped"] == 1


@pytest.mark.asyncio
async def test_normal_acquisition_is_unaffected_by_the_new_guards(db_session):
    """Regression: a paper whose text matches its record and whose chunks are all clean
    prose is stored exactly as before, and the job result keeps its original three-key
    shape (no new keys appear unless a guard actually fires)."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="A Study of Feedback",
        authors=[{"name": "Jane Smith"}],
        doi="10.1/feedback-study-2",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    crafted_chunks = [{"section": "results", "text": GOOD_CHUNK_TEXT}]

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/feedback-study-2.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value="A Study of Feedback by Jane Smith. " + GOOD_CHUNK_TEXT,
        ),
        patch("app.services.fulltext.chunk_text", return_value=crafted_chunks),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert metadata["fulltext_chunks"] == crafted_chunks
    assert "fulltext_reason" not in metadata
    assert "fulltext_dropped_reference_chunks" not in metadata

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result == {"acquired": 1, "abstract_only": 0, "already_acquired": 0}


@pytest.mark.asyncio
async def test_a_later_successful_acquisition_clears_a_stale_wrong_work_marker(db_session):
    """`acquire_full_texts` only skips papers
    already marked "acquired", so a paper rejected as `wrong_work` on one run is retried on
    the next. A later success must not leave the earlier rejection's `fulltext_reason` and
    `fulltext_wrong_work_fragment` markers behind -- an acquired paper must not keep a
    metadata field claiming its full text was rejected as the wrong work."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=WRONG_WORK_TITLE,
        authors=WRONG_WORK_AUTHORS,
        doi="10.55593/ej.27108a7",
    )
    job1 = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://www.iier.org.au/iier32/ambele.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=WRONG_WORK_EXTRACTED_TEXT,
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job1.id, session_factory=factory, paper_ids=None
        )

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_reason"] == "wrong_work"
    assert "fulltext_wrong_work_fragment" in metadata

    job2 = await seed_job(db_session, project, JobType.fulltext_acquire)
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/matching.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-fake-bytes",
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=MATCHING_EXTRACTED_TEXT,
        ),
        patch(
            "app.services.fulltext.chunk_text",
            return_value=[{"section": "results", "text": GOOD_CHUNK_TEXT}],
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job2.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert "fulltext_reason" not in metadata
    assert "fulltext_wrong_work_fragment" not in metadata

    await db_session.refresh(job2)
    assert job2.status.value == "completed", job2.error
    assert job2.result["acquired"] == 1
