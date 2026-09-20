"""Pure-function tests for two fixes to the write job's gated loop
(``app.services.writing``):

* ``_strip_duplicate_leading_heading`` drops every leading markdown heading line or
  whole-line bold run the model writes, whatever it says: the section title is the
  only heading a saved section ever carries, so the saved section never opens with a
  second heading node right after it, whether that second heading repeats the
  section title or is worded differently again.
* ``_close_citation_link_coverage_gap`` runs the bounded coverage-gap retry
  (``app.services.fulltext.resolve_unclassified_sentences``) over one citation-link
  call's own results, using this section's own linker as the retry's linker, and folds
  the retry's own provenance into the section's running totals.

No network, no database: `build_citation_link_map`'s own ``link_citations`` call is
monkeypatched.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.agents.citation_link_agent import CitationLinkResult  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402
from app.services import writing  # noqa: E402
from app.services.writing import (  # noqa: E402
    _close_citation_link_coverage_gap,
    _strip_duplicate_leading_heading,
)


def _fake_provenance() -> LLMCallProvenance:
    return LLMCallProvenance(
        agent="citation_link",
        model_configured="deepseek-chat",
        model_reported="deepseek-chat",
        provider_response_id="resp-1",
        system_fingerprint=None,
        temperature=0.0,
        prompt_version="sha256:test",
        input_tokens=10,
        output_tokens=5,
    )


# --- _strip_duplicate_leading_heading -----------------------------------------------


def test_strip_duplicate_leading_heading_drops_a_matching_markdown_heading():
    text = "# Literature Review\n\nFirst paragraph.\n\nSecond paragraph."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "First paragraph.\n\nSecond paragraph."


def test_strip_duplicate_leading_heading_is_case_and_whitespace_insensitive():
    text = "##  literature   review \n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_drops_a_differently_worded_leading_heading():
    """The section title is the only heading a saved section ever carries now: a
    differently-worded leading heading -- a genuine, distinctly-worded sub-heading
    (for example "Effects of Written Corrective Feedback on L2 Writing Accuracy and
    Revision", "Literature Review: Written Corrective Feedback in Second-Language
    Writing", "2.1 Feedback Form and Accuracy Outcomes", none of them a repeat of the
    section title or the generic default) -- is dropped whole exactly like an exact
    repeat is, not kept as a second heading node."""
    text = "# Background and Motivation\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_drops_two_leading_headings_of_any_wording():
    """The model's raw text can open with TWO consecutive heading lines, ``#
    Literature Review`` directly followed by ``### Effects of Feedback Form on
    Writing Accuracy`` (a differently-worded heading). Both are dropped, in one
    pass: the function loops, re-examining the new first block after each drop, so
    the second heading left behind by the first drop is recognised as heading-shaped
    too and is dropped in the same call, rather than surviving as a second heading
    node right after the section's own injected H2."""
    text = "# Literature Review\n\n### Effects of Feedback Form on Writing Accuracy\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_drops_two_consecutive_matching_headings():
    text = "# Literature Review\n\n## literature review\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_leaves_a_heading_elsewhere_in_the_body():
    text = "First paragraph.\n\n# Literature Review\n\nSecond paragraph."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == text


def test_strip_duplicate_leading_heading_drops_a_heading_line_with_extra_trailing_words():
    """A heading line with extra words tacked on the same line (``"# Literature
    Review carries more text"``) still matches ``_MARKDOWN_HEADING_RE`` in full --
    it is dropped whole regardless of its own wording, the same as any other
    leading heading."""
    text = "# Literature Review carries more text\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_leaves_empty_text_unchanged():
    assert _strip_duplicate_leading_heading("", "Literature Review") == ""


def test_strip_duplicate_leading_heading_drops_a_matching_bold_run():
    """The model's other way of marking a heading is a first block of exactly
    ``**Literature Review**``. `_build_section_tiptap_nodes` promotes this same shape to
    a level-3 heading node, which would otherwise duplicate the section's own H2."""
    text = "**Literature Review**\n\nFirst paragraph.\n\nSecond paragraph."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "First paragraph.\n\nSecond paragraph."


def test_strip_duplicate_leading_heading_drops_a_numbered_markdown_heading():
    """The model can number its heading, e.g. ``## 2. Literature Review``. Comparing
    "Literature Review" (the section's own H2) against "2. Literature Review" (the
    model's numbered repeat) as plain strings would see two different strings, so the
    duplicate would go undetected without normalising the leading number away."""
    text = "## 2. Literature Review\n\nFirst paragraph.\n\nSecond paragraph."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "First paragraph.\n\nSecond paragraph."


def test_strip_duplicate_leading_heading_drops_a_closed_atx_heading():
    """A trailing run of ``#`` characters ("closed ATX" markdown
    style) around an otherwise-matching heading."""
    text = "## Literature Review ##\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_drops_a_differently_numbered_and_worded_heading():
    text = "## 3. Background and Motivation\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


def test_strip_duplicate_leading_heading_leaves_a_bold_run_with_more_text():
    text = "**Literature Review** carries more text\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == text


def test_strip_duplicate_leading_heading_drops_a_trailing_period():
    """``_normalized_heading_text`` must tolerate a leading
    section number and a trailing ``#`` run as well as trailing sentence punctuation,
    so "**Literature Review.**" is recognised as the same duplicate a bare repeat
    already is -- one punctuation mark away from the closed-ATX and numbered cases
    above."""
    text = "**Literature Review.**\n\nFirst paragraph.\n\nSecond paragraph."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "First paragraph.\n\nSecond paragraph."


def test_strip_duplicate_leading_heading_drops_a_markdown_heading_with_trailing_punctuation():
    text = "## Literature Review?\n\nBody."
    result = _strip_duplicate_leading_heading(text, "Literature Review")
    assert result == "Body."


# --- _close_citation_link_coverage_gap ------------------------------------------------


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_is_a_no_op_when_nothing_is_missed():
    text = "Smith (2020) reported large gains."
    links = [{"sentence": text, "keys": ["smith_2020"], "citation_text": "(Smith, 2020)"}]
    uncited: list[dict] = []
    provenance = {"total_calls": 1}

    new_links, new_uncited, new_provenance, retry_failed = (
        await _close_citation_link_coverage_gap(
            text, links, uncited, {"smith_2020": None}, None, provenance
        )
    )

    assert new_links == links
    assert new_uncited == []
    # No retry call was made, so the provenance dict must be unchanged.
    assert new_provenance == provenance
    assert retry_failed is False


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_resolves_a_gap_into_a_link(monkeypatch):
    gap_sentence = "Mixed-effect models showed a durable four-week advantage."
    text = f"Jones (2019) confirmed prior work. {gap_sentence}"
    links = [
        {"sentence": "Jones (2019) confirmed prior work.", "keys": ["jones_2019"],
         "citation_text": "(Jones, 2019)"}
    ]
    uncited: list[dict] = []
    provenance = {"total_calls": 1, "total_input_tokens": 100, "total_output_tokens": 20}

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        assert gap_sentence in content
        return CitationLinkResult(
            links=[
                {"sentence": gap_sentence, "keys": ["lee_2021"], "citation_text": "(Lee, 2021)"}
            ],
            provenance=_fake_provenance(),
            uncited_sentences=[],
        )

    monkeypatch.setattr(writing, "link_citations", fake_link_citations)

    new_links, new_uncited, new_provenance, retry_failed = (
        await _close_citation_link_coverage_gap(
            text, links, uncited, {"jones_2019": None, "lee_2021": None}, None, provenance
        )
    )

    assert any(
        link["sentence"] == gap_sentence and link["keys"] == ["lee_2021"] for link in new_links
    )
    assert new_uncited == []
    assert retry_failed is False
    # The retry call's own provenance was folded into the section's running totals.
    assert new_provenance["total_calls"] == 2
    assert new_provenance["citation_link_calls"][-1] is not None


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_marks_a_removable_finding_on_retry_answer(
    monkeypatch,
):
    """The retry call itself succeeded; it simply never named the gap. That is a
    positive answer, not a transient failure, so the sentence is an ordinary,
    removable "finding" with no ``"unclassified"`` flag."""
    gap_sentence = "Cognitive load was lower for one feedback type on grammatical errors."
    text = gap_sentence
    provenance = {"total_calls": 1}

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        return CitationLinkResult(links=[], provenance=_fake_provenance(), uncited_sentences=[])

    monkeypatch.setattr(writing, "link_citations", fake_link_citations)

    new_links, new_uncited, new_provenance, retry_failed = (
        await _close_citation_link_coverage_gap(text, [], [], {}, None, provenance)
    )

    assert new_links == []
    assert retry_failed is False
    assert new_uncited == [{"paragraph_index": 0, "sentence": gap_sentence, "tag": "finding"}]
    assert new_provenance["total_calls"] == 2


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_keeps_unclassified_only_on_retry_failure(
    monkeypatch,
):
    """The transient-failure guarantee: when the retry call itself raises, the gap is
    never actually looked at, so it is kept, marked ``"unclassified": True``."""
    gap_sentence = "Cognitive load was lower for one feedback type on grammatical errors."
    text = gap_sentence
    provenance = {"total_calls": 1}

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        raise RuntimeError("provider outage")

    monkeypatch.setattr(writing, "link_citations", fake_link_citations)

    new_links, new_uncited, new_provenance, retry_failed = (
        await _close_citation_link_coverage_gap(text, [], [], {}, None, provenance)
    )

    assert new_links == []
    assert retry_failed is True
    assert new_uncited == [
        {"paragraph_index": 0, "sentence": gap_sentence, "tag": "finding", "unclassified": True}
    ]
    # No retry provenance to fold in (the call itself failed), so total_calls is
    # unchanged from before this coverage step ran.
    assert new_provenance["total_calls"] == 1


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_makes_at_most_one_retry_call(monkeypatch):
    text = "First orphan sentence. Second orphan sentence. Third orphan sentence."
    calls = []

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        calls.append(content)
        return CitationLinkResult(links=[], provenance=_fake_provenance(), uncited_sentences=[])

    monkeypatch.setattr(writing, "link_citations", fake_link_citations)

    await _close_citation_link_coverage_gap(text, [], [], {}, None, {"total_calls": 1})

    assert len(calls) == 1
