"""Citation link validation V1-V6.

Every test here is a pure-function test against ``validate_citation_link`` /
``validate_citation_links`` and a stubbed agent for ``link_citations`` -- no network, no
database.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.citation_link_agent import (
    CitationLink,
    CitationLinkMap,
    CitationLinkResult,
    UncitedSentence,
    build_link_prompt,
    link_citations,
    split_paragraphs,
    validate_citation_link,
    validate_citation_links,
    validate_uncited_sentence,
    validate_uncited_sentences,
)

CONTENT = (
    "Tutoring improves outcomes (Smith, 2020). Brown et al. (2018) concur.\n\n"
    "Some claim otherwise (Nguyen, 2021)."
)


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_split_paragraphs_drops_blank_blocks_but_does_not_strip_kept_ones():
    content = "  P1 with leading space\n\nP2\n\n\n\nP3"
    assert split_paragraphs(content) == ["  P1 with leading space", "P2", "P3"]


def test_v1_rejects_a_sentence_not_present_in_the_generated_text():
    """An invented sentence never survives, regardless of how plausible it looks."""
    link = CitationLink(
        paragraph_index=0,
        sentence="This sentence was never written by the model (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
    )
    assert validate_citation_link(link, _norm(CONTENT)) is None


def test_v2_rejects_a_citation_text_not_inside_the_claimed_sentence():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["nguyen_2021"],
        citation_text="(Nguyen, 2021)",  # really in the next sentence, not this one
    )
    assert validate_citation_link(link, _norm(CONTENT)) is None


def test_v3_rejects_a_link_on_a_sentence_the_audit_finds_no_real_citation_in():
    """citation_audit.citation_spans is authoritative: an invented citation on a real,
    citation-free sentence is dropped even though V1/V2 alone could not catch it."""
    content = "This sentence has no citation at all. " + CONTENT
    link = CitationLink(
        paragraph_index=0,
        sentence="This sentence has no citation at all.",
        keys=["ghost_1999"],
        citation_text="no citation at all",
    )
    assert validate_citation_link(link, _norm(content)) is None


def test_v4_keeps_a_key_absent_from_the_library_as_unresolved_not_dropped():
    """Validation never consults a paper library; a key merely absent from one project's
    papers is not a disagreement and is kept (`no_full_text` handles it later)."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result == {
        "paragraph_index": 0,
        "sentence": "Tutoring improves outcomes (Smith, 2020).",
        "keys": ["smith_2020"],
        "citation_text": "(Smith, 2020)",
        # No proposition was given, so the link's own sentence is used.
        "proposition": "Tutoring improves outcomes (Smith, 2020).",
        "evidence_ids": [],
    }


def test_v5_drops_a_key_whose_surname_or_year_disagrees_with_the_citation_text():
    """The model claimed the citation was Jones/2019, but the citation_text it also
    reported plainly reads Smith, 2020 -- the key is dropped, not the whole link."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["jones_2019"],
        citation_text="(Smith, 2020)",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result == {
        "paragraph_index": 0,
        "sentence": "Tutoring improves outcomes (Smith, 2020).",
        "keys": [],
        "citation_text": "(Smith, 2020)",
        "proposition": "Tutoring improves outcomes (Smith, 2020).",
        "evidence_ids": [],
    }


def test_v5_keeps_the_agreeing_key_and_drops_only_the_disagreeing_one():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020", "jones_2019"],
        citation_text="(Smith, 2020)",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result["keys"] == ["smith_2020"]


def test_a_valid_proposition_is_kept_verbatim():
    """``proposition`` is the citation link's own narrowed span of its
    sentence -- kept exactly as the model wrote it once it is confirmed to be a real,
    contiguous substring of that sentence (checked whitespace-insensitively)."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes for most students (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        proposition="Tutoring improves outcomes for most students",
    )
    result = validate_citation_link(link, _norm(CONTENT.replace(
        "Tutoring improves outcomes (Smith, 2020).",
        "Tutoring improves outcomes for most students (Smith, 2020).",
    )))
    assert result["proposition"] == "Tutoring improves outcomes for most students"


def test_a_proposition_not_found_in_the_sentence_falls_back_to_the_sentence():
    """A hallucinated or paraphrased proposition -- one that is not an actual substring
    of the sentence it is attached to -- is exactly as unusable as no proposition at
    all, so the sentence is used instead (never an invented span)."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        proposition="This text never appears in the sentence at all",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result["proposition"] == "Tutoring improves outcomes (Smith, 2020)."


def test_a_whitespace_only_proposition_falls_back_to_the_sentence():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        proposition="   ",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result["proposition"] == "Tutoring improves outcomes (Smith, 2020)."


def test_a_proposition_is_matched_after_whitespace_normalisation():
    """The occurrence check tolerates a whitespace difference between the model's
    ``proposition`` and the sentence it is checked against, but the stored value is the
    proposition exactly as given, stripped of only its own leading/trailing whitespace."""
    content = "Tutoring  improves outcomes (Smith, 2020). Brown et al. (2018) concur."
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring  improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        proposition="  Tutoring improves outcomes  ",
    )
    result = validate_citation_link(link, _norm(content))
    assert result["proposition"] == "Tutoring improves outcomes"


def test_a_span_with_no_link_survives_into_the_fallback():
    """`validate_citation_links` only ever removes links; it never invents one for a real
    citation the model's map omitted. The extractor is what falls back
    to the regexes for that sentence -- this module just confirms nothing here manufactures
    a link to paper over the gap."""
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence="Tutoring improves outcomes (Smith, 2020).",
                keys=["smith_2020"],
                citation_text="(Smith, 2020)",
            )
        ]
    )
    # "Brown et al. (2018) concur." has a real citation the audit would find, but the
    # model's map never mentioned it -- validate_citation_links neither drops the first
    # link nor fabricates a second one.
    validated = validate_citation_links(link_map, CONTENT)
    assert len(validated) == 1
    assert validated[0]["sentence"] == "Tutoring improves outcomes (Smith, 2020)."


# --- Fix B: snap the validated sentence to finalize's own splitter fragment -----------


def test_the_snap_replaces_a_linker_reported_merged_sentence_with_its_own_fragment():
    """Real shape: the linker reports TWO of
    finalize's own sentence fragments as one ``sentence``. Snapped to the fragment that
    actually contains this link's own ``citation_text``, so the validated link is keyed
    on exactly the string `_finalize_paragraph_text` will look it up by -- not the
    model's own, wider copy of it."""
    cited = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    framing = "Learner preferences point in a compatible direction."
    content = (
        "Opening framing paragraph with no citation of its own.\n\n%s %s" % (cited, framing)
    )
    link = CitationLink(
        paragraph_index=1,
        sentence="%s %s" % (cited, framing),
        keys=["liu_2019"],
        citation_text="(Liu, 2019)",
    )
    result = validate_citation_link(link, _norm(content), content=content)
    assert result["sentence"] == cited


def test_the_snap_replaces_an_abbreviation_split_sentence_with_its_own_fragment():
    """Real shape: finalize's splitter cuts on "e.g." (not in its
    abbreviation allow-list), so a sentence the linker reported whole is actually two
    fragments; the link is snapped to the one carrying the citation."""
    sentence = (
        "Coded feedback narrows the options available to a learner, e.g. tense and "
        "article errors, and still outperformed no feedback (Mao, 2024)."
    )
    content = "Opening framing paragraph with no citation of its own.\n\n%s" % sentence
    link = CitationLink(
        paragraph_index=1,
        sentence=sentence,
        keys=["mao_2024"],
        citation_text="(Mao, 2024)",
    )
    result = validate_citation_link(link, _norm(content), content=content)
    assert result["sentence"] == (
        "tense and article errors, and still outperformed no feedback (Mao, 2024)."
    )


def test_the_snap_is_the_identity_on_a_sentence_that_is_already_a_real_fragment():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
    )
    result = validate_citation_link(link, _norm(CONTENT), content=CONTENT)
    assert result["sentence"] == "Tutoring improves outcomes (Smith, 2020)."


def test_no_content_given_skips_the_snap():
    """A caller that does not pass *content* (every pure V1-V6 test above) gets the
    model's own sentence back unchanged -- the snap is opt-in, not a silent behaviour
    change for a caller that never had the raw text to snap against."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result["sentence"] == "Tutoring improves outcomes (Smith, 2020)."


def test_the_snapped_sentence_also_narrows_the_proposition_fallback():
    """The proposition occurrence check runs against the SNAPPED sentence, not the
    model's wider one -- a proposition that is a real span of the model's merged
    sentence but happens to also occur inside the snapped fragment is kept; the
    fallback, when needed, is the snapped fragment, not the model's original string."""
    cited = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    framing = "Learner preferences point in a compatible direction."
    content = (
        "Opening framing paragraph with no citation of its own.\n\n%s %s" % (cited, framing)
    )
    link = CitationLink(
        paragraph_index=1,
        sentence="%s %s" % (cited, framing),
        keys=["liu_2019"],
        citation_text="(Liu, 2019)",
        proposition="Learner preferences point in a compatible direction.",
    )
    result = validate_citation_link(link, _norm(content), content=content)
    # The given proposition is not a span of the snapped sentence (it belongs to the
    # OTHER, dropped half of the model's merged string), so it falls back to the
    # snapped sentence itself, never the model's wider one.
    assert result["proposition"] == cited


def test_validate_citation_links_snaps_every_link_against_the_real_content():
    """End-to-end through the batch validator (what `link_citations` actually calls):
    the snap fires without any extra wiring at the call site."""
    cited = "Direct correction outperformed coded feedback on accuracy (Liu, 2019)."
    framing = "Learner preferences point in a compatible direction."
    content = (
        "Opening framing paragraph with no citation of its own.\n\n%s %s" % (cited, framing)
    )
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=1,
                sentence="%s %s" % (cited, framing),
                keys=["liu_2019"],
                citation_text="(Liu, 2019)",
            )
        ]
    )
    validated = validate_citation_links(link_map, content)
    assert len(validated) == 1
    assert validated[0]["sentence"] == cited


# --- Fix A companion: a link V3 rejects is recorded as an uncited finding -------------


def test_validate_citation_links_records_a_v3_rejection_as_an_uncited_finding():
    """A link the model reported on a real sentence, with a citation_text that really
    occurs inside it, but that ``citation_spans`` itself finds no real citation in
    (V3) is the one place this module knows the model named a genuine sentence AND the
    audit found no citation in it -- the definition of an uncited finding. Recorded
    here instead of discarded, so a paid retry is not spent rediscovering it."""
    content = "This sentence has no citation at all. " + CONTENT
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence="This sentence has no citation at all.",
                keys=["ghost_1999"],
                citation_text="no citation at all",
            )
        ]
    )
    uncited_from_rejected: list[dict] = []
    validated = validate_citation_links(
        link_map, content, uncited_from_rejected=uncited_from_rejected
    )
    assert validated == []
    assert uncited_from_rejected == [
        {
            "paragraph_index": 0,
            "sentence": "This sentence has no citation at all.",
            "tag": "finding",
        }
    ]


def test_validate_citation_links_does_not_record_a_v1_rejection_as_an_uncited_finding():
    """An invented sentence (V1) is not a real sentence of the text at all, so it is
    dropped silently, not recorded as an uncited finding -- only a real sentence V3
    alone rejects qualifies."""
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence="This sentence was never written by the model (Smith, 2020).",
                keys=["smith_2020"],
                citation_text="(Smith, 2020)",
            )
        ]
    )
    uncited_from_rejected: list[dict] = []
    validated = validate_citation_links(
        link_map, CONTENT, uncited_from_rejected=uncited_from_rejected
    )
    assert validated == []
    assert uncited_from_rejected == []


def test_validate_citation_links_uncited_from_rejected_defaults_to_none_and_is_optional():
    """The default (no list given) preserves every existing caller's own behaviour:
    nothing is recorded, nothing raises."""
    content = "This sentence has no citation at all. " + CONTENT
    link_map = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence="This sentence has no citation at all.",
                keys=["ghost_1999"],
                citation_text="no citation at all",
            )
        ]
    )
    assert validate_citation_links(link_map, content) == []


def test_build_link_prompt_lists_paragraphs_and_keys():
    prompt = build_link_prompt(CONTENT, ["smith_2020", "brown_2018"])
    assert "[Paragraph 0]" in prompt
    assert "[Paragraph 1]" in prompt
    assert "smith_2020" in prompt and "brown_2018" in prompt


def test_build_link_prompt_tolerates_no_papers():
    prompt = build_link_prompt(CONTENT, [])
    assert "no papers in the library" in prompt


def _fake_link_run_result(links: list[CitationLink], *, input_tokens=910, output_tokens=140):
    response = SimpleNamespace(
        model_name="deepseek-v4-flash",
        provider_response_id="resp-link-1",
        provider_details={"system_fingerprint": "fp_link"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        output=CitationLinkMap(links=links), response=response, usage=lambda: usage
    )


@pytest.mark.asyncio
async def test_link_citations_runs_the_agent_and_validates_its_output():
    fake_result = _fake_link_run_result(
        [
            CitationLink(
                paragraph_index=0,
                sentence="Tutoring improves outcomes (Smith, 2020).",
                keys=["smith_2020"],
                citation_text="(Smith, 2020)",
            ),
            CitationLink(
                paragraph_index=0,
                sentence="Invented sentence never in the text (Ghost, 1999).",
                keys=["ghost_1999"],
                citation_text="(Ghost, 1999)",
            ),
        ]
    )

    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch(
        "app.agents.citation_link_agent.get_citation_link_agent", return_value=fake_agent
    ):
        result = await link_citations(CONTENT, ["smith_2020"])

    assert isinstance(result, CitationLinkResult)
    assert result.links == [
        {
            "paragraph_index": 0,
            "sentence": "Tutoring improves outcomes (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
            "proposition": "Tutoring improves outcomes (Smith, 2020).",
            "evidence_ids": [],
            # Deterministic, no-model-call coverage/meta-evaluation flags added by
            # `_flag_sentence_coverage`, off the model's own structured-output schema.
            "coverage_incomplete": False,
            "residue_spans": [],
            "meta_evaluation": False,
        }
    ]
    assert result.uncited_sentences == []
    fake_agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_link_citations_records_the_calls_own_provenance_and_usage():
    """The linker's call is a real,
    billed DeepSeek request, so its own model, tokens and prompt version must be
    recoverable exactly like every other agent's, not silently absent."""
    fake_result = _fake_link_run_result([], input_tokens=910, output_tokens=140)

    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch(
        "app.agents.citation_link_agent.get_citation_link_agent", return_value=fake_agent
    ):
        result = await link_citations(CONTENT, ["smith_2020"])

    assert result.provenance.agent == "citation_link"
    assert result.provenance.model_reported == "deepseek-v4-flash"
    assert result.provenance.provider_response_id == "resp-link-1"
    assert result.provenance.system_fingerprint == "fp_link"
    assert result.provenance.input_tokens == 910
    assert result.provenance.output_tokens == 140
    assert result.provenance.prompt_version


def test_the_prompt_asks_for_a_proposition_span_not_the_whole_sentence():
    """The linker's own instructions must ask for the narrower span, not
    rely on the model volunteering it -- this changes ``CITATION_LINK_PROMPT``'s content
    (and therefore its recorded ``prompt_version``, a sha256 of the text itself)."""
    from app.agents.citation_link_agent import CITATION_LINK_PROMPT

    assert "proposition" in CITATION_LINK_PROMPT
    assert "verbatim" in CITATION_LINK_PROMPT


# --------------------------------------------------------------------------------------
# evidence_ids (V6) and uncited-sentence tagging.
# --------------------------------------------------------------------------------------


def test_the_prompt_asks_for_evidence_ids_and_uncited_sentences():
    from app.agents.citation_link_agent import CITATION_LINK_PROMPT

    assert "evidence_ids" in CITATION_LINK_PROMPT
    assert "EVIDENCE CATALOG" in CITATION_LINK_PROMPT
    assert "framing" in CITATION_LINK_PROMPT
    assert "finding" in CITATION_LINK_PROMPT


def test_build_link_prompt_lists_the_evidence_catalog():
    catalog = [{"id": "e1", "quote": "Tutoring improved outcomes.", "key": "smith_2020"}]
    prompt = build_link_prompt(CONTENT, ["smith_2020"], catalog)
    assert "[e1] (smith_2020)" in prompt
    assert "Tutoring improved outcomes." in prompt


def test_build_link_prompt_tolerates_no_evidence_catalog():
    prompt = build_link_prompt(CONTENT, ["smith_2020"])
    assert "(none given)" in prompt


def test_v6_keeps_an_evidence_id_whose_paper_matches_a_kept_key():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        evidence_ids=["e1"],
    )
    result = validate_citation_link(link, _norm(CONTENT), {"e1": "smith_2020"})
    assert result["evidence_ids"] == ["e1"]


def test_v6_drops_an_evidence_id_belonging_to_a_different_paper():
    """The model attached an id from Nguyen's evidence to a Smith citation -- V6 drops
    it, since restating another paper's evidence proves nothing about this one."""
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        evidence_ids=["e9"],
    )
    result = validate_citation_link(link, _norm(CONTENT), {"e9": "nguyen_2021"})
    assert result["evidence_ids"] == []


def test_v6_drops_an_unknown_evidence_id():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        evidence_ids=["e404"],
    )
    result = validate_citation_link(link, _norm(CONTENT), {"e1": "smith_2020"})
    assert result["evidence_ids"] == []


def test_v6_defaults_to_no_evidence_ids_when_no_catalog_was_shown():
    link = CitationLink(
        paragraph_index=0,
        sentence="Tutoring improves outcomes (Smith, 2020).",
        keys=["smith_2020"],
        citation_text="(Smith, 2020)",
        evidence_ids=["e1"],
    )
    result = validate_citation_link(link, _norm(CONTENT))
    assert result["evidence_ids"] == []


def test_an_uncited_sentence_present_in_the_text_and_really_uncited_is_kept():
    content = "This section reviews tutoring research. " + CONTENT
    sentence = UncitedSentence(
        paragraph_index=0, sentence="This section reviews tutoring research.", tag="framing"
    )
    result = validate_uncited_sentence(sentence, _norm(content))
    assert result == {
        "paragraph_index": 0,
        "sentence": "This section reviews tutoring research.",
        "tag": "framing",
    }


def test_an_uncited_sentence_not_in_the_text_is_dropped():
    sentence = UncitedSentence(paragraph_index=0, sentence="Invented sentence.", tag="framing")
    assert validate_uncited_sentence(sentence, _norm(CONTENT)) is None


def test_an_uncited_sentence_that_actually_carries_a_citation_is_dropped():
    """Defence against the model mislabelling a cited sentence as uncited."""
    sentence = UncitedSentence(
        paragraph_index=0, sentence="Tutoring improves outcomes (Smith, 2020).", tag="finding"
    )
    assert validate_uncited_sentence(sentence, _norm(CONTENT)) is None


def test_an_invalid_tag_defaults_to_finding():
    content = "This is a genuinely uncited sentence with no citation at all."
    sentence = UncitedSentence(
        paragraph_index=0, sentence=content, tag="not-a-real-tag"
    )
    result = validate_uncited_sentence(sentence, _norm(content))
    assert result["tag"] == "finding"


def test_validate_uncited_sentences_filters_the_whole_list():
    link_map = CitationLinkMap(
        links=[],
        uncited_sentences=[
            UncitedSentence(paragraph_index=0, sentence="Invented.", tag="framing"),
            UncitedSentence(
                paragraph_index=0,
                sentence="This section reviews tutoring research.",
                tag="framing",
            ),
        ],
    )
    content = "This section reviews tutoring research. " + CONTENT
    validated = validate_uncited_sentences(link_map, content)
    assert validated == [
        {
            "paragraph_index": 0,
            "sentence": "This section reviews tutoring research.",
            "tag": "framing",
        }
    ]


@pytest.mark.asyncio
async def test_link_citations_passes_the_evidence_catalog_and_returns_uncited_sentences():
    fake_result = _fake_link_run_result([])
    fake_result.output = CitationLinkMap(
        links=[
            CitationLink(
                paragraph_index=0,
                sentence="Tutoring improves outcomes (Smith, 2020).",
                keys=["smith_2020"],
                citation_text="(Smith, 2020)",
                evidence_ids=["e1"],
            )
        ],
        uncited_sentences=[
            UncitedSentence(
                paragraph_index=0,
                sentence="This section reviews tutoring research.",
                tag="framing",
            )
        ],
    )
    catalog = [{"id": "e1", "quote": "Tutoring improved outcomes.", "key": "smith_2020"}]
    content = "This section reviews tutoring research. " + CONTENT

    fake_agent = type("A", (), {"run": AsyncMock(return_value=fake_result)})()
    with patch(
        "app.agents.citation_link_agent.get_citation_link_agent", return_value=fake_agent
    ):
        result = await link_citations(content, ["smith_2020"], catalog)

    assert result.links[0]["evidence_ids"] == ["e1"]
    assert result.uncited_sentences == [
        {
            "paragraph_index": 0,
            "sentence": "This section reviews tutoring research.",
            "tag": "framing",
        }
    ]
