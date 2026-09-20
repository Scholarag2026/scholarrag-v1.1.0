"""Tests for the citation-link coverage rule: a sentence the citation-link call
returns in neither ``links`` nor ``uncited_sentences`` would otherwise be invisible both
to the verification gate and to finalize's own uncited-"finding" removal rule, so it
could reach the delivered text with no citation, no verification row, and no record that
it was never checked. ``unclassified_body_sentences`` names every such gap;
``resolve_unclassified_sentences`` sends every gap back to the linker together, once;
whatever it still cannot classify is marked ``"unclassified": True`` and KEPT by
``_finalize_paragraph_text`` under its own ``sentences_unclassified_kept`` counter: a gap
the retry never actually, positively classified must never be removed. Removal is
reserved for a gap the retry classified as an uncited finding, either on the first pass
or its own bounded retry.

The last two tests in this file are regression tests built directly from a promoted
demo run's own ``draft_content.json`` and ``writing_result.json``
(`demo/output/20260911-171918/`): one exercises the standalone action's own coverage
step (`_close_citation_link_coverage_gap`, over the saved draft's paragraph node,
`draft_content.json`), the other exercises the write job's own gated-loop path
(`resolve_unclassified_sentences` folded into `finalize_generated_section`, over the
whole section's own flat link/uncited lists, `writing_result.json`) -- the two places
the coverage rule closes the same gap. Both show the orphaned sentences, unresolved even
by a worst-case retry failure, now KEPT rather than removed.
"""

import json
import os
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

import pytest  # noqa: E402

from app.agents import citation_link_agent  # noqa: E402
from app.agents.citation_link_agent import CitationLinkResult  # noqa: E402
from app.schemas.provenance import LLMCallProvenance  # noqa: E402
from app.services.fulltext import (  # noqa: E402
    _close_citation_link_coverage_gap,
    _finalize_paragraph_text,
    finalize_draft_document,
    finalize_generated_section,
    resolve_unclassified_sentences,
    unclassified_body_sentences,
)
from tests.conftest import skip_unless_run_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_RUN = REPO_ROOT / "demo" / "output" / "20260911-171918"


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


# --- unclassified_body_sentences ------------------------------------------------------


def test_unclassified_body_sentences_finds_a_sentence_in_neither_list():
    text = "Smith (2020) reported gains. This claim carries no citation at all."
    links = [{"sentence": "Smith (2020) reported gains.", "keys": ["smith_2020"]}]
    gaps = unclassified_body_sentences(text, links, [])
    assert [g["sentence"] for g in gaps] == ["This claim carries no citation at all."]
    assert gaps[0]["paragraph_index"] == 0


def test_unclassified_body_sentences_treats_an_uncited_entry_as_covered():
    text = "Smith (2020) reported gains. Framing sentence with no citation."
    links = [{"sentence": "Smith (2020) reported gains.", "keys": ["smith_2020"]}]
    uncited = [{"sentence": "Framing sentence with no citation.", "tag": "framing"}]
    assert unclassified_body_sentences(text, links, uncited) == []


def test_unclassified_body_sentences_skips_heading_blocks():
    text = "# Literature Review\n\nAn uncited body sentence here."
    gaps = unclassified_body_sentences(text, [], [])
    assert [g["sentence"] for g in gaps] == ["An uncited body sentence here."]


def test_unclassified_body_sentences_returns_empty_when_fully_covered():
    text = "Smith (2020) reported gains."
    links = [{"sentence": "Smith (2020) reported gains.", "keys": ["smith_2020"]}]
    assert unclassified_body_sentences(text, links, []) == []


def test_unclassified_body_sentences_reports_a_repeated_sentence_once():
    text = "An uncited sentence repeats. An uncited sentence repeats."
    gaps = unclassified_body_sentences(text, [], [])
    assert len(gaps) == 1
    assert gaps[0]["paragraph_index"] == 0


def test_unclassified_body_sentences_does_not_treat_a_stale_unclassified_entry_as_covered():
    """A prior pass's own ``"unclassified": True`` placeholder
    records that no model call ever positively classified the sentence -- it must not
    count as coverage, or the sentence is frozen unclassified forever the moment one
    retry call happens to fail, with no later heal ever trying it again."""
    text = "An orphaned empirical claim with no citation and no tag."
    uncited = [{"sentence": text, "tag": "finding", "unclassified": True}]
    gaps = unclassified_body_sentences(text, [], uncited)
    assert [g["sentence"] for g in gaps] == [text]


def test_unclassified_body_sentences_treats_a_split_fragment_of_a_covered_sentence_as_covered():
    """Real shape, ``demo/output/20260909-043006``: an abbreviation
    outside `_ABBREVIATIONS_NOT_SENTENCE_FINAL` ("U.S.") splits a sentence the linker
    reported whole into two fragments; neither fragment equals the linker's own
    ``sentence`` exactly, but each is a substring of it, so neither is a gap."""
    whole_sentence = (
        "Surveying 70 students and 16 teachers in a U.S. intensive English program, "
        "Liu and Wu (2019) found that most preferred direct correction."
    )
    text = whole_sentence
    links = [{"sentence": whole_sentence, "keys": ["liu_2019"]}]
    assert unclassified_body_sentences(text, links, []) == []


# --- resolve_unclassified_sentences ---------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_makes_no_call_with_no_gaps():
    text = "Smith (2020) reported gains."
    links = [{"sentence": text, "keys": ["smith_2020"]}]

    calls = []

    async def linker(retry_text):
        calls.append(retry_text)
        raise AssertionError("should not be called")

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, links, [], linker)
    )
    assert new_links == links
    assert new_links is not links  # a fresh copy, per the docstring
    assert new_uncited == []
    assert provenance is None
    assert calls_made == 0
    assert retry_failed is False
    assert calls == []


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_resolves_a_gap_into_a_link():
    gap = "Mixed-effect models showed a durable four-week advantage."
    text = f"Jones (2019) confirmed prior work. {gap}"
    links = [{"sentence": "Jones (2019) confirmed prior work.", "keys": ["jones_2019"]}]

    async def linker(retry_text):
        assert retry_text == gap
        return CitationLinkResult(
            links=[{"sentence": gap, "keys": ["lee_2021"], "citation_text": "(Lee, 2021)"}],
            provenance=_fake_provenance(),
            uncited_sentences=[],
        )

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, links, [], linker)
    )
    assert calls_made == 1
    assert provenance is not None
    assert retry_failed is False
    resolved = [link for link in new_links if link["sentence"] == gap]
    assert len(resolved) == 1
    assert resolved[0]["keys"] == ["lee_2021"]
    assert resolved[0]["paragraph_index"] == 0
    assert new_uncited == []


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_resolves_a_gap_into_an_uncited_entry():
    gap = "This framing sentence carries no citation at all."
    text = gap

    async def linker(retry_text):
        return CitationLinkResult(
            links=[],
            provenance=_fake_provenance(),
            uncited_sentences=[{"sentence": gap, "tag": "framing"}],
        )

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, [], [], linker)
    )
    assert calls_made == 1
    assert retry_failed is False
    assert new_links == []
    assert len(new_uncited) == 1
    assert new_uncited[0]["tag"] == "framing"
    assert "unclassified" not in new_uncited[0]


@pytest.mark.asyncio
async def test_resolve_unclassified_marks_a_removable_finding_on_retry_answer():
    """The retry call itself succeeded -- the model saw the gap's own miniature text
    and chose not to report it as a link, a framing sentence, or a finding. That is
    a positive (if unhelpful) answer, not a transient failure, so the synthetic entry
    is an ordinary, removable "finding" with no ``"unclassified"`` flag: rule 1 of
    ``_finalize_paragraph_text`` removes it and counts it under
    ``sentences_removed_uncited_finding``, where a reader can see it."""
    gap = "Cognitive load was lower for one feedback type on grammatical errors."
    text = gap

    async def linker(retry_text):
        return CitationLinkResult(links=[], provenance=_fake_provenance(), uncited_sentences=[])

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, [], [], linker)
    )
    assert calls_made == 1
    assert retry_failed is False
    assert new_links == []
    assert new_uncited == [{"paragraph_index": 0, "sentence": gap, "tag": "finding"}]


@pytest.mark.asyncio
async def test_resolve_unclassified_keeps_unclassified_only_on_retry_call_failure():
    """The transient-failure guarantee (design amendment A4) is untouched: when the
    retry call itself raises, nothing about the gap was ever actually looked at, so
    the synthetic entry still carries ``"unclassified": True`` and is kept, not
    removed."""
    gap = "Cognitive load was lower for one feedback type on grammatical errors."
    text = gap

    async def linker(retry_text):
        raise RuntimeError("provider outage")

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, [], [], linker)
    )
    assert calls_made == 1
    assert retry_failed is True
    assert new_links == []
    assert new_uncited == [
        {"paragraph_index": 0, "sentence": gap, "tag": "finding", "unclassified": True}
    ]


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_replaces_a_stale_unclassified_entry_on_success():
    """A sentence a PRIOR pass's own retry could not classify is
    retried again here, and once this pass's own retry succeeds, the fresh classification
    REPLACES the stale ``"unclassified": True`` placeholder -- it is not left in
    ``uncited_sentences`` alongside the new entry."""
    gap = "Mixed-effect models showed a durable advantage four weeks later."
    text = gap
    stale_uncited = [{"sentence": gap, "tag": "finding", "unclassified": True}]

    async def linker(retry_text):
        assert retry_text == gap
        return CitationLinkResult(
            links=[], provenance=_fake_provenance(),
            uncited_sentences=[{"sentence": gap, "tag": "framing"}],
        )

    _links, new_uncited, _prov, calls_made, retry_failed = await resolve_unclassified_sentences(
        text, [], stale_uncited, linker
    )
    assert calls_made == 1
    assert retry_failed is False
    assert new_uncited == [{"sentence": gap, "tag": "framing", "paragraph_index": 0}]


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_replaces_a_stale_unclassified_entry_on_retry_fail():
    """A sentence retried again after a prior failure, whose retry fails AGAIN, keeps
    exactly one ``"unclassified": True`` entry -- the stale one from the earlier pass is
    dropped, and one fresh one (this pass's own bounded attempt) takes its place, never
    both at once."""
    gap = "A sentence two consecutive retries both fail to classify."
    text = gap
    stale_uncited = [{"sentence": gap, "tag": "finding", "unclassified": True}]

    async def linker(retry_text):
        raise RuntimeError("provider outage, again")

    _links, new_uncited, _prov, calls_made, retry_failed = await resolve_unclassified_sentences(
        text, [], stale_uncited, linker
    )
    assert calls_made == 1
    assert retry_failed is True
    assert new_uncited == [
        {"paragraph_index": 0, "sentence": gap, "tag": "finding", "unclassified": True}
    ]


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_survives_a_failed_retry_call():
    gap = "A sentence the retry call itself never reaches."
    text = gap

    async def linker(retry_text):
        raise RuntimeError("provider outage")

    new_links, new_uncited, provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(text, [], [], linker)
    )
    assert calls_made == 1
    assert provenance is None
    assert retry_failed is True
    assert new_uncited == [
        {"paragraph_index": 0, "sentence": gap, "tag": "finding", "unclassified": True}
    ]


@pytest.mark.asyncio
async def test_resolve_unclassified_sentences_makes_at_most_one_call_for_several_gaps():
    text = "First orphan claim here. Second orphan claim here. Third orphan claim here."
    calls = []

    async def linker(retry_text):
        calls.append(retry_text)
        return CitationLinkResult(links=[], provenance=_fake_provenance(), uncited_sentences=[])

    _links, uncited, _prov, calls_made, retry_failed = await resolve_unclassified_sentences(
        text, [], [], linker
    )
    assert calls_made == 1
    assert retry_failed is False
    assert len(calls) == 1
    assert len(uncited) == 3
    # The retry call itself succeeded; it simply never named any of the three gaps, so
    # each becomes an ordinary, removable finding, not a transient-failure placeholder.
    assert all("unclassified" not in entry for entry in uncited)
    assert all(entry["tag"] == "finding" for entry in uncited)


# --- _finalize_paragraph_text: sentences_unclassified_kept -----------------------------


def test_finalize_paragraph_text_keeps_an_unclassified_finding_and_counts_it_separately():
    """A gap `resolve_unclassified_sentences` marked
    ``unclassified`` -- its own bounded retry never positively classified it -- is kept,
    not removed, and counted under its own statistic, exactly like a framing sentence."""
    orphan = "An orphaned empirical claim with no citation and no tag."
    text = f"Smith (2020) reported gains. {orphan}"
    links = [{"sentence": "Smith (2020) reported gains.", "keys": ["smith_2020"]}]
    uncited = [{"sentence": orphan, "tag": "finding", "unclassified": True}]
    claim_status = {
        ("Smith (2020) reported gains.", "Smith (2020) reported gains.", "smith_2020"): "verified"
    }

    new_text, surviving_links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        text, links, uncited, claim_status
    )

    assert orphan in new_text
    assert "Smith (2020) reported gains." in new_text
    assert stats["sentences_unclassified_kept"] == 1
    assert stats["sentences_removed_uncited_finding"] == 0
    assert surviving_uncited == [{"sentence": orphan, "tag": "finding", "unclassified": True}]


def test_finalize_paragraph_text_removes_an_ordinary_finding_the_retry_actually_classified():
    ordinary_finding = "A finding sentence the linker itself tagged, not a coverage gap."
    text = ordinary_finding
    uncited = [{"sentence": ordinary_finding, "tag": "finding"}]

    new_text, _links, stats, _healed, surviving_uncited = _finalize_paragraph_text(
        text, [], uncited, {}
    )

    assert ordinary_finding not in new_text
    assert stats["sentences_removed_uncited_finding"] == 1
    assert stats["sentences_unclassified_kept"] == 0
    assert surviving_uncited == []


# --- Regression: the promoted run's own orphaned sentences --------------------------


def _load_promoted_third_paragraph_node() -> dict:
    run_dir = skip_unless_run_dir(PROMOTED_RUN)
    draft_content = json.loads(
        (run_dir / "draft_content.json").read_text(encoding="utf-8")
    )
    # content[3] is the paragraph that opens with the orphaned "Mixed-effect linear
    # models ..." sentence, carries one real citation link (Liu and Wu, 2019), and
    # closes with a second, equally orphaned limitation sentence about that same
    # study's own survey.
    return draft_content["content"][3]


def test_promoted_run_paragraph_carries_the_orphaned_sentences_this_fix_closes():
    """Sanity check on the fixture itself: confirms the real artefact still has the
    exact orphaned-sentence shape described above, so the regression test
    below is not accidentally checking a paragraph that no longer reproduces the bug."""
    node = _load_promoted_third_paragraph_node()
    text = node["content"][0]["text"]
    links = node["attrs"]["citationLinks"]
    uncited = node["attrs"].get("uncitedSentences") or []

    gaps = unclassified_body_sentences(text, links, uncited)
    gap_sentences = {g["sentence"] for g in gaps}
    assert (
        "Mixed-effect linear models showed that both direct corrections and codes "
        "enhanced immediate accuracy during revision, but only direct corrections "
        "produced a durable advantage four weeks later; cognitive load was lower for "
        "direct corrections on grammatical errors." in gap_sentences
    )
    assert (
        "Their survey of 70 students and 16 teachers was not designed to project "
        "population trends, and the teacher questionnaire did not allow responses to "
        "vary by student proficiency." in gap_sentences
    )
    assert len(gaps) == 2


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_keeps_orphaned_sentences_on_retry_failure(
    monkeypatch,
):
    """A worst-case retry failure (a fresh provider outage,
    or a model that still misses both orphaned sentences) must never delete them -- it
    records them as unresolved and leaves them in the delivered text, so the offline
    checker can flag the gap for a human instead of the pipeline silently destroying
    real content."""
    node = _load_promoted_third_paragraph_node()
    draft_content = {"type": "doc", "content": [dict(node)]}
    liu_wu_sentence = node["attrs"]["citationLinks"][0]["sentence"]

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        return CitationLinkResult(links=[], provenance=_fake_provenance(), uncited_sentences=[])

    monkeypatch.setattr(citation_link_agent, "link_citations", fake_link_citations)

    await _close_citation_link_coverage_gap(draft_content, {"liu_2019": None}, uuid4())

    healed_attrs = draft_content["content"][0]["attrs"]
    unclassified_sentences = {
        entry["sentence"] for entry in healed_attrs["uncitedSentences"]
    }
    assert liu_wu_sentence not in unclassified_sentences
    assert len(healed_attrs["uncitedSentences"]) == 2
    assert all(entry["unclassified"] for entry in healed_attrs["uncitedSentences"])

    claims = [
        (liu_wu_sentence, "liu_2019", "Liu and Wu (2019)", liu_wu_sentence),
    ]
    claim_status = {(liu_wu_sentence, liu_wu_sentence, "liu_2019"): "verified"}

    result = finalize_draft_document(draft_content, claims, claim_status)

    final_text = "".join(
        n.get("text", "")
        for node in result.content["content"]
        for n in node.get("content") or []
        if n.get("type") == "text"
    )
    assert "Mixed-effect linear models" in final_text
    assert "Their survey of 70 students and 16 teachers" in final_text
    assert liu_wu_sentence in final_text
    assert result.stats["sentences_unclassified_kept"] == 2
    assert result.content["content"][0]["attrs"]["uncitedSentences"] == (
        healed_attrs["uncitedSentences"]
    )


@pytest.mark.asyncio
async def test_close_citation_link_coverage_gap_replaces_a_stale_unclassified_entry_on_success(
    monkeypatch,
):
    """A paragraph node already carrying a stale
    ``"unclassified": True`` placeholder from an earlier call to this same action is
    retried again, and a successful retry this time REPLACES that placeholder rather
    than leaving it beside the fresh classification."""
    sentence = "A sentence one earlier heal could not classify."
    draft_content = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": sentence}],
                "attrs": {
                    "uncitedSentences": [
                        {
                            "paragraph_index": 0,
                            "sentence": sentence,
                            "tag": "finding",
                            "unclassified": True,
                        }
                    ]
                },
            }
        ],
    }

    async def fake_link_citations(content, paper_keys, evidence_catalog=None):
        return CitationLinkResult(
            links=[], provenance=_fake_provenance(),
            uncited_sentences=[{"sentence": sentence, "tag": "framing"}],
        )

    monkeypatch.setattr(citation_link_agent, "link_citations", fake_link_citations)

    await _close_citation_link_coverage_gap(draft_content, {}, uuid4())

    healed_attrs = draft_content["content"][0]["attrs"]
    assert healed_attrs["uncitedSentences"] == [
        {"paragraph_index": 0, "sentence": sentence, "tag": "framing"}
    ]


@pytest.mark.asyncio
async def test_resolve_and_finalize_keep_orphaned_sentences_from_writing_result_on_retry_failure(
    monkeypatch,
):
    """The write job's own gated-loop path (`writing.py`'s `_close_citation_link_
    coverage_gap`, backed by this module's `resolve_unclassified_sentences`), exercised
    directly against `writing_result.json` -- the flat, whole-section ``content``,
    ``citation_links`` and ``uncited_sentences`` a real run actually produced,
    before either fixture claim was ever appended. A worst-case retry failure must keep
    every orphaned sentence in the delivered text, not delete it."""
    run_dir = skip_unless_run_dir(PROMOTED_RUN)
    writing_result = json.loads(
        (run_dir / "writing_result.json").read_text(encoding="utf-8")
    )
    text = writing_result["content"]
    citation_links = writing_result["citation_links"]
    uncited_sentences = writing_result["uncited_sentences"]

    # Every currently-linked sentence is already known-good (it survived into this same
    # ``content``, the write job's own already-finalized text) -- "verified" here mirrors
    # that, so re-finalizing does not remove text finalize already decided to keep.
    claim_status = {
        (link["sentence"], link.get("proposition") or link["sentence"], key): "verified"
        for link in citation_links
        for key in link.get("keys") or []
    }

    async def fake_linker(retry_text):
        # Worst case, as in the draft-side regression test above: the retry call
        # itself fails outright, so neither orphaned sentence is ever actually looked
        # at.
        raise RuntimeError("provider outage")

    new_links, new_uncited, _prov, calls_made, retry_failed = (
        await resolve_unclassified_sentences(
            text, citation_links, uncited_sentences, fake_linker
        )
    )
    assert calls_made == 1
    assert retry_failed is True

    result = finalize_generated_section(text, new_links, new_uncited, claim_status)

    assert "Mixed-effect linear models" in result.text
    assert "Their survey of 70 students and 16 teachers" in result.text
    # A third gap this whole-section pass also finds, past the one paragraph the
    # draft-side regression test above checks in isolation: the section's own closing
    # paragraph makes an unattributed claim about the whole library with no citation
    # and no framing/finding tag either.
    assert "No study in the library compares teacher, peer, and automated feedback" in (
        result.text
    )
    assert (
        "Liu and Wu (2019) found that 48.10% of students favoured direct correction"
        in result.text
    )
    assert result.stats["sentences_unclassified_kept"] == 3
    assert result.stats.get("sentences_removed_uncited_finding", 0) == 0
