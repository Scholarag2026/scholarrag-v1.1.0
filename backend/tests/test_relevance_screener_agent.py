"""Relevance screener: protocol-eligibility screening with three statuses, criterion
anchoring and loud failure.

The screener stops asking
whether a record is topically relevant and asks whether the shown text fails a numbered
criterion of the review's own protocol. ``SCREENER_PROMPT_V1`` is the original binary
prompt, kept byte-exact so its sha stays reproducible for anyone re-scoring the v1
evaluation runs. ``SCREENER_PROMPT_V2`` is the active protocol-eligibility prompt: default
INCLUDE, an EXCLUDE must name a criterion id and quote the shown text verbatim, and a
record the shown text cannot decide is NEEDS_REVIEW rather than a soft EXCLUDE.
``apply_decision_guard`` is a deterministic, model-free check that demotes an unanchored
EXCLUDE to NEEDS_REVIEW and never does anything else.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from pydantic_ai import Agent, PromptedOutput  # noqa: E402

from app.agents import relevance_screener_agent as screener  # noqa: E402
from app.agents.model_config import (  # noqa: E402
    DETERMINISTIC_FAST_MODEL_SETTINGS,
    SECOND_PASS_V2_MODEL_SETTINGS,
)
from app.config import settings  # noqa: E402
from app.schemas.provenance import LLMCallProvenance, prompt_version  # noqa: E402

PAPERS = [
    {"title": "Paper A", "abstract": "About the topic."},
    {"title": "Paper B", "abstract": "About something else."},
    {"title": "Paper C", "abstract": "Also about the topic."},
]


def _fake_run_result(
    decisions: list[screener.PaperDecision],
    *,
    input_tokens: int = 321,
    output_tokens: int = 45,
    model_name: str = "deepseek-v4-flash",
    system_fingerprint: str = "fp_abc",
):
    response = SimpleNamespace(
        model_name=model_name,
        provider_response_id="resp-42",
        provider_details={"system_fingerprint": system_fingerprint, "finish_reason": "stop"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        output=screener.ScreeningDecisions(decisions=decisions),
        response=response,
        usage=lambda: usage,
    )


def _fake_provenance() -> LLMCallProvenance:
    return LLMCallProvenance(
        agent="relevance_screener",
        model_configured="deepseek-chat",
        prompt_version=screener.SCREENER_PROMPT_VERSION,
    )


# --- 1. SCREENER_PROMPT_V1: renamed, byte-exact, sha pinned -----------------------------


def test_screener_prompt_v1_is_byte_exact_and_matches_the_committed_sha():
    assert screener.SCREENER_PROMPT_V1_VERSION == "sha256:3c422c45cd8b"
    assert screener.SCREENER_PROMPT_V1_VERSION == prompt_version(screener.SCREENER_PROMPT_V1)
    assert "DIRECTLY addresses" in screener.SCREENER_PROMPT_V1
    assert "substantively contributes" in screener.SCREENER_PROMPT_V1
    assert "PRISMA framework" in screener.SCREENER_PROMPT_V1


# --- 2. SCREENER_PROMPT_V2: the active prompt --------------------------------------------


def test_screener_prompt_v2_digest_is_pinned_though_no_longer_active():
    """SCREENER_PROMPT moved from v2 to v3 (below), but v2 itself stays
    byte-identical so every v2 evaluation run stays reproducible."""
    assert prompt_version(screener.SCREENER_PROMPT_V2) == "sha256:fb9de89e0534"


def test_screener_prompt_is_v3_and_the_active_sha_matches_it():
    """Production wiring of the v2 screener design moves the active pointer
    to v3, so the anchor ledger, the title-only TOPIC exclude and (via the guard functions
    app.services.smart_search.run_smart_search already calls) the type/table-of-contents
    demotions and the inclusion-only second pass apply to every production job."""
    assert screener.SCREENER_PROMPT == screener.SCREENER_PROMPT_V3
    assert screener.SCREENER_PROMPT_VERSION == prompt_version(screener.SCREENER_PROMPT_V3)
    assert screener.SCREENER_PROMPT_VERSION != screener.SCREENER_PROMPT_V1_VERSION
    assert screener.SCREENER_PROMPT_VERSION != prompt_version(screener.SCREENER_PROMPT_V2)
    assert screener._ACTIVE_PROMPT_VERSION == "v3"


def test_screener_prompt_v2_defaults_to_include():
    text = screener.SCREENER_PROMPT_V2
    assert "Default to INCLUDE" in text
    assert "unless the shown title and abstract give you a" in text


def test_screener_prompt_v2_requires_a_criterion_id_and_a_verbatim_quote_for_exclude():
    text = screener.SCREENER_PROMPT_V2
    assert "EXCLUDE only when the shown text itself fails a numbered criterion" in text
    assert '"criterion"' in text
    assert '"quote"' in text
    assert "quote a phrase copied" in text
    assert "character for character" in text


def test_screener_prompt_v2_quote_supports_reason_self_check_is_on_the_exclude_bullet():
    """The self-check sentence guards against a P3 case (a therapy judged by its brand
    name) governed by the kinds-not-words bullet, not the absence bullet. It sits on the
    EXCLUDE bullet itself so it reads as a general check on every EXCLUDE, not one scoped
    to absence criteria."""
    text = screener.SCREENER_PROMPT_V2
    self_check = (
        "Before returning an\n  EXCLUDE, check that the phrase you quoted states the "
        "failure you wrote in the reason; if\n  it does not, you do not have a ground."
    )
    assert self_check in text
    exclude_bullet_start = text.index(
        "EXCLUDE only when the shown text itself fails a numbered criterion"
    )
    kinds_bullet_start = text.index("When a criterion gives a category by listing kinds")
    assert exclude_bullet_start < text.index(self_check) < kinds_bullet_start
    # no longer duplicated inside the absence bullet
    absence_bullet_start = text.index('A criterion phrased as an absence ("did not use"')
    ordering_bullet_start = text.index(
        "Judge the numbered criteria that are not listed under a Full-text"
    )
    absence_bullet = text[absence_bullet_start:ordering_bullet_start]
    assert "you do not have a ground" not in absence_bullet


def test_screener_prompt_v2_states_the_kinds_not_words_rule():
    """Inserted
    immediately after the EXCLUDE bullet. A criterion that gives a category by listing kinds
    or examples is satisfied by a member of the category, not only by a listed word; when the
    shown text cannot be classified, the answer is NEEDS_REVIEW, not EXCLUDE. Absent from V1,
    which has no criteria to read this way at all."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        'When a criterion gives a category by listing kinds or examples, whether with '
        '"included",\n  "such as", "for example", or a parenthetical, the list shows what '
        "the category covers;\n  it is not the whole of it." in text
    )
    assert (
        "A feature of the shown text that is of a listed kind\n  satisfies the criterion "
        "even when it is called something else" in text
    )
    assert (
        "EXCLUDE on such a criterion only when\n  the shown text names something plainly "
        "outside the category; when you cannot tell,\n  answer NEEDS_REVIEW." in text
    )
    # sits immediately after the EXCLUDE bullet, before the absence-criterion bullet
    assert (
        text.index("EXCLUDE only when the shown text itself fails a numbered criterion")
        < text.index("When a criterion gives a category by listing kinds or examples")
        < text.index('A criterion phrased as an absence ("did not use"')
    )
    assert "When a criterion gives a category by listing kinds or examples" not in (
        screener.SCREENER_PROMPT_V1
    )


def test_screener_prompt_v2_kinds_not_words_rule_defers_full_text_criteria():
    """Without a carve-out, the kinds-not-words bullet's own "when you
    cannot tell, answer NEEDS_REVIEW" would collide with the full-text inclusion bullet's
    rule that an unconfirmable full-text criterion is INCLUDE with the id in "to_confirm",
    never NEEDS_REVIEW. This carve-out sentence closes that collision."""
    text = screener.SCREENER_PROMPT_V2
    carve_out = (
        "This bullet governs only a criterion that is not listed under a\n  Full-text "
        "heading; a criterion listed under a Full-text heading is governed by the two\n  "
        "Full-text bullets below instead, not by this one."
    )
    assert carve_out in text
    kinds_bullet_start = text.index("When a criterion gives a category by listing kinds")
    absence_bullet_start = text.index('A criterion phrased as an absence ("did not use"')
    assert kinds_bullet_start < text.index(carve_out) < absence_bullet_start
    assert carve_out not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_does_not_state_a_defined_term_governs_everywhere_it_appears():
    """A defined-term-governs-everywhere sentence was tried on the kinds-not-words
    bullet, aimed at model errors that judged a criterion by the everyday sense of a term
    another criterion had already defined, but it measured worse on evaluation and was
    dropped again."""
    text = screener.SCREENER_PROMPT_V2
    sentence = (
        "When one numbered criterion defines a\n  term, that definition governs that term "
        "everywhere else in the protocol, including\n  inside another criterion and inside "
        "the examples a criterion gives. So a criterion that\n  asks whether something is "
        "that term, including one phrased as a negation of it, is\n  judged by the "
        "definition and not by the name the study gives what it evaluates: if the\n  shown "
        "text says that any part of what the study evaluates has the component the\n  "
        "definition names, the definition is met, whatever the study also names alongside "
        "it."
    )
    assert sentence not in text
    assert sentence not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_states_the_absence_is_shown_not_assumed_rule():
    """A criterion phrased as an absence
    is failed only when the shown text shows the thing missing from the study as a whole;
    naming it anywhere in any arm, condition, measure or outcome satisfies the criterion;
    silence about it is undecidable, not a failure. Absent from V1. The self-check sentence
    that used to close this bullet now lives on the EXCLUDE bullet instead (see the
    dedicated self-check test above)."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        'A criterion phrased as an absence ("did not use", "did not report", "no X") is '
        "failed\n  only when the shown text shows the thing missing from the study as a "
        "whole." in text
    )
    assert (
        "If the shown\n  text names the thing anywhere, in any arm, condition, measure or "
        "outcome, the criterion\n  is met, whatever else the study also contains." in text
    )
    assert (
        "If the shown text does not mention the\n  thing at all, you cannot tell: answer "
        "NEEDS_REVIEW, not EXCLUDE." in text
    )
    # sits immediately after edit 1, before the (unchanged) ordering bullet
    assert (
        text.index("EXCLUDE on such a criterion only when")
        < text.index('A criterion phrased as an absence ("did not use"')
        < text.index("Judge the numbered criteria that are not listed under a Full-text")
    )
    assert 'A criterion phrased as an absence ("did not use"' not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_absence_rule_defers_full_text_criteria():
    """Without a carve-out, the absence bullet's own "you cannot tell: answer
    NEEDS_REVIEW" would collide with that bullet's "silence about one of these has no
    effect on your decision" -- both rules are exclusive and cover the identical case of a
    shown text that says nothing about the criterion's subject. This carve-out sentence
    closes that collision."""
    text = screener.SCREENER_PROMPT_V2
    carve_out = (
        "This bullet governs only\n  a criterion that is not listed under a Full-text "
        "heading; silence about a criterion\n  listed under a Full-text heading is "
        "governed by the two Full-text bullets below instead."
    )
    assert carve_out in text
    absence_bullet_start = text.index('A criterion phrased as an absence ("did not use"')
    ordering_bullet_start = text.index(
        "Judge the numbered criteria that are not listed under a Full-text"
    )
    assert absence_bullet_start < text.index(carve_out) < ordering_bullet_start
    assert carve_out not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_states_the_ordering_rule():
    """One bullet, inserted between the EXCLUDE bullet and the two
    full-text bullets, states that the abstract-stage criteria and the research question
    are judged first, and that a record failing one of those (or off topic) is EXCLUDE on
    that ground without ever reaching a full-text criterion."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        "Judge the numbered criteria that are not listed under a Full-text heading, and "
        "the\n  research question, before anything else." in text
    )
    assert "do not consider a\n  full-text criterion." in text
    # sits between the EXCLUDE bullet and the two full-text bullets
    assert (
        text.index("EXCLUDE only when the shown text itself fails a numbered criterion")
        < text.index("Judge the numbered criteria that are not listed under a Full-text")
        < text.index("A criterion listed under Full-text inclusion criteria")
        < text.index("A criterion listed under Full-text exclusion criteria")
    )


def test_screener_prompt_v2_states_the_off_topic_definition():
    """The off-topic definition inside the ordering bullet: a different population,
    phenomenon or outcome, not merely an unconfirmed detail some criterion asks about --
    and a record with no abstract can never be judged off topic at all. The naming
    instruction (a criterion id or the reserved "TOPIC", a verbatim quote like any other
    EXCLUDE) is unchanged, right after the definition. A later sentence widens the
    no-abstract rule further: see the dedicated test below."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        "A record is off topic when the shown text is about a different\n  population, "
        "phenomenon or outcome from the one the research question names." in text
    )
    assert (
        "It is not\n  off topic merely because the shown text fails to confirm a detail "
        "some criterion asks\n  about: if your reason for calling a record off topic could "
        'be written as "criterion Cn\n  is not confirmed", then judge Cn under its own '
        "rules instead, and if Cn is a full-text\n  criterion, do not EXCLUDE at all."
        in text
    )
    assert (
        "can never be judged off topic: answer NEEDS_REVIEW." in text
    )
    assert (
        "An off-topic\n  EXCLUDE names the criterion the mismatch fails, or \"TOPIC\" when "
        "no criterion covers it,\n  and quotes the shown text verbatim like any other "
        "EXCLUDE." in text
    )
    assert '"TOPIC"' in text
    assert "quotes the shown text verbatim like any other EXCLUDE" in text
    # the old wording is gone, not merely superseded by new text elsewhere
    assert "does not concern the" not in text
    for absent in (
        "is about a different\n  population",
        "can never be judged off topic: answer NEEDS_REVIEW.",
        'if your reason for calling a record off topic could be written as "criterion Cn',
    ):
        assert absent not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_no_abstract_sentence_forbids_any_criterion_grounded_exclude():
    """Forbidding only the off-topic reading of a
    no-abstract record would leave room for an EXCLUDE naming a numbered criterion and
    quoting the title (a real failure case, "Moclobemide in Social Phobia" on I2). The
    sentence is absolute: a no-abstract record can never ground an EXCLUDE on any
    criterion, full stop, not only the off-topic one."""
    text = screener.SCREENER_PROMPT_V2
    widened = (
        "A record with no abstract can never ground an EXCLUDE\n  on any criterion, and "
        "can never be judged off topic: answer NEEDS_REVIEW."
    )
    assert widened in text
    assert widened not in screener.SCREENER_PROMPT_V1


def test_screener_prompt_v2_states_the_full_text_inclusion_rule():
    """The single "Full-text criteria" rule bullet splits into two,
    one per kind, in the same position (after the EXCLUDE bullet). The inclusion side defers
    the criterion rather than flagging it: judge the rest, INCLUDE with a to-confirm note,
    and silence never grounds an EXCLUDE or a NEEDS_REVIEW."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        "A criterion listed under Full-text inclusion criteria cannot be decided from a "
        "title and\n  abstract." in text
    )
    assert '"to_confirm"' in text
    assert (
        "Silence about one of these is not a reason to EXCLUDE and not a\n  reason for "
        "NEEDS_REVIEW." in text
    )
    # additive: the exact byte sequence of the surrounding bullets is unchanged
    assert "EXCLUDE only when the shown text itself fails a numbered criterion" in text
    assert "NEEDS_REVIEW when the shown text cannot decide" in text


def test_screener_prompt_v2_states_the_full_text_exclusion_rule():
    """The exclusion side can never ground an EXCLUDE, and grounds a
    NEEDS_REVIEW only behind explicit contrary evidence; silence about it has no effect."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        "A criterion listed under Full-text exclusion criteria can never ground an EXCLUDE."
        in text
    )
    assert "explicit contrary evidence" in text
    assert "Silence about one of\n  these has no effect on your decision." in text


def test_screener_prompt_v2_schema_line_covers_a_needs_review_quote():
    """The "quote" schema line must cover a NEEDS_REVIEW that names a
    criterion (the full-text-exclusion contrary-evidence case), not only an EXCLUDE, or a
    model reading the schema literally would never emit the evidence the exclusion-side rule
    bullet asks it to quote."""
    text = screener.SCREENER_PROMPT_V2
    assert (
        '- "quote": for an EXCLUDE, and for any NEEDS_REVIEW that names a criterion, a '
        "phrase\n  copied verbatim from the shown title or abstract, \"\" otherwise"
    ) in text


def test_screener_prompt_v2_schema_line_states_to_confirm():
    text = screener.SCREENER_PROMPT_V2
    assert (
        '- "to_confirm": on an INCLUDE, the full-text inclusion criterion ids the shown '
        "text does\n  not already confirm; [] otherwise"
    ) in text


def test_paper_decision_criterion_field_description_agrees_with_the_prompt_schema():
    """The pydantic-ai structured-output schema (``PaperDecision.criterion``'s ``description``)
    must state the same rule as the prompt's own schema line, or the model sees two
    contradictory contracts for the same field."""
    description = screener.PaperDecision.model_fields["criterion"].description
    assert "NEEDS_REVIEW" in description
    assert "otherwise" in description
    assert description != "The criterion id for an EXCLUDE (for example 'E3'); '' otherwise."


def test_paper_decision_quote_field_description_agrees_with_the_prompt_schema():
    """The "quote" schema line widens to cover a NEEDS_REVIEW that
    names a criterion, so the field description must say the same thing."""
    description = screener.PaperDecision.model_fields["quote"].description
    assert "NEEDS_REVIEW" in description
    assert description != (
        "For an EXCLUDE, a phrase copied verbatim from the shown text; '' otherwise."
    )


def test_screener_prompt_v2_states_the_needs_review_rule():
    text = screener.SCREENER_PROMPT_V2
    assert "NEEDS_REVIEW" in text
    assert "NEEDS_REVIEW is not a soft EXCLUDE" in text


def test_screener_prompt_v2_states_the_cut_abstract_rule():
    text = screener.SCREENER_PROMPT_V2
    assert "cut abstract" in text
    assert "[...]" in text
    assert "Absence of evidence in a cut abstract is not evidence" in text


def test_screener_prompt_v2_screens_each_record_on_its_own():
    text = screener.SCREENER_PROMPT_V2
    assert "Screen each record on its own" in text
    assert "Do not compare records" in text


def test_screener_prompt_v2_deletes_the_four_v1_precision_first_clauses():
    text = screener.SCREENER_PROMPT_V2
    for deleted_clause in (
        "DIRECTLY addresses or substantively contributes",
        "merely shares some keywords",
        "uses methods FROM the topic's domain",
        "When genuinely uncertain",
    ):
        assert deleted_clause not in text


def test_system_prompt_asks_for_a_short_reason_per_paper():
    assert "reason" in screener.SCREENER_PROMPT.lower()
    assert "20 words" in screener.SCREENER_PROMPT


# --- 3. Abstract cap constants and the version-bound renderer ---------------------------


def test_abstract_cap_constants_are_a_7_3_split_of_the_limit():
    assert screener.ABSTRACT_CHAR_LIMIT == 10000
    assert screener.ABSTRACT_HEAD_CHARS == 7000
    assert screener.ABSTRACT_TAIL_CHARS == 3000
    total = screener.ABSTRACT_HEAD_CHARS + screener.ABSTRACT_TAIL_CHARS
    assert total == screener.ABSTRACT_CHAR_LIMIT
    assert screener.ABSTRACT_CUT_MARKER == " [...] "


def test_render_abstract_v1_reproduces_the_committed_behaviour_byte_for_byte():
    long_abstract = "x" * 600
    rendered = screener.render_abstract(long_abstract, prompt_version="v1", limit=500)
    assert rendered == long_abstract[:500] + "..."

    short_abstract = "a short abstract"
    rendered_short = screener.render_abstract(short_abstract, prompt_version="v1", limit=500)
    assert rendered_short == short_abstract


def test_render_abstract_v2_shows_the_whole_abstract_at_or_under_the_limit():
    abstract = "y" * 500
    assert screener.render_abstract(abstract, prompt_version="v2", limit=500) == abstract


def test_render_abstract_v2_shows_head_marker_tail_above_the_limit():
    abstract = "".join(f"{i:04d}" for i in range(200))  # 800 deterministic characters
    limit = 500
    rendered = screener.render_abstract(abstract, prompt_version="v2", limit=limit)
    assert rendered == abstract[:350] + screener.ABSTRACT_CUT_MARKER + abstract[-150:]
    assert screener.ABSTRACT_CUT_MARKER in rendered


def test_render_abstract_v2_at_limit_500_differs_from_v1():
    abstract = "z" * 700
    v1 = screener.render_abstract(abstract, prompt_version="v1", limit=500)
    v2 = screener.render_abstract(abstract, prompt_version="v2", limit=500)
    assert v1 == abstract[:500] + "..."
    assert screener.ABSTRACT_CUT_MARKER in v2
    assert v1 != v2


def test_render_abstract_rejects_an_unknown_prompt_version():
    with pytest.raises(ValueError):
        screener.render_abstract("abc", prompt_version="v4", limit=100)


# --- 4. Numbered criteria and build_shown_texts ------------------------------------------


def test_build_screening_prompt_numbers_criteria_in_protocol_order():
    prompt = screener._build_screening_prompt(
        "topic",
        PAPERS,
        inclusion_criteria=["is peer reviewed", "is about the topic"],
        exclusion_criteria=["is a conference abstract only"],
    )
    assert "I1: is peer reviewed" in prompt
    assert "I2: is about the topic" in prompt
    assert "E1: is a conference abstract only" in prompt


def test_build_shown_texts_returns_title_and_abstract_exactly_as_rendered():
    papers = [
        {"title": "T1", "abstract": "A1"},
        {"title": "T2", "abstract": "B" * 15000},
    ]
    texts = screener.build_shown_texts(papers)
    assert texts[0] == "Title: T1\nAbstract: A1"
    assert texts[1].startswith("Title: T2\nAbstract: ")
    assert screener.ABSTRACT_CUT_MARKER in texts[1]


def test_build_shown_texts_indents_two_spaces_at_v1_to_match_the_user_turn():
    """The v1 user turn indents its "Title:"/"Abstract:"
    lines by two spaces, so the canonical helper must match that margin at v1, not only
    render the unindented v2 form."""
    papers = [{"title": "T1", "abstract": "A1"}]
    texts = screener.build_shown_texts(papers, prompt_version="v1", limit=500)
    assert texts[0] == "  Title: T1\n  Abstract: A1"
    prompt = screener._build_screening_prompt(
        "teacher burnout", papers, prompt_version="v1", limit=500
    )
    assert texts[0] in prompt


def test_build_shown_texts_v2_is_unindented_still():
    papers = [{"title": "T1", "abstract": "A1"}]
    texts = screener.build_shown_texts(papers, prompt_version="v2", limit=500)
    assert texts[0] == "Title: T1\nAbstract: A1"


def test_build_shown_texts_treats_a_whitespace_only_abstract_as_missing():
    """``paper.get("abstract") or NO_ABSTRACT_PLACEHOLDER``
    would leave a whitespace-only abstract truthy, rendering "Abstract:    " instead of
    the placeholder -- so ``_shown_abstract_is_missing``'s suffix match would fail and an
    EXCLUDE quoting only the title would stand. Stripping first makes the renderer agree
    with the harness, which already treats ``(rec.get("abstract") or "").strip()`` falsy
    as no abstract."""
    papers = [{"title": "T1", "abstract": "   "}]
    texts = screener.build_shown_texts(papers)
    assert texts[0] == f"Title: T1\nAbstract: {screener.NO_ABSTRACT_PLACEHOLDER}"
    assert screener._shown_abstract_is_missing(texts[0])


def test_build_shown_texts_matches_what_the_prompt_shows_per_record():
    prompt = screener._build_screening_prompt("q", PAPERS)
    for text in screener.build_shown_texts(PAPERS):
        assert text in prompt


# --- 4b. The "Full-text inclusion/exclusion criteria" lines ---------


def test_build_screening_prompt_v2_renders_the_full_text_inclusion_and_exclusion_lines():
    prompt = screener._build_screening_prompt(
        "topic",
        PAPERS,
        inclusion_criteria=["is peer reviewed", "reports an effect size"],
        exclusion_criteria=["is a conference abstract only", "no comparator reported"],
        inclusion_stages=["abstract", "full_text"],
        exclusion_stages=["abstract", "full_text"],
    )
    assert "Full-text inclusion criteria: I2" in prompt
    assert "Full-text exclusion criteria: E2" in prompt
    assert '"to_confirm"' in prompt
    assert "These can never ground an EXCLUDE." in prompt
    assert "explicit contrary evidence" in prompt
    # inclusion heading first, both after the exclusion block
    assert (
        prompt.index("E2: no comparator reported")
        < prompt.index("Full-text inclusion criteria:")
        < prompt.index("Full-text exclusion criteria:")
    )


def test_build_screening_prompt_v2_renders_only_the_side_that_has_a_full_text_member():
    """Either heading is independent: an inclusion-only full-text criterion renders no
    exclusion heading, and vice versa."""
    inclusion_only = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["full_text"],
        exclusion_stages=["abstract"],
    )
    assert "Full-text inclusion criteria: I1" in inclusion_only
    assert "Full-text exclusion criteria" not in inclusion_only

    exclusion_only = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["full_text"],
    )
    assert "Full-text inclusion criteria" not in exclusion_only
    assert "Full-text exclusion criteria: E1" in exclusion_only


def test_build_screening_prompt_v2_omits_both_full_text_headings_when_none_is_full_text():
    prompt = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["abstract"],
    )
    assert "Full-text inclusion criteria" not in prompt
    assert "Full-text exclusion criteria" not in prompt


def test_build_screening_prompt_v2_omits_the_heading_when_no_criterion_is_full_text():
    prompt = screener._build_screening_prompt(
        "topic",
        PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["abstract"],
    )
    assert "Full-text criteria" not in prompt
    # and when no stage lists are given at all (every pre-S8 call site)
    prompt_no_stages = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
    )
    assert "Full-text criteria" not in prompt_no_stages
    assert prompt_no_stages == screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["abstract"],
    )


def test_build_screening_prompt_v2_renders_the_order_line_before_either_heading():
    """The order line sits at the position of the Full-text inclusion
    heading, and appears whenever either side has a full-text member -- even when only the
    exclusion side does, so no inclusion heading renders at all."""
    both = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed", "reports an effect size"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract", "full_text"],
        exclusion_stages=["full_text"],
    )
    assert "Order of judgement: decide the criteria above and the research question" in both
    assert (
        both.index("Order of judgement:")
        < both.index("Full-text inclusion criteria:")
        < both.index("Full-text exclusion criteria:")
    )

    exclusion_only = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["full_text"],
    )
    assert "Order of judgement:" in exclusion_only
    assert "Full-text inclusion criteria" not in exclusion_only
    assert (
        exclusion_only.index("Order of judgement:")
        < exclusion_only.index("Full-text exclusion criteria:")
    )


def test_build_screening_prompt_v2_omits_order_line_when_no_criterion_is_full_text():
    prompt = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        inclusion_stages=["abstract"],
        exclusion_stages=["abstract"],
    )
    assert "Order of judgement" not in prompt
    # and when no stage lists are given at all (every pre-S8 call site)
    prompt_no_stages = screener._build_screening_prompt(
        "topic", PAPERS,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
    )
    assert "Order of judgement" not in prompt_no_stages


def test_build_screening_prompt_v1_ignores_stage_arguments():
    """The stage-aware line is v2-only; a v1 call must render byte-identically whether or
    not stages are passed (v1 has no criterion ids for a model to route against)."""
    papers = [{"title": "Paper A", "abstract": "About the topic."}]
    with_stages = screener._build_screening_prompt(
        "q", papers, exclusion_criteria=["off topic"],
        prompt_version="v1", inclusion_stages=[], exclusion_stages=["full_text"],
    )
    without_stages = screener._build_screening_prompt(
        "q", papers, exclusion_criteria=["off topic"], prompt_version="v1",
    )
    assert with_stages == without_stages
    assert "Full-text criteria" not in with_stages


def test_build_screening_prompt_v2_notes_when_no_criteria_are_given():
    """The user turn must not promise numbered
    criteria it does not supply."""
    prompt = screener._build_screening_prompt("q", PAPERS)
    assert "No inclusion or exclusion criteria were provided" in prompt


def test_build_screening_prompt_v1_matches_the_parent_format_byte_for_byte():
    """The v1 framing (criteria bullets, per-record
    layout) must reproduce the original parent format exactly, not just the abstract
    truncation style, so a --prompt-version v1 run stays byte-identical to the six
    committed v1 runs wherever the protocol text itself is unchanged."""
    papers = [
        {"title": "Paper A", "abstract": "About the topic."},
        {"title": "Paper B", "abstract": "x" * 600},
    ]
    prompt = screener._build_screening_prompt(
        "teacher burnout",
        papers,
        inclusion_criteria=["is peer reviewed"],
        exclusion_criteria=["is a conference abstract only"],
        prompt_version="v1",
        limit=500,
    )
    expected = (
        "Research topic: teacher burnout\n"
        "\n"
        "INCLUDE papers that:\n"
        "  - is peer reviewed\n"
        "\n"
        "EXCLUDE papers that:\n"
        "  - is a conference abstract only\n"
        "\n"
        "Papers to screen (title and abstract):\n"
        "\n"
        "Paper 1:\n"
        "  Title: Paper A\n"
        "  Abstract: About the topic.\n"
        "\n"
        "Paper 2:\n"
        "  Title: Paper B\n"
        "  Abstract: " + "x" * 500 + "..."
    )
    assert prompt == expected


def test_build_screening_prompt_rejects_an_unknown_prompt_version():
    with pytest.raises(ValueError):
        screener._build_screening_prompt("q", PAPERS, prompt_version="v4")


# --- 5. PaperDecision and ScreeningBatchResult schemas ------------------------------------


def test_paper_decision_accepts_the_three_statuses_and_defaults_criterion_and_quote():
    d = screener.PaperDecision(decision="needs_review")
    assert d.decision == "NEEDS_REVIEW"
    assert d.criterion == ""
    assert d.quote == ""
    with pytest.raises(ValueError):
        screener.PaperDecision(decision="MAYBE")


def test_paper_decision_normalises_case_and_truncates_long_reasons():
    d = screener.PaperDecision(decision="include", reason="x" * 500)
    assert d.decision == "INCLUDE"
    assert len(d.reason) == 300


def test_paper_decision_truncates_overlong_criterion_and_quote_gracefully():
    d = screener.PaperDecision(decision="EXCLUDE", criterion="X" * 20, quote="y" * 400)
    assert len(d.criterion) == 8
    assert len(d.quote) == 300


def test_paper_decision_normalises_criterion_case_and_whitespace():
    """A model that answers "e1" or " E1 " must still
    match a known id of "E1", the same way ``_normalise_decision`` already strips and
    upper-cases ``decision``."""
    assert screener.PaperDecision(decision="EXCLUDE", criterion="e1").criterion == "E1"
    assert screener.PaperDecision(decision="EXCLUDE", criterion=" E1 ").criterion == "E1"
    assert screener.PaperDecision(decision="EXCLUDE", criterion="e1.").criterion == "E1."


def test_paper_decision_normalises_then_truncates_an_overlong_criterion():
    d = screener.PaperDecision(decision="EXCLUDE", criterion=" " + "e" * 20)
    assert d.criterion == "E" * 8


def test_paper_decision_to_confirm_defaults_to_empty():
    assert screener.PaperDecision(decision="INCLUDE").to_confirm == []


def test_paper_decision_normalises_to_confirm_ids():
    """Each id is stripped/upper-cased the same way ``criterion`` is,
    and a blank entry is dropped, so the model's own casing never fails a set-membership
    check in :func:`screen_papers`."""
    d = screener.PaperDecision(decision="INCLUDE", to_confirm=["i1", " I3 ", ""])
    assert d.to_confirm == ["I1", "I3"]


def test_screening_batch_result_include_is_computed_from_status():
    result = screener.ScreeningBatchResult(
        statuses=["INCLUDE", "EXCLUDE", "NEEDS_REVIEW"],
        criteria_ids=["", "E1", ""],
        quotes=["", "something else", ""],
        reasons=["r1", "r2", "r3"],
        provenance=_fake_provenance(),
    )
    assert result.include == [True, False, False]


def test_screening_batch_result_requires_provenance():
    """Contract from the plan's Produces block: provenance is not optional."""
    with pytest.raises(ValueError):
        screener.ScreeningBatchResult(
            statuses=["INCLUDE"], criteria_ids=[""], quotes=[""], reasons=["r"]
        )


def test_screening_batch_result_guard_fields_default_to_empty():
    result = screener.ScreeningBatchResult(
        statuses=["INCLUDE"],
        criteria_ids=[""],
        quotes=[""],
        reasons=["r"],
        provenance=_fake_provenance(),
    )
    assert result.guard_conversions == 0
    assert result.guard_applied == []
    assert result.padded == 0
    # guard_reasons: parallel to statuses, defaulted to "" per record when omitted
    assert result.guard_reasons == [""]
    # to_confirm: parallel to statuses, defaulted to [] per record when omitted
    assert result.to_confirm == [[]]


def test_screening_batch_result_rejects_a_mismatched_guard_reasons_length():
    with pytest.raises(ValueError):
        screener.ScreeningBatchResult(
            statuses=["INCLUDE", "EXCLUDE"],
            criteria_ids=["", "E1"],
            quotes=["", "q"],
            reasons=["r1", "r2"],
            provenance=_fake_provenance(),
            guard_reasons=["full_text_criterion"],
        )


def test_screening_batch_result_rejects_a_mismatched_to_confirm_length():
    with pytest.raises(ValueError):
        screener.ScreeningBatchResult(
            statuses=["INCLUDE", "EXCLUDE"],
            criteria_ids=["", "E1"],
            quotes=["", "q"],
            reasons=["r1", "r2"],
            provenance=_fake_provenance(),
            to_confirm=[["I1"]],
        )


def test_screening_batch_result_accepts_explicit_matching_to_confirm():
    result = screener.ScreeningBatchResult(
        statuses=["INCLUDE", "EXCLUDE"],
        criteria_ids=["", "E1"],
        quotes=["", "q"],
        reasons=["r1", "r2"],
        provenance=_fake_provenance(),
        to_confirm=[["I1", "I3"], []],
    )
    assert result.to_confirm == [["I1", "I3"], []]


def test_screening_batch_result_accepts_explicit_matching_guard_reasons():
    result = screener.ScreeningBatchResult(
        statuses=["INCLUDE", "NEEDS_REVIEW"],
        criteria_ids=["", "E1"],
        quotes=["", "q"],
        reasons=["r1", "r2"],
        provenance=_fake_provenance(),
        guard_reasons=["", "cut_abstract"],
    )
    assert result.guard_reasons == ["", "cut_abstract"]


def test_screening_batch_result_rejects_mismatched_list_lengths():
    """The four per-record lists must stay parallel, or
    ``smart_search.py``'s five-way zip would truncate silently."""
    with pytest.raises(ValueError):
        screener.ScreeningBatchResult(
            statuses=["INCLUDE", "EXCLUDE", "INCLUDE"],
            criteria_ids=[""],
            quotes=["", "", ""],
            reasons=["r1", "r2", "r3"],
            provenance=_fake_provenance(),
        )


def test_screening_batch_result_rejects_a_mismatched_guard_applied_length():
    with pytest.raises(ValueError):
        screener.ScreeningBatchResult(
            statuses=["INCLUDE", "EXCLUDE"],
            criteria_ids=["", ""],
            quotes=["", ""],
            reasons=["r1", "r2"],
            provenance=_fake_provenance(),
            guard_applied=[True],
        )


def test_screening_batch_result_accepts_matching_lengths_and_empty_guard_applied():
    result = screener.ScreeningBatchResult(
        statuses=["INCLUDE", "EXCLUDE"],
        criteria_ids=["", "E1"],
        quotes=["", "q"],
        reasons=["r1", "r2"],
        provenance=_fake_provenance(),
    )
    assert result.guard_applied == []


def test_screening_batch_result_accepts_legacy_include_until_the_shim_is_deleted():
    """A concurrent phase's test fixtures still build
    ScreeningBatchResult(include=..., reasons=..., provenance=...); this shim keeps that
    path working until that phase moves its fixtures to statuses=. Delete this test (and
    the _accept_legacy_include validator) once it does."""
    result = screener.ScreeningBatchResult(
        include=[True, False, True],
        reasons=["r1", "r2", "r3"],
        provenance=_fake_provenance(),
    )
    assert result.statuses == ["INCLUDE", "EXCLUDE", "INCLUDE"]
    assert result.criteria_ids == ["", "", ""]
    assert result.quotes == ["", "", ""]
    assert result.include == [True, False, True]


def test_screening_batch_result_legacy_include_does_not_override_explicit_statuses():
    result = screener.ScreeningBatchResult(
        include=[True],
        statuses=["EXCLUDE"],
        criteria_ids=["E1"],
        quotes=["q"],
        reasons=["r"],
        provenance=_fake_provenance(),
    )
    assert result.statuses == ["EXCLUDE"]


# --- 6. apply_decision_guard --------------------------------------------------------------

SHOWN_TEXT = "Title: T\nAbstract: This study uses rats exclusively for the intervention."


def test_guard_passes_through_a_properly_anchored_exclude():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"I1", "E1"})
    assert result.decision == "EXCLUDE"
    assert result.criterion == "E1"
    assert result.quote == "uses rats exclusively"
    assert guard_reason == ""


def test_guard_demotes_exclude_with_an_unknown_criterion_id():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E9", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"I1", "E1"})
    assert result.decision == "NEEDS_REVIEW"
    assert result.criterion == "E9"  # kept for audit
    assert result.quote == "uses rats exclusively"  # kept for audit
    assert "guard" in result.reason.lower()
    assert guard_reason == "unanchored_exclude"


def test_guard_demotes_exclude_with_an_empty_quote():
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="")
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
    assert result.decision == "NEEDS_REVIEW"
    assert "guard" in result.reason.lower()
    assert guard_reason == "unanchored_exclude"


def test_guard_demotes_exclude_with_a_non_verbatim_quote():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="a phrase never shown"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
    assert result.decision == "NEEDS_REVIEW"
    assert "guard" in result.reason.lower()
    assert guard_reason == "unanchored_exclude"


def test_guard_verbatim_check_is_whitespace_normalised_and_casefolded():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="USES   rats   exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


def test_guard_never_changes_include_or_needs_review_monotonicity():
    for status in ("INCLUDE", "NEEDS_REVIEW"):
        decision = screener.PaperDecision(decision=status, criterion="", quote="")
        result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
        assert result.decision == status
        assert result.reason == decision.reason
        assert guard_reason == ""


def test_guard_never_produces_an_include_from_an_exclude():
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="bad", quote="bad")
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
    assert result.decision in ("EXCLUDE", "NEEDS_REVIEW")
    assert result.decision != "INCLUDE"
    assert guard_reason in ("", "unanchored_exclude", "full_text_criterion", "cut_abstract")


def test_guard_with_no_protocol_criteria_requires_only_a_verbatim_quote():
    """With an empty protocol no id could ever be
    "known", so the criterion-id half of the anchor is dropped and a verbatim quote alone
    anchors the EXCLUDE."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, set())
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


def test_guard_accepts_a_lowercase_criterion_id_after_normalisation():
    """Normalisation happens on ``PaperDecision``
    construction, so the guard sees "E1" even when the model answered "e1"."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="e1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"I1", "E1"})
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


def test_guard_with_no_protocol_criteria_still_demotes_a_non_verbatim_quote():
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="", quote="not shown at all")
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, set())
    assert result.decision == "NEEDS_REVIEW"
    assert "guard" in result.reason.lower()
    assert guard_reason == "unanchored_exclude"


# --- 6a. The no-abstract guard rule ---------------------------------------------

NO_ABSTRACT_SHOWN_TEXT = (
    f"Title: Moclobemide in Social Phobia\nAbstract: {screener.NO_ABSTRACT_PLACEHOLDER}"
)


def test_guard_reason_no_abstract_constant_and_placeholder():
    assert screener.GUARD_REASON_NO_ABSTRACT == "no_abstract"
    assert screener.NO_ABSTRACT_PLACEHOLDER == "(no abstract)"


def test_shown_abstract_is_missing_matches_the_rendered_placeholder():
    """The private helper is exact, not a heuristic: it fires only when the shown text ends
    in the literal placeholder line, not merely because a real abstract mentions similar
    words somewhere in its body."""
    assert screener._shown_abstract_is_missing(NO_ABSTRACT_SHOWN_TEXT) is True
    assert screener._shown_abstract_is_missing(SHOWN_TEXT) is False
    assert screener._shown_abstract_is_missing("") is False
    assert screener._shown_abstract_is_missing(None) is False
    # a real abstract that happens to mention the placeholder text mid-sentence is not the
    # placeholder itself -- the check is a suffix match, not a substring search
    mentions_placeholder = (
        "Title: T\nAbstract: Reviewers often write (no abstract) as a shorthand complaint, "
        "but this record has a full one."
    )
    assert screener._shown_abstract_is_missing(mentions_placeholder) is False


def test_guard_demotes_exclude_on_a_record_with_no_abstract():
    """An EXCLUDE on a record whose rendered
    abstract is the no-abstract placeholder is always demoted, with reason "no_abstract"."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2", quote="Moclobemide",
        reason="Moclobemide is a medication, not CBT",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, NO_ABSTRACT_SHOWN_TEXT, {"I2"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert result.criterion == "I2"  # kept for audit
    assert result.quote == "Moclobemide"  # kept for audit
    assert "guard" in result.reason.lower()
    assert guard_reason == "no_abstract"


def test_guard_no_abstract_fires_even_when_the_quote_is_a_verbatim_substring_of_the_title():
    """The diagnosis's exact failure mode (record 18): the model quotes only the title,
    which is still part of the shown text and so passes the ordinary verbatim-substring
    check. Without the no-abstract check running first, this EXCLUDE would otherwise stand
    unchanged -- a real (empty-abstract) known criterion id and a genuinely verbatim quote."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2", quote="Moclobemide in Social Phobia",
        reason="no abstract shown",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, NO_ABSTRACT_SHOWN_TEXT, {"I2"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "no_abstract"


def test_guard_demotes_exclude_on_an_empty_string_abstract():
    """An empty-string abstract renders through the same placeholder as a missing one
    (``build_shown_texts``: ``(paper.get("abstract") or "").strip() or
    NO_ABSTRACT_PLACEHOLDER``), so the guard treats the two identically."""
    shown_text = f"Title: T\nAbstract: {screener.NO_ABSTRACT_PLACEHOLDER}"
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="T")
    result, guard_reason = screener.apply_decision_guard(decision, shown_text, {"E1"})
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "no_abstract"


def test_guard_demotes_exclude_on_a_whitespace_only_abstract():
    """A whitespace-only abstract would otherwise render truthy
    ("Abstract:    "), which is not the placeholder suffix, so an EXCLUDE quoting only the
    title would survive the guard. ``build_shown_texts`` strips before falling back to the
    placeholder, so this shown text is identical to the no-abstract case."""
    shown_text = screener.build_shown_texts(
        [{"title": "Moclobemide in Social Phobia", "abstract": "   "}]
    )[0]
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2", quote="Moclobemide in Social Phobia",
    )
    result, guard_reason = screener.apply_decision_guard(decision, shown_text, {"I2"})
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "no_abstract"


def test_guard_no_abstract_check_precedes_full_text_and_cut_abstract_and_unanchored():
    """The no-abstract check is unconditional and runs before every other EXCLUDE
    check, so a criterion that is also full-text-exclusion, absence-tested with a cut
    abstract, or simply unknown, is still reported "no_abstract" on a record with no
    abstract -- never "full_text_criterion", "cut_abstract" or "unanchored_exclude" -- as
    long as the quote is verbatim. This test's name does not mention cut-abstract or
    non-verbatim-quote coverage, but both are exercised here (the non-verbatim-quote case
    is asserted in
    ``test_guard_no_abstract_falls_through_to_unanchored_exclude_on_a_fabricated_quote``
    below, which gets the other label instead)."""
    verbatim_on_title = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="Moclobemide in Social Phobia",
    )
    _, guard_reason = screener.apply_decision_guard(
        verbatim_on_title, NO_ABSTRACT_SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert guard_reason == "no_abstract"

    unknown_id = screener.PaperDecision(
        decision="EXCLUDE", criterion="E9", quote="Moclobemide in Social Phobia",
    )
    _, guard_reason = screener.apply_decision_guard(unknown_id, NO_ABSTRACT_SHOWN_TEXT, {"E1"})
    assert guard_reason == "no_abstract"

    full_text_inclusion = screener.PaperDecision(
        decision="EXCLUDE", criterion="I1", quote="Moclobemide in Social Phobia",
    )
    _, guard_reason = screener.apply_decision_guard(
        full_text_inclusion, NO_ABSTRACT_SHOWN_TEXT, {"I1"}, full_text_inclusion_ids={"I1"},
    )
    assert guard_reason == "no_abstract"

    absence_criterion_cut_abstract_text = (
        f"Title: Moclobemide [...] Phobia\nAbstract: {screener.NO_ABSTRACT_PLACEHOLDER}"
    )
    absence_with_cut_marker = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="Moclobemide [...] Phobia",
    )
    _, guard_reason = screener.apply_decision_guard(
        absence_with_cut_marker, absence_criterion_cut_abstract_text, {"E1"},
        absence_ids={"E1"},
    )
    assert guard_reason == "no_abstract"


def test_guard_no_abstract_falls_through_to_unanchored_exclude_on_a_fabricated_quote():
    """The no-abstract check being unconditional (it
    always runs first, ahead of every other EXCLUDE check) does not mean it always wins the
    same gentler label. An EXCLUDE on a no-abstract record whose quote is not in the shown
    text at all -- the model invented the quote rather than merely copying the title -- is
    demoted to "unanchored_exclude", the same fabricated-quote distinction the full-text-
    exclusion branch already makes ("an EXCLUDE that both breaks the full-text rule and
    fabricates its quote is not owed the gentler label"), not to the benign "no_abstract"."""
    fabricated = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2",
        quote="a 6-month trial of paroxetine 40mg in adults",
    )
    result, guard_reason = screener.apply_decision_guard(
        fabricated, NO_ABSTRACT_SHOWN_TEXT, {"I2"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_exclude"

    # an empty quote on a no-abstract record is fabricated too (nothing to anchor to)
    empty_quote = screener.PaperDecision(decision="EXCLUDE", criterion="I2", quote="")
    _, guard_reason = screener.apply_decision_guard(empty_quote, NO_ABSTRACT_SHOWN_TEXT, {"I2"})
    assert guard_reason == "unanchored_exclude"

    # the verbatim-title case from the diagnosis still gets the gentler label unchanged
    verbatim = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2", quote="Moclobemide in Social Phobia",
    )
    _, guard_reason = screener.apply_decision_guard(verbatim, NO_ABSTRACT_SHOWN_TEXT, {"I2"})
    assert guard_reason == "no_abstract"


def test_guard_no_abstract_check_does_not_apply_to_needs_review():
    """The rule is scoped to EXCLUDE (S11 task scope): a model-native NEEDS_REVIEW on a
    no-abstract record is untouched by this check -- it already routes through the
    pre-existing NEEDS_REVIEW branch (criterion/quote attribution), which S11 does not
    change."""
    decision = screener.PaperDecision(
        decision="NEEDS_REVIEW", criterion="", reason="no abstract shown",
    )
    result, guard_reason = screener.apply_decision_guard(decision, NO_ABSTRACT_SHOWN_TEXT, {"E1"})
    assert result is decision
    assert guard_reason == ""


def test_guard_no_abstract_check_does_not_fire_on_a_record_with_a_real_abstract():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1"})
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


# --- 6b. Full-text inclusion/exclusion and cut-abstract demotions ----


def test_guard_demotes_exclude_naming_a_full_text_exclusion_id_with_a_verbatim_quote():
    """An EXCLUDE naming a full-text exclusion criterion is demoted to
    "full_text_criterion", not "unanchored_exclude", when its quote is verbatim -- the model
    broke the never-EXCLUDE rule, but at least anchored the claim in the shown text."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert result.criterion == "E1"  # kept for audit
    assert result.quote == "uses rats exclusively"  # kept for audit
    assert "guard" in result.reason.lower()
    assert guard_reason == "full_text_criterion"


def test_guard_demotes_exclude_naming_a_full_text_exclusion_id_without_a_verbatim_quote():
    """The gentler label is not owed to an EXCLUDE that both breaks the never-EXCLUDE rule
    and fabricates its quote."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="a phrase never shown"
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_exclude"


def test_guard_demotes_exclude_naming_a_full_text_inclusion_id_as_unanchored_regardless_of_quote():
    """A full-text inclusion criterion is never tested at abstract stage at all, so an
    EXCLUDE naming one is a rule the prompt forbids outright: demoted to
    "unanchored_exclude" whatever the quote says -- there is no "full_text_criterion" reward
    for an EXCLUDE the model should never have made."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="I1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"I1"}, full_text_inclusion_ids={"I1"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert result.criterion == "I1"  # kept for audit
    assert guard_reason == "unanchored_exclude"


def test_guard_demotes_exclude_on_an_absence_criterion_in_a_cut_abstract():
    cut_text = (
        f"Title: T\nAbstract: head of the abstract{screener.ABSTRACT_CUT_MARKER}tail of it"
    )
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="head of the")
    result, guard_reason = screener.apply_decision_guard(
        decision, cut_text, {"E1"}, absence_ids={"E1"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "cut_abstract"


def test_guard_absence_check_does_not_fire_on_an_uncut_abstract():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="E1", quote="uses rats exclusively"
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"E1"}, absence_ids={"E1"},
    )
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


def test_guard_checks_run_in_order_full_text_exclusion_then_cut_abstract_then_unanchored():
    """full_text_exclusion_ids (behind a verbatim quote) is checked
    before absence_ids + a cut abstract, which is checked before the pre-existing unanchored
    test -- so a criterion that is both full-text-exclusion and absence-tested, quoted
    verbatim, is reported as full_text_criterion, not cut_abstract."""
    cut_text = f"Title: T\nAbstract: head{screener.ABSTRACT_CUT_MARKER}tail"
    decision = screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="head")
    _, guard_reason = screener.apply_decision_guard(
        decision, cut_text, {"E1"}, full_text_exclusion_ids={"E1"}, absence_ids={"E1"},
    )
    assert guard_reason == "full_text_criterion"
    # neither full-text nor absence-cut: falls through to the unanchored test
    unanchored = screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="never shown")
    _, guard_reason = screener.apply_decision_guard(
        unanchored, cut_text, {"E1"}, full_text_exclusion_ids=set(), absence_ids=set(),
    )
    assert guard_reason == "unanchored_exclude"


def test_guard_attributes_full_text_criterion_to_a_compliant_needs_review_with_a_quote():
    """A model-native NEEDS_REVIEW naming a full-text exclusion
    criterion is attributed "full_text_criterion" only when it also backs the claim with a
    non-empty, verbatim quote -- the decision was already NEEDS_REVIEW, so the returned
    object is the same instance, unchanged."""
    decision = screener.PaperDecision(
        decision="NEEDS_REVIEW", criterion="E1", quote="uses rats exclusively",
        reason="a different method was used",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result is decision
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "full_text_criterion"


def test_guard_attributes_unquoted_criterion_to_a_needs_review_with_no_verbatim_quote():
    """The fifth reason value: a NEEDS_REVIEW naming a full-text exclusion criterion without
    a non-empty, verbatim quote keeps its id and quote for audit but is not trusted as the
    amendment's contrary-evidence case."""
    no_quote = screener.PaperDecision(
        decision="NEEDS_REVIEW", criterion="E1", reason="cannot tell"
    )
    result, guard_reason = screener.apply_decision_guard(
        no_quote, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result is no_quote
    assert guard_reason == "unquoted_criterion"

    fabricated_quote = screener.PaperDecision(
        decision="NEEDS_REVIEW", criterion="E1", quote="never shown", reason="cannot tell",
    )
    _, guard_reason = screener.apply_decision_guard(
        fabricated_quote, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert guard_reason == "unquoted_criterion"


def test_guard_does_not_attribute_anything_to_a_needs_review_naming_no_criterion():
    """A NEEDS_REVIEW naming no criterion at all still gets "" -- there is nothing to check
    a quote against, so it stays indistinguishable from a genuine undecidable record."""
    no_criterion = screener.PaperDecision(decision="NEEDS_REVIEW", reason="cannot tell")
    result, guard_reason = screener.apply_decision_guard(
        no_criterion, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result is no_criterion
    assert guard_reason == ""


def test_guard_attributes_unquoted_criterion_to_an_abstract_stage_needs_review_without_a_quote():
    """The non-empty-and-verbatim quote test applies
    to any non-empty ``criterion``, not only a full-text exclusion id. A NEEDS_REVIEW naming an
    ordinary abstract-stage id with no quote is one of the amendment's own defect populations
    (section 2 still requires a criterion id and a verbatim quote there) and must not fall
    through to guard_reason == "", indistinguishable from a record the model legitimately
    could not decide."""
    abstract_stage = screener.PaperDecision(decision="NEEDS_REVIEW", criterion="I1")
    result, guard_reason = screener.apply_decision_guard(
        abstract_stage, SHOWN_TEXT, {"I1"}, full_text_exclusion_ids={"E1"},
    )
    assert result is abstract_stage
    assert guard_reason == "unquoted_criterion"


def test_guard_attributes_unquoted_criterion_to_a_full_text_inclusion_needs_review_without_quote():
    """The other defect population: a NEEDS_REVIEW naming a full-text *inclusion* id is a
    routing the prompt's section 3 bullet forbids outright, so it must not fall through to
    guard_reason == "" either -- it is attributed "unquoted_criterion" the same as any other
    criterion-naming NEEDS_REVIEW with no verbatim quote."""
    decision = screener.PaperDecision(
        decision="NEEDS_REVIEW", criterion="I1", reason="cannot tell"
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"I1"}, full_text_inclusion_ids={"I1"},
    )
    assert result is decision
    assert guard_reason == "unquoted_criterion"


def test_guard_never_attributes_full_text_criterion_to_an_include():
    decision = screener.PaperDecision(decision="INCLUDE", criterion="E1", reason="ok")
    result, guard_reason = screener.apply_decision_guard(
        decision, SHOWN_TEXT, {"E1"}, full_text_exclusion_ids={"E1"},
    )
    assert result is decision
    assert guard_reason == ""


def test_guard_never_returns_include():
    """An alternative design -- routing an EXCLUDE
    naming a full-text inclusion id to INCLUDE with a to-confirm note -- is rejected. The
    guard's only permitted transition stays EXCLUDE to NEEDS_REVIEW."""
    for criterion, kwargs in (
        ("I1", {"full_text_inclusion_ids": {"I1"}}),
        ("E1", {"full_text_exclusion_ids": {"E1"}}),
        ("E9", {}),
    ):
        decision = screener.PaperDecision(decision="EXCLUDE", criterion=criterion, quote="q")
        result, _ = screener.apply_decision_guard(decision, SHOWN_TEXT, {"E1", "I1"}, **kwargs)
        assert result.decision != "INCLUDE"


def test_full_text_inclusion_criterion_ids_matches_the_guards_internal_check():
    ids = screener.full_text_inclusion_criterion_ids(
        ["i1", "i2"], ["abstract", "full_text"],
    )
    assert ids == {"I2"}


def test_full_text_exclusion_criterion_ids_matches_the_guards_internal_check():
    ids = screener.full_text_exclusion_criterion_ids(
        ["e1", "e2"], ["full_text", "abstract"],
    )
    assert ids == {"E1"}


def test_full_text_inclusion_and_exclusion_criterion_ids_are_empty_with_no_stage_lists():
    assert screener.full_text_inclusion_criterion_ids(["i1"], None) == set()
    assert screener.full_text_exclusion_criterion_ids(["e1"], None) == set()


# --- 6d. The reserved TOPIC criterion id ----------------------------


def test_known_criterion_ids_always_includes_the_reserved_topic_id():
    """``_known_criterion_ids`` adds the reserved
    ``TOPIC`` id, whether or not the protocol names any criterion at all, so an EXCLUDE
    naming it with a verbatim quote is returned unchanged rather than demoted."""
    assert screener.TOPIC_CRITERION_ID == "TOPIC"
    assert "TOPIC" in screener._known_criterion_ids(None, None)
    assert "TOPIC" in screener._known_criterion_ids(["a", "b"], ["c"])


def _load_evaluation_common_quote_is_verbatim():
    # Loaded by file path, not by package import: the backend cannot import evaluation/
    # (separate environment, not a dependency), so the test reaches the file directly to
    # check that the private in-module copy agrees with it on a shared fixture. The module
    # is registered in sys.modules before exec so its own @dataclass fields (evaluated
    # lazily under `from __future__ import annotations`) can resolve their module globals.
    repo_root = Path(__file__).resolve().parents[2]
    common_path = repo_root / "evaluation" / "common.py"
    module_name = "_eval_common_for_screener_test"
    spec = importlib.util.spec_from_file_location(module_name, common_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module.quote_is_verbatim


def test_private_verbatim_check_agrees_with_evaluation_common_on_a_shared_fixture():
    eval_quote_is_verbatim = _load_evaluation_common_quote_is_verbatim()
    unicode_shown_text = "Title: T\nAbstract: A double‐blind trial reported “significant” gains."
    fixtures = [
        (SHOWN_TEXT, "uses rats exclusively", True),
        (SHOWN_TEXT, "USES   RATS   exclusively", True),
        (SHOWN_TEXT, "a phrase never shown", False),
        (SHOWN_TEXT, "", False),
        # record 1065: the guard's outcome must not flip on one U+2010 code point.
        # The shown text carries U+2010; the quote uses a plain ASCII hyphen.
        (unicode_shown_text, "double-blind trial", True),
        (unicode_shown_text, "double‐blind trial", True),
        (unicode_shown_text, '"significant" gains', True),
    ]
    for shown_text, quote, expected in fixtures:
        assert screener._quote_is_verbatim(quote, shown_text) == expected
        assert eval_quote_is_verbatim(quote, shown_text, casefold=True) == expected


# --- 6f. Anchored INCLUDE and unanchored_include -------

ANCHOR_SHOWN_TEXT = (
    "Title: A trial of hand hygiene reminders\n"
    "Abstract: Nurses in three ICUs received a computer screen saver hand hygiene "
    "reminder. Compliance improved by 12 percentage points."
)


def _anchor_decision(**anchors: str) -> "screener.PaperDecision":
    """An INCLUDE with one AnchorEntry per keyword arg (slot=quote)."""
    return screener.PaperDecision(
        decision="INCLUDE",
        anchors=[{"slot": slot, "quote": quote} for slot, quote in anchors.items()],
    )


def test_guard_passes_a_fully_anchored_include_with_all_three_slots_verbatim():
    decision = _anchor_decision(
        population="Nurses in three ICUs",
        subject="hand hygiene reminder",
        outcome="Compliance improved by 12 percentage points",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision == "INCLUDE"
    assert guard_reason == ""


def test_guard_demotes_an_include_missing_an_anchored_slot_entirely():
    decision = _anchor_decision(
        population="Nurses in three ICUs",
        outcome="Compliance improved by 12 percentage points",
    )  # no "subject" entry at all
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision == "NEEDS_REVIEW"
    assert "guard" in result.reason.lower()
    assert guard_reason == "unanchored_include"


def test_guard_demotes_an_include_with_a_not_established_anchored_slot():
    decision = _anchor_decision(
        population="Nurses in three ICUs",
        subject="not_established",
        outcome="Compliance improved by 12 percentage points",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_include"


def test_guard_demotes_an_include_with_a_fabricated_anchor_quote():
    decision = _anchor_decision(
        population="Nurses in three ICUs",
        subject="a phrase never shown in the abstract",
        outcome="Compliance improved by 12 percentage points",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_include"


def test_guard_only_checks_the_slots_this_run_treats_as_anchored():
    """A slot outside ``anchored_slots`` is never checked, even when it is missing or
    fabricated -- the offline re-scoring design depends on this: the same ledger
    scores differently for "population only" vs "all three" without another model call."""
    decision = _anchor_decision(population="Nurses in three ICUs")  # subject/outcome absent
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset({"population"}),
    )
    assert result.decision == "INCLUDE"
    assert guard_reason == ""


def test_guard_with_no_anchored_slots_never_demotes_an_include():
    """``anchored_slots`` empty (the default) is a no-op: identical to v2's guard, whatever
    the model's own ledger contains -- an existing caller that never passes it is unaffected."""
    decision = _anchor_decision()  # no anchors at all
    result, guard_reason = screener.apply_decision_guard(decision, ANCHOR_SHOWN_TEXT, set())
    assert result.decision == "INCLUDE"
    assert guard_reason == ""


def test_guard_unanchored_include_keeps_the_ledger_for_audit():
    decision = _anchor_decision(population="Nurses in three ICUs")
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset({"subject"}),
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_include"
    assert result.anchors == decision.anchors


def test_anchor_entry_normalises_a_not_established_answer_under_any_casing():
    entry = screener.AnchorEntry(slot="Subject", quote="  Not_Established  ")
    assert entry.slot == "subject"
    assert entry.quote == screener.NOT_ESTABLISHED


def test_guard_never_returns_include_still_holds_with_anchored_slots(monkeypatch=None):
    """The one new INCLUDE-side transition is EXCLUDE-shaped in reverse: INCLUDE can only
    ever become NEEDS_REVIEW here, never EXCLUDE."""
    decision = _anchor_decision()
    result, _ = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, set(), anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision in ("INCLUDE", "NEEDS_REVIEW")


# --- 6g. Title-only TOPIC exclude on no-abstract records

NO_ABSTRACT_OFF_TOPIC_TEXT = (
    f"Title: Resuscitation guidelines for cardiac arrest\n"
    f"Abstract: {screener.NO_ABSTRACT_PLACEHOLDER}"
)


def test_guard_no_abstract_topic_exclude_stands_when_the_quote_is_verbatim_in_the_title():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="TOPIC", quote="Resuscitation guidelines for cardiac arrest",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, NO_ABSTRACT_OFF_TOPIC_TEXT, {"TOPIC"},
    )
    assert result.decision == "EXCLUDE"
    assert guard_reason == ""


def test_guard_no_abstract_topic_exclude_is_demoted_when_the_quote_is_not_in_the_title():
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="TOPIC", quote="a phrase never shown anywhere",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, NO_ABSTRACT_OFF_TOPIC_TEXT, {"TOPIC"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "unanchored_exclude"


def test_guard_no_abstract_numbered_criterion_exclude_is_still_demoted_not_carved_out():
    """The carve-out is scoped to criterion "TOPIC" only:
    a numbered-criterion EXCLUDE quoting only the title still gets the ordinary "no_abstract"
    label, exactly as it did before this design (see
    ``test_guard_no_abstract_fires_even_when_the_quote_is_a_verbatim_substring_of_the_title``
    above, unchanged)."""
    decision = screener.PaperDecision(
        decision="EXCLUDE", criterion="I2",
        quote="Resuscitation guidelines for cardiac arrest",
    )
    result, guard_reason = screener.apply_decision_guard(
        decision, NO_ABSTRACT_OFF_TOPIC_TEXT, {"I2"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "no_abstract"


def test_shown_title_extracts_the_title_line_from_a_rendered_shown_text():
    assert screener._shown_title(SHOWN_TEXT) == "T"
    assert (
        screener._shown_title(NO_ABSTRACT_OFF_TOPIC_TEXT)
        == "Resuscitation guidelines for cardiac arrest"
    )
    assert screener._shown_title("") == ""
    assert screener._shown_title(None) == ""


# --- 6h. The inclusion-only second pass's guard --------


def test_apply_second_pass_guard_demotes_an_include_on_not_established():
    status, guard_reason = screener.apply_second_pass_guard("INCLUDE", "not_established")
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_leaves_a_confirmed_include_unchanged():
    status, guard_reason = screener.apply_second_pass_guard("INCLUDE", "confirmed")
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_second_pass_guard_never_touches_a_non_include_status():
    for status in ("EXCLUDE", "NEEDS_REVIEW"):
        result, guard_reason = screener.apply_second_pass_guard(status, "not_established")
        assert result == status
        assert guard_reason == ""


def test_anchor_entry_slot_and_quote_are_normalised_independently_of_second_pass():
    entry = screener.AnchorEntry(slot="  Population ", quote="a verbatim phrase")
    assert entry.slot == "population"
    assert entry.quote == "a verbatim phrase"


# --- Agent configuration -------------------------------------------------------------------


def test_agent_is_deterministic_and_keeps_the_fast_timeout_tier():
    screener._agents.clear()
    agent = screener.get_relevance_screener_agent()
    assert agent.model_settings["temperature"] == 0.0
    assert agent.model_settings["timeout"] == DETERMINISTIC_FAST_MODEL_SETTINGS["timeout"]
    assert agent.model_settings["timeout"] == settings.llm_timeout_seconds
    # The model is the DeepSeek chat model (best-effort system_fingerprint subclass).
    assert agent.model.model_name == settings.deepseek_model
    assert agent.model.system == "deepseek"


def test_get_relevance_screener_agent_caches_one_agent_per_prompt_version():
    """A v1 call and a default (v3) call
    must each reach an agent instructed with the matching system prompt, and neither may
    overwrite the other's cached agent."""
    screener._agents.clear()
    agent_default = screener.get_relevance_screener_agent()
    agent_v1 = screener.get_relevance_screener_agent(prompt_version="v1")
    assert agent_v1 is not agent_default
    assert agent_v1._instructions == [screener.SCREENER_PROMPT_V1]
    assert agent_default._instructions == [screener.SCREENER_PROMPT_V3]
    # Cached, not rebuilt, on a second call for the same version.
    assert screener.get_relevance_screener_agent(prompt_version="v1") is agent_v1
    assert screener.get_relevance_screener_agent() is agent_default


def test_get_relevance_screener_agent_v2_still_reachable_explicitly():
    """v2 is no longer the default, but it stays a selectable prompt_version, on
    its own cached agent, since a v2 evaluation run must still be able to reproduce it."""
    screener._agents.clear()
    agent_v2 = screener.get_relevance_screener_agent(prompt_version="v2")
    agent_v3 = screener.get_relevance_screener_agent(prompt_version="v3")
    assert agent_v2 is not agent_v3
    assert agent_v2._instructions == [screener.SCREENER_PROMPT_V2]
    assert agent_v3._instructions == [screener.SCREENER_PROMPT_V3]


def test_get_relevance_screener_agent_rejects_an_unknown_prompt_version():
    with pytest.raises(ValueError):
        screener.get_relevance_screener_agent(prompt_version="v4")


# --- screen_papers: decisions, guard, padding, provenance ---------------------------------


@pytest.mark.asyncio
async def test_screen_papers_returns_statuses_criteria_quotes_reasons_and_provenance():
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="Directly studies the topic"),
        screener.PaperDecision(
            decision="EXCLUDE",
            criterion="E1",
            quote="something else",
            reason="Different field, shared keyword",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="Applies the topic's framework"),
    ]
    run = AsyncMock(return_value=_fake_run_result(decisions))
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers(
            "teacher burnout", PAPERS, exclusion_criteria=["different field"]
        )

    assert isinstance(result, screener.ScreeningBatchResult)
    assert result.statuses == ["INCLUDE", "EXCLUDE", "INCLUDE"]
    assert result.criteria_ids == ["", "E1", ""]
    assert result.quotes == ["", "something else", ""]
    assert result.include == [True, False, True]
    assert result.reasons == [
        "Directly studies the topic",
        "Different field, shared keyword",
        "Applies the topic's framework",
    ]
    prov = result.provenance
    assert isinstance(prov, LLMCallProvenance)
    assert prov.agent == "relevance_screener"
    assert prov.model_configured == settings.deepseek_model
    assert prov.model_reported == "deepseek-v4-flash"
    assert prov.system_fingerprint == "fp_abc"
    assert prov.provider_response_id == "resp-42"
    assert prov.temperature == 0.0
    assert prov.prompt_version == screener.SCREENER_PROMPT_VERSION
    assert prov.input_tokens == 321
    assert prov.output_tokens == 45
    # The user prompt carries the topic and every paper.
    sent_prompt = run.await_args.args[0]
    assert "teacher burnout" in sent_prompt
    assert "Paper C" in sent_prompt
    assert result.padded == 0


@pytest.mark.asyncio
async def test_screen_papers_guards_an_unanchored_exclude_from_the_model():
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E9", quote="not in the text"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert "guard" in result.reasons[0].lower()
    assert result.include[0] is False


@pytest.mark.asyncio
async def test_screen_papers_counts_a_guard_conversion():
    """An unanchored EXCLUDE the guard demotes must
    be counted, both as a total and per record, so downstream gate checks have a signal."""
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E9", quote="not in the text"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.guard_conversions == 1
    assert result.guard_applied == [True, False, False]


@pytest.mark.asyncio
async def test_screen_papers_populates_the_provenance_guard_conversions():
    """``LLMCallProvenance`` gains ``guard_conversions``, populated by the agent
    from the same count as ``ScreeningBatchResult.guard_conversions``."""
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E9", quote="not in the text"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.provenance.guard_conversions == 1 == result.guard_conversions


@pytest.mark.asyncio
async def test_screen_papers_guard_conversions_is_zero_for_a_clean_batch():
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(
            decision="EXCLUDE", criterion="E1", quote="something else", reason="off topic"
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.guard_conversions == 0
    assert result.guard_applied == [False, False, False]
    assert result.guard_reasons == ["", "", ""]


@pytest.mark.asyncio
async def test_screen_papers_demotes_an_exclude_naming_a_full_text_exclusion_id_with_a_quote():
    """An EXCLUDE naming a criterion the caller marked full-text
    exclusion is routed to NEEDS_REVIEW with guard_reason "full_text_criterion" when its
    quote is verbatim against the shown text."""
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="E1", quote="About the topic", reason="off topic"
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS, exclusion_criteria=["off topic"], exclusion_stages=["full_text"],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "full_text_criterion"
    assert result.guard_applied == [True, False, False]
    assert result.guard_reasons[1:] == ["", ""]


@pytest.mark.asyncio
async def test_screen_papers_demotes_an_exclude_naming_a_full_text_exclusion_id_without_a_quote():
    """Without a verbatim quote the demotion is "unanchored_exclude", not the gentler
    "full_text_criterion" label -- the model both broke the never-EXCLUDE rule and
    fabricated its evidence."""
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="E1", quote="About something else", reason="off topic"
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS, exclusion_criteria=["off topic"], exclusion_stages=["full_text"],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "unanchored_exclude"


@pytest.mark.asyncio
async def test_screen_papers_demotes_an_exclude_naming_a_full_text_inclusion_id():
    """A full-text inclusion criterion is never tested at abstract stage at all: an EXCLUDE
    naming one is unanchored regardless of its quote."""
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="I1", quote="About the topic",
            reason="not peer reviewed",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS,
            inclusion_criteria=["is peer reviewed"], inclusion_stages=["full_text"],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "unanchored_exclude"


@pytest.mark.asyncio
async def test_screen_papers_attributes_full_text_criterion_to_a_compliant_needs_review_quoted():
    """A compliant model NEEDS_REVIEW naming a full-text exclusion
    criterion, backed by a verbatim quote, surfaces guard_reason == "full_text_criterion"
    (not "" or "undecidable") and is not counted as a guard conversion (no demotion
    happened)."""
    decisions = [
        screener.PaperDecision(
            decision="NEEDS_REVIEW", criterion="E1", quote="About the topic",
            reason="a different method was used",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS, exclusion_criteria=["off topic"], exclusion_stages=["full_text"],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.criteria_ids[0] == "E1"
    assert result.guard_reasons[0] == "full_text_criterion"
    assert result.guard_conversions == 0
    assert result.guard_applied == [False, False, False]


@pytest.mark.asyncio
async def test_screen_papers_attributes_unquoted_criterion_to_a_needs_review_without_a_quote():
    """The same compliant NEEDS_REVIEW, without a verbatim quote, surfaces guard_reason ==
    "unquoted_criterion" instead: the model named the criterion but did not back it."""
    decisions = [
        screener.PaperDecision(decision="NEEDS_REVIEW", criterion="E1", reason="cannot tell"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS, exclusion_criteria=["off topic"], exclusion_stages=["full_text"],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "unquoted_criterion"
    assert result.guard_conversions == 0


# --- 6c. to_confirm ---------------------------------------------------


@pytest.mark.asyncio
async def test_screen_papers_defaults_to_confirm_to_the_full_set_when_the_model_leaves_it_empty():
    """An INCLUDE with no (or an empty) to_confirm list defaults to
    every full-text inclusion id the protocol names -- silence never means "confirmed"."""
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS,
            inclusion_criteria=["uses LASSI", "reports an effect size"],
            inclusion_stages=["full_text", "full_text"],
        )

    assert result.to_confirm == [["I1", "I2"], ["I1", "I2"], ["I1", "I2"]]


@pytest.mark.asyncio
async def test_screen_papers_intersects_to_confirm_with_the_protocols_full_text_inclusion_ids():
    """A model-named id outside the protocol's full-text inclusion set is dropped, and one
    the model already confirmed is not defaulted back in."""
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok", to_confirm=["I2", "E1"]),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS,
            inclusion_criteria=["uses LASSI", "reports an effect size"],
            inclusion_stages=["full_text", "full_text"],
        )

    # "E1" is not a known inclusion id, so it is dropped; only "I2" survives the intersection.
    assert result.to_confirm[0] == ["I2"]


@pytest.mark.asyncio
async def test_screen_papers_forces_to_confirm_empty_on_every_non_include():
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="something else"),
        screener.PaperDecision(decision="NEEDS_REVIEW", reason="cannot tell"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS,
            inclusion_criteria=["uses LASSI"], inclusion_stages=["full_text"],
            exclusion_criteria=["off topic"],
        )

    assert result.statuses[1] == "EXCLUDE"
    assert result.statuses[2] == "NEEDS_REVIEW"
    assert result.to_confirm[1] == []
    assert result.to_confirm[2] == []


@pytest.mark.asyncio
async def test_screen_papers_to_confirm_is_empty_with_no_full_text_inclusion_criteria():
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok") for _ in PAPERS]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS)

    assert result.to_confirm == [[], [], []]


# --- 6d. The reserved TOPIC id, through screen_papers -------------


@pytest.mark.asyncio
async def test_screen_papers_exclude_naming_topic_with_a_verbatim_quote_survives_the_guard():
    """An off-topic EXCLUDE naming the reserved TOPIC id, backed by a
    verbatim quote, is returned unchanged -- not demoted as unanchored_exclude."""
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="TOPIC", quote="About the topic",
            reason="does not concern the population the question names",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "EXCLUDE"
    assert result.criteria_ids[0] == "TOPIC"
    assert result.guard_reasons[0] == ""


@pytest.mark.asyncio
async def test_screen_papers_exclude_naming_topic_without_a_verbatim_quote_is_unanchored():
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="TOPIC", quote="a phrase never shown",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "unanchored_exclude"


@pytest.mark.asyncio
async def test_screen_papers_with_no_protocol_at_all_keeps_a_verbatim_anchored_exclude():
    """``_known_criterion_ids`` always adds the
    reserved ``TOPIC`` id, so ``known_ids`` is never empty even with no protocol criteria at
    all. The guard's empty-protocol fallback (``criterion_ok = ... if known_ids else True``)
    must key off whether the protocol names any *numbered* criterion, not off whether
    ``known_ids`` itself is empty -- otherwise a compliant EXCLUDE with an empty criterion id
    (the v2 user turn tells the model to leave it empty when no protocol was supplied) and a
    verbatim quote is wrongly demoted. This is the default product path:
    ``SmartSearchRequest.inclusion_criteria``/``exclusion_criteria`` default to ``[]``, which
    ``backend/app/api/smart_search.py`` forwards as ``None``, landing here with
    ``inclusion_criteria=None, exclusion_criteria=None``."""
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="", quote="About the topic", reason="off topic",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS, inclusion_criteria=None, exclusion_criteria=None,
        )

    assert result.statuses[0] == "EXCLUDE"
    assert result.guard_reasons[0] == ""
    assert result.guard_applied == [False, False, False]
    assert result.guard_conversions == 0


@pytest.mark.asyncio
async def test_screen_papers_topic_is_never_attributed_full_text_criterion():
    """TOPIC is in neither full-text set, so a NEEDS_REVIEW naming it with a verbatim quote
    is left unattributed ("") -- never "full_text_criterion", which is reserved for a
    full-text exclusion id backed by a verbatim quote."""
    decisions = [
        screener.PaperDecision(
            decision="NEEDS_REVIEW", criterion="TOPIC", quote="About the topic",
            reason="ambiguous",
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == ""


@pytest.mark.asyncio
async def test_screen_papers_topic_never_enters_to_confirm():
    """TOPIC is intersected out of to_confirm like any id outside the protocol's full-text
    inclusion set -- it can in practice only ever appear on an EXCLUDE, never an INCLUDE, but
    the resolution is defensive regardless of what the model returns."""
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok", to_confirm=["TOPIC", "I1"]),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS,
            inclusion_criteria=["uses LASSI"], inclusion_stages=["full_text"],
        )

    assert result.to_confirm[0] == ["I1"]


@pytest.mark.asyncio
async def test_screen_papers_demotes_an_exclude_on_an_absence_criterion_in_a_cut_abstract():
    long_abstract = "x" * 20000  # exceeds ABSTRACT_CHAR_LIMIT, forces a cut
    papers = [{"title": "T", "abstract": long_abstract}]
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="E1", quote=long_abstract[:20], reason="not found"
        ),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", papers, exclusion_criteria=["off topic"], exclusion_absence=[True],
        )

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "cut_abstract"


# --- 6e. The no-abstract guard rule, through screen_papers ----------------------


@pytest.mark.asyncio
async def test_screen_papers_demotes_an_exclude_on_a_record_with_no_abstract():
    """End to end: a paper with no abstract at all, EXCLUDEd on a real criterion
    with a quote copied from its title, is demoted to NEEDS_REVIEW with guard_reason
    "no_abstract" -- a real example (a drug-trial title, no abstract shown)."""
    papers = [{"title": "Moclobemide in Social Phobia", "abstract": None}]
    decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="I2", quote="Moclobemide in Social Phobia",
            reason="Moclobemide is a medication, not CBT; no abstract shown.",
        ),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", papers, exclusion_criteria=["uses CBT"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "no_abstract"
    assert result.guard_applied == [True]
    assert result.criteria_ids[0] == "I2"  # kept for audit


@pytest.mark.asyncio
async def test_screen_papers_no_abstract_guard_reason_survives_an_empty_string_abstract():
    papers = [{"title": "T", "abstract": ""}]
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="T", reason="off"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", papers, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "no_abstract"


@pytest.mark.asyncio
async def test_screen_papers_no_abstract_guard_reason_survives_a_whitespace_only_abstract():
    """A whitespace-only abstract must land on
    guard_reason "no_abstract" end to end through ``screen_papers``, the same as an empty
    or missing one, since ``build_shown_texts`` strips before falling back to the
    placeholder."""
    papers = [{"title": "T", "abstract": "   "}]
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="T", reason="off"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", papers, exclusion_criteria=["off topic"])

    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "no_abstract"


@pytest.mark.asyncio
async def test_screen_papers_v1_sends_the_v1_prompt_and_leaves_an_exclude_undemoted():
    """A --prompt-version v1 call must send the v1
    system prompt, record the v1 sha in provenance, and never run the guard -- v1 asks for
    no criterion and no quote, so an EXCLUDE with neither must survive."""
    decisions = [
        screener.PaperDecision(decision="EXCLUDE", reason="off topic"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    run = AsyncMock(return_value=_fake_run_result(decisions))
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("teacher burnout", PAPERS, prompt_version="v1")

    assert result.statuses[0] == "EXCLUDE"  # not demoted: the guard does not run for v1
    assert result.guard_conversions == 0
    assert result.guard_applied == [False, False, False]
    assert result.provenance.prompt_version == screener.SCREENER_PROMPT_V1_VERSION
    sent_prompt = run.await_args.args[0]
    assert "Papers to screen (title and abstract):" in sent_prompt
    assert "Records to screen (title and abstract):" not in sent_prompt


@pytest.mark.asyncio
async def test_screen_papers_v1_abstract_limit_reproduces_the_committed_truncation():
    papers = [{"title": "T", "abstract": "z" * 600}]
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    run = AsyncMock(return_value=_fake_run_result(decisions))
    with patch.object(Agent, "run", new=run):
        await screener.screen_papers(
            "q", papers, prompt_version="v1", abstract_limit=500
        )

    sent_prompt = run.await_args.args[0]
    assert "z" * 500 + "..." in sent_prompt
    assert screener.ABSTRACT_CUT_MARKER not in sent_prompt


@pytest.mark.asyncio
async def test_screen_papers_pads_missing_decisions_with_needs_review_and_a_reason():
    """The one-time re-ask also comes back short here, so the two
    records neither call decided are padded."""
    decisions = [screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="the topic")]
    run = AsyncMock(side_effect=[_fake_run_result(decisions), _fake_run_result([])])
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert run.await_count == 2
    assert result.statuses == ["EXCLUDE", "NEEDS_REVIEW", "NEEDS_REVIEW"]
    assert result.include == [False, False, False]
    assert result.reasons[1] == screener.NO_DECISION_REASON
    assert result.reasons[2] == screener.NO_DECISION_REASON
    assert result.reasons[1] != ""
    assert result.padded == 2


@pytest.mark.asyncio
async def test_screen_papers_padded_is_zero_when_every_paper_gets_a_decision():
    """A harness scoring a v1-completed run asserts
    ``padded == 0`` before treating it as comparable to the six committed v1 runs."""
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok") for _ in PAPERS]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS)

    assert result.padded == 0


@pytest.mark.asyncio
async def test_screen_papers_reasks_missing_decisions_once_before_padding():
    """A short batch response is re-asked once, as a fresh batch
    containing only the records the first call left undecided, before anything is padded."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    second_decisions = [
        screener.PaperDecision(
            decision="EXCLUDE", criterion="E1", quote="something else", reason="off topic"
        ),
        screener.PaperDecision(decision="INCLUDE", reason="ok too"),
    ]
    run = AsyncMock(
        side_effect=[_fake_run_result(first_decisions), _fake_run_result(second_decisions)]
    )
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert run.await_count == 2
    second_prompt = run.await_args_list[1].args[0]
    assert "Paper B" in second_prompt
    assert "Paper C" in second_prompt
    assert "Paper A" not in second_prompt

    assert result.statuses == ["INCLUDE", "EXCLUDE", "INCLUDE"]
    assert result.reasons[1] != screener.NO_DECISION_REASON
    assert result.padded == 0


@pytest.mark.asyncio
async def test_screen_papers_guards_a_decision_the_reask_call_returned():
    """The guard runs on a reasked decision exactly as it would on a first-call one, so a
    guard conversion produced by a re-ask is counted the same way."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    second_decisions = [
        screener.PaperDecision(decision="EXCLUDE", criterion="E9", quote="not shown"),
        screener.PaperDecision(decision="INCLUDE", reason="ok"),
    ]
    run = AsyncMock(
        side_effect=[_fake_run_result(first_decisions), _fake_run_result(second_decisions)]
    )
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS, exclusion_criteria=["off topic"])

    assert result.statuses[1] == "NEEDS_REVIEW"  # guard demoted the reasked EXCLUDE
    assert result.guard_conversions == 1
    assert result.guard_applied == [False, True, False]
    assert result.padded == 0


@pytest.mark.asyncio
async def test_screen_papers_pads_only_what_the_reask_still_leaves_undecided():
    """The re-ask itself can also come back short; only the still-missing records are
    padded, and the reported ``padded`` count reflects only those."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    second_decisions: list[screener.PaperDecision] = []
    run = AsyncMock(
        side_effect=[_fake_run_result(first_decisions), _fake_run_result(second_decisions)]
    )
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS)

    assert run.await_count == 2
    assert result.statuses == ["INCLUDE", "NEEDS_REVIEW", "NEEDS_REVIEW"]
    assert result.reasons[1] == screener.NO_DECISION_REASON
    assert result.reasons[2] == screener.NO_DECISION_REASON
    assert result.padded == 2


@pytest.mark.asyncio
async def test_screen_papers_pads_when_the_reask_call_itself_raises():
    """A re-ask that fails outright (network error, model error) degrades to padding the
    still-missing records rather than failing the whole batch."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    run = AsyncMock(side_effect=[_fake_run_result(first_decisions), TimeoutError("timed out")])
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS)

    assert run.await_count == 2
    assert result.statuses == ["INCLUDE", "NEEDS_REVIEW", "NEEDS_REVIEW"]
    assert result.padded == 2


@pytest.mark.asyncio
async def test_screen_papers_does_not_reask_when_nothing_is_missing():
    """No short tail at all: exactly one model call, unchanged from before this fix."""
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok") for _ in PAPERS]
    run = AsyncMock(return_value=_fake_run_result(decisions))
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS)

    assert run.await_count == 1
    assert result.padded == 0


@pytest.mark.asyncio
async def test_screen_papers_v1_also_reasks_before_padding():
    """The re-ask sits before the v1/v2 branch, so a v1 call is re-asked too."""
    papers = [{"title": "T1", "abstract": "A1"}, {"title": "T2", "abstract": "A2"}]
    first = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    second = [screener.PaperDecision(decision="INCLUDE", reason="ok too")]
    run = AsyncMock(side_effect=[_fake_run_result(first), _fake_run_result(second)])
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", papers, prompt_version="v1")

    assert run.await_count == 2
    assert result.padded == 0
    assert result.statuses == ["INCLUDE", "INCLUDE"]


# --- Re-ask provenance is not dropped ----------------


@pytest.mark.asyncio
async def test_screen_papers_reask_provenance_carries_the_second_calls_usage_and_identity():
    """The re-ask is a second paid call: its own tokens, model and fingerprint must stay
    visible on ``reask_provenance``, not merged into or dropped from ``provenance``."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    second_decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="ok too"),
        screener.PaperDecision(decision="INCLUDE", reason="ok three"),
    ]
    run = AsyncMock(
        side_effect=[
            _fake_run_result(first_decisions, input_tokens=1000, output_tokens=100),
            _fake_run_result(
                second_decisions,
                input_tokens=700,
                output_tokens=70,
                model_name="deepseek-v4-flash-2",
                system_fingerprint="fp_reask",
            ),
        ]
    )
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS)

    assert run.await_count == 2
    assert result.padded == 0
    assert result.provenance.input_tokens == 1000
    assert result.provenance.output_tokens == 100
    assert result.provenance.model_reported == "deepseek-v4-flash"
    assert result.provenance.system_fingerprint == "fp_abc"
    assert result.reask_provenance is not None
    assert result.reask_provenance.input_tokens == 700
    assert result.reask_provenance.output_tokens == 70
    assert result.reask_provenance.model_reported == "deepseek-v4-flash-2"
    assert result.reask_provenance.system_fingerprint == "fp_reask"
    total_input = result.provenance.input_tokens + result.reask_provenance.input_tokens
    total_output = result.provenance.output_tokens + result.reask_provenance.output_tokens
    assert (total_input, total_output) == (1700, 170)


@pytest.mark.asyncio
async def test_screen_papers_reask_provenance_is_none_without_a_reask():
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok") for _ in PAPERS]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS)

    assert result.reask_provenance is None
    assert result.reask_error is None


@pytest.mark.asyncio
async def test_screen_papers_reask_provenance_and_error_when_the_reask_call_raises():
    """A still-padded record caused by the re-ask call itself
    raising must be distinguishable from one caused by the model quietly skipping it again."""
    first_decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")]
    run = AsyncMock(side_effect=[_fake_run_result(first_decisions), TimeoutError("timed out")])
    with patch.object(Agent, "run", new=run):
        result = await screener.screen_papers("q", PAPERS)

    assert result.padded == 2
    assert result.reask_provenance is None
    assert result.reask_error == "TimeoutError: timed out"


@pytest.mark.asyncio
async def test_screen_papers_truncates_surplus_decisions():
    decisions = [screener.PaperDecision(decision="INCLUDE", reason=f"r{i}") for i in range(5)]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers("q", PAPERS)

    assert len(result.include) == len(PAPERS) == len(result.reasons) == len(result.statuses)


@pytest.mark.asyncio
async def test_screen_papers_raises_screening_error_instead_of_including_everything():
    with patch.object(Agent, "run", new=AsyncMock(side_effect=TimeoutError("model timed out"))):
        with pytest.raises(screener.ScreeningError) as excinfo:
            await screener.screen_papers("q", PAPERS)

    # Root-cause class and message stay visible for the audit record.
    assert "TimeoutError" in str(excinfo.value)
    assert "model timed out" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_screen_papers_with_no_papers_raises_without_a_model_call():
    """An empty batch is a caller error: no model call, no result without provenance."""
    run = AsyncMock()
    with patch.object(Agent, "run", new=run):
        with pytest.raises(screener.ScreeningError, match="empty batch"):
            await screener.screen_papers("q", [])

    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_screen_papers_raises_when_the_model_returns_no_decisions():
    """Zero decisions is a failed call, not a short tail to pad with INCLUDE."""
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result([]))):
        with pytest.raises(screener.ScreeningError, match="no decisions"):
            await screener.screen_papers("q", PAPERS)


@pytest.mark.asyncio
async def test_screen_papers_wraps_provenance_failures_in_screening_error():
    """Every failure inside screen_papers surfaces as ScreeningError (plan contract)."""
    decisions = [screener.PaperDecision(decision="INCLUDE", reason="ok")] * len(PAPERS)
    with (
        patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))),
        patch.object(
            screener, "provenance_from_run", side_effect=RuntimeError("provenance broke")
        ),
    ):
        with pytest.raises(screener.ScreeningError) as excinfo:
            await screener.screen_papers("q", PAPERS)

    assert "RuntimeError: provenance broke" in str(excinfo.value)


# --- Best-effort system_fingerprint capture (model_config.build_deepseek_model) -----------


def _chat_completion(fingerprint: str | None):
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    return ChatCompletion(
        id="resp-1",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        created=1,
        model="deepseek-v4-flash",
        object="chat.completion",
        system_fingerprint=fingerprint,
    )


def test_deepseek_model_copies_system_fingerprint_into_provider_details():
    from app.agents.model_config import DeepSeekChatModel, build_deepseek_model

    model = build_deepseek_model("deepseek-chat")
    assert isinstance(model, DeepSeekChatModel)
    response = model._process_response(_chat_completion("fp_deepseek_123"))
    assert response.model_name == "deepseek-v4-flash"
    assert response.provider_details["system_fingerprint"] == "fp_deepseek_123"
    # pydantic-ai's own mapping is preserved.
    assert response.provider_details["finish_reason"] == "stop"


def test_deepseek_model_degrades_to_none_without_fingerprint():
    from app.agents.model_config import build_deepseek_model

    model = build_deepseek_model("deepseek-chat")
    response = model._process_response(_chat_completion(None))
    assert "system_fingerprint" not in (response.provider_details or {})


def test_attach_system_fingerprint_helper_is_defensive():
    from app.agents.model_config import attach_system_fingerprint

    # No provider_details yet -> a dict is created.
    target = SimpleNamespace(provider_details=None)
    attach_system_fingerprint(target, SimpleNamespace(system_fingerprint="fp_x"))
    assert target.provider_details == {"system_fingerprint": "fp_x"}

    # Raw response without the attribute -> untouched.
    target = SimpleNamespace(provider_details={"finish_reason": "stop"})
    attach_system_fingerprint(target, SimpleNamespace())
    assert target.provider_details == {"finish_reason": "stop"}

    # A read-only target must not raise.
    class _Frozen:
        provider_details = None

        def __setattr__(self, name, value):
            raise AttributeError("frozen")

    attach_system_fingerprint(_Frozen(), SimpleNamespace(system_fingerprint="fp_y"))


# --- confirm_inclusions (the second-pass agent call) ----

SECOND_PASS_PAPERS = [
    {"title": "A trial of hand hygiene reminders", "abstract": "Nurses in three ICUs."},
    {"title": "A survey of feedback beliefs", "abstract": "Teachers describe their views."},
]


def _fake_second_pass_run_result(
    answers: list["screener.SecondPassAnswer"],
    *,
    input_tokens: int = 111,
    output_tokens: int = 22,
    model_name: str = "deepseek-v4-flash",
    system_fingerprint: str = "fp_second",
):
    response = SimpleNamespace(
        model_name=model_name,
        provider_response_id="resp-second-1",
        provider_details={"system_fingerprint": system_fingerprint, "finish_reason": "stop"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        output=screener.SecondPassAnswers(answers=answers),
        response=response,
        usage=lambda: usage,
    )


def test_build_second_pass_prompt_lists_every_record_as_include():
    prompt = screener._build_second_pass_prompt(SECOND_PASS_PAPERS)
    assert "Record 1:" in prompt
    assert "Record 2:" in prompt
    assert "Nurses in three ICUs" in prompt
    assert "Teachers describe their views" in prompt


async def test_confirm_inclusions_returns_verdicts_missing_slots_anchors_and_provenance():
    answers = [
        screener.SecondPassAnswer(
            verdict="confirmed",
            anchors=[
                {"slot": "population", "quote": "Nurses in three ICUs"},
                {"slot": "subject", "quote": "hand hygiene reminders"},
                {"slot": "outcome", "quote": "Compliance improved"},
            ],
        ),
        screener.SecondPassAnswer(verdict="not_established", missing_slot="outcome"),
    ]
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_run_result(answers))
    ):
        result = await screener.confirm_inclusions(SECOND_PASS_PAPERS)
    assert result.verdicts == ["confirmed", "not_established"]
    assert result.missing_slots == ["", "outcome"]
    assert result.anchors[0] == [
        {"slot": "population", "quote": "Nurses in three ICUs"},
        {"slot": "subject", "quote": "hand hygiene reminders"},
        {"slot": "outcome", "quote": "Compliance improved"},
    ]
    assert result.anchors[1] == []
    assert result.provenance.model_configured == settings.deepseek_model
    assert result.padded == 0


async def test_confirm_inclusions_pads_a_short_response_with_not_established():
    """A response short of the batch is padded to "not_established", never "confirmed" --
    the same "never default to a lenient outcome silently" rule the rest of the module
    applies to a missing screening decision."""
    answers = [screener.SecondPassAnswer(verdict="confirmed")]  # only 1 of 2 papers answered
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_run_result(answers))
    ):
        result = await screener.confirm_inclusions(SECOND_PASS_PAPERS)
    assert result.verdicts == ["confirmed", "not_established"]
    assert result.padded == 1


async def test_confirm_inclusions_raises_on_an_empty_batch():
    with pytest.raises(screener.ScreeningError):
        await screener.confirm_inclusions([])


async def test_confirm_inclusions_raises_when_the_model_returns_no_answers_at_all():
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_run_result([]))
    ):
        with pytest.raises(screener.ScreeningError):
            await screener.confirm_inclusions(SECOND_PASS_PAPERS)


async def test_confirm_inclusions_raises_when_the_model_call_itself_raises():
    with patch.object(Agent, "run", new=AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(screener.ScreeningError):
            await screener.confirm_inclusions(SECOND_PASS_PAPERS)


# ============================================================================================
# Screener design v2
# ============================================================================================

# --- Design decision A: full-text criteria route to the queue, never to INCLUDE -------------


def test_resolve_to_confirm_is_never_empty_when_the_protocol_has_a_full_text_inclusion_id():
    """The mechanism behind decision A's "expected 0" INCLUDE-only recall on a full-text-
    defining protocol: a full-text inclusion criterion is undecidable from a title and
    abstract, so its id is either named by the model or defaulted in -- there is no way for
    the resolved list to come back empty once the protocol names one."""
    silent = screener.PaperDecision(decision="INCLUDE", reason="ok")
    named_subset = screener.PaperDecision(decision="INCLUDE", reason="ok", to_confirm=["I1"])
    for decision in (silent, named_subset):
        assert screener._resolve_to_confirm(decision, {"I1"}) == ["I1"]


def test_guard_demotes_an_include_with_an_unconfirmed_full_text_inclusion_criterion():
    decision = screener.PaperDecision(decision="INCLUDE", reason="passes I2 and I3")
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, {"I1"}, full_text_inclusion_ids={"I1"},
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "full_text_to_confirm"
    assert "full-text" in result.reason.lower()


def test_guard_full_text_to_confirm_runs_ahead_of_the_anchor_check():
    """Decision A's check fires first: an INCLUDE that also fails the anchor check is still
    reported as full_text_to_confirm, not unanchored_include."""
    decision = _anchor_decision()  # no anchors at all -- would also fail the anchor check
    result, guard_reason = screener.apply_decision_guard(
        decision, ANCHOR_SHOWN_TEXT, {"I1"},
        full_text_inclusion_ids={"I1"},
        anchored_slots=frozenset(screener.ANCHOR_SLOTS),
    )
    assert result.decision == "NEEDS_REVIEW"
    assert guard_reason == "full_text_to_confirm"


def test_guard_full_text_to_confirm_is_a_no_op_with_no_full_text_inclusion_criteria():
    decision = screener.PaperDecision(decision="INCLUDE", reason="ok")
    result, guard_reason = screener.apply_decision_guard(decision, ANCHOR_SHOWN_TEXT, set())
    assert result.decision == "INCLUDE"
    assert guard_reason == ""


@pytest.mark.asyncio
async def test_screen_papers_routes_every_include_to_needs_review_when_i1_is_full_text():
    """Integration-level: on a protocol whose sole inclusion criterion is full-text (the
    Nagtegaal_2019 shape), screen_papers's own INCLUDE-only recall is structurally 0 -- every
    INCLUDE the model returns is routed to the queue, and its to_confirm audit trail
    survives the demotion."""
    decisions = [
        screener.PaperDecision(decision="INCLUDE", reason="on topic"),
        screener.PaperDecision(decision="EXCLUDE", criterion="E1", quote="About something else"),
    ]
    with patch.object(Agent, "run", new=AsyncMock(return_value=_fake_run_result(decisions))):
        result = await screener.screen_papers(
            "q", PAPERS[:2],
            inclusion_criteria=["the defining taxonomy criterion"],
            inclusion_stages=["full_text"],
            prompt_version="v3",
        )
    assert result.statuses[0] == "NEEDS_REVIEW"
    assert result.guard_reasons[0] == "full_text_to_confirm"
    assert result.to_confirm[0] == ["I1"]


# --- Design decision C, element 1d: the narrow type demotion --------------------------------


def test_apply_type_demotion_demotes_a_paratext_include():
    status, guard_reason = screener.apply_type_demotion(
        "INCLUDE", work_type="article", is_paratext=True
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "nonarticle_type"


def test_apply_type_demotion_demotes_each_listed_nonarticle_type():
    for work_type in screener.NONARTICLE_TYPES:
        status, guard_reason = screener.apply_type_demotion(
            "INCLUDE", work_type=work_type, is_paratext=False
        )
        assert status == "NEEDS_REVIEW", work_type
        assert guard_reason == "nonarticle_type"


def test_apply_type_demotion_leaves_an_article_include_unchanged():
    status, guard_reason = screener.apply_type_demotion(
        "INCLUDE", work_type="article", is_paratext=False
    )
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_type_demotion_does_not_touch_types_the_design_left_out():
    """conference-paper, dissertation, report and other are deliberately not on the list
    (the design's own measurement: both directions of error, S-115 and S-111)."""
    for work_type in ("conference-paper", "dissertation", "report", "other", None, ""):
        status, guard_reason = screener.apply_type_demotion(
            "INCLUDE", work_type=work_type, is_paratext=False
        )
        assert status == "INCLUDE"
        assert guard_reason == ""


def test_apply_type_demotion_is_a_no_op_on_a_non_include_status():
    for status in ("EXCLUDE", "NEEDS_REVIEW"):
        result, guard_reason = screener.apply_type_demotion(
            status, work_type="book", is_paratext=True
        )
        assert result == status
        assert guard_reason == ""


# --- Design decision C, element 1d: the table-of-contents detector -------------------------


def test_is_table_of_contents_detects_four_numbered_chapters():
    text = "1. Introduction 2. Methods 3. Results 4. Discussion"
    assert screener._is_table_of_contents(text) is True


def test_is_table_of_contents_detects_two_chapters_with_a_part_marker():
    text = "Part I. Foundations 1. Introduction 2. Background"
    assert screener._is_table_of_contents(text) is True


def test_is_table_of_contents_detects_two_part_markers_alone():
    text = "Part I. Foundations Part II. Applications"
    assert screener._is_table_of_contents(text) is True


def test_is_table_of_contents_is_false_for_an_ordinary_abstract():
    text = (
        "Title: A trial of hand hygiene reminders\n"
        "Abstract: Nurses in three ICUs received a reminder. Compliance improved by 12 "
        "percentage points."
    )
    assert screener._is_table_of_contents(text) is False


def test_is_table_of_contents_handles_none_and_empty_text():
    assert screener._is_table_of_contents(None) is False
    assert screener._is_table_of_contents("") is False


def test_apply_table_of_contents_demotion_demotes_a_chapter_list_include():
    text = "1. Intro 2. Theory 3. Method 4. Results"
    status, guard_reason = screener.apply_table_of_contents_demotion("INCLUDE", text)
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "table_of_contents"


def test_apply_table_of_contents_demotion_leaves_an_ordinary_abstract_include_unchanged():
    status, guard_reason = screener.apply_table_of_contents_demotion(
        "INCLUDE", ANCHOR_SHOWN_TEXT
    )
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_table_of_contents_demotion_is_a_no_op_on_a_non_include_status():
    for status in ("EXCLUDE", "NEEDS_REVIEW"):
        result, guard_reason = screener.apply_table_of_contents_demotion(
            status, "1. A 2. B 3. C 4. D"
        )
        assert result == status
        assert guard_reason == ""


# --- Design decision C: the v2 second-pass judge (SECOND_PASS_PROMPT_V2) -------------------

SECOND_PASS_V2_QUESTION = (
    "What is the effect of hand hygiene reminders on nurses' compliance?"
)
SECOND_PASS_V2_INCLUSION_CRITERIA = ["Nurses in an ICU are the population."]
SECOND_PASS_V2_EXCLUSION_CRITERIA = ["Not a comparative study."]


def test_build_second_pass_prompt_v2_lists_the_research_question_and_numbered_criteria():
    prompt = screener._build_second_pass_prompt_v2(
        SECOND_PASS_PAPERS,
        SECOND_PASS_V2_QUESTION,
        SECOND_PASS_V2_INCLUSION_CRITERIA,
        SECOND_PASS_V2_EXCLUSION_CRITERIA,
    )
    assert SECOND_PASS_V2_QUESTION in prompt
    assert "I1: Nurses in an ICU are the population." in prompt
    assert "E1: Not a comparative study." in prompt
    assert "Record 1:" in prompt
    assert "Nurses in three ICUs" in prompt


def test_get_second_pass_agent_caches_separately_per_prompt_version_and_model():
    screener._second_pass_agents.clear()
    v1_agent = screener.get_second_pass_agent()
    v2_agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert v1_agent is not v2_agent
    assert screener.get_second_pass_agent() is v1_agent
    assert (
        screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
        is v2_agent
    )


def test_get_second_pass_agent_rejects_an_unknown_prompt_version():
    with pytest.raises(ValueError):
        screener.get_second_pass_agent(prompt_version="v3")


def test_second_pass_v2_agent_carries_no_thinking_disable():
    """An independent review found that disabling thinking (as every other tier does, for
    deepseek-flash tool-calling compatibility) collapsed this judge's own demotion rate from
    18 of 38 candidates to 5 of 38 on the same model, prompt, schema and shown text -- the V2
    second pass must keep reasoning on, unlike every other agent in this module."""
    screener._second_pass_agents.clear()
    v2_agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert v2_agent.model_settings == SECOND_PASS_V2_MODEL_SETTINGS
    assert "extra_body" not in v2_agent.model_settings


def test_second_pass_v2_agent_timeout_covers_the_observed_reasoning_on_latency():
    """Reasoning-on calls at this judge's own configured model were observed to take 43 to
    315 seconds each; DETERMINISTIC_LONG_MODEL_SETTINGS's 300s budget would time out the
    slowest of them and route their batch to second_pass_unavailable."""
    screener._second_pass_agents.clear()
    v2_agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert v2_agent.model_settings["timeout"] > 315
    assert v2_agent.model_settings["temperature"] == 0.0


def test_second_pass_v1_agent_still_disables_thinking():
    """Regression guard: only the V2 second pass gets reasoning left on; the V1 anchor-ledger
    re-ask still runs at settings.deepseek_model (deepseek-flash), which still needs
    thinking disabled for its own tool-calling structured output to work at all."""
    screener._second_pass_agents.clear()
    v1_agent = screener.get_second_pass_agent()
    assert v1_agent.model_settings == DETERMINISTIC_FAST_MODEL_SETTINGS
    assert v1_agent.model_settings["extra_body"] == {"thinking": {"type": "disabled"}}


def _fake_second_pass_v2_run_result(
    answers: list["screener.SecondPassAnswerV2"],
    *,
    input_tokens: int = 555,
    output_tokens: int = 77,
    model_name: str = "deepseek-v4-pro",
    system_fingerprint: str = "fp_second_v2",
):
    response = SimpleNamespace(
        model_name=model_name,
        provider_response_id="resp-second-v2-1",
        provider_details={"system_fingerprint": system_fingerprint, "finish_reason": "stop"},
        provider_name="deepseek",
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        output=screener.SecondPassAnswersV2(answers=answers),
        response=response,
        usage=lambda: usage,
    )


async def test_confirm_inclusions_v2_returns_answers_and_provenance_when_research_question_given():
    answers = [
        screener.SecondPassAnswerV2(
            population={"established": "established", "quote": "Nurses in three ICUs"},
            outcome={"established": "established", "quote": "Compliance improved"},
            study_type="study",
        ),
        screener.SecondPassAnswerV2(
            population={"established": "not_established", "quote": ""},
            outcome={"established": "not_established", "quote": ""},
            study_type="not_established",
        ),
    ]
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_v2_run_result(answers))
    ):
        result = await screener.confirm_inclusions(
            SECOND_PASS_PAPERS,
            research_question=SECOND_PASS_V2_QUESTION,
            inclusion_criteria=SECOND_PASS_V2_INCLUSION_CRITERIA,
            exclusion_criteria=SECOND_PASS_V2_EXCLUSION_CRITERIA,
            model="deepseek-v4-pro",
        )
    assert isinstance(result, screener.SecondPassBatchResultV2)
    assert result.answers[0].study_type == "study"
    assert result.answers[1].study_type == "not_established"
    assert result.provenance.model_configured == "deepseek-v4-pro"
    assert result.padded == 0


async def test_confirm_inclusions_v2_pads_a_short_response_with_not_established():
    answers = [
        screener.SecondPassAnswerV2(
            population={"established": "established", "quote": "Nurses in three ICUs"},
            outcome={"established": "established", "quote": "Compliance improved"},
            study_type="study",
        ),
    ]
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_v2_run_result(answers))
    ):
        result = await screener.confirm_inclusions(
            SECOND_PASS_PAPERS, research_question=SECOND_PASS_V2_QUESTION,
        )
    assert len(result.answers) == 2
    assert result.answers[1].study_type == "not_established"
    assert result.answers[1].population.established == "not_established"
    assert result.padded == 1


async def test_confirm_inclusions_without_a_research_question_keeps_the_v1_shape():
    """research_question empty (the default) is byte-for-byte v1 behaviour."""
    answers = [screener.SecondPassAnswer(verdict="confirmed")]
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_run_result(answers * 2))
    ):
        result = await screener.confirm_inclusions(SECOND_PASS_PAPERS)
    assert isinstance(result, screener.SecondPassBatchResult)


# --- screener_second_pass_output_mode ("tool" vs "prompted") ------


def test_get_second_pass_agent_uses_the_bare_schema_as_output_type_in_tool_mode(monkeypatch):
    """output_mode "tool": the V2 agent's own output_type is the bare SecondPassAnswersV2
    schema, pydantic-ai's own tool-calling output (a forced tool_choice) -- unchanged from
    every prior round for a judge model that does not reject that combination."""
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "tool")
    screener._second_pass_agents.clear()
    agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert agent.output_type is screener.SecondPassAnswersV2


def test_get_second_pass_agent_wraps_the_schema_in_promptedoutput_in_prompted_mode(monkeypatch):
    """output_mode "prompted": the schema goes into the prompt instead of a forced
    tool_choice -- pydantic_ai.PromptedOutput wraps the same SecondPassAnswersV2 schema,
    never a different one, so the two modes stay decision- and provenance-compatible."""
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "prompted")
    screener._second_pass_agents.clear()
    agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert isinstance(agent.output_type, PromptedOutput)
    assert agent.output_type.outputs is screener.SecondPassAnswersV2


def test_get_second_pass_agent_leaves_the_v1_agent_in_tool_mode_regardless(monkeypatch):
    """output_mode only ever applies to the V2 second pass; the V1 anchor-ledger re-ask
    stays bare tool-calling output even when the setting itself says "prompted"."""
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "prompted")
    screener._second_pass_agents.clear()
    v1_agent = screener.get_second_pass_agent()
    assert v1_agent.output_type is screener.SecondPassAnswers


def test_get_second_pass_agent_rejects_an_unknown_output_mode(monkeypatch):
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "narrated")
    screener._second_pass_agents.clear()
    with pytest.raises(ValueError):
        screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")


def test_get_second_pass_agent_caches_separately_per_output_mode(monkeypatch):
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "tool")
    screener._second_pass_agents.clear()
    tool_agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "prompted")
    prompted_agent = screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro")
    assert tool_agent is not prompted_agent
    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "tool")
    assert screener.get_second_pass_agent(prompt_version="v2", model="deepseek-v4-pro") is tool_agent


async def test_confirm_inclusions_v2_reports_output_mode_and_reasoning_effort_never_temperature():
    """Provenance fields: output_mode and reasoning_effort take the place of a temperature
    this judge's own reasoning-mode call never actually applies (DeepSeek's thinking-mode
    guide: temperature is not supported in thinking mode, so reporting it would be false)."""
    answers = [
        screener.SecondPassAnswerV2(
            population={"established": "established", "quote": "Nurses in three ICUs"},
            outcome={"established": "established", "quote": "Compliance improved"},
            study_type="study",
        ),
        screener.SecondPassAnswerV2(),
    ]
    with patch.object(
        Agent, "run", new=AsyncMock(return_value=_fake_second_pass_v2_run_result(answers))
    ):
        result = await screener.confirm_inclusions(
            SECOND_PASS_PAPERS,
            research_question=SECOND_PASS_V2_QUESTION,
            inclusion_criteria=SECOND_PASS_V2_INCLUSION_CRITERIA,
            exclusion_criteria=SECOND_PASS_V2_EXCLUSION_CRITERIA,
            model="deepseek-v4-pro",
        )
    assert result.provenance.temperature is None
    assert result.provenance.output_mode == settings.screener_second_pass_output_mode
    assert result.provenance.reasoning_effort == screener.SECOND_PASS_REASONING_EFFORT_DEFAULT


async def test_confirm_inclusions_v2_wraps_a_malformed_prompted_reply_in_a_screening_error(
    monkeypatch,
):
    """A "prompted" reply that does not parse or validate as SecondPassAnswerV2 (pydantic-ai
    exhausts its own retries and raises UnexpectedModelBehavior) is not silently lost --
    confirm_inclusions wraps it in a ScreeningError carrying the exception's own type and
    message, the same contract every other model-call failure in this function keeps."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "prompted")
    screener._second_pass_agents.clear()
    with patch.object(
        Agent,
        "run",
        new=AsyncMock(
            side_effect=UnexpectedModelBehavior(
                "Exceeded maximum retries (1) for output validation", body="not valid json"
            )
        ),
    ):
        with pytest.raises(screener.ScreeningError) as excinfo:
            await screener.confirm_inclusions(
                SECOND_PASS_PAPERS,
                research_question=SECOND_PASS_V2_QUESTION,
                inclusion_criteria=SECOND_PASS_V2_INCLUSION_CRITERIA,
                exclusion_criteria=SECOND_PASS_V2_EXCLUSION_CRITERIA,
                model="deepseek-flash",
            )
    assert "UnexpectedModelBehavior" in str(excinfo.value)
    assert "Exceeded maximum retries" in str(excinfo.value)


async def test_run_second_pass_stage_routes_a_malformed_prompted_reply_with_its_error_text(
    monkeypatch,
):
    """End to end: a "prompted" judge whose reply fails schema validation reaches
    run_second_pass_stage as any other judge failure does -- routed to NEEDS_REVIEW with
    second_pass_unavailable, its own decision carrying the recoverable error text."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    monkeypatch.setattr(settings, "screener_second_pass_output_mode", "prompted")
    screener._second_pass_agents.clear()
    with patch.object(
        Agent,
        "run",
        new=AsyncMock(
            side_effect=UnexpectedModelBehavior(
                "Exceeded maximum retries (1) for output validation", body="not valid json"
            )
        ),
    ):
        result = await screener.run_second_pass_stage(
            SECOND_PASS_PAPERS,
            ["SHOWN", "SHOWN"],
            research_question=SECOND_PASS_V2_QUESTION,
            inclusion_criteria=SECOND_PASS_V2_INCLUSION_CRITERIA,
            exclusion_criteria=SECOND_PASS_V2_EXCLUSION_CRITERIA,
            model="deepseek-flash",
            concurrency=8,
        )
    assert result.decisions[0].guard_reason == "second_pass_unavailable"
    assert result.decisions[0].second_pass is None
    assert result.decisions[0].error is not None
    assert "UnexpectedModelBehavior" in result.decisions[0].error
    assert "Exceeded maximum retries" in result.decisions[0].error


# --- Design decision C, element 3a: the second pass's own verbatim check -------------------


def test_apply_second_pass_guard_v2_keeps_a_fully_established_verbatim_study():
    answers = screener.SecondPassAnswerV2(
        population={"established": "established", "quote": "Nurses in three ICUs"},
        outcome={"established": "established", "quote": "Compliance improved"},
        study_type="study",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_second_pass_guard_v2_allows_a_synthesis_study_type():
    answers = screener.SecondPassAnswerV2(
        population={"established": "established", "quote": "Nurses in three ICUs"},
        outcome={"established": "established", "quote": "Compliance improved"},
        study_type="synthesis",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_second_pass_guard_v2_demotes_on_not_established_population():
    answers = screener.SecondPassAnswerV2(
        population={"established": "not_established", "quote": ""},
        outcome={"established": "established", "quote": "Compliance improved"},
        study_type="study",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v2_demotes_on_not_established_outcome():
    """The population and study_type slots each already had their own coverage;
    this closes the same gap for the outcome slot on its own, population established."""
    answers = screener.SecondPassAnswerV2(
        population={"established": "established", "quote": "Nurses in three ICUs"},
        outcome={"established": "not_established", "quote": ""},
        study_type="study",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v2_demotes_on_not_established_study_type():
    answers = screener.SecondPassAnswerV2(
        population={"established": "established", "quote": "Nurses in three ICUs"},
        outcome={"established": "established", "quote": "Compliance improved"},
        study_type="not_established",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v2_demotes_a_fabricated_quote_even_if_marked_established():
    """The element 3a verbatim check: a quote the shown text does not contain is treated as
    not_established, whatever the model's own "established" verdict said."""
    answers = screener.SecondPassAnswerV2(
        population={"established": "established", "quote": "a phrase never shown"},
        outcome={"established": "established", "quote": "Compliance improved"},
        study_type="study",
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v2_is_a_no_op_on_a_non_include_status():
    answers = screener.SecondPassAnswerV2(study_type="not_established")
    for status in ("EXCLUDE", "NEEDS_REVIEW"):
        result, guard_reason = screener.apply_second_pass_guard(
            status, "", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
        )
        assert result == status
        assert guard_reason == ""


def test_apply_second_pass_guard_v1_verbatim_check_keeps_a_confirmed_answer_all_verbatim():
    """apply_second_pass_guard enforces
    the verbatim check on the V1 judge's own anchor quotes, not only the verdict string."""
    answers = screener.SecondPassAnswer(
        verdict="confirmed",
        anchors=[
            {"slot": "population", "quote": "Nurses in three ICUs"},
            {"slot": "subject", "quote": "hand hygiene reminder"},
            {"slot": "outcome", "quote": "Compliance improved by 12 percentage points"},
        ],
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "confirmed", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "INCLUDE"
    assert guard_reason == ""


def test_apply_second_pass_guard_v1_verbatim_check_demotes_a_confirmed_answer_with_a_fabricated_quote():
    answers = screener.SecondPassAnswer(
        verdict="confirmed",
        anchors=[
            {"slot": "population", "quote": "a population never shown"},
            {"slot": "subject", "quote": "hand hygiene reminder"},
            {"slot": "outcome", "quote": "Compliance improved by 12 percentage points"},
        ],
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "confirmed", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v1_verbatim_check_demotes_a_confirmed_answer_missing_a_slot():
    answers = screener.SecondPassAnswer(
        verdict="confirmed",
        anchors=[
            {"slot": "population", "quote": "Nurses in three ICUs"},
            {"slot": "outcome", "quote": "Compliance improved by 12 percentage points"},
        ],  # subject missing entirely
    )
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "confirmed", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


def test_apply_second_pass_guard_v1_answers_path_demotes_on_a_not_established_verdict():
    answers = screener.SecondPassAnswer(verdict="not_established", missing_slot="outcome")
    status, guard_reason = screener.apply_second_pass_guard(
        "INCLUDE", "not_established", answers=answers, shown_text=ANCHOR_SHOWN_TEXT
    )
    assert status == "NEEDS_REVIEW"
    assert guard_reason == "not_established"


# --- The second-pass model config default ------------------------------------------------


def test_screener_second_pass_model_defaults_to_the_same_model_reasoning_on():
    """Evaluation found deepseek-flash with reasoning on
    and screener_second_pass_output_mode "prompted" matches a materially stronger model's
    own inclusions at a fraction of its cost and latency, so the second pass defaults to
    the same model settings.deepseek_model uses."""
    assert settings.screener_second_pass_model == "deepseek-flash"
    assert settings.screener_second_pass_model == settings.deepseek_model
    assert settings.screener_second_pass_output_mode == "prompted"


# --- run_second_pass_stage: the concurrent second-pass stage ---------------------------


def _fake_second_pass_provenance(**overrides) -> LLMCallProvenance:
    fields = {
        "agent": "relevance_screener_second_pass_v2",
        "model_configured": "deepseek-v4-pro",
        "model_reported": "deepseek-v4-pro",
        "system_fingerprint": "fp_stage_test",
        "prompt_version": screener.SECOND_PASS_PROMPT_V2_VERSION,
        "input_tokens": 100,
        "output_tokens": 20,
    }
    fields.update(overrides)
    return LLMCallProvenance(**fields)


def _stub_second_pass_result(papers, *, established: str = "established", study_type: str = "study"):
    """A ``SecondPassBatchResultV2``-shaped stand-in: every answer quotes the fixed string
    every candidate's own ``shown_text`` is set to in these tests ("SHOWN"), so
    ``apply_second_pass_guard``'s own verbatim check passes and the candidate is confirmed
    whenever ``established == "established"``."""
    answers = [
        screener.SecondPassAnswerV2(
            population={"established": established, "quote": "SHOWN"},
            outcome={"established": established, "quote": "SHOWN"},
            study_type=study_type,
        )
        for _ in papers
    ]
    return SimpleNamespace(answers=answers, provenance=_fake_second_pass_provenance())


@pytest.mark.asyncio
async def test_run_second_pass_stage_batches_candidates_into_chunks_of_five():
    """Batching: SECOND_PASS_STAGE_BATCH_SIZE (5) records per call, in input order, one
    call per chunk -- not one call per candidate and not one call for the whole round."""
    seen_chunk_sizes: list[int] = []

    async def fake_judge(papers, **kwargs):
        seen_chunk_sizes.append(len(papers))
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(12)]
    shown = ["SHOWN"] * 12
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, judge=fake_judge,
    )
    assert sorted(seen_chunk_sizes) == [2, 5, 5]
    assert len(result.calls) == 3
    assert len(result.decisions) == 12
    assert all(d.status == "INCLUDE" for d in result.decisions)
    assert result.n_skipped_budget == 0
    # Every candidate index appears in exactly one call's own indices, covering 0..11 --
    # harness parity (run_screening.py) attributes each row to its own call by this field.
    all_indices = sorted(i for call in result.calls for i in call.indices)
    assert all_indices == list(range(12))


@pytest.mark.asyncio
async def test_run_second_pass_stage_respects_the_concurrency_limit():
    """Concurrency bound: with 20 candidates (4 chunks of 5) and concurrency=2, no more
    than 2 judge calls are ever in flight at once."""
    in_flight = {"current": 0, "peak": 0}

    async def fake_judge(papers, **kwargs):
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        await asyncio.sleep(0.01)  # yield control so other chunks can actually overlap
        in_flight["current"] -= 1
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(20)]
    shown = ["SHOWN"] * 20
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=2, judge=fake_judge,
    )
    assert in_flight["peak"] == 2  # actually reached the limit, and never exceeded it
    assert len(result.calls) == 4
    assert len(result.decisions) == 20


@pytest.mark.asyncio
async def test_run_second_pass_stage_checks_should_abort_only_after_the_semaphore():
    """should_abort (a real session checkout in production,
    app.services.smart_search's own _stage_should_abort) must be checked only after
    acquiring the semaphore, so at most `concurrency`-many checks are ever in flight at
    once -- not once per chunk, unbounded, the moment the stage starts. With 20 candidates
    (4 chunks) and concurrency=2, a should_abort that yields control while it "checks out"
    would show 4 concurrent calls if the pre-semaphore check still ran it; bounded to 2
    proves it runs only inside the semaphore."""
    in_flight = {"current": 0, "peak": 0}

    async def fake_should_abort() -> bool:
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        await asyncio.sleep(0.01)
        in_flight["current"] -= 1
        return False

    async def fake_judge(papers, **kwargs):
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(20)]
    shown = ["SHOWN"] * 20
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=2, judge=fake_judge,
        should_abort=fake_should_abort,
    )
    assert in_flight["peak"] <= 2
    assert len(result.decisions) == 20
    assert all(d.status == "INCLUDE" for d in result.decisions)


@pytest.mark.asyncio
async def test_run_second_pass_stage_routes_a_failed_or_timed_out_chunk_to_needs_review():
    """Timeout/failure routing: a chunk whose own judge call raises (a real timeout would
    surface the same way, since confirm_inclusions wraps every failure in ScreeningError)
    is routed to NEEDS_REVIEW with second_pass_unavailable and no second_pass answer,
    without disturbing any other chunk's own outcome."""

    async def flaky_judge(papers, **kwargs):
        if papers[0]["title"] == "P0":
            raise TimeoutError("simulated 420s timeout")
        return _stub_second_pass_result(papers)

    candidates = [{"title": "P0"}, {"title": "P1"}]
    shown = ["SHOWN", "SHOWN"]
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, batch_size=1, judge=flaky_judge,
    )
    assert result.decisions[0].status == "NEEDS_REVIEW"
    assert result.decisions[0].guard_reason == "second_pass_unavailable"
    assert result.decisions[0].second_pass is None
    assert result.decisions[1].status == "INCLUDE"
    assert result.decisions[1].second_pass is not None
    assert len(result.calls) == 1  # only the successful chunk's call is recorded
    assert result.n_skipped_budget == 0


@pytest.mark.asyncio
async def test_run_second_pass_stage_records_and_logs_the_failed_chunks_own_exception(caplog):
    """The failure reason must be recoverable, not just its
    count -- the exception's own type and message are carried on the decision and logged,
    so a reader of production logs or of this result can see why a record was
    second_pass_unavailable, not merely that it was."""
    import logging

    async def flaky_judge(papers, **kwargs):
        raise TimeoutError("simulated 420s timeout")

    with caplog.at_level(logging.WARNING, logger="app.agents.relevance_screener_agent"):
        result = await screener.run_second_pass_stage(
            [{"title": "P0"}], ["SHOWN"],
            research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
            model="deepseek-v4-pro", concurrency=8, judge=flaky_judge,
        )
    assert result.decisions[0].error == "TimeoutError: simulated 420s timeout"
    assert any(
        "simulated 420s timeout" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_run_second_pass_stage_leaves_error_none_for_a_confirmed_or_budget_skipped_chunk():
    """A decision's own error is only ever set on an actual judge-call failure -- never on
    a confirmed/demoted answer, and never on a chunk the stage skipped for lack of budget
    (no exception occurred there at all)."""
    async def fake_judge(papers, **kwargs):
        return _stub_second_pass_result(papers)

    result_ok = await screener.run_second_pass_stage(
        [{"title": "P0"}], ["SHOWN"],
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, judge=fake_judge,
    )
    assert result_ok.decisions[0].error is None

    result_skipped = await screener.run_second_pass_stage(
        [{"title": "P0"}], ["SHOWN"],
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, stage_deadline=-1.0, judge=fake_judge,
    )
    assert result_skipped.decisions[0].error is None
    assert result_skipped.decisions[0].guard_reason == "second_pass_unavailable"


@pytest.mark.asyncio
async def test_run_second_pass_stage_concurrency_none_falls_back_to_the_configured_setting(
    monkeypatch,
):
    """run_smart_search and the harness both call
    run_second_pass_stage without passing concurrency at all -- that path must read
    settings.screener_second_pass_concurrency, not some other value baked into the
    function's own signature."""
    monkeypatch.setattr(settings, "screener_second_pass_concurrency", 3)
    in_flight = {"current": 0, "peak": 0}

    async def fake_judge(papers, **kwargs):
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        await asyncio.sleep(0.01)
        in_flight["current"] -= 1
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(15)]  # 3 chunks of 5
    shown = ["SHOWN"] * 15
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", judge=fake_judge,  # concurrency not passed at all
    )
    assert in_flight["peak"] == 3
    assert len(result.calls) == 3


@pytest.mark.asyncio
async def test_run_second_pass_stage_never_launches_a_chunk_after_its_deadline():
    """Budget accounting, fully exhausted: a stage_deadline already in the past when the
    stage starts means no chunk is ever launched at all, and every candidate is routed to
    NEEDS_REVIEW with second_pass_unavailable, counted in n_skipped_budget."""
    calls: list[int] = []

    async def fake_judge(papers, **kwargs):
        calls.append(len(papers))
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(6)]
    shown = ["SHOWN"] * 6
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, batch_size=5,
        stage_deadline=-1.0,  # already in the past under the real time.monotonic() clock
        judge=fake_judge,
    )
    assert calls == []
    assert result.n_skipped_budget == 6
    assert all(d.status == "NEEDS_REVIEW" for d in result.decisions)
    assert all(d.guard_reason == "second_pass_unavailable" for d in result.decisions)


@pytest.mark.asyncio
async def test_run_second_pass_stage_stops_launching_new_chunks_once_the_deadline_passes():
    """Budget accounting, partial: a deadline crossed midway through the stage lets a chunk
    already in flight finish (a call already paid for is not abandoned), but never launches
    the next one -- a fake, manually-advanced clock makes this deterministic without a real
    sleep. concurrency=1 forces the second chunk to wait behind the first, so it queries the
    deadline again after the first chunk's own call has already advanced the clock past it."""
    clock = {"t": 0.0}

    def fake_now() -> float:
        return clock["t"]

    calls: list[int] = []

    async def fake_judge(papers, **kwargs):
        calls.append(len(papers))
        clock["t"] += 1.0  # simulate this call itself taking 1.0 (fake) second
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(10)]
    shown = ["SHOWN"] * 10
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=1, batch_size=5,
        stage_deadline=1.0,  # exactly when the first chunk's own call finishes
        now_fn=fake_now,
        judge=fake_judge,
    )
    assert calls == [5]  # the second chunk was never launched
    assert result.n_skipped_budget == 5
    statuses = [d.status for d in result.decisions]
    assert statuses[:5] == ["INCLUDE"] * 5
    assert statuses[5:] == ["NEEDS_REVIEW"] * 5
    assert all(r == "second_pass_unavailable" for r in [d.guard_reason for d in result.decisions[5:]])


@pytest.mark.asyncio
async def test_run_second_pass_stage_default_judge_resolves_confirm_inclusions_at_call_time():
    """The ``judge`` parameter's default is resolved from this module's own globals at call
    time, not bound once at import time: a caller that never passes ``judge`` explicitly
    (app.services.smart_search's own real usage) must still be patchable via
    ``patch.object(relevance_screener_agent, "confirm_inclusions", ...)``, the same pattern
    every other agent function in this module already supports."""
    candidates = [{"title": "P0"}]
    shown = ["SHOWN"]
    fake = AsyncMock(return_value=_stub_second_pass_result(candidates))
    with patch.object(screener, "confirm_inclusions", new=fake):
        result = await screener.run_second_pass_stage(
            candidates, shown,
            research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
            model="deepseek-v4-pro", concurrency=8,
        )
    fake.assert_awaited_once()
    assert result.decisions[0].status == "INCLUDE"


@pytest.mark.asyncio
async def test_run_second_pass_stage_wall_time_reflects_concurrent_not_summed_latency():
    """The stage's own wall_time_s is one span around the whole concurrent gather, not the
    sum of the calls' own latencies -- with concurrency high enough that both chunks run
    at once, the wall time is close to one call's own duration, not both added together."""

    async def slow_judge(papers, **kwargs):
        await asyncio.sleep(0.05)
        return _stub_second_pass_result(papers)

    candidates = [{"title": f"P{i}"} for i in range(10)]
    shown = ["SHOWN"] * 10
    result = await screener.run_second_pass_stage(
        candidates, shown,
        research_question="rq", inclusion_criteria=["i"], exclusion_criteria=["e"],
        model="deepseek-v4-pro", concurrency=8, batch_size=5, judge=slow_judge,
    )
    assert len(result.calls) == 2
    # Both chunks ran concurrently under concurrency=8, so the stage's own wall time is
    # close to one call's latency (~0.05s), well under the ~0.1s two sequential calls
    # would take.
    assert result.wall_time_s < 0.09
