"""The writer is told a target length and a hard maximum, the body word
count (headings excluded) is checked after generation, and a body over the hard maximum
is regenerated once, naming the previous count. ``provenance.length`` records the
outcome. ``target_words=None`` (the default, matching every request made before this
field existed) runs generation with no length control at all.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.services.writing import (  # noqa: E402
    WritingCompletion,
    _count_body_words,
    _hard_maximum_words,
    _length_instruction,
    _trim_body_to_hard_maximum,
)


def test_hard_maximum_is_the_target_rounded_up_by_25_percent():
    assert _hard_maximum_words(400) == 500
    assert _hard_maximum_words(300) == 375
    # 401 * 1.25 = 501.25 -> rounds up, not down or to the nearest.
    assert _hard_maximum_words(401) == 502


def test_length_instruction_names_the_target_and_the_hard_maximum():
    assert _length_instruction(400) == "Target length: 400 words. Hard maximum: 500 words."


def test_count_body_words_excludes_markdown_heading_lines():
    text = "## Sub-theme heading\n\nOne two three four five."
    assert _count_body_words(text) == 5


def test_count_body_words_counts_everything_when_there_is_no_heading():
    assert _count_body_words("One two three.") == 3


def test_count_body_words_excludes_a_whole_line_bold_sub_heading():
    """A sub-heading the writer marked with a whole-line
    bold run, not a ``#`` line, must not inflate the count that regeneration is
    triggered against."""
    text = "**Sub-Heading Label**\n\nOne two three four five."
    assert _count_body_words(text) == 5


def test_count_body_words_keeps_a_bold_claim_sentence_as_body_text():
    """A whole-line bold run that ends in sentence punctuation, or carries a citation,
    is a claim sentence the writer happened to bold, not a sub-heading
    (`app.services.fulltext._is_bold_heading_candidate`), so it still counts."""
    text = "**Direct corrections outperformed codes by 32% (Smith, 2020).**"
    assert _count_body_words(text) == 8


# ---------------------------------------------------------------------------
# _trim_body_to_hard_maximum
# ---------------------------------------------------------------------------


def test_trim_returns_the_text_unchanged_when_already_under_the_maximum():
    text = "One two three."
    result_text, links, uncited, trimmed = _trim_body_to_hard_maximum(text, 10, [], [])
    assert not trimmed
    assert result_text == text
    assert links == []
    assert uncited == []


def test_trim_drops_trailing_blocks_until_under_the_maximum():
    text = "One two three four five.\n\nSix seven eight nine ten.\n\nEleven twelve."
    result_text, _links, _uncited, trimmed = _trim_body_to_hard_maximum(text, 5, [], [])
    assert trimmed
    assert result_text == "One two three four five."
    assert _count_body_words(result_text) <= 5


def test_trim_never_empties_the_section_even_if_still_over_the_maximum():
    text = "One two three four five six seven eight nine ten."
    result_text, _links, _uncited, trimmed = _trim_body_to_hard_maximum(text, 2, [], [])
    assert not trimmed
    assert result_text == text


def test_trim_drops_citation_links_and_uncited_sentences_of_dropped_blocks():
    text = "One two three four five.\n\nSix seven eight nine ten."
    links = [
        {"paragraph_index": 0, "sentence": "One two three four five.", "keys": ["a"]},
        {"paragraph_index": 1, "sentence": "Six seven eight nine ten.", "keys": ["b"]},
    ]
    uncited = [{"paragraph_index": 1, "sentence": "Six seven eight nine ten.", "tag": "framing"}]
    result_text, new_links, new_uncited, trimmed = _trim_body_to_hard_maximum(
        text, 5, links, uncited
    )
    assert trimmed
    assert result_text == "One two three four five."
    assert new_links == [links[0]]
    assert new_uncited == []


def _fake_completion(word_count: int, **provenance_overrides) -> WritingCompletion:
    content = " ".join(f"w{i}" for i in range(word_count)) + "."
    provenance = {
        "agent": "writing",
        "model_configured": "deepseek-chat",
        "model_reported": "deepseek-v4-flash",
        "provider": "deepseek",
        "provider_response_id": "chatcmpl-1",
        "system_fingerprint": "fp_test",
        "temperature": 0.7,
        "prompt_version": "sha256:aaaaaaaaaaaa",
        "input_tokens": 1000,
        "output_tokens": 200,
        "max_tokens": 4096,
    }
    provenance.update(provenance_overrides)
    return WritingCompletion(content=content, provenance=provenance)


async def _run_generate_section(db_session, *, target_words, call_deepseek_mock):
    from app.models.analysis_job import JobType
    from app.services.writing import generate_section
    from tests.t5_fixtures import (
        make_session_factory,
        seed_draft,
        seed_job,
        seed_project,
        seed_user,
    )

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    draft = await seed_draft(db_session, project, user, {"type": "doc", "content": []})
    job = await seed_job(db_session, project, JobType.writing)

    factory, engine = make_session_factory()
    with patch(
        "app.agents.writing_rules_agent.merge_writing_rules",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM down"),
    ), patch("app.services.writing.call_deepseek", new=call_deepseek_mock), patch(
        "app.services.writing.build_citation_link_map",
        new_callable=AsyncMock,
        side_effect=RuntimeError("linker not exercised by this test"),
    ):
        await generate_section(
            draft=draft, section_type="methods", context=None, job_id=job.id,
            session_factory=factory, target_words=target_words,
        )
    await engine.dispose()
    await db_session.refresh(job)
    return job


@pytest.mark.asyncio
async def test_generate_section_sends_the_length_instruction_when_target_words_is_given(
    db_session,
):
    captured_prompts: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        captured_prompts.append(user_prompt)
        return _fake_completion(300)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    assert job.status.value == "completed", job.error
    assert len(captured_prompts) == 1
    assert "Target length: 400 words. Hard maximum: 500 words." in captured_prompts[0]


@pytest.mark.asyncio
async def test_generate_section_omits_the_length_instruction_when_no_target_words_is_given(
    db_session,
):
    captured_prompts: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        captured_prompts.append(user_prompt)
        return _fake_completion(300)

    job = await _run_generate_section(
        db_session, target_words=None, call_deepseek_mock=fake_call_deepseek
    )

    assert job.status.value == "completed", job.error
    assert "Target length" not in captured_prompts[0]
    assert "length" not in job.result["provenance"]
    # ``False`` ("no regeneration happened"), not
    # ``None`` -- the hard-maximum check never runs at all with no target_words, which
    # is the same outcome a reader of this boolean field cares about as a body that ran
    # the check and stayed under it.
    assert job.result["loop_stats"]["length_regenerated"] is False


@pytest.mark.asyncio
async def test_a_body_under_the_hard_maximum_is_not_regenerated(db_session):
    async def fake_call_deepseek(system_prompt, user_prompt):
        return _fake_completion(300)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    assert job.status.value == "completed", job.error
    length = job.result["provenance"]["length"]
    assert length == {
        "target_words": 400,
        "hard_maximum": 500,
        "words": 300,
        "regenerated": False,
        "over_target": False,
        # No revision ran, so the text finalize saved
        # is the same one this length record was built from.
        "final_words": 300,
    }


@pytest.mark.asyncio
async def test_a_body_over_the_target_but_under_the_hard_maximum_is_not_regenerated(
    db_session,
):
    """``over_target`` and ``regenerated`` are independent: a body between the target and
    the hard maximum is over target but is not worth a second paid call for."""
    async def fake_call_deepseek(system_prompt, user_prompt):
        return _fake_completion(450)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    length = job.result["provenance"]["length"]
    assert length["regenerated"] is False
    assert length["over_target"] is True
    assert length["words"] == 450


@pytest.mark.asyncio
async def test_a_body_over_the_hard_maximum_is_regenerated_once_naming_the_previous_count(
    db_session,
):
    calls: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        calls.append(user_prompt)
        if len(calls) == 1:
            return _fake_completion(700)
        return _fake_completion(380)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    assert job.status.value == "completed", job.error
    assert len(calls) == 2, "exactly one regeneration, not a retry loop"
    assert "700 words" in calls[1], "the regeneration prompt names the previous count"
    assert "500 words" in calls[1], "the regeneration prompt repeats the hard maximum"

    # The final, saved content is the second (regenerated) attempt's, not the first.
    assert job.result["content"] == " ".join(f"w{i}" for i in range(380)) + "."

    length = job.result["provenance"]["length"]
    assert length == {
        "target_words": 400,
        "hard_maximum": 500,
        "words": 380,
        "regenerated": True,
        "over_target": False,
        # No gated-loop revision ran here (no claims,
        # since the citation-link map failed in this test's setup), so finalize saved
        # the same 380-word text this length record already describes.
        "final_words": 380,
    }


@pytest.mark.asyncio
async def test_regeneration_still_over_the_hard_maximum_is_not_retried_again(db_session):
    """Exactly one regeneration attempt, win or lose -- never an unbounded retry loop."""
    calls: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        calls.append(user_prompt)
        return _fake_completion(700)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    assert job.status.value == "completed", job.error
    assert len(calls) == 2
    length = job.result["provenance"]["length"]
    assert length["regenerated"] is True
    assert length["words"] == 700
    assert length["over_target"] is True


@pytest.mark.asyncio
async def test_regeneration_folds_both_calls_tokens_into_the_running_totals(db_session):
    calls: list[str] = []

    async def fake_call_deepseek(system_prompt, user_prompt):
        calls.append(user_prompt)
        if len(calls) == 1:
            return _fake_completion(700, input_tokens=1000, output_tokens=200)
        return _fake_completion(380, input_tokens=1100, output_tokens=150)

    job = await _run_generate_section(
        db_session, target_words=400, call_deepseek_mock=fake_call_deepseek
    )

    provenance = job.result["provenance"]
    # Top-level fields describe the final (used) call alone, as every existing reader
    # of them already expects.
    assert provenance["input_tokens"] == 1100
    assert provenance["output_tokens"] == 150
    # The running totals cover both writer calls, so cost reporting is not silently
    # short by a whole real paid call.
    assert provenance["total_calls"] == 2
    assert provenance["total_input_tokens"] == 2100
    assert provenance["total_output_tokens"] == 350
