"""The shared verification policy, Part B of the task authorisation 2026-09-10 redesign,
repair turn added 2026-09-11: ``app.services.fulltext.verify_claim_with_policy``.

One shared function, used by both the production path (``_verify_claims``'s ``_verify_one``)
and the evaluation harness (``evaluation/claims/verify_common.py``'s ``ProductionVerifier``,
see ``evaluation/tests/test_claims_verify_common.py`` for that side's own tests), so there is
exactly one verification policy, not two copies that could drift.

Part A (``relocate_evidence_quotes``, restricted to provably neutral edits --
``test_quote_relocation.py``) is run inside every pass, unchanged. Part B, tested here, is
the bounded repair turn: if, and only if, the first pass's own shape is exactly
``needs_nuance`` with ``machine_reasons == ["quote_not_verbatim"]`` after a model status of
``verified``, the model is asked once more, in the same conversation, to repair the quote
segments the guard found not verbatim, and that second answer -- whatever it is -- is
trusted as final. This never fires more than once per claim. ``verify_claim_with_policy``
itself never touches a raw model message: it hands ``call_model`` a `RepairRequest` naming
the failed segments and trusts the caller's own closure to continue the conversation (see
``test_verify_and_heal_claims_second_pass_fires_end_to_end`` below for the production
closure's own message-history wiring, against a `FakeAgent`).
"""

from __future__ import annotations

import pytest

CHUNK = (
    "The authors report that the intervention improved reading fluency for every "
    "participant enrolled in the study across every recorded session."
)
BAD_QUOTE = (
    "The authors report that the intervention improved reading fluency for every "
    "participant enrolled we the study across every recorded session."
)
CLAIM_TEXT = (
    "The intervention improved reading fluency for every participant enrolled in the "
    "study across every recorded session."
)


def _sequenced_call_model(answers):
    """A ``call_model`` stand-in that returns each of *answers* in order, one per call, and
    raises ``AssertionError`` if called more times than there are answers (so a test that
    expects exactly N calls fails loudly on an unexpected extra one). Every ``RepairRequest``
    (or ``None``, for the first call) it was given is recorded, in order, on its own
    ``received`` attribute, so a test can check what the policy asked it to repair."""
    it = iter(answers)
    received: list[object] = []

    async def call_model(repair_request=None):
        received.append(repair_request)
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - defensive
            raise AssertionError("call_model invoked more times than scripted") from None

    call_model.received = received
    return call_model


def _answer(**kw):
    from app.services.fulltext import ModelPassAnswer

    kw.setdefault("assertions", [])
    kw.setdefault("evidence_quotes", [kw.get("evidence_quote")] if kw.get("evidence_quote") else [])
    return ModelPassAnswer(**kw)


# --------------------------------------------------------------------------------------
# The trigger fires only on the exact condition, and takes the repair turn as final.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_turn_fires_and_resolves_a_miscopied_quote():
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first"),
            _answer(status="verified", evidence_quote=CHUNK, explanation="second"),
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "verified"
    assert result.machine_reasons == []
    assert result.model_status == "verified"
    assert len(result.passes) == 2
    assert result.passes[0]["model_status"] == "verified"
    assert result.passes[0]["machine_reasons"] == ["quote_not_verbatim"]
    assert result.passes[1]["machine_reasons"] == []
    assert result.evidence_quote == CHUNK
    assert result.explanation == "second"  # the final pass's own explanation


@pytest.mark.asyncio
async def test_repair_turn_receives_the_failed_segment_and_only_fires_once():
    """``call_model``'s first call gets ``None``; its second (and last) call gets a
    `RepairRequest` naming the bad quote as a failed segment -- the trigger condition
    itself is checked directly, and the stub's own "called more times than scripted"
    guard proves a repaired-but-still-verified answer never licenses a third call."""
    from app.services.fulltext import RepairRequest, verify_claim_with_policy

    call_model = _sequenced_call_model(
        [
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first"),
            _answer(status="verified", evidence_quote=CHUNK, explanation="second"),
        ]
    )
    await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert call_model.received[0] is None
    assert isinstance(call_model.received[1], RepairRequest)
    assert call_model.received[1].failed_segments == [BAD_QUOTE]


@pytest.mark.asyncio
async def test_repair_turn_that_again_miscopies_stays_needs_nuance():
    """The repair turn's result is trusted as final "whatever it is" (task authorisation):
    a second mis-copy is not retried a third time."""
    from app.services.fulltext import verify_claim_with_policy

    other_bad_quote = BAD_QUOTE.replace("we the study", "they the study")
    call_model = _sequenced_call_model(
        [
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first"),
            _answer(status="verified", evidence_quote=other_bad_quote, explanation="second"),
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "needs_nuance"
    assert result.machine_reasons == ["quote_not_verbatim"]
    assert len(result.passes) == 2


@pytest.mark.asyncio
async def test_repair_turn_whose_model_says_unsupported_is_unsupported():
    """A downgrade the repair turn's own answer makes is kept, not discarded."""
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first"),
            _answer(status="unsupported", evidence_quote=None, explanation="second"),
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "unsupported"
    assert result.model_status == "unsupported"
    assert len(result.passes) == 2


@pytest.mark.asyncio
async def test_never_fires_when_the_first_pass_is_unsupported():
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [_answer(status="unsupported", evidence_quote=None, explanation="only")]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "unsupported"
    assert len(result.passes) == 1


@pytest.mark.asyncio
async def test_never_fires_on_attribution_mismatch():
    """Guard 1 caps a wrong-paper claim to `unsupported` with `["attribution_mismatch"]`,
    never `needs_nuance` with `["quote_not_verbatim"]` -- the trigger's own reasons check
    excludes this shape even though the model's own status was `verified`."""
    from app.services.fulltext import verify_claim_with_policy

    wrong_paper_claim = "Researchers reported an entirely unrelated pattern (Smith, 2020)."
    call_model = _sequenced_call_model(
        [_answer(status="verified", evidence_quote=BAD_QUOTE, explanation="only")]
    )
    result = await verify_claim_with_policy(
        wrong_paper_claim, [CHUNK], [{"text": CHUNK}], call_model
    )

    assert result.status == "unsupported"
    assert result.machine_reasons == ["attribution_mismatch"]
    assert len(result.passes) == 1


@pytest.mark.asyncio
async def test_never_fires_when_the_models_own_first_status_was_needs_nuance():
    """`_guard_quote_fidelity` only ever runs its check when the incoming status is
    `verified` (it returns unchanged for any other status), so a model that already said
    `needs_nuance` can never even reach `machine_reasons == ["quote_not_verbatim"]` -- the
    trigger's own `first_answer.status == "verified"` condition is a second, independent
    guarantee that the repair turn never fires for this case, exercised directly here."""
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [_answer(status="needs_nuance", evidence_quote=BAD_QUOTE, explanation="only")]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "needs_nuance"
    assert result.machine_reasons == []
    assert len(result.passes) == 1


@pytest.mark.asyncio
async def test_never_fires_twice():
    """A trigger-shaped result from the repair turn itself is still taken as final, never
    triggering a third call -- pinned directly against `_sequenced_call_model`'s own
    "called more times than scripted" guard."""
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first"),
            _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="second"),
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "needs_nuance"
    assert len(result.passes) == 2


@pytest.mark.asyncio
async def test_never_fires_by_default_when_the_first_pass_is_already_verified():
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [_answer(status="verified", evidence_quote=CHUNK, explanation="only")]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "verified"
    assert result.machine_reasons == []
    assert len(result.passes) == 1


@pytest.mark.asyncio
async def test_never_fires_when_the_first_answer_has_no_quote_at_all():
    """Guard 3 fires ``quote_not_verbatim`` on two different shapes --
    a non-verbatim quote, and no quote at all (``not any(quotes)``). The second shape has
    no segment to repair (``_non_verbatim_quote_segments`` returns ``[]`` for it), so the
    repair turn must not fire: it would only solicit evidence the model never produced,
    under a prompt that must not forbid changing the verdict back. The guard's
    own demotion must be the one, final answer -- pinned by `_sequenced_call_model`'s own
    "called more times than scripted" guard, which fails the test loudly if a second call
    is attempted."""
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [_answer(status="verified", evidence_quote=None, explanation="only")]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "needs_nuance"
    assert result.machine_reasons == ["quote_not_verbatim"]
    assert result.model_status == "verified"
    assert len(result.passes) == 1


def test_repair_request_rejects_an_empty_failed_segments_list():
    """Assert the non-emptiness invariant where the
    `RepairRequest` is built, so a future caller cannot reintroduce the degenerate-prompt
    bug by constructing one with nothing to repair."""
    from app.services.fulltext import RepairRequest

    with pytest.raises(AssertionError):
        RepairRequest(failed_segments=[])


@pytest.mark.asyncio
async def test_an_exception_from_call_model_propagates_uncaught():
    """The shared function never wraps a `call_model` failure in a status; it propagates,
    exactly as a single, unwrapped model call would have -- the caller's own error handling
    (production's `_verify_one` try/except, the evaluation harness's `verify_with_retries`)
    is unchanged by this policy existing. This is the *first* call's own exception
    behaviour, unchanged by the repair-turn redesign (`test_repair_turn_exception_keeps_
    the_first_passes_result_as_final` below covers the *second*, optional call, which is
    caught)."""
    from app.services.fulltext import verify_claim_with_policy

    async def _raising_call_model(repair_request=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], _raising_call_model)


@pytest.mark.asyncio
async def test_repair_turn_exception_keeps_the_first_passes_result_as_final():
    """A transient failure on the repair turn must not destroy the first pass's own
    completed, guarded verdict. The first call succeeds (in the trigger shape), the second
    raises; the function must not raise, and the returned result must be exactly the first
    pass's own result."""
    from app.services.fulltext import verify_claim_with_policy

    calls = {"n": 0}

    async def call_model(repair_request=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first")
        raise RuntimeError("transient provider failure")

    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert calls["n"] == 2  # the repair call was actually attempted
    assert result.status == "needs_nuance"
    assert result.machine_reasons == ["quote_not_verbatim"]
    assert result.model_status == "verified"
    assert result.explanation == "first"  # the first pass's own explanation, not lost
    assert len(result.passes) == 2
    assert result.passes[0]["machine_reasons"] == ["quote_not_verbatim"]
    assert result.passes[0]["repair_prompt_version"] is None
    assert result.passes[1]["model_status"] is None
    assert result.passes[1]["diagnostics"] == ["second_pass_error"]
    assert "transient provider failure" in result.passes[1]["error"]
    # A repair call was attempted (and failed), so its own record still names
    # which repair-prompt version it was attempted under, distinct from `None` on a pass
    # that was never a repair attempt at all.
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION

    assert result.passes[1]["repair_prompt_version"] == QUOTE_REPAIR_PROMPT_VERSION


@pytest.mark.asyncio
async def test_repair_turn_exception_does_not_trigger_a_third_call():
    """The trigger only ever fires once, and a caught repair-turn exception must not be
    mistaken for a shape that licenses trying again."""
    from app.services.fulltext import verify_claim_with_policy

    calls = {"n": 0}

    async def call_model(repair_request=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _answer(status="verified", evidence_quote=BAD_QUOTE, explanation="first")
        raise RuntimeError("boom")

    await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_passes_carries_tokens_fingerprint_and_repair_prompt_version_per_call():
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT_VERSION
    from app.services.fulltext import verify_claim_with_policy

    call_model = _sequenced_call_model(
        [
            _answer(
                status="verified", evidence_quote=BAD_QUOTE, explanation="first",
                input_tokens=100, output_tokens=20, system_fingerprint="fp-1",
                model_reported="deepseek-v4-flash",
            ),
            _answer(
                status="verified", evidence_quote=CHUNK, explanation="second",
                input_tokens=110, output_tokens=25, system_fingerprint="fp-2",
                model_reported="deepseek-v4-flash",
            ),
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.passes[0]["tokens"] == {"input_tokens": 100, "output_tokens": 20}
    assert result.passes[0]["fingerprint"] == "fp-1"
    assert result.passes[0]["repair_prompt_version"] is None
    assert result.passes[1]["tokens"] == {"input_tokens": 110, "output_tokens": 25}
    assert result.passes[1]["fingerprint"] == "fp-2"
    assert result.passes[1]["repair_prompt_version"] == QUOTE_REPAIR_PROMPT_VERSION
    assert result.model_reported == "deepseek-v4-flash"  # the final pass's own value


# --------------------------------------------------------------------------------------
# A paraphrase the repair turn removed from the guarded top-level
# `evidence_quotes` can still survive, unchecked, on `assertions[].quote` /
# `assertions[].quotes`, because Guard 3 (`_guard_quote_fidelity`) never reads them. The
# filter that drops such a quote from what is stored and shown, without moving the status
# the guards already decided, is `_drop_non_verbatim_assertion_quotes`, applied
# by `verify_claim_with_policy`'s own caller (`test_claim_verification_service.py`,
# `test_non_verbatim_assertion_quote_is_dropped_before_storage`), not by this function --
# see `_compute_verification_policy_version`'s docstring for why: a display-only filter has
# no business moving the digest that identifies the call-relocate-guard-repair policy.
# --------------------------------------------------------------------------------------


def test_drop_non_verbatim_assertion_quotes_drops_only_the_non_verbatim_ones():
    """Unit test of the filter function itself: the singular `quote` is cleared when it is
    not verbatim, and `quotes` keeps only its verbatim members -- this is the exact shape
    of the "we" for "and" Mao et al. misquote (on `assertions[2].quotes[1]` of
    a `verified` row)."""
    from app.services.fulltext import _drop_non_verbatim_assertion_quotes

    good_assertion_quote = CHUNK
    assertions = [
        {
            "text": "the finding holds",
            "kind": "other",
            "verdict": "supported",
            "quote": BAD_QUOTE,
            "quotes": [good_assertion_quote, BAD_QUOTE],
        }
    ]
    (cleaned,) = _drop_non_verbatim_assertion_quotes(assertions, [CHUNK])
    assert cleaned["quote"] is None  # the non-verbatim singular quote is dropped
    assert cleaned["quotes"] == [good_assertion_quote]  # only the verbatim one survives


def test_drop_non_verbatim_assertion_quotes_leaves_a_fully_verbatim_assertion_unchanged():
    """The filter must not touch an assertion whose own quotes are all verbatim -- it
    removes exactly the non-verbatim ones, nothing more."""
    from app.services.fulltext import _drop_non_verbatim_assertion_quotes

    assertions = [
        {
            "text": "the finding holds",
            "kind": "other",
            "verdict": "supported",
            "quote": CHUNK,
            "quotes": [CHUNK],
        }
    ]
    (cleaned,) = _drop_non_verbatim_assertion_quotes(assertions, [CHUNK])
    assert cleaned["quote"] == CHUNK
    assert cleaned["quotes"] == [CHUNK]


@pytest.mark.asyncio
async def test_verify_claim_with_policy_returns_assertion_quotes_unfiltered():
    """`verify_claim_with_policy` itself must not run the display-only filter: a
    non-verbatim assertion quote comes back exactly as the model answered, so that
    `VERIFICATION_POLICY_VERSION` identifies only the call-relocate-guard-repair policy, not
    a display concern its own caller applies afterwards."""
    from app.services.fulltext import verify_claim_with_policy

    good_assertion_quote = CHUNK
    call_model = _sequenced_call_model(
        [
            _answer(
                status="verified",
                evidence_quote=CHUNK,
                explanation="only",
                assertions=[
                    {
                        "text": "the finding holds",
                        "kind": "other",
                        "verdict": "supported",
                        "quote": BAD_QUOTE,
                        "quotes": [good_assertion_quote, BAD_QUOTE],
                    }
                ],
            )
        ]
    )
    result = await verify_claim_with_policy(CLAIM_TEXT, [CHUNK], [{"text": CHUNK}], call_model)

    assert result.status == "verified"  # the guard never saw the assertion's own quote
    assert result.machine_reasons == []
    (assertion,) = result.assertions
    assert assertion["quote"] == BAD_QUOTE  # unfiltered: the caller's job now
    assert assertion["quotes"] == [good_assertion_quote, BAD_QUOTE]


# --------------------------------------------------------------------------------------
# VERIFICATION_POLICY_VERSION: hashes the shared function, the whole relocation set, and
# (added 2026-09-11) the repair turn's own segment-collection helper and prompt version.
# --------------------------------------------------------------------------------------


def test_verification_policy_version_is_a_stable_16_character_hex_digest():
    from app.services.fulltext import VERIFICATION_POLICY_VERSION

    assert len(VERIFICATION_POLICY_VERSION) == 16
    int(VERIFICATION_POLICY_VERSION, 16)  # raises ValueError if not hex


def test_verification_policy_version_is_pinned_at_the_value_results_v5_was_measured_under():
    """``VERIFICATION_POLICY_VERSION`` is pinned to the value `evaluation/claims/results/v5`
    was measured under, `667bcc9209126ddc`: excluding the display-only
    `_drop_non_verbatim_assertion_quotes` filter from what this digest hashes keeps the
    verdict-affecting policy's own identity stable even though the filter itself moved to
    `verify_claim_with_policy`'s caller, with no change to any verdict this policy can
    reach."""
    from app.services.fulltext import VERIFICATION_POLICY_VERSION

    assert VERIFICATION_POLICY_VERSION == "667bcc9209126ddc"


def test_verification_policy_version_is_unmoved_by_the_display_only_assertion_filter(
    monkeypatch,
):
    """The filter itself is free to change (it never touches a verdict) without moving the
    digest that identifies the policy, because it is no longer one of the digest's inputs
    and is no longer called from inside the hashed `verify_claim_with_policy` body."""
    import app.services.fulltext as ft

    original = ft._compute_verification_policy_version()
    monkeypatch.setattr(
        ft, "_drop_non_verbatim_assertion_quotes", lambda assertions, chunk_texts: []
    )
    changed = ft._compute_verification_policy_version()
    assert changed == original


def test_verification_policy_version_differs_from_quote_relocation_version():
    """Distinct digests for a reason: `QUOTE_RELOCATION_VERSION` identifies Part A alone
    (still used on its own, e.g. by a caller that only wants to know if the relocation rule
    changed); `VERIFICATION_POLICY_VERSION` identifies the whole policy, including Part B."""
    from app.services.fulltext import QUOTE_RELOCATION_VERSION, VERIFICATION_POLICY_VERSION

    assert VERIFICATION_POLICY_VERSION != QUOTE_RELOCATION_VERSION


def test_verification_policy_version_changes_when_the_trigger_reasons_change(monkeypatch):
    """`_SECOND_PASS_TRIGGER_REASONS` is a constant referenced by name inside
    `verify_claim_with_policy`'s body, so its *value* is not on that function's own source
    line and must be hashed by `repr`, the same way the relocation set's own word lists are
    (`test_quote_relocation.py`'s constant-moves-the-digest tests)."""
    import app.services.fulltext as ft

    original = ft._compute_verification_policy_version()
    monkeypatch.setattr(ft, "_SECOND_PASS_TRIGGER_REASONS", ("quote_not_verbatim", "extra"))
    changed = ft._compute_verification_policy_version()
    assert changed != original


def test_verification_policy_version_changes_when_the_relocation_set_changes(monkeypatch):
    """Part A is one half of the policy: a change to the relocation set (here, the article
    allowlist) must move `VERIFICATION_POLICY_VERSION` too, not just `QUOTE_RELOCATION_
    VERSION`."""
    import app.services.fulltext as ft

    original = ft._compute_verification_policy_version()
    monkeypatch.setattr(ft, "_RELOCATION_ARTICLES", frozenset(ft._RELOCATION_ARTICLES - {"an"}))
    changed = ft._compute_verification_policy_version()
    assert changed != original


def test_verification_policy_version_changes_when_the_repair_prompt_changes(monkeypatch):
    """The repair turn's own prompt text is not on `verify_claim_with_policy`'s own source
    line either (it lives in `app.agents.claim_verification_agent`), so it is folded in by
    its own version string, the same way the trigger reasons are folded in by `repr`
    (task authorisation 2026-09-11)."""
    import app.agents.claim_verification_agent as cva
    import app.services.fulltext as ft

    original = ft._compute_verification_policy_version()
    monkeypatch.setattr(cva, "QUOTE_REPAIR_PROMPT_VERSION", "sha256:changedchanged")
    changed = ft._compute_verification_policy_version()
    assert changed != original


def test_compute_verification_policy_version_falls_back_to_none_when_source_is_unavailable(
    monkeypatch, caplog
):
    import app.services.fulltext as ft

    def _raise(*_args, **_kwargs):
        raise OSError("source not available")

    monkeypatch.setattr(ft.inspect, "getsource", _raise)
    with caplog.at_level("WARNING"):
        result = ft._compute_verification_policy_version()
    assert result is None
    assert "VERIFICATION_POLICY_VERSION" in caplog.text


# --------------------------------------------------------------------------------------
# _non_verbatim_quote_segments: the repair turn's own input-gathering helper.
# --------------------------------------------------------------------------------------


def test_non_verbatim_quote_segments_finds_only_the_segments_that_fail():
    from app.services.fulltext import _non_verbatim_quote_segments

    assert _non_verbatim_quote_segments([CHUNK], [CHUNK]) == []
    assert _non_verbatim_quote_segments([BAD_QUOTE], [CHUNK]) == [BAD_QUOTE]
    assert _non_verbatim_quote_segments([CHUNK, BAD_QUOTE], [CHUNK]) == [BAD_QUOTE]


# --------------------------------------------------------------------------------------
# End to end: the production caller (`_verify_claims`, via `verify_and_heal_claims`)
# actually threads the shared policy through, including a repair turn when licensed, and
# that repair turn's own request really does continue the same pydantic-ai conversation.
# --------------------------------------------------------------------------------------

import os  # noqa: E402

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.models.analysis_job import JobType  # noqa: E402
from app.schemas.fulltext import ClaimVerification  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    FakeAgent,
    make_session_factory,
    seed_draft,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
    tiptap_paragraph,
)

_E2E_CHUNK_TEXT = (
    "Corrections could not be processed and the L2 learning could not take place for "
    "every enrolled participant across the whole trial period at every recruited site."
)
_E2E_BAD_QUOTE = _E2E_CHUNK_TEXT.replace(
    "and the L2 learning could not", "we the L2 learning could not"
)
_E2E_CHUNKS = [{"section": "results", "text": _E2E_CHUNK_TEXT}]
_E2E_CLAIM_TEXT = (
    "Corrections could not be processed and the L2 learning could not take place "
    "(Smith, 2020)."
)


async def _seed_policy_world(db_session):
    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": _E2E_CHUNKS},
    )
    draft = await seed_draft(db_session, project, user, tiptap_paragraph(_E2E_CLAIM_TEXT))
    job = await seed_job(db_session, project, JobType.claim_verify)
    return project, paper, draft, job


@pytest.mark.asyncio
async def test_verify_and_heal_claims_second_pass_fires_end_to_end(db_session):
    """"we" for "and" is the authorised motivating case: neither word is an article, a
    spelling variant or a citation, so Part A refuses it (`test_quote_relocation.py`'s
    ``test_we_and_case_no_longer_relocates_via_part_a``); the shared policy's bounded
    repair turn is what now resolves it end to end, and the production closure really
    does continue the first call's own pydantic-ai conversation (``message_history``) with
    the versioned repair prompt, not a fresh copy of the first request."""
    from app.agents.claim_verification_agent import (
        QUOTE_REPAIR_PROMPT_VERSION,
        format_quote_repair_prompt,
    )
    from app.services.fulltext import _non_verbatim_quote_segments, verify_and_heal_claims

    project, paper, draft, job = await _seed_policy_world(db_session)

    outputs = [
        ClaimVerification(
            claim_text=_E2E_CLAIM_TEXT,
            paper_id=paper.id,
            status="verified",
            evidence_quote=_E2E_BAD_QUOTE,
            explanation="First pass.",
        ),
        ClaimVerification(
            claim_text=_E2E_CLAIM_TEXT,
            paper_id=paper.id,
            status="verified",
            evidence_quote=_E2E_CHUNK_TEXT,
            explanation="Second pass, correctly copied.",
        ),
    ]
    calls = {"n": 0}

    def _next_output(_prompt):
        output = outputs[calls["n"]]
        calls["n"] += 1
        return output

    agent = FakeAgent(output=_next_output)

    factory, engine = make_session_factory()
    from unittest.mock import patch

    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await verify_and_heal_claims(
            project_id=project.id,
            draft_id=draft.id,
            job_id=job.id,
            session_factory=factory,
        )
    await engine.dispose()

    await db_session.refresh(job)
    assert job.error is None, job.error
    (verification,) = job.result["verifications"]
    assert len(agent.calls) == 2
    assert verification["status"] == "verified"
    assert verification["model_status"] == "verified"
    assert verification["machine_reasons"] == []
    assert len(verification["passes"]) == 2
    assert verification["passes"][0]["machine_reasons"] == ["quote_not_verbatim"]
    assert verification["passes"][0]["repair_prompt_version"] is None
    assert verification["passes"][1]["machine_reasons"] == []
    assert verification["passes"][1]["repair_prompt_version"] == QUOTE_REPAIR_PROMPT_VERSION
    assert verification["explanation"] == "Second pass, correctly copied."

    # The repair turn's own prompt names the exact segment that failed, and the first
    # call's prompt (the frozen ``format_verification_prompt`` output) is sent unchanged.
    first_prompt, _first_deps = agent.calls[0]
    second_prompt, _second_deps = agent.calls[1]
    failed_segments = _non_verbatim_quote_segments([_E2E_BAD_QUOTE], [_E2E_CHUNK_TEXT])
    assert second_prompt == format_quote_repair_prompt(failed_segments)
    assert second_prompt != first_prompt

    # The repair call continues the SAME conversation: its own `message_history` is
    # exactly the first call's own result's `all_messages()`.
    assert agent.message_histories[0] is None
    assert agent.message_histories[1] == agent.results[0].all_messages()

    provenance = job.result["provenance"]
    assert provenance["calls"] == 2
    assert provenance["repair_prompt_version"] == QUOTE_REPAIR_PROMPT_VERSION
