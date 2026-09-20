"""Tests for the Claim Verification Agent."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def _normalized(text: str) -> str:
    """Collapse all whitespace runs (including line wraps) to single spaces, so a
    substring assertion is robust to how a sentence happens to be word-wrapped in the
    triple-quoted prompt source."""
    return " ".join(text.split())


def test_claim_verification_agent_exists():
    from app.agents.claim_verification_agent import get_claim_verification_agent

    agent = get_claim_verification_agent()
    assert agent is not None


def test_format_verification_prompt_includes_all_inputs():
    from app.agents.claim_verification_agent import format_verification_prompt

    result = format_verification_prompt(
        claim_text="AI tutors improve student performance by 20%.",
        chunks=[
            "In our study, students using AI-assisted tutoring showed a 20% improvement.",
            "The control group showed no significant change in performance.",
        ],
        paper_title="AI-Assisted Tutoring in Higher Education",
        paper_authors=["Smith, J.", "Lee, K."],
    )
    assert "AI tutors improve student performance by 20%." in result
    assert "AI-Assisted Tutoring in Higher Education" in result
    assert "Smith, J." in result
    assert "Lee, K." in result
    assert "Chunk 1" in result
    assert "Chunk 2" in result
    assert "20% improvement" in result


def test_format_verification_prompt_no_chunks():
    from app.agents.claim_verification_agent import format_verification_prompt

    result = format_verification_prompt(
        claim_text="Some claim about the paper.",
        chunks=[],
        paper_title="A Paper Title",
        paper_authors=None,
    )
    assert "Some claim about the paper." in result
    assert "A Paper Title" in result
    assert "Unknown" in result  # No authors provided
    assert "No full-text chunks available" in result


def test_format_verification_prompt_with_no_authors():
    from app.agents.claim_verification_agent import format_verification_prompt

    result = format_verification_prompt(
        claim_text="Test claim.",
        chunks=["Some text chunk."],
        paper_title="Test Paper",
        paper_authors=[],
    )
    # Empty list should still produce a sensible output
    assert "Test Paper" in result
    assert "Test claim." in result
    assert "Chunk 1" in result


def test_claim_verification_agent_is_deterministic_and_versioned():
    """Temperature must be pinned to 0, the long timeout kept, and the prompt version exposed."""
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        VERIFICATION_PROMPT,
        get_claim_verification_agent,
    )
    from app.agents.model_config import DETERMINISTIC_LONG_MODEL_SETTINGS, prompt_version

    agent = get_claim_verification_agent()
    assert agent.model_settings["temperature"] == 0.0
    assert agent.model_settings["timeout"] == DETERMINISTIC_LONG_MODEL_SETTINGS["timeout"]
    assert CLAIM_VERIFICATION_PROMPT_VERSION == prompt_version(VERIFICATION_PROMPT)
    assert CLAIM_VERIFICATION_PROMPT_VERSION.startswith("sha256:")


def test_model_facing_output_schema_has_no_provenance_fields():
    """``model_reported`` is filled by the service; the model never sees it."""
    from app.agents.claim_verification_agent import get_claim_verification_agent
    from app.schemas.fulltext import ClaimVerification, ClaimVerificationOutput

    agent = get_claim_verification_agent()
    assert agent.output_type is ClaimVerificationOutput
    assert "model_reported" not in ClaimVerificationOutput.model_json_schema()["properties"]
    # The stored record is a strict superset of what the model returns.
    assert set(ClaimVerificationOutput.model_fields) < set(ClaimVerification.model_fields)


def test_claim_verification_agent_uses_fingerprint_capturing_deepseek_model():
    """The claim verifier, like the screener, is built via
    ``build_deepseek_model`` so DeepSeek's ``system_fingerprint`` lands in the
    provenance record; the long timeout tier is unchanged."""
    from app.agents.claim_verification_agent import get_claim_verification_agent
    from app.agents.model_config import LONG_MODEL_SETTINGS, DeepSeekChatModel
    from app.config import settings

    agent = get_claim_verification_agent()
    assert isinstance(agent.model, DeepSeekChatModel)
    assert agent.model.model_name == settings.deepseek_model
    assert agent.model_settings["timeout"] == LONG_MODEL_SETTINGS["timeout"] == 300


# --- v1 / v2 / v3 identity -------------------------------------------------------


def test_verification_prompt_v1_is_retained_byte_identical():
    """v1 stays exported and its sha is unchanged, so anyone re-scoring the v1
    evaluation runs (evaluation/claims/results/, frozen) gets the exact prompt that
    produced them."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V1
    from app.agents.model_config import prompt_version

    assert prompt_version(VERIFICATION_PROMPT_V1) == "sha256:e5fcad829dd9"
    assert "no supporting evidence is found" in VERIFICATION_PROMPT_V1
    assert "needs_nuance" in VERIFICATION_PROMPT_V1


def test_verification_prompt_v2_is_retained_byte_identical():
    """v2 stays exported and byte exact even though v3 is
    now the active prompt, so anyone re-scoring the v2 evaluation runs
    (evaluation/claims/results/v2/, frozen) gets the exact prompt that produced them."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V2
    from app.agents.model_config import prompt_version

    assert prompt_version(VERIFICATION_PROMPT_V2) == "sha256:e3c9f5e99a43"
    assert "contradicted" in VERIFICATION_PROMPT_V2
    assert "One refuted proposition condemns the whole claim" in VERIFICATION_PROMPT_V2


def test_verification_prompt_v3_is_the_active_prompt():
    """v3 is what the agent actually runs.
    ``CLAIM_VERIFICATION_PROMPT_VERSION`` is derived from it and differs from both the
    v1 and the v2 shas, which stay byte exact and unaffected by the repoint."""
    from app.agents.claim_verification_agent import (
        CLAIM_VERIFICATION_PROMPT_VERSION,
        VERIFICATION_PROMPT,
        VERIFICATION_PROMPT_V1,
        VERIFICATION_PROMPT_V2,
        VERIFICATION_PROMPT_V3,
    )
    from app.agents.model_config import prompt_version

    assert VERIFICATION_PROMPT is VERIFICATION_PROMPT_V3
    assert VERIFICATION_PROMPT != VERIFICATION_PROMPT_V1
    assert VERIFICATION_PROMPT != VERIFICATION_PROMPT_V2
    assert prompt_version(VERIFICATION_PROMPT_V1) == "sha256:e5fcad829dd9"
    assert prompt_version(VERIFICATION_PROMPT_V2) == "sha256:e3c9f5e99a43"
    assert CLAIM_VERIFICATION_PROMPT_VERSION == prompt_version(VERIFICATION_PROMPT_V3)
    assert CLAIM_VERIFICATION_PROMPT_VERSION != prompt_version(VERIFICATION_PROMPT_V1)
    assert CLAIM_VERIFICATION_PROMPT_VERSION != prompt_version(VERIFICATION_PROMPT_V2)


# --- v3 required content (table rows V3-1..V3-13) ------


def test_verification_prompt_v3_states_closed_assertion_inventory():
    """V3-1: the assertion inventory is closed to the propositions the claim's own
    words assert; an assertion may never encode the absence of a qualifier, and the
    named forbidden forms are excluded literally."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "Decompose the claim into the propositions its own words assert and only those."
        in prompt
    )
    assert "Never write an assertion whose content is the absence of a qualifier." in prompt
    for forbidden in (
        '"generally"',
        '"without restriction"',
        '"for all"',
        '"unqualified"',
        '"categorical"',
        '"asserted without limitation"',
        '"stated as certain"',
    ):
        assert forbidden in prompt


def test_verification_prompt_v3_states_explicit_quantifier_licensing():
    """V3-2: only an explicit quantifier word in the claim licenses a quantifier
    assertion, and that assertion is the word itself and nothing more."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        'only an explicit quantifier in the claim ("all", "every", "always", "never", '
        '"none", "only", "most", "few", "no") licenses a quantifier assertion, and the '
        "assertion is that word and nothing more" in prompt
    )


def test_verification_prompt_v3_states_anchored_contradiction():
    """V3-3 (changed from v2): ``contradicted`` fires only on a same-entity,
    same-measure span with a different value, direction, polarity or quantifier, and
    all five identifying fields are recorded."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        '"contradicted" only when one quoted span has the same entity, the same '
        "measure and a different value, direction, polarity or quantifier; entity, "
        "measure, claim_value, source_value and the span are all recorded." in prompt
    )


def test_verification_prompt_v3_states_distinct_entity_never_contradicts():
    """V3-4: a source statement about a subgroup, a different outcome, data source,
    time window or model system is about a different entity or measure, so it can
    never contradict the claim's assertion."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "A source statement about a subgroup, a different outcome, data source, time "
        "window or model system is about a different entity or measure and never "
        "contradicts." in prompt
    )


def test_verification_prompt_v3_states_supported_composition_rules():
    """V3-5: ``supported`` covers paraphrase, instance/class generalisation, bounded
    two-statement conjunction, and unit/rate/denominator/fraction/percentage
    conversion at the claim's stated precision; every span used is quoted."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        '"supported" when the chunks state it in other words with the same meaning; '
        "about a named instance of a class the assertion names, or as a class "
        "generalising over an instance it names; across two statements whose "
        "conjunction yields it with no added premise; or as a numerically equivalent "
        "value after unit, rate, denominator, fraction and percentage conversion at "
        "the precision the claim states." in prompt
    )
    assert "Every span used is quoted" in prompt


def test_verification_prompt_v3_states_bounded_composition():
    """V3-6: composition may never change a value, direction word, polarity or
    quantifier, and a round-multiple claim number is never treated as a unit
    conversion of a source number."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "Composition may never change a value, direction word, polarity or "
        "quantifier, and a claim number that is a round multiple of a source number "
        "is never a conversion." in prompt
    )


def test_verification_prompt_v3_states_single_central_assertion():
    """V3-7: exactly one assertion is CENTRAL, the proposition the sentence exists to
    state; every other assertion is peripheral."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "Exactly one assertion is marked CENTRAL: the proposition the sentence exists "
        "to state, its main predicate over its main subject. All others are "
        "peripheral." in prompt
    )


def test_verification_prompt_v3_states_precedence_order():
    """V3-8 (changed from v2): the six-step precedence order -- contradicted anywhere,
    central absent, principal entity absent, peripheral absent, hedge fires,
    otherwise verified."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        'Precedence in order: contradicted anywhere gives "unsupported"; central '
        'absent gives "unsupported"; principal entity absent from every chunk gives '
        '"unsupported"; peripheral absent gives "needs_nuance" and enters '
        'unstated_details; the hedge test firing gives "needs_nuance"; otherwise '
        '"verified".' in prompt
    )


def test_verification_prompt_v3_states_hedge_test():
    """V3-9 (changed from v2): a hedge assertion exists only when the source hedges and
    the claim removes it; a categorical source with a hedged claim, or equal force,
    gives no assertion; reporting-verb synonyms, the causal verb for a demonstrated
    relation, and naming the model system are not hedges."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "A hedge assertion exists only when the source hedges and the claim removes "
        "the hedge." in prompt
    )
    assert "Categorical source with hedged claim, or equal force, gives no assertion." in prompt
    assert (
        "Reporting-verb synonyms, the causal verb chosen for a demonstrated relation, "
        "and naming the model system are not hedges." in prompt
    )


def test_verification_prompt_v3_states_principal_entity_absent():
    """V3-10 (kept): when no chunk mentions the claim's principal entity the status is
    unsupported. The bullet names
    the CENTRAL assertion's entity as the entity it means, so this bullet and the
    peripheral/principal-entity precedence clause (V3-12's clause (b), which defines the
    principal entity the same way) read as one rule instead of two that a model could see
    as competing."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "when no chunk mentions the claim's principal entity, meaning the CENTRAL "
        'assertion\'s entity, the status is "unsupported"' in prompt
    )


def test_verification_prompt_v3_states_judge_as_written():
    """V3-11: judge the claim exactly as written, including apparent typos; a claim
    and source differ if their words differ, however plausible the difference."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert "Judge the claim exactly as written." in prompt
    assert "A word you believe to be a typographical error is still the claim's word." in prompt
    assert (
        '"no X" and "X" differ, "increase" and "decrease" differ, 1.62 and 0.81 differ.'
        in prompt
    )
    assert "Do not infer from context what the author meant." in prompt


def test_verification_prompt_v3_states_disagreement_never_a_nuance():
    """V3-12 (new, verbatim): a value/direction/polarity/quantifier disagreement on the
    same entity and measure is unsupported, not a nuance, regardless of how much of
    the claim otherwise reproduces the source or how plausible a transcription error
    would be."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "A disagreement in a value, a direction, a polarity or a quantifier is never "
        "a nuance. If a source span exists about the same entity and the same "
        "measure and it states a different value, direction, polarity or quantifier, "
        "the status is unsupported, no matter how much of the rest of the claim "
        "reproduces the source word for word, and no matter how plausible it is that "
        "the difference was an error of transcription." in prompt
    )


# --- v3 STEP 3 precedence refinement ------


def test_verification_prompt_v3_states_added_detail_absent_is_peripheral():
    """P11b clause (a): a detail the claim adds, absent from every chunk, is peripheral
    and absent -- it never gets promoted into the CENTRAL or principal-entity checks."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "An assertion whose content is a detail the claim adds, and which is absent "
        "from every chunk, is peripheral and absent." in prompt
    )


def test_verification_prompt_v3_states_principal_entity_is_central_entity_only():
    """P11b clause (b): the principal entity is the CENTRAL assertion's entity and no
    other's, so an added modifier can never be read as the principal entity."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "The claim's principal entity is the entity of the CENTRAL assertion and of no "
        "other assertion, so a modifier the claim adds can never be the principal "
        "entity." in prompt
    )


def test_verification_prompt_v3_states_enumeration_without_naming_is_different_measure():
    """P11b clause (c): a source passage that lists other instruments, settings,
    populations or time windows without naming the claim's is a different measure and
    never contradicts."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "A source passage that enumerates other instruments, settings, populations or "
        "time windows without naming the claim's is a different measure and never "
        "contradicts." in prompt
    )


def test_verification_prompt_v3_states_supported_central_with_absent_peripheral_never_unsupported():
    """P11b clause (d): when the CENTRAL assertion is supported and nothing is
    contradicted, an absent peripheral assertion floors the claim at needs_nuance and
    the status is never unsupported on that ground."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        'When the central assertion is supported and no assertion is contradicted, the '
        'status is "needs_nuance" whenever any peripheral assertion is absent, and it '
        'is never "unsupported".' in prompt
    )


def test_verification_prompt_v3_states_step4_field_list():
    """V3-13 (changed from v2): STEP 4's output field list is v2's five fields plus
    ``unstated_details`` and ``evidence_quotes``; ``no_full_text`` stays forbidden as a
    model answer."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    for field in (
        "- status:",
        "- assertions:",
        "- evidence_quote:",
        "- explanation:",
        "- suggested_revision:",
        "- unstated_details:",
        "- evidence_quotes:",
    ):
        assert field in prompt
    assert 'Do NOT return "no_full_text".' in prompt
    assert (
        'For "unsupported" this is the span that refutes the claim, or null when nothing '
        "in the chunks bears on it." in prompt
    )
    assert 'For "verified" and "needs_nuance" a quote is required.' in prompt


def test_verification_prompt_v3_states_assertion_kind_vocabulary():
    """STEP 4's assertions bullet must name the
    ``kind`` field and its allowed vocabulary, matching the ClaimAssertion.kind
    comment in backend/app/schemas/fulltext.py:21-22 exactly, so the model does not
    invent a value for a required schema field with no default."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        'a kind: one of "population", "intervention", "quantity", "direction", '
        '"scope", "attribution" or "other"' in prompt
    )


# --- v3 required absences (V3-A..V3-D) -----------------


def test_verification_prompt_v3_omits_look_elsewhere_for_generality():
    """V3-A: no clause reads "unless the source itself questions whether the finding
    extends beyond that system" and no other instruction sends the model looking
    elsewhere in the source for a finding's generality (the iteration-3 regression on
    hss-verbatim-05)."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        "unless the source itself questions whether the finding extends beyond that "
        "system" not in prompt
    )
    assert "extends beyond that system" not in prompt
    assert "look elsewhere in the source" not in prompt


def test_verification_prompt_v3_omits_unqualified_any_assertion_absent():
    """V3-B: v2's unqualified "if ANY assertion is absent, the status is unsupported"
    is gone. Absence only floors unsupported when it is the CENTRAL assertion;
    absence of a peripheral assertion floors only needs_nuance."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3).lower()
    assert 'if any assertion is "absent"' not in prompt


def test_verification_prompt_v3_omits_v2_single_span_verified_rule():
    """V3-C: v2's single-span rule, "Never mark verified unless a verbatim quote
    supports every assertion", is gone -- V3-5 allows a supported verdict built by
    composing more than one quoted span."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert (
        'Never mark "verified" unless a verbatim quote from the chunks supports every '
        "assertion" not in prompt
    )
    assert "verbatim quote from the chunks supports every assertion" not in prompt


def test_verification_prompt_v3_omits_unbounded_wording_charity_clause():
    """V3-D: v2's unbounded charity clause about wording (any reporting-verb
    difference is never itself an overstatement; any omitted methodological detail is
    never itself grounds for needs_nuance) is gone. V3-9's hedge exceptions are a
    closed, named list instead of an open licence."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    assert "is never by itself an overstatement" not in prompt
    assert "is never by itself grounds for" not in prompt
    assert "omitting a methodological detail" not in prompt


def test_verification_prompt_v3_documents_the_four_unsupported_anchors_jointly():
    """One documentation test (brief section 1.1): V3-3, V3-6, V3-11 and V3-12 are the
    four sentences that jointly hold every altered construction at unsupported, and
    they must all be present together in the released prompt."""
    from app.agents.claim_verification_agent import VERIFICATION_PROMPT_V3

    prompt = _normalized(VERIFICATION_PROMPT_V3)
    # V3-3
    assert (
        '"contradicted" only when one quoted span has the same entity, the same '
        "measure and a different value, direction, polarity or quantifier" in prompt
    )
    # V3-6
    assert (
        "Composition may never change a value, direction word, polarity or "
        "quantifier, and a claim number that is a round multiple of a source number "
        "is never a conversion." in prompt
    )
    # V3-11
    assert (
        '"no X" and "X" differ, "increase" and "decrease" differ, 1.62 and 0.81 '
        "differ." in prompt
    )
    # V3-12, verbatim
    assert (
        "A disagreement in a value, a direction, a polarity or a quantifier is never "
        "a nuance. If a source span exists about the same entity and the same "
        "measure and it states a different value, direction, polarity or quantifier, "
        "the status is unsupported, no matter how much of the rest of the claim "
        "reproduces the source word for word, and no matter how plausible it is that "
        "the difference was an error of transcription." in prompt
    )


# --- QUOTE_REPAIR_PROMPT: must license a status change ------------


def test_quote_repair_prompt_does_not_forbid_a_status_change():
    """M1: the repair turn is only ever sent after the model said `verified` and the
    quote guard demoted it, so "do not change status" always means "answer verified
    again" -- exactly the outcome that lets a fabricated-but-verbatim, unrelated span
    slip back to `verified`. The prompt must not name `status` among the fields the
    model is told to leave unchanged."""
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT

    prompt = _normalized(QUOTE_REPAIR_PROMPT)
    assert "do not change status" not in prompt
    assert "do not change suggested_revision" in prompt


def test_quote_repair_prompt_states_the_no_support_alternative():
    """M1's required fix: the prompt must tell the model plainly that when no exact
    span of the source supports the claim, it should change status instead of
    substituting an unrelated but verbatim span."""
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT

    prompt = _normalized(QUOTE_REPAIR_PROMPT)
    assert "no exact span of the source supports the claim" in prompt
    assert "change status to reflect that" in prompt


def test_quote_repair_prompt_licenses_explanation_and_verdict_to_follow_a_status_change():
    """A repair answer that honestly reports no
    supporting span must not be forced to keep the first pass's own explanation (which
    argued the claim WAS supported) or assertion verdicts still reading `supported` next
    to a `status` of `unsupported` -- that contradiction is exactly what the claim
    report, the claim-record export and the UI would then show beside the new status.
    The rest of the do-not-change list (suggested_revision, and any assertion's entity,
    measure, claim_value or source_value) must stay intact."""
    from app.agents.claim_verification_agent import QUOTE_REPAIR_PROMPT

    prompt = _normalized(QUOTE_REPAIR_PROMPT)
    assert (
        "do not change suggested_revision, or any assertion's entity, measure, "
        "claim_value or source_value" in prompt
    )
    assert "also update explanation and any assertion's verdict" in prompt
