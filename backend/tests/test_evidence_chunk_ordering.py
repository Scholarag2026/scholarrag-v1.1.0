"""Evidence-first chunk ordering.

When a claim's own citation link carries ``evidence_ids``, the chunk(s) holding those
evidence quotes are moved to the front of the list sent to the verifier for that one
claim; every other chunk keeps its stored relative order and the total chunk count never
changes. The claim-verification prompt template and every guard function are untouched --
only the order of what `_verify_claims` hands them changes, pinned here by re-asserting
both frozen digests: the claim-verification prompt and `apply_verification_guards` and
its seven guard functions are frozen and never to be edited.
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobType  # noqa: E402
from app.schemas.fulltext import ClaimVerification  # noqa: E402
from app.services.fulltext import (  # noqa: E402
    _build_paper_lookup,
    _evidence_first_chunk_order,
    claims_from_citation_links,
    evidence_quotes_by_claim_map,
)
from tests.t5_fixtures import (  # noqa: E402
    FakeAgent,
    make_session_factory,
    seed_job,
    seed_project,
    seed_user,
)

CHUNK_A = {"section": "intro", "text": "Chunk A text, unrelated to the claim."}
CHUNK_B = {"section": "results", "text": "Chunk B has the key finding: tutoring works great."}
CHUNK_C = {"section": "discussion", "text": "Chunk C text, also unrelated."}


# --------------------------------------------------------------------------------------
# `_evidence_first_chunk_order`: pure function.
# --------------------------------------------------------------------------------------


def test_evidence_first_chunk_order_moves_the_matching_chunk_first():
    chunks = [CHUNK_A, CHUNK_B, CHUNK_C]
    texts = [c["text"] for c in chunks]

    ordered_chunks, ordered_texts = _evidence_first_chunk_order(
        chunks, texts, ["Chunk B has the key finding: tutoring works great."]
    )

    assert ordered_texts[0] == CHUNK_B["text"]
    assert len(ordered_chunks) == len(chunks) == 3
    # The rest keep their original relative order.
    assert ordered_texts[1:] == [CHUNK_A["text"], CHUNK_C["text"]]
    # Original inputs are never mutated.
    assert texts == [CHUNK_A["text"], CHUNK_B["text"], CHUNK_C["text"]]


def test_evidence_first_chunk_order_is_a_noop_without_evidence_quotes():
    chunks = [CHUNK_A, CHUNK_B]
    texts = [c["text"] for c in chunks]

    ordered_chunks, ordered_texts = _evidence_first_chunk_order(chunks, texts, None)

    assert ordered_chunks is chunks
    assert ordered_texts is texts


def test_evidence_first_chunk_order_is_a_noop_when_the_quote_is_not_found():
    chunks = [CHUNK_A, CHUNK_B]
    texts = [c["text"] for c in chunks]

    ordered_chunks, ordered_texts = _evidence_first_chunk_order(
        chunks, texts, ["This sentence appears nowhere in any chunk."]
    )

    assert ordered_chunks is chunks
    assert ordered_texts is texts


def test_evidence_first_chunk_order_total_chunk_count_is_always_unchanged():
    chunks = [CHUNK_A, CHUNK_B, CHUNK_C]
    texts = [c["text"] for c in chunks]

    ordered_chunks, ordered_texts = _evidence_first_chunk_order(
        chunks, texts, [CHUNK_C["text"]]
    )

    assert len(ordered_chunks) == 3
    assert sorted(ordered_texts) == sorted(texts)


# --------------------------------------------------------------------------------------
# `claims_from_citation_links` / `evidence_quotes_by_claim_map`.
# --------------------------------------------------------------------------------------


def test_claims_from_citation_links_builds_one_claim_per_key_and_dedupes():
    links = [
        {
            "sentence": "Tutoring works (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
            "proposition": "Tutoring works",
        },
        # A duplicate (sentence, key) pair -- collapses to the first citation_text.
        {
            "sentence": "Tutoring works (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020) [dup]",
            "proposition": "Tutoring works",
        },
        {
            "sentence": "Two papers agree (Jones, 2019; Lee, 2021).",
            "keys": ["jones_2019", "lee_2021"],
            "citation_text": "(Jones, 2019; Lee, 2021)",
            "proposition": "",
        },
    ]

    claims = claims_from_citation_links(links)

    assert claims == [
        ("Tutoring works", "smith_2020", "(Smith, 2020)", "Tutoring works (Smith, 2020)."),
        (
            "Two papers agree (Jones, 2019; Lee, 2021).",
            "jones_2019",
            "(Jones, 2019; Lee, 2021)",
            "Two papers agree (Jones, 2019; Lee, 2021).",
        ),
        (
            "Two papers agree (Jones, 2019; Lee, 2021).",
            "lee_2021",
            "(Jones, 2019; Lee, 2021)",
            "Two papers agree (Jones, 2019; Lee, 2021).",
        ),
    ]


def test_evidence_quotes_by_claim_map_resolves_catalog_ids_to_quotes():
    links = [
        {
            "sentence": "Tutoring works (Smith, 2020).",
            "keys": ["smith_2020"],
            "citation_text": "(Smith, 2020)",
            "evidence_ids": ["e1", "e9-unknown"],
        },
        {
            "sentence": "No evidence here (Jones, 2019).",
            "keys": ["jones_2019"],
            "citation_text": "(Jones, 2019)",
            "evidence_ids": [],
        },
    ]
    catalog = [{"id": "e1", "key": "smith_2020", "quote": "Tutoring works great."}]

    result = evidence_quotes_by_claim_map(links, catalog)

    assert result == {("Tutoring works (Smith, 2020).", "smith_2020"): ["Tutoring works great."]}


def test_evidence_quotes_by_claim_map_is_empty_without_a_catalog():
    links = [{"sentence": "S (Smith, 2020).", "keys": ["smith_2020"], "evidence_ids": ["e1"]}]
    assert evidence_quotes_by_claim_map(links, None) == {}
    assert evidence_quotes_by_claim_map(links, []) == {}


# --------------------------------------------------------------------------------------
# End-to-end through `_verify_claims`: the prompt actually sent, and the resulting
# evidence_location, both reflect the reordered list.
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_claims_sends_the_evidence_chunk_first(db_session):
    from app.services.fulltext import _verify_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    from tests.t5_fixtures import seed_paper

    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [CHUNK_A, CHUNK_B, CHUNK_C],
        },
    )
    job = await seed_job(db_session, project, JobType.claim_verify)
    paper_lookup = _build_paper_lookup([paper])

    claim_sentence = "Tutoring works great (Smith, 2020)."
    claims = [(claim_sentence, "smith_2020", "(Smith, 2020)", claim_sentence)]
    evidence_map = {(claim_sentence, "smith_2020"): [CHUNK_B["text"]]}

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text=claim_sentence,
            paper_id=paper.id,
            status="verified",
            evidence_quote="Chunk B has the key finding: tutoring works great.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        outcome = await _verify_claims(
            claims,
            paper_lookup,
            SimpleNamespace(),
            job.id,
            factory,
            evidence_quotes_by_claim=evidence_map,
        )
    await engine.dispose()

    assert len(agent.calls) == 1
    prompt, _deps = agent.calls[0]
    # Chunk B (the evidence chunk) is shown first; the total chunk count (3) is
    # unchanged; the other two keep their original relative order after it.
    chunk_order = [
        line for line in prompt.splitlines() if line.startswith("--- Chunk")
    ]
    assert len(chunk_order) == 3
    b_index = prompt.index(CHUNK_B["text"])
    a_index = prompt.index(CHUNK_A["text"])
    c_index = prompt.index(CHUNK_C["text"])
    assert b_index < a_index < c_index

    # `evidence_location` numbers the chunk against the same reordered list the model
    # was shown, so "chunk 1" here correctly names the evidence chunk, not its stored
    # position (2).
    verification = outcome["verifications"][0]
    assert verification.status == "verified"
    assert verification.evidence_location == "chunk 1 (results)"


@pytest.mark.asyncio
async def test_verify_claims_keeps_stored_order_without_evidence_ids(db_session):
    """The pre-existing, byte-identical path (design section 6 amendment A6): a claim
    with no evidence quotes reaches the verifier with chunks in stored order, exactly
    as before this feature existed."""
    from app.services.fulltext import _verify_claims

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    from tests.t5_fixtures import seed_paper

    paper = await seed_paper(
        db_session,
        project,
        title="Cited Work",
        authors=[{"name": "Jane Smith"}],
        year=2020,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [CHUNK_A, CHUNK_B, CHUNK_C],
        },
    )
    job = await seed_job(db_session, project, JobType.claim_verify)
    paper_lookup = _build_paper_lookup([paper])

    claim_sentence = "Tutoring works great (Smith, 2020)."
    claims = [(claim_sentence, "smith_2020", "(Smith, 2020)", claim_sentence)]

    agent = FakeAgent(
        output=ClaimVerification(
            claim_text=claim_sentence,
            paper_id=paper.id,
            status="verified",
            evidence_quote="Chunk A text, unrelated to the claim.",
            explanation="Supported.",
        )
    )

    factory, engine = make_session_factory()
    with patch(
        "app.agents.claim_verification_agent.get_claim_verification_agent",
        return_value=agent,
    ):
        await _verify_claims(claims, paper_lookup, SimpleNamespace(), job.id, factory)
    await engine.dispose()

    prompt, _deps = agent.calls[0]
    a_index = prompt.index(CHUNK_A["text"])
    b_index = prompt.index(CHUNK_B["text"])
    c_index = prompt.index(CHUNK_C["text"])
    assert a_index < b_index < c_index


# --------------------------------------------------------------------------------------
# Frozen digests: this feature reorders chunks in code only. Both digests the
# authorisation names must be exactly what they were before this task.
# --------------------------------------------------------------------------------------


def test_claim_verification_prompt_is_still_frozen():
    from app.agents.claim_verification_agent import CLAIM_VERIFICATION_PROMPT_VERSION

    assert CLAIM_VERIFICATION_PROMPT_VERSION == "sha256:49fcfbfaf2f6"


def test_verification_guards_are_still_frozen():
    """Second, independent copy of `test_verification_guards.py`'s own digest computation
    (that file's docstring gives the full reasoning); moved once, deliberately, for guard 7
    (task authorisation 2026-09-14)."""
    import hashlib
    import inspect

    from app.services import fulltext

    names = [
        "apply_verification_guards",
        "_guard_no_full_text_with_chunks",
        "_guard_attribution",
        "_guard_assertion_status_consistency",
        "_guard_numeric_tier_b",
        "_guard_numeric_value_absent",
        "_guard_quote_fidelity",
        "_guard_scale_fidelity",
        "scale_alignment_findings",
        "_scale_negation_scope_shift",
        "_scale_tokens",
        "_scale_stem",
        "_scale_is_content_token",
        "_scale_content_indices",
        "_scale_head_set",
        "_scale_context_set",
        "_scale_occurrences",
        "_scale_admissible_pairs",
        "_scale_align",
    ]
    digest = hashlib.sha256()
    for name in names:
        digest.update(inspect.getsource(getattr(fulltext, name)).encode("utf-8"))
    for value in (
        fulltext._SCALE_CURLY_FOLD,
        fulltext._SCALE_EXTENT_WORDS,
        fulltext._SCALE_DEGREE_WORDS,
        fulltext._SCALE_HEDGE_WORDS,
        fulltext._SCALE_BAND_BY_RANK,
    ):
        digest.update(repr(value).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_FUNCTION_WORDS)).encode("utf-8"))
    digest.update(repr(fulltext._SCALE_EPISTEMIC_VERB_STEMS).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_ABSOLUTE_NEGATORS)).encode("utf-8"))
    digest.update(repr(sorted(fulltext._SCALE_NEGATORS)).encode("utf-8"))
    for value in (
        fulltext._SCALE_HEAD_RIGHT_SPAN,
        fulltext._SCALE_HEAD_RIGHT_MAX,
        fulltext._SCALE_HEAD_LEFT_SPAN,
        fulltext._SCALE_HEAD_LEFT_MAX,
        fulltext._SCALE_CONTEXT_SPAN,
        fulltext._SCALE_CONTEXT_MAX,
        fulltext._SCALE_MIN_HEAD_OVERLAP,
        fulltext._SCALE_MIN_CONTEXT_OVERLAP,
    ):
        digest.update(repr(value).encode("utf-8"))
    assert digest.hexdigest()[:16] == fulltext.GUARD_DIGEST == "8a6c833ffc329f89"
