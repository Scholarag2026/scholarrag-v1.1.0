"""Tests for full-text and claim verification schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError


def test_claim_verification_minimal():
    from app.schemas.fulltext import ClaimVerification

    paper_id = uuid4()
    cv = ClaimVerification(
        claim_text="Students who use AI tutors perform 20% better.",
        paper_id=paper_id,
        status="verified",
        explanation="The paper explicitly states this finding in section 4.2.",
    )
    assert cv.claim_text == "Students who use AI tutors perform 20% better."
    assert cv.paper_id == paper_id
    assert cv.status == "verified"
    assert cv.evidence_quote is None
    assert cv.suggested_revision is None


def test_claim_verification_full():
    from app.schemas.fulltext import ClaimVerification

    paper_id = uuid4()
    cv = ClaimVerification(
        claim_text="Gamification always improves learning outcomes.",
        paper_id=paper_id,
        status="needs_nuance",
        evidence_quote="Gamification improved outcomes in 7 of 12 studies reviewed.",
        explanation="The claim overstates; the paper shows mixed results.",
        suggested_revision="Gamification improved learning outcomes in a majority of studies reviewed.",
    )
    assert cv.status == "needs_nuance"
    assert cv.evidence_quote is not None
    assert cv.suggested_revision is not None


def test_claim_verification_missing_required_field():
    from app.schemas.fulltext import ClaimVerification

    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_text="Some claim",
            # missing paper_id, status, explanation
        )


def test_claim_verification_report():
    from app.schemas.fulltext import ClaimVerification, ClaimVerificationReport

    draft_id = uuid4()
    paper_id = uuid4()
    verifications = [
        ClaimVerification(
            claim_text="Claim A",
            paper_id=paper_id,
            status="verified",
            explanation="Supported by evidence.",
        ),
        ClaimVerification(
            claim_text="Claim B",
            paper_id=paper_id,
            status="unsupported",
            explanation="No evidence found.",
        ),
        ClaimVerification(
            claim_text="Claim C",
            paper_id=paper_id,
            status="needs_nuance",
            explanation="Partially supported.",
        ),
    ]
    report = ClaimVerificationReport(
        draft_id=draft_id,
        verifications=verifications,
        verified_count=1,
        unsupported_count=1,
        nuance_count=1,
        abstract_only_count=0,
    )
    assert report.draft_id == draft_id
    assert len(report.verifications) == 3
    assert report.verified_count == 1
    assert report.unsupported_count == 1
    assert report.nuance_count == 1
    assert report.abstract_only_count == 0


def test_claim_verification_report_empty():
    from app.schemas.fulltext import ClaimVerificationReport

    report = ClaimVerificationReport(
        draft_id=uuid4(),
        verifications=[],
        verified_count=0,
        unsupported_count=0,
        nuance_count=0,
        abstract_only_count=0,
    )
    assert report.verifications == []
    assert report.verified_count == 0


def test_paste_fulltext_request():
    from app.schemas.fulltext import PasteFulltextRequest

    req = PasteFulltextRequest(text="This is the full text of the paper...")
    assert req.text == "This is the full text of the paper..."


def test_paste_fulltext_request_missing_text():
    from app.schemas.fulltext import PasteFulltextRequest

    with pytest.raises(ValidationError):
        PasteFulltextRequest()


# ---------- ClaimAssertion, assertions, model_status, machine_reasons ----------


def test_claim_assertion_requires_text_kind_verdict_only():
    from app.schemas.fulltext import ClaimAssertion

    a = ClaimAssertion(
        text="Sample size was 40 participants.", kind="population", verdict="supported"
    )
    assert a.claim_value is None and a.source_value is None and a.quote is None


def test_claim_verification_output_assertions_default_empty_and_is_additive():
    """The flat read contract: assertions is optional,
    so an old-shape dict (no ``assertions`` key at all) still validates."""
    from app.schemas.fulltext import ClaimAssertion, ClaimVerificationOutput

    old_shape = ClaimVerificationOutput(
        claim_text="Old-shape claim.",
        paper_id=uuid4(),
        status="verified",
        explanation="ok",
    )
    assert old_shape.assertions == []

    out = ClaimVerificationOutput(
        claim_text="Scores rose 12%.",
        paper_id=uuid4(),
        status="unsupported",
        explanation="The source reports 8%, not 12%.",
        assertions=[
            ClaimAssertion(
                text="Scores rose 12%.",
                kind="quantity",
                verdict="contradicted",
                claim_value="12%",
                source_value="8%",
                quote="scores rose by 8 percent",
            )
        ],
    )
    assert out.assertions[0].verdict == "contradicted"
    assert out.assertions[0].source_value == "8%"


def test_claim_verification_gains_model_status_and_machine_reasons():
    from app.schemas.fulltext import ClaimVerification

    cv = ClaimVerification(
        claim_text="Scores rose 12%.",
        paper_id=uuid4(),
        status="unsupported",
        explanation="A number in the claim is absent from the source.",
    )
    # Additive, defaulted: an old-shape ClaimVerification (no model_status/machine_reasons
    # keys) still validates.
    assert cv.model_status is None
    assert cv.machine_reasons == []

    cv2 = ClaimVerification(
        claim_text="Scores rose 12%.",
        paper_id=uuid4(),
        status="unsupported",
        explanation="A number in the claim is absent from the source.",
        model_status="needs_nuance",
        machine_reasons=["numeric_not_in_source"],
    )
    assert cv2.model_status == "needs_nuance"
    assert cv2.machine_reasons == ["numeric_not_in_source"]


def test_claim_verification_report_gains_contradicted_and_guarded_counts():
    from app.schemas.fulltext import ClaimVerificationReport

    # Old-shape dict: no contradicted_count/guarded_count
    # keys at all. Must still validate (acceptance criterion 13).
    old_shape = {
        "draft_id": uuid4(),
        "verifications": [],
        "verified_count": 0,
        "unsupported_count": 0,
        "nuance_count": 0,
        "abstract_only_count": 0,
    }
    report = ClaimVerificationReport(**old_shape)
    assert report.contradicted_count == 0
    assert report.guarded_count == 0

    report2 = ClaimVerificationReport(**old_shape, contradicted_count=2, guarded_count=3)
    assert report2.contradicted_count == 2
    assert report2.guarded_count == 3


# ---------- v3 schema fields, additive with defaults ----------


def test_claim_assertion_gains_central_default_false():
    """``central`` is additive: an old-shape ``ClaimAssertion`` dict (no ``central`` key)
    still validates, and defaults to False."""
    from app.schemas.fulltext import ClaimAssertion

    old_shape = ClaimAssertion(text="Sample size was 40.", kind="population", verdict="supported")
    assert old_shape.central is False

    a = ClaimAssertion(
        text="Sample size was 40.",
        kind="population",
        verdict="supported",
        central=True,
    )
    assert a.central is True


def test_claim_assertion_gains_entity_and_measure_default_none():
    from app.schemas.fulltext import ClaimAssertion

    old_shape = ClaimAssertion(text="Sample size was 40.", kind="population", verdict="supported")
    assert old_shape.entity is None
    assert old_shape.measure is None

    a = ClaimAssertion(
        text="Sample size was 40.",
        kind="population",
        verdict="supported",
        entity="participants",
        measure="sample size",
    )
    assert a.entity == "participants"
    assert a.measure == "sample size"


def test_claim_assertion_gains_quotes_default_empty_list():
    """``quotes`` holds every composed span; additive, defaults to an empty list."""
    from app.schemas.fulltext import ClaimAssertion

    old_shape = ClaimAssertion(text="Sample size was 40.", kind="population", verdict="supported")
    assert old_shape.quotes == []

    a = ClaimAssertion(
        text="Scores rose from 8% to 16%, doubling.",
        kind="quantity",
        verdict="supported",
        quotes=["scores rose from 8 percent", "to 16 percent"],
    )
    assert a.quotes == ["scores rose from 8 percent", "to 16 percent"]


def test_claim_verification_output_gains_unstated_details_default_empty_list():
    from app.schemas.fulltext import ClaimVerificationOutput

    old_shape = ClaimVerificationOutput(
        claim_text="Old-shape claim.",
        paper_id=uuid4(),
        status="verified",
        explanation="ok",
    )
    assert old_shape.unstated_details == []

    out = ClaimVerificationOutput(
        claim_text="In a small sample, scores rose 12%.",
        paper_id=uuid4(),
        status="needs_nuance",
        explanation="The source does not state the sample was small.",
        unstated_details=["The claim adds 'in a small sample', absent from the source."],
    )
    assert out.unstated_details == [
        "The claim adds 'in a small sample', absent from the source."
    ]


def test_claim_verification_output_gains_evidence_quotes_default_empty_list():
    from app.schemas.fulltext import ClaimVerificationOutput

    old_shape = ClaimVerificationOutput(
        claim_text="Old-shape claim.",
        paper_id=uuid4(),
        status="verified",
        explanation="ok",
    )
    assert old_shape.evidence_quotes == []

    out = ClaimVerificationOutput(
        claim_text="Scores rose from 8% to 16%.",
        paper_id=uuid4(),
        status="verified",
        explanation="Two spans compose to support the claim.",
        evidence_quotes=["scores rose from 8 percent", "to 16 percent"],
    )
    assert out.evidence_quotes == ["scores rose from 8 percent", "to 16 percent"]


def test_claim_verification_gains_diagnostics_default_empty_list():
    """``diagnostics`` is a sibling of ``machine_reasons``, never a member of it, and
    never changes ``status``."""
    from app.schemas.fulltext import ClaimVerification

    old_shape = ClaimVerification(
        claim_text="Scores rose 12%.",
        paper_id=uuid4(),
        status="unsupported",
        explanation="A number in the claim is absent from the source.",
    )
    assert old_shape.diagnostics == []
    assert old_shape.machine_reasons == []

    cv = ClaimVerification(
        claim_text="Scores rose 12%.",
        paper_id=uuid4(),
        status="unsupported",
        explanation="A number in the claim is absent from the source.",
        machine_reasons=["attribution_mismatch"],
        diagnostics=["numeric_not_in_source"],
    )
    assert cv.machine_reasons == ["attribution_mismatch"]
    assert cv.diagnostics == ["numeric_not_in_source"]
    # diagnostics is a distinct field, not folded into machine_reasons.
    assert "numeric_not_in_source" not in cv.machine_reasons
