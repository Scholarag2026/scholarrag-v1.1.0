"""Tests for research design and data collection schemas."""

import pytest
from pydantic import ValidationError


def test_methodology_plan_valid():
    from app.schemas.research_design import MethodologyPlan
    mp = MethodologyPlan(
        approach="mixed_methods",
        design_type="explanatory sequential",
        justification="Combines breadth of surveys with depth of interviews",
        research_questions=["RQ1: How does X affect Y?"],
        theoretical_framework="Social Cognitive Theory",
    )
    assert mp.approach == "mixed_methods"


def test_methodology_plan_invalid_approach():
    from app.schemas.research_design import MethodologyPlan
    with pytest.raises(ValidationError):
        MethodologyPlan(
            approach="experimental",
            design_type="RCT",
            justification="j",
            research_questions=["RQ1"],
            theoretical_framework="t",
        )


def test_ethics_plan():
    from app.schemas.research_design import EthicsPlan
    ep = EthicsPlan(
        considerations=["Informed consent", "Anonymity"],
        consent_template="I consent to participate...",
        data_protection="All data encrypted at rest",
        irb_notes="Submit to university IRB",
    )
    assert len(ep.considerations) == 2


def test_instrument():
    from app.schemas.research_design import Instrument
    inst = Instrument(
        name="Student Perception Survey",
        type="survey",
        description="5-point Likert scale",
        sample_questions=["Q1: Rate your experience", "Q2: How often do you..."],
        administration="Online via Qualtrics",
    )
    assert inst.type == "survey"


def test_instrument_invalid_type():
    from app.schemas.research_design import Instrument
    with pytest.raises(ValidationError):
        Instrument(
            name="Test", type="experiment",
            description="d", sample_questions=["q"], administration="a",
        )


def test_sampling_strategy():
    from app.schemas.research_design import SamplingStrategy
    ss = SamplingStrategy(
        strategy_type="purposive",
        target_population="University students aged 18-25",
        sample_size=50,
        justification="Allows for theoretical saturation",
        recruitment_plan="Email invitations via department",
        inclusion_criteria=["Enrolled full-time", "Age 18+"],
        exclusion_criteria=["Exchange students"],
    )
    assert ss.sample_size == 50


def test_validity_plan():
    from app.schemas.research_design import ValidityPlan
    vp = ValidityPlan(
        internal_validity=["Control for confounding variables"],
        external_validity=["Multi-site sampling"],
        reliability_measures=["Inter-rater reliability > 0.8"],
        triangulation="Data triangulation: surveys + interviews + documents",
    )
    assert "triangulation" in vp.triangulation.lower()


def test_research_design_complete():
    from app.schemas.research_design import (
        EthicsPlan,
        Instrument,
        MethodologyPlan,
        ResearchDesign,
        SamplingStrategy,
        ValidityPlan,
    )
    rd = ResearchDesign(
        methodology=MethodologyPlan(
            approach="qualitative", design_type="case study",
            justification="j", research_questions=["RQ1"],
            theoretical_framework="t",
        ),
        ethics=EthicsPlan(
            considerations=["c"], consent_template="ct",
            data_protection="dp", irb_notes="irb",
        ),
        instruments=[Instrument(
            name="n", type="interview", description="d",
            sample_questions=["q"], administration="a",
        )],
        sampling=SamplingStrategy(
            strategy_type="purposive", target_population="tp",
            sample_size=30, justification="j", recruitment_plan="rp",
            inclusion_criteria=["ic"], exclusion_criteria=["ec"],
        ),
        validity=ValidityPlan(
            internal_validity=["iv"], external_validity=["ev"],
            reliability_measures=["rm"], triangulation="t",
        ),
    )
    assert rd.methodology.approach == "qualitative"
    assert len(rd.instruments) == 1


def test_collection_phase():
    from app.schemas.research_design import CollectionPhase
    cp = CollectionPhase(
        phase_number=1, name="Pilot Testing",
        description="Test instruments with small group",
        steps=["Recruit 5 participants", "Administer survey", "Debrief"],
        duration="1 week", deliverables=["Pilot report"],
    )
    assert cp.phase_number == 1


def test_quality_check():
    from app.schemas.research_design import QualityCheck
    qc = QualityCheck(
        check_name="Response completeness",
        when="After each batch of 10 responses",
        how="Check for >80% completion rate",
        action_if_failed="Follow up with incomplete respondents",
    )
    assert qc.check_name == "Response completeness"


def test_data_storage_plan():
    from app.schemas.research_design import DataStoragePlan
    dsp = DataStoragePlan(
        storage_method="University secure server",
        backup_strategy="Daily automated backups",
        access_control="Password-protected, PI access only",
        anonymization="Remove names, assign participant codes",
        retention_period="5 years after publication",
    )
    assert "5 years" in dsp.retention_period


def test_timeline_item():
    from app.schemas.research_design import TimelineItem
    ti = TimelineItem(week="Week 1-2", activity="Pilot testing", milestone="Pilot complete")
    assert ti.milestone == "Pilot complete"

    ti2 = TimelineItem(week="Week 3", activity="Main data collection")
    assert ti2.milestone is None


def test_data_collection_plan_complete():
    from app.schemas.research_design import (
        CollectionPhase,
        DataCollectionPlan,
        DataStoragePlan,
        QualityCheck,
        TimelineItem,
    )
    dcp = DataCollectionPlan(
        phases=[CollectionPhase(
            phase_number=1, name="p", description="d",
            steps=["s1"], duration="1w", deliverables=["d1"],
        )],
        quality_checks=[QualityCheck(
            check_name="c", when="w", how="h", action_if_failed="a",
        )],
        data_storage=DataStoragePlan(
            storage_method="s", backup_strategy="b",
            access_control="a", anonymization="an", retention_period="r",
        ),
        timeline=[TimelineItem(week="W1", activity="a")],
        ethical_reminders=["Obtain consent before each session"],
    )
    assert len(dcp.phases) == 1
    assert len(dcp.ethical_reminders) == 1


def test_research_design_request():
    from app.schemas.research_design import ResearchDesignRequest
    req = ResearchDesignRequest(research_question="How does X affect Y?")
    assert "How does" in req.research_question
