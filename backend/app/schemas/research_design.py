"""Schemas for research design and data collection planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# --- Research Design Agent output ---

class MethodologyPlan(BaseModel):
    approach: Literal["qualitative", "quantitative", "mixed_methods"]
    design_type: str
    justification: str
    research_questions: list[str]
    theoretical_framework: str


class EthicsPlan(BaseModel):
    considerations: list[str]
    consent_template: str
    data_protection: str
    irb_notes: str


class Instrument(BaseModel):
    name: str
    type: Literal["survey", "interview", "observation", "focus_group", "document_analysis"]
    description: str
    sample_questions: list[str]
    administration: str


class SamplingStrategy(BaseModel):
    strategy_type: str
    target_population: str
    sample_size: int
    justification: str
    recruitment_plan: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]


class ValidityPlan(BaseModel):
    internal_validity: list[str]
    external_validity: list[str]
    reliability_measures: list[str]
    triangulation: str


class ResearchDesign(BaseModel):
    methodology: MethodologyPlan
    ethics: EthicsPlan
    instruments: list[Instrument]
    sampling: SamplingStrategy
    validity: ValidityPlan


# --- Data Collection Agent output ---

class CollectionPhase(BaseModel):
    phase_number: int
    name: str
    description: str
    steps: list[str]
    duration: str
    deliverables: list[str]


class QualityCheck(BaseModel):
    check_name: str
    when: str
    how: str
    action_if_failed: str


class DataStoragePlan(BaseModel):
    storage_method: str
    backup_strategy: str
    access_control: str
    anonymization: str
    retention_period: str


class TimelineItem(BaseModel):
    week: str
    activity: str
    milestone: str | None = None


class DataCollectionPlan(BaseModel):
    phases: list[CollectionPhase]
    quality_checks: list[QualityCheck]
    data_storage: DataStoragePlan
    timeline: list[TimelineItem]
    ethical_reminders: list[str]


# --- API request schemas ---

class ResearchDesignRequest(BaseModel):
    research_question: str
