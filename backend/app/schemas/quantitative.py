"""Schemas for quantitative analysis — analysis plans, descriptive stats, and results."""

from __future__ import annotations

from pydantic import BaseModel


class AnalysisMethod(BaseModel):
    name: str
    justification: str
    assumptions: list[str]
    variables: list[str]


class VariableMapping(BaseModel):
    column: str
    role: str  # independent, dependent, control, covariate
    suggested_type: str  # continuous, categorical, ordinal


class AnalysisPlan(BaseModel):
    methods: list[AnalysisMethod]
    variable_mappings: list[VariableMapping]
    assumptions: list[str]
    warnings: list[str]


class NumericColumnStats(BaseModel):
    column: str
    mean: float
    std: float
    median: float
    min: float
    max: float
    skewness: float
    kurtosis: float


class CategoricalColumnStats(BaseModel):
    column: str
    frequencies: dict[str, int]
    percentages: dict[str, float]
    mode: str


class NormalityTest(BaseModel):
    column: str
    statistic: float
    p_value: float
    is_normal: bool


class DescriptiveStats(BaseModel):
    numeric_stats: list[NumericColumnStats]
    categorical_stats: list[CategoricalColumnStats]
    correlation_matrix: dict[str, dict[str, float]]
    normality_tests: list[NormalityTest]


class CodeTemplate(BaseModel):
    method: str
    python: str
    r: str
    spss: str
    explanation: str


class Prediction(BaseModel):
    description: str
    confidence: str  # high, medium, low
    basis: str


class Interpretation(BaseModel):
    finding: str
    explanation: str
    apa_report: str  # How to report this in APA format


class AssumptionCheck(BaseModel):
    assumption: str
    result: str  # met, violated, inconclusive
    details: str
    recommendation: str


class AnalysisResults(BaseModel):
    plan: AnalysisPlan
    descriptive_stats: DescriptiveStats
    predictions: list[Prediction]
    code_templates: list[CodeTemplate]
    interpretations: list[Interpretation]
    assumption_checks: list[AssumptionCheck]
