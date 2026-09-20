"""Tests for quantitative analysis Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.quantitative import (
    AnalysisMethod,
    AnalysisPlan,
    AssumptionCheck,
    CodeTemplate,
    AnalysisResults,
    DescriptiveStats,
    NumericColumnStats,
    CategoricalColumnStats,
    NormalityTest,
    Prediction,
    Interpretation,
    VariableMapping,
)


def test_analysis_method():
    m = AnalysisMethod(
        name="Independent t-test",
        justification="Compare means between two groups",
        assumptions=["Normality", "Equal variances"],
        variables=["score", "gender"],
    )
    assert m.name == "Independent t-test"


def test_variable_mapping():
    vm = VariableMapping(column="score", role="dependent", suggested_type="continuous")
    assert vm.role == "dependent"


def test_analysis_plan():
    plan = AnalysisPlan(
        methods=[AnalysisMethod(name="t-test", justification="compare groups", assumptions=[], variables=[])],
        variable_mappings=[VariableMapping(column="age", role="independent", suggested_type="continuous")],
        assumptions=["Data is normally distributed"],
        warnings=["Small sample size (n=10)"],
    )
    assert len(plan.methods) == 1
    assert len(plan.warnings) == 1


def test_numeric_column_stats():
    stats = NumericColumnStats(
        column="age",
        mean=28.4,
        std=3.95,
        median=28.5,
        min=22.0,
        max=35.0,
        skewness=0.12,
        kurtosis=-0.85,
    )
    assert stats.mean == 28.4


def test_categorical_column_stats():
    stats = CategoricalColumnStats(
        column="gender",
        frequencies={"Male": 5, "Female": 5},
        percentages={"Male": 50.0, "Female": 50.0},
        mode="Male",
    )
    assert stats.mode == "Male"


def test_normality_test():
    nt = NormalityTest(column="age", statistic=0.95, p_value=0.67, is_normal=True)
    assert nt.is_normal is True


def test_descriptive_stats():
    ds = DescriptiveStats(
        numeric_stats=[],
        categorical_stats=[],
        correlation_matrix={},
        normality_tests=[],
    )
    assert isinstance(ds.correlation_matrix, dict)


def test_code_template():
    ct = CodeTemplate(
        method="t-test",
        python="from scipy.stats import ttest_ind\n...",
        r="t.test(group1, group2)",
        spss="T-TEST GROUPS=gender(1 2) /VARIABLES=score.",
        explanation="Compares means between two independent groups.",
    )
    assert "scipy" in ct.python


def test_prediction():
    p = Prediction(
        description="Based on r=0.45, expect significant positive relationship.",
        confidence="medium",
        basis="Correlation between X and Y",
    )
    assert p.confidence == "medium"


def test_interpretation():
    interp = Interpretation(
        finding="Significant difference found between groups",
        explanation="The t-test revealed a significant difference (p < 0.05)",
        apa_report="t(18) = 2.45, p = .025, d = 0.89",
    )
    assert "significant" in interp.finding.lower()


def test_assumption_check():
    ac = AssumptionCheck(
        assumption="Normality",
        result="met",
        details="Shapiro-Wilk W=0.95, p=0.67",
        recommendation="Proceed with parametric test",
    )
    assert ac.result == "met"


def test_analysis_results():
    ar = AnalysisResults(
        plan=AnalysisPlan(methods=[], variable_mappings=[], assumptions=[], warnings=[]),
        descriptive_stats=DescriptiveStats(numeric_stats=[], categorical_stats=[], correlation_matrix={}, normality_tests=[]),
        predictions=[],
        code_templates=[],
        interpretations=[],
        assumption_checks=[],
    )
    assert ar.plan is not None
