"""Tests for descriptive statistics computation service."""

import os
import pytest
import pandas as pd

from app.services.descriptive_stats import (
    compute_numeric_stats,
    compute_categorical_stats,
    compute_correlation_matrix,
    compute_normality_tests,
    compute_all_descriptive_stats,
)


@pytest.fixture
def sample_df():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "fixtures", "sample_data.csv"))


def test_compute_numeric_stats(sample_df):
    stats = compute_numeric_stats(sample_df)
    names = {s["column"] for s in stats}
    assert "age" in names
    assert "score" in names
    age_stat = next(s for s in stats if s["column"] == "age")
    assert 20 < age_stat["mean"] < 40
    assert age_stat["std"] > 0


def test_compute_categorical_stats(sample_df):
    stats = compute_categorical_stats(sample_df)
    names = {s["column"] for s in stats}
    assert "gender" in names
    gender = next(s for s in stats if s["column"] == "gender")
    assert "Male" in gender["frequencies"]
    assert "Female" in gender["frequencies"]
    assert abs(sum(gender["percentages"].values()) - 100.0) < 0.1


def test_compute_correlation_matrix(sample_df):
    matrix = compute_correlation_matrix(sample_df)
    assert "age" in matrix
    assert "score" in matrix["age"]
    assert matrix["age"]["age"] == pytest.approx(1.0)


def test_compute_normality_tests(sample_df):
    tests = compute_normality_tests(sample_df)
    names = {t["column"] for t in tests}
    assert "age" in names
    for t in tests:
        assert 0 <= t["p_value"] <= 1


def test_compute_all(sample_df):
    result = compute_all_descriptive_stats(sample_df)
    assert "numeric_stats" in result
    assert "categorical_stats" in result
    assert "correlation_matrix" in result
    assert "normality_tests" in result


# ---------- JSON-safety of descriptive stats (T5 / issue #12) ----------


def test_normality_is_normal_is_plain_bool(sample_df):
    """`is_normal` must be a Python bool, not numpy.bool_ (not JSON serializable)."""
    tests = compute_normality_tests(sample_df)
    assert tests, "fixture should have at least one numeric column"
    for t in tests:
        assert type(t["is_normal"]) is bool


def test_all_descriptive_stats_are_json_serializable(sample_df):
    import json

    stats = compute_all_descriptive_stats(sample_df)
    json.dumps(stats)  # must not raise TypeError


def test_column_names_are_strings():
    import pandas as pd

    df = pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0], 1: ["a", "b", "a", "b"]})
    stats = compute_all_descriptive_stats(df)
    assert all(isinstance(s["column"], str) for s in stats["numeric_stats"])
    assert all(isinstance(s["column"], str) for s in stats["categorical_stats"])
    assert all(isinstance(s["column"], str) for s in stats["normality_tests"])


def test_descriptive_stats_validate_against_schema(sample_df):
    """The quantitative persist path validates through DescriptiveStats — it must fit."""
    from app.schemas.quantitative import DescriptiveStats

    stats = compute_all_descriptive_stats(sample_df)
    validated = DescriptiveStats.model_validate(stats).model_dump(mode="json")
    assert set(validated) == {
        "numeric_stats",
        "categorical_stats",
        "correlation_matrix",
        "normality_tests",
    }
