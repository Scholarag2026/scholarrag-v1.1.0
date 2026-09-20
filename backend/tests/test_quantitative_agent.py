"""Tests for Quantitative Analysis agents."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_quantitative_plan_agent_exists():
    from app.agents.quantitative_agent import get_quantitative_plan_agent

    agent = get_quantitative_plan_agent()
    assert agent is not None


def test_quantitative_plan_agent_singleton():
    from app.agents.quantitative_agent import get_quantitative_plan_agent

    agent1 = get_quantitative_plan_agent()
    agent2 = get_quantitative_plan_agent()
    assert agent1 is agent2


def test_quantitative_interpretation_agent_exists():
    from app.agents.quantitative_agent import get_quantitative_interpretation_agent

    agent = get_quantitative_interpretation_agent()
    assert agent is not None


def test_format_plan_prompt():
    from app.agents.quantitative_agent import format_plan_prompt

    columns = [
        {"name": "age", "dtype": "numeric", "role": "independent", "missing_count": 0, "unique_count": 45},
        {"name": "score", "dtype": "numeric", "role": "dependent", "missing_count": 2, "unique_count": 30},
        {"name": "group", "dtype": "categorical", "role": "control", "missing_count": 0, "unique_count": 3},
    ]
    rqs = ["Does age predict score?", "Do groups differ in score?"]
    result = format_plan_prompt(
        columns=columns,
        research_questions=rqs,
        methodology="quantitative",
        project_description="Education study",
    )
    assert "age" in result
    assert "score" in result
    assert "Does age predict score?" in result
    assert "Education study" in result
    assert "quantitative" in result


def test_format_interpretation_prompt():
    from app.agents.quantitative_agent import format_interpretation_prompt

    descriptive_stats = {
        "numeric_stats": [
            {
                "column": "score",
                "mean": 75.4,
                "std": 12.1,
                "median": 76.0,
                "min": 40.0,
                "max": 100.0,
                "skewness": -0.3,
                "kurtosis": 0.1,
            }
        ],
        "categorical_stats": [
            {
                "column": "group",
                "frequencies": {"A": 30, "B": 25, "C": 20},
                "percentages": {"A": 40.0, "B": 33.3, "C": 26.7},
                "mode": "A",
            }
        ],
        "correlation_matrix": {},
        "normality_tests": [
            {"column": "score", "statistic": 0.98, "p_value": 0.42, "is_normal": True}
        ],
    }
    plan_methods = [
        {
            "name": "One-way ANOVA",
            "justification": "Compare score across groups",
            "assumptions": ["normality", "homogeneity of variance"],
            "variables": ["group", "score"],
        }
    ]
    rqs = ["Do groups differ in score?"]
    result = format_interpretation_prompt(
        descriptive_stats=descriptive_stats,
        plan_methods=plan_methods,
        research_questions=rqs,
    )
    assert "score" in result
    assert "ANOVA" in result
    assert "Do groups differ" in result
    assert "75.4" in result
