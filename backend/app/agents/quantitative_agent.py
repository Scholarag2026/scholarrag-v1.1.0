"""Quantitative Analysis Agent — produces analysis plans and interprets results."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.quantitative import AnalysisPlan, AnalysisResults

QUANTITATIVE_PLAN_PROMPT = """\
You are a Quantitative Analysis Agent for academic research in humanities and social sciences.
Given a dataset's column descriptions and research questions, produce a comprehensive analysis plan.

Your plan must include:
- Appropriate statistical methods (e.g. t-test, ANOVA, regression, correlation, chi-square)
- Variable role mappings (independent, dependent, control, covariate)
- Assumptions that must be checked before each method
- Warnings about potential data quality issues

Choose methods that are appropriate for the variable types (numeric vs. categorical).
Justify each method in the context of the research questions.
For regression methods, clearly specify predictor and outcome variables.
For group comparisons, specify the grouping variable and target variable.
"""

QUANTITATIVE_INTERPRETATION_PROMPT = """\
You are a Quantitative Interpretation Agent for academic research in humanities and social sciences.
Given descriptive statistics, an analysis plan, and (optionally) research questions, produce
a full set of analysis results with interpretations.

Your results must include:
- A reference to the provided analysis plan
- Descriptive statistics summary (pass through from input)
- Predictions about likely outcomes based on the data patterns
- Code templates in Python, R, and SPSS for each planned method
- APA-style interpretations of what the results might show
- Assumption check results for each method (met/violated/inconclusive)

Be specific about the data patterns visible in the descriptive statistics.
Write APA report strings that researchers can directly use in their papers.
"""

_plan_agent: Agent | None = None
_interpretation_agent: Agent | None = None


def get_quantitative_plan_agent() -> Agent:
    """Lazily create the quantitative plan agent (singleton)."""
    global _plan_agent
    if _plan_agent is None:
        _plan_agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=AnalysisPlan,
            instructions=QUANTITATIVE_PLAN_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _plan_agent


def get_quantitative_interpretation_agent() -> Agent:
    """Lazily create the quantitative interpretation agent (singleton)."""
    global _interpretation_agent
    if _interpretation_agent is None:
        _interpretation_agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=AnalysisResults,
            instructions=QUANTITATIVE_INTERPRETATION_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _interpretation_agent


def format_plan_prompt(
    columns: list[dict],
    research_questions: list[str],
    methodology: str | None = None,
    project_description: str | None = None,
) -> str:
    """Format the user prompt for the Quantitative Plan Agent."""
    parts = []

    if project_description:
        parts.append(f"Project Description: {project_description}")

    if methodology:
        parts.append(f"Methodology: {methodology}")

    if research_questions:
        rqs = "\n".join(f"- {rq}" for rq in research_questions)
        parts.append(f"Research Questions:\n{rqs}")

    if columns:
        col_lines = "\n".join(
            f"- {col.get('name', 'unknown')} "
            f"(type: {col.get('dtype', 'unknown')}, "
            f"role: {col.get('role', 'unassigned')}, "
            f"missing: {col.get('missing_count', 0)}, "
            f"unique: {col.get('unique_count', 0)})"
            for col in columns
        )
        parts.append(f"Dataset Columns:\n{col_lines}")

    return "\n\n".join(parts)


def format_interpretation_prompt(
    descriptive_stats: dict,
    plan_methods: list[dict],
    research_questions: list[str] | None = None,
) -> str:
    """Format the user prompt for the Quantitative Interpretation Agent."""
    parts = []

    if research_questions:
        rqs = "\n".join(f"- {rq}" for rq in research_questions)
        parts.append(f"Research Questions:\n{rqs}")

    if plan_methods:
        method_lines = "\n".join(
            f"- {m.get('name', 'unknown')}: {m.get('justification', '')}"
            for m in plan_methods
        )
        parts.append(f"Planned Analysis Methods:\n{method_lines}")

    if descriptive_stats:
        numeric = descriptive_stats.get("numeric_stats", [])
        categorical = descriptive_stats.get("categorical_stats", [])
        normality = descriptive_stats.get("normality_tests", [])

        if numeric:
            num_lines = "\n".join(
                f"  {s.get('column', '?')}: mean={s.get('mean', '?')}, "
                f"sd={s.get('std', '?')}, median={s.get('median', '?')}, "
                f"skew={s.get('skewness', '?')}"
                for s in numeric
            )
            parts.append(f"Numeric Statistics:\n{num_lines}")

        if categorical:
            cat_lines = "\n".join(
                f"  {s.get('column', '?')}: mode={s.get('mode', '?')}, "
                f"categories={list(s.get('frequencies', {}).keys())[:5]}"
                for s in categorical
            )
            parts.append(f"Categorical Statistics:\n{cat_lines}")

        if normality:
            norm_lines = "\n".join(
                f"  {s.get('column', '?')}: normal={s.get('is_normal', '?')}, "
                f"p={s.get('p_value', '?')}"
                for s in normality
            )
            parts.append(f"Normality Tests:\n{norm_lines}")

    return "\n\n".join(parts)
