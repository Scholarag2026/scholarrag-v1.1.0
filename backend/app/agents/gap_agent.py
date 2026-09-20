"""Gap Analysis Agent — identifies research gaps and controversies."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.analysis import GapReport

GAP_ANALYSIS_PROMPT = """\
You are a Research Gap Analysis Agent for academic research in humanities and social sciences.
Given a collection of paper analyses from a research project, identify:

1. Research gaps — Areas that are unexplored, under-explored,
   controversial, or have methodological weaknesses.
   For each gap, classify as: unexplored, under_explored, controversial, or methodological.
   Rate severity as: high, medium, or low.
   Provide evidence from the analyzed papers.

2. Controversies — Topics where studies disagree or present conflicting findings.
   For each controversy, list the different positions and which papers support them.

3. Suggested research questions — Actionable research questions that address the identified gaps.
   For each question, explain why it matters, suggest a methodology,
   and reference which gaps it addresses.

4. Theoretical landscape — List all theories and frameworks used across the literature.

5. Summary — Write a 2-3 paragraph synthesis of the state of the field.

Be specific and evidence-based. Reference paper titles and findings in your analysis.
"""

_gap_agent: Agent[AnalysisDependencies, GapReport] | None = None


def get_gap_agent() -> Agent[AnalysisDependencies, GapReport]:
    """Lazily create the gap analysis agent (avoids requiring API key at import time)."""
    global _gap_agent
    if _gap_agent is None:
        _gap_agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=GapReport,
            instructions=GAP_ANALYSIS_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _gap_agent


def format_analyses_for_gap_prompt(
    analyses: list[dict],
    project_description: str | None = None,
    target_journal: str | None = None,
) -> str:
    """Format all paper analyses as text for the gap analysis agent."""
    parts = []

    if project_description:
        parts.append(f"Project Description: {project_description}\n")
    if target_journal:
        parts.append(f"Target Journal: {target_journal}\n")

    parts.append(f"Total papers analyzed: {len(analyses)}\n")
    parts.append("---\n")

    for i, a in enumerate(analyses, 1):
        findings = "; ".join(a.get("key_findings", []))
        limitations = "; ".join(a.get("limitations", []))
        theories = ", ".join(a.get("theories_used", []))

        parts.append(
            f"Paper {i}: {a.get('paper_title', 'Untitled')} ({a.get('paper_year', 'N/A')})\n"
            f"- Quality Score: {a.get('quality_score', 'N/A')}\n"
            f"- Methodology: {a.get('methodology', 'N/A')} "
            f"(Rigor: {a.get('methodology_rigor', 'N/A')})\n"
            f"- Key Findings: {findings or 'None'}\n"
            f"- Limitations: {limitations or 'None'}\n"
            f"- Theories: {theories or 'None'}\n"
        )
    return "\n".join(parts)
