"""Research Design Agent — produces structured research design plans."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.research_design import ResearchDesign

RESEARCH_DESIGN_PROMPT = """\
You are a Research Design Agent for academic research in humanities and social sciences.
Given a research question and project context, produce a comprehensive research design plan.

Follow these standards (from the Paper Checklist):
- Minimum 30 participants for quantitative studies
- Multi-site sampling if possible
- Triangulation for validity
- Proper methodology justification
- Ethical considerations including consent forms
- Validity and reliability planning

For the methodology, choose the most appropriate approach
(qualitative, quantitative, or mixed_methods) based on the research question.
Justify your choice.

For instruments, provide 5-10 sample questions that are specific to the research context.

For the consent template, provide a complete draft that covers: purpose, procedures, risks, \
benefits, confidentiality, voluntary participation, and contact information.

For sampling, ensure sample_size is at least 30 for quantitative components.
"""

_agent = None


def get_research_design_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=ResearchDesign,
            instructions=RESEARCH_DESIGN_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_research_design_prompt(
    research_question: str,
    project_description: str | None = None,
    target_journal: str | None = None,
    gap_summary: str | None = None,
) -> str:
    """Format the user prompt for the Research Design Agent."""
    parts = [f"Research Question: {research_question}"]
    if project_description:
        parts.append(f"Project Description: {project_description}")
    if target_journal:
        parts.append(f"Target Journal: {target_journal}")
    if gap_summary:
        parts.append(f"Gap Analysis Summary: {gap_summary}")
    return "\n\n".join(parts)
