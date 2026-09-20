"""Qualitative Coding Agent — produces structured codebooks and coded segments."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.qualitative import CodingResult

QUALITATIVE_CODING_PROMPT = """\
You are a Qualitative Coding Agent for academic research in humanities and social sciences.
Given text segments, research questions, and optionally an existing codebook and methodology,
produce a comprehensive qualitative coding analysis.

Follow these standards:
- Use thematic analysis or grounded theory as appropriate
- Create a hierarchical codebook with themes and codes
- Assign unique IDs (e.g., t1, t2 for themes; c1, c2 for codes)
- Assign descriptive colors to each code (hex format, e.g., #FF6B6B)
- Code each segment with one or more codes
- Provide confidence levels (high, medium, low) for each coding decision
- Explain reasoning for each coding decision
- Flag uncertainties where coding is ambiguous
- Add annotation notes for the researcher

If an existing codebook is provided, use it as a starting point and extend it as needed.
If a methodology is specified (e.g., grounded theory, thematic analysis), follow that approach.
"""

_agent = None


def get_qualitative_coding_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=CodingResult,
            instructions=QUALITATIVE_CODING_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_coding_prompt(
    text_segments: list[str],
    research_questions: list[str],
    existing_codebook: dict | None = None,
    methodology: str | None = None,
) -> str:
    """Format the user prompt for the Qualitative Coding Agent."""
    parts = ["Text Segments to Code:"]
    for i, segment in enumerate(text_segments, 1):
        parts.append(f"  Segment {i}: {segment}")

    parts.append("")
    parts.append("Research Questions:")
    for rq in research_questions:
        parts.append(f"  - {rq}")

    if existing_codebook:
        parts.append("")
        parts.append(f"Existing Codebook: {existing_codebook}")

    if methodology:
        parts.append("")
        parts.append(f"Methodology: {methodology}")

    return "\n".join(parts)
