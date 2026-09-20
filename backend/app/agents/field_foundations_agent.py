"""Field Foundations Agent — identifies foundational works for a research field."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.field_foundations import FoundationalWorksList

FOUNDATIONS_PROMPT = """\
You are an expert academic librarian and research methodologist specializing in \
humanities and social sciences.

Given a research topic (and optionally research questions), identify 20-30 \
foundational works that any researcher in this field MUST know. These are the \
seminal papers, books, and landmark studies that define the field.

For each work, provide:
- suggested_title: The exact title of the work as commonly cited
- suggested_authors: A list of author names (e.g., ["Vygotsky, L. S."])
- suggested_year: The publication year
- why_essential: A 1-2 sentence explanation of why this work is foundational

IMPORTANT:
- Focus on genuinely influential, highly-cited works — not obscure papers.
- Include a mix of: theoretical frameworks, seminal empirical studies, \
  landmark reviews, and methodological contributions.
- Cover the historical development of the field (early foundations through \
  recent influential works).
- Do NOT fabricate works. Only suggest works you are confident actually exist.
- If research questions are provided, weight suggestions toward works relevant \
  to those specific questions while still covering the broader field.
- If existing paper titles are listed, avoid duplicating them — suggest \
  works NOT already in the researcher's library.

Output valid JSON matching the FoundationalWorksList schema."""

_agent: Agent[AnalysisDependencies, FoundationalWorksList] | None = None


def get_field_foundations_agent() -> Agent[AnalysisDependencies, FoundationalWorksList]:
    """Lazily create the field foundations agent (avoids requiring API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=AnalysisDependencies,
            output_type=FoundationalWorksList,
            instructions=FOUNDATIONS_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_foundations_prompt(
    topic: str,
    research_questions: list[str] | None = None,
    existing_titles: list[str] | None = None,
) -> str:
    """Format the user prompt for the field foundations agent."""
    parts = [f"Research field/topic: {topic}"]

    if research_questions:
        parts.append("")
        parts.append("Research questions:")
        for i, rq in enumerate(research_questions, 1):
            parts.append(f"  {i}. {rq}")

    if existing_titles:
        parts.append("")
        parts.append("Papers already in the researcher's library (do NOT duplicate these):")
        for i, title in enumerate(existing_titles[:50], 1):
            parts.append(f"  {i}. {title}")

    parts.append("")
    parts.append(
        "Please identify 20-30 foundational works for this field. "
        "Include seminal papers, landmark studies, and key theoretical contributions."
    )

    return "\n".join(parts)
