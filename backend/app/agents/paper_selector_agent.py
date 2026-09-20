"""Paper Selector Agent — selects the 15-20 most relevant papers for a given article section."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings

PAPER_SELECTOR_PROMPT = """\
You are an academic paper selector agent for humanities and social sciences research articles.

Given a section type and a list of paper summaries (id, title, year, themes), select the 15-20
most relevant papers for that section.

Different sections require different papers:
- introduction: papers about research context, gaps, and significance — papers that establish
  why the topic matters, what is unknown, and what the study contributes
- literature_review: papers covering the main themes, findings, and theories — broad coverage
  of the field, grouped by concept or debate
- discussion: papers whose findings can be compared or contrasted with the user's own results —
  papers with concrete findings that align, diverge, or extend the user's conclusions
- implications: a mix of key theoretical and practical papers — papers that support the
  significance of the findings for theory, practice, and pedagogy
- conclusion: a mix of key papers that anchor the overall contribution — seminal works and
  papers most central to the research questions
- abstract: a mix of the most important papers spanning the article's core argument
- chat: papers most relevant to answering the user's specific question — select papers whose
  titles, themes, or findings are directly related to the question being asked

A foundational or seminal work may appear in multiple sections.
Across ALL sections of an article, every paper in the pool should be selected at least once.

Select exactly 15-20 paper IDs. Prefer breadth within the section's needs.
Output valid JSON with:
- "selected_paper_ids": list of selected paper ID strings (15-20 items)
- "reasoning": brief explanation of the selection strategy for this section
"""

_agent: Agent[None, PaperSelection] | None = None


class PaperSelection(BaseModel):
    selected_paper_ids: list[str] = Field(
        description="IDs of the 15-20 most relevant papers for this section"
    )
    reasoning: str = Field(description="Brief explanation of selection strategy")


def get_paper_selector_agent() -> Agent[None, PaperSelection]:
    """Lazily create the paper selector agent (avoids requiring API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=PaperSelection,
            instructions=PAPER_SELECTOR_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def _build_selector_prompt(
    section_type: str, paper_summaries: list[dict], question: str | None = None
) -> str:
    """Assemble the user-turn prompt listing all papers with their metadata."""
    parts: list[str] = [
        f"Section type: {section_type}",
        f"Total papers available: {len(paper_summaries)}",
        "",
        "Papers to select from:",
    ]
    for paper in paper_summaries:
        paper_id = paper.get("id", "?")
        title = paper.get("title", "(no title)")
        year = paper.get("year", "n.d.")
        themes = paper.get("themes", [])
        themes_str = ", ".join(themes) if isinstance(themes, list) else str(themes)
        parts.append(f"- ID: {paper_id} | Year: {year} | Title: {title} | Themes: {themes_str}")
    parts.append("")
    parts.append(
        f"Select 15-20 paper IDs that are most relevant for writing the {section_type} section."
    )
    prompt = "\n".join(parts)
    if question:
        prompt += f"\n\nThe user is asking: {question}\nSelect papers most relevant to answering this specific question."
    return prompt


async def select_papers_for_section(
    section_type: str,
    paper_summaries: list[dict],
    question: str | None = None,
) -> list[str]:
    """Select 15-20 most relevant papers for a given section type.

    Args:
        section_type: one of "introduction", "literature_review", "discussion",
                      "implications", "conclusion", "abstract", "chat"
        paper_summaries: list of {id, title, year, themes} dicts
        question: optional user question (used for "chat" section type)

    Returns:
        List of paper ID strings
    """
    if not paper_summaries:
        return []

    # If 20 or fewer papers, return all — no selection needed
    if len(paper_summaries) <= 20:
        return [p["id"] for p in paper_summaries if p.get("id")]

    agent = get_paper_selector_agent()
    prompt = _build_selector_prompt(section_type, paper_summaries, question=question)
    result = await agent.run(prompt)
    return result.output.selected_paper_ids
