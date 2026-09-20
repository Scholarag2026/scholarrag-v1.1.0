"""Quality Scoring Agent — analyzes papers and produces quality scores."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, LONG_MODEL_SETTINGS
from app.config import settings
from app.schemas.analysis import BatchAnalysisResult


@dataclass
class AnalysisDependencies:
    """Dependencies injected into analysis agents. No DB — agents are pure transformers."""

    project_id: UUID | str
    project_description: str | None
    target_journal: str | None
    citation_style: str


QUALITY_SCORING_PROMPT = """\
You are a Literature Analysis Agent for academic research in humanities and social sciences.
Analyze each paper based on its abstract, metadata, and available information.

For each paper, assess:
1. Methodology rigor — Is the research design sound?
   Are methods appropriate for the research question?
2. Sample adequacy — Is the sample size sufficient? Is sampling strategy appropriate?
3. Key findings — What are the 3-5 most important findings?
4. Limitations — What limitations are acknowledged or apparent?
5. Theoretical framework — What theories or frameworks does this paper use?
6. Overall quality — Score 0-1 based on:
   - Methodology rigor (40%): high=0.9, medium=0.6, low=0.3
   - Sample adequacy (20%): adequate=1.0, limited=0.5, unclear=0.3
   - Journal quality (20%): WoS indexed=1.0, not indexed=0.4
   - Citation impact (10%): normalize by field average
   - Recency (10%): last 5 years=1.0, 5-10 years=0.7, >10 years=0.4

IMPORTANT:
- Return the paper_id EXACTLY as provided in the input for each paper.
- Base assessments only on available information.
- Do not hallucinate details not present in the abstract.
- If information is insufficient for a dimension, note it in limitations and score conservatively.
"""

_analysis_agent: Agent[AnalysisDependencies, BatchAnalysisResult] | None = None


def get_analysis_agent() -> Agent[AnalysisDependencies, BatchAnalysisResult]:
    """Lazily create the analysis agent (avoids requiring API key at import time)."""
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = Agent(
            f"deepseek:{settings.deepseek_reasoner_model}",
            deps_type=AnalysisDependencies,
            output_type=BatchAnalysisResult,
            instructions=QUALITY_SCORING_PROMPT,
            model_settings=LONG_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _analysis_agent


def format_papers_for_prompt(papers: list[dict]) -> str:
    """Format a batch of papers as text for the agent prompt."""
    parts = []
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(a.get("name", "Unknown") for a in paper.get("authors", []))
        if not authors_str:
            authors_str = "Unknown"

        wos = "yes" if paper.get("is_wos_indexed") else "no"
        abstract = paper.get("abstract") or "Not available"
        year = paper.get("year") or "Unknown"
        journal = paper.get("journal_name") or "Unknown"
        citations = paper.get("citation_count")
        citations_str = str(citations) if citations is not None else "Unknown"

        parts.append(
            f"Paper {i}:\n"
            f"- ID: {paper['id']}\n"
            f"- Title: {paper['title']}\n"
            f"- Authors: {authors_str}\n"
            f"- Year: {year}\n"
            f"- Journal: {journal} (WoS indexed: {wos})\n"
            f"- Citation count: {citations_str}\n"
            f"- Abstract: {abstract}\n"
        )
    return "\n".join(parts)
