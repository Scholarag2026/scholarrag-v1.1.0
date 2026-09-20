"""Query expansion agent — generates synonym/related search queries."""

from __future__ import annotations

from pydantic_ai import Agent

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.deep_search import QueryExpansionResult

EXPANSION_PROMPT = """\
You are an expert academic search strategist for humanities and social sciences.

Given a research query and titles/abstracts of papers found so far, generate 3-5 alternative search
queries that would find DIFFERENT relevant papers. Strategies:
- Use synonyms and related terminology
- Broaden or narrow the scope
- Use different theoretical frameworks or methodological terms
- Target specific sub-topics identified from the existing results

HARD CONSTRAINTS — these queries are executed against the OpenAlex `search` parameter, which
rejects the following with an HTTP 400 error. Violating any of them makes the query return
zero results:
- NEVER use wildcard or truncation characters. Do not emit `*` or `?` anywhere. Write out the
  full word forms joined with OR instead: use (comparison OR comparative OR comparing),
  NOT `compar*`; use (process OR processes OR processing), NOT `process*`.
- NEVER use proximity operators (NEAR, NEAR/n, W/n, ADJ, PRE/n).
- NEVER use field tags or prefixes (TS=, TI=, AB=, AU=, SO=, ti:, abs:).
- NEVER use regular-expression or fuzzy syntax (`~`, `^`, `/.../`).
- Keep each query under 300 characters.
- OpenAlex already stems words, so plural and inflected forms are matched automatically —
  you never need truncation to catch them.

Output valid JSON with a "queries" list of 3-5 search strings."""

_agent: Agent[AnalysisDependencies, QueryExpansionResult] | None = None


def get_query_expansion_agent() -> Agent[AnalysisDependencies, QueryExpansionResult]:
    """Lazily create the query expansion agent (avoids requiring API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=AnalysisDependencies,
            output_type=QueryExpansionResult,
            instructions=EXPANSION_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def format_expansion_prompt(
    original_query: str,
    paper_titles: list[str],
    paper_abstracts: list[str | None] | None = None,
) -> str:
    """Format the user prompt for query expansion with existing paper context."""
    parts = [f"Original query: {original_query}", "", "Papers found so far:"]
    for i, title in enumerate(paper_titles[:10]):
        abstract = paper_abstracts[i] if paper_abstracts and i < len(paper_abstracts) else None
        parts.append(f"  {i + 1}. {title}")
        if abstract:
            parts.append(f"     Abstract: {abstract[:200]}...")
    return "\n".join(parts)
