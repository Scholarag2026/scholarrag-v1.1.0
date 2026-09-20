"""Query generator agent — converts user research topics into academic keyword queries."""

from __future__ import annotations

import re
from typing import NamedTuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.provenance import LLMCallProvenance, provenance_from_run

QUERY_GENERATOR_PROMPT = """\
You are an expert academic search strategist specialising in humanities and social sciences.

Given a research topic or question, generate 3-7 optimised academic keyword queries suitable
for searching databases such as Web of Science, Scopus, and Google Scholar.

Guidelines:
- Use Boolean operators AND / OR to combine terms
- Use quoted phrases for exact multi-word concepts (e.g. "social capital")
- Include synonyms and alternative terminology for key concepts
- Cover both broader and narrower formulations of the topic
- Include sub-topics, related theoretical frameworks, and methodological terms where relevant
- Vary query specificity so the set collectively casts a wide but targeted net

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

Output valid JSON with:
- "queries": a list of 3-7 search strings
- "reasoning": a one- or two-sentence explanation of the strategy you applied"""

_agent: Agent[None, GeneratedQueries] | None = None

_WILDCARD_RE = re.compile(r"[*?]")
_MULTISPACE_RE = re.compile(r"\s{2,}")


class GeneratedQueries(BaseModel):
    """Structured output from the query generator agent."""

    queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=7,
        description="3-7 academic keyword query strings using Boolean operators and quoted phrases.",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the query generation strategy.",
    )


def get_query_generator_agent() -> Agent[None, GeneratedQueries]:
    """Lazily create the query generator agent (avoids requiring API key at import time)."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=GeneratedQueries,
            instructions=QUERY_GENERATOR_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


def _build_prompt(
    user_query: str,
    project_description: str | None,
    existing_papers: list[str] | None,
) -> str:
    """Assemble the user-turn prompt from the query and optional context."""
    parts: list[str] = [f"Research topic / question: {user_query}"]

    if project_description:
        parts.append(f"\nProject description: {project_description}")

    if existing_papers:
        parts.append(
            "\nThe following papers have already been retrieved — generate queries that "
            "are likely to surface DIFFERENT, complementary papers not yet found:"
        )
        for i, title in enumerate(existing_papers[:15], start=1):
            parts.append(f"  {i}. {title}")

    return "\n".join(parts)


def _strip_wildcards(query: str) -> str:
    """Remove truncation wildcards the LLM may still emit.

    OpenAlex's stemmed ``search=`` parameter returns HTTP 400 for any query containing
    ``*`` or ``?`` (OPENALEX-WILDCARD-400). The prompt forbids them; this is the
    deterministic backstop for when the model ignores the prompt.
    """
    cleaned = _WILDCARD_RE.sub("", query)
    return _MULTISPACE_RE.sub(" ", cleaned).strip()


class GeneratedQueriesResult(NamedTuple):
    """One round's generated queries plus the call's own provenance (mirrors
    ``app.agents.deep_analysis_agent.DeepAnalysisResult``):
    ``generate_search_queries`` alone discards the pydantic-ai run result, so calling it
    directly would fold the query generator's own calls, tokens, model and fingerprint
    into no provenance record at all, leaving every Smart Search round's own
    query-generation call invisible to ``summary.json`` and the reported token/cost
    totals."""

    queries: list[str]
    provenance: LLMCallProvenance


async def generate_search_queries_with_provenance(
    user_query: str,
    project_description: str | None = None,
    existing_papers: list[str] | None = None,
) -> GeneratedQueriesResult:
    """Run the query generator and return its own call provenance alongside the
    queries. ``generate_search_queries`` below is a thin
    wrapper over this that keeps its existing call sites and return type unchanged.
    """
    agent = get_query_generator_agent()
    prompt = _build_prompt(user_query, project_description, existing_papers)
    result = await agent.run(prompt)
    cleaned = [_strip_wildcards(q) for q in result.output.queries]
    provenance = provenance_from_run(
        "query_generator",
        result,
        model_configured=settings.deepseek_model,
        temperature=FAST_MODEL_SETTINGS.get("temperature"),
        prompt=QUERY_GENERATOR_PROMPT,
    )
    return GeneratedQueriesResult(
        queries=[q for q in cleaned if q],
        provenance=provenance,
    )


async def generate_search_queries(
    user_query: str,
    project_description: str | None = None,
    existing_papers: list[str] | None = None,
) -> list[str]:
    """Generate 3-7 academic keyword queries for the given research topic.

    Args:
        user_query: The research topic or question entered by the user.
        project_description: Optional project context to guide query generation.
        existing_papers: Optional list of paper titles already retrieved (used in
            expansion rounds to steer the LLM toward different coverage).

    Returns:
        A list of query strings ready to be sent to academic search APIs.
    """
    result = await generate_search_queries_with_provenance(
        user_query, project_description, existing_papers
    )
    return result.queries
