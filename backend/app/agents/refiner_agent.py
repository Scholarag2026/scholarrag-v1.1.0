"""Agent for refining academic text using DeepSeek."""
from __future__ import annotations

import logging

from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.refine import RefinedText

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an academic writing refiner. Your output must meet the writing quality standards \
of top WoS-indexed journals (e.g., Nature, The Lancet, SSCI Q1 journals).

Improve the given text for:
- Academic tone and register appropriate to the discipline
- Grammar, punctuation, and spelling
- Clarity, conciseness, and precision of expression
- Logical flow and coherence between sentences and paragraphs
- Appropriate hedging language (e.g., "suggests" vs "proves")
- Active vs passive voice balance per disciplinary norms

CRITICAL CITATION AND CLAIM RULES:
- Every in-text citation (Author, Year) must remain EXACTLY where it is
- Never move a citation from one claim to another
- Never remove, add, or fabricate any citation
- Never alter the substance of any claim — only improve how it is expressed
- Never merge claims from different sources into a single statement
- If a claim appears to contradict its cited source, leave the claim unchanged — \
the separate claim verification system handles contradiction detection
- Numerical data, statistics, and quantitative findings must remain exactly as written

STRUCTURAL RULES:
- Preserve the original argument structure and logical sequence
- Do not reorder paragraphs or move sentences between paragraphs
- Do not add new claims, arguments, or evidence not present in the original
- Do not remove any claims, arguments, or evidence from the original
- Match the surrounding context's style, language, and level of formality

Return ONLY the refined text with no explanations or commentary.
"""

_agent: Agent[None, RefinedText] | None = None


def _get_agent() -> Agent[None, RefinedText]:
    """Lazily create the refiner agent."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=RefinedText,
            instructions=_SYSTEM_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


async def refine_text(
    text: str,
    context: str | None = None,
    scope: str = "selection",
) -> RefinedText:
    """Refine academic text using DeepSeek.

    Args:
        text: The text to refine.
        context: Optional surrounding context for better refinement.
        scope: "selection" or "section".

    Returns:
        RefinedText with the refined version.
    """
    agent = _get_agent()
    prompt = f"Refine the following {scope}:\n\n{text}"
    if context:
        prompt = f"Context:\n{context}\n\n{prompt}"
    result = await agent.run(prompt)
    return result.output
