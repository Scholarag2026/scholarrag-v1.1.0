"""Agent for extracting metadata from academic paper text using DeepSeek."""
from __future__ import annotations

import logging

from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings
from app.schemas.paper_upload import ExtractedPaperMetadata

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an academic paper metadata extractor. Given the full text of an academic paper, \
extract the following fields:

- title: The paper's title
- authors: List of author names as [{"name": "Full Name"}, ...]
- year: Publication year as integer
- journal_name: Name of the journal or conference
- doi: The DOI identifier (e.g., "10.1234/example")
- abstract: The paper's abstract text

Return valid JSON matching the schema. If a field cannot be determined from the text, \
return null for that field. For authors, extract all authors you can find.
"""

_agent: Agent[None, ExtractedPaperMetadata] | None = None


def _get_agent() -> Agent[None, ExtractedPaperMetadata]:
    """Lazily create the metadata extraction agent."""
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=ExtractedPaperMetadata,
            instructions=_SYSTEM_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


async def extract_metadata_from_text(text: str) -> ExtractedPaperMetadata:
    """Extract metadata from paper text using DeepSeek.

    Args:
        text: The full text of the paper (first ~8000 chars recommended).

    Returns:
        ExtractedPaperMetadata with extracted fields.
    """
    agent = _get_agent()
    truncated = text[:8000] if len(text) > 8000 else text
    try:
        result = await agent.run(
            f"Extract metadata from this academic paper:\n\n{truncated}"
        )
        return result.output
    except Exception as e:
        logger.warning("Metadata extraction failed: %s", e)
        return ExtractedPaperMetadata(title="Untitled")
