"""Compliance Rules Merge Agent — intelligently merges golden standard with journal guidelines
for compliance checking purposes.

Decides which rules to use from each source based on the paper type, resolving conflicts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings


class MergedComplianceRules(BaseModel):
    """Unified compliance rules produced by AI merge."""

    word_limit_min: int = Field(description="Minimum word count for the paper")
    word_limit_max: int = Field(description="Maximum word count for the paper")
    abstract_word_min: int = Field(description="Minimum abstract word count")
    abstract_word_max: int = Field(description="Maximum abstract word count")
    abstract_structure: str = Field(description="'structured' or 'unstructured'")
    required_sections: list[str] = Field(description="Required sections in order, appropriate for the paper type")
    citation_style: str = Field(description="Citation style to use")
    reasoning: str = Field(description="Brief explanation of merge decisions")


MERGE_PROMPT = """\
You are an expert academic compliance advisor. You merge two sources of rules to produce
unified compliance checking standards for a specific paper type.

1. GOLDEN STANDARD — default best-practice requirements for this paper type
2. JOURNAL GUIDELINES — specific requirements from the target journal

CRITICAL RULES:
- For REQUIRED SECTIONS: Use the golden standard's sections for the paper type.
  A literature review article needs: abstract, introduction, literature review, discussion, conclusion.
  A research article needs: abstract, introduction, literature review, methods, results, discussion, implications, conclusion.
  The journal's section list is usually generic and doesn't know the paper type.
  ONLY use the journal's sections if they clearly match the paper type.

- For WORD LIMITS: Use the journal's limits if specified. Fall back to golden standard.
  Calculate reasonable min/max range (e.g., if journal says 8000 max, set min to 7000).

- For ABSTRACT: Use the journal's abstract requirements if specified (word limit, structured/unstructured).
  Fall back to golden standard.

- For CITATION STYLE: Use the journal's citation style if specified. Fall back to golden standard.

Output section names in lowercase (e.g., "introduction", "literature review", "methods").
"""

_agent: Agent[None, MergedComplianceRules] | None = None


def get_compliance_rules_agent() -> Agent[None, MergedComplianceRules]:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=MergedComplianceRules,
            instructions=MERGE_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


GOLDEN_STANDARDS_TEXT = {
    "literature_review": """
Paper type: Literature Review Article
- Word count: 6000-10000 words
- Abstract: 150-300 words, unstructured
- Required sections: abstract, introduction, literature review, discussion, conclusion
- Citation style: APA 7th edition
""",
    "research_article": """
Paper type: Research Article
- Word count: 5000-8000 words
- Abstract: 150-300 words, unstructured
- Required sections: abstract, introduction, literature review, methods, results, discussion, implications, conclusion
- Citation style: APA 7th edition
""",
}


async def merge_compliance_rules(
    paper_type: str,
    journal_guidelines: dict | None = None,
) -> MergedComplianceRules:
    """AI-powered merge of golden standard + journal guidelines for compliance.

    Args:
        paper_type: "literature_review" or "research_article"
        journal_guidelines: dict from project.target_journal_guidelines

    Returns:
        MergedComplianceRules with unified standards
    """
    golden_text = GOLDEN_STANDARDS_TEXT.get(paper_type, GOLDEN_STANDARDS_TEXT["research_article"])

    prompt = f"GOLDEN STANDARD:\n{golden_text}\n"

    if journal_guidelines:
        prompt += f"\nJOURNAL GUIDELINES ({journal_guidelines.get('journal_name', 'Unknown')}):\n"
        for key in ["word_limit_total", "word_limit_abstract", "abstract_structure",
                     "required_sections", "citation_style", "reference_format",
                     "heading_style", "line_spacing", "font_requirements",
                     "keyword_requirements", "special_requirements"]:
            val = journal_guidelines.get(key)
            if val:
                prompt += f"- {key}: {val}\n"
    else:
        prompt += "\nNo journal guidelines available. Use golden standard only.\n"

    prompt += f"\nProduce unified compliance rules for a {paper_type} paper."

    agent = get_compliance_rules_agent()
    result = await agent.run(prompt)
    return result.output
