"""Writing Rules Merge Agent — intelligently merges golden standard with journal guidelines.

Produces a unified set of writing instructions tailored to the specific paper type
and target journal, resolving conflicts between the two sources.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings


class MergedWritingRules(BaseModel):
    """Unified writing rules produced by merging golden standard + journal guidelines."""

    section_structure: str = Field(description="How this section should be structured (headings, flow, organization)")
    word_limit: str = Field(description="Word limit for this section, derived from total paper limit")
    citation_rules: str = Field(description="Citation style and formatting rules")
    tone_and_style: str = Field(description="Academic tone, tense, voice requirements")
    content_requirements: str = Field(description="What must be included in this section")
    formatting: str = Field(description="Font, spacing, heading style requirements")
    special_instructions: str = Field(description="Any journal-specific or paper-type-specific rules")


MERGE_PROMPT = """\
You are an expert academic writing advisor who creates precise writing instructions
by merging two sources of rules:

1. GOLDEN STANDARD — the default best-practice structure for this type of academic paper
2. JOURNAL GUIDELINES — specific requirements from the target journal

Your job is to produce a UNIFIED set of writing instructions that:
- Uses the journal's requirements wherever they specify something (word limits, formatting)
- Uses the golden standard wherever the journal doesn't specify (section structure, content flow, academic conventions)
- Resolves any conflicts in favor of the journal's requirements
- Calculates section-specific word limits from the total word limit (e.g., if total is 8000 words and there are 6 sections, Introduction might get ~1000 words, Literature Review ~2500, etc.)
- Includes specific, actionable instructions that a writing AI can follow exactly

In-text citations in the drafted text are always author-year (e.g., "(Smith, 2020)" or
"Smith (2020)"), never the journal's own citation style: the journal's style is applied
only when the paper is exported, not while it is being drafted, so your citation_rules
must not mention any style other than author-year.

Be precise and specific — don't say "follow standard conventions", say exactly what to do.
"""

_agent: Agent[None, MergedWritingRules] | None = None


def get_writing_rules_agent() -> Agent[None, MergedWritingRules]:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=MergedWritingRules,
            instructions=MERGE_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


# Golden standards for each paper type
GOLDEN_STANDARDS = {
    "literature_review": {
        "sections": ["Abstract", "Introduction", "Literature Review", "Discussion", "Conclusion"],
        "total_words": "6000-10000",
        "abstract_words": "150-300",
        "citation_style": "APA 7th edition",
        "structure": "Thematic synthesis organized by themes, not chronologically or by individual papers. "
                     "Evaluate studies for content, design, interesting points, and gaps. "
                     "Primary sources preferred; max 2 'as cited in' references. "
                     "Present tense for views; past tense for study findings.",
        "font": "Times New Roman 12pt",
        "spacing": "Double-spaced",
    },
    "research_article": {
        "sections": ["Abstract", "Introduction", "Literature Review", "Methods", "Results", "Discussion", "Implications", "Conclusion"],
        "total_words": "5000-8000",
        "abstract_words": "150-300",
        "citation_style": "APA 7th edition",
        "structure": "IMRaD structure. Introduction states purpose and gaps. "
                     "Methods describes procedure in past tense. "
                     "Results presents findings without interpretation. "
                     "Discussion interprets results against literature.",
        "font": "Times New Roman 12pt",
        "spacing": "Double-spaced",
    },
}


async def merge_writing_rules(
    section_type: str,
    paper_type: str,
    journal_guidelines: dict | None = None,
) -> MergedWritingRules:
    """Merge golden standard with journal guidelines for a specific section.

    Args:
        section_type: e.g., "introduction", "literature_review", "methods"
        paper_type: "literature_review" or "research_article"
        journal_guidelines: dict from project.target_journal_guidelines

    Returns:
        Unified MergedWritingRules for the AI writer to follow
    """
    golden = GOLDEN_STANDARDS.get(paper_type, GOLDEN_STANDARDS["research_article"])

    prompt = f"""Paper type: {paper_type}
Section to write: {section_type}

GOLDEN STANDARD (default best practice):
- Required sections: {', '.join(golden['sections'])}
- Total word range: {golden['total_words']}
- Abstract words: {golden['abstract_words']}
- Citation style: {golden['citation_style']}
- Structure notes: {golden['structure']}
- Font: {golden['font']}
- Spacing: {golden['spacing']}
"""

    if journal_guidelines:
        prompt += f"""
JOURNAL GUIDELINES (from target journal: {journal_guidelines.get('journal_name', 'Unknown')}):
"""
        if journal_guidelines.get("word_limit_total"):
            prompt += f"- Total word limit: {journal_guidelines['word_limit_total']} words\n"
        if journal_guidelines.get("word_limit_abstract"):
            prompt += f"- Abstract word limit: {journal_guidelines['word_limit_abstract']} words\n"
        if journal_guidelines.get("abstract_structure"):
            prompt += f"- Abstract type: {journal_guidelines['abstract_structure']}\n"
        if journal_guidelines.get("required_sections"):
            prompt += f"- Required sections: {', '.join(journal_guidelines['required_sections'])}\n"
        # citation_style and reference_format are deliberately never passed here: in-text
        # citations in a draft are always author-year, and the journal's own style is
        # applied only at export
        # (app.services.export), so telling this prompt the journal's style would only
        # risk it producing a citation_rules instruction that contradicts _BASE_RULES.
        if journal_guidelines.get("heading_style"):
            prompt += f"- Heading style: {journal_guidelines['heading_style']}\n"
        if journal_guidelines.get("line_spacing"):
            prompt += f"- Line spacing: {journal_guidelines['line_spacing']}\n"
        if journal_guidelines.get("font_requirements"):
            prompt += f"- Font: {journal_guidelines['font_requirements']}\n"
        if journal_guidelines.get("keyword_requirements"):
            prompt += f"- Keywords: {journal_guidelines['keyword_requirements']}\n"
        if journal_guidelines.get("figure_table_rules"):
            prompt += f"- Figures/Tables: {journal_guidelines['figure_table_rules']}\n"
        if journal_guidelines.get("special_requirements"):
            prompt += f"- Special: {journal_guidelines['special_requirements']}\n"
    else:
        prompt += "\nNo journal-specific guidelines available. Use golden standard only.\n"

    prompt += f"""
Produce unified writing instructions for the "{section_type}" section of this {paper_type}.
Calculate the appropriate word count for this specific section based on total limits.
Resolve any conflicts between golden standard and journal guidelines (journal wins).
"""

    agent = get_writing_rules_agent()
    result = await agent.run(prompt)
    return result.output
