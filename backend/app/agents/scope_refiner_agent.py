"""Scope Refiner Agent — asks clarifying questions to narrow research scope before search.

The goal is to turn a broad topic into a precise, narrow scope so the search
returns only highly relevant papers. Keeps asking until the scope is specific
enough for exhaustive search.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from app.agents.model_config import AGENT_RETRIES, FAST_MODEL_SETTINGS
from app.config import settings


class ScopeRefinement(BaseModel):
    """Output from the scope refiner agent."""

    is_specific_enough: bool = Field(
        description="True if the topic is narrow enough for targeted academic search"
    )
    clarifying_question: str | None = Field(
        default=None,
        description="Question to ask the user to narrow scope (None if specific enough)"
    )
    options: list[str] = Field(
        default_factory=list,
        description="2-4 suggested answers for the clarifying question"
    )
    refined_topic: str | None = Field(
        default=None,
        description="The refined, narrowed topic statement (only when specific enough)"
    )
    inclusion_criteria: list[str] = Field(
        default_factory=list,
        description="Specific criteria for what papers should be included"
    )
    exclusion_criteria: list[str] = Field(
        default_factory=list,
        description="Specific criteria for what papers should be excluded"
    )


SCOPE_REFINER_PROMPT = """\
You are an expert research librarian conducting a reference interview to help a researcher
define the precise scope of their systematic literature search.

Your job is to ask clarifying questions ONE AT A TIME until you have gathered ALL the
information needed to construct precise inclusion and exclusion criteria. Keep asking
until YOU are satisfied that you understand exactly what the researcher wants.

DIMENSIONS TO EXPLORE (ask about each that is relevant to the topic):
1. Sub-topic focus — which specific aspect of the broad topic?
2. Research methodology — empirical studies? reviews? theoretical? qualitative? quantitative?
3. Population/context — specific groups? settings? countries? language pairs?
4. Time period — recent papers only? historical perspective?
5. Theoretical framework — specific theories or paradigms?
6. Application domain — academic? industry? specific use cases?
7. Outcome focus — what results or effects are they interested in?

HOW TO ASK:
- Ask ONE question per turn
- Each question should target a different dimension from the list above
- Provide 3-5 concrete options for each question
- ALWAYS include "All of the above / Comprehensive review" as the LAST option
- Include options the user might not have considered
- The user can also type a custom answer
- Ask questions in order of importance (sub-topic first, then narrow further)

WHEN TO STOP (declare is_specific_enough = true):
- Stop ONLY when you have enough information across multiple dimensions to write
  precise inclusion/exclusion criteria
- If the user keeps choosing "All of the above", that IS valid information — it means
  they want comprehensive coverage of that dimension. Move to the NEXT dimension.
- A topic like "AI tools in translator training across all educational levels focusing
  on post-editing and MT literacy" IS specific enough — you know the sub-topic,
  application, and focus areas
- DO NOT keep asking if you already have 3+ dimensions clarified

WHEN DECLARING SPECIFIC ENOUGH:
- refined_topic MUST be a complete, detailed statement incorporating ALL the user's answers
- List 4-6 inclusion criteria (specific, actionable rules for what papers to include)
- List 4-6 exclusion criteria (specific, actionable rules for what papers to exclude)
- Inclusion/exclusion criteria should be precise enough that two independent reviewers
  would agree on most papers

Always respond in the same language as the user's input."""


_agent: Agent[None, ScopeRefinement] | None = None


def get_scope_refiner_agent() -> Agent[None, ScopeRefinement]:
    global _agent
    if _agent is None:
        _agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=ScopeRefinement,
            instructions=SCOPE_REFINER_PROMPT,
            model_settings=FAST_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
    return _agent


async def refine_scope(
    user_query: str,
    previous_answers: list[dict] | None = None,
) -> ScopeRefinement:
    """Ask the scope refiner to evaluate and narrow the research topic.

    Args:
        user_query: The user's research topic
        previous_answers: List of {question, answer} dicts from previous rounds

    Returns:
        ScopeRefinement with either a clarifying question or the final refined scope
    """
    prompt = f"Research topic: {user_query}"

    if previous_answers:
        prompt += "\n\nThe user has already answered these clarifying questions:"
        for qa in previous_answers:
            prompt += f"\n  Q: {qa['question']}\n  A: {qa['answer']}"
        prompt += f"\n\nYou have asked {len(previous_answers)} question(s) so far."
        prompt += "\nDo you have enough information to define precise inclusion/exclusion criteria, or do you need to ask about another dimension?"

    agent = get_scope_refiner_agent()
    result = await agent.run(prompt)

    # Safety valve: after 7 questions, force conclusion to prevent infinite loops
    if previous_answers and len(previous_answers) >= 7 and not result.output.is_specific_enough:
        answers_summary = "; ".join(f"{qa['question']} → {qa['answer']}" for qa in previous_answers)
        result.output.is_specific_enough = True
        result.output.refined_topic = f"{user_query} — scope: {answers_summary}"
        result.output.clarifying_question = None
        result.output.options = []

    return result.output
