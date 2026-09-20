"""Expertise-level system prompt modifiers for AI services."""

_PROMPTS: dict[str, str] = {
    "student": (
        "The user is a student researcher. "
        "Provide detailed step-by-step guidance, explain methodology choices, "
        "define technical terms on first use, and include academic writing "
        "best practices reminders."
    ),
    "researcher": (
        "The user is an experienced researcher. "
        "Provide concise, direct feedback. Assume familiarity with standard "
        "research methods and academic conventions. Focus on substance over form."
    ),
    "faculty": (
        "The user is a faculty member / senior researcher. "
        "Engage as a peer collaborator. Focus on argumentation depth, novelty, "
        "positioning within the field, and strategic framing. "
        "Minimize basic guidance."
    ),
}


def get_expertise_prompt(expertise_level: str | None) -> str:
    """Return the system-prompt modifier for the given expertise level.

    Returns an empty string for unknown or None levels so callers can
    safely concatenate without conditionals.
    """
    if expertise_level is None:
        return ""
    return _PROMPTS.get(expertise_level, "")
