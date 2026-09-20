"""Deep analysis agent — produces comprehensive structured analysis of a paper's full text."""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from app.agents.model_config import AGENT_RETRIES, ANALYSIS_MODEL_SETTINGS
from app.config import settings
from app.schemas.provenance import LLMCallProvenance, provenance_from_run
from app.services.expertise import get_expertise_prompt

logger = logging.getLogger(__name__)

DEEP_ANALYSIS_PROMPT = """\
You are an expert academic analyst. Your task is to read a paper's full text carefully and \
extract a comprehensive, structured analysis. This analysis will be the foundation for all \
future academic writing derived from this paper, so thoroughness is essential.

Your whole reply has to fit in one response, so each field below carries a limit. Stay \
inside every limit: a reply that runs past the limit is cut off mid-answer and the whole \
analysis of this paper is then lost, including the evidence items.

Instructions for each field:

key_findings:
  Extract the main results and conclusions. Be specific — include numbers, effect sizes, \
  and named outcomes where present. At most 200 words.

methodology:
  Describe the research design, sample characteristics, data collection procedures, \
  and analysis methods. Include software, instruments, or corpora used. At most 200 words.

theoretical_framework:
  Identify every theory, model, or paradigm the paper explicitly uses or references as a \
  conceptual lens. Name authors associated with each framework where mentioned. At most \
  150 words.

key_quotes_with_citations:
  Extract EXACT verbatim quotes that the paper's authors cite from OTHER researchers. \
  Preserve the original in-text citation as it appears in the paper. \
  Format each entry as a complete sentence showing how the paper's authors use the quote, e.g.: \
  'As Baker (2018) argued, "translation technology has fundamentally altered the \
translator role".' \
  These quotes will be used as secondary citations in generated writing, so accuracy is critical. \
  At most 8 entries.

claims_and_evidence:
  State the paper's main claims and describe the evidence presented for each. \
  Include both supported and contested claims. At most 200 words.

builds_on:
  Identify the prior works, researchers, and intellectual traditions this paper builds upon. \
  Name specific authors and papers where possible. At most 150 words.

limitations:
  Extract the acknowledged limitations. Include relevant quotes from the authors where they \
  explicitly discuss limitations, caveats, or scope restrictions. At most 150 words.

future_directions:
  List the future research directions the authors suggest, with specifics. At most 150 words.

themes:
  List 3-8 key topics and themes addressed by this paper (short noun phrases, e.g. \
  "machine translation post-editing", "translator agency", "corpus linguistics").

evidence:
  Extract evidence items that could ground a citation to this specific paper, up to the \
  maximum the message below names. Each item's quote must be a character for character \
  copy out of the full text below: start it at the first word of a sentence, end it \
  at that sentence's own closing full stop, question mark or exclamation mark, and change \
  nothing in between. Do not reword it, do not shorten it, do not tidy up its punctuation \
  or spelling, and do not join sentences that are not next to each other in the text. \
  Copy one to three consecutive sentences, at least 8 words in total. Code checks every \
  quote against the paper's own text and throws away anything that is not an exact copy, \
  so an item you have reworded is worse than no item at all. Give the number of the chunk \
  you copied from as chunk_index. Prefer quotes that state a concrete finding, method \
  detail, sample characteristic, or limitation over generic or introductory sentences. \
  For each item:
  - finding: one sentence, in your own words, stating what the quote shows or claims.
  - kind: "finding" (a result), "method" (how the study was done), "sample" (who or what \
    was studied), or "limitation" (an acknowledged limitation or caveat).
  - origin: "own" when the quote states this paper's own authors' claim; "reported" when \
    the quote is this paper describing or citing another work's claim (for example inside \
    a literature review). Mark a quote "reported" whenever it is really about someone \
    else's study, even if this paper's authors endorse it -- only "own" evidence is ever \
    shown to a later writer citing this paper.
  - concepts: 1-4 short topic tags for this evidence (lower case noun phrases), e.g. \
    "translation quality", "sample size".

Be specific rather than general, do not omit fields, and keep every field inside its limit.\
"""

#: The per-paper evidence-item caps ``analyze_paper_text_with_provenance`` walks down when
#: a reply comes back truncated.
#:
#: The first value is the cap asked for normally. On the twelve demo seed papers the
#: earlier prompt, which asked for "5-15" items, drew 19 to 21 items per paper and ran 8
#: of 12 calls past the output ceiling, and a call that ends that way yields nothing at
#: all -- no analysis, no evidence. A cap of ten items keeps a reply inside the budget in
#: ``app.agents.model_config.ANALYSIS_MODEL_SETTINGS`` while still leaving room for a
#: paper to be rejected on several items and clear the six-per-paper target.
EVIDENCE_ITEM_CAPS: tuple[int, ...] = (10, 6, 3)

# Cached per resolved expertise-prompt suffix ("" for none/unknown levels).
_agents: dict[str, Agent[None, DeepPaperAnalysis]] = {}


class EvidenceItem(BaseModel):
    """One evidence item the analysis agent proposes. Code, not the model, decides
    which proposed items are accepted -- see
    ``app.services.deep_analysis.validate_evidence_items``."""

    quote: str = Field(
        description="One to three consecutive complete sentences, copied character for "
        "character from the full text, at least 8 words long."
    )
    chunk_index: int = Field(
        description="0-based index of the numbered chunk this quote is copied from, "
        "exactly as shown in the full text above. Code looks the quote up in this chunk "
        "first and in the paper's other chunks after it, so a mistake here costs "
        "nothing as long as the quote itself is an exact copy."
    )
    finding: str = Field(
        description="One sentence, in your own words, stating what this quote shows or claims."
    )
    kind: Literal["finding", "method", "sample", "limitation"] = Field(
        description="What sort of evidence this is."
    )
    origin: Literal["own", "reported"] = Field(
        description="'own' when the quote states this paper's own authors' claim; "
        "'reported' when the paper is describing or citing another work's claim."
    )
    concepts: list[str] = Field(
        default_factory=list,
        description="1-4 short topic tags for this evidence (lower case noun phrases).",
    )


class DeepPaperAnalysis(BaseModel):
    """Structured full-text analysis of an academic paper."""

    key_findings: str = Field(description="Main results and conclusions of this paper")
    methodology: str = Field(
        description="Research design, sample, data collection, analysis methods"
    )
    theoretical_framework: str = Field(
        description="Theories, models, paradigms used or referenced"
    )
    key_quotes_with_citations: list[str] = Field(
        default_factory=list,
        description=(
            "Important quotes from OTHER authors cited in this paper, preserved with their "
            "original in-text citations. E.g., 'As Smith (2020) argued, translation technology "
            "has fundamentally altered the translator role'"
        ),
    )
    claims_and_evidence: str = Field(
        description="What this paper claims and what evidence supports it"
    )
    builds_on: str = Field(
        description="Which prior works and researchers this paper builds upon"
    )
    limitations: str = Field(
        description="Acknowledged limitations, with relevant quotes if available"
    )
    future_directions: str = Field(description="Suggested future research directions")
    themes: list[str] = Field(
        default_factory=list, description="Key topics and themes addressed"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Verbatim, chunk-located evidence items.",
    )


def _resolve_instructions(expertise_level: str | None) -> str:
    """The exact instructions text ``get_deep_analysis_agent`` builds for
    *expertise_level*, shared with ``analyze_paper_text_with_provenance`` so the
    provenance's ``prompt_version`` hashes the same text the agent was actually
    constructed with."""
    expertise_suffix = get_expertise_prompt(expertise_level)
    return (
        f"{DEEP_ANALYSIS_PROMPT}\n\n{expertise_suffix}"
        if expertise_suffix
        else DEEP_ANALYSIS_PROMPT
    )


def get_deep_analysis_agent(
    expertise_level: str | None = None,
) -> Agent[None, DeepPaperAnalysis]:
    """Return the cached deep-analysis agent for *expertise_level*.

    Agents are cached per resolved expertise-prompt suffix (so ``None`` and any
    unrecognised level share one agent). Each pydantic-ai ``Agent`` owns an
    ``httpx.AsyncClient`` that is never closed, so constructing one per paper leaked
    a connection pool per paper.
    """
    expertise_suffix = get_expertise_prompt(expertise_level)
    agent = _agents.get(expertise_suffix)
    if agent is None:
        instructions = _resolve_instructions(expertise_level)
        agent = Agent(
            f"deepseek:{settings.deepseek_model}",
            deps_type=None,
            output_type=DeepPaperAnalysis,
            instructions=instructions,
            model_settings=ANALYSIS_MODEL_SETTINGS,
            retries=AGENT_RETRIES,
        )
        _agents[expertise_suffix] = agent
    return agent


def _build_numbered_chunks(chunks: list[dict]) -> str:
    """The paper's full text, split into 0-based numbered chunks: the
    ``evidence`` field asks the model to name the chunk a quote came from using these
    same numbers, so ``app.services.deep_analysis.validate_evidence_items`` can look the
    quote up in ``chunks[chunk_index]`` and confirm it verbatim -- code, not the model,
    decides acceptance."""
    if not chunks:
        return "No full-text chunks available."
    blocks = []
    for i, chunk in enumerate(chunks):
        section = chunk.get("section") if isinstance(chunk, dict) else None
        header = f"--- Chunk {i} ({section}) ---" if section else f"--- Chunk {i} ---"
        text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _build_analysis_prompt(
    title: str,
    authors: str,
    year: int | None,
    chunks: list[dict],
    max_evidence_items: int = EVIDENCE_ITEM_CAPS[0],
) -> str:
    """Assemble the user-turn prompt from paper metadata and the paper's full-text
    chunks, numbered so the model can name the chunk each evidence
    quote is copied from.

    *max_evidence_items* is stated here rather than in the instructions because it is
    what the shrink-and-retry ladder varies per attempt: the instructions, and
    so the agent and its ``prompt_version``, stay the same across a paper's attempts."""
    year_str = str(year) if year is not None else "unknown"
    parts: list[str] = [
        f"Paper title: {title}",
        f"Authors: {authors}",
        f"Year: {year_str}",
        "",
        f"Extract at most {max_evidence_items} evidence items for this paper.",
        "",
        "Full text, split into numbered chunks (report chunk_index using these same "
        "numbers when you extract evidence):",
        "",
        _build_numbered_chunks(chunks),
    ]
    return "\n".join(parts)


class DeepAnalysisResult(NamedTuple):
    """One full-text analysis plus the call's own provenance (mirrors
    ``app.agents.citation_link_agent.CitationLinkResult``): the write job's grounding
    step needs the model, prompt version and token usage of this call to fold into the
    section's own provenance, which ``analyze_paper_text`` alone cannot provide since it
    discards the pydantic-ai run result."""

    output: DeepPaperAnalysis
    provenance: LLMCallProvenance


async def analyze_paper_text_with_provenance(
    title: str,
    authors: str,
    year: int | None,
    chunks: list[dict],
    expertise_level: str | None = None,
) -> DeepAnalysisResult:
    """Run AI deep analysis on a paper's full-text chunks and return the call's
    provenance alongside the analysis. ``analyze_paper_text`` below is a
    thin wrapper over this that keeps its existing call sites and return type unchanged.

    ``chunks`` is the paper's raw ``fulltext_chunks`` -- shown to the
    model numbered, not joined into one string, so the model can name which chunk each
    evidence quote is copied from and the caller can check that quote against the paper's
    own text verbatim.

    A reply that runs past the output budget comes back as
    ``pydantic_ai.exceptions.IncompleteToolCall`` (a subclass of
    ``UnexpectedModelBehavior``), which loses the whole paper's analysis, not just the
    items that did not fit. So each cap in ``EVIDENCE_ITEM_CAPS`` is tried in turn and
    the paper is asked for fewer evidence items each time; the failure from the
    smallest cap is raised, since both callers already catch one paper's analysis failing
    and carry on with the rest.
    """
    agent = get_deep_analysis_agent(expertise_level)
    last_error: UnexpectedModelBehavior | None = None
    for attempt, cap in enumerate(EVIDENCE_ITEM_CAPS, 1):
        prompt = _build_analysis_prompt(title, authors, year, chunks, max_evidence_items=cap)
        try:
            result = await agent.run(prompt)
        except UnexpectedModelBehavior as exc:
            last_error = exc
            logger.warning(
                "Analysis of %r came back unusable at a cap of %d evidence items "
                "(attempt %d of %d): %s",
                title,
                cap,
                attempt,
                len(EVIDENCE_ITEM_CAPS),
                exc,
            )
            continue
        provenance = provenance_from_run(
            "deep_analysis",
            result,
            model_configured=settings.deepseek_model,
            temperature=ANALYSIS_MODEL_SETTINGS.get("temperature"),
            prompt=_resolve_instructions(expertise_level),
        )
        return DeepAnalysisResult(output=result.output, provenance=provenance)
    raise last_error if last_error is not None else RuntimeError(
        "No evidence-item cap was tried"
    )


async def analyze_paper_text(
    title: str,
    authors: str,
    year: int | None,
    chunks: list[dict],
    expertise_level: str | None = None,
) -> DeepPaperAnalysis:
    """Run AI deep analysis on a paper's full-text chunks.

    Args:
        title: The paper's title.
        authors: Author names as a single string.
        year: Publication year, or None if unknown.
        chunks: The paper's ``fulltext_chunks`` (a list of ``{"section", "text"}``
            dicts, in order), shown to the model numbered.
        expertise_level: Optional expertise level (e.g. "student", "researcher",
            "faculty") used to tailor the analysis tone and detail.

    Returns:
        A DeepPaperAnalysis with all fields populated by the LLM.
    """
    result = await analyze_paper_text_with_provenance(
        title, authors, year, chunks, expertise_level=expertise_level
    )
    return result.output


def build_abstract_only_analysis(
    title: str, authors: str, year: int | None, abstract: str
) -> dict:
    """For papers without full text: wrap the abstract as-is in the analysis schema.

    No AI call is made. The abstract is placed in key_findings; all other fields
    are empty. This provides a consistent shape for downstream consumers even when
    full-text analysis is not available.

    Args:
        title: The paper's title (unused in output but accepted for call-site symmetry).
        authors: Author names (unused in output but accepted for call-site symmetry).
        year: Publication year (unused in output but accepted for call-site symmetry).
        abstract: The paper's abstract text.

    Returns:
        A dict matching the DeepPaperAnalysis field names.
    """
    return {
        "key_findings": abstract,
        "methodology": "",
        "theoretical_framework": "",
        "key_quotes_with_citations": [],
        "claims_and_evidence": "",
        "builds_on": "",
        "limitations": "",
        "future_directions": "",
        "themes": [],
    }
