import logging
import math
import re
from typing import Any, Mapping, NamedTuple, Sequence
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.orm import joinedload

# Local import inside ``build_citation_link_map`` (below) would also work, but the linker
# is exercised by every generation and is cheap to import eagerly, unlike the other local
# imports in this module that exist to avoid a circular import at module load time.
from app.agents.citation_link_agent import CitationLinkResult, link_citations
from app.config import settings
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.dataset import Codebook, Dataset
from app.models.draft import Draft
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.schemas.provenance import LLMCallProvenance, prompt_version
from app.services import task as task_service
from app.services.citation_audit import audit_citations
from app.services.deep_analysis import format_authors, is_full_text_analysis
from app.services.expertise import get_expertise_prompt

logger = logging.getLogger(__name__)

# Sampling temperature of the drafting call; recorded in every writing job result.
# max_tokens is configurable (settings.writing_max_tokens, default 4096, the fixed
# value this constant used to hold); call_deepseek reads it at call time instead.
WRITING_TEMPERATURE = 0.7

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
}

_BASE_RULES = """RULES FOR ALL SECTIONS:
- Cite a paper, in APA 7th format ((Author, Year) or Author (Year)), only for a proposition that is stated in the material supplied for that paper.
- ONLY cite papers provided in the context below — NEVER invent citations
- When one sentence cites several papers, each cited paper must support the clause it is attached to. A claim about the field as a whole, or about other papers, is never attributed to a paper.
- A sentence that carries a citation asserts only what that citation supports. Put a second finding, a study-design detail, or a population detail in its own sentence with its own citation, one citation each — never add it as a trailing or leading clause of a sentence whose citation does not support it.
- Never rank or compare sources ("the strongest evidence", "treated most directly", "the clearest demonstration"). State the finding and cite it.
- A statement that no supplied material supports is written without a citation and flagged [NEEDS CITATION].
- Each paragraph: topic sentence → supporting body → wrap-up sentence
- Smooth transitions between paragraphs
- Output in Markdown format"""

SECTION_PROMPTS: dict[str, str] = {
    "literature_review": """You are an academic writing assistant specializing in humanities and social sciences literature reviews.

SECTION-SPECIFIC RULES:
1. SYNTHESIZE, don't list — group by theme/issue discussed, not by individual papers
2. DO NOT use author as the subject at the beginning of a sentence
3. Present tense for views/arguments; past tense for specific study findings
4. Proper attribution patterns: "Scholar X argues...", "demonstrates...", "suggests..."
5. Evaluate each study: research content/design, interesting points, gaps
6. Primary sources preferred (max 2 "as cited" references)

""" + _BASE_RULES,

    "introduction": """You are an academic writing assistant specializing in research paper introductions for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. State the purpose and aim of the research clearly
2. Establish significance with research context — why does this matter?
3. Identify the GAP: what is not explored, under-explored, controversial, or needs boundary-pushing
4. Present research questions (qualitative: research question; quantitative: hypothesis)
5. Briefly outline ethical considerations if relevant
6. Mention validity and reliability approach
7. End with a roadmap of the paper structure
8. Do NOT include findings or conclusions — this is the introduction

""" + _BASE_RULES,

    "methods": """You are an academic writing assistant specializing in research methodology sections for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. Justify the chosen methodology and instruments — explain WHY this approach
2. Describe participants: How chosen, Whom, What criteria, Why these participants
3. Define variables/factors if appropriate
4. Detail data collection procedures step by step
5. Explain data analysis approach
6. Address ethical considerations (consent, anonymity, IRB approval)
7. Write in past tense (methods were applied, data were collected)
8. Be specific and replicable — another researcher should be able to follow this

""" + _BASE_RULES,

    "results": """You are an academic writing assistant specializing in results/findings sections for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. Write ENTIRELY in past tense
2. State findings WITHOUT bias or interpretation — just the facts
3. Use subheadings to organize different findings
4. Reference tables/charts by number: "Table 1 shows..." or "As shown in Figure 2..."
5. Tables/charts must be numbered, titled, captioned, and standalone (understandable without text)
6. Keep each table/description concise — no longer than half a page
7. Present quantitative results with statistical details (p-values, effect sizes, confidence intervals)
8. Present qualitative findings with representative quotes
9. Do NOT discuss implications — save that for the Discussion section

""" + _BASE_RULES,

    "discussion": """You are an academic writing assistant specializing in discussion sections for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. INTERPRET the results — refer back to the theoretical framework
2. Relate findings to the literature review and existing studies — how do they compare?
3. Reference the theoretical framework and discuss new insights
4. Move the reader's understanding forward — what does this mean for the field?
5. Address unexpected findings and explain possible reasons
6. Compare with similar studies — where do findings align or diverge?
7. Acknowledge limitations naturally within the discussion
8. DO NOT simply repeat the results — analyze and contextualize them

""" + _BASE_RULES,

    "implications": """You are an academic writing assistant specializing in implications sections for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. Present THEORETICAL implications: new insights from the discussion, contributions to theory
2. Present PRACTICAL implications: how findings can be applied in practice
3. Present PEDAGOGICAL implications: educational applications (4-6 sentences with at least 2 references)
4. Each implication should flow logically from the discussion
5. Be specific and actionable — not vague generalizations
6. Connect each implication back to the research findings

""" + _BASE_RULES,

    "conclusion": """You are an academic writing assistant specializing in conclusion sections for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. Summarize the key results, findings, and new insights concisely
2. Concluding statements should be SHORT and PUNCHY — impactful final thoughts
3. Present limitations in a forward-looking, positive tone (opportunities for future research)
4. Suggest specific directions for future research
5. End with a strong closing statement that reinforces the paper's contribution
6. Do NOT introduce new information or citations not discussed earlier
7. Keep it concise — this is a wrap-up, not a new discussion

""" + _BASE_RULES,

    "abstract": """You are an academic writing assistant specializing in academic abstracts for humanities and social sciences.

SECTION-SPECIFIC RULES:
1. WORD LIMIT IS CRITICAL: Follow the word limit specified in the UNIFIED WRITING RULES below. If no specific limit is given, target 250 words (±10%). COUNT YOUR WORDS and ensure you are within the limit.
2. Structure: Introduction → Body → Conclusion
3. Must include: Purpose, Methods/Approach, Results/Findings, Conclusions
4. No new information — only summarize what is in the paper
5. Must be understandable to a wide audience without reading the full paper
6. Do NOT include citations in the abstract
7. Use concise, direct language — every word must earn its place
8. Write in present tense for general statements, past tense for specific findings
9. Include keywords after the abstract if specified in the rules below

""" + _BASE_RULES,
}

_DEFAULT_PROMPT = """You are an academic writing assistant for humanities and social sciences research papers.

Write the requested section following academic conventions and standards.

""" + _BASE_RULES


def get_section_prompt(section_type: str, language: str = "en") -> str:
    """Get the system prompt for a specific paper section, with optional language instruction."""
    base = SECTION_PROMPTS.get(section_type, _DEFAULT_PROMPT)
    return base + build_language_instruction(language)


def build_language_instruction(language: str) -> str:
    """Return a language instruction suffix for non-English languages, or empty string for English."""
    if language == "en":
        return ""
    language_name = LANGUAGE_NAMES.get(language, "English")
    return (
        f"\n\nIMPORTANT: Write the entire output in {language_name}. "
        f"Use academic conventions and terminology appropriate for "
        f"{language_name}-language scholarship."
    )


def _author_display_names(p) -> list[str]:
    """The names to render this paper's citation from: the acquired full text's own
    byline, when acquisition parsed and accepted one
    (``metadata["fulltext_byline"]["source"] == "fulltext"``,
    ``app.services.fulltext._parse_byline_surnames`` / the acceptance test there), else
    OpenAlex's own author list.

    The byline is stored as surnames only (`app.services.fulltext._accept_byline`), so
    only the surname is rendered here too -- APA 7th narrative/parenthetical citation
    form never needs a given name, and the writer's own prompt already produces
    "Ghane et al. (2024)" style citations from a plain surname list."""
    metadata = getattr(p, "metadata_", None) or {}
    byline = metadata.get("fulltext_byline") or {}
    if byline.get("source") == "fulltext" and byline.get("surnames"):
        return list(byline["surnames"])
    return [a.get("name", "?") for a in (p.authors or [])]


def _build_papers_context(papers: list) -> str:
    """Format papers for the AI prompt context."""
    if not papers:
        return (
            "No papers are currently in the project library. "
            "Generate a general outline with [NEEDS CITATION] markers."
        )

    lines = ["## Papers in Project Library\n"]
    for i, pp in enumerate(papers, 1):
        p = pp.paper
        authors_str = ", ".join(_author_display_names(p))
        lines.append(f"### Paper {i}")
        lines.append(f"- **Title**: {p.title}")
        lines.append(f"- **Authors**: {authors_str}")
        lines.append(f"- **Year**: {p.year or 'n.d.'}")
        lines.append(f"- **Journal**: {p.journal_name or 'Unknown'}")
        if p.abstract:
            lines.append(f"- **Abstract**: {p.abstract}")
        lines.append("")
    return "\n".join(lines)


def _paper_display_line(p) -> str:
    """"Authors (Year) - Title", the one line format every evidence-context listing
    below shares for a paper it is not otherwise detailing."""
    authors_str = ", ".join(_author_display_names(p))
    return f"{authors_str} ({p.year or 'n.d.'}) - {p.title}"


def _has_full_text(paper) -> bool:
    """A paper counts as having full text when its metadata carries at least one
    ``fulltext_chunks`` entry -- the same test ``_ground_selected_papers`` uses to
    decide whether a paper is eligible for grounding at all."""
    return bool((paper.metadata_ or {}).get("fulltext_chunks"))


class EvidenceContext(NamedTuple):
    """What ``_build_evidence_context`` produces: the context text sent
    to the writer, and the evidence catalog (``{"id", "quote", "key"}`` per shown item)
    the citation-link call validates ``evidence_ids`` against."""

    text: str
    catalog: list[dict]


async def _build_evidence_context(papers: list, session_factory) -> EvidenceContext:
    """Build the writer's context from each paper's own stored evidence, not its
    abstract or a prose summary of its analysis.

    A paper with full text (``_has_full_text``) is shown under its own heading, with its
    ``origin == "own"`` evidence rows (``app.models.evidence.Evidence``)
    grouped by concept, each item given a short id ("e1", "e2", ...) unique across this
    whole context and shown once per concept heading it belongs to. A paper without full
    text is listed by title only under "not available for citation (no full text)" and
    contributes nothing to the evidence catalog: it is excluded from composition, not
    merely discouraged. The writer is told it exists so
    it is not simply forgotten, but has nothing to cite it with.

    The returned catalog maps each shown id to the quote and the citing paper's
    author-year key (``app.services.fulltext._build_paper_lookup``'s key space), for the
    citation-link call to validate ``evidence_ids`` against. A paper
    with no derivable key (no authors, or no year) is shown but contributes no catalog
    entries, since nothing could cite it by key regardless.
    """
    from app.models.evidence import Evidence
    from app.services.fulltext import _build_paper_lookup

    full_text_papers = []
    no_full_text_papers = []
    for pp in papers:
        p = pp.paper if hasattr(pp, "paper") else pp
        (full_text_papers if _has_full_text(p) else no_full_text_papers).append(p)

    if not full_text_papers and not no_full_text_papers:
        return EvidenceContext(text="No papers available.", catalog=[])

    key_by_paper_id = {p.id: key for key, p in _build_paper_lookup(full_text_papers).items()}

    evidence_by_paper: dict = {}
    if full_text_papers:
        async with session_factory() as db:
            result = await db.execute(
                select(Evidence)
                .where(Evidence.paper_id.in_([p.id for p in full_text_papers]))
                .where(Evidence.origin == "own")
                .order_by(Evidence.paper_id, Evidence.chunk_index, Evidence.id)
            )
            for row in result.scalars().all():
                evidence_by_paper.setdefault(row.paper_id, []).append(row)

    lines: list[str] = []
    catalog: list[dict] = []
    counter = 0
    if full_text_papers:
        lines.append("## Evidence available for citation\n")
        for p in full_text_papers:
            rows = evidence_by_paper.get(p.id, [])
            lines.append(f"### {_paper_display_line(p)}")
            if not rows:
                lines.append("(no evidence extracted from this paper's full text)")
                lines.append("")
                continue

            row_ids: dict = {}
            for row in rows:
                counter += 1
                row_ids[row.id] = f"e{counter}"

            by_concept: dict[str, list] = {}
            for row in rows:
                for concept in (row.concepts or ["general"]):
                    by_concept.setdefault(concept, []).append(row)
            for concept in sorted(by_concept):
                lines.append(f"Concept: {concept}")
                for row in by_concept[concept]:
                    lines.append(f'- [{row_ids[row.id]}] "{row.quote}"')
            lines.append("")

            key = key_by_paper_id.get(p.id)
            if key:
                for row in rows:
                    catalog.append({"id": row_ids[row.id], "quote": row.quote, "key": key})

        lines.append(
            "RULES FOR USING THIS EVIDENCE:\n"
            "- A sentence citing one of the papers above must restate one of the "
            "evidence items listed for that paper, not a claim absent from them.\n"
            "- Do not write an evidence id such as [e12] in the visible text -- ids are "
            "recorded separately, later, during citation linking.\n"
        )

    if no_full_text_papers:
        lines.append("## Not available for citation (no full text)\n")
        for p in no_full_text_papers:
            lines.append(f"- {_paper_display_line(p)}")
        lines.append(
            "\nDo not cite these papers: write about the topic in general terms "
            "instead, or omit it.\n"
        )

    return EvidenceContext(text="\n".join(lines), catalog=catalog)


async def _load_journal_guidelines(project_id: UUID, session_factory) -> dict | None:
    """Load cached journal guidelines for the project."""
    async with session_factory() as db:
        result = await db.execute(
            select(Project).where(Project.id == project_id)
        )
        project = result.scalar_one_or_none()
        if project and project.target_journal_guidelines:
            return project.target_journal_guidelines
    return None


def _build_journal_rules(guidelines: dict) -> str:
    """Build a journal-specific rules block from guidelines dict."""
    journal_rules = f"""
JOURNAL-SPECIFIC REQUIREMENTS ({guidelines.get('journal_name', 'Unknown')}):
"""
    if guidelines.get('word_limit_total'):
        journal_rules += f"- Total word limit: {guidelines['word_limit_total']} words\n"
    if guidelines.get('word_limit_abstract'):
        journal_rules += f"- Abstract word limit: {guidelines['word_limit_abstract']} words\n"
    if guidelines.get('abstract_structure'):
        journal_rules += f"- Abstract structure: {guidelines['abstract_structure']}\n"
    if guidelines.get('required_sections'):
        journal_rules += f"- Required sections: {', '.join(guidelines['required_sections'])}\n"
    # citation_style and reference_format are deliberately omitted here: drafts are
    # always written with author-year citations; the journal's own style is applied
    # only at export.
    if guidelines.get('heading_style'):
        journal_rules += f"- Heading style: {guidelines['heading_style']}\n"
    if guidelines.get('line_spacing'):
        journal_rules += f"- Line spacing: {guidelines['line_spacing']}\n"
    if guidelines.get('font_requirements'):
        journal_rules += f"- Font: {guidelines['font_requirements']}\n"
    if guidelines.get('special_requirements'):
        journal_rules += f"- Special requirements: {guidelines['special_requirements']}\n"
    journal_rules += "\nYou MUST follow these journal-specific requirements exactly.\n"
    return journal_rules


class WritingCompletion(NamedTuple):
    """Generated text plus the provenance of the call that produced it."""

    content: str
    provenance: dict


_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]+.*$", re.MULTILINE)

#: A whole line that is nothing but one bold run, scanned
#: across the full text with `re.MULTILINE` -- the multi-line counterpart of
#: `_MARKDOWN_BOLD_HEADING_RE` (defined further down, once *per block*), used only by
#: `_count_body_words` to find every candidate bold sub-heading line in one pass rather
#: than one call per `split_paragraphs` block.
_BOLD_HEADING_LINE_RE = re.compile(r"^\*\*([^*\n]+)\*\*[ \t]*$", re.MULTILINE)


def _hard_maximum_words(target_words: int) -> int:
    """The regeneration threshold: 25% over the requested target,
    rounded up, so a body exactly at the target is never flagged as over length."""
    return math.ceil(target_words * 1.25)


def _length_instruction(target_words: int) -> str:
    """The literal length instruction sent to the model for one section request."""
    hard_maximum = _hard_maximum_words(target_words)
    return f"Target length: {target_words} words. Hard maximum: {hard_maximum} words."


def _count_body_words(text: str) -> int:
    """Word count of *text* with every markdown heading line (``# ...`` to ``###### ...``)
    and every whole-line bold sub-heading run (``**Sub-Heading Text**``) removed first,
    so a section's own inline heading -- however the
    model chose to mark it -- never counts toward the body length a length instruction
    is enforced against.

    A bold line only counts as a heading under the same predicate
    `_build_section_tiptap_nodes` uses to promote that exact shape to a real heading
    node, `app.services.fulltext._is_bold_heading_candidate` (imported here rather than
    duplicated, so the two can never drift apart): a bold line ending in sentence
    punctuation, or carrying a citation, is a claim sentence the writer happened to
    bold, not a sub-heading, and still counts as body text. A section
    whose sub-headings are written as whole-line bold runs instead of ``#`` lines would
    otherwise be measured -- and regenerated -- against a count inflated by every one of
    those heading words."""
    from app.services.fulltext import _is_bold_heading_candidate

    without_hash_headings = _HEADING_LINE_RE.sub("", text)
    without_bold_headings = _BOLD_HEADING_LINE_RE.sub(
        lambda m: "" if _is_bold_heading_candidate(m.group(1)) else m.group(0),
        without_hash_headings,
    )
    return len(without_bold_headings.split())


#: The bounded regeneration backstop's own floor: a section finalize leaves with
#: fewer distinct cited sentences than this is thin enough to regenerate once,
#: whole -- see ``_cited_sentence_count`` and the backstop's own comment in
#: ``generate_section`` for why the bound is "fewer than three", not "zero".
THIN_SECTION_CITED_SENTENCE_MINIMUM = 3


def _cited_sentence_count(citation_links: list[dict]) -> int:
    """The number of DISTINCT sentences *citation_links* (a finalize result's own
    surviving link list) carries a citation for -- several links can name the same
    sentence (a co-cited group, or two separate facts drawn from the same sentence),
    so this counts sentences, not links, exactly as the reader would when judging
    whether a section reads as more than a token gesture toward its own citations."""
    return len({link.get("sentence") for link in citation_links if link.get("sentence")})


class _RegenerationAttempt(NamedTuple):
    """One write attempt's own full state -- the section's original attempt, or
    the empty-section backstop's own regenerated one -- captured whole so the two
    can be compared and the better one kept, rather than the backstop always
    shipping the regenerated attempt's own outcome regardless of whether it is
    actually better. The regenerated attempt restarts from the section's
    ORIGINAL, pre-revision prompt (unchanged by this comparison), so a first
    attempt that already carries the gate's own successful revision can otherwise
    lose ground it had already made when the regenerated attempt does no better on
    its own citation-link call."""

    generated_text: str
    citation_links: list[dict] | None
    uncited_sentences: list[dict] | None
    citation_audit: dict | None
    claims: list[tuple[str, str, str, str]]
    verifications: list
    first_pass_verified_rate: float | None
    link_map_failed: bool
    claim_status: dict
    stripped_uncited_sentences: list[dict] | None
    finalize_result: Any


def _trim_body_to_hard_maximum(
    text: str,
    hard_maximum: int,
    citation_links: list[dict],
    uncited_sentences: list[dict] | None,
) -> tuple[str, list[dict], list[dict] | None, bool]:
    """Drop whole trailing blocks of the FINALIZED *text* until its body word count
    (`_count_body_words`) is at or under *hard_maximum*:
    the hard maximum is enforced once, right after the very first generation call, and
    the gated loop's own revision carries no length constraint
    of its own -- a section that survives the verification gate could still be saved
    well over the maximum. This runs last, on the exact text and citation links
    `generate_section` is about to save, after `finalize_generated_section` has already
    removed whatever verification rejected, so it is the final word on length nothing
    upstream can undo.

    Blocks are dropped one at a time from the end (`app.agents.citation_link_agent.
    split_paragraphs`'s own blank-line-separated blocks), never past the section's last
    remaining block, so a section already at or under the maximum is returned
    unchanged, and a section this trim cannot shrink without emptying it entirely is
    returned as short as it can safely go rather than left with no content at all.

    Every citation link and uncited-sentence entry whose own ``paragraph_index``
    pointed at a dropped block is dropped with it -- indices of every SURVIVING block
    are unchanged, since only the tail is ever removed, so no entry needs
    re-indexing -- and the caller must re-run `surviving_verifications` against the
    returned links: membership there is what actually decides which verified claim
    rows the saved `claim_report` and draft agree on, so a claim whose only citation
    lived in a dropped block must stop being reported as verified, not merely have its
    text silently vanish from view."""
    from app.agents.citation_link_agent import split_paragraphs

    blocks = split_paragraphs(text)
    trimmed = False
    while len(blocks) > 1 and _count_body_words("\n\n".join(blocks)) > hard_maximum:
        blocks.pop()
        trimmed = True

    if not trimmed:
        return text, citation_links, uncited_sentences, False

    kept_count = len(blocks)
    new_text = "\n\n".join(blocks)
    new_citation_links = [
        link
        for link in (citation_links or [])
        if not (
            isinstance(link.get("paragraph_index"), int)
            and link["paragraph_index"] >= kept_count
        )
    ]
    new_uncited_sentences = (
        [
            entry
            for entry in uncited_sentences
            if not (
                isinstance(entry.get("paragraph_index"), int)
                and entry["paragraph_index"] >= kept_count
            )
        ]
        if uncited_sentences is not None
        else None
    )
    return new_text, new_citation_links, new_uncited_sentences, True


async def call_deepseek(system_prompt: str, user_prompt: str) -> WritingCompletion:
    """Call the DeepSeek chat API; return the generated text and the call's provenance.

    ``provenance`` is an ``LLMCallProvenance`` dump (agent ``"writing"``, the configured
    model alias, the ``model`` the provider reports, temperature, prompt version, token
    usage) plus ``max_tokens``, ``thinking`` and, when the response's usage carries it,
    ``reasoning_tokens``. The model sent is ``settings.writing_model`` when set,
    otherwise ``settings.deepseek_model``: this writer-only override lets the
    drafting call use a different DeepSeek model than every other agent, which all keep
    reading ``settings.deepseek_model`` directly.

    ``settings.writing_max_tokens`` (default 4096, the fixed value every writing call
    sent before this setting existed) is always sent as ``max_tokens``. Both are
    writer-only, read nowhere else.

    ``settings.writing_thinking`` ("enabled" or "disabled") is sent as the request's
    ``thinking`` field when set; ``None`` (the default, unset ``WRITING_THINKING``)
    does not mean the request carries no ``thinking`` field at all -- it is sent as
    ``{"type": "disabled"}``, the same shape ``app.agents.model_config``'s shared
    ``ModelSettings`` tiers set via ``extra_body`` for every other agent. This raw
    call builds its own request body rather than going through pydantic-ai, so it
    gets none of that shared wiring's protection on its own; without an explicit
    default here, moving ``settings.deepseek_model``/``settings.writing_model`` from
    the retired ``deepseek-chat`` alias to a model that defaults to thinking mode ON
    (``deepseek-flash``) would silently turn thinking on for every writer call that
    never set ``WRITING_THINKING``, while every frozen writer run and the promoted
    demo ran with it off.

    The visible content returned is always ``message.content``; the chain-of-thought
    text a thinking-mode response carries in ``message.reasoning_content`` is never
    read here and never stored.
    """
    if not settings.deepseek_api_key:
        raise ValueError("DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env")
    model = settings.writing_model or settings.deepseek_model
    max_tokens = settings.writing_max_tokens
    thinking_type = settings.writing_thinking or "disabled"
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": WRITING_TEMPERATURE,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking_type},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    record = LLMCallProvenance(
        agent="writing",
        model_configured=model,
        model_reported=data.get("model"),
        provider_response_id=data.get("id"),
        system_fingerprint=data.get("system_fingerprint"),
        temperature=WRITING_TEMPERATURE,
        prompt_version=prompt_version(system_prompt),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )
    provenance = {
        **record.model_dump(mode="json"),
        "max_tokens": max_tokens,
        "thinking": thinking_type,
    }
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning_tokens is not None:
        provenance["reasoning_tokens"] = reasoning_tokens
    return WritingCompletion(content=content, provenance=provenance)


async def build_citation_link_map(
    content: str, paper_lookup: dict, evidence_catalog: list[dict] | None = None
) -> CitationLinkResult:
    """Second structured call over already-generated text.

    ``paper_lookup`` is the same ``surname_year``-keyed dict
    ``app.services.fulltext._build_paper_lookup`` builds; only its keys are shown to the
    model, and only as a hint of which citations are in this project's library, never as
    something the returned map is filtered against. ``evidence_catalog`` is the same catalog
    ``app.services.writing._build_evidence_context`` returned when building the writer's
    context, so the linker can report, per citation, which evidence ids its proposition
    restates. Any exception (provider outage, malformed structured response) is left to
    the caller's own try/except, mirroring the citation-audit call just above it in
    ``generate_section``. Returns the validated map alongside the call's own
    ``LLMCallProvenance``: this is a second, real DeepSeek call, and its tokens
    are recorded here.
    """
    return await link_citations(content, list(paper_lookup.keys()), evidence_catalog)


async def _close_citation_link_coverage_gap(
    generated_text: str,
    citation_links: list[dict],
    uncited_sentences: list[dict],
    paper_lookup: dict,
    evidence_catalog: list[dict] | None,
    provenance: dict,
) -> tuple[list[dict], list[dict], dict, bool]:
    """Run ``app.services.fulltext.resolve_unclassified_sentences`` over one citation-link
    call's own results, using this same section's `build_citation_link_map` as the
    bounded retry's own linker (so a sentence it resolves is validated exactly as the
    first call's own results were), and fold the retry call's own provenance into
    *provenance* exactly as every other citation-link call already is
    (`_with_citation_link_provenance`).

    Called right after EVERY citation-link call the gated loop makes (the section's
    first pass and, when the gate triggers a revision, its second pass too) rather than
    once before finalize: applying it here means a sentence the retry resolves into a
    genuine citation link is turned into a claim by `claims_from_citation_links` exactly
    like any other and reaches the verification gate below in the normal way, instead of
    being kept or dropped by a special case of its own. A sentence the retry call
    itself failed to reach becomes a synthetic uncited "finding" entry marked
    ``"unclassified": True``, which needs no verification at all; that flag is what
    stops finalize's own rule 1 from removing it -- it is
    kept instead, exactly like a framing-tagged sentence, and counted under
    ``sentences_unclassified_kept``. A sentence the retry call answered but never
    mentioned becomes an ordinary, removable "finding" entry instead, with no
    ``"unclassified"`` flag, so rule 1 removes it and counts it under
    ``sentences_removed_uncited_finding``.

    Returns the merged links and uncited sentences, *provenance* (updated when a retry
    call was made), and whether the retry call itself failed -- the caller folds this
    into ``loop_stats["link_coverage_retry_failed"]``.
    """
    from app.services.fulltext import resolve_unclassified_sentences

    async def _retry_linker(retry_text: str) -> CitationLinkResult:
        return await build_citation_link_map(retry_text, paper_lookup, evidence_catalog)

    new_links, new_uncited, retry_provenance, calls_made, retry_failed = (
        await resolve_unclassified_sentences(
            generated_text, citation_links, uncited_sentences, _retry_linker
        )
    )
    if calls_made:
        provenance = _with_citation_link_provenance(provenance, retry_provenance)
    return new_links, new_uncited, provenance, retry_failed


def _sum_tokens(*values: int | None) -> int | None:
    """Sum the non-``None`` values in *values*, or ``None`` when none are present.
    Shared by every provenance-folding function below so a section's running totals
    never silently turn into 0 just because one call reported no usage."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _with_analysis_provenance(provenance: dict, analysis_calls: list[dict]) -> dict:
    """Fold every full-text grounding analysis call into the section's
    provenance dict, exactly as ``_with_citation_link_provenance`` (below) folds the
    citation-link call: the existing top-level fields keep describing the generation
    call alone, and the analysis calls' own records plus the running totals are added
    under new keys. Composes with ``_with_citation_link_provenance`` regardless of which
    of the two runs first, since both read the running totals already in *provenance*
    (defaulting to the generation call's own numbers) rather than assuming they are the
    first fold to run. ``analysis_calls`` is ``[]`` when no paper needed grounding, so
    ``provenance["analysis_calls"]`` is always a list, never missing."""
    provenance = dict(provenance)
    provenance["analysis_calls"] = analysis_calls
    base_calls = provenance.get("total_calls", 1)
    base_input = provenance.get("total_input_tokens", provenance.get("input_tokens"))
    base_output = provenance.get("total_output_tokens", provenance.get("output_tokens"))
    provenance["total_calls"] = base_calls + len(analysis_calls)
    provenance["total_input_tokens"] = _sum_tokens(
        base_input, *(c.get("input_tokens") for c in analysis_calls)
    )
    provenance["total_output_tokens"] = _sum_tokens(
        base_output, *(c.get("output_tokens") for c in analysis_calls)
    )
    return provenance


def _with_citation_link_provenance(
    provenance: dict, link_provenance: LLMCallProvenance | None
) -> dict:
    """Fold the citation-link call's own provenance into the section's provenance dict,
    so a section's totals cover
    both DeepSeek calls it makes, not only the generation call.

    ``provenance``'s existing top-level fields (``model_configured``, ``input_tokens``,
    ...) keep describing the generation call alone, unchanged, since every existing
    reader (``demo/run_demo.py``, the frontend provenance line) already treats them that
    way; the link call's own record and the combined totals are added under new keys
    instead. The running totals default to the generation call's own numbers, but read
    whatever ``_with_analysis_provenance`` already folded in first when grounding
    analysed at least one paper, so the two folds compose in either
    order. ``link_provenance`` is ``None`` when the link call itself failed (a provider
    outage, a malformed structured response), in which case the totals are unchanged and
    ``citation_link_call`` is ``None``, so a reader can always look at these three keys
    rather than checking which ones exist.

    The gated write loop can call this
    function a second time, from its own revision pass, after a first citation-link
    call already succeeded. ``citation_link_call`` is kept as the most recent call's own
    record only, for every existing reader of that single key; every call's record
    (including a ``None`` for one that failed) is additionally appended, in call order,
    to ``citation_link_calls``, the same list shape ``_with_verification_provenance``
    already uses for ``verification_calls`` -- so a revision's second call can no longer
    silently destroy the first call's response id, fingerprint and token split."""

    provenance = dict(provenance)
    dumped_call = (
        link_provenance.model_dump(mode="json") if link_provenance is not None else None
    )
    provenance["citation_link_call"] = dumped_call
    calls_list = list(provenance.get("citation_link_calls") or [])
    calls_list.append(dumped_call)
    provenance["citation_link_calls"] = calls_list
    base_calls = provenance.get("total_calls", 1)
    base_input = provenance.get("total_input_tokens", provenance.get("input_tokens"))
    base_output = provenance.get("total_output_tokens", provenance.get("output_tokens"))
    provenance["total_calls"] = base_calls + (1 if link_provenance is not None else 0)
    provenance["total_input_tokens"] = _sum_tokens(
        base_input,
        link_provenance.input_tokens if link_provenance is not None else None,
    )
    provenance["total_output_tokens"] = _sum_tokens(
        base_output,
        link_provenance.output_tokens if link_provenance is not None else None,
    )
    return provenance


def _with_verification_provenance(provenance: dict, verify_provenance: dict | None) -> dict:
    """Fold one `_verify_claims` aggregate provenance record into the section's own
    provenance dict, the same running-totals pattern
    `_with_citation_link_provenance` uses. Called once for the first verification pass
    and again for a revision's own (smaller) re-verify pass, so
    ``provenance["verification_calls"]`` is a list with one entry per pass, in order,
    alongside the running ``total_calls``/``total_input_tokens``/``total_output_tokens``.
    ``verify_provenance`` with zero calls (every claim resolved deterministically, no
    full-text chunks at all) is still appended, for a complete trace, but never changes
    the running totals (`_sum_tokens` treats its own ``None`` token fields as absent)."""
    provenance = dict(provenance)
    calls_list = list(provenance.get("verification_calls") or [])
    calls_list.append(verify_provenance)
    provenance["verification_calls"] = calls_list
    base_calls = provenance.get("total_calls", 1)
    base_input = provenance.get("total_input_tokens", provenance.get("input_tokens"))
    base_output = provenance.get("total_output_tokens", provenance.get("output_tokens"))
    added_calls = verify_provenance.get("calls") or 0
    provenance["total_calls"] = base_calls + added_calls
    provenance["total_input_tokens"] = _sum_tokens(
        base_input, verify_provenance.get("input_tokens")
    )
    provenance["total_output_tokens"] = _sum_tokens(
        base_output, verify_provenance.get("output_tokens")
    )
    return provenance


#: A markdown heading line at the start of a finalize/generation "paragraph" block
#: (`app.agents.citation_link_agent.split_paragraphs`'s own blank-line blocks), for
#: `_build_section_tiptap_nodes` to render as a real Tiptap ``heading`` node rather than
#: a paragraph carrying literal "#" characters.
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")

#: A whole paragraph block that is nothing but one bold run, start to end, with no other
#: ``*`` character anywhere in it: the writer
#: sometimes emits a sub-heading as ``**Sub-Heading Text**`` on its own line instead of a
#: markdown ``###`` line for the same structural role. Matched narrowly -- a line with more
#: than one bold run (``**A** and **B**``) is running prose, not a heading, and must not
#: match -- so only the exact shape a lone bold-line sub-heading takes is affected; every
#: other use of ``**bold**`` inside running text is untouched (see the function's own
#: docstring below).
_MARKDOWN_BOLD_HEADING_RE = re.compile(r"^\*\*([^*]+)\*\*$")

def _strip_duplicate_leading_heading(
    text: str, title: str, section_type_default: str | None = None
) -> str:
    """Drop every leading markdown heading line or whole-line bold run the model
    writes before its own real body. The saved section already opens with its own
    H2 heading node carrying *title* (`_build_section_tiptap_nodes`); the section
    title is the only heading a saved section ever carries, so ANY model-written
    heading line or bold run as the very first block of the generated body --
    whether it repeats *title* or *section_type_default*, or is worded differently
    again -- is dropped whole, whatever it says. Left in place, it would become a
    second heading node once the text is split into paragraphs and rendered
    (`_MARKDOWN_HEADING_RE`/`_MARKDOWN_BOLD_HEADING_RE` in
    `_build_section_tiptap_nodes`).
    `app.services.fulltext.finalize_generated_section` makes the identical,
    unconditional drop for every OTHER heading the model writes further into the
    body, once finalize's own removals are known; this is the early half of the
    same rule, run right after generation, before the citation-link call or the
    word count ever see the text.

    Loops over every CONSECUTIVE leading heading block, dropping each one and
    re-examining the new first block, so a run of several leading headings in a row
    -- for example ``# Literature Review`` directly followed by ``### Effects of
    Feedback Form on Writing Accuracy`` -- is fully unwound in one pass rather than
    leaving the second one standing as a second heading node beside the section's
    own injected H2. Stops the moment the current first block is not itself
    heading-shaped: a genuine paragraph is always left untouched, whether it is the
    original first block or one reached only after some number of leading headings
    were dropped.

    *title* and *section_type_default* are accepted for call-site compatibility (a
    caller resolving one of the two into the section's own saved title) but no
    longer change the outcome: every leading heading is dropped regardless of its
    own wording, so nothing about this function's decision depends on either value.
    Applied right after every fresh generation call (initial, the length
    hard-maximum regeneration, and the gated loop's own revision), before the
    citation-link call, the word count, or finalize ever see the text, so none of
    them have to reason about a leading heading block."""
    from app.agents.citation_link_agent import split_paragraphs

    blocks = split_paragraphs(text)
    if not blocks:
        return text
    while blocks:
        first_block = blocks[0].strip()
        is_heading = bool(
            _MARKDOWN_HEADING_RE.match(first_block)
            or _MARKDOWN_BOLD_HEADING_RE.match(first_block)
        )
        if not is_heading:
            # A genuine paragraph: left exactly as it stands. The loop stops
            # here; nothing further down *blocks* is inspected.
            break
        blocks = blocks[1:]
    return "\n\n".join(blocks)


#: One inline emphasis run, ``**bold**`` or ``*italic*``, on a single line
#: -- never spanning a blank-line block boundary, so a lone,
#: unmatched ``*`` used as a multiplication sign or a list bullet is never touched (it
#: has no closing partner on the same line).
_INLINE_EMPHASIS_RUN_RE = re.compile(r"\*{1,2}([^*\n]+?)\*{1,2}")


def _strip_inline_emphasis(text: str) -> str:
    """Remove markdown emphasis markers from every block of *text* that will end up as
    a plain paragraph node: ``_build_section_tiptap_nodes``
    does not convert inline ``**bold**``/``*italic*`` runs to a real Tiptap mark (an
    accepted, documented limitation -- see its own docstring), so a literal
    ``(*d* = 1.32)`` the model wrote inside running prose reached the saved draft with
    its asterisks intact and no italics applied.

    Applied once, right after `_strip_duplicate_leading_heading`, on the SAME text
    every downstream step (the citation-link call, verification, and
    `_build_section_tiptap_nodes` itself) reads from here on -- not only at save time --
    so a claim's own ``sentence``/``claim_text`` and the text the saved draft actually
    carries stay byte-identical. Stripping only when the paragraph node is built, after
    citations were already linked and claims verified against the model's original
    (still-emphasised) sentence text, would break the exit invariant that every
    verified claim's own text occurs verbatim in the saved draft.

    A block that will be promoted to a heading node -- a markdown ``#`` line, or a
    whole-line bold run that still reads as a sub-heading
    (`app.services.fulltext._is_bold_heading_candidate`) -- is left untouched: its own
    markers are what `_build_section_tiptap_nodes` (and, for the very first block,
    `_strip_duplicate_leading_heading`, which already ran before this on the
    unstripped text) matches on to recognise it as a heading at all."""
    from app.agents.citation_link_agent import split_paragraphs
    from app.services.fulltext import _is_bold_heading_candidate

    blocks = split_paragraphs(text)
    if not blocks:
        return text
    cleaned_blocks = []
    for block in blocks:
        stripped_block = block.strip()
        if _MARKDOWN_HEADING_RE.match(stripped_block):
            cleaned_blocks.append(block)
            continue
        bold_heading_match = _MARKDOWN_BOLD_HEADING_RE.match(stripped_block)
        if bold_heading_match and _is_bold_heading_candidate(bold_heading_match.group(1)):
            cleaned_blocks.append(block)
            continue
        cleaned_blocks.append(_INLINE_EMPHASIS_RUN_RE.sub(r"\1", block))
    return "\n\n".join(cleaned_blocks)


def _normalize_generated_text(
    text: str, title: str, section_type_default: str | None = None
) -> str:
    """The two text-shape fixes every fresh generation call needs before anything else
    (the citation-link call, the word count, or finalize) ever sees the text: drop a
    duplicated leading heading (`_strip_duplicate_leading_heading`), then strip inline
    markdown emphasis from every block that is not itself a heading
    (`_strip_inline_emphasis`). Order matters -- emphasis-stripping runs second because
    it must not touch the FIRST block until heading-detection has already had its own
    unmodified look at it."""
    return _strip_inline_emphasis(
        _strip_duplicate_leading_heading(text, title, section_type_default)
    )


def _build_section_tiptap_nodes(
    section_type: str,
    title: str,
    text: str,
    citation_links: list[dict],
    uncited_sentences: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict]:
    """One finalized section's own Tiptap nodes: an H2
    heading tagged ``attrs.sectionType`` (so a later regeneration of this same section
    can find and replace exactly this block, `_merge_section_into_draft`), then one node
    per ``split_paragraphs(text)`` block -- a real ``heading`` node for a markdown
    heading line the model wrote inside its own body, a ``paragraph`` node otherwise,
    carrying ``attrs.citationLinks`` grouped by the link's own ``paragraph_index``
    (an index into that same block list, exactly as the citation-link agent's own
    prompt defines it).

    Each stored link also carries ``proposition``: the narrowed,
    verbatim span of the sentence that citation actually supports, exactly as
    `extract_claims_from_document` (fulltext.py) reads it back on the saved-draft path
    -- the standalone action, the demo runner and ``claim_report.json`` all use that
    function, not the write job's own in-memory claim list. Without this field, a
    grouped-citation sentence round-trips through the saved draft and is sent to the
    verifier whole, once per citation, instead of narrowed to the one fact each
    citation supports.

    *uncited_sentences* -- `app.services.fulltext.FinalizeResult.uncited_sentences`, the
    surviving framing/still-unresolved entries finalize already recomputed against this
    same *text*'s own blocks -- is persisted onto ``attrs.uncitedSentences``, grouped by
    ``paragraph_index`` exactly as ``citation_links`` is.
    Without persisting this attribute, every framing sentence
    the write job's own linker had already classified would read, on every future heal, as a
    fresh, unclassified coverage gap (`app.services.fulltext.unclassified_body_sentences`)
    -- and a coverage-gap retry call that then failed, or omitted one of those already-
    classified sentences, would remove it from a real user's saved draft on the "verify and
    heal" action, never having been given the chance to see it was already accounted
    for. Persisting it here means a later heal only ever sends a genuinely new gap back
    to the retry.

    A full markdown-to-Tiptap conversion (bold/italic marks, bullet lists, tables) is
    out of this feature's scope; a block that is not a markdown heading line, and not a
    whole-line bold run that still reads as a sub-heading (``_MARKDOWN_BOLD_HEADING_RE``
    plus `app.services.fulltext._is_bold_heading_candidate` -- the writer's other way of marking a
    sub-heading, rendered as a level-3 heading, one level under this section's own H2,
    UNLESS it carries a citation or ends in sentence punctuation, in which case it is a
    claim sentence the writer happened to bold, not a heading) always becomes one plain
    paragraph node, markdown syntax such as ``**bold**`` inside running prose included
    verbatim as text. This remains an accepted, documented limitation for inline bold,
    not a silent data loss: no text is dropped, only left unstyled.

    The same predicate `app.services.fulltext._is_heading_block` uses for the identical
    shape, imported rather than duplicated: this module already
    imports private helpers from `app.services.fulltext` elsewhere
    (`resolve_unclassified_sentences`, `_build_paper_lookup`), never the other way round,
    so a citation-bearing or sentence-ending bold block is exempted from the heading
    promotion here in exactly the same cases finalize exempts it from every removal and
    coverage rule -- one condition, not two copies that could drift apart."""
    from app.agents.citation_link_agent import split_paragraphs
    from app.services.fulltext import _is_bold_heading_candidate

    nodes: list[dict] = [
        {
            "type": "heading",
            "attrs": {"level": 2, "sectionType": section_type},
            "content": [{"type": "text", "text": title}],
        }
    ]
    links_by_paragraph: dict[int, list[dict]] = {}
    for link in citation_links or []:
        index = link.get("paragraph_index")
        if isinstance(index, int):
            links_by_paragraph.setdefault(index, []).append(
                {
                    "sentence": link.get("sentence"),
                    "keys": link.get("keys") or [],
                    "citation_text": link.get("citation_text"),
                    "evidence_ids": link.get("evidence_ids") or [],
                    "proposition": link.get("proposition"),
                }
            )
    uncited_by_paragraph: dict[int, list[dict]] = {}
    for entry in uncited_sentences or []:
        index = entry.get("paragraph_index")
        if isinstance(index, int):
            uncited_by_paragraph.setdefault(index, []).append(
                {
                    "sentence": entry.get("sentence"),
                    "tag": entry.get("tag"),
                    **({"unclassified": True} if entry.get("unclassified") else {}),
                }
            )

    for i, block in enumerate(split_paragraphs(text)):
        stripped_block = block.strip()
        heading_match = _MARKDOWN_HEADING_RE.match(stripped_block)
        if heading_match:
            nodes.append({
                "type": "heading",
                "attrs": {"level": len(heading_match.group(1))},
                "content": [{"type": "text", "text": heading_match.group(2).strip()}],
            })
            continue
        bold_heading_match = _MARKDOWN_BOLD_HEADING_RE.match(stripped_block)
        if bold_heading_match and _is_bold_heading_candidate(bold_heading_match.group(1)):
            nodes.append({
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": bold_heading_match.group(1).strip()}],
            })
            continue
        node: dict = {"type": "paragraph", "content": [{"type": "text", "text": block}]}
        node_attrs: dict[str, Any] = {}
        block_links = links_by_paragraph.get(i)
        if block_links:
            node_attrs["citationLinks"] = block_links
        block_uncited = uncited_by_paragraph.get(i)
        if block_uncited:
            node_attrs["uncitedSentences"] = block_uncited
        if node_attrs:
            node["attrs"] = node_attrs
        nodes.append(node)
    return nodes


def _is_heading_at_or_above(node: dict, level: int) -> bool:
    """True when *node* is a heading node -- of any kind, ``attrs.sectionType``-tagged
    or not -- at *level* or higher (a smaller or equal Tiptap heading level number):
    the boundary `_merge_section_into_draft` stops a section replacement at.
    A heading with no ``attrs.level`` at all (a user-typed heading, or one
    written before section tagging existed) defaults to level 1, so it always counts
    as a boundary rather than being silently swallowed into whatever section happens
    to precede it."""
    if node.get("type") != "heading":
        return False
    node_level = (node.get("attrs") or {}).get("level", 1)
    return node_level <= level


def _merge_section_into_draft(existing_content: dict | None, section_type: str, section_nodes: list[dict]) -> dict:
    """Splice *section_nodes* into *existing_content*: a section
    already present (a heading node whose own ``attrs.sectionType`` matches) is replaced
    in place, from its own heading up to (not including) the next heading at or above
    that heading's own level -- of any kind, `_is_heading_at_or_above`, not only one this
    feature's own code tagged with ``attrs.sectionType`` (a user-typed
    heading, a section written before section tagging existed, or a hand-written
    References list, must never be silently deleted just because it carries no tag) --
    or the end of the document; a section not yet present is appended. An empty or
    missing draft becomes exactly *section_nodes*. Every other section, and any content
    outside section boundaries entirely (this draft's own manual edits before section
    tagging existed), is left untouched. A heading nested more deeply than the replaced
    section's own heading is still treated as part of that section's own content, not as
    a boundary. A second heading tagged with the same ``section_type`` -- a shape this
    backend never creates on its own, but a hand-edited or imported draft could -- is
    dropped along with its own content rather than given a second copy of
    *section_nodes*: only the first such
    heading is ever replaced."""
    existing_nodes = list(((existing_content or {}).get("content")) or [])
    new_nodes: list[dict] = []
    replaced = False
    i = 0
    while i < len(existing_nodes):
        node = existing_nodes[i]
        node_attrs = node.get("attrs") or {}
        if node.get("type") == "heading" and node_attrs.get("sectionType") == section_type:
            section_level = node_attrs.get("level", 2)
            # Only the FIRST heading tagged with this
            # section_type is replaced with *section_nodes* -- a later one (a shape the
            # backend never creates itself, but a hand-edited or imported draft could)
            # is dropped along with its own content, not given a second copy of the
            # freshly generated section.
            if not replaced:
                new_nodes.extend(section_nodes)
                replaced = True
            i += 1
            while i < len(existing_nodes) and not _is_heading_at_or_above(
                existing_nodes[i], section_level
            ):
                i += 1
            continue
        new_nodes.append(node)
        i += 1
    if not replaced:
        new_nodes.extend(section_nodes)
    return {"type": "doc", "content": new_nodes}


def _normalize_ws(text: str) -> str:
    """Collapse whitespace runs to one space. Mirrors
    `app.services.fulltext._doc_normalize_ws` exactly; kept as a separate, tiny
    function here rather than imported, since it is a one-line pure string operation
    and this module does not otherwise depend on that one."""
    return " ".join((text or "").split())


def _strip_locked_block_findings(
    uncited_sentences: Sequence[Mapping[str, Any]] | None,
    locked_blocks: Sequence[Mapping[str, Any]] | None,
) -> list[dict]:
    """Drop any uncited-sentence entry whose own text is reproduced from a locked
    block, before finalize's rule 1 (design amendment A4) ever sees it. ``locked_blocks`` text is reproduced by the model verbatim
    inside the generated section, and is then subject to the same uncited-"finding"
    removal rule as anything else the model wrote; a user's own locked paragraph that
    states an uncited fact must never be removed just because it carries no citation
    -- the user wrote it, not the model, and finalize has no other way to tell the two
    apart once the model has echoed the text back.

    The frontend does not send ``locked_blocks`` today, so this exemption is inert in
    production; it is real code, not a placeholder, for the day it does."""
    kept = list(uncited_sentences or [])
    locked_norms = [
        norm
        for norm in (_normalize_ws(block.get("text") or "") for block in (locked_blocks or []))
        if norm
    ]
    if not locked_norms:
        return kept
    return [
        entry
        for entry in kept
        if not (
            (sentence_norm := _normalize_ws(entry.get("sentence") or ""))
            and any(sentence_norm in locked for locked in locked_norms)
        )
    ]


class GroundingResult(NamedTuple):
    """What one grounding pass over a section's selected papers produced:
    one ``LLMCallProvenance`` dump per paper actually analysed, in call order, folded
    into the section's provenance exactly as the citation-link call is folded, plus the
    ``analysis`` node metrics recorded on the write job's own result."""

    analysis_calls: list[dict]
    metrics: dict


def _verbatim_rate(evidence_items: int, rejected_items: int) -> float:
    """``evidence_items / (evidence_items + rejected_items)``, or ``1.0`` when neither
    paper produced or rejected any item at all -- vacuously true, and never a
    divide-by-zero."""
    total = evidence_items + rejected_items
    return (evidence_items / total) if total else 1.0


async def _ground_selected_papers(
    selected_papers: list,
    session_factory,
    job_id: UUID,
    expertise_level: str | None,
) -> GroundingResult:
    """Ensure every one of *selected_papers* that has full text but no full-text
    analysis yet is analysed before the section context is built.

    A paper qualifies when its metadata carries ``fulltext_chunks`` but
    ``is_full_text_analysis`` says its stored ``deep_analysis`` (if any) is only the
    abstract-only placeholder, or is missing altogether. It is analysed with the
    existing full-text analysis agent -- ``analyze_paper_text_with_provenance``, reused
    (not duplicated) from ``app.agents.deep_analysis_agent`` exactly as
    ``app.services.deep_analysis.analyze_all_papers`` reuses ``analyze_paper_text`` --
    and the result is persisted as that paper's current ``deep_analysis``, so later
    sections and paper chat see it too and the same paper is never re-analysed on a
    later call. The in-memory ``ProjectPaper.paper.metadata_`` is also updated in place,
    so the context-building call right after this one sees the fresh analysis without a
    re-fetch.

    The analysis's proposed evidence items are validated and stored exactly as
    ``app.services.deep_analysis.analyze_all_papers`` stores them (the same
    ``validate_evidence_items``/``replace_paper_evidence`` helpers, not duplicated),
    replacing that paper's earlier evidence rows in the same transaction as the
    ``deep_analysis`` metadata write, so the two are never out of step.

    A single paper's analysis failing (provider outage, malformed structured response)
    or its persistence failing (a stale row, a serialisation error) is logged and
    skipped -- exactly the fault-tolerance ``analyze_all_papers`` already applies per
    paper -- so it never fails the whole writing job; that paper is simply presented
    under its abstract, as if it had no full text.
    """
    from app.agents.deep_analysis_agent import analyze_paper_text_with_provenance
    from app.models.paper import Paper
    from app.services.deep_analysis import replace_paper_evidence, validate_evidence_items

    to_analyze = []
    for pp in selected_papers:
        p = pp.paper
        meta = p.metadata_ or {}
        chunks = meta.get("fulltext_chunks") or []
        if chunks and not is_full_text_analysis(meta.get("deep_analysis")):
            to_analyze.append(p)

    analysis_calls: list[dict] = []
    papers_analyzed = 0
    evidence_items_total = 0
    rejected_items_total = 0
    for i, p in enumerate(to_analyze, 1):
        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id,
                progress=0.35,
                progress_message=(
                    f"Grounding paper {i}/{len(to_analyze)} in its full text: "
                    f"{(p.title or '(untitled)')[:60]}"
                ),
                status=JobStatus.running,
            )

        meta = p.metadata_ or {}
        chunks = meta.get("fulltext_chunks") or []
        authors_str = format_authors(p.authors or [])
        try:
            result = await analyze_paper_text_with_provenance(
                p.title or "", authors_str, p.year, chunks,
                expertise_level=expertise_level,
            )
        except Exception:
            logger.warning(
                "Full-text grounding analysis failed for paper %s", p.id, exc_info=True
            )
            continue

        analysis_dict = result.output.model_dump()
        analysis_calls.append(result.provenance.model_dump(mode="json"))
        raw_evidence = analysis_dict.pop("evidence", [])
        evidence_rows, rejected = validate_evidence_items(raw_evidence, chunks)

        # Counted only once the write below actually
        # commits, not right after the analysis call -- a paper whose persistence
        # fails (a stale row, a serialisation error, ``fresh is None``) is caught and
        # skipped exactly like an analysis failure, so it must land in
        # ``papers_failed`` below, and its evidence rows (already validated, but never
        # stored) must not inflate ``evidence_items``/``rejected_items``.
        try:
            async with session_factory() as bg_db:
                fresh = await bg_db.get(Paper, p.id)
                if fresh is not None:
                    new_meta = dict(fresh.metadata_ or {})
                    new_meta["deep_analysis"] = analysis_dict
                    fresh.metadata_ = new_meta
                    await replace_paper_evidence(
                        bg_db, p.id, evidence_rows, result.provenance.prompt_version
                    )
                    await bg_db.commit()
                    papers_analyzed += 1
                    evidence_items_total += len(evidence_rows)
                    rejected_items_total += rejected
        except Exception:
            logger.warning(
                "Could not persist grounding analysis for paper %s", p.id, exc_info=True
            )

        # Update the in-memory object this same request builds context from, whether or
        # not the persistence above succeeded.
        p.metadata_ = {**(p.metadata_ or {}), "deep_analysis": analysis_dict}

    metrics = {
        "papers": papers_analyzed,
        # ``papers`` alone cannot tell a reader whether
        # every paper that needed grounding got it, since a single paper's analysis
        # failing (provider outage, a token-limit error) is caught above and simply
        # skipped. ``papers_attempted`` is the size of ``to_analyze`` before any of them
        # ran; ``papers_failed`` is the gap between the two numbers.
        "papers_attempted": len(to_analyze),
        "papers_failed": len(to_analyze) - papers_analyzed,
        "evidence_items": evidence_items_total,
        "rejected_items": rejected_items_total,
        "verbatim_rate": _verbatim_rate(evidence_items_total, rejected_items_total),
    }
    return GroundingResult(analysis_calls=analysis_calls, metrics=metrics)


async def generate_section(
    draft: Draft,
    section_type: str,
    context: str | None,
    job_id: UUID,
    session_factory,
    language: str = "en",
    locked_blocks: list | None = None,
    expertise_level: str | None = None,
    target_words: int | None = None,
    title: str | None = None,
) -> None:
    """Background task: generate a section using the three-pass writing flow.

    Pass 1: Load all project papers
    Pass 2: Select relevant papers for this section type (via paper selector agent)
    Pass 3: Build deep-analysis context and generate with DeepSeek

    ``target_words``, when given, is sent to the model as a target
    length plus a hard maximum 25% over it; a body (headings excluded) still over the
    hard maximum after generation is regenerated once, naming the previous count, and
    ``provenance["length"]`` records the outcome. ``None`` runs generation with no
    length control at all.

    ``title``, when given, is used verbatim as this
    section's own saved heading and as the text `_strip_duplicate_leading_heading`
    compares the model's own opening heading line against. ``None`` derives it
    from *section_type* -- ``section_type.replace("_", " ").title()``.
    """
    try:
        # --- Pass 1: Load papers ---
        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id,
                status=JobStatus.running,
                progress=0.1,
                progress_message="Loading paper library...",
            )

        async with session_factory() as bg_db:
            result = await bg_db.execute(
                select(ProjectPaper)
                .options(joinedload(ProjectPaper.paper))
                .where(ProjectPaper.project_id == draft.project_id)
            )
            project_papers = list(result.scalars().unique().all())

        # Determine what this section needs
        needs_papers = section_type in (
            "introduction", "literature_review", "discussion",
            "implications", "conclusion", "abstract",
        )
        needs_data = section_type in (
            "methods", "results", "discussion",
            "implications", "conclusion", "abstract",
        )

        selected_papers = project_papers  # default for papers_used count
        analysis_calls: list[dict] = []
        analysis_metrics = {
            "papers": 0, "papers_attempted": 0, "papers_failed": 0,
            "evidence_items": 0, "rejected_items": 0, "verbatim_rate": 1.0,
        }
        evidence_catalog: list[dict] = []

        papers_context = ""
        if needs_papers and project_papers:
            # --- Pass 2: Select relevant papers ---
            async with session_factory() as bg_db:
                await task_service.update_job_status(
                    bg_db, job_id,
                    progress=0.3,
                    progress_message=f"Selecting relevant papers for {section_type}...",
                    status=JobStatus.running,
                )

            paper_summaries = []
            for pp in project_papers:
                p = pp.paper
                metadata = p.metadata_ or {}
                analysis = metadata.get("deep_analysis", {})
                themes = analysis.get("themes", []) if isinstance(analysis, dict) else []
                paper_summaries.append({
                    "id": str(p.id),
                    "title": p.title,
                    "year": p.year,
                    "themes": ", ".join(themes) if themes else "N/A",
                })

            from app.agents.paper_selector_agent import select_papers_for_section
            selected_ids = await select_papers_for_section(section_type, paper_summaries)

            # Filter to selected papers
            selected_papers = [
                pp for pp in project_papers if str(pp.paper.id) in selected_ids
            ]
            if not selected_papers:
                selected_papers = project_papers  # Fallback: use all

            # --- Pass 3: Ground every selected paper that has full text but no
            # full-text analysis yet, storing its accepted evidence,
            # then build the writer's context from that evidence. ---
            grounding = await _ground_selected_papers(
                selected_papers, session_factory, job_id, expertise_level
            )
            analysis_calls = grounding.analysis_calls
            analysis_metrics = grounding.metrics
            evidence_context = await _build_evidence_context(selected_papers, session_factory)
            papers_context = evidence_context.text
            evidence_catalog = evidence_context.catalog

        data_context = ""
        if needs_data:
            async with session_factory() as bg_db:
                await task_service.update_job_status(
                    bg_db, job_id,
                    progress=0.5,
                    progress_message="Loading research data...",
                    status=JobStatus.running,
                )
            data_context = await _build_data_context(draft.project_id, session_factory)

        # Load journal guidelines and merge with golden standard via AI
        guidelines = await _load_journal_guidelines(draft.project_id, session_factory)

        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id,
                progress=0.55,
                progress_message="Merging journal rules with writing standards...",
                status=JobStatus.running,
            )

        # AI-powered merge of golden standard + journal guidelines
        from app.agents.writing_rules_agent import merge_writing_rules
        try:
            merged_rules = await merge_writing_rules(
                section_type=section_type,
                paper_type=draft.paper_type.value if hasattr(draft.paper_type, 'value') else str(draft.paper_type),
                journal_guidelines=guidelines,
            )
            # merged_rules.citation_rules is deliberately never appended here:
            # _BASE_RULES already
            # fixes in-text citations to author-year, and appending a second, model-
            # produced citation instruction risked contradicting it whenever a journal
            # guideline named a different style.
            merged_rules_text = (
                f"\n\nUNIFIED WRITING RULES FOR THIS SECTION:\n"
                f"- Structure: {merged_rules.section_structure}\n"
                f"- Word limit: {merged_rules.word_limit}\n"
                f"- Tone/Style: {merged_rules.tone_and_style}\n"
                f"- Content requirements: {merged_rules.content_requirements}\n"
                f"- Formatting: {merged_rules.formatting}\n"
                f"- Special instructions: {merged_rules.special_instructions}\n"
                f"\nYou MUST follow these unified rules exactly.\n"
            )
        except Exception as e:
            logger.warning("Failed to merge writing rules via AI: %s. Falling back.", e)
            # Fallback to simple append
            merged_rules_text = ""
            if guidelines:
                merged_rules_text = "\n\n" + _build_journal_rules(guidelines)

        # Load project research topic (refined topic preferred over title)
        from app.models.project import Project
        project_title = ""
        project_description = ""
        refined_topic = ""
        async with session_factory() as bg_db:
            project = await bg_db.get(Project, draft.project_id)
            if project:
                project_title = project.title or ""
                project_description = project.description or ""
                refined_topic = project.refined_topic or ""

        # Build final prompt
        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id,
                progress=0.6,
                progress_message=f"Writing {section_type}...",
                status=JobStatus.running,
            )

        topic_context = ""
        if refined_topic:
            topic_context = f"Research topic (refined): {refined_topic}\n\n"
        elif project_title:
            topic_context = f"Research topic: {project_title}\n"
            if project_description:
                topic_context += f"Research description: {project_description}\n"
            topic_context += "\n"

        locked_context = ""
        if locked_blocks:
            locked_context = (
                "\n\nIMPORTANT: The following paragraphs were written by the user. "
                "You MUST preserve them EXACTLY as written, in their current positions. "
                "Generate AI content around them.\n\n"
            )
            for block in locked_blocks:
                locked_context += f"[Position {block['position']}] {block['text']}\n\n"

        user_prompt = f"Write a {section_type} section for an academic paper.\n\n{topic_context}"
        if papers_context:
            user_prompt += papers_context + "\n\n"
        if data_context:
            user_prompt += data_context + "\n\n"
        if context:
            user_prompt += f"Additional instructions: {context}\n\n"
        user_prompt += (
            f"Write a comprehensive, well-synthesized {section_type} "
            f"based on the information above. Follow all rules in your system prompt."
        )

        if locked_context:
            user_prompt += locked_context

        hard_maximum = _hard_maximum_words(target_words) if target_words is not None else None
        if target_words is not None:
            user_prompt += f"\n\n{_length_instruction(target_words)}"

        expertise_modifier = get_expertise_prompt(expertise_level)
        expertise_section = f"\n\n{expertise_modifier}" if expertise_modifier else ""
        system_prompt = get_section_prompt(section_type, language) + merged_rules_text + expertise_section

        # The section's own
        # title, computed once and reused everywhere this function needs it -- by
        # `_strip_duplicate_leading_heading` right after every fresh generation call
        # below, and by `_build_section_tiptap_nodes` at the very end -- so the two
        # never compare against two different renderings of the "same" title.
        # The caller's own requested *title*, when given,
        # wins over the section_type-derived default.
        section_title = title or section_type.replace("_", " ").title()
        # The generic default, computed regardless of
        # whether *title* was given, so `_strip_duplicate_leading_heading` also
        # recognises the model's own generic heading even when the caller requested a
        # more specific *title* the user/system prompt above never actually names.
        section_type_default_title = section_type.replace("_", " ").title()

        generated_text, provenance = await call_deepseek(system_prompt, user_prompt)
        generated_text = _normalize_generated_text(
            generated_text, section_title, section_type_default_title
        )

        # A body still over the hard maximum after generation is
        # regenerated once, naming the previous count -- never an unbounded retry loop.
        # Folding both calls' tokens into the running totals here, before
        # ``_with_analysis_provenance``/``_with_citation_link_provenance`` run below,
        # keeps the section's cost reporting from silently missing a whole second paid
        # call, the same concern already addressed for the citation-link call.
        if target_words is not None:
            words = _count_body_words(generated_text)
            # Named distinctly from the gated loop's
            # own ``loop_revised`` below -- the two are independent events (this one
            # fires on a body still over the hard maximum right after generation; that
            # one fires when the verification gate finds a problem later).
            length_regenerated = words > hard_maximum
            if length_regenerated:
                first_provenance = provenance
                regen_prompt = user_prompt + (
                    f"\n\nYour previous draft was about {words} words, over the hard "
                    f"maximum of {hard_maximum} words. Rewrite it so the body text does "
                    f"not exceed {hard_maximum} words."
                )
                generated_text, provenance = await call_deepseek(system_prompt, regen_prompt)
                generated_text = _normalize_generated_text(
                    generated_text, section_title, section_type_default_title
                )
                words = _count_body_words(generated_text)
                provenance["total_calls"] = provenance.get("total_calls", 1) + 1
                provenance["total_input_tokens"] = _sum_tokens(
                    first_provenance.get("input_tokens"), provenance.get("input_tokens")
                )
                provenance["total_output_tokens"] = _sum_tokens(
                    first_provenance.get("output_tokens"), provenance.get("output_tokens")
                )
            provenance["length"] = {
                "target_words": target_words,
                "hard_maximum": hard_maximum,
                # This is the pre-revision,
                # pre-finalize body -- a later gated-loop revision (below) replaces
                # ``generated_text`` wholesale, and finalize removes sentences from it.
                # ``final_words``, set after finalize runs, is the length of the text
                # actually saved; this field is left describing what it always has, the
                # generation call's own output, so an existing reader is not surprised.
                "words": words,
                "regenerated": length_regenerated,
                "over_target": words > target_words,
            }
        else:
            # ``False`` ("no regeneration happened"),
            # not ``None`` -- no ``target_words`` means the hard-maximum check above
            # never runs at all, which is the same outcome a reader of this boolean
            # field cares about as a body that ran the check and stayed under it.
            length_regenerated = False

        # ``prompt_version`` hashes the full prompt actually sent (template + merged
        # writing rules + expertise modifier); ``prompt_template_version`` identifies
        # the section template alone so runs can be grouped by template.
        provenance["prompt_template_version"] = prompt_version(
            get_section_prompt(section_type, language)
        )
        # Fold every full-text grounding analysis call made while building the context,
        # before the citation-link fold below, so both compose onto the
        # same running totals regardless of which ran first.
        provenance = _with_analysis_provenance(provenance, analysis_calls)

        # Code-level check of the citations the model produced against the whole
        # library. The text itself is stored untouched, and a failure of the
        # audit must never fail a job whose (paid) generation already succeeded.
        try:
            citation_audit = audit_citations(
                generated_text, [pp.paper for pp in project_papers]
            ).model_dump()
        except Exception:
            logger.warning("Citation audit failed for draft %s", draft.id, exc_info=True)
            citation_audit = None

        # A second, structured call over the text just generated:
        # a sentence-to-reference map so claim verification no longer has to guess a
        # citation's shape from the finished markdown. Try/except mirrors the citation
        # audit above -- a failure here (provider outage, malformed structured response)
        # costs only the mapping, never the (already paid-for) generated text.
        uncited_sentences = None
        # A citation-link failure (here or after the revision
        # below) leaves `citation_links` empty, so the gate below verifies nothing and
        # finalize removes nothing -- the exit invariant is silently not enforced. Set
        # once either failure fires, so the final report can say so rather than claim
        # an empty, all-verified set.
        link_map_failed = False
        # Set when any citation-link coverage retry's own call fails outright
        # (a provider outage, a malformed structured response), across the section's
        # first pass and, when the gate triggers a revision, its second pass too --
        # never reset once set, so the final report shows a call failure even if a
        # later pass's own retry succeeds.
        link_coverage_retry_failed = False
        from app.services.fulltext import _build_paper_lookup

        paper_lookup = _build_paper_lookup([pp.paper for pp in project_papers])
        try:
            link_result = await build_citation_link_map(
                generated_text, paper_lookup, evidence_catalog
            )
            citation_links = link_result.links
            uncited_sentences = link_result.uncited_sentences
            provenance = _with_citation_link_provenance(provenance, link_result.provenance)
            # Close the coverage gap left in *this* call's own results
            # before anything downstream (the verification gate, finalize) ever sees
            # them -- see `_close_citation_link_coverage_gap`'s own docstring.
            citation_links, uncited_sentences, provenance, retry_failed = (
                await _close_citation_link_coverage_gap(
                    generated_text, citation_links, uncited_sentences,
                    paper_lookup, evidence_catalog, provenance,
                )
            )
            link_coverage_retry_failed = link_coverage_retry_failed or retry_failed
        except Exception:
            logger.warning("Citation link map failed for draft %s", draft.id, exc_info=True)
            citation_links = None
            link_map_failed = True
            provenance = _with_citation_link_provenance(provenance, None)

        # --- Verify -> gate -> (revise once) -> finalize.
        # ``regen_count`` is 0 or 1 -- exactly one whole-section revision, never a loop --
        # and only fires when at least one claim's status is neither "verified" nor the
        # uncheckable "no_full_text" (design section 6 amendment A3: revision lists the
        # sentences that already passed verbatim, so a regeneration never forces their
        # re-verification). Unchanged (sentence, key) pairs after a revision keep their
        # first-pass verdict instead of paying for a second call. ---
        from app.services.fulltext import (
            _load_analysis_deps,
            _verify_claims,
            claims_from_citation_links,
            evidence_quotes_by_claim_map,
            finalize_generated_section,
            surviving_verifications,
        )

        claims = claims_from_citation_links(citation_links) if citation_links else []
        verifications: list = []
        first_pass_verified_rate: float | None = None
        # Named distinctly from ``length_regenerated``
        # above -- this fires when the verification gate below finds a problem and the
        # whole section is rewritten once, an independent event from the hard-maximum
        # regeneration that may or may not also have happened on the same run.
        loop_revised = False

        if claims:
            async with session_factory() as bg_db:
                await task_service.update_job_status(
                    bg_db, job_id, JobStatus.running, progress=0.8,
                    progress_message="Verifying claims against full text...",
                    result={"state": "verify", "claims": len(claims)},
                )
            evidence_map = evidence_quotes_by_claim_map(citation_links, evidence_catalog)
            deps = await _load_analysis_deps(draft.project_id, session_factory)
            outcome = await _verify_claims(
                claims, paper_lookup, deps, job_id, session_factory,
                progress_label="Verifying", evidence_quotes_by_claim=evidence_map,
            )
            verifications = outcome["verifications"]
            provenance = _with_verification_provenance(provenance, outcome["provenance"])
            first_pass_verified_rate = (
                sum(1 for v in verifications if v.status == "verified") / len(claims)
            )

            # A coverage or comparative-meta-evaluation flag
            # (`app.agents.citation_link_agent._flag_sentence_coverage`, edits 3/8)
            # also forces the one revision pass, even when every reported proposition
            # itself came back "verified" -- the residue outside them (or the ranking
            # phrase inside one) is what needs fixing, and this is outcome 1's own
            # "revise first" of the three the finalize gate now applies.
            sentence_has_coverage_flag = {
                link.get("sentence")
                for link in (citation_links or [])
                if link.get("coverage_incomplete") or link.get("meta_evaluation")
            }
            needs_revision = (
                any(v.status in ("unsupported", "needs_nuance", "error") for v in verifications)
                or bool(sentence_has_coverage_flag)
            )
            if needs_revision:
                loop_revised = True
                async with session_factory() as bg_db:
                    await task_service.update_job_status(
                        bg_db, job_id, JobStatus.running, progress=0.85,
                        progress_message="Revising unverified sentences...",
                        result={"state": "revise"},
                    )

                # A sentence with a mix of verified and
                # not-verified propositions, or one flagged for coverage/meta-
                # evaluation even though every reported proposition verified, is
                # neither told to stay verbatim (it still asserts unverified material)
                # nor listed as a plain failure (part of it did verify) -- it gets its
                # own, third instruction below instead, so the model is never given
                # two contradictory instructions about the same sentence.
                sentence_all_verified: dict[str, bool] = {}
                sentence_any_verified: dict[str, bool] = {}
                for (_ct, _key, _cx, sentence), v in zip(claims, verifications):
                    verified = v.status == "verified"
                    sentence_all_verified[sentence] = sentence_all_verified.get(sentence, True) and verified
                    sentence_any_verified[sentence] = sentence_any_verified.get(sentence, False) or verified
                fully_verified_sentences = {
                    sentence for sentence, all_verified in sentence_all_verified.items() if all_verified
                }
                coverage_issue_sentences = {
                    sentence
                    for sentence, all_verified in sentence_all_verified.items()
                    if not all_verified and sentence_any_verified.get(sentence)
                } | (sentence_has_coverage_flag & fully_verified_sentences)

                verified_sentences = sorted({
                    sentence
                    for (_ct, _key, _cx, sentence), v in zip(claims, verifications)
                    if v.status == "verified" and sentence not in coverage_issue_sentences
                })
                failed_lines: list[str] = []
                for (_ct, key, _cx, sentence), v in zip(claims, verifications):
                    if (
                        v.status in ("unsupported", "needs_nuance", "error")
                        and sentence not in coverage_issue_sentences
                    ):
                        quote_note = (
                            f' Nearest supported material: "{v.evidence_quote}".'
                            if v.evidence_quote else ""
                        )
                        failed_lines.append(
                            f'- "{sentence}" (citing {key}): {v.status} -- '
                            f"{v.explanation}{quote_note}"
                        )

                coverage_issue_lines: list[str] = []
                for sentence in sorted(coverage_issue_sentences):
                    verified_props = sorted({
                        ct
                        for (ct, _key, _cx, claim_sentence), v in zip(claims, verifications)
                        if claim_sentence == sentence and v.status == "verified"
                    })
                    residue_spans = sorted({
                        span
                        for link in (citation_links or [])
                        if link.get("sentence") == sentence
                        for span in (link.get("residue_spans") or [])
                    })
                    verified_text = (
                        ", ".join(f'"{p}"' for p in verified_props) if verified_props else "(none)"
                    )
                    not_covered_text = (
                        ", ".join(f'"{s}"' for s in residue_spans) if residue_spans else "(unspecified)"
                    )
                    coverage_issue_lines.append(
                        f'  - "{sentence}"\n'
                        f"    verified: {verified_text}\n"
                        f"    not covered: {not_covered_text}"
                    )
                # `finalize_generated_section` is about to remove every
                # sentence above (rule 2), plus every uncited "finding" sentence (rule
                # 1), from the FIRST pass's own citation-link call -- the same list
                # `stripped_uncited_sentences` (built below from whatever
                # `uncited_sentences` holds when this revision finishes) would carry had
                # no revision happened at all. Telling the model about them now, while
                # it is already paying for one whole-section rewrite, is what lets a
                # surviving framing sentence stop pointing at material that is about to
                # disappear (a surviving "However,
                # the same authors ..." or "Three gaps emerge." with nothing to follow
                # it, left dangling because the removal path never told the writer what
                # it was about to remove). An "unclassified" entry is not itself removed
                # by finalize (`_finalize_paragraph_text` keeps it, uncertain rather than
                # wrong), but it is still named here: the model is already rewriting the
                # whole section, and a sentence the citation-link call could not classify
                # is exactly the shape it can fix in one pass -- give it a citation,
                # rewrite it as unambiguous context, or drop it.
                pre_revision_uncited = _strip_locked_block_findings(
                    uncited_sentences, locked_blocks
                )
                uncited_finding_lines = [
                    f'- "{u["sentence"]}"'
                    for u in pre_revision_uncited
                    if u.get("tag") == "finding" and not u.get("unclassified")
                ]
                unclassified_lines = [
                    f'- "{u["sentence"]}"'
                    for u in pre_revision_uncited
                    if u.get("unclassified")
                ]
                revision_block = (
                    "\n\nREVISION REQUIRED: verification found problems with some cited "
                    "sentences below. Rewrite the whole section once more, following "
                    "every rule already given, with these additional constraints:\n"
                    "- Keep every one of these sentences exactly as written, verbatim, "
                    "in the same position -- they already passed verification:\n"
                    + "\n".join(f'  - "{s}"' for s in verified_sentences)
                    + "\n- These sentences failed verification and must be rewritten, "
                    "removed, or re-cited so every remaining citation is supported by "
                    "the material already given to you:\n"
                    + "\n".join(failed_lines)
                )
                if coverage_issue_lines:
                    revision_block += (
                        "\n- These sentences assert material that no verified claim "
                        "covers. Rewrite each one so it asserts only the verified "
                        "propositions listed under it, keeping its citation exactly "
                        "as written, and put nothing else in the sentence. Add no new "
                        "claim:\n" + "\n".join(coverage_issue_lines)
                    )
                if uncited_finding_lines:
                    revision_block += (
                        "\n- These sentences state an empirical finding with no "
                        "citation and will be removed:\n"
                        + "\n".join(uncited_finding_lines)
                    )
                if unclassified_lines:
                    revision_block += (
                        "\n- These sentences were not recognised as either a cited "
                        "claim or scene-setting context: give each one a citation, "
                        "rewrite it as unambiguous context, or remove it:\n"
                        + "\n".join(unclassified_lines)
                    )
                if uncited_finding_lines or unclassified_lines:
                    revision_block += (
                        "\n- For every sentence you remove or rewrite above, also "
                        "rewrite the paragraph it belongs to so the surrounding text "
                        "reads coherently without it: drop or rewrite any other "
                        "sentence in the same paragraph that only made sense pointing "
                        "at what you just removed (an opening that announces a count "
                        "with nothing left to satisfy it, or a \"however\"/\"the same "
                        "authors\"/\"this finding\" with nothing left before it). Add "
                        "no new claim while doing this."
                    )
                if target_words is not None:
                    # Without a length constraint here, a revision that rewrites the
                    # whole section would have every chance to land back over the hard
                    # maximum (or push a body that was already over it further over)
                    # with nothing telling the model otherwise. Restating the same
                    # instruction the initial generation call already sent gives the
                    # model a real chance to stay within it on its own; the length
                    # re-check after finalize, below, is what actually guarantees it.
                    revision_block += f"\n- {_length_instruction(target_words)}"
                revised_text, revise_provenance = await call_deepseek(
                    system_prompt, user_prompt + revision_block
                )
                generated_text = _normalize_generated_text(
                    revised_text, section_title, section_type_default_title
                )
                provenance["total_calls"] = provenance.get("total_calls", 1) + 1
                provenance["total_input_tokens"] = _sum_tokens(
                    provenance.get("total_input_tokens", provenance.get("input_tokens")),
                    revise_provenance.get("input_tokens"),
                )
                provenance["total_output_tokens"] = _sum_tokens(
                    provenance.get("total_output_tokens", provenance.get("output_tokens")),
                    revise_provenance.get("output_tokens"),
                )

                try:
                    citation_audit = audit_citations(
                        generated_text, [pp.paper for pp in project_papers]
                    ).model_dump()
                except Exception:
                    logger.warning(
                        "Citation audit failed for draft %s (revision)", draft.id,
                        exc_info=True,
                    )
                    citation_audit = None

                try:
                    link_result = await build_citation_link_map(
                        generated_text, paper_lookup, evidence_catalog
                    )
                    citation_links = link_result.links
                    uncited_sentences = link_result.uncited_sentences
                    provenance = _with_citation_link_provenance(
                        provenance, link_result.provenance
                    )
                    # Close the coverage gap left in the revision's own
                    # citation-link call too, not only the section's first pass -- see
                    # `_close_citation_link_coverage_gap`'s own docstring.
                    citation_links, uncited_sentences, provenance, revision_retry_failed = (
                        await _close_citation_link_coverage_gap(
                            generated_text, citation_links, uncited_sentences,
                            paper_lookup, evidence_catalog, provenance,
                        )
                    )
                    link_coverage_retry_failed = (
                        link_coverage_retry_failed or revision_retry_failed
                    )
                except Exception:
                    logger.warning(
                        "Citation link map failed for draft %s (revision)", draft.id,
                        exc_info=True,
                    )
                    citation_links = None
                    uncited_sentences = None
                    link_map_failed = True
                    provenance = _with_citation_link_provenance(provenance, None)

                # Only a (sentence, proposition, key) triple whose text actually
                # changed is sent back to the verifier; an
                # unchanged one keeps the verdict it already earned. Keying on the
                # proposition too (not (sentence, key) alone) means a re-linked,
                # re-narrowed proposition for a sentence whose own text did not change
                # is re-verified rather than silently inheriting a DIFFERENT
                # proposition's old verdict.
                old_status_by_pair = {
                    (sentence, ct, key): v
                    for (ct, key, _cx, sentence), v in zip(claims, verifications)
                }
                new_claims = (
                    claims_from_citation_links(citation_links) if citation_links else []
                )
                to_reverify: list[tuple[str, str, str, str]] = []
                reverify_positions: list[int] = []
                prior_by_position: dict[int, object] = {}
                for i, (ct, key, cx, sentence) in enumerate(new_claims):
                    prior = old_status_by_pair.get((sentence, ct, key))
                    if prior is not None:
                        prior_by_position[i] = prior
                    else:
                        to_reverify.append((ct, key, cx, sentence))
                        reverify_positions.append(i)
                if to_reverify:
                    evidence_map2 = evidence_quotes_by_claim_map(
                        citation_links, evidence_catalog
                    )
                    deps2 = await _load_analysis_deps(draft.project_id, session_factory)
                    outcome2 = await _verify_claims(
                        to_reverify, paper_lookup, deps2, job_id, session_factory,
                        progress_label="Re-verifying", evidence_quotes_by_claim=evidence_map2,
                    )
                    provenance = _with_verification_provenance(provenance, outcome2["provenance"])
                    for pos, v in zip(reverify_positions, outcome2["verifications"]):
                        prior_by_position[pos] = v
                claims = new_claims
                verifications = [prior_by_position[i] for i in range(len(new_claims))]

        # --- finalize: remove every cited sentence whose final status is not verified,
        # remove every uncited "finding" sentence, drop a no_full_text citation (and its
        # sentence when no verified citation remains), strip [NEEDS CITATION]. ---
        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id, JobStatus.running, progress=0.95,
                progress_message="Finalizing verified text...",
                result={"state": "finalize"},
            )

        # Keyed on (sentence, proposition, key).
        claim_status = {
            (sentence, ct, key): v.status
            for (ct, key, _cx, sentence), v in zip(claims, verifications)
        }
        # The same key, but to the claim's own verified evidence quote(s) --
        # narrows the coverage gate's own numeral disqualification
        # (`finalize_generated_section`'s own ``claim_evidence_quotes`` parameter,
        # `sentence_coverage.covered_numeral_keys`): a sentence whose only problem
        # is a residue numeral no proposition covers is not removed for that
        # reason alone when the same number is verbatim in one of its own kept
        # claims' evidence quotes.
        claim_evidence_quotes = {
            (sentence, ct, key): (v.evidence_quotes or ([v.evidence_quote] if v.evidence_quote else []))
            for (ct, key, _cx, sentence), v in zip(claims, verifications)
        }
        # Finalize is handed the locked-block-stripped
        # list, not the raw one -- the job result must report that same stripped list,
        # not the raw `uncited_sentences`, so a reader never sees an entry finalize
        # deliberately never acted on (a locked block's own text, design amendment A4's
        # exemption).
        stripped_uncited_sentences = _strip_locked_block_findings(uncited_sentences, locked_blocks)
        finalize_result = finalize_generated_section(
            generated_text,
            citation_links or [],
            stripped_uncited_sentences,
            claim_status,
            section_title=section_title,
            claim_evidence_quotes=claim_evidence_quotes,
        )
        final_text = finalize_result.text
        final_citation_links = finalize_result.citation_links

        # --- Bounded regeneration backstop: a section whose pre-finalize citation
        # links were non-empty, or whose own citation audit found at least one
        # citation, but that finalize reduces to fewer than
        # `THIN_SECTION_CITED_SENTENCE_MINIMUM` distinct cited sentences is
        # regenerated once, whole, rather than shipped as a heading over a hollow or
        # near-hollow paragraph. The bound is "fewer than three", not "zero": a
        # section that finalizes to exactly one or two cited sentences after twelve
        # of its own findings were removed reads, to any actual reader, as the same
        # kind of failure a literally empty section is, and both shapes come from
        # the same cause -- the writer drafted more than the library's verified
        # claims can support, and finalize's own coverage/verification gates
        # correctly refused to ship what could not be verified. This is also the
        # exact signature of a citation link finalize could not key back to a
        # sentence of its own splitter's fragments (the defect Fix B's sentence snap,
        # in ``validate_citation_link``, targets at the source); this backstop is what
        # makes the failure survivable when something else still produces the same
        # shape, never the fix itself. A link-map failure is excluded (`link_map_
        # failed`) since an empty result there is already a recorded outage, not a
        # silent loss. Exactly one extra whole-section generation call, counted
        # independently of the gate's own revision above and of the hard-maximum
        # regeneration below -- never a loop: the second attempt runs the same
        # write, link and verification passes once and finalizes again, restarting
        # from the section's own ORIGINAL, pre-revision prompt every time (this
        # backstop never continues from an already-revised draft). Both attempts
        # are kept in full (`_RegenerationAttempt`) and compared once the second
        # one finalizes; whichever survives finalize with MORE distinct cited
        # sentences ships, on a tie the first attempt (`loop_stats
        # ["regeneration_kept"]`), so a regenerated attempt that does no better
        # than the one already in hand -- including one whose own citation-link
        # call fails outright -- can never make the delivered section worse.
        empty_section_regenerated = False
        empty_section_after_regeneration = False
        regeneration_kept = "first"
        regeneration_calls: list[dict] = []
        pre_finalize_had_citations = bool(citation_links) or bool(
            citation_audit and citation_audit.get("total")
        )
        if (
            not link_map_failed
            and pre_finalize_had_citations
            and _cited_sentence_count(finalize_result.citation_links)
            < THIN_SECTION_CITED_SENTENCE_MINIMUM
        ):
            empty_section_regenerated = True
            # The first attempt's own full state, captured before any of it is
            # overwritten below, so it can be weighed against the regenerated
            # attempt once that one finalizes too.
            first_attempt = _RegenerationAttempt(
                generated_text=generated_text,
                citation_links=citation_links,
                uncited_sentences=uncited_sentences,
                citation_audit=citation_audit,
                claims=claims,
                verifications=verifications,
                first_pass_verified_rate=first_pass_verified_rate,
                link_map_failed=link_map_failed,
                claim_status=claim_status,
                stripped_uncited_sentences=stripped_uncited_sentences,
                finalize_result=finalize_result,
            )
            async with session_factory() as bg_db:
                await task_service.update_job_status(
                    bg_db, job_id, JobStatus.running, progress=0.96,
                    progress_message="Regenerating an empty section...",
                    result={"state": "regenerate"},
                )

            regen_text, regen_gen_provenance = await call_deepseek(system_prompt, user_prompt)
            regen_text = _normalize_generated_text(
                regen_text, section_title, section_type_default_title
            )
            regeneration_calls.append(regen_gen_provenance)
            provenance["total_calls"] = provenance.get("total_calls", 1) + 1
            provenance["total_input_tokens"] = _sum_tokens(
                provenance.get("total_input_tokens", provenance.get("input_tokens")),
                regen_gen_provenance.get("input_tokens"),
            )
            provenance["total_output_tokens"] = _sum_tokens(
                provenance.get("total_output_tokens", provenance.get("output_tokens")),
                regen_gen_provenance.get("output_tokens"),
            )

            try:
                regen_citation_audit = audit_citations(
                    regen_text, [pp.paper for pp in project_papers]
                ).model_dump()
            except Exception:
                logger.warning(
                    "Citation audit failed for draft %s (empty-section regeneration)",
                    draft.id, exc_info=True,
                )
                regen_citation_audit = None

            regen_link_map_failed = False
            try:
                regen_link_result = await build_citation_link_map(
                    regen_text, paper_lookup, evidence_catalog
                )
                regen_citation_links = regen_link_result.links
                regen_uncited_sentences = regen_link_result.uncited_sentences
                provenance = _with_citation_link_provenance(
                    provenance, regen_link_result.provenance
                )
                regen_citation_links, regen_uncited_sentences, provenance, regen_retry_failed = (
                    await _close_citation_link_coverage_gap(
                        regen_text, regen_citation_links, regen_uncited_sentences,
                        paper_lookup, evidence_catalog, provenance,
                    )
                )
                link_coverage_retry_failed = link_coverage_retry_failed or regen_retry_failed
            except Exception:
                logger.warning(
                    "Citation link map failed for draft %s (empty-section regeneration)",
                    draft.id, exc_info=True,
                )
                regen_citation_links = None
                regen_uncited_sentences = None
                regen_link_map_failed = True
                provenance = _with_citation_link_provenance(provenance, None)

            regen_claims = (
                claims_from_citation_links(regen_citation_links) if regen_citation_links else []
            )
            regen_verifications: list = []
            regen_first_pass_verified_rate: float | None = None
            if regen_claims:
                regen_evidence_map = evidence_quotes_by_claim_map(
                    regen_citation_links, evidence_catalog
                )
                regen_deps = await _load_analysis_deps(draft.project_id, session_factory)
                regen_outcome = await _verify_claims(
                    regen_claims, paper_lookup, regen_deps, job_id, session_factory,
                    progress_label="Verifying (regenerated section)",
                    evidence_quotes_by_claim=regen_evidence_map,
                )
                regen_verifications = regen_outcome["verifications"]
                provenance = _with_verification_provenance(provenance, regen_outcome["provenance"])
                regen_first_pass_verified_rate = (
                    sum(1 for v in regen_verifications if v.status == "verified")
                    / len(regen_claims)
                )

            regen_claim_status = {
                (sentence, ct, key): v.status
                for (ct, key, _cx, sentence), v in zip(regen_claims, regen_verifications)
            }
            regen_claim_evidence_quotes = {
                (sentence, ct, key): (v.evidence_quotes or ([v.evidence_quote] if v.evidence_quote else []))
                for (ct, key, _cx, sentence), v in zip(regen_claims, regen_verifications)
            }
            regen_stripped_uncited_sentences = _strip_locked_block_findings(
                regen_uncited_sentences, locked_blocks
            )
            regen_finalize_result = finalize_generated_section(
                regen_text,
                regen_citation_links or [],
                regen_stripped_uncited_sentences,
                regen_claim_status,
                section_title=section_title,
                claim_evidence_quotes=regen_claim_evidence_quotes,
            )
            second_attempt = _RegenerationAttempt(
                generated_text=regen_text,
                citation_links=regen_citation_links,
                uncited_sentences=regen_uncited_sentences,
                citation_audit=regen_citation_audit,
                claims=regen_claims,
                verifications=regen_verifications,
                first_pass_verified_rate=regen_first_pass_verified_rate,
                link_map_failed=regen_link_map_failed,
                claim_status=regen_claim_status,
                stripped_uncited_sentences=regen_stripped_uncited_sentences,
                finalize_result=regen_finalize_result,
            )

            # Never make the section worse: whichever attempt survives finalize
            # with MORE distinct cited sentences is the one kept; on a tie the
            # first attempt wins, since it may already carry the gate's own
            # successful revision, which the regenerated attempt -- restarting
            # from the section's ORIGINAL prompt -- does not.
            first_cited_count = _cited_sentence_count(
                first_attempt.finalize_result.citation_links
            )
            second_cited_count = _cited_sentence_count(
                second_attempt.finalize_result.citation_links
            )
            if second_cited_count > first_cited_count:
                chosen = second_attempt
                regeneration_kept = "second"
            else:
                chosen = first_attempt
                regeneration_kept = "first"

            generated_text = chosen.generated_text
            citation_links = chosen.citation_links
            uncited_sentences = chosen.uncited_sentences
            citation_audit = chosen.citation_audit
            claims = chosen.claims
            verifications = chosen.verifications
            first_pass_verified_rate = chosen.first_pass_verified_rate
            link_map_failed = chosen.link_map_failed
            claim_status = chosen.claim_status
            stripped_uncited_sentences = chosen.stripped_uncited_sentences
            finalize_result = chosen.finalize_result
            final_text = finalize_result.text
            final_citation_links = finalize_result.citation_links
            # Only when the KEPT attempt itself has zero cited sentences -- rule 14
            # (`empty_cited_section`, `demo/check_delivered.py`) treats this flag
            # as an unconditional violation, and the kept attempt is not always
            # the regenerated one.
            empty_section_after_regeneration = not finalize_result.citation_links

        provenance["regeneration_calls"] = regeneration_calls

        final_words = _count_body_words(final_text)
        # The hard maximum was enforced once, right after
        # the very first generation call; the gated loop's own revision above (when it
        # fires) rewrites the whole section with no length constraint of its own, and
        # finalize only ever removes sentences verification rejected, never enforces a
        # target length either. This is the last check, on the exact text about to be
        # saved, and it is unconditional: trimming, not another paid regeneration, is
        # what guarantees no delivered section exceeds *hard_maximum* regardless of how
        # it got there.
        length_trimmed = False
        if target_words is not None and hard_maximum is not None and final_words > hard_maximum:
            final_text, final_citation_links, trimmed_uncited_sentences, length_trimmed = (
                _trim_body_to_hard_maximum(
                    final_text,
                    hard_maximum,
                    final_citation_links,
                    finalize_result.uncited_sentences,
                )
            )
            if length_trimmed:
                finalize_result = finalize_result._replace(
                    text=final_text,
                    citation_links=final_citation_links,
                    uncited_sentences=trimmed_uncited_sentences,
                )
                final_words = _count_body_words(final_text)
        survival_rate = (final_words / target_words) if target_words else None
        # ``provenance["length"]`` was built right
        # after generation (and the over-length regeneration, if any) and never
        # touched again, so it describes a draft the gated loop's own revision (if it
        # ran) and finalize's sentence removals may since have changed. ``final_words``
        # is the length of the text actually saved; it is added here, not used to
        # overwrite ``words``, so an existing reader of ``length.words`` still sees the
        # generation call's own output, unchanged.
        if "length" in provenance:
            provenance["length"] = {**provenance["length"], "final_words": final_words}
        # Built from the links finalize actually kept, not from the
        # raw verdict list -- rule 2 of `_finalize_paragraph_text` removes a whole
        # sentence, and every citation on it, the moment any one of its citations is
        # not verified, so a verified citation sharing a sentence with an unsupported
        # one is gone from the final text even though its own status says "verified".
        final_verifications = surviving_verifications(
            claims, verifications, final_citation_links, finalize_result.healed_sentences
        )
        loop_stats = {
            **finalize_result.stats,
            # Two independent events, each under its
            # own name -- see the comments where each variable is set, above.
            "length_regenerated": length_regenerated,
            "loop_revised": loop_revised,
            "first_pass_verified_rate": first_pass_verified_rate,
            "survival_rate": survival_rate,
            "link_map_failed": link_map_failed,
            # Whether a citation-link coverage retry's own call (the section's first
            # pass or its revision pass) raised outright, as opposed to answering but
            # never mentioning a gap sentence -- see
            # ``resolve_unclassified_sentences``'s own docstring for why the two are
            # recorded, and treated by finalize, differently.
            "link_coverage_retry_failed": link_coverage_retry_failed,
            # Whether the length re-check above actually
            # dropped trailing content to bring the saved text at or under
            # `hard_maximum` -- ``False`` when no target length was set, and when the
            # finalized text already fit without it.
            "length_trimmed": length_trimmed,
            # Whether the empty-section regeneration backstop fired for this section
            # (0 or 1, like the other write-stage counters this dict otherwise
            # carries) -- true whenever a second attempt ran, whether or not it is
            # the one kept; "regeneration_kept" names which attempt actually
            # shipped ("first" or "second"), and "empty_section_after_
            # regeneration" is true only when the KEPT attempt itself still
            # finalizes to zero cited sentences.
            "empty_section_regenerated": int(empty_section_regenerated),
            "regeneration_kept": regeneration_kept,
            "empty_section_after_regeneration": empty_section_after_regeneration,
            # The write job's own record of which proposition lacked a verdict or
            # which residue caused each `sentences_removed_coverage_incomplete`
            # sentence, so a reader (the demo summary that already carries
            # `loop_stats` whole) can justify each one rather than reading a bare
            # count.
            "coverage_incomplete_reasons": finalize_result.coverage_incomplete_reasons,
            # The same treatment for every `sentences_removed_meta_evaluation_uncited`
            # sentence's own matched span (see `FinalizeResult`'s own docstring).
            "meta_evaluation_reasons": finalize_result.meta_evaluation_reasons,
            # The same treatment again for every `sentences_removed_verb_initial`
            # sentence's own before/after text.
            "verb_initial_reasons": finalize_result.verb_initial_reasons,
            # The same treatment again for every `sentences_removed_heading_shaped`
            # sentence -- the second, independent guard against a heading-shaped
            # fragment reaching the delivered draft as a body paragraph.
            "heading_shaped_reasons": finalize_result.heading_shaped_reasons,
            # The same treatment again for every heading `headings_removed` counted
            # -- each dropped heading's own verbatim text (see `FinalizeResult`'s
            # own docstring).
            "headings_removed_reasons": finalize_result.headings_removed_reasons,
            # The same treatment again for every marker-headed fragment dropped
            # rather than merged into its predecessor -- see `FinalizeResult`'s own
            # docstring and `_finalize_paragraph_text`'s `sentences_truncated_
            # fragment_repaired` comment.
            "truncated_fragment_reasons": finalize_result.truncated_fragment_reasons,
        }
        # A citation-link failure means nothing was verified
        # and finalize removed nothing -- the exit invariant was never enforced on this
        # text, so the report must say so rather than claim an empty, all-verified set.
        claim_report = None if link_map_failed else {
            "draft_id": str(draft.id),
            "verifications": [v.model_dump(mode="json") for v in final_verifications],
            "verified_count": len(final_verifications),
            "unsupported_count": 0,
            "nuance_count": 0,
            "abstract_only_count": 0,
            "error_count": 0,
        }

        # Save the finalized section into the draft, citationLinks repaired. A fresh
        # section is appended; regenerating an already-present section (same
        # section_type, tagged on its own heading node) replaces just that section's
        # block, so no other section of a multi-section draft is ever touched.
        section_nodes = _build_section_tiptap_nodes(
            section_type,
            section_title,
            final_text,
            final_citation_links,
            finalize_result.uncited_sentences,
        )
        async with session_factory() as bg_db:
            from app.services import draft as draft_service

            fresh_draft = await bg_db.get(Draft, draft.id)
            if fresh_draft is not None:
                merged_content = _merge_section_into_draft(
                    fresh_draft.content, section_type, section_nodes
                )
                await draft_service.update_draft(
                    bg_db, draft.id, fresh_draft.user_id, content=merged_content
                )
                await bg_db.commit()

        # Done
        async with session_factory() as bg_db:
            await task_service.update_job_status(
                bg_db, job_id,
                status=JobStatus.completed,
                progress=1.0,
                progress_message="Generation complete",
                result={
                    "section_type": section_type,
                    "content": final_text,
                    "papers_used": len(selected_papers) if needs_papers else 0,
                    "citation_audit": citation_audit,
                    "citation_links": final_citation_links,
                    "uncited_sentences": stripped_uncited_sentences,
                    "provenance": provenance,
                    "metrics": {"analysis": analysis_metrics},
                    "claim_report": claim_report,
                    "loop_stats": loop_stats,
                },
            )

    except Exception as e:
        logger.exception("Section generation failed for draft %s", draft.id)
        async with session_factory() as err_db:
            await task_service.update_job_status(
                err_db, job_id,
                status=JobStatus.failed,
                error=str(e),
            )


async def _build_data_context(project_id: UUID, session_factory) -> str:
    """Load user's research data and analysis results for research article sections."""
    parts: list[str] = []

    async with session_factory() as db:
        # 1. Load datasets
        ds_result = await db.execute(
            select(Dataset).where(Dataset.project_id == project_id)
        )
        datasets = list(ds_result.scalars().all())

        # 2. Load completed quantitative analysis jobs
        quant_result = await db.execute(
            select(AnalysisJob).where(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.quantitative,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        quant_jobs = list(quant_result.scalars().all())

        # 3. Load completed qualitative analysis jobs
        qual_result = await db.execute(
            select(AnalysisJob).where(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.qualitative,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        qual_jobs = list(qual_result.scalars().all())

        # 4. Load codebooks
        cb_result = await db.execute(
            select(Codebook).where(Codebook.project_id == project_id)
        )
        codebooks = list(cb_result.scalars().all())

    # Return empty string if nothing exists
    if not datasets and not quant_jobs and not qual_jobs and not codebooks:
        return ""

    parts.append("## User's Research Data\n")

    # Datasets
    if datasets:
        for ds in datasets:
            parts.append("### Dataset")
            parts.append(f"- Filename: {ds.filename}")
            parts.append(f"- Rows: {ds.row_count}, Columns: {len(ds.columns) if ds.columns else 0}")
            if ds.columns:
                col_entries = []
                for col_name, col_info in ds.columns.items():
                    if isinstance(col_info, dict):
                        dtype = col_info.get("dtype", col_info.get("type", "unknown"))
                        col_entries.append(f"{col_name} ({dtype})")
                    else:
                        col_entries.append(str(col_name))
                parts.append(f"- Variables: {', '.join(col_entries)}")
            parts.append("")

    # Quantitative analysis results
    if quant_jobs:
        parts.append("### Quantitative Analysis Results")
        for job in quant_jobs:
            if job.result:
                result_data = job.result
                # Descriptive statistics
                if "descriptive_stats" in result_data:
                    parts.append("#### Descriptive Statistics")
                    stats = result_data["descriptive_stats"]
                    if isinstance(stats, dict):
                        for var, var_stats in stats.items():
                            if isinstance(var_stats, dict):
                                stat_str = ", ".join(
                                    f"{k}: {v}" for k, v in var_stats.items()
                                    if k in ("mean", "std", "min", "max", "median", "count")
                                )
                                parts.append(f"- {var}: {stat_str}")
                    elif isinstance(stats, str):
                        parts.append(stats)
                # Correlations
                if "correlations" in result_data:
                    parts.append("#### Correlations")
                    corr = result_data["correlations"]
                    if isinstance(corr, dict):
                        for pair, value in corr.items():
                            parts.append(f"- {pair}: r = {value}")
                    elif isinstance(corr, str):
                        parts.append(corr)
                # Predictions / regression
                if "predictions" in result_data:
                    parts.append("#### Predictions / Regression")
                    preds = result_data["predictions"]
                    if isinstance(preds, str):
                        parts.append(preds)
                    elif isinstance(preds, dict):
                        for k, v in preds.items():
                            parts.append(f"- {k}: {v}")
                # Fallback: dump remaining top-level keys as summary text
                known_keys = {"descriptive_stats", "correlations", "predictions"}
                extra = {k: v for k, v in result_data.items() if k not in known_keys}
                if extra and not any(k in result_data for k in known_keys):
                    import json
                    parts.append(json.dumps(extra, ensure_ascii=False, indent=2))
        parts.append("")

    # Qualitative analysis results
    if qual_jobs:
        parts.append("### Qualitative Analysis Results")
        for job in qual_jobs:
            if job.result:
                result_data = job.result
                if "codebook" in result_data:
                    parts.append("#### Codebook")
                    cb = result_data["codebook"]
                    if isinstance(cb, dict):
                        for code, desc in cb.items():
                            parts.append(f"- {code}: {desc}")
                    elif isinstance(cb, str):
                        parts.append(cb)
                if "coded_segments" in result_data:
                    parts.append("#### Coded Segments (sample)")
                    segments = result_data["coded_segments"]
                    if isinstance(segments, list):
                        for seg in segments[:5]:  # show first 5 to keep context manageable
                            parts.append(f"- {seg}")
                    elif isinstance(segments, str):
                        parts.append(segments)
                # Fallback for other result structures
                known_keys = {"codebook", "coded_segments"}
                extra = {k: v for k, v in result_data.items() if k not in known_keys}
                if extra and not any(k in result_data for k in known_keys):
                    import json
                    parts.append(json.dumps(extra, ensure_ascii=False, indent=2))
        parts.append("")

    # Codebooks (standalone, not tied to a completed job)
    if codebooks:
        parts.append("### Codebooks")
        for cb in codebooks:
            if cb.codes:
                parts.append("#### Codes")
                if isinstance(cb.codes, dict):
                    for code, desc in cb.codes.items():
                        parts.append(f"- {code}: {desc}")
            if cb.themes:
                parts.append("#### Themes")
                if isinstance(cb.themes, dict):
                    for theme, desc in cb.themes.items():
                        parts.append(f"- {theme}: {desc}")
        parts.append("")

    return "\n".join(parts)
