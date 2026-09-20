"""Compliance checker — validates drafts against journal guidelines or golden standards."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.schemas.compliance import ComplianceCheck, ComplianceReport

logger = logging.getLogger(__name__)

# ── Golden Standard Defaults (used when no journal guidelines available) ──────

GOLDEN_STANDARDS = {
    "research_article": {
        "word_limit_min": 5000,
        "word_limit_max": 8000,
        "word_limit_abstract_min": 150,
        "word_limit_abstract_max": 300,
        "abstract_structure": "unstructured",
        "required_sections": [
            "abstract", "introduction", "literature review", "methods",
            "results", "discussion", "implications", "conclusion",
        ],
        "citation_style": "APA 7th",
        "keyword_requirements": "3-6 keywords",
        "line_spacing": "double",
        "font_requirements": "Times New Roman 12pt",
    },
    "literature_review": {
        "word_limit_min": 6000,
        "word_limit_max": 10000,
        "word_limit_abstract_min": 150,
        "word_limit_abstract_max": 300,
        "abstract_structure": "unstructured",
        "required_sections": [
            "abstract", "introduction", "literature review",
            "discussion", "conclusion",
        ],
        "citation_style": "APA 7th",
        "keyword_requirements": "3-6 keywords",
        "line_spacing": "double",
        "font_requirements": "Times New Roman 12pt",
    },
}

# Normalized aliases for section detection
SECTION_ALIASES: dict[str, str] = {
    "intro": "introduction",
    "lit review": "literature review",
    "review of literature": "literature review",
    "review of the literature": "literature review",
    "theoretical framework": "literature review",
    "methodology": "methods",
    "method": "methods",
    "research methodology": "methods",
    "research design": "methods",
    "findings": "results",
    "results and discussion": "results",
    "results/findings": "results",
    "implications": "implications",
    "practical implications": "implications",
    "theoretical implications": "implications",
    "recommendations": "implications",
    "abstract": "abstract",
    "summary": "conclusion",
    "concluding remarks": "conclusion",
    "references": "references",
    "bibliography": "references",
    "works cited": "references",
}


def _count_words_in_tiptap(content: dict | None) -> int:
    """Count total words in a Tiptap JSON document."""
    if not content:
        return 0

    def _extract_text(node: dict) -> str:
        if node.get("type") == "text":
            return node.get("text", "")
        parts = []
        for child in node.get("content", []):
            parts.append(_extract_text(child))
        return " ".join(parts)

    text = _extract_text(content)
    return len(text.split()) if text.strip() else 0


def _extract_sections_from_tiptap(content: dict | None) -> list[str]:
    """Extract section headings from Tiptap JSON, normalized to lowercase."""
    if not content:
        return []
    sections = []
    for node in content.get("content", []):
        if node.get("type") == "heading":
            text_parts = []
            for child in node.get("content", []):
                if child.get("type") == "text":
                    text_parts.append(child.get("text", ""))
            heading = " ".join(text_parts).strip().lower()
            normalized = SECTION_ALIASES.get(heading, heading)
            if normalized:
                sections.append(normalized)
    return sections


def _extract_abstract_words(content: dict | None, sections: list[str]) -> int:
    """Count words in the abstract section."""
    if not content or "abstract" not in sections:
        return 0

    in_abstract = False
    abstract_texts = []
    for node in content.get("content", []):
        if node.get("type") == "heading":
            heading_text = ""
            for child in node.get("content", []):
                if child.get("type") == "text":
                    heading_text += child.get("text", "")
            key = heading_text.strip().lower()
            normalized = SECTION_ALIASES.get(key, key)
            if normalized == "abstract":
                in_abstract = True
                continue
            elif in_abstract:
                break
        elif in_abstract and node.get("type") == "paragraph":
            for child in node.get("content", []):
                if child.get("type") == "text":
                    abstract_texts.append(child.get("text", ""))

    abstract_text = " ".join(abstract_texts)
    return len(abstract_text.split()) if abstract_text.strip() else 0


def _build_standards(paper_type: str, journal_guidelines: dict | None) -> dict:
    """Merge journal guidelines with golden standards. Journal rules take priority."""
    standards = dict(GOLDEN_STANDARDS.get(paper_type, GOLDEN_STANDARDS["research_article"]))

    if not journal_guidelines:
        return standards

    # Override with journal-specific values where available
    if journal_guidelines.get("word_limit_total"):
        limit = int(journal_guidelines["word_limit_total"])
        standards["word_limit_min"] = max(1000, limit - 1000)
        standards["word_limit_max"] = limit

    if journal_guidelines.get("word_limit_abstract"):
        abs_limit = int(journal_guidelines["word_limit_abstract"])
        standards["word_limit_abstract_min"] = max(50, abs_limit - 50)
        standards["word_limit_abstract_max"] = abs_limit + 50

    if journal_guidelines.get("abstract_structure"):
        standards["abstract_structure"] = journal_guidelines["abstract_structure"]

    # NOTE: Do NOT override required_sections from journal guidelines.
    # The journal's section list is generic (usually for research articles)
    # and doesn't know the user's paper type. The golden standard's sections
    # for the specific paper_type (literature_review vs research_article) are
    # always correct.

    if journal_guidelines.get("citation_style"):
        standards["citation_style"] = journal_guidelines["citation_style"]

    if journal_guidelines.get("keyword_requirements"):
        standards["keyword_requirements"] = journal_guidelines["keyword_requirements"]

    if journal_guidelines.get("line_spacing"):
        standards["line_spacing"] = journal_guidelines["line_spacing"]

    if journal_guidelines.get("font_requirements"):
        standards["font_requirements"] = journal_guidelines["font_requirements"]

    return standards


def _check_word_count(word_count: int, standards: dict) -> ComplianceCheck:
    """Check total word count against standards."""
    wmin = standards.get("word_limit_min", 5000)
    wmax = standards.get("word_limit_max", 8000)

    if wmin <= word_count <= wmax:
        return ComplianceCheck(
            check_type="word_count",
            status="pass",
            message=f"Word count: {word_count} (target: {wmin}-{wmax})",
            details={"word_count": word_count, "target_min": wmin, "target_max": wmax},
        )
    if word_count > wmax:
        return ComplianceCheck(
            check_type="word_count",
            status="warning",
            message=f"Word count {word_count} exceeds limit of {wmax}",
            details={"word_count": word_count, "target_max": wmax},
        )
    if word_count >= wmin * 0.5:
        return ComplianceCheck(
            check_type="word_count",
            status="warning",
            message=f"Word count {word_count} is below target ({wmin}-{wmax})",
            details={"word_count": word_count, "target_min": wmin},
        )
    return ComplianceCheck(
        check_type="word_count",
        status="fail",
        message=f"Word count is only {word_count} — paper appears incomplete (target: {wmin}-{wmax})",
        details={"word_count": word_count, "target_min": wmin},
    )


def _check_abstract_length(abstract_words: int, standards: dict) -> ComplianceCheck:
    """Check abstract word count against standards."""
    amin = standards.get("word_limit_abstract_min", 150)
    amax = standards.get("word_limit_abstract_max", 300)

    if abstract_words == 0:
        return ComplianceCheck(
            check_type="abstract_length",
            status="fail",
            message="No abstract found or abstract is empty",
            details={"abstract_words": 0, "target_min": amin, "target_max": amax},
        )
    if amin <= abstract_words <= amax:
        return ComplianceCheck(
            check_type="abstract_length",
            status="pass",
            message=f"Abstract: {abstract_words} words (target: {amin}-{amax})",
            details={"abstract_words": abstract_words},
        )
    return ComplianceCheck(
        check_type="abstract_length",
        status="warning",
        message=f"Abstract: {abstract_words} words (target: {amin}-{amax})",
        details={"abstract_words": abstract_words, "target_min": amin, "target_max": amax},
    )


def _check_sections_present(found_sections: list[str], standards: dict) -> ComplianceCheck:
    """Check if all required sections are present."""
    required = standards.get("required_sections", [])
    missing = [s for s in required if s not in found_sections]

    if not missing:
        return ComplianceCheck(
            check_type="sections_present",
            status="pass",
            message=f"All {len(required)} required sections found",
            details={"required": required, "found": found_sections},
        )
    return ComplianceCheck(
        check_type="sections_present",
        status="fail" if len(missing) > 2 else "warning",
        message=f"Missing sections: {', '.join(s.title() for s in missing)}",
        details={"required": required, "found": found_sections, "missing": missing},
    )


def _check_citation_style(standards: dict) -> ComplianceCheck:
    """Report the expected citation style (informational — cannot auto-verify)."""
    style = standards.get("citation_style", "APA 7th")
    return ComplianceCheck(
        check_type="citation_style",
        status="pass",
        message=f"Expected citation style: {style}",
        details={"citation_style": style},
    )


async def check_compliance(
    content: dict | None,
    paper_type: str,
    journal_guidelines: dict | None = None,
) -> ComplianceReport:
    """Run all compliance checks against AI-merged rules or golden standards.

    Args:
        content: Tiptap JSON document content
        paper_type: "research_article" or "literature_review"
        journal_guidelines: Optional journal-specific rules from the project
    """
    # Try AI-powered merge of golden standard + journal guidelines
    try:
        from app.agents.compliance_rules_agent import merge_compliance_rules
        merged = await merge_compliance_rules(paper_type, journal_guidelines)
        standards = {
            "word_limit_min": merged.word_limit_min,
            "word_limit_max": merged.word_limit_max,
            "word_limit_abstract_min": merged.abstract_word_min,
            "word_limit_abstract_max": merged.abstract_word_max,
            "abstract_structure": merged.abstract_structure,
            "required_sections": merged.required_sections,
            "citation_style": merged.citation_style,
        }
    except Exception:
        logger.exception("AI compliance rules merge failed, falling back to hardcoded merge")
        # Fallback to hardcoded merge
        standards = _build_standards(paper_type, journal_guidelines)

    word_count = _count_words_in_tiptap(content)
    sections = _extract_sections_from_tiptap(content)
    abstract_words = _extract_abstract_words(content, sections)

    checks: list[ComplianceCheck] = []
    checks.append(_check_word_count(word_count, standards))
    checks.append(_check_sections_present(sections, standards))
    checks.append(_check_abstract_length(abstract_words, standards))
    checks.append(_check_citation_style(standards))

    # Determine overall status
    statuses = [c.status for c in checks]
    if "fail" in statuses:
        overall = "fail"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "pass"

    return ComplianceReport(
        overall_status=overall,
        checks=checks,
        total_word_count=word_count,
        sections_found=sections,
    )


async def run_compliance_check(draft_id: UUID, job_id: UUID, session_factory) -> None:
    """Background task: run the compliance checks for a draft and store the report."""
    from app.models.analysis_job import JobStatus
    from app.models.draft import Draft
    from app.services import task as task_service

    try:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Loading draft...",
            )
            draft = await db.get(Draft, draft_id)
            if draft is None:
                raise ValueError(f"Draft {draft_id} not found")
            content = draft.content
            paper_type = draft.paper_type.value
            project = await db.get(Project, draft.project_id)
            journal_guidelines = project.target_journal_guidelines if project else None

        report = await check_compliance(content, paper_type, journal_guidelines)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Compliance check complete.",
                result=report.model_dump(mode="json"),
            )
    except Exception as e:
        logger.exception("Compliance check failed for draft %s", draft_id)
        async with session_factory() as db:
            await task_service.update_job_status(db, job_id, JobStatus.failed, error=str(e))
