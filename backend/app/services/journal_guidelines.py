"""Background job: extract journal author guidelines and persist them on the project."""

from __future__ import annotations

import logging
from uuid import UUID

from app.agents.journal_guidelines_agent import (
    fetch_journal_guidelines,
    get_journal_guidelines_agent,
)
from app.models.analysis_job import JobStatus
from app.models.project import CitationStyle, Project
from app.services import task as task_service

logger = logging.getLogger(__name__)


def _match_citation_style(extracted_style: str | None) -> CitationStyle | None:
    """Map a free-text citation style onto the CitationStyle enum."""
    if not extracted_style:
        return None
    style_upper = extracted_style.strip().upper()
    for cs in CitationStyle:
        if cs.value.upper() in style_upper or style_upper in cs.value.upper():
            return cs
    return CitationStyle.custom


async def run_fetch_journal_guidelines(
    project_id: UUID,
    guidelines_text: str | None,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: extract guidelines from pasted text or web search, then persist."""
    try:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Reading journal guidelines...",
            )
            project = await db.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found")
            target_journal = project.target_journal

        if guidelines_text and guidelines_text.strip():
            agent = get_journal_guidelines_agent()
            prompt = (
                f"Journal: {target_journal}\n\n"
                f"Author Guidelines Text:\n{guidelines_text[:15000]}"
            )
            result = await agent.run(prompt)
            guidelines = result.output
            guidelines.journal_name = target_journal
            guidelines.raw_guidelines_text = guidelines_text[:5000]
        else:
            guidelines = await fetch_journal_guidelines(target_journal)

        guidelines_dict = guidelines.model_dump()

        async with session_factory() as db:
            project = await db.get(Project, project_id)
            if project is not None:
                project.target_journal_guidelines = guidelines_dict
                matched = _match_citation_style(guidelines.citation_style)
                if matched is not None:
                    project.citation_style = matched
                await db.commit()

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Journal guidelines extracted.",
                result=guidelines_dict,
            )
    except Exception as e:
        logger.exception("Journal guidelines extraction failed for project %s", project_id)
        async with session_factory() as db:
            await task_service.update_job_status(db, job_id, JobStatus.failed, error=str(e))
