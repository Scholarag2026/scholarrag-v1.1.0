"""Research design service — orchestrates research design and data collection agents."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.data_collection_agent import (
    format_collection_prompt,
    get_data_collection_agent,
)
from app.agents.research_design_agent import (
    format_research_design_prompt,
    get_research_design_agent,
)
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.project import Project
from app.services import task as task_service

logger = logging.getLogger(__name__)


async def get_latest_research_design(
    db: AsyncSession, project_id: UUID
) -> dict | None:
    """Get the most recent completed research design result."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.research_design,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None


async def get_latest_collection_plan(
    db: AsyncSession, project_id: UUID
) -> dict | None:
    """Get the most recent completed data collection plan result."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.project_id == project_id,
                AnalysisJob.job_type == JobType.data_collection,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None


async def run_research_design(
    project_id: UUID,
    job_id: UUID,
    session_factory,
    research_question: str,
) -> None:
    """Background task: generate research design."""
    try:
        async with session_factory() as db:
            project = await db.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Generating research design...",
            )

        # Get gap analysis summary if available
        gap_summary = None
        async with session_factory() as db:
            gap_result = await db.execute(
                select(AnalysisJob)
                .where(
                    and_(
                        AnalysisJob.project_id == project_id,
                        AnalysisJob.job_type == JobType.gap_analysis,
                        AnalysisJob.status == JobStatus.completed,
                    )
                )
                .order_by(AnalysisJob.created_at.desc())
                .limit(1)
            )
            gap_job = gap_result.scalar_one_or_none()
            if gap_job and gap_job.result:
                gap_summary = gap_job.result.get("summary")

        prompt = format_research_design_prompt(
            research_question=research_question,
            project_description=project.description,
            target_journal=project.target_journal,
            gap_summary=gap_summary,
        )

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description,
            target_journal=project.target_journal,
            citation_style=(
                project.citation_style.value if project else "APA"
            ),
        )

        agent = get_research_design_agent()
        result = await agent.run(prompt, deps=deps)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Research design complete",
                result=result.output.model_dump(),
            )

    except Exception as e:
        logger.exception(
            "Research design failed for project %s", project_id
        )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )


async def run_data_collection_plan(
    project_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: generate data collection plan from latest research design."""
    try:
        async with session_factory() as db:
            project = await db.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

        # Get latest research design
        async with session_factory() as db:
            design = await get_latest_research_design(db, project_id)
            if not design:
                await task_service.update_job_status(
                    db, job_id, JobStatus.failed,
                    error="No research design found. Generate one first.",
                )
                return

            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Generating data collection plan...",
            )

        prompt = format_collection_prompt(
            design=design,
            project_description=project.description,
        )

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description,
            target_journal=project.target_journal,
            citation_style=(
                project.citation_style.value if project else "APA"
            ),
        )

        agent = get_data_collection_agent()
        result = await agent.run(prompt, deps=deps)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Data collection plan complete",
                result=result.output.model_dump(),
            )

    except Exception as e:
        logger.exception(
            "Data collection plan failed for project %s", project_id
        )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )
