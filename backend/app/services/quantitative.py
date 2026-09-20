"""Quantitative analysis service — orchestrates plan, descriptive stats, and interpretation."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.quantitative_agent import (
    format_interpretation_prompt,
    format_plan_prompt,
    get_quantitative_interpretation_agent,
    get_quantitative_plan_agent,
)
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.dataset import Dataset
from app.models.project import Project
from app.schemas.quantitative import DescriptiveStats
from app.services import task as task_service
from app.services.dataset import load_dataset_df_async
from app.services.descriptive_stats import compute_all_descriptive_stats

logger = logging.getLogger(__name__)


async def get_latest_quantitative(
    db: AsyncSession, dataset_id: UUID
) -> dict | None:
    """Get the most recent completed quantitative analysis job result for a dataset."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.dataset_id == dataset_id,
                AnalysisJob.job_type == JobType.quantitative,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None


async def run_quantitative_analysis(
    project_id: UUID,
    dataset_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: 3-stage pipeline — plan → descriptive stats → interpretation."""
    try:
        # --- Stage 1: Load project and dataset, mark running ---
        async with session_factory() as db:
            project = await db.get(Project, project_id)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            dataset = await db.get(Dataset, dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")

            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Generating analysis plan...",
            )

        # --- Stage 2: Build analysis plan via agent ---
        columns = dataset.columns or []
        research_questions = []
        if project.description:
            research_questions = [project.description]

        plan_prompt = format_plan_prompt(
            columns=columns,
            research_questions=research_questions,
            project_description=project.description,
        )

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description,
            target_journal=project.target_journal,
            citation_style=(
                project.citation_style.value if project.citation_style else "APA"
            ),
        )

        plan_agent = get_quantitative_plan_agent()
        plan_result = await plan_agent.run(plan_prompt, deps=deps)
        plan = plan_result.output

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.4,
                progress_message="Computing descriptive statistics...",
            )

        # --- Stage 3: Compute descriptive statistics ---
        # Parquet read and pandas/scipy stats computation are both blocking; keep them
        # off the event loop (D20 / issue #27).
        df = await load_dataset_df_async(dataset)
        stats_result = await asyncio.to_thread(compute_all_descriptive_stats, df)
        # Validate through the pydantic schema so numpy scalars are coerced away before
        # the value is ever handed to the JSONB writer (issue #12).
        descriptive_stats = DescriptiveStats.model_validate(
            stats_result
        ).model_dump(mode="json")

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.7,
                progress_message="Interpreting results...",
            )

        # --- Stage 4: Interpret results via agent ---
        interp_prompt = format_interpretation_prompt(
            descriptive_stats=descriptive_stats,
            plan_methods=[m.model_dump() for m in plan.methods],
            research_questions=research_questions if research_questions else None,
        )

        interp_agent = get_quantitative_interpretation_agent()
        interp_result = await interp_agent.run(interp_prompt, deps=deps)
        final = interp_result.output

        # Merge the computed plan into the result
        result_dict = final.model_dump()
        result_dict["plan"] = plan.model_dump()
        result_dict["descriptive_stats"] = descriptive_stats

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Quantitative analysis complete",
                result=result_dict,
            )

    except Exception as e:
        logger.exception(
            "Quantitative analysis failed for dataset %s", dataset_id
        )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )
