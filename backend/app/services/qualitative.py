"""Qualitative coding service — orchestrates coding agent, codebook management, and kappa."""

from __future__ import annotations

import asyncio
import logging
import math
from uuid import UUID

from sklearn.metrics import cohen_kappa_score
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.analysis_agent import AnalysisDependencies
from app.agents.qualitative_agent import (
    format_coding_prompt,
    get_qualitative_coding_agent,
)
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.dataset import Codebook, CoderType, CodingSession, CodingStatus, Dataset
from app.models.project import Project
from app.services import task as task_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inter-coder reliability (Cohen's kappa)
# ---------------------------------------------------------------------------


def _finite_or_none(value) -> float | None:
    """Return a plain float, or None when the value is missing or non-finite.

    Cohen's kappa is genuinely undefined (NaN) whenever observed and chance agreement are
    both 1.0. Product decision D5: report that as null, never as a fabricated 1.0/0.0.
    """
    if value is None:
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def compute_inter_coder_reliability(
    ai_codes: dict[str, list[str]],
    human_codes: dict[str, list[str]],
    all_codes: list[str],
) -> dict:
    """Compute Cohen's kappa between AI and human coding.

    Parameters
    ----------
    ai_codes : dict mapping segment_id -> list of code names (first = primary)
    human_codes : dict mapping segment_id -> list of code names (first = primary)
    all_codes : list of all possible code names

    Returns
    -------
    dict with overall_kappa (float | None), overall_kappa_explanation (str | None),
    per_code_kappa, agreement_pct, disagreement_segments.

    Kappa is ``None`` whenever it is mathematically undefined — never a fabricated
    1.0 or 0.0 (product decision D5).
    """
    # Only compare segments present in both
    common_segments = sorted(set(ai_codes.keys()) & set(human_codes.keys()))

    if not common_segments:
        return {
            "overall_kappa": None,
            "overall_kappa_explanation": (
                "No segments were coded by both coders, so Cohen's kappa is not computable."
            ),
            "per_code_kappa": [],
            "agreement_pct": 0.0,
            "disagreement_segments": [],
        }

    # Build primary code vectors
    ai_primary = [ai_codes[s][0] if ai_codes[s] else "" for s in common_segments]
    human_primary = [human_codes[s][0] if human_codes[s] else "" for s in common_segments]

    # Overall kappa on primary codes
    labels = sorted(set(ai_primary + human_primary + all_codes))
    overall_kappa = _finite_or_none(
        cohen_kappa_score(ai_primary, human_primary, labels=labels)
    )
    overall_kappa_explanation = None
    if overall_kappa is None:
        overall_kappa_explanation = (
            "Cohen's kappa is undefined here: both coders assigned the same single code to "
            "every shared segment, so chance agreement is already 100%."
        )

    # Agreement percentage
    agreements = sum(1 for a, h in zip(ai_primary, human_primary) if a == h)
    agreement_pct = (agreements / len(common_segments)) * 100.0

    # Per-code binary kappa
    per_code_kappa = []
    for code in all_codes:
        ai_binary = [1 if a == code else 0 for a in ai_primary]
        human_binary = [1 if h == code else 0 for h in human_primary]

        if not any(ai_binary) and not any(human_binary):
            # Neither coder ever used this code — kappa is undefined, not perfect (D5).
            kappa = None
        else:
            kappa = _finite_or_none(cohen_kappa_score(ai_binary, human_binary))

        code_agreements = sum(1 for a, h in zip(ai_binary, human_binary) if a == h)
        code_agreement_pct = (code_agreements / len(common_segments)) * 100.0

        per_code_kappa.append({
            "code_name": code,
            "kappa": kappa,
            "agreement_pct": float(code_agreement_pct),
        })

    # Disagreement segments
    disagreement_segments = []
    for seg_id, a, h in zip(common_segments, ai_primary, human_primary):
        if a != h:
            disagreement_segments.append({
                "segment_id": seg_id,
                "ai_codes": ai_codes[seg_id],
                "human_codes": human_codes[seg_id],
            })

    return {
        "overall_kappa": overall_kappa,
        "overall_kappa_explanation": overall_kappa_explanation,
        "per_code_kappa": per_code_kappa,
        "agreement_pct": float(agreement_pct),
        "disagreement_segments": disagreement_segments,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def get_latest_coding_result(
    db: AsyncSession, dataset_id: UUID
) -> dict | None:
    """Get the most recent completed qualitative job result for a dataset."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.dataset_id == dataset_id,
                AnalysisJob.job_type == JobType.qualitative,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None


async def get_codebook(
    db: AsyncSession, dataset_id: UUID
) -> dict | None:
    """Get the latest codebook for a dataset."""
    result = await db.execute(
        select(Codebook)
        .where(Codebook.dataset_id == dataset_id)
        .order_by(Codebook.created_at.desc())
        .limit(1)
    )
    codebook = result.scalar_one_or_none()
    if codebook is None:
        return None
    return {
        "id": str(codebook.id),
        "codes": codebook.codes,
        "themes": codebook.themes,
    }


async def get_latest_inter_coder_result(
    db: AsyncSession, dataset_id: UUID
) -> dict | None:
    """Get the most recent completed inter-coder reliability result."""
    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.dataset_id == dataset_id,
                AnalysisJob.job_type == JobType.inter_coder,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return job.result if job else None


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def run_qualitative_coding(
    project_id: UUID,
    dataset_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: generate qualitative coding for a dataset."""
    try:
        # --- Phase 1: Load project and dataset ---
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
                progress_message="Preparing qualitative coding...",
            )

        # --- Phase 2: Get existing codebook if available ---
        existing_codebook = None
        async with session_factory() as db:
            codebook_data = await get_codebook(db, dataset_id)
            if codebook_data:
                existing_codebook = codebook_data

        # --- Phase 3: Build text segments from dataset columns ---
        # Extract text columns from dataset metadata
        text_segments = []
        columns = dataset.columns or []
        text_cols = [c["name"] for c in columns if c.get("dtype") == "text"]

        if text_cols:
            # Parquet read and row materialization are both blocking; keep them off the
            # event loop (D20 / issue #27).
            from app.services.dataset import load_dataset_df_async
            df = await load_dataset_df_async(dataset)

            def _build_segments() -> list[str]:
                segments: list[str] = []
                for col in text_cols:
                    for val in df[col].dropna().tolist():
                        segments.append(str(val))
                return segments

            text_segments = await asyncio.to_thread(_build_segments)

        if not text_segments:
            text_segments = ["No text data available for coding."]

        # --- Phase 4: Build research questions ---
        research_questions = []
        if project.description:
            research_questions = [project.description]

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.4,
                progress_message="Running qualitative coding agent...",
            )

        # --- Phase 5: Run agent ---
        prompt = format_coding_prompt(
            text_segments=text_segments,
            research_questions=research_questions,
            existing_codebook=existing_codebook,
        )

        deps = AnalysisDependencies(
            project_id=str(project_id),
            project_description=project.description,
            target_journal=project.target_journal,
            citation_style=(
                project.citation_style.value if project.citation_style else "APA"
            ),
        )

        agent = get_qualitative_coding_agent()
        result = await agent.run(prompt, deps=deps)
        result_dict = result.output.model_dump()

        # --- Phase 6: Save codebook and coding session ---
        async with session_factory() as db:
            # Save codebook
            new_codebook = Codebook(
                project_id=project_id,
                user_id=dataset.user_id,
                dataset_id=dataset_id,
                codes=result_dict["codebook"]["codes"],
                themes=result_dict["codebook"]["themes"],
            )
            db.add(new_codebook)
            await db.flush()

            # Save AI coding session
            coding_session = CodingSession(
                user_id=dataset.user_id,
                codebook_id=new_codebook.id,
                coder_type=CoderType.ai,
                coded_segments=result_dict["coded_segments"],
                progress=1.0,
                status=CodingStatus.completed,
            )
            db.add(coding_session)
            await db.commit()

        # --- Phase 7: Mark complete ---
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Qualitative coding complete",
                result=result_dict,
            )

    except Exception as e:
        logger.exception(
            "Qualitative coding failed for dataset %s", dataset_id
        )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )


async def run_inter_coder_reliability(
    dataset_id: UUID,
    job_id: UUID,
    session_factory,
) -> None:
    """Background task: compute inter-coder reliability (Cohen's kappa)."""
    try:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.1,
                progress_message="Loading coding sessions...",
            )

        # Get the latest codebook for this dataset
        async with session_factory() as db:
            codebook_result = await db.execute(
                select(Codebook)
                .where(Codebook.dataset_id == dataset_id)
                .order_by(Codebook.created_at.desc())
                .limit(1)
            )
            codebook = codebook_result.scalar_one_or_none()
            if not codebook:
                raise ValueError("No codebook found for dataset")

            # Get AI coding session
            ai_result = await db.execute(
                select(CodingSession)
                .where(
                    and_(
                        CodingSession.codebook_id == codebook.id,
                        CodingSession.coder_type == CoderType.ai,
                        CodingSession.status == CodingStatus.completed,
                    )
                )
                .order_by(CodingSession.created_at.desc())
                .limit(1)
            )
            ai_session = ai_result.scalar_one_or_none()

            # Get human coding session
            human_result = await db.execute(
                select(CodingSession)
                .where(
                    and_(
                        CodingSession.codebook_id == codebook.id,
                        CodingSession.coder_type == CoderType.human,
                        CodingSession.status == CodingStatus.completed,
                    )
                )
                .order_by(CodingSession.created_at.desc())
                .limit(1)
            )
            human_session = human_result.scalar_one_or_none()

        if not ai_session or not human_session:
            async with session_factory() as db:
                await task_service.update_job_status(
                    db, job_id, JobStatus.failed,
                    error="Both AI and human coding sessions are required.",
                )
            return

        # Build code maps from segments
        ai_codes: dict[str, list[str]] = {}
        for seg in (ai_session.coded_segments or []):
            ai_codes[seg["segment_id"]] = seg.get("codes", [])

        human_codes: dict[str, list[str]] = {}
        for seg in (human_session.coded_segments or []):
            human_codes[seg["segment_id"]] = seg.get("codes", [])

        # Gather all code names from codebook
        all_codes = [c["name"] for c in (codebook.codes or [])]

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.running,
                progress=0.5,
                progress_message="Computing Cohen's kappa...",
            )

        # Compute reliability
        report = compute_inter_coder_reliability(ai_codes, human_codes, all_codes)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.completed,
                progress=1.0,
                progress_message="Inter-coder reliability analysis complete",
                result=report,
            )

    except Exception as e:
        logger.exception(
            "Inter-coder reliability failed for dataset %s", dataset_id
        )
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id, JobStatus.failed, error=str(e)
            )
