"""API routes for full-text acquisition, upload, paste, and claim verification."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.paper import Paper
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.schemas.fulltext import (
    ClaimVerificationReport,
    FulltextChunkResponse,
    PasteFulltextRequest,
)
from app.schemas.task import TaskCreateResponse
from app.services import draft as draft_service
from app.services import project as project_service
from app.services import task as task_service
from app.services.fulltext import (
    acquire_full_texts,
    chunk_text,
    drop_reference_and_backmatter_chunks,
    extract_text_from_pdf,
    verify_and_heal_claims,
    verify_user_edits,
)

router = APIRouter()
_session_factory = async_session_factory

# Newest completed claim_verify jobs to inspect for a given project.
CLAIM_VERIFY_LOOKBACK: int = 20


# ---------------------------------------------------------------------------
# Request schemas (endpoint-specific, kept local to avoid circular imports)
# ---------------------------------------------------------------------------


class AcquireFullTextsRequest(BaseModel):
    """Optional list of paper IDs to acquire. Empty = all papers."""
    paper_ids: list[UUID] = []


class VerifyAndHealRequest(BaseModel):
    """Request body for the verify-and-heal endpoint."""

    draft_id: UUID


class VerifyEditsRequest(BaseModel):
    """Request body for the verify-edits endpoint."""

    draft_id: UUID
    changed_sections: list[str]


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/acquire-full-texts  (202)
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/acquire-full-texts",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def acquire_full_texts_endpoint(
    project_id: UUID,
    req: AcquireFullTextsRequest = AcquireFullTextsRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start Unpaywall-first cascade to acquire full texts for all project papers."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.fulltext_acquire,
        conflict_detail="Full-text acquisition already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.fulltext_acquire
    )

    background_tasks.add_task(
        acquire_full_texts,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        paper_ids=req.paper_ids or None,
    )

    return TaskCreateResponse(task_id=job.id)


# ---------------------------------------------------------------------------
# POST /papers/{paper_id}/upload-fulltext  (200)
# ---------------------------------------------------------------------------


@router.post("/papers/{paper_id}/upload-fulltext", status_code=200)
async def upload_fulltext(
    paper_id: UUID,
    file: UploadFile,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF file and extract full text for a paper."""
    paper = await _get_paper(db, paper_id)

    if not file.content_type or "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # PDF parsing (pymupdf) and chunking are CPU-bound and must not block the
    # event loop (D20 / issue #27).
    text = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)
    chunks = await asyncio.to_thread(chunk_text, text)

    metadata = dict(paper.metadata_) if paper.metadata_ else {}
    metadata["fulltext_status"] = "acquired"
    metadata["fulltext_source"] = "upload"
    metadata["fulltext_chunks"] = chunks
    paper.metadata_ = metadata

    await db.flush()
    await db.commit()

    return {
        "paper_id": str(paper.id),
        "fulltext_status": "acquired",
        "fulltext_source": "upload",
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------------------
# POST /papers/{paper_id}/paste-fulltext  (200)
# ---------------------------------------------------------------------------


@router.post("/papers/{paper_id}/paste-fulltext", status_code=200)
async def paste_fulltext(
    paper_id: UUID,
    req: PasteFulltextRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Accept pasted full text for a paper, chunk and store it."""
    paper = await _get_paper(db, paper_id)

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # chunk_text is CPU-bound and must not block the event loop (D20 / issue #27).
    chunks = await asyncio.to_thread(chunk_text, req.text)

    metadata = dict(paper.metadata_) if paper.metadata_ else {}
    metadata["fulltext_status"] = "acquired"
    metadata["fulltext_source"] = "paste"
    metadata["fulltext_chunks"] = chunks
    paper.metadata_ = metadata

    await db.flush()
    await db.commit()

    return {
        "paper_id": str(paper.id),
        "fulltext_status": "acquired",
        "fulltext_source": "paste",
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/verify-and-heal  (202)
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/verify-and-heal",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def verify_and_heal_endpoint(
    project_id: UUID,
    req: VerifyAndHealRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Phase A: verify all claims in a draft and self-heal failures."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.claim_verify,
        conflict_detail="Claim verification already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.claim_verify
    )

    background_tasks.add_task(
        verify_and_heal_claims,
        project_id=project_id,
        draft_id=req.draft_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/verify-edits  (202)
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/verify-edits",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def verify_edits_endpoint(
    project_id: UUID,
    req: VerifyEditsRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Phase B: re-verify only changed sections after user edits."""
    await project_service.get_project(db, project_id, user.id)

    # Concurrency guard (D12c: reaps stale jobs before checking; same job type as
    # verify-and-heal)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.claim_verify,
        conflict_detail="Claim verification already in progress",
    )

    job = await task_service.create_job(
        db, project_id, user.id, JobType.claim_verify
    )

    background_tasks.add_task(
        verify_user_edits,
        project_id=project_id,
        draft_id=req.draft_id,
        changed_sections=req.changed_sections,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


# ---------------------------------------------------------------------------
# GET /drafts/{draft_id}/claim-verification
# ---------------------------------------------------------------------------


@router.get(
    "/drafts/{draft_id}/claim-verification",
    response_model=ClaimVerificationReport,
)
async def get_claim_verification(
    draft_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current claim verification report for a draft.

    The draft is authorized first (404 for anything the caller does not own); the job lookup is
    then scoped to that draft's project instead of scanning every completed job platform-wide.
    """
    draft = await draft_service.get_draft(db, draft_id, user.id)

    result = await db.execute(
        select(AnalysisJob)
        .where(
            and_(
                AnalysisJob.project_id == draft.project_id,
                AnalysisJob.job_type == JobType.claim_verify,
                AnalysisJob.status == JobStatus.completed,
            )
        )
        .order_by(AnalysisJob.created_at.desc())
        .limit(CLAIM_VERIFY_LOOKBACK)
    )
    jobs = result.scalars().all()

    for job in jobs:
        if job.result and str(job.result.get("draft_id")) == str(draft_id):
            # task_id always reflects the job actually matched here, overriding whatever
            # (if anything) is stored in the job's own result JSON.
            return ClaimVerificationReport(**{**job.result, "task_id": job.id})

    # Project has no matching verification job yet — report honestly as empty.
    return ClaimVerificationReport(
        draft_id=draft_id,
        verifications=[],
        verified_count=0,
        unsupported_count=0,
        nuance_count=0,
        abstract_only_count=0,
    )


# ---------------------------------------------------------------------------
# GET /papers/{paper_id}/fulltext-chunks/{chunk_index}
# ---------------------------------------------------------------------------


@router.get(
    "/papers/{paper_id}/fulltext-chunks/{chunk_index}",
    response_model=FulltextChunkResponse,
)
async def get_fulltext_chunk(
    paper_id: UUID,
    chunk_index: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read-only: one paper's full-text chunk, by 0-based index.
    Filtered the same way `_verify_one` reads a paper's chunks
    (`drop_reference_and_backmatter_chunks`), so index N here is the same chunk a claim
    quoting "chunk N+1" (`ClaimVerification.evidence_location`) was actually verified
    against. Used by the delivered-evidence record (``demo/run_demo.py``), which needs a
    paper's chunk text by index.

    Scoped to a project the caller owns: papers are deduplicated by DOI, so a bare paper
    id does not prove the caller ever added it to their own library, and this is the
    first route in the file that returns stored full text rather than accepting a write
    to an id the caller must already hold.

    ``abstract`` is returned alongside every chunk of the
    same paper, not only chunk 0 -- it is not itself one of the numbered chunks the
    verifier's own ``evidence_location`` counts, so there is no dedicated index for it,
    but the delivered-evidence record's own passage locator needs it regardless of
    which chunk a given claim's ``evidence_location`` happens to name.
    """
    paper = await _get_paper_for_user(db, paper_id, user.id)
    stored_chunks = [c for c in (paper.metadata_ or {}).get("fulltext_chunks", []) if "text" in c]
    filtered_chunks, _dropped = drop_reference_and_backmatter_chunks(stored_chunks)
    if chunk_index < 0 or chunk_index >= len(filtered_chunks):
        raise HTTPException(status_code=404, detail="Chunk index out of range")
    chunk = filtered_chunks[chunk_index]
    return FulltextChunkResponse(
        paper_id=paper.id,
        chunk_index=chunk_index,
        chunk_count=len(filtered_chunks),
        section=chunk.get("section"),
        text=chunk["text"],
        abstract=paper.abstract,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_paper(db: AsyncSession, paper_id: UUID) -> Paper:
    """Fetch a paper by ID or raise 404."""
    result = await db.execute(select(Paper).where(Paper.id == paper_id))
    paper = result.scalar_one_or_none()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


async def _get_paper_for_user(db: AsyncSession, paper_id: UUID, user_id: UUID) -> Paper:
    """Fetch a paper by ID, scoped to a project the caller owns, or raise 404.

    Papers are deduplicated by DOI, so knowing
    a paper id is not proof the caller ever added that paper to a project of their own.
    """
    result = await db.execute(
        select(Paper)
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .join(Project, Project.id == ProjectPaper.project_id)
        .where(Paper.id == paper_id, Project.user_id == user_id)
    )
    paper = result.scalars().first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper
