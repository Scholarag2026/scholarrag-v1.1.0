import asyncio
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.refiner_agent import refine_text
from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import JobStatus, JobType
from app.models.paper import Paper
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.schemas.draft import (
    DraftCreate,
    DraftDetailResponse,
    DraftListResponse,
    DraftResponse,
    DraftUpdate,
    DraftVersionResponse,
    GenerateRequest,
)
from app.schemas.refine import RefineRequest
from app.schemas.task import TaskCreateResponse
from app.services import draft as draft_service
from app.services import project as project_service
from app.services import task as task_service
from app.services.citation_render import render_document
from app.services.compliance import run_compliance_check
from app.services.writing import generate_section

router = APIRouter(tags=["drafts"])

# Module-level session factory — overridden in tests to use test DB
_session_factory = async_session_factory


@router.post("/projects/{project_id}/drafts", response_model=DraftResponse, status_code=201)
async def create_draft(
    project_id: UUID,
    req: DraftCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    draft = await draft_service.create_draft(
        db, project_id, user.id, req.title, req.paper_type,
    )
    return DraftResponse.model_validate(draft)


@router.get("/projects/{project_id}/drafts", response_model=DraftListResponse)
async def list_drafts(
    project_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    drafts, total = await draft_service.list_drafts(
        db, project_id, page=page, limit=limit
    )
    return DraftListResponse(
        drafts=[DraftResponse.model_validate(d) for d in drafts],
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/drafts/{draft_id}", response_model=DraftDetailResponse)
async def get_draft(
    draft_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await draft_service.get_draft(db, draft_id, user.id)
    versions = await draft_service.list_draft_version_metadata(db, draft_id)
    return DraftDetailResponse(
        draft=DraftResponse.model_validate(draft),
        versions=[DraftVersionResponse(**v) for v in versions],
    )


@router.put("/drafts/{draft_id}", response_model=DraftResponse)
async def update_draft(
    draft_id: UUID,
    req: DraftUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await draft_service.update_draft(
        db, draft_id, user.id,
        title=req.title,
        content=req.content,
        status=req.status,
    )
    return DraftResponse.model_validate(draft)


@router.delete("/drafts/{draft_id}")
async def delete_draft(
    draft_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await draft_service.delete_draft(db, draft_id, user.id)
    return {"success": True}


@router.post("/drafts/{draft_id}/generate", response_model=TaskCreateResponse)
async def generate_draft_section(
    draft_id: UUID,
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    draft = await draft_service.get_draft(db, draft_id, user.id)
    job = await task_service.create_job(db, draft.project_id, user.id, JobType.writing)

    background_tasks.add_task(
        generate_section,
        draft=draft,
        section_type=req.section_type,
        context=req.context,
        job_id=job.id,
        session_factory=_session_factory,
        language=req.language,
        locked_blocks=[b.model_dump() for b in req.locked_blocks] if req.locked_blocks else None,
        expertise_level=user.expertise_level.value if user.expertise_level else None,
        target_words=req.target_words,
        title=req.section_title,
    )

    return TaskCreateResponse(task_id=job.id)


@router.post(
    "/drafts/{draft_id}/check-compliance",
    response_model=TaskCreateResponse,
    status_code=202,
)
async def check_draft_compliance(
    draft_id: UUID,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a background compliance check. Poll GET /tasks/{task_id} for the report."""
    draft = await draft_service.get_draft(db, draft_id, user.id)
    job = await task_service.create_job(
        db, draft.project_id, user.id, JobType.compliance_check
    )

    background_tasks.add_task(
        run_compliance_check,
        draft_id=draft_id,
        job_id=job.id,
        session_factory=_session_factory,
    )

    return TaskCreateResponse(task_id=job.id)


@router.get("/drafts/{draft_id}/export")
async def export_draft(
    draft_id: UUID,
    format: str = "docx",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a draft in the specified format (docx, pdf, latex), with in-text citations
    and a reference list rendered in the project's chosen citation style. The draft row
    itself is never written."""
    from app.services import export as export_service

    draft = await draft_service.get_draft(db, draft_id, user.id)
    project = await project_service.get_project(db, draft.project_id, user.id)
    # Only the columns `render_document` reads:
    # `select(Paper)` would also pull the `metadata` JSONB column, holding
    # `fulltext_chunks` and `deep_analysis`, for every paper in the project, on a
    # synchronous request path served by two workers. `app.api.paper_chat` avoids the
    # same hazard the same way.
    papers_result = await db.execute(
        select(
            Paper.authors.label("authors"),
            Paper.year.label("year"),
            Paper.title.label("title"),
            Paper.journal_name.label("journal_name"),
            Paper.doi.label("doi"),
        )
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .where(ProjectPaper.project_id == draft.project_id)
    )
    papers = list(papers_result.all())
    citation_style = getattr(project.citation_style, "value", project.citation_style) or "APA"
    rendered = await asyncio.to_thread(render_document, draft.content, papers, citation_style)
    content = rendered.content

    # weasyprint and python-docx are pure-CPU and blocking; with --workers 2 running them
    # on the event loop stalls half the app's request capacity for the whole export (D20).
    #
    # Decision (issue #27 remediation): D20's "convert export to a BackgroundTasks job"
    # requirement is explicitly dropped in favor of keeping this a synchronous streaming
    # response wrapped in asyncio.to_thread. Exports are single-document, bounded-size CPU
    # work with no chunked/bulk-insert equivalent (unlike WoS import), and the client needs
    # the file bytes back on this request rather than polling a job id. If export payloads
    # grow large enough to need the job pattern, revisit this decision then.
    if format == "docx":
        output = await asyncio.to_thread(export_service.export_docx, draft.title, content)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ext = "docx"
    elif format == "pdf":
        output = await asyncio.to_thread(export_service.export_pdf, draft.title, content)
        media_type = "application/pdf"
        ext = "pdf"
    elif format == "latex":
        output = await asyncio.to_thread(export_service.export_latex, draft.title, content)
        media_type = "application/x-latex"
        ext = "tex"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: docx, pdf, latex",
        )

    filename = f"{draft.title.replace(' ', '_')}.{ext}"
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _run_refine(
    draft_id: UUID,
    job_id: UUID,
    session_factory,
    text: str,
    context: str | None,
    scope: str,
    section_key: str | None = None,
) -> None:
    """Background task for text refinement."""
    try:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.running,
                progress=0.1,
                progress_message="Refining text...",
            )

        result = await refine_text(text=text, context=context, scope=scope)

        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.completed,
                progress=1.0,
                progress_message="Refinement complete",
                result={"original": text, "refined": result.refined, "refined_text": result.refined, "section_key": section_key},
            )
    except Exception as e:
        async with session_factory() as db:
            await task_service.update_job_status(
                db, job_id,
                status=JobStatus.failed,
                error=str(e),
            )


@router.post("/drafts/{draft_id}/refine", response_model=TaskCreateResponse, status_code=202)
async def refine_draft_text(
    draft_id: UUID,
    req: RefineRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Refine selected text or an entire section."""
    draft = await draft_service.get_draft(db, draft_id, user.id)
    job = await task_service.create_job(db, draft.project_id, user.id, JobType.refine)

    background_tasks.add_task(
        _run_refine,
        draft_id=draft_id,
        job_id=job.id,
        session_factory=_session_factory,
        text=req.text,
        context=req.context,
        scope=req.scope,
        section_key=req.section_key,
    )

    return TaskCreateResponse(task_id=job.id)
