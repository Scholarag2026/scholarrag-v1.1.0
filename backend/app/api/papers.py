import os
import re
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory, get_db
from app.dependencies import get_current_user
from app.models.analysis_job import AnalysisJob, JobType
from app.models.user import User
from app.schemas.paper import (
    AddPaperRequest,
    PaperData,
    ProjectPaperListResponse,
    ProjectPaperResponse,
)
from app.schemas.paper_upload import UploadConfirmRequest
from app.schemas.task import TaskCreateResponse
from app.services import paper as paper_service
from app.services import project as project_service
from app.services import task as task_service
from app.services.paper_upload import process_uploaded_files

router = APIRouter(prefix="/projects/{project_id}/papers", tags=["papers"])

_session_factory = async_session_factory

MAX_FILES = 30
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".bib"}
BIBTEX_EXPORT_PAGE_SIZE = 200


@router.post("", response_model=ProjectPaperResponse, status_code=201)
async def add_paper(
    project_id: UUID,
    req: AddPaperRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify project ownership
    await project_service.get_project(db, project_id, user.id)

    if req.paper_data:
        paper = await paper_service.get_or_create_paper(db, req.paper_data)
    else:
        raise HTTPException(status_code=400, detail="paper_data is required")

    pp = await paper_service.add_paper_to_project(
        db, project_id, paper.id,
        relevance_score=req.relevance_score,
        user_notes=req.user_notes,
        tags=req.tags,
    )
    return ProjectPaperResponse.model_validate(pp)


@router.get("", response_model=ProjectPaperListResponse)
async def list_papers(
    project_id: UUID,
    page: int = Query(1, ge=1, le=1_000_000),
    limit: int = Query(50, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    papers, total = await paper_service.list_project_papers(db, project_id, page=page, limit=limit)
    return ProjectPaperListResponse(
        papers=[ProjectPaperResponse.model_validate(pp) for pp in papers],
        total=total,
        page=page,
        limit=limit,
    )


def _paper_to_bibtex(paper) -> str:
    """Convert a Paper model to a BibTeX entry."""
    authors = paper.authors or []
    if authors and isinstance(authors, list) and authors:
        first_author = authors[0].get("name", "Unknown").split()[-1]
    else:
        first_author = "Unknown"
    year = paper.year or "n.d."
    cite_key = f"{first_author}{year}".lower().replace(" ", "")

    lines = [f"@article{{{cite_key},"]
    lines.append(f"  title = {{{paper.title}}},")

    if authors:
        author_str = " and ".join(a.get("name", "") for a in authors)
        lines.append(f"  author = {{{author_str}}},")

    if paper.year:
        lines.append(f"  year = {{{paper.year}}},")
    if paper.journal_name:
        lines.append(f"  journal = {{{paper.journal_name}}},")
    if paper.doi:
        lines.append(f"  doi = {{{paper.doi}}},")
    if paper.abstract:
        lines.append(f"  abstract = {{{paper.abstract}}},")

    lines.append("}")
    return "\n".join(lines)


def _parse_bibtex(content: str) -> list[dict]:
    """Simple BibTeX parser. Extracts title, author, year, journal, doi from entries."""
    entries = []
    pattern = r'@\w+\{[^@]*\}'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        entry = {}
        for field in ["title", "author", "year", "journal", "doi", "abstract"]:
            field_pattern = rf'{field}\s*=\s*\{{([^}}]*)\}}'
            field_match = re.search(field_pattern, match, re.IGNORECASE)
            if field_match:
                entry[field] = field_match.group(1).strip()
        if entry.get("title"):
            entries.append(entry)
    return entries


@router.get("/export")
async def export_papers(
    project_id: UUID,
    format: str = "bibtex",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    papers = []
    page = 1
    while True:
        batch, total = await paper_service.list_project_papers(
            db, project_id, page=page, limit=BIBTEX_EXPORT_PAGE_SIZE
        )
        if not batch:
            break
        papers.extend(batch)
        if len(papers) >= total:
            break
        page += 1

    if format == "bibtex":
        entries = [_paper_to_bibtex(pp.paper) for pp in papers]
        content = "\n\n".join(entries)
        return PlainTextResponse(
            content=content,
            media_type="application/x-bibtex",
            headers={"Content-Disposition": "attachment; filename=references.bib"},
        )

    raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")


@router.post("/import")
async def import_papers(
    project_id: UUID,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)

    content = (await file.read()).decode("utf-8")
    entries = _parse_bibtex(content)

    imported_count = 0
    for entry in entries:
        authors_raw = entry.get("author", "")
        authors = [{"name": a.strip()} for a in authors_raw.split(" and ") if a.strip()]
        year_str = entry.get("year", "")
        year = int(year_str) if year_str.isdigit() else None

        paper_data = PaperData(
            doi=entry.get("doi"),
            title=entry["title"],
            authors=authors,
            year=year,
            journal_name=entry.get("journal"),
            abstract=entry.get("abstract"),
            source_api="manual",
        )

        paper = await paper_service.get_or_create_paper(db, paper_data)
        try:
            await paper_service.add_paper_to_project(db, project_id, paper.id)
            imported_count += 1
        except Exception:
            # Skip duplicates silently
            pass

    return {"imported_count": imported_count}


@router.delete("/{paper_id}")
async def remove_paper(
    project_id: UUID,
    paper_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.get_project(db, project_id, user.id)
    await paper_service.remove_paper_from_project(db, project_id, paper_id)
    return {"success": True}


@router.post("/upload", response_model=TaskCreateResponse, status_code=202)
async def upload_papers(
    project_id: UUID,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload article files for metadata extraction."""
    await project_service.get_project(db, project_id, user.id)

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files per upload")

    files_data = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {f.filename} exceeds 50MB limit")
        files_data.append({
            "name": f.filename or "unknown",
            "content": content,
            "extension": ext,
        })

    # Concurrency guard (D12c: reaps stale jobs before checking)
    await task_service.guard_no_active_job(
        db,
        project_id=project_id,
        job_type=JobType.paper_upload,
        conflict_detail="Upload already in progress",
    )

    job = await task_service.create_job(db, project_id, user.id, JobType.paper_upload)

    background_tasks.add_task(
        process_uploaded_files,
        project_id=project_id,
        job_id=job.id,
        session_factory=_session_factory,
        files_data=files_data,
    )

    return TaskCreateResponse(task_id=job.id)


@router.post("/upload-confirm")
async def confirm_upload(
    project_id: UUID,
    req: UploadConfirmRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm and save uploaded papers after user review."""
    await project_service.get_project(db, project_id, user.id)

    job_result = await db.execute(
        select(AnalysisJob).where(
            and_(
                AnalysisJob.id == req.task_id,
                AnalysisJob.project_id == project_id,
                AnalysisJob.user_id == user.id,
            )
        )
    )
    job = job_result.scalar_one_or_none()
    if not job or not job.result:
        raise HTTPException(status_code=404, detail="Upload job not found")

    fulltext_data = job.result.get("_fulltext_data", {})
    imported_count = 0

    for paper in req.papers:
        paper_data = PaperData(
            doi=paper.doi,
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            journal_name=paper.journal_name,
            abstract=paper.abstract,
            source_api="upload",
        )

        db_paper = await paper_service.get_or_create_paper(db, paper_data)

        ft = fulltext_data.get(str(paper.index))
        if ft:
            metadata = dict(db_paper.metadata_) if db_paper.metadata_ else {}
            metadata["fulltext_status"] = "acquired"
            metadata["fulltext_source"] = "upload"
            metadata["fulltext_chunks"] = ft["chunks"]
            metadata["fulltext_char_count"] = ft["char_count"]
            db_paper.metadata_ = metadata
            await db.flush()

        try:
            await paper_service.add_paper_to_project(db, project_id, db_paper.id)
            imported_count += 1
        except Exception:
            pass  # Skip duplicates

    await db.commit()
    return {"imported_count": imported_count}
