from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from app.services import project as project_service

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=201)
async def create(
    req: ProjectCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    p = await project_service.create_project(
        db, user.id, req.title, req.description, req.target_journal, req.citation_style
    )
    return ProjectResponse.model_validate(p)


@router.get("", response_model=ProjectListResponse)
async def list_all(
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    projects, total = await project_service.list_projects(
        db, user.id, page=page, limit=limit
    )
    return ProjectListResponse(
        projects=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        page=page,
        limit=limit,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_one(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    p = await project_service.get_project(db, project_id, user.id)
    return ProjectResponse.model_validate(p)


@router.put("/{project_id}", response_model=ProjectResponse)
async def update(
    project_id: UUID,
    req: ProjectUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    p = await project_service.update_project(
        db, project_id, user.id, **req.model_dump(exclude_unset=True)
    )
    return ProjectResponse.model_validate(p)


@router.delete("/{project_id}")
async def delete(
    project_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await project_service.delete_project(db, project_id, user.id)
    return {"success": True}
