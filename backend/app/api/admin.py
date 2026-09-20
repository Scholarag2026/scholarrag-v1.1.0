from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_admin_user
from app.models.analysis_job import AnalysisJob
from app.models.draft import Draft
from app.models.paper import Paper
from app.models.project import Project
from app.models.user import User, UserRole
from app.models.wos_journal import WosJournal
from app.schemas.admin import AdminUserResponse, SystemStats, UpdateUserRoleRequest
from app.services.wos_import import _detect_collection
from app.services.wos_lookup import import_wos_csv

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[AdminUserResponse])
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(200, ge=1, le=200),
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(
            User,
            func.count(Project.id).label("project_count"),
        )
        .outerjoin(Project, Project.user_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
    )
    rows = await db.execute(stmt)
    results = []
    for user, project_count in rows.all():
        results.append(
            AdminUserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                role=user.role.value,
                expertise_level=user.expertise_level.value,
                created_at=user.created_at,
                project_count=project_count,
            )
        )
    return results


@router.get("/stats", response_model=SystemStats)
async def system_stats(
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    total_projects = (await db.execute(select(func.count(Project.id)))).scalar_one()
    total_papers = (await db.execute(select(func.count(Paper.id)))).scalar_one()
    total_drafts = (await db.execute(select(func.count(Draft.id)))).scalar_one()
    total_jobs = (await db.execute(select(func.count(AnalysisJob.id)))).scalar_one()
    return SystemStats(
        total_users=total_users,
        total_projects=total_projects,
        total_papers=total_papers,
        total_drafts=total_drafts,
        total_jobs=total_jobs,
    )


@router.patch("/users/{user_id}/role", response_model=AdminUserResponse)
async def update_user_role(
    user_id: UUID,
    req: UpdateUserRoleRequest,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        new_role = UserRole(req.role)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid role: {req.role}")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = new_role
    await db.commit()
    await db.refresh(user)

    project_count_result = await db.execute(
        select(func.count(Project.id)).where(Project.user_id == user.id)
    )
    project_count = project_count_result.scalar_one()

    return AdminUserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        expertise_level=user.expertise_level.value,
        created_at=user.created_at,
        project_count=project_count,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: UUID,
    admin: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()
    return {"success": True}


@router.post("/wos-journals/import", status_code=200)
async def import_wos_journals(
    file: UploadFile,
    user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Import WoS Master Journal List from CSV file."""
    content = await file.read()
    csv_text = content.decode("utf-8")
    collection = _detect_collection(file.filename or "")
    count = await import_wos_csv(db, csv_text, collection)
    return {"imported": count, "collection": collection}


@router.get("/wos-journals/count")
async def wos_journal_count(
    user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the number of WoS journals currently loaded."""
    result = await db.execute(select(func.count()).select_from(WosJournal))
    return {"count": result.scalar()}
