from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import CitationStyle, Project, ProjectStatus


async def create_project(
    db: AsyncSession,
    user_id: UUID,
    title: str,
    description: str | None = None,
    target_journal: str | None = None,
    citation_style: str = "APA",
) -> Project:
    # Convert string to enum
    cs = CitationStyle(citation_style)
    project = Project(
        user_id=user_id,
        title=title,
        description=description,
        target_journal=target_journal,
        citation_style=cs,
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return project


async def list_projects(
    db: AsyncSession, user_id: UUID, page: int = 1, limit: int = 50
) -> tuple[list[Project], int]:
    """List a user's projects with pagination. Returns (rows, total)."""
    count_result = await db.execute(
        select(func.count()).select_from(Project).where(Project.user_id == user_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.updated_at.desc(), Project.id)
        .offset((page - 1) * limit)
        .limit(limit)
    )
    return list(result.scalars().all()), total


async def get_project(db: AsyncSession, project_id: UUID, user_id: UUID) -> Project:
    result = await db.execute(
        select(Project).where(Project.id == project_id, Project.user_id == user_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


async def update_project(
    db: AsyncSession, project_id: UUID, user_id: UUID, **kwargs
) -> Project:
    project = await get_project(db, project_id, user_id)

    # Track if journal changed so we can clear stale guidelines
    old_journal = project.target_journal

    for key, value in kwargs.items():
        # Convert string values to enums where needed
        if key == "citation_style" and value is not None:
            value = CitationStyle(value)
        elif key == "status" and value is not None:
            value = ProjectStatus(value)

        # Allow clearing fields: empty string → None for nullable fields
        if key in ("target_journal", "description") and value == "":
            value = None

        if value is not None or key in ("target_journal", "description"):
            setattr(project, key, value)

    # If journal was changed or cleared, invalidate stored guidelines
    if "target_journal" in kwargs and project.target_journal != old_journal:
        project.target_journal_guidelines = None

    await db.flush()
    await db.refresh(project)
    return project


async def delete_project(
    db: AsyncSession, project_id: UUID, user_id: UUID
) -> None:
    project = await get_project(db, project_id, user_id)
    await db.delete(project)
