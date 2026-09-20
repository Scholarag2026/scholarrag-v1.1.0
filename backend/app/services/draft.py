from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.draft import Draft, DraftStatus, DraftVersion, PaperType

# Newest N snapshots retained per draft; older ones are pruned on every new snapshot.
MAX_DRAFT_VERSIONS = 50


async def create_draft(
    db: AsyncSession,
    project_id: UUID,
    user_id: UUID,
    title: str,
    paper_type: str = "literature_review",
) -> Draft:
    draft = Draft(
        project_id=project_id,
        user_id=user_id,
        title=title,
        paper_type=PaperType(paper_type),
        current_version=1,
    )
    db.add(draft)
    # Commit here, not just flush: `Depends(get_db)`'s own post-yield commit
    # (`app.database.get_db`) runs after the HTTP response has already been sent
    # (FastAPI 0.128's `AsyncExitStackMiddleware` sends the response before closing
    # the request-scoped `AsyncExitStack` the dependency is attached to), so a client
    # that immediately calls `POST /drafts/{id}/generate` on receiving this endpoint's
    # 201 can open a brand-new session before that deferred commit runs and get a 404
    # for a draft that, from the client's perspective, was already created (see
    # `tests/test_draft_create_commit.py`).
    # `app.services.task.create_job` already commits explicitly for the same reason.
    await db.commit()
    await db.refresh(draft)
    return draft


async def list_drafts(
    db: AsyncSession, project_id: UUID, page: int = 1, limit: int = 50
) -> tuple[list[Draft], int]:
    """List a project's drafts with pagination. Returns (rows, total)."""
    count_result = await db.execute(
        select(func.count()).select_from(Draft).where(Draft.project_id == project_id)
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        select(Draft)
        .where(Draft.project_id == project_id)
        .order_by(Draft.updated_at.desc(), Draft.id)
        .offset((page - 1) * limit)
        .limit(limit)
    )
    return list(result.scalars().all()), total


async def get_draft(db: AsyncSession, draft_id: UUID, user_id: UUID) -> Draft:
    """Load a draft's own columns. Version history is NOT eager-loaded — use
    ``list_draft_version_metadata`` when the caller needs it."""
    result = await db.execute(
        select(Draft).where(Draft.id == draft_id, Draft.user_id == user_id)
    )
    draft = result.scalar_one_or_none()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


async def list_draft_version_metadata(
    db: AsyncSession, draft_id: UUID
) -> list[dict[str, Any]]:
    """Return version metadata (never the snapshot content), newest version first."""
    result = await db.execute(
        select(
            DraftVersion.id,
            DraftVersion.version,
            DraftVersion.change_summary,
            DraftVersion.created_at,
        )
        .where(DraftVersion.draft_id == draft_id)
        .order_by(DraftVersion.version.desc())
    )
    return [
        {
            "id": row.id,
            "version": row.version,
            "change_summary": row.change_summary,
            "created_at": row.created_at,
        }
        for row in result.all()
    ]


async def prune_draft_versions(
    db: AsyncSession, draft_id: UUID, keep: int = MAX_DRAFT_VERSIONS
) -> int:
    """Delete all but the ``keep`` newest snapshots for a draft. Returns rows deleted."""
    stale = await db.execute(
        select(DraftVersion.id)
        .where(DraftVersion.draft_id == draft_id)
        .order_by(DraftVersion.version.desc())
        .offset(keep)
    )
    stale_ids = [row[0] for row in stale.all()]
    if not stale_ids:
        return 0
    await db.execute(
        delete(DraftVersion)
        .where(DraftVersion.id.in_(stale_ids))
        .execution_options(synchronize_session=False)
    )
    return len(stale_ids)


async def update_draft(
    db: AsyncSession,
    draft_id: UUID,
    user_id: UUID,
    title: str | None = None,
    content: dict | None = None,
    status: str | None = None,
) -> Draft:
    draft = await get_draft(db, draft_id, user_id)

    # If content changed, snapshot the old version and prune the tail of the history.
    if content is not None and content != draft.content:
        version = DraftVersion(
            draft_id=draft.id,
            version=draft.current_version,
            content=draft.content,
            change_summary=f"Version {draft.current_version} snapshot",
        )
        db.add(version)
        draft.content = content
        draft.current_version += 1
        await db.flush()
        await prune_draft_versions(db, draft.id)

    if title is not None:
        draft.title = title
    if status is not None:
        draft.status = DraftStatus(status)

    await db.flush()
    await db.refresh(draft)
    return draft


async def delete_draft(db: AsyncSession, draft_id: UUID, user_id: UUID) -> None:
    draft = await get_draft(db, draft_id, user_id)
    await db.delete(draft)
