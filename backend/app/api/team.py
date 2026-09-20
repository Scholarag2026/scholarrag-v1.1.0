from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.schemas.team import (
    AddMemberRequest,
    CreateTeamRequest,
    ShareProjectRequest,
    TeamListResponse,
    TeamMemberResponse,
    TeamResponse,
    UpdateMemberRoleRequest,
)
from app.services import team as team_service

router = APIRouter(prefix="/teams", tags=["teams"])


def _member_to_response(member, user_email: str, user_name: str) -> dict:
    return {
        "id": member.id,
        "user_id": member.user_id,
        "email": user_email,
        "name": user_name,
        "role": member.role.value if hasattr(member.role, "value") else member.role,
        "joined_at": member.created_at,
    }


async def _build_team_response(db: AsyncSession, team) -> TeamResponse:
    """Build a TeamResponse with member email/name populated (single batched user query)."""
    member_ids = [m.user_id for m in team.members]
    users_by_id: dict = {}
    if member_ids:
        rows = await db.execute(select(User).where(User.id.in_(member_ids)))
        users_by_id = {u.id: u for u in rows.scalars().all()}

    members_data = []
    for m in team.members:
        u = users_by_id.get(m.user_id)
        if u is not None:
            members_data.append(_member_to_response(m, u.email, u.name))

    return TeamResponse(
        id=team.id,
        name=team.name,
        description=team.description,
        created_by=team.created_by,
        members=[TeamMemberResponse(**md) for md in members_data],
        created_at=team.created_at,
    )


@router.post("", response_model=TeamResponse, status_code=201)
async def create_team(
    req: CreateTeamRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    team = await team_service.create_team(db, req.name, req.description, user.id)
    return await _build_team_response(db, team)


@router.get("", response_model=list[TeamListResponse])
async def list_teams(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    teams = await team_service.get_user_teams(db, user.id)
    return [TeamListResponse(**t) for t in teams]


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    team = await team_service.get_team(db, team_id, user.id)
    return await _build_team_response(db, team)


@router.post("/{team_id}/members", response_model=TeamMemberResponse, status_code=201)
async def add_member(
    team_id: UUID,
    req: AddMemberRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    member = await team_service.add_member(db, team_id, req.email, req.role, user.id)
    # Look up the added user for email/name
    result = await db.execute(select(User).where(User.id == member.user_id))
    added_user = result.scalar_one()
    return TeamMemberResponse(**_member_to_response(member, added_user.email, added_user.name))


@router.patch("/{team_id}/members/{user_id}", response_model=TeamMemberResponse)
async def update_member_role(
    team_id: UUID,
    user_id: UUID,
    req: UpdateMemberRoleRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    member = await team_service.update_member_role(db, team_id, user_id, req.role, user.id)
    result = await db.execute(select(User).where(User.id == member.user_id))
    target_user = result.scalar_one()
    return TeamMemberResponse(**_member_to_response(member, target_user.email, target_user.name))


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: UUID,
    user_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await team_service.remove_member(db, team_id, user_id, user.id)
    return {"success": True}


@router.post("/{team_id}/projects", status_code=201)
async def share_project(
    team_id: UUID,
    req: ShareProjectRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tp = await team_service.share_project(db, team_id, req.project_id, user.id)
    return {"success": True, "team_project_id": str(tp.id)}


@router.delete("/{team_id}")
async def delete_team(
    team_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await team_service.delete_team(db, team_id, user.id)
    return {"success": True}
