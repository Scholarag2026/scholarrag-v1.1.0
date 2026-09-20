from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.team import Team, TeamMember, TeamProject, TeamRole
from app.models.user import User


async def create_team(
    db: AsyncSession,
    name: str,
    description: str | None,
    user_id: UUID,
) -> Team:
    team = Team(name=name, description=description, created_by=user_id)
    db.add(team)
    await db.flush()

    # Add creator as owner
    member = TeamMember(team_id=team.id, user_id=user_id, role=TeamRole.owner)
    db.add(member)
    await db.flush()
    await db.refresh(team, attribute_names=["members"])
    return team


async def get_user_teams(db: AsyncSession, user_id: UUID) -> list[dict]:
    """List teams the user belongs to, with member counts."""
    stmt = (
        select(
            Team.id,
            Team.name,
            Team.description,
            Team.created_at,
            func.count(TeamMember.id).label("member_count"),
        )
        .join(TeamMember, TeamMember.team_id == Team.id)
        .where(
            Team.id.in_(
                select(TeamMember.team_id).where(TeamMember.user_id == user_id)
            )
        )
        .group_by(Team.id, Team.name, Team.description, Team.created_at)
        .order_by(Team.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()
    return [
        {
            "id": row.id,
            "name": row.name,
            "description": row.description,
            "member_count": row.member_count,
            "created_at": row.created_at,
        }
        for row in rows
    ]


async def get_team(db: AsyncSession, team_id: UUID, user_id: UUID) -> Team:
    """Get team details; caller must be a member."""
    # Verify membership
    membership = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user_id
        )
    )
    if not membership.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Team not found")

    result = await db.execute(
        select(Team)
        .where(Team.id == team_id)
        .options(selectinload(Team.members))
    )
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


async def _get_member_role(db: AsyncSession, team_id: UUID, user_id: UUID) -> TeamRole | None:
    result = await db.execute(
        select(TeamMember.role).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user_id
        )
    )
    return result.scalar_one_or_none()


async def add_member(
    db: AsyncSession,
    team_id: UUID,
    email: str,
    role: str,
    requesting_user_id: UUID,
) -> TeamMember:
    # Check requester is owner or editor
    requester_role = await _get_member_role(db, team_id, requesting_user_id)
    if requester_role not in (TeamRole.owner, TeamRole.editor):
        raise HTTPException(status_code=403, detail="Only owners and editors can add members")

    # Look up user by email
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check not already a member
    existing = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user.id
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="User is already a team member")

    team_role = TeamRole(role)
    member = TeamMember(team_id=team_id, user_id=user.id, role=team_role)
    db.add(member)
    await db.flush()
    await db.refresh(member)
    return member


async def remove_member(
    db: AsyncSession,
    team_id: UUID,
    user_id: UUID,
    requesting_user_id: UUID,
) -> None:
    requester_role = await _get_member_role(db, team_id, requesting_user_id)
    if requester_role != TeamRole.owner:
        raise HTTPException(status_code=403, detail="Only the team owner can remove members")

    if user_id == requesting_user_id:
        raise HTTPException(status_code=400, detail="Owner cannot remove themselves")

    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user_id
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    await db.delete(member)


async def update_member_role(
    db: AsyncSession,
    team_id: UUID,
    user_id: UUID,
    role: str,
    requesting_user_id: UUID,
) -> TeamMember:
    requester_role = await _get_member_role(db, team_id, requesting_user_id)
    if requester_role != TeamRole.owner:
        raise HTTPException(status_code=403, detail="Only the team owner can update roles")

    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id, TeamMember.user_id == user_id
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    member.role = TeamRole(role)
    await db.flush()
    await db.refresh(member)
    return member


async def share_project(
    db: AsyncSession,
    team_id: UUID,
    project_id: UUID,
    user_id: UUID,
) -> TeamProject:
    # Verify user is a member
    requester_role = await _get_member_role(db, team_id, user_id)
    if requester_role is None:
        raise HTTPException(status_code=403, detail="Not a team member")

    # Check not already shared
    existing = await db.execute(
        select(TeamProject).where(
            TeamProject.team_id == team_id, TeamProject.project_id == project_id
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Project already shared with this team")

    tp = TeamProject(team_id=team_id, project_id=project_id)
    db.add(tp)
    await db.flush()
    await db.refresh(tp)
    return tp


async def delete_team(
    db: AsyncSession,
    team_id: UUID,
    user_id: UUID,
) -> None:
    requester_role = await _get_member_role(db, team_id, user_id)
    if requester_role != TeamRole.owner:
        raise HTTPException(status_code=403, detail="Only the team owner can delete the team")

    result = await db.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    await db.delete(team)
