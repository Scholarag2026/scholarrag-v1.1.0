from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CreateTeamRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class AddMemberRequest(BaseModel):
    email: str
    role: str = "viewer"


class UpdateMemberRoleRequest(BaseModel):
    role: str


class ShareProjectRequest(BaseModel):
    project_id: UUID


class TeamMemberResponse(BaseModel):
    id: UUID
    user_id: UUID
    email: str
    name: str
    role: str
    joined_at: datetime

    model_config = {"from_attributes": True}


class TeamResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    created_by: UUID
    members: list[TeamMemberResponse]
    created_at: datetime

    model_config = {"from_attributes": True}


class TeamListResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    member_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
