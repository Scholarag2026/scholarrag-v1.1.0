from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AdminUserResponse(BaseModel):
    id: UUID
    email: str
    name: str
    role: str
    expertise_level: str
    created_at: datetime
    project_count: int

    model_config = {"from_attributes": True}


class SystemStats(BaseModel):
    total_users: int
    total_projects: int
    total_papers: int
    total_drafts: int
    total_jobs: int


class UpdateUserRoleRequest(BaseModel):
    role: str  # "user" or "admin"
