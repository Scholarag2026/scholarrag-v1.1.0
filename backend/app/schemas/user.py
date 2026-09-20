from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UserResponse(BaseModel):
    id: UUID
    email: str
    name: str
    expertise_level: str
    preferred_language: str
    created_at: datetime

    model_config = {"from_attributes": True}
