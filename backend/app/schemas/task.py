# backend/app/schemas/task.py
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class TaskResponse(BaseModel):
    id: UUID
    job_type: str
    status: str
    progress: float
    progress_message: str | None
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskCreateResponse(BaseModel):
    task_id: UUID
