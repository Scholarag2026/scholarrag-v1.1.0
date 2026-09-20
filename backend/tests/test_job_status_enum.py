"""The cancelled job status and its Alembic revision."""

import re
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.models.analysis_job import AnalysisJob, JobStatus, JobType

VERSIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def test_job_status_has_cancelled_member():
    assert JobStatus.cancelled.value == "cancelled"


def test_an_alembic_revision_adds_cancelled_to_the_db_enum():
    sources = [p.read_text(encoding="utf-8") for p in VERSIONS_DIR.glob("*.py")]
    pattern = re.compile(r"ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'cancelled'")
    assert any(pattern.search(s) for s in sources), (
        "no alembic revision adds 'cancelled' to the jobstatus enum"
    )


def test_the_cancelled_revision_chains_from_the_previous_head():
    src = (VERSIONS_DIR / "h1a2b3c4d5e6_add_cancelled_job_status.py").read_text(encoding="utf-8")
    assert 'revision: str = "h1a2b3c4d5e6"' in src
    assert 'down_revision: Union[str, None] = "g1a2b3c4d5e6"' in src


async def _seed_project_and_user(client) -> tuple[UUID, UUID]:
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "cancel-enum@example.com",
            "password": "StrongPass123!",
            "name": "Enum User",
            "expertise_level": "researcher",
        },
    )
    token = res.json()["access_token"]
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    user_id = UUID(me.json()["id"])
    proj = await client.post(
        "/api/v1/projects",
        json={"title": "Enum Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return UUID(proj.json()["id"]), user_id


@pytest.mark.asyncio
async def test_cancelled_status_round_trips_through_postgres(client, db_session):
    project_id, user_id = await _seed_project_and_user(client)
    job_id = uuid4()
    db_session.add(
        AnalysisJob(
            id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.smart_search,
            status=JobStatus.cancelled,
            progress=0.4,
        )
    )
    await db_session.commit()

    result = await db_session.execute(
        select(AnalysisJob.status).where(AnalysisJob.id == job_id)
    )
    assert result.scalar_one() is JobStatus.cancelled
