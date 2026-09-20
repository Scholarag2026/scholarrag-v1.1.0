"""GraphData.last_build lets the UI distinguish 'never built' from 'built, 0 links'."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.project import Project
from app.models.user import User
from app.services.graph import get_graph_data


async def _project(db) -> tuple[uuid.UUID, uuid.UUID]:
    user = User(
        email=f"lastbuild-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="not-a-real-hash",
        name="Last Build Tester",
    )
    db.add(user)
    await db.flush()
    project = Project(user_id=user.id, title="Last Build Project")
    db.add(project)
    await db.flush()
    return project.id, user.id


async def test_last_build_is_none_when_the_graph_was_never_built(db_session):
    project_id, _ = await _project(db_session)

    data = await get_graph_data(db_session, project_id)

    assert data.nodes == []
    assert data.edges == []
    assert data.last_build is None


async def test_last_build_reports_a_degraded_completed_build(db_session):
    project_id, user_id = await _project(db_session)
    db_session.add(
        AnalysisJob(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.graph_building,
            status=JobStatus.completed,
            progress=1.0,
            result={
                "edges_created": 0,
                "papers_processed": 0,
                "papers_failed": 4,
                "total_papers": 4,
            },
        )
    )
    await db_session.flush()

    data = await get_graph_data(db_session, project_id)

    assert data.last_build is not None
    assert data.last_build.status == "completed"
    assert data.last_build.edges_created == 0
    assert data.last_build.papers_failed == 4


async def test_last_build_reports_a_failed_build_with_its_error(db_session):
    project_id, user_id = await _project(db_session)
    db_session.add(
        AnalysisJob(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.graph_building,
            status=JobStatus.failed,
            progress=1.0,
            result={"edges_created": 0, "papers_processed": 0, "papers_failed": 3},
            error="OpenAlex returned errors for 3 of 3 papers",
        )
    )
    await db_session.flush()

    data = await get_graph_data(db_session, project_id)

    assert data.last_build.status == "failed"
    assert "OpenAlex" in data.last_build.error
