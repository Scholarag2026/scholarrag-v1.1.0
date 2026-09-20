from __future__ import annotations

import uuid

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis_job import AnalysisJob, JobType
from app.models.draft import Draft
from app.models.paper import Paper
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.schemas.analytics import MonthCount, SystemAnalytics, UserAnalytics


async def get_user_analytics(db: AsyncSession, user_id: uuid.UUID) -> UserAnalytics:
    """Aggregate analytics for a single user."""

    # Total papers linked to this user's projects
    total_papers_q = (
        select(func.count(ProjectPaper.id))
        .join(Project, ProjectPaper.project_id == Project.id)
        .where(Project.user_id == user_id)
    )
    total_papers = (await db.execute(total_papers_q)).scalar_one() or 0

    # Total drafts
    total_drafts_q = select(func.count(Draft.id)).where(Draft.user_id == user_id)
    total_drafts = (await db.execute(total_drafts_q)).scalar_one() or 0

    # Total search jobs
    total_searches_q = (
        select(func.count(AnalysisJob.id))
        .where(AnalysisJob.user_id == user_id, AnalysisJob.job_type == JobType.search)
    )
    total_searches = (await db.execute(total_searches_q)).scalar_one() or 0

    # No dedicated export tracking yet; report 0
    total_exports = 0

    # Papers by month (using project_papers.added_at)
    month_expr = func.to_char(ProjectPaper.added_at, "YYYY-MM")
    month_q = (
        select(
            month_expr.label("month"),
            func.count(ProjectPaper.id).label("cnt"),
        )
        .join(Project, ProjectPaper.project_id == Project.id)
        .where(Project.user_id == user_id)
        .group_by(literal_column("month"))
        .order_by(literal_column("month"))
    )
    month_rows = (await db.execute(month_q)).all()
    papers_by_month = [MonthCount(month=r.month, count=r.cnt) for r in month_rows]

    # Papers by source
    source_q = (
        select(Paper.source_api, func.count(Paper.id).label("cnt"))
        .join(ProjectPaper, ProjectPaper.paper_id == Paper.id)
        .join(Project, ProjectPaper.project_id == Project.id)
        .where(Project.user_id == user_id)
        .group_by(Paper.source_api)
    )
    source_rows = (await db.execute(source_q)).all()
    papers_by_source = {str(r.source_api.value): r.cnt for r in source_rows}

    return UserAnalytics(
        total_papers=total_papers,
        total_drafts=total_drafts,
        total_searches=total_searches,
        total_exports=total_exports,
        papers_by_month=papers_by_month,
        papers_by_source=papers_by_source,
    )


async def get_system_analytics(db: AsyncSession) -> SystemAnalytics:
    """Aggregate analytics across the whole system."""

    total_users = (await db.execute(select(func.count(User.id)))).scalar_one() or 0
    total_projects = (await db.execute(select(func.count(Project.id)))).scalar_one() or 0
    total_papers = (await db.execute(select(func.count(Paper.id)))).scalar_one() or 0
    total_drafts = (await db.execute(select(func.count(Draft.id)))).scalar_one() or 0
    total_jobs = (await db.execute(select(func.count(AnalysisJob.id)))).scalar_one() or 0

    # Users by month
    user_month_expr = func.to_char(User.created_at, "YYYY-MM")
    users_month_q = (
        select(
            user_month_expr.label("month"),
            func.count(User.id).label("cnt"),
        )
        .group_by(literal_column("month"))
        .order_by(literal_column("month"))
    )
    users_month_rows = (await db.execute(users_month_q)).all()
    users_by_month = [MonthCount(month=r.month, count=r.cnt) for r in users_month_rows]

    # Papers by source
    source_q = (
        select(Paper.source_api, func.count(Paper.id).label("cnt"))
        .group_by(Paper.source_api)
    )
    source_rows = (await db.execute(source_q)).all()
    papers_by_source = {str(r.source_api.value): r.cnt for r in source_rows}

    # Jobs by type
    jobs_q = (
        select(AnalysisJob.job_type, func.count(AnalysisJob.id).label("cnt"))
        .group_by(AnalysisJob.job_type)
    )
    jobs_rows = (await db.execute(jobs_q)).all()
    jobs_by_type = {str(r.job_type.value): r.cnt for r in jobs_rows}

    return SystemAnalytics(
        total_users=total_users,
        total_projects=total_projects,
        total_papers=total_papers,
        total_drafts=total_drafts,
        total_jobs=total_jobs,
        users_by_month=users_by_month,
        papers_by_source=papers_by_source,
        jobs_by_type=jobs_by_type,
    )
