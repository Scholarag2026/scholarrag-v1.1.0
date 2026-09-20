from pydantic import BaseModel


class MonthCount(BaseModel):
    month: str  # "2026-01"
    count: int


class UserAnalytics(BaseModel):
    total_papers: int
    total_drafts: int
    total_searches: int
    total_exports: int
    papers_by_month: list[MonthCount]
    papers_by_source: dict[str, int]


class SystemAnalytics(BaseModel):
    total_users: int
    total_projects: int
    total_papers: int
    total_drafts: int
    total_jobs: int
    users_by_month: list[MonthCount]
    papers_by_source: dict[str, int]
    jobs_by_type: dict[str, int]
