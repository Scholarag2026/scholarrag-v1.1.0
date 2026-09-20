from fastapi import APIRouter

from app.api.admin import router as admin_router
from app.api.analysis import router as analysis_router
from app.api.analytics import router as analytics_router
from app.api.auth import router as auth_router
from app.api.datasets import router as datasets_router
from app.api.deep_search import router as deep_search_router
from app.api.drafts import router as drafts_router
from app.api.field_foundations import router as field_foundations_router
from app.api.fulltext import router as fulltext_router
from app.api.team import router as team_router
from app.api.graph import router as graph_router
from app.api.papers import router as papers_router
from app.api.projects import router as projects_router
from app.api.qualitative import router as qualitative_router
from app.api.quantitative import router as quantitative_router
from app.api.research_design import router as research_design_router
from app.api.search import router as search_router
from app.api.seed_papers import router as seed_papers_router
from app.api.tasks import router as tasks_router
from app.api.verification import router as verification_router
from app.api.scope import router as scope_router
from app.api.smart_search import router as smart_search_router
from app.api.deep_analysis import router as deep_analysis_router
from app.api.paper_chat import router as paper_chat_router
from app.api.journal_guidelines import router as journal_guidelines_router
from app.api.wos import router as wos_router

api_router = APIRouter()
api_router.include_router(admin_router)
api_router.include_router(auth_router)
api_router.include_router(projects_router)
api_router.include_router(papers_router)
api_router.include_router(search_router)
api_router.include_router(tasks_router)
api_router.include_router(drafts_router)
api_router.include_router(verification_router)
api_router.include_router(analysis_router)
api_router.include_router(graph_router)
api_router.include_router(deep_search_router)
api_router.include_router(seed_papers_router)
api_router.include_router(research_design_router)
api_router.include_router(datasets_router)
api_router.include_router(quantitative_router)
api_router.include_router(qualitative_router)
api_router.include_router(field_foundations_router)
api_router.include_router(fulltext_router)
api_router.include_router(team_router)
api_router.include_router(analytics_router)
api_router.include_router(scope_router)
api_router.include_router(smart_search_router)
api_router.include_router(wos_router)
api_router.include_router(deep_analysis_router)
api_router.include_router(paper_chat_router)
api_router.include_router(journal_guidelines_router)
