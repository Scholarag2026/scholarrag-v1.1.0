from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_admin_user, get_current_user
from app.models.user import User
from app.schemas.analytics import SystemAnalytics, UserAnalytics
from app.services import analytics as analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/me", response_model=UserAnalytics)
async def my_analytics(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return analytics for the authenticated user."""
    return await analytics_service.get_user_analytics(db, user.id)


@router.get("/system", response_model=SystemAnalytics)
async def system_analytics(
    user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Return system-wide analytics (admin only)."""
    return await analytics_service.get_system_analytics(db)
