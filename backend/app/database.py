from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.core.json_utils import json_serializer

# One engine is shared by request handlers (get_db) and every BackgroundTasks job.
# 2 uvicorn workers x (pool_size + max_overflow) = the real ceiling against Postgres.
# pool_timeout is deliberately short so exhaustion surfaces as an error instead of a
# 30-second stall on unrelated requests (#44).
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=settings.db_pool_recycle,
    pool_pre_ping=True,
    echo=False,
    json_serializer=json_serializer,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db():
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
