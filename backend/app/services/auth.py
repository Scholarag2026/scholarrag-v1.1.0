from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password_async, verify_password_async
from app.models.user import User


async def register_user(
    db: AsyncSession, email: str, password: str, name: str, expertise_level: str
) -> User:
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=email,
        password_hash=await hash_password_async(password),
        name=name,
        expertise_level=expertise_level,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not await verify_password_async(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user
