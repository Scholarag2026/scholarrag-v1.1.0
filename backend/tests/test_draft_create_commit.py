"""Regression test for the `create_draft` -> `generate` 404 race.

A client can see `POST /drafts/{id}/generate` return 404 seconds after the same
draft's own `POST .../drafts` returned 201. This is not host flakiness: FastAPI
0.128's `AsyncExitStackMiddleware` (`fastapi.routing.request_response`) sends the HTTP
response (`await response(scope, receive, send)`) BEFORE it exits the request-scoped
`AsyncExitStack` that `Depends(get_db)` is attached to, and `app.database.get_db`'s own
commit (`await session.commit()`) is the code that runs when that exit stack closes --
i.e., after the response bytes have already left the process. A client (a demo script,
a browser, another service) that fires the very next request as soon as it receives the
201 can therefore reach the backend, open a brand-new session for that request, and
query for a draft whose creating transaction has not committed yet.

`app.services.task.create_job` already commits explicitly, for exactly this reason
(a job row a client polls, or a background task reads through its OWN session, right
after creation). `app.services.draft.create_draft` must do the same, which is the gap
this test closes.

This suite's own `client` fixture cannot reproduce the race: `conftest.py`'s
`override_get_db` hands every request in a test the SAME session object, so a second
request's `get_draft` call always sees the first request's flushed-but-uncommitted
insert (read-your-own-writes on one shared transaction), regardless of whether
`create_draft` itself commits. This test instead opens two independent engines/
sessions against the test database, matching two unrelated HTTP requests in
production, and requires the draft to be visible on the second session without either
session ever reading through the other.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import hash_password
from app.models.project import Project
from app.models.user import User
from app.services import draft as draft_service
from tests.conftest import TEST_DATABASE_URL


@pytest.mark.asyncio
async def test_create_draft_is_visible_to_a_brand_new_session_with_no_further_commit():
    engine_a = create_async_engine(TEST_DATABASE_URL)
    engine_b = create_async_engine(TEST_DATABASE_URL)
    factory_a = async_sessionmaker(bind=engine_a, class_=AsyncSession, expire_on_commit=False)
    factory_b = async_sessionmaker(bind=engine_b, class_=AsyncSession, expire_on_commit=False)
    try:
        async with factory_a() as setup_session:
            user = User(
                email=f"draft-commit-{uuid.uuid4().hex[:8]}@example.com",
                password_hash=hash_password("StrongPass123!"),
                name="Commit Race User",
            )
            setup_session.add(user)
            await setup_session.flush()
            project = Project(user_id=user.id, title="Commit Race Project")
            setup_session.add(project)
            await setup_session.commit()
            user_id, project_id = user.id, project.id

        # Simulates POST /projects/{id}/drafts: a fresh, request-scoped session that
        # never commits itself -- in production, `get_db`'s post-yield commit does
        # that, after the response has already been sent (see module docstring).
        async with factory_a() as write_session:
            draft = await draft_service.create_draft(
                write_session, project_id, user_id, "Commit Race Draft",
            )
            draft_id = draft.id

        # Simulates POST /drafts/{id}/generate: a second, independent request-scoped
        # session, opened only after the first has been closed with no explicit
        # commit of its own -- exactly what a client that already received the first
        # request's response can trigger.
        async with factory_b() as read_session:
            found = await draft_service.get_draft(read_session, draft_id, user_id)
            assert found.id == draft_id
    finally:
        await engine_a.dispose()
        await engine_b.dispose()
