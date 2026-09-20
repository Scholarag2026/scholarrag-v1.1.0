"""Shared seeding helpers for T5 (analysis & writing) service tests.

Background-task services take a ``session_factory`` and open their own sessions, so tests
need a factory that is independent of the request-scoped session provided by the autouse
``db_session`` fixture. Both point at the same test database.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.analysis_job import AnalysisJob, JobStatus, JobType
from app.models.draft import Draft
from app.models.paper import Paper, SourceApi
from app.models.project import CitationStyle, Project
from app.models.project_paper import ProjectPaper
from app.models.user import User
from tests.conftest import TEST_DATABASE_URL


def make_session_factory():
    """Return ``(session_factory, engine)`` bound to the test database.

    Always ``await engine.dispose()`` at the end of the test.
    """
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return factory, engine


async def seed_user(db: AsyncSession) -> User:
    user = User(
        email=f"t5-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="not-a-real-hash",
        name="T5 User",
    )
    db.add(user)
    await db.commit()
    return user


async def seed_project(db: AsyncSession, user: User, **kwargs) -> Project:
    project = Project(
        user_id=user.id,
        title=kwargs.pop("title", "T5 Project"),
        description=kwargs.pop("description", "A project about AI tutoring."),
        target_journal=kwargs.pop("target_journal", "Computers & Education"),
        citation_style=kwargs.pop("citation_style", CitationStyle.apa),
        **kwargs,
    )
    db.add(project)
    await db.commit()
    return project


async def seed_paper(db: AsyncSession, project: Project, **kwargs) -> Paper:
    paper = Paper(
        title=kwargs.pop("title", "A Paper"),
        authors=kwargs.pop("authors", [{"name": "Jane Smith"}]),
        year=kwargs.pop("year", 2020),
        source_api=kwargs.pop("source_api", SourceApi.manual),
        **kwargs,
    )
    db.add(paper)
    await db.flush()
    db.add(ProjectPaper(project_id=project.id, paper_id=paper.id))
    await db.commit()
    return paper


async def seed_draft(db: AsyncSession, project: Project, user: User, content: dict) -> Draft:
    draft = Draft(project_id=project.id, user_id=user.id, title="T5 Draft", content=content)
    db.add(draft)
    await db.commit()
    return draft


async def seed_job(db: AsyncSession, project: Project, job_type: JobType) -> AnalysisJob:
    job = AnalysisJob(
        project_id=project.id,
        user_id=project.user_id,
        job_type=job_type,
        status=JobStatus.pending,
        progress=0.0,
    )
    db.add(job)
    await db.commit()
    return job


def tiptap_paragraph(text: str) -> dict:
    """Minimal Tiptap document containing a single paragraph."""
    return {
        "type": "doc",
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }


class FakeResult:
    """Stand-in for a pydantic-ai ``AgentRunResult``.

    ``all_messages()`` (task authorisation 2026-09-11, the repair turn) is a stand-in for
    pydantic-ai's own method of the same name: it never models pydantic-ai's real message
    types, it just echoes back whatever ``message_history`` this call was given, plus this
    call's own prompt/output as one further turn -- enough for a test to check that a
    later call's ``message_history`` really does carry an earlier call's own prompt and
    reply.
    """

    def __init__(self, output, prompt: object | None = None, message_history: object = None):
        self.output = output
        self._prompt = prompt
        self._message_history = list(message_history or [])

    def all_messages(self):
        return [*self._message_history, ("user", self._prompt), ("model", self.output)]


class FakeAgent:
    """Stand-in for a pydantic-ai ``Agent`` that records every call.

    ``output`` may be a plain value (returned for every call) or a callable taking the
    prompt and returning the output. Set ``raises`` to make every call raise instead.

    ``calls`` stays a list of ``(prompt, deps)`` 2-tuples, exactly as every existing
    caller already unpacks it. ``message_histories``/``results`` (task authorisation
    2026-09-11) are additive, parallel lists: whatever ``message_history`` each call was
    given (``None`` for a call that gave none) and each call's own `FakeResult`, so a
    repair-turn test can check that call 2's ``message_history`` is call 1's own
    ``all_messages()``.
    """

    def __init__(self, output=None, raises: Exception | None = None):
        self._output = output
        self._raises = raises
        self.calls: list[tuple[str, object]] = []
        self.message_histories: list[object] = []
        self.results: list[FakeResult] = []

    async def run(self, prompt, deps=None, message_history=None):
        self.calls.append((prompt, deps))
        self.message_histories.append(message_history)
        if self._raises is not None:
            raise self._raises
        output = self._output(prompt) if callable(self._output) else self._output
        result = FakeResult(output, prompt=prompt, message_history=message_history)
        self.results.append(result)
        return result
