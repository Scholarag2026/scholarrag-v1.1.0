"""expand_node on OpenAlex (issue EXPAND-NODE-504).

Refs and cites are fetched concurrently through the mocked client, so the whole call is
two round trips deep and comfortably inside the API's 15s ceiling.
"""

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.models.citation_edge import CitationEdge
from app.models.paper import Paper, SourceApi
from app.models.project import Project
from app.models.project_paper import ProjectPaper
from app.models.user import User
from app.services.graph import expand_node
from tests.conftest import TEST_DATABASE_URL


@pytest.fixture
async def session_factory():
    engine = create_async_engine(TEST_DATABASE_URL)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


def _work(w_id: str, title: str, *, refs=None, citations: int = 0) -> dict:
    return {
        "doi": None,
        "title": title,
        "authors": [{"name": "Author One"}],
        "year": 2019,
        "journal_name": None,
        "journal_issn": None,
        "citation_count": citations,
        "abstract": None,
        "source_api": "openalex",
        "external_id": w_id,
        "full_text_url": None,
        "wos_collection": None,
        "wos_categories": None,
        "openalex_id": w_id,
        "referenced_works": list(refs or []),
    }


def _fake_client(*, work=None, batch=None, cites=None, error=None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.get_work = AsyncMock(side_effect=error)
    else:
        client.get_work = AsyncMock(return_value=work)
    client.get_works_batch = AsyncMock(return_value=list(batch or []))
    client.get_citations = AsyncMock(return_value=list(cites or []))
    return client


async def _seed(db, *, external_id: str | None = "W1"):
    user = User(
        email=f"expand-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="not-a-real-hash",
        name="Expand Tester",
    )
    db.add(user)
    await db.flush()
    project = Project(user_id=user.id, title="Expand Project")
    db.add(project)
    await db.flush()
    paper = Paper(
        title="Focus paper", authors=[], external_id=external_id, source_api=SourceApi.openalex
    )
    db.add(paper)
    await db.flush()
    db.add(ProjectPaper(project_id=project.id, paper_id=paper.id))
    await db.commit()
    return project.id, paper.id


async def test_expand_returns_new_nodes_and_persists_edges(db_session, session_factory):
    project_id, paper_id = await _seed(db_session)
    fake = _fake_client(
        work=_work("W1", "Focus paper", refs=["W2"]),
        batch=[_work("W2", "Reference", citations=9)],
        cites=[_work("W3", "Citing", citations=4)],
    )

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        expansion = await expand_node(
            paper_id=paper_id, project_id=project_id, session_factory=session_factory
        )

    assert {n.title for n in expansion.new_nodes} == {"Reference", "Citing"}
    assert len(expansion.new_edges) == 2

    async with session_factory() as db:
        edges = (
            await db.execute(
                select(CitationEdge).where(CitationEdge.project_id == project_id)
            )
        ).scalars().all()
    assert len(edges) == 2


async def test_expand_caps_refs_and_cites_by_citation_count(db_session, session_factory):
    project_id, paper_id = await _seed(db_session)
    many_refs = [_work(f"W{i}", f"Ref {i}", citations=i) for i in range(100, 140)]
    many_cites = [_work(f"C{i}", f"Cite {i}", citations=i) for i in range(200, 240)]
    fake = _fake_client(
        work=_work("W1", "Focus paper", refs=[w["openalex_id"] for w in many_refs]),
        batch=many_refs,
        cites=many_cites,
    )

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        expansion = await expand_node(
            paper_id=paper_id, project_id=project_id, session_factory=session_factory
        )

    # graph_expand_max_refs=15 + graph_expand_max_cites=15
    assert len(expansion.new_nodes) == 30
    assert "Ref 139" in {n.title for n in expansion.new_nodes}
    assert "Ref 100" not in {n.title for n in expansion.new_nodes}


async def test_expand_returns_empty_for_a_paper_without_an_openalex_id(
    db_session, session_factory
):
    project_id, paper_id = await _seed(db_session, external_id=None)
    fake = _fake_client(work=_work("W1", "unused"))

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        expansion = await expand_node(
            paper_id=paper_id, project_id=project_id, session_factory=session_factory
        )

    assert expansion.new_nodes == []
    assert expansion.new_edges == []
    assert fake.get_work.await_count == 0


async def test_expand_returns_empty_when_openalex_has_no_record(db_session, session_factory):
    project_id, paper_id = await _seed(db_session)
    fake = _fake_client(work=None)

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        expansion = await expand_node(
            paper_id=paper_id, project_id=project_id, session_factory=session_factory
        )

    assert expansion.new_nodes == []


async def test_expand_propagates_upstream_errors(db_session, session_factory):
    """The API layer turns this into a 502 with a message the UI can show."""
    project_id, paper_id = await _seed(db_session)
    response = MagicMock()
    response.status_code = 503
    fake = _fake_client(
        error=httpx.HTTPStatusError("503", request=MagicMock(), response=response)
    )

    with patch("app.services.graph.OpenAlexClient", return_value=fake):
        with pytest.raises(httpx.HTTPStatusError):
            await expand_node(
                paper_id=paper_id, project_id=project_id, session_factory=session_factory
            )


async def test_expand_treats_throttled_cites_as_empty(db_session, session_factory):
    """fetched[1] is None when cites: is rate-limited — expand must not crash."""
    project_id, paper_id = await _seed(db_session)

    fake = _fake_client(work=_work("W1", "Focus paper", refs=["W2"]))
    with patch("app.services.graph.OpenAlexClient", return_value=fake), patch(
        "app.services.graph.fetch_relations",
        new=AsyncMock(return_value=([_work("W2", "Ref")], None)),
    ):
        expansion = await expand_node(
            paper_id=paper_id, project_id=project_id, session_factory=session_factory
        )

    assert [n.title for n in expansion.new_nodes] == ["Ref"]
