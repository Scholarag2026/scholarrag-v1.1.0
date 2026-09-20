# backend/tests/test_search_api.py
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.schemas.paper import PaperData
from tests.conftest import TEST_DATABASE_URL


async def get_auth_token(client):
    res = await client.post("/api/v1/auth/register", json={
        "email": "search@example.com", "password": "StrongPass123!",
        "name": "Search User", "expertise_level": "researcher",
    })
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post("/api/v1/projects", json={"title": "Search Project"},
                            headers={"Authorization": f"Bearer {token}"})
    return res.json()["id"]


MOCK_RESULTS = (
    [
        PaperData(
            doi="10.1/test",
            title="Test Paper",
            authors=[{"name": "Author"}],
            year=2023,
            citation_count=50,
            source_api="openalex",
        )
    ],
    {"openalex": 1},
)


@pytest.fixture(autouse=True)
async def _patch_search_session_factory():
    """Override the background task's session factory to use the test database."""
    import app.api.search as search_module
    engine = create_async_engine(TEST_DATABASE_URL)
    test_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    original = search_module._session_factory
    search_module._session_factory = test_factory
    yield
    search_module._session_factory = original
    await engine.dispose()


@pytest.mark.asyncio
async def test_search_creates_task(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch("app.api.search.SearchService") as mock_svc:
        instance = mock_svc.return_value
        instance.search = AsyncMock(return_value=MOCK_RESULTS)

        response = await client.post(
            f"/api/v1/projects/{project_id}/search",
            json={"query": "machine learning"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_get_task_status(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch("app.api.search.SearchService") as mock_svc:
        instance = mock_svc.return_value
        instance.search = AsyncMock(return_value=MOCK_RESULTS)

        search_res = await client.post(
            f"/api/v1/projects/{project_id}/search",
            json={"query": "machine learning"},
            headers={"Authorization": f"Bearer {token}"},
        )
        task_id = search_res.json()["task_id"]

    # Wait briefly for background task to complete
    await asyncio.sleep(0.5)

    response = await client.get(
        f"/api/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("pending", "running", "completed")
    assert data["job_type"] == "search"


@pytest.mark.asyncio
async def test_search_requires_query(client):
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    response = await client.post(
        f"/api/v1/projects/{project_id}/search",
        json={"query": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_task_not_found(client):
    token = await get_auth_token(client)

    response = await client.get(
        "/api/v1/tasks/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_no_db_session_is_held_across_the_network_wait():
    """DB-SESSION-AND-POOL-PRESSURE: the pooled session must not span svc.search()."""
    from uuid import uuid4

    import app.api.search as search_module
    from app.services.search import SearchService

    state = {"open": 0, "peak_during_search": 0, "acquisitions": 0}
    real_factory = search_module._session_factory

    class _TrackingCtx:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            state["open"] += 1
            state["acquisitions"] += 1
            return await self._inner.__aenter__()

        async def __aexit__(self, *exc_info):
            state["open"] -= 1
            return await self._inner.__aexit__(*exc_info)

    def _tracking_factory():
        return _TrackingCtx(real_factory())

    async def _fake_search(self, query, **kwargs):
        state["peak_during_search"] = max(state["peak_during_search"], state["open"])
        return MOCK_RESULTS

    search_module._session_factory = _tracking_factory
    try:
        with patch.object(SearchService, "search", new=_fake_search):
            await search_module._run_search_background(uuid4(), "q", None, None, None)
    finally:
        search_module._session_factory = real_factory

    assert state["peak_during_search"] == 0
    assert state["open"] == 0
    assert state["acquisitions"] >= 2
