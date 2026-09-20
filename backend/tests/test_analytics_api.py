import pytest


async def _register_user(client, email="analytics@example.com", name="Analytics User"):
    res = await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "StrongPass123!",
        "name": name,
        "expertise_level": "researcher",
    })
    return res.json()["access_token"]


@pytest.mark.asyncio
async def test_analytics_me_returns_schema(client):
    """GET /analytics/me should return valid UserAnalytics JSON."""
    token = await _register_user(client)
    res = await client.get(
        "/api/v1/analytics/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "total_papers" in data
    assert "total_drafts" in data
    assert "total_searches" in data
    assert "total_exports" in data
    assert isinstance(data["papers_by_month"], list)
    assert isinstance(data["papers_by_source"], dict)


@pytest.mark.asyncio
async def test_analytics_me_requires_auth(client):
    """GET /analytics/me without a token should return 401/403."""
    res = await client.get("/api/v1/analytics/me")
    assert res.status_code in (401, 403)


@pytest.mark.asyncio
async def test_analytics_system_requires_admin(client):
    """GET /analytics/system should reject non-admin users."""
    token = await _register_user(client, email="nonadmin@example.com")
    res = await client.get(
        "/api/v1/analytics/system",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code == 403
