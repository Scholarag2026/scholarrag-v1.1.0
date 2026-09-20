import pytest

from app.models.user import UserRole


async def _register_and_get_token(client, email="user@example.com", name="Test User"):
    """Helper: register a user and return (token, user_id)."""
    resp = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": name,
            "expertise_level": "student",
        },
    )
    data = resp.json()
    return data["access_token"], data["user"]["id"]


async def _promote_to_admin(db_session, user_id):
    """Helper: set a user's role to admin directly in the DB."""
    from uuid import UUID

    from sqlalchemy import update

    from app.models.user import User

    await db_session.execute(
        update(User).where(User.id == UUID(user_id)).values(role=UserRole.admin)
    )
    await db_session.commit()


@pytest.mark.asyncio
async def test_non_admin_gets_403(client):
    token, _ = await _register_and_get_token(client, email="nonadmin@example.com")
    resp = await client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_list_users(client, db_session):
    token, uid = await _register_and_get_token(client, email="admin@example.com", name="Admin")
    await _promote_to_admin(db_session, uid)

    resp = await client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    users = resp.json()
    assert isinstance(users, list)
    assert len(users) >= 1
    assert users[0]["email"] == "admin@example.com"


@pytest.mark.asyncio
async def test_admin_stats(client, db_session):
    token, uid = await _register_and_get_token(client, email="stats@example.com", name="Stats")
    await _promote_to_admin(db_session, uid)

    resp = await client.get(
        "/api/v1/admin/stats",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "total_users" in data
    assert "total_projects" in data
    assert "total_papers" in data
    assert "total_drafts" in data
    assert "total_jobs" in data
    assert data["total_users"] >= 1


@pytest.mark.asyncio
async def test_admin_update_user_role(client, db_session):
    # Register two users: one will be admin, one will be target
    admin_token, admin_id = await _register_and_get_token(
        client, email="roleadmin@example.com", name="RoleAdmin"
    )
    await _promote_to_admin(db_session, admin_id)

    _, target_id = await _register_and_get_token(
        client, email="target@example.com", name="Target"
    )

    resp = await client.patch(
        f"/api/v1/admin/users/{target_id}/role",
        json={"role": "admin"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


async def test_admin_wos_import_csv(client, db_session):
    """POST /admin/wos-journals/import passes the detected collection through (#T7 latent bug)."""
    admin_token, admin_id = await _register_and_get_token(
        client, email="wosadmin@example.com", name="WoS Admin"
    )
    await _promote_to_admin(db_session, admin_id)

    csv_text = (
        "Journal title,ISSN,eISSN,Publisher name,Publisher address,Languages,"
        "Web of Science Categories\n"
        "Journal of Testing,1234-5678,8765-4321,TestPub,Addr,English,Testing\n"
    )
    resp = await client.post(
        "/api/v1/admin/wos-journals/import",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("Social Sciences Citation Index (SSCI).csv", csv_text, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["imported"] == 1
    assert body["collection"] == "SSCI"
