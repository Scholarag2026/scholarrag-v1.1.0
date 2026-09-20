import pytest


async def register_user(client, email, name="Test User"):
    """Helper to register a user and get their auth token + user id."""
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": name,
            "expertise_level": "researcher",
        },
    )
    data = res.json()
    return data["access_token"], data.get("user", {}).get("id")


async def get_auth_header(client, email, name="Test User"):
    token, user_id = await register_user(client, email, name)
    return {"Authorization": f"Bearer {token}"}, user_id


@pytest.mark.asyncio
async def test_create_team(client):
    headers, _ = await get_auth_header(client, "team_owner@example.com", "Owner")
    response = await client.post(
        "/api/v1/teams",
        json={"name": "My Team", "description": "A test team"},
        headers=headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Team"
    assert data["description"] == "A test team"
    assert len(data["members"]) == 1
    assert data["members"][0]["role"] == "owner"
    assert data["members"][0]["email"] == "team_owner@example.com"


@pytest.mark.asyncio
async def test_list_teams(client):
    headers, _ = await get_auth_header(client, "lister@example.com", "Lister")
    await client.post(
        "/api/v1/teams",
        json={"name": "Team A"},
        headers=headers,
    )
    await client.post(
        "/api/v1/teams",
        json={"name": "Team B"},
        headers=headers,
    )
    response = await client.get("/api/v1/teams", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = {t["name"] for t in data}
    assert names == {"Team A", "Team B"}


@pytest.mark.asyncio
async def test_add_member(client):
    owner_headers, _ = await get_auth_header(client, "owner2@example.com", "Owner")
    # Register a second user to add as member
    await register_user(client, "member@example.com", "Member")

    # Create team
    create_res = await client.post(
        "/api/v1/teams",
        json={"name": "Collab Team"},
        headers=owner_headers,
    )
    team_id = create_res.json()["id"]

    # Add member by email
    response = await client.post(
        f"/api/v1/teams/{team_id}/members",
        json={"email": "member@example.com", "role": "editor"},
        headers=owner_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "member@example.com"
    assert data["role"] == "editor"

    # Verify team now has 2 members
    team_res = await client.get(f"/api/v1/teams/{team_id}", headers=owner_headers)
    assert len(team_res.json()["members"]) == 2


@pytest.mark.asyncio
async def test_share_project_with_team(client):
    owner_headers, _ = await get_auth_header(client, "sharer@example.com", "Sharer")

    # Create a project
    proj_res = await client.post(
        "/api/v1/projects",
        json={"title": "Shared Project"},
        headers=owner_headers,
    )
    project_id = proj_res.json()["id"]

    # Create a team
    team_res = await client.post(
        "/api/v1/teams",
        json={"name": "Project Team"},
        headers=owner_headers,
    )
    team_id = team_res.json()["id"]

    # Share project
    response = await client.post(
        f"/api/v1/teams/{team_id}/projects",
        json={"project_id": project_id},
        headers=owner_headers,
    )
    assert response.status_code == 201
    assert response.json()["success"] is True


@pytest.mark.asyncio
async def test_remove_member(client):
    owner_headers, _ = await get_auth_header(client, "remover@example.com", "Remover")
    member_headers, member_id = await get_auth_header(client, "removee@example.com", "Removee")

    # Create team
    create_res = await client.post(
        "/api/v1/teams",
        json={"name": "Remove Test"},
        headers=owner_headers,
    )
    team_id = create_res.json()["id"]

    # Add member
    add_res = await client.post(
        f"/api/v1/teams/{team_id}/members",
        json={"email": "removee@example.com", "role": "viewer"},
        headers=owner_headers,
    )
    added_user_id = add_res.json()["user_id"]

    # Remove member
    response = await client.delete(
        f"/api/v1/teams/{team_id}/members/{added_user_id}",
        headers=owner_headers,
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Verify team now has 1 member
    team_res = await client.get(f"/api/v1/teams/{team_id}", headers=owner_headers)
    assert len(team_res.json()["members"]) == 1


@pytest.mark.asyncio
async def test_delete_team(client):
    headers, _ = await get_auth_header(client, "deleter@example.com", "Deleter")

    create_res = await client.post(
        "/api/v1/teams",
        json={"name": "Delete Me"},
        headers=headers,
    )
    team_id = create_res.json()["id"]

    response = await client.delete(f"/api/v1/teams/{team_id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["success"] is True

    # Verify team is gone
    list_res = await client.get("/api/v1/teams", headers=headers)
    assert len(list_res.json()) == 0


@pytest.mark.asyncio
async def test_non_owner_cannot_remove_member(client):
    owner_headers, _ = await get_auth_header(client, "owner_perm@example.com", "Owner")
    editor_headers, _ = await get_auth_header(client, "editor_perm@example.com", "Editor")
    await register_user(client, "viewer_perm@example.com", "Viewer")

    # Create team
    create_res = await client.post(
        "/api/v1/teams",
        json={"name": "Perm Test"},
        headers=owner_headers,
    )
    team_id = create_res.json()["id"]

    # Add editor
    await client.post(
        f"/api/v1/teams/{team_id}/members",
        json={"email": "editor_perm@example.com", "role": "editor"},
        headers=owner_headers,
    )

    # Add viewer
    add_res = await client.post(
        f"/api/v1/teams/{team_id}/members",
        json={"email": "viewer_perm@example.com", "role": "viewer"},
        headers=owner_headers,
    )
    viewer_user_id = add_res.json()["user_id"]

    # Editor tries to remove viewer — should fail
    response = await client.delete(
        f"/api/v1/teams/{team_id}/members/{viewer_user_id}",
        headers=editor_headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_team_detail_uses_single_user_query(client, db_session):
    """GET /teams/{id} must not issue one users SELECT per member."""
    from sqlalchemy import event

    owner_headers, _ = await get_auth_header(client, "nplus1_owner@example.com", "Owner")
    await register_user(client, "nplus1_a@example.com", "A")
    await register_user(client, "nplus1_b@example.com", "B")

    create_res = await client.post(
        "/api/v1/teams", json={"name": "N+1 Team"}, headers=owner_headers
    )
    team_id = create_res.json()["id"]

    for email in ("nplus1_a@example.com", "nplus1_b@example.com"):
        await client.post(
            f"/api/v1/teams/{team_id}/members",
            json={"email": email, "role": "viewer"},
            headers=owner_headers,
        )

    statements: list[str] = []
    bind = db_session.sync_session.get_bind()

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(bind, "before_cursor_execute", _capture)
    try:
        res = await client.get(f"/api/v1/teams/{team_id}", headers=owner_headers)
    finally:
        event.remove(bind, "before_cursor_execute", _capture)

    assert res.status_code == 200
    assert len(res.json()["members"]) == 3

    user_selects = [s for s in statements if "FROM users" in s]
    # 1 for get_current_user + 1 batched IN query for the three members.
    assert len(user_selects) == 2, user_selects
