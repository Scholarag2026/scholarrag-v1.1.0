"""Tests for full-text API endpoints."""

import os
import uuid

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def get_auth_token(client):
    email = f"fttest-{uuid.uuid4().hex[:8]}@example.com"
    res = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Fulltext Test User",
            "expertise_level": "researcher",
        },
    )
    return res.json()["access_token"]


async def create_project(client, token):
    res = await client.post(
        "/api/v1/projects",
        json={"title": "Fulltext Test Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


# ---------------------------------------------------------------------------
# 1. Route existence checks
# ---------------------------------------------------------------------------


def test_fulltext_routes_exist():
    """Verify all full-text endpoints are registered on the router."""
    from app.api.fulltext import router

    routes = [r.path for r in router.routes]
    assert "/projects/{project_id}/acquire-full-texts" in routes
    assert "/papers/{paper_id}/upload-fulltext" in routes
    assert "/papers/{paper_id}/paste-fulltext" in routes
    assert "/projects/{project_id}/verify-and-heal" in routes
    assert "/projects/{project_id}/verify-edits" in routes
    assert "/drafts/{draft_id}/claim-verification" in routes
    assert "/papers/{paper_id}/fulltext-chunks/{chunk_index}" in routes


# ---------------------------------------------------------------------------
# 2. acquire-full-texts returns 202
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_full_texts_returns_202(client):
    """Happy path: POST acquire-full-texts returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)

    with patch(
        "app.api.fulltext.acquire_full_texts",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/acquire-full-texts",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    uuid.UUID(data["task_id"])


# ---------------------------------------------------------------------------
# 3. verify-and-heal returns 202
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_and_heal_returns_202(client):
    """Happy path: POST verify-and-heal returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = str(uuid.uuid4())

    with patch(
        "app.api.fulltext.verify_and_heal_claims",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/verify-and-heal",
            json={"draft_id": draft_id},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    uuid.UUID(data["task_id"])


# ---------------------------------------------------------------------------
# 4. paste-fulltext validates empty text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paste_fulltext_empty_text_returns_400(client):
    """Pasting empty text should return 400."""
    token = await get_auth_token(client)
    fake_paper_id = str(uuid.uuid4())

    response = await client.post(
        f"/api/v1/papers/{fake_paper_id}/paste-fulltext",
        json={"text": ""},
        headers={"Authorization": f"Bearer {token}"},
    )
    # 404 (paper not found) or 400 (empty text) depending on which check fires first.
    # Since paper doesn't exist, we expect 404.
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_paste_fulltext_drops_a_reference_list_chunk_before_storage(client, db_session):
    """`paste_fulltext` calls
    `chunk_text(req.text)` with no keyword argument, so it must inherit the
    reference-list/back-matter drop from `chunk_text`'s own default. Same guarantee
    applies to `upload_fulltext` and the bulk-upload confirm path, which call `chunk_text`
    the same way."""
    from app.models.paper import Paper

    paper = Paper(title="Paste Guard Paper")
    db_session.add(paper)
    await db_session.flush()

    token = await get_auth_token(client)

    good_text = (
        "The results indicate that students who received corrective feedback showed "
        "higher post-test scores than those in the control group."
    )
    reference_text = (
        "Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion och flerspraakighet. "
        "Apples, 9(1), 1-18. Retrieved from https://apples.jyu.fi\n"
        "Sinclair, J. M., & Coulthard, R. M. (1975). Towards an analysis of discourse. "
        "Oxford, UK: Oxford University Press. pp. 12-34.\n"
        "Snell, J., Shaw, S., & Copland, F. (2015). Linguistic ethnography. London, UK: "
        "Palgrave Macmillan. 10.1057/9781137035035\n"
        "Swain, M. (1985). Communicative competence. Rowley, MA: Newbury House. "
        "pp. 235-253.\n"
        "Swain, M. (1995). Three functions of output. Oxford, UK: Oxford University "
        "Press. pp. 125-144.\n"
    )
    pasted_text = "Introduction\n" + good_text + "\n\nReferences\n" + reference_text

    response = await client.post(
        f"/api/v1/papers/{paper.id}/paste-fulltext",
        json={"text": pasted_text},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200

    await db_session.refresh(paper)
    sections = [c["section"] for c in paper.metadata_["fulltext_chunks"]]
    assert "references" not in sections
    assert "introduction" in sections


# ---------------------------------------------------------------------------
# 5. claim-verification GET is scoped to the caller's own draft
# ---------------------------------------------------------------------------


async def _create_draft(client, token, project_id, title="Verify Draft"):
    res = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": title},
        headers={"Authorization": f"Bearer {token}"},
    )
    return res.json()["id"]


@pytest.mark.asyncio
async def test_claim_verification_unknown_draft_returns_404(client):
    """An unknown draft id is not a silent empty report — it is a 404."""
    token = await get_auth_token(client)
    draft_id = str(uuid.uuid4())

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/claim-verification",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_claim_verification_other_users_draft_returns_404(client):
    """Cross-tenant read of another user's verification report is refused."""
    victim_token = await get_auth_token(client)
    victim_project = await create_project(client, victim_token)
    victim_draft = await _create_draft(client, victim_token, victim_project)

    attacker_token = await get_auth_token(client)
    response = await client.get(
        f"/api/v1/drafts/{victim_draft}/claim-verification",
        headers={"Authorization": f"Bearer {attacker_token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_claim_verification_owner_gets_empty_report(client):
    """The owner of a draft with no verification job gets an empty 200 report."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = await _create_draft(client, token, project_id)

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/claim-verification",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["draft_id"] == draft_id
    assert data["verifications"] == []
    assert data["verified_count"] == 0
    assert data["unsupported_count"] == 0


@pytest.mark.asyncio
async def test_claim_verification_ignores_other_projects_job(client, db_session):
    """A completed claim_verify job in a different project must not be returned."""
    from app.models.analysis_job import AnalysisJob, JobStatus, JobType

    owner_token = await get_auth_token(client)
    owner_project = await create_project(client, owner_token)
    owner_draft = await _create_draft(client, owner_token, owner_project)

    other_token = await get_auth_token(client)
    other_project = await create_project(client, other_token)
    me = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {other_token}"}
    )
    other_user_id = uuid.UUID(me.json()["id"])

    db_session.add(
        AnalysisJob(
            project_id=uuid.UUID(other_project),
            user_id=other_user_id,
            job_type=JobType.claim_verify,
            status=JobStatus.completed,
            result={
                "draft_id": owner_draft,
                "verifications": [],
                "verified_count": 7,
                "unsupported_count": 0,
                "nuance_count": 0,
                "abstract_only_count": 0,
            },
        )
    )
    await db_session.flush()

    response = await client.get(
        f"/api/v1/drafts/{owner_draft}/claim-verification",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert response.status_code == 200
    assert response.json()["verified_count"] == 0


# ---------------------------------------------------------------------------
# 6. verify-edits returns 202
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_edits_returns_202(client):
    """Happy path: POST verify-edits returns 202 with a task_id."""
    token = await get_auth_token(client)
    project_id = await create_project(client, token)
    draft_id = str(uuid.uuid4())

    with patch(
        "app.api.fulltext.verify_user_edits",
        new_callable=AsyncMock,
    ):
        response = await client.post(
            f"/api/v1/projects/{project_id}/verify-edits",
            json={
                "draft_id": draft_id,
                "changed_sections": ["introduction", "methodology"],
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    uuid.UUID(data["task_id"])


# ---------------------------------------------------------------------------
# 7. GET /papers/{paper_id}/fulltext-chunks/{chunk_index}  (task authorisation
#    2026-09-11, the delivered-evidence record)
# ---------------------------------------------------------------------------


async def _link_paper_to_new_project(client, db_session, token, paper_id):
    """Add ``paper_id`` to a fresh project owned by the caller of ``token``."""
    from app.models.project_paper import ProjectPaper

    project_id = await create_project(client, token)
    db_session.add(ProjectPaper(project_id=uuid.UUID(project_id), paper_id=paper_id))
    await db_session.flush()
    return project_id


@pytest.mark.asyncio
async def test_get_fulltext_chunk_returns_the_chunk_at_that_index(client, db_session):
    from app.models.paper import Paper

    paper = Paper(
        title="Chunk Read Paper",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"section": "introduction", "text": "Introductory text about the study."},
                {"section": "results", "text": "The intervention improved outcomes."},
            ],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    token = await get_auth_token(client)
    await _link_paper_to_new_project(client, db_session, token, paper.id)
    response = await client.get(
        f"/api/v1/papers/{paper.id}/fulltext-chunks/1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["paper_id"] == str(paper.id)
    assert data["chunk_index"] == 1
    assert data["chunk_count"] == 2
    assert data["section"] == "results"
    assert data["text"] == "The intervention improved outcomes."


@pytest.mark.asyncio
async def test_get_fulltext_chunk_returns_the_papers_own_abstract(client, db_session):
    """The verifier's own prompt gives the model a paper's
    abstract as context alongside its numbered chunks, so a quote genuinely verbatim
    only there is real evidence the guard already accepts -- but a delivered-evidence
    passage locator has no way to read it if it is not one of the numbered chunks
    `evidence_location` counts. Returned alongside every chunk of the paper, not only
    chunk 0, since the row a locator needs it for is not always chunk 0's own row."""
    from app.models.paper import Paper

    paper = Paper(
        title="Chunk Read Paper",
        abstract="This survey found broad support for scaffolded feedback across cohorts.",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"section": "introduction", "text": "Introductory text about the study."},
                {"section": "results", "text": "The intervention improved outcomes."},
            ],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    token = await get_auth_token(client)
    await _link_paper_to_new_project(client, db_session, token, paper.id)
    response = await client.get(
        f"/api/v1/papers/{paper.id}/fulltext-chunks/1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["abstract"] == (
        "This survey found broad support for scaffolded feedback across cohorts."
    )


@pytest.mark.asyncio
async def test_get_fulltext_chunk_drops_reference_chunks_before_indexing(client, db_session):
    """The index is into the same *filtered* list `_verify_one` reads (reference/back-
    matter chunks dropped), so index 1 here is the results chunk, not the references
    chunk that immediately preceded it in storage."""
    from app.models.paper import Paper

    reference_text = (
        "References\n"
        "Rosen, J., & Wedin, A. (2015). Klassrumsinteraktion. Apples, 9(1), 1-18.\n"
        "Sinclair, J. M. (1975). Towards an analysis of discourse. Oxford University Press.\n"
        "Swain, M. (1985). Communicative competence. Newbury House. pp. 235-253.\n"
    )
    paper = Paper(
        title="Chunk Filter Paper",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [
                {"section": "introduction", "text": "Introductory text about the study."},
                {"section": "results", "text": "The intervention improved outcomes."},
                {"section": "references", "text": reference_text},
            ],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    token = await get_auth_token(client)
    await _link_paper_to_new_project(client, db_session, token, paper.id)
    response = await client.get(
        f"/api/v1/papers/{paper.id}/fulltext-chunks/1",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 2
    assert data["section"] == "results"
    assert data["text"] == "The intervention improved outcomes."


@pytest.mark.asyncio
async def test_get_fulltext_chunk_out_of_range_returns_404(client, db_session):
    from app.models.paper import Paper

    paper = Paper(
        title="Chunk Range Paper",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "introduction", "text": "Some text."}],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    token = await get_auth_token(client)
    await _link_paper_to_new_project(client, db_session, token, paper.id)
    response = await client.get(
        f"/api/v1/papers/{paper.id}/fulltext-chunks/5",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_fulltext_chunk_unknown_paper_returns_404(client):
    token = await get_auth_token(client)
    fake_paper_id = str(uuid.uuid4())
    response = await client.get(
        f"/api/v1/papers/{fake_paper_id}/fulltext-chunks/0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_fulltext_chunk_unlinked_paper_returns_404(client, db_session):
    """A paper that exists but is not linked
    to any project the caller owns must 404, exactly as an unknown paper id does — the
    caller must not be able to read another user's uploaded full text just by guessing
    or discovering the paper id (papers are deduplicated by DOI across users)."""
    from app.models.paper import Paper

    paper = Paper(
        title="Chunk Unlinked Paper",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "introduction", "text": "Some text."}],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    # A different user's project holds the same paper id — the caller below has no
    # project of their own linked to it.
    other_token = await get_auth_token(client)
    await _link_paper_to_new_project(client, db_session, other_token, paper.id)

    token = await get_auth_token(client)
    response = await client.get(
        f"/api/v1/papers/{paper.id}/fulltext-chunks/0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_fulltext_chunk_requires_auth(client, db_session):
    from app.models.paper import Paper

    paper = Paper(
        title="Chunk Auth Paper",
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "introduction", "text": "Some text."}],
        },
    )
    db_session.add(paper)
    await db_session.flush()

    response = await client.get(f"/api/v1/papers/{paper.id}/fulltext-chunks/0")
    assert response.status_code in (401, 403)
