"""``fulltext_byline`` backfill for a paper already marked acquired.

``acquire_full_texts`` returns early, before any byline parse, for a paper whose own
metadata already carries ``fulltext_status == "acquired"`` (the idempotency guard that
skips the network cascade). The byline parse sits inside the SAME branch that
sets ``fulltext_status = "acquired"``, so a paper acquired before that parse existed --
an already-acquired paper, the Razmi and Ghane paper among them -- never gets a
``fulltext_byline`` entry at all, and its citation keeps rendering from OpenAlex's own
(differently ordered) author list.

This uses the identical committed fixture chunk `test_byline_author_string.py` already
validates the parser against.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from app.models.analysis_job import JobType  # noqa: E402
from tests.t5_fixtures import (  # noqa: E402
    make_session_factory,
    seed_job,
    seed_paper,
    seed_project,
    seed_user,
)

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/evidence/seeds12/53ae2131-383c-4642-ae68-6e14983b73f9.json"
)


def _fixture_chunk() -> dict:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["chunks"][0]


@pytest.mark.asyncio
async def test_backfills_the_byline_for_a_paper_already_marked_acquired(db_session):
    """The idempotency guard still makes no network call, but the byline this paper's
    own already-stored chunks name is written on the same pass, so its citation can
    render "Ghane et al. (2024)" rather than staying stuck on OpenAlex's own two-author,
    Razmi-first order forever."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="The impact of written corrective feedback",
        doi="10.1/ghane-razmi-2024",
        authors=[{"name": "Mohammad Hasan Razmi"}, {"name": "Mohammad Hossein Ghane"}],
        year=2024,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [_fixture_chunk()],
        },
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with patch(
        "app.clients.unpaywall.UnpaywallClient.lookup_locations", new_callable=AsyncMock
    ) as mock_lookup, patch(
        "app.services.fulltext.fetch_pdf_from_url", new_callable=AsyncMock
    ) as mock_fetch:
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_lookup.assert_not_awaited()
    mock_fetch.assert_not_awaited()

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result == {"acquired": 0, "abstract_only": 0, "already_acquired": 1}

    await db_session.refresh(paper)
    assert paper.metadata_["fulltext_byline"] == {
        "surnames": ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"],
        "source": "fulltext",
    }
    # The rest of the already-stored metadata is untouched.
    assert paper.metadata_["fulltext_status"] == "acquired"
    assert paper.metadata_["fulltext_chunks"] == [_fixture_chunk()]


@pytest.mark.asyncio
async def test_an_already_acquired_paper_whose_chunks_do_not_name_every_openalex_author_falls_back(
    db_session,
):
    """The same acceptance rule edit 9 applies at acquisition time also governs the
    backfill: a layout the parser cannot read, or a chunk that does not actually name
    every OpenAlex author, keeps today's OpenAlex-derived rendering rather than writing
    a wrong or partial byline."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title="Some Other Paper",
        doi="10.1/no-byline-match",
        authors=[{"name": "Jane Smith"}],
        year=2021,
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [{"section": "preamble", "text": "Abstract: nothing here names an author byline."}],
        },
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with patch(
        "app.clients.unpaywall.UnpaywallClient.lookup_locations", new_callable=AsyncMock
    ) as mock_lookup:
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()
    mock_lookup.assert_not_awaited()

    await db_session.refresh(paper)
    assert paper.metadata_["fulltext_byline"] == {"source": "openalex"}


@pytest.mark.asyncio
async def test_an_already_acquired_paper_that_already_carries_a_byline_is_left_alone(db_session):
    """A paper backfilled (or acquired) on an earlier pass already carries
    ``fulltext_byline``; the guard must not re-parse or overwrite it on every later
    run."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    existing_byline = {"surnames": ["Smith"], "source": "fulltext"}
    paper = await seed_paper(
        db_session,
        project,
        title="Already Has A Byline",
        doi="10.1/already-has-byline",
        authors=[{"name": "Jane Smith"}],
        metadata_={
            "fulltext_status": "acquired",
            "fulltext_chunks": [],
            "fulltext_byline": existing_byline,
        },
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with patch(
        "app.clients.unpaywall.UnpaywallClient.lookup_locations", new_callable=AsyncMock
    ) as mock_lookup:
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()
    mock_lookup.assert_not_awaited()

    await db_session.refresh(paper)
    assert paper.metadata_["fulltext_byline"] == existing_byline
