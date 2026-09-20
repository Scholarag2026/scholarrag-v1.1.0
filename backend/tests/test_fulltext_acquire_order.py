"""Acquisition must skip already-acquired papers BEFORE paying for the network cascade."""

import os

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


@pytest.mark.asyncio
async def test_already_acquired_paper_makes_no_network_call(db_session):
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        title="Already Acquired",
        doi="10.1/already-acquired",
        full_text_url="https://example.org/paper.pdf",
        metadata_={"fulltext_status": "acquired", "fulltext_chunks": []},
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


@pytest.mark.asyncio
async def test_new_paper_still_runs_the_unpaywall_cascade(db_session):
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(db_session, project, title="Fresh Paper", doi="10.1/fresh")
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with patch(
        "app.clients.unpaywall.UnpaywallClient.lookup_locations",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_lookup, patch(
        "app.services.fulltext.fetch_pdf_from_url", new_callable=AsyncMock, return_value=None
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_lookup.assert_awaited_once_with("10.1/fresh")
    await db_session.refresh(job)
    assert job.result == {"acquired": 0, "abstract_only": 1, "already_acquired": 0}


def test_fulltext_service_has_no_core_leg():
    import app.services.fulltext as fulltext_module

    assert not hasattr(fulltext_module, "CoreClient")


# ---------------------------------------------------------------------------------------
# F4 acquisition: try every Unpaywall location (already ordered by `lookup_locations`),
# then the stored `full_text_url`, stopping at the first download that passes the
# identity guard -- a download that yields a real PDF about the WRONG paper must not end
# the search early.
# ---------------------------------------------------------------------------------------


MATCHING_TITLE = "A Study of Feedback"
MATCHING_AUTHORS = [{"name": "Jane Smith"}]
WRONG_WORK_TEXT_1 = "An Unrelated Paper About Something Else.\n" + "Filler text. " * 40
WRONG_WORK_TEXT_2 = "Yet Another Unrelated Paper.\n" + "Filler text. " * 40
MATCHING_TEXT = f"{MATCHING_TITLE}\nJane Smith\n\n" + "Real content about feedback. " * 40


@pytest.mark.asyncio
async def test_unpaywall_locations_are_tried_in_order_stopping_at_first_identity_match(
    db_session,
):
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/multi-location",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    url1 = "https://example.org/published.pdf"
    url2 = "https://example.org/accepted.pdf"
    url3 = "https://example.org/submitted.pdf"
    pdf_by_url = {url1: b"%PDF-1", url2: b"%PDF-2", url3: b"%PDF-3"}
    text_by_bytes = {
        b"%PDF-1": WRONG_WORK_TEXT_1,
        b"%PDF-2": WRONG_WORK_TEXT_2,
        b"%PDF-3": MATCHING_TEXT,
    }

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=[url1, url2, url3],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=lambda url: pdf_by_url[url],
        ) as mock_fetch,
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            side_effect=lambda pdf_bytes: text_by_bytes[pdf_bytes],
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    assert [c.args[0] for c in mock_fetch.await_args_list] == [url1, url2, url3]

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert metadata["fulltext_source"] == "unpaywall"
    assert "fulltext_reason" not in metadata

    await db_session.refresh(job)
    assert job.status.value == "completed", job.error
    assert job.result["acquired"] == 1
    assert "wrong_work" not in job.result


@pytest.mark.asyncio
async def test_falls_through_to_full_text_url_when_every_unpaywall_location_fails_identity(
    db_session,
):
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/falls-through",
        full_text_url="https://example.org/stored.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    unpaywall_url = "https://example.org/unpaywall-wrong.pdf"
    stored_url = "https://example.org/stored.pdf"
    pdf_by_url = {unpaywall_url: b"%PDF-WRONG", stored_url: b"%PDF-RIGHT"}
    text_by_bytes = {b"%PDF-WRONG": WRONG_WORK_TEXT_1, b"%PDF-RIGHT": MATCHING_TEXT}

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=[unpaywall_url],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=lambda url: pdf_by_url[url],
        ) as mock_fetch,
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            side_effect=lambda pdf_bytes: text_by_bytes[pdf_bytes],
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    assert [c.args[0] for c in mock_fetch.await_args_list] == [unpaywall_url, stored_url]

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert metadata["fulltext_source"] == "stored_url"

    await db_session.refresh(job)
    assert job.result["acquired"] == 1


@pytest.mark.asyncio
async def test_wrong_work_reported_only_when_no_candidate_ever_matches(db_session):
    """When every candidate across the whole cascade yields a real PDF about the wrong
    paper, the paper is rejected as `wrong_work` (not silently left `abstract_only` with
    no reason), and the job counts it."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/never-matches",
        full_text_url="https://example.org/stored-wrong.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    unpaywall_url = "https://example.org/unpaywall-wrong.pdf"
    stored_url = "https://example.org/stored-wrong.pdf"
    pdf_by_url = {unpaywall_url: b"%PDF-WRONG-1", stored_url: b"%PDF-WRONG-2"}
    text_by_bytes = {b"%PDF-WRONG-1": WRONG_WORK_TEXT_1, b"%PDF-WRONG-2": WRONG_WORK_TEXT_2}

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=[unpaywall_url],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=lambda url: pdf_by_url[url],
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            side_effect=lambda pdf_bytes: text_by_bytes[pdf_bytes],
        ),
        patch("app.services.fulltext.chunk_text") as mock_chunk_text,
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_chunk_text.assert_not_called()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "abstract_only"
    assert metadata["fulltext_reason"] == "wrong_work"
    # The reported fragment is from the LAST candidate tried, not the first.
    assert "unrelated paper" in metadata["fulltext_wrong_work_fragment"].lower()

    await db_session.refresh(job)
    assert job.result["wrong_work"] == 1


@pytest.mark.asyncio
async def test_metadata_oa_url_is_still_tried_as_a_final_fallback(db_session):
    """The pre-existing OpenAlex/metadata `oa_url` fallback is kept as a trailing
    candidate after Unpaywall and the stored `full_text_url` are both exhausted."""
    from app.services.fulltext import acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/metadata-fallback",
        metadata_={"oa_url": "https://example.org/from-metadata.pdf"},
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            return_value=b"%PDF-META",
        ) as mock_fetch,
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=MATCHING_TEXT,
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_fetch.assert_awaited_once_with("https://example.org/from-metadata.pdf")

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "acquired"
    assert metadata["fulltext_source"] == "metadata_oa_url"

    await db_session.refresh(job)
    assert job.result["acquired"] == 1


# ---------------------------------------------------------------------------------------
# F4 acquisition: a paper where every attempt across the whole cascade returned a
# publisher interstitial (never a real PDF) is reported with `fulltext_reason
# publisher_interstitial`, distinct from `wrong_work` (a real PDF about another work).
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_attempt_returning_html_marks_publisher_interstitial_and_counts_it(
    db_session,
):
    from app.services.fulltext import PublisherInterstitialError, acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/all-interstitial",
        full_text_url="https://example.org/stored.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/unpaywall.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=PublisherInterstitialError("blocked"),
        ),
        patch("app.services.fulltext.chunk_text") as mock_chunk_text,
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    mock_chunk_text.assert_not_called()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "abstract_only"
    assert metadata["fulltext_reason"] == "publisher_interstitial"
    assert "fulltext_wrong_work_fragment" not in metadata

    await db_session.refresh(job)
    assert job.result["interstitial"] == 1
    assert "wrong_work" not in job.result


@pytest.mark.asyncio
async def test_a_generic_failure_alongside_an_interstitial_is_not_counted_as_interstitial(
    db_session,
):
    """Only when EVERY attempt looked like an interstitial does the paper get that
    reason -- a candidate that failed for some other reason (here: a plain download
    failure) leaves the paper a bare, unexplained `abstract_only`."""
    from app.services.fulltext import PublisherInterstitialError, acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/mixed-failure",
        full_text_url="https://example.org/stored.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/unpaywall.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=[PublisherInterstitialError("blocked"), None],
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "abstract_only"
    assert "fulltext_reason" not in metadata

    await db_session.refresh(job)
    assert "interstitial" not in job.result
    assert "wrong_work" not in job.result


@pytest.mark.asyncio
async def test_an_interstitial_then_a_real_wrong_work_pdf_reports_wrong_work(db_session):
    """A real (if wrong-identity) PDF anywhere in the cascade rules out
    `publisher_interstitial` -- the paper was never actually blocked everywhere."""
    from app.services.fulltext import PublisherInterstitialError, acquire_full_texts

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    paper = await seed_paper(
        db_session,
        project,
        title=MATCHING_TITLE,
        authors=MATCHING_AUTHORS,
        doi="10.1/interstitial-then-wrong-work",
        full_text_url="https://example.org/stored.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    factory, engine = make_session_factory()
    with (
        patch(
            "app.clients.unpaywall.UnpaywallClient.lookup_locations",
            new_callable=AsyncMock,
            return_value=["https://example.org/unpaywall.pdf"],
        ),
        patch(
            "app.services.fulltext.fetch_pdf_from_url",
            new_callable=AsyncMock,
            side_effect=[PublisherInterstitialError("blocked"), b"%PDF-WRONG"],
        ),
        patch(
            "app.services.fulltext.extract_text_from_pdf",
            return_value=WRONG_WORK_TEXT_1,
        ),
    ):
        await acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    await db_session.refresh(paper)
    metadata = paper.metadata_ or {}
    assert metadata["fulltext_status"] == "abstract_only"
    assert metadata["fulltext_reason"] == "wrong_work"

    await db_session.refresh(job)
    assert job.result["wrong_work"] == 1
    assert "interstitial" not in job.result
