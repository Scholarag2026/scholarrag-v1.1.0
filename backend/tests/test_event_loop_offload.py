"""D20: CPU/IO-bound sync work must not run on the request event loop."""

import threading
import uuid


async def test_register_and_login_run_argon2_off_the_event_loop(client, monkeypatch):
    from app.core import security

    real = security.ph
    seen: dict[str, str] = {}

    class RecordingHasher:
        def hash(self, password: str) -> str:
            seen["hash_thread"] = threading.current_thread().name
            return real.hash(password)

        def verify(self, hash: str, password: str) -> bool:
            seen["verify_thread"] = threading.current_thread().name
            return real.verify(hash, password)

    monkeypatch.setattr(security, "ph", RecordingHasher())

    email = f"offload-{uuid.uuid4().hex[:8]}@example.com"
    register = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Offload User",
            "expertise_level": "researcher",
        },
    )
    assert register.status_code == 201

    login = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123!"},
    )
    assert login.status_code == 200

    main_thread = threading.main_thread().name
    assert seen["hash_thread"] != main_thread
    assert seen["verify_thread"] != main_thread


async def test_password_async_helpers_round_trip():
    from app.core.security import hash_password_async, verify_password_async

    digest = await hash_password_async("StrongPass123!")
    assert await verify_password_async("StrongPass123!", digest) is True
    assert await verify_password_async("WrongPass123!", digest) is False


class _FakeStorage:
    """In-memory stand-in for StorageService so tests never touch the filesystem."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def save(self, key: str, content: bytes) -> str:
        self.files[key] = content
        return key

    def read(self, key: str) -> bytes:
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def delete(self, key: str) -> None:
        self.files.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self.files


async def test_dataset_upload_parses_and_serializes_off_the_event_loop(db_session, monkeypatch):
    from app.models.project import Project
    from app.models.user import User
    from app.services import dataset as dataset_service

    storage = _FakeStorage()
    monkeypatch.setattr(dataset_service, "storage_service", storage)

    seen: dict[str, str] = {}
    real_to_parquet = dataset_service.df_to_parquet_bytes

    def recording_to_parquet(df):
        seen["thread"] = threading.current_thread().name
        return real_to_parquet(df)

    monkeypatch.setattr(dataset_service, "df_to_parquet_bytes", recording_to_parquet)

    user = User(
        email=f"ds-{uuid.uuid4().hex[:8]}@example.com", password_hash="x", name="DS User"
    )
    db_session.add(user)
    await db_session.flush()
    project = Project(user_id=user.id, title="DS Project")
    db_session.add(project)
    await db_session.flush()

    content = b"name,score\nalice,1\nbob,2\n"
    result = await dataset_service.upload_dataset(
        db_session, project.id, user.id, "scores.csv", content
    )

    assert result["row_count"] == 2
    assert [c["name"] for c in result["columns"]] == ["name", "score"]
    assert result["preview"] == [{"name": "alice", "score": 1}, {"name": "bob", "score": 2}]
    assert len(storage.files) == 2
    assert seen["thread"] != threading.main_thread().name


async def test_dataset_preview_reads_parquet_off_the_event_loop(db_session, monkeypatch):
    from app.models.project import Project
    from app.models.user import User
    from app.services import dataset as dataset_service

    storage = _FakeStorage()
    monkeypatch.setattr(dataset_service, "storage_service", storage)

    user = User(
        email=f"ds-{uuid.uuid4().hex[:8]}@example.com", password_hash="x", name="DS User"
    )
    db_session.add(user)
    await db_session.flush()
    project = Project(user_id=user.id, title="DS Project")
    db_session.add(project)
    await db_session.flush()

    uploaded = await dataset_service.upload_dataset(
        db_session, project.id, user.id, "scores.csv", b"name,score\nalice,1\nbob,2\n"
    )

    seen: dict[str, str] = {}
    real_from_parquet = dataset_service.parquet_bytes_to_df

    def recording_from_parquet(content):
        seen["thread"] = threading.current_thread().name
        return real_from_parquet(content)

    monkeypatch.setattr(dataset_service, "parquet_bytes_to_df", recording_from_parquet)

    preview = await dataset_service.get_dataset_preview(
        db_session, uuid.UUID(uploaded["id"]), user.id
    )

    assert preview["row_count"] == 2
    assert preview["preview"][0]["name"] == "alice"
    assert seen["thread"] != threading.main_thread().name


async def _register_and_project(client) -> tuple[str, str]:
    email = f"export-offload-{uuid.uuid4().hex[:8]}@example.com"
    register = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Export User",
            "expertise_level": "researcher",
        },
    )
    token = register.json()["access_token"]
    project = await client.post(
        "/api/v1/projects",
        json={"title": "Export Offload Project"},
        headers={"Authorization": f"Bearer {token}"},
    )
    return token, project.json()["id"]


async def test_docx_export_runs_off_the_event_loop(client, monkeypatch):
    from app.services import export as export_service

    token, project_id = await _register_and_project(client)
    created = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Offload Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = created.json()["id"]

    seen: dict[str, str] = {}
    real_export_docx = export_service.export_docx

    def recording_export_docx(title, content):
        seen["thread"] = threading.current_thread().name
        return real_export_docx(title, content)

    monkeypatch.setattr(export_service, "export_docx", recording_export_docx)

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=docx",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.content[:2] == b"PK"
    assert seen["thread"] != threading.main_thread().name


async def test_upload_fulltext_extraction_runs_off_the_event_loop(client, db_session, monkeypatch):
    """POST /papers/{id}/upload-fulltext must extract/chunk the PDF off the event loop
    (BLOCKING-SYNC-EVENT-LOOP #27)."""
    from pathlib import Path

    import app.api.fulltext as ft_module
    from app.models.paper import Paper

    paper = Paper(title="Offload Upload Paper")
    db_session.add(paper)
    await db_session.flush()

    seen: dict[str, str] = {}
    real_extract = ft_module.extract_text_from_pdf

    def recording_extract(pdf_bytes):
        seen["thread"] = threading.current_thread().name
        return real_extract(pdf_bytes)

    monkeypatch.setattr(ft_module, "extract_text_from_pdf", recording_extract)

    email = f"upload-offload-{uuid.uuid4().hex[:8]}@example.com"
    register = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Upload Offload User",
            "expertise_level": "researcher",
        },
    )
    token = register.json()["access_token"]

    pdf_bytes = (Path(__file__).parent / "fixtures" / "sample_paper.pdf").read_bytes()
    response = await client.post(
        f"/api/v1/papers/{paper.id}/upload-fulltext",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert seen["thread"] != threading.main_thread().name


async def test_paste_fulltext_chunking_runs_off_the_event_loop(client, db_session, monkeypatch):
    """POST /papers/{id}/paste-fulltext must chunk the pasted text off the event loop
    (BLOCKING-SYNC-EVENT-LOOP #27)."""
    import app.api.fulltext as ft_module
    from app.models.paper import Paper

    paper = Paper(title="Offload Paste Paper")
    db_session.add(paper)
    await db_session.flush()

    seen: dict[str, str] = {}
    real_chunk = ft_module.chunk_text

    def recording_chunk(text):
        seen["thread"] = threading.current_thread().name
        return real_chunk(text)

    monkeypatch.setattr(ft_module, "chunk_text", recording_chunk)

    email = f"paste-offload-{uuid.uuid4().hex[:8]}@example.com"
    register = await client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPass123!",
            "name": "Paste Offload User",
            "expertise_level": "researcher",
        },
    )
    token = register.json()["access_token"]

    response = await client.post(
        f"/api/v1/papers/{paper.id}/paste-fulltext",
        json={"text": "Some pasted full text for offload testing."},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert seen["thread"] != threading.main_thread().name


async def test_paper_upload_pdf_and_docx_extraction_run_off_the_event_loop(monkeypatch):
    """process_uploaded_files must run pymupdf/python-docx extraction and chunking in a
    worker thread for every uploaded file (BLOCKING-SYNC-EVENT-LOOP #27)."""
    import io
    from pathlib import Path
    from types import SimpleNamespace

    from docx import Document

    import app.services.paper_upload as pu_module
    from tests.t5_fixtures import make_session_factory

    pdf_bytes = (Path(__file__).parent / "fixtures" / "sample_paper.pdf").read_bytes()

    buf = io.BytesIO()
    doc = Document()
    doc.add_paragraph("A DOCX paragraph used for event-loop offload testing.")
    doc.save(buf)
    docx_bytes = buf.getvalue()

    seen: dict[str, str] = {}
    real_extract_pdf = pu_module.extract_text_from_pdf
    real_extract_docx = pu_module._extract_text_from_docx

    def recording_extract_pdf(content):
        seen["pdf_thread"] = threading.current_thread().name
        return real_extract_pdf(content)

    def recording_extract_docx(content):
        seen["docx_thread"] = threading.current_thread().name
        return real_extract_docx(content)

    monkeypatch.setattr(pu_module, "extract_text_from_pdf", recording_extract_pdf)
    monkeypatch.setattr(pu_module, "_extract_text_from_docx", recording_extract_docx)

    async def fake_extract_metadata(text):
        return SimpleNamespace(
            title="T", authors=[], year=None, journal_name=None, doi=None, abstract=None
        )

    monkeypatch.setattr(pu_module, "extract_metadata_from_text", fake_extract_metadata)

    factory, engine = make_session_factory()
    try:
        await pu_module.process_uploaded_files(
            project_id=uuid.uuid4(),
            job_id=uuid.uuid4(),
            session_factory=factory,
            files_data=[
                {"name": "a.pdf", "content": pdf_bytes, "extension": ".pdf"},
                {"name": "b.docx", "content": docx_bytes, "extension": ".docx"},
            ],
        )
    finally:
        await engine.dispose()

    main = threading.main_thread().name
    assert seen["pdf_thread"] != main
    assert seen["docx_thread"] != main


async def test_acquire_full_texts_extraction_runs_off_the_event_loop(db_session, monkeypatch):
    """acquire_full_texts must extract/chunk downloaded PDFs off the event loop
    (BLOCKING-SYNC-EVENT-LOOP #27)."""
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    import app.services.fulltext as fulltext_module
    from app.models.analysis_job import JobType
    from tests.t5_fixtures import (
        make_session_factory,
        seed_job,
        seed_paper,
        seed_project,
        seed_user,
    )

    pdf_bytes = (Path(__file__).parent / "fixtures" / "sample_paper.pdf").read_bytes()

    user = await seed_user(db_session)
    project = await seed_project(db_session, user)
    await seed_paper(
        db_session,
        project,
        # Must match sample_paper.pdf's own real title ("Effects of AI Tutoring
        # on Student Performance") -- app/services/fulltext.py rejects a full text as
        # wrong_work when the paper's title/author does not describe the extracted text,
        # so an unrelated administrative label here would (correctly) be rejected before
        # this test's own offload assertion is ever reached.
        title="Effects of AI Tutoring on Student Performance",
        doi="10.1/needs-extraction-offload",
        full_text_url="https://example.org/paper.pdf",
    )
    job = await seed_job(db_session, project, JobType.fulltext_acquire)

    seen: dict[str, str] = {}
    real_extract = fulltext_module.extract_text_from_pdf

    def recording_extract(pdf_bytes_arg):
        seen["thread"] = threading.current_thread().name
        return real_extract(pdf_bytes_arg)

    monkeypatch.setattr(fulltext_module, "extract_text_from_pdf", recording_extract)

    factory, engine = make_session_factory()
    with patch(
        "app.services.fulltext.fetch_pdf_from_url",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ):
        await fulltext_module.acquire_full_texts(
            project_id=project.id, job_id=job.id, session_factory=factory, paper_ids=None
        )
    await engine.dispose()

    assert seen["thread"] != threading.main_thread().name
    await db_session.refresh(job)
    assert job.result == {"acquired": 1, "abstract_only": 0, "already_acquired": 0}


async def test_quantitative_descriptive_stats_run_off_the_event_loop(db_session, monkeypatch):
    """run_quantitative_analysis must load the parquet and compute descriptive stats off
    the event loop (BLOCKING-SYNC-EVENT-LOOP #27)."""
    from app.models.dataset import Dataset
    from app.models.project import Project
    from app.models.user import User
    from app.schemas.quantitative import AnalysisPlan, AnalysisResults, DescriptiveStats
    from app.services import dataset as dataset_service
    from app.services import quantitative as quant_module
    from tests.t5_fixtures import FakeAgent, make_session_factory

    storage = _FakeStorage()
    monkeypatch.setattr(dataset_service, "storage_service", storage)

    user = User(
        email=f"quant-offload-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
        name="Quant User",
    )
    db_session.add(user)
    await db_session.commit()
    project = Project(user_id=user.id, title="Quant Offload Project", description="A quant study.")
    db_session.add(project)
    await db_session.commit()

    uploaded = await dataset_service.upload_dataset(
        db_session, project.id, user.id, "scores.csv", b"name,score\nalice,1\nbob,2\n"
    )
    await db_session.commit()
    dataset = await db_session.get(Dataset, uuid.UUID(uploaded["id"]))

    plan = AnalysisPlan(methods=[], variable_mappings=[], assumptions=[], warnings=[])
    stats = DescriptiveStats(
        numeric_stats=[], categorical_stats=[], correlation_matrix={}, normality_tests=[]
    )
    final = AnalysisResults(
        plan=plan,
        descriptive_stats=stats,
        predictions=[],
        code_templates=[],
        interpretations=[],
        assumption_checks=[],
    )

    monkeypatch.setattr(quant_module, "get_quantitative_plan_agent", lambda: FakeAgent(output=plan))
    monkeypatch.setattr(
        quant_module, "get_quantitative_interpretation_agent", lambda: FakeAgent(output=final)
    )

    seen: dict[str, str] = {}
    real_compute = quant_module.compute_all_descriptive_stats

    def recording_compute(df):
        seen["compute_thread"] = threading.current_thread().name
        return real_compute(df)

    monkeypatch.setattr(quant_module, "compute_all_descriptive_stats", recording_compute)

    real_from_parquet = dataset_service.parquet_bytes_to_df

    def recording_from_parquet(content):
        seen["parquet_thread"] = threading.current_thread().name
        return real_from_parquet(content)

    monkeypatch.setattr(dataset_service, "parquet_bytes_to_df", recording_from_parquet)

    factory, engine = make_session_factory()
    try:
        await quant_module.run_quantitative_analysis(
            project_id=project.id,
            dataset_id=dataset.id,
            job_id=uuid.uuid4(),
            session_factory=factory,
        )
    finally:
        await engine.dispose()

    main_thread = threading.main_thread().name
    assert seen["compute_thread"] != main_thread
    assert seen["parquet_thread"] != main_thread


async def test_qualitative_coding_dataset_load_runs_off_the_event_loop(db_session, monkeypatch):
    """run_qualitative_coding must load the parquet dataset off the event loop
    (BLOCKING-SYNC-EVENT-LOOP #27)."""
    from app.models.dataset import Dataset
    from app.models.project import Project
    from app.models.user import User
    from app.schemas.qualitative import CodebookSchema, CodingResult
    from app.services import dataset as dataset_service
    from app.services import qualitative as qual_module
    from tests.t5_fixtures import FakeAgent, make_session_factory

    storage = _FakeStorage()
    monkeypatch.setattr(dataset_service, "storage_service", storage)

    user = User(
        email=f"qual-offload-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="x",
        name="Qual User",
    )
    db_session.add(user)
    await db_session.commit()
    project = Project(user_id=user.id, title="Qual Offload Project", description="A qual study.")
    db_session.add(project)
    await db_session.commit()

    long_comment_1 = "This is a sufficiently long free-text comment for text classification. " * 2
    long_comment_2 = "Another sufficiently long free-text comment for classification purposes. " * 2
    csv_content = f'comment,score\n"{long_comment_1}",1\n"{long_comment_2}",2\n'.encode("utf-8")

    uploaded = await dataset_service.upload_dataset(
        db_session, project.id, user.id, "comments.csv", csv_content
    )
    await db_session.commit()
    dataset = await db_session.get(Dataset, uuid.UUID(uploaded["id"]))
    assert any(c["dtype"] == "text" for c in dataset.columns), dataset.columns

    coding_result = CodingResult(
        codebook=CodebookSchema(codes=[], themes=[]),
        coded_segments=[],
        uncertainties=[],
        annotation_notes=[],
    )
    monkeypatch.setattr(
        qual_module, "get_qualitative_coding_agent", lambda: FakeAgent(output=coding_result)
    )

    seen: dict[str, str] = {}
    real_from_parquet = dataset_service.parquet_bytes_to_df

    def recording_from_parquet(content):
        seen["thread"] = threading.current_thread().name
        return real_from_parquet(content)

    monkeypatch.setattr(dataset_service, "parquet_bytes_to_df", recording_from_parquet)

    factory, engine = make_session_factory()
    try:
        await qual_module.run_qualitative_coding(
            project_id=project.id,
            dataset_id=dataset.id,
            job_id=uuid.uuid4(),
            session_factory=factory,
        )
    finally:
        await engine.dispose()

    assert seen["thread"] != threading.main_thread().name


async def test_pdf_export_runs_off_the_event_loop(client, monkeypatch):
    from app.services import export as export_service

    token, project_id = await _register_and_project(client)
    created = await client.post(
        f"/api/v1/projects/{project_id}/drafts",
        json={"title": "Offload PDF Draft"},
        headers={"Authorization": f"Bearer {token}"},
    )
    draft_id = created.json()["id"]

    seen: dict[str, str] = {}

    def fake_export_pdf(title, content):
        seen["thread"] = threading.current_thread().name
        import io

        return io.BytesIO(b"%PDF-1.4 stub")

    monkeypatch.setattr(export_service, "export_pdf", fake_export_pdf)

    response = await client.get(
        f"/api/v1/drafts/{draft_id}/export?format=pdf",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert seen["thread"] != threading.main_thread().name
