"""Tests for paper upload service."""
import os
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.paper_upload import parse_bibtex, process_uploaded_files


def test_parse_bibtex_extracts_entries():
    bib = """
    @article{doe2024,
      title = {Test Paper Title},
      author = {John Doe and Jane Smith},
      year = {2024},
      journal = {Nature},
      doi = {10.1234/test},
      abstract = {A test abstract.},
    }
    """
    entries = parse_bibtex(bib)
    assert len(entries) == 1
    assert entries[0]["title"] == "Test Paper Title"
    assert entries[0]["year"] == "2024"


def test_paper_upload_service_imports():
    assert callable(process_uploaded_files)


def test_extract_text_from_docx():
    from app.services.paper_upload import _extract_text_from_docx
    assert callable(_extract_text_from_docx)


def test_upload_routes_exist():
    from app.api.papers import router
    routes = [r.path for r in router.routes]
    assert any("upload" in r for r in routes)


def test_confirm_routes_exist():
    from app.api.papers import router
    routes = [r.path for r in router.routes]
    assert any("upload-confirm" in r for r in routes)
