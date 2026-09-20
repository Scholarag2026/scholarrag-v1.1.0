"""Tests for paper metadata extraction agent."""
import os
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.schemas.paper_upload import ExtractedPaperMetadata


def test_extracted_paper_metadata_schema():
    """Verify the output schema accepts valid data."""
    data = ExtractedPaperMetadata(
        title="Test Paper Title",
        authors=[{"name": "John Doe"}, {"name": "Jane Smith"}],
        year=2024,
        journal_name="Nature",
        doi="10.1234/test",
        abstract="This is a test abstract.",
    )
    assert data.title == "Test Paper Title"
    assert len(data.authors) == 2
    assert data.year == 2024


def test_extracted_paper_metadata_defaults():
    """Verify defaults when fields are missing."""
    data = ExtractedPaperMetadata()
    assert data.title == "Untitled"
    assert data.authors == []
    assert data.year is None
    assert data.doi is None


def test_agent_module_imports():
    """Verify the agent module can be imported."""
    from app.agents.paper_metadata_agent import extract_metadata_from_text
    assert callable(extract_metadata_from_text)
