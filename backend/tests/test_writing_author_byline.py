"""Tests for the writer's own author-string rendering from the acquired full text's
byline.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")

from app.services.writing import _author_display_names, _paper_display_line  # noqa: E402


def test_renders_from_the_byline_when_accepted():
    paper = SimpleNamespace(
        title="The impact of written corrective feedback...",
        year=2024,
        authors=[{"name": "Mohammad Hasan Razmi"}, {"name": "Mohammad Hossein Ghane"}],
        metadata_={
            "fulltext_byline": {
                "surnames": ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"],
                "source": "fulltext",
            }
        },
    )
    assert _author_display_names(paper) == ["Ghane", "Razmi", "Dehghanpoor", "Nematollahi"]
    assert _paper_display_line(paper).startswith("Ghane, Razmi, Dehghanpoor, Nematollahi (2024)")


def test_falls_back_to_openalex_when_byline_was_not_accepted():
    paper = SimpleNamespace(
        title="Some paper",
        year=2020,
        authors=[{"name": "Jane Smith"}],
        metadata_={"fulltext_byline": {"source": "openalex"}},
    )
    assert _author_display_names(paper) == ["Jane Smith"]


def test_falls_back_to_openalex_when_no_byline_metadata_at_all():
    paper = SimpleNamespace(title="Some paper", year=2020, authors=[{"name": "Jane Smith"}])
    assert _author_display_names(paper) == ["Jane Smith"]
