"""Tests for graph service helper functions."""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key-not-real")


def test_paper_to_graph_node_library():
    from app.services.graph import _paper_to_graph_node

    class MockPaper:
        id = "uuid-1"
        title = "Test Paper"
        authors = [{"name": "Smith"}]
        year = 2023
        citation_count = 50
        quality_score = 0.8
        doi = "10.1000/test"
        abstract = "An abstract"

    node = _paper_to_graph_node(MockPaper(), in_library=True)
    assert node.id == "uuid-1"
    assert node.in_library is True
    assert node.authors == ["Smith"]
    assert node.quality_score == 0.8


def test_paper_to_graph_node_external():
    from app.services.graph import _paper_to_graph_node

    class MockPaper:
        id = "uuid-2"
        title = "External Paper"
        authors = []
        year = None
        citation_count = None
        quality_score = None
        doi = None
        abstract = None

    node = _paper_to_graph_node(MockPaper(), in_library=False)
    assert node.in_library is False
    assert node.authors == []


def test_paper_to_graph_node_with_is_seed():
    from app.services.graph import _paper_to_graph_node

    class MockPaper:
        id = "uuid-1"
        title = "Test Paper"
        authors = [{"name": "Smith"}]
        year = 2023
        citation_count = 50
        quality_score = 0.8
        doi = "10.1000/test"
        abstract = "An abstract"

    node = _paper_to_graph_node(MockPaper(), in_library=True, is_seed=True)
    assert node.is_seed is True

    node2 = _paper_to_graph_node(MockPaper(), in_library=False, is_seed=False)
    assert node2.is_seed is False


def test_compute_doi_scores_library_papers_score_highest():
    from app.services.graph import _compute_doi_scores

    scores = _compute_doi_scores(
        node_ids=["lib1", "lib2", "ext1", "ext2"],
        adjacency={"lib1": ["ext1"], "ext1": ["lib1", "ext2"], "ext2": ["ext1"], "lib2": []},
        library_ids={"lib1", "lib2"},
        citation_counts={"lib1": 100, "lib2": 50, "ext1": 200, "ext2": 10},
        focus_id=None,
    )
    assert scores["lib1"] > scores["ext2"]
    assert scores["lib2"] > scores["ext2"]
    assert scores["ext1"] > scores["ext2"]


def test_compute_doi_scores_focus_boosts_nearby():
    from app.services.graph import _compute_doi_scores

    scores_no_focus = _compute_doi_scores(
        node_ids=["lib1", "ext1", "ext2"],
        adjacency={"lib1": ["ext1"], "ext1": ["lib1", "ext2"], "ext2": ["ext1"]},
        library_ids={"lib1"},
        citation_counts={"lib1": 100, "ext1": 50, "ext2": 50},
        focus_id=None,
    )
    scores_focus_ext1 = _compute_doi_scores(
        node_ids=["lib1", "ext1", "ext2"],
        adjacency={"lib1": ["ext1"], "ext1": ["lib1", "ext2"], "ext2": ["ext1"]},
        library_ids={"lib1"},
        citation_counts={"lib1": 100, "ext1": 50, "ext2": 50},
        focus_id="ext1",
    )
    assert scores_focus_ext1["ext2"] > scores_no_focus["ext2"]


def test_graph_service_has_no_sleep_and_no_semantic_scholar():
    """The build loop must be limiter-bound, not sleep-bound (issue #17), and OpenAlex-only."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "app" / "services" / "graph.py"
    ).read_text(encoding="utf-8")

    assert "asyncio.sleep" not in source
    assert "SemanticScholarClient" not in source
    assert "_resolve_s2_id" not in source
