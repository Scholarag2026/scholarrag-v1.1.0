from app.services.graph import _compute_clusters


def test_compute_clusters_basic():
    """Two disconnected cliques should get different cluster IDs."""
    edges = [
        ("a", "b"), ("b", "c"), ("a", "c"),  # clique 1
        ("d", "e"), ("e", "f"), ("d", "f"),  # clique 2
    ]
    clusters = _compute_clusters(edges)
    assert isinstance(clusters, dict)
    assert clusters["a"] == clusters["b"] == clusters["c"]
    assert clusters["d"] == clusters["e"] == clusters["f"]
    assert clusters["a"] != clusters["d"]


def test_compute_clusters_empty():
    clusters = _compute_clusters([])
    assert clusters == {}


def test_compute_clusters_single_edge():
    clusters = _compute_clusters([("a", "b")])
    assert "a" in clusters
    assert "b" in clusters
