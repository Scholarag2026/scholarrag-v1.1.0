"""Tests for citation graph schemas."""



def test_graph_node_valid():
    from app.schemas.graph import GraphNode

    node = GraphNode(
        id="550e8400-e29b-41d4-a716-446655440000",
        title="Impact of Technology on Learning",
        authors=["Smith, J.", "Jones, A."],
        year=2023,
        citation_count=45,
        quality_score=0.75,
        in_library=True,
        doi="10.1234/test",
        abstract="This study examines...",
    )
    assert node.in_library is True
    assert node.quality_score == 0.75


def test_graph_node_minimal():
    from app.schemas.graph import GraphNode

    node = GraphNode(
        id="id1",
        title="A Paper",
        authors=[],
        year=None,
        citation_count=None,
        quality_score=None,
        in_library=False,
        doi=None,
        abstract=None,
    )
    assert node.in_library is False


def test_graph_edge():
    from app.schemas.graph import GraphEdge

    edge = GraphEdge(source="paper-1", target="paper-2")
    assert edge.source == "paper-1"
    assert edge.target == "paper-2"


def test_graph_data():
    from app.schemas.graph import GraphData, GraphEdge, GraphNode

    data = GraphData(
        nodes=[
            GraphNode(id="1", title="P1", authors=[], year=2023,
                      citation_count=10, quality_score=None, in_library=True,
                      doi=None, abstract=None),
        ],
        edges=[GraphEdge(source="1", target="2")],
    )
    assert len(data.nodes) == 1
    assert len(data.edges) == 1


def test_graph_expansion():
    from app.schemas.graph import GraphEdge, GraphExpansion, GraphNode

    exp = GraphExpansion(
        new_nodes=[
            GraphNode(id="3", title="New Paper", authors=[], year=2022,
                      citation_count=5, quality_score=None, in_library=False,
                      doi=None, abstract=None),
        ],
        new_edges=[GraphEdge(source="1", target="3")],
    )
    assert len(exp.new_nodes) == 1
    assert exp.new_nodes[0].in_library is False


def test_graph_node_is_seed_default_false():
    from app.schemas.graph import GraphNode

    node = GraphNode(
        id="id1", title="A Paper", authors=[], year=None,
        citation_count=None, quality_score=None, in_library=False,
        doi=None, abstract=None,
    )
    assert node.is_seed is False


def test_graph_node_is_seed_true():
    from app.schemas.graph import GraphNode

    node = GraphNode(
        id="id1", title="Seed Paper", authors=[], year=2023,
        citation_count=10, quality_score=0.8, in_library=True,
        doi="10.1234/test", abstract=None, is_seed=True,
    )
    assert node.is_seed is True


def test_expand_node_request():
    from app.schemas.graph import ExpandNodeRequest

    req = ExpandNodeRequest(paper_id="550e8400-e29b-41d4-a716-446655440000")
    assert str(req.paper_id) == "550e8400-e29b-41d4-a716-446655440000"
