"""papers.external_id must be indexed (issue PAPER-LOOKUP-N-PLUS-1).

get_or_create_papers_bulk resolves a whole reference list with
`WHERE doi IN (...) OR external_id IN (...)`; doi is already unique+indexed, external_id
was a sequential scan on a 5k-row table that grows with every expansion.
"""

from pathlib import Path

from sqlalchemy import inspect

from app.models.paper import Paper

VERSIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def test_model_declares_the_index():
    column = inspect(Paper).columns["external_id"]
    assert column.index is True


def test_metadata_creates_the_named_index():
    names = {index.name for index in Paper.__table__.indexes}
    assert "ix_papers_external_id" in names


def test_alembic_revision_chains_from_the_current_head():
    src = (VERSIONS_DIR / "i1a2b3c4d5e6_add_papers_external_id_index.py").read_text(
        encoding="utf-8"
    )
    assert 'revision: str = "i1a2b3c4d5e6"' in src
    assert 'down_revision: Union[str, None] = "h1a2b3c4d5e6"' in src
    assert "ix_papers_external_id" in src
    assert "op.create_index" in src
    assert "op.drop_index" in src
