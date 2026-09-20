"""The evidence table.

One verbatim, code-accepted quote per row, always tied to the paper it was extracted
from and replaced wholesale on that paper's next analysis (``app.services.deep_analysis``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, select

from app.models.evidence import EVIDENCE_KINDS, EVIDENCE_ORIGINS, Evidence
from app.models.paper import Paper, SourceApi

VERSIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def test_model_declares_every_spec_column():
    columns = inspect(Evidence).columns
    names = set(columns.keys())
    assert names == {
        "id", "paper_id", "quote", "chunk_index", "section",
        "finding", "kind", "origin", "concepts", "prompt_version", "created_at",
    }


def test_evidence_kinds_and_origins_are_the_spec_values():
    assert EVIDENCE_KINDS == ("finding", "method", "sample", "limitation")
    assert EVIDENCE_ORIGINS == ("own", "reported")


def test_alembic_revision_chains_from_the_current_head():
    src = (VERSIONS_DIR / "k1a2b3c4d5e6_add_evidence_table.py").read_text(encoding="utf-8")
    assert 'revision: str = "k1a2b3c4d5e6"' in src
    assert 'down_revision: Union[str, None] = "j1a2b3c4d5e6"' in src
    assert "op.create_table(\n        \"evidence\"" in src
    assert "op.drop_table(\"evidence\")" in src


@pytest.mark.asyncio
async def test_evidence_row_round_trips_through_the_database(db_session):
    paper = Paper(title="A Paper", authors=[{"name": "Jane Smith"}], year=2020,
                  source_api=SourceApi.manual)
    db_session.add(paper)
    await db_session.flush()

    row = Evidence(
        paper_id=paper.id,
        quote="Tutoring improved outcomes for most students in the sample.",
        chunk_index=2,
        section="Results",
        finding="Tutoring improves student outcomes.",
        kind="finding",
        origin="own",
        concepts=["tutoring", "student outcomes"],
        prompt_version="sha256:deadbeef0000",
    )
    db_session.add(row)
    await db_session.commit()

    fetched = (
        await db_session.execute(select(Evidence).where(Evidence.paper_id == paper.id))
    ).scalar_one()
    assert fetched.quote == row.quote
    assert fetched.chunk_index == 2
    assert fetched.section == "Results"
    assert fetched.kind == "finding"
    assert fetched.origin == "own"
    assert fetched.concepts == ["tutoring", "student outcomes"]
    assert fetched.prompt_version == "sha256:deadbeef0000"
    assert fetched.created_at is not None


@pytest.mark.asyncio
async def test_deleting_a_paper_cascades_to_its_evidence_rows(db_session):
    paper = Paper(title="A Paper", authors=[], year=2020, source_api=SourceApi.manual)
    db_session.add(paper)
    await db_session.flush()
    db_session.add(Evidence(
        paper_id=paper.id, quote="q", chunk_index=0, finding="f",
        kind="finding", origin="own", concepts=[], prompt_version="sha256:x",
    ))
    await db_session.commit()

    await db_session.delete(paper)
    await db_session.commit()

    remaining = (await db_session.execute(select(Evidence))).scalars().all()
    assert remaining == []
