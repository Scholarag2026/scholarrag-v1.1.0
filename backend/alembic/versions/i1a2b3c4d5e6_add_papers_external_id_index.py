"""Add index on papers.external_id

Revision ID: i1a2b3c4d5e6
Revises: h1a2b3c4d5e6
Create Date: 2026-07-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "i1a2b3c4d5e6"
down_revision: Union[str, None] = "h1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Citation-graph builds resolve references with
    # `WHERE doi IN (...) OR external_id IN (...)`; without this index the
    # external_id half is a sequential scan on every batch.
    op.create_index(
        "ix_papers_external_id", "papers", ["external_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_papers_external_id", table_name="papers")
