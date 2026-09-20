"""add paper_upload refine and upload enum values

Revision ID: 588c313b5020
Revises: 451270d9394a
Create Date: 2026-03-23 05:14:37.402962

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '588c313b5020'
down_revision: Union[str, None] = '451270d9394a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Fix: smart_search and deep_analysis were missing from previous migrations
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'smart_search'")
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'deep_analysis'")
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'paper_upload'")
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'refine'")
    op.execute("ALTER TYPE sourceapi ADD VALUE IF NOT EXISTS 'upload'")


def downgrade() -> None:
    # PostgreSQL does not support removing enum values easily.
    # These are additive-only changes.
    pass
