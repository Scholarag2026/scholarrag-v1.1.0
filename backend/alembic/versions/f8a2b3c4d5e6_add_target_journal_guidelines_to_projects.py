"""add target_journal_guidelines to projects

Revision ID: f8a2b3c4d5e6
Revises: e47c10652a7a
Create Date: 2026-03-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'f8a2b3c4d5e6'
down_revision: Union[str, None] = 'e47c10652a7a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('target_journal_guidelines', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'target_journal_guidelines')
