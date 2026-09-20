"""add journal_guidelines and compliance_check job types

Revision ID: j1a2b3c4d5e6
Revises: h1a2b3c4d5e6
Create Date: 2026-07-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "j1a2b3c4d5e6"
down_revision: Union[str, None] = "i1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("COMMIT")
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'journal_guidelines'")
    op.execute("ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'compliance_check'")
    op.execute("BEGIN")


def downgrade() -> None:
    # PostgreSQL cannot remove enum values; these changes are additive-only.
    pass
