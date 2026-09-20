"""Add Harvard, IEEE, Vancouver to citation_style enum

Revision ID: g1a2b3c4d5e6
Revises: f8a2b3c4d5e6
Create Date: 2026-03-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "g1a2b3c4d5e6"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction in PostgreSQL
    op.execute("COMMIT")
    op.execute("ALTER TYPE citationstyle ADD VALUE IF NOT EXISTS 'Harvard'")
    op.execute("ALTER TYPE citationstyle ADD VALUE IF NOT EXISTS 'IEEE'")
    op.execute("ALTER TYPE citationstyle ADD VALUE IF NOT EXISTS 'Vancouver'")
    op.execute("BEGIN")


def downgrade() -> None:
    # PostgreSQL does not support removing values from enums easily.
    # This is a no-op for downgrade.
    pass
