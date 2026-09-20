"""add wos_collection and wos_categories to papers"""

from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "588c313b5020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("wos_collection", sa.String(10), nullable=True))
    op.add_column("papers", sa.Column("wos_categories", sa.String(1000), nullable=True))


def downgrade() -> None:
    op.drop_column("papers", "wos_categories")
    op.drop_column("papers", "wos_collection")
