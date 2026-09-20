"""add user role

Revision ID: 53a570555365
Revises: 23015df767f4
Create Date: 2026-03-21 08:59:56.595206

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '53a570555365'
down_revision: Union[str, None] = '23015df767f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    userrole = sa.Enum('user', 'admin', name='userrole')
    userrole.create(op.get_bind(), checkfirst=True)
    op.add_column(
        'users',
        sa.Column('role', userrole, nullable=False, server_default='user'),
    )


def downgrade() -> None:
    op.drop_column('users', 'role')
    sa.Enum(name='userrole').drop(op.get_bind(), checkfirst=True)
