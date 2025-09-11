"""enable pgvector extension

Revision ID: dcea459ec292
Revises: 483d076af940
Create Date: 2025-09-11 22:46:31.284790

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import pgvector


# revision identifiers, used by Alembic.
revision: str = "dcea459ec292"
down_revision: Union[str, Sequence[str], None] = "483d076af940"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP EXTENSION IF EXISTS vector")
