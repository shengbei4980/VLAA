"""Keep workflow history in a recoverable trash view."""

from alembic import op
import sqlalchemy as sa

from app.core.time import UTCDateTime

revision = "0021_workflow_run_trash"
down_revision = "0020_visual_workflows"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workflow_runs", sa.Column("deleted_at", UTCDateTime(), nullable=True))
    op.create_index("ix_workflow_runs_deleted_at", "workflow_runs", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_workflow_runs_deleted_at", table_name="workflow_runs")
    op.drop_column("workflow_runs", "deleted_at")
