from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.core.models import TimestampMixin
from app.core.time import UTCDateTime


class WorkflowTaskProfile(Base, TimestampMixin):
    __tablename__ = "workflow_task_profiles"
    __table_args__ = (UniqueConstraint("key", "version", name="uq_workflow_profile_version"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(255))
    config_json: Mapped[dict] = mapped_column(JSON)


class WorkflowTemplate(Base, TimestampMixin):
    __tablename__ = "workflow_templates"
    __table_args__ = (UniqueConstraint("key", "version", name="uq_workflow_template_version"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(255))
    nodes_json: Mapped[list] = mapped_column(JSON)


class WorkflowRun(Base, TimestampMixin):
    __tablename__ = "workflow_runs"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_workflow_run_idempotency"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    fingerprint: Mapped[str] = mapped_column(String(64))
    template_id: Mapped[str] = mapped_column(ForeignKey("workflow_templates.id"))
    profile_id: Mapped[str] = mapped_column(ForeignKey("workflow_task_profiles.id"))
    template_snapshot: Mapped[dict] = mapped_column(JSON)
    profile_snapshot: Mapped[dict] = mapped_column(JSON)
    context_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_run_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True, index=True)


class WorkflowItem(Base, TimestampMixin):
    __tablename__ = "workflow_items"
    __table_args__ = (UniqueConstraint("run_id", "subject_ref", name="uq_workflow_item_subject"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    subject_ref: Mapped[str] = mapped_column(String(128))
    context_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    disposition: Mapped[str | None] = mapped_column(String(32), nullable=True)
    receipt_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkflowAttempt(Base, TimestampMixin):
    __tablename__ = "workflow_attempts"
    __table_args__ = (
        UniqueConstraint("run_id", "unit_key", "node_id", "attempt", name="uq_workflow_attempt"),
        Index("ix_workflow_attempt_run_status", "run_id", "status"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_items.id", ondelete="CASCADE"), nullable=True)
    unit_key: Mapped[str] = mapped_column(String(64))
    node_id: Mapped[str] = mapped_column(String(64))
    node_type: Mapped[str] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer)
    generation: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    input_json: Mapped[dict] = mapped_column(JSON)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reserved_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    task_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class WorkflowArtifact(Base, TimestampMixin):
    __tablename__ = "workflow_artifacts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_items.id", ondelete="CASCADE"), nullable=True)
    attempt_id: Mapped[str | None] = mapped_column(ForeignKey("workflow_attempts.id", ondelete="SET NULL"), nullable=True)
    kind: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_name: Mapped[str] = mapped_column(String(128))
    payload_json: Mapped[dict] = mapped_column(JSON)
    parent_ids_json: Mapped[list] = mapped_column(JSON, default=list)


class WorkflowReview(Base, TimestampMixin):
    __tablename__ = "workflow_reviews"
    __table_args__ = (UniqueConstraint("attempt_id", name="uq_workflow_review_attempt"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[str] = mapped_column(ForeignKey("workflow_items.id", ondelete="CASCADE"), index=True)
    attempt_id: Mapped[str] = mapped_column(ForeignKey("workflow_attempts.id", ondelete="CASCADE"))
    artifact_id: Mapped[str] = mapped_column(ForeignKey("workflow_artifacts.id"))
    revision: Mapped[int] = mapped_column(Integer, default=1)
    base_revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    receipt_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class WorkflowEvent(Base, TimestampMixin):
    __tablename__ = "workflow_events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
