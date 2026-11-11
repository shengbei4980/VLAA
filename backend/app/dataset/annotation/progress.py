"""Read current workflow drafts alongside applied dataset labels, without copying them."""
from __future__ import annotations

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from app.core.models import Annotation, AnnotationDraft, ImageItem
from app.workflow.models import WorkflowArtifact, WorkflowItem, WorkflowReview, WorkflowRun


def candidate_count(payload: dict) -> int:
    objects = payload.get("objects", payload.get("annotations", []))
    return sum(isinstance(item, dict) and item.get("decision") != "rejected" for item in objects or [])


def image_progress(session: Session, images: list[ImageItem]) -> dict[str, dict]:
    """Use a bounded number of queries for a whole dataset/list/page.

    A formal receipt remains recorded when a new draft is created. Such images
    are pending again for new snapshots, while explicitly chosen old snapshots
    retain their immutable labels. Cancelled/deleted and stale runs do not block.
    """
    if not images:
        return {}
    ids = [image.id for image in images]
    counts = dict(session.execute(select(Annotation.image_id, func.count()).where(
        Annotation.image_id.in_(ids)).group_by(Annotation.image_id)).all())
    drafts = {draft.image_id: draft for draft in session.scalars(select(AnnotationDraft).where(
        AnnotationDraft.image_id.in_(ids)))}
    # Prefer an active run to a cancelled one. Within that range the latest run
    # is the editing context, independent of how many immutable artifacts it has.
    matches_revision = or_(WorkflowItem.context_json["base_revision"].as_integer() == ImageItem.annotation_revision,
                           and_(WorkflowItem.confirmed.is_(True),
                                WorkflowItem.receipt_json["revision"].as_integer() == ImageItem.annotation_revision))
    rows = session.execute(select(WorkflowItem, WorkflowRun).join(WorkflowRun).join(
        ImageItem, ImageItem.id == WorkflowItem.subject_ref).where(
        WorkflowItem.subject_ref.in_(ids), WorkflowRun.deleted_at.is_(None),
    ).order_by(case((or_(WorkflowRun.cancel_requested.is_(True), WorkflowRun.status == "cancelled"), 1), else_=0),
               case((matches_revision, 0), else_=1),
               WorkflowRun.created_at.desc(), WorkflowItem.created_at.desc(), WorkflowItem.id.desc())).all()
    selected = {}
    for item, run in rows:
        if item.subject_ref in selected:
            continue
        nodes = run.template_snapshot.get("nodes", [])
        # Continuing a confirmed annotation opens a manual revision without
        # calling a model. It still supersedes the applied receipt until review.
        annotation_edit_of = run.context_json.get("annotation_edit_of")
        manual_revision = isinstance(annotation_edit_of, str) and bool(annotation_edit_of.strip())
        if manual_revision or any(node.get("type") in {"vision_propose", "managed_inference"} for node in nodes):
            selected[item.subject_ref] = (item, run)
    item_ids = [item.id for item, _ in selected.values()]
    artifacts, reviews = {}, {}
    if item_ids:
        ranked = select(WorkflowArtifact.id, func.row_number().over(
            partition_by=WorkflowArtifact.item_id,
            order_by=(WorkflowArtifact.created_at.desc(), WorkflowArtifact.id.desc()),
        ).label("position")).where(WorkflowArtifact.item_id.in_(item_ids), WorkflowArtifact.kind.in_(
            ("proposals", "review_proposals", "confirmed_annotations"))).subquery()
        artifacts = {row.item_id: row for row in session.scalars(select(WorkflowArtifact).join(
            ranked, WorkflowArtifact.id == ranked.c.id).where(ranked.c.position == 1))}
        ranked_reviews = select(WorkflowReview.id, func.row_number().over(
            partition_by=WorkflowReview.item_id,
            order_by=(WorkflowReview.created_at.desc(), WorkflowReview.id.desc()),
        ).label("position")).where(WorkflowReview.item_id.in_(item_ids)).subquery()
        reviews = {row.item_id: row for row in session.scalars(select(WorkflowReview).join(
            ranked_reviews, WorkflowReview.id == ranked_reviews.c.id).where(ranked_reviews.c.position == 1))}
    result = {}
    for image in images:
        draft = drafts.get(image.id)
        current_draft = draft is not None and draft.base_revision == image.annotation_revision
        formal_confirmed = (image.review_status in {"confirmed", "confirmed_empty"}
                            and image.review_revision == image.annotation_revision and bool(image.review_fingerprint))
        value = {"annotation_count": counts.get(image.id, 0), "has_draft": current_draft,
                 "draft_annotation_count": candidate_count(draft.payload_json) if current_draft else 0,
                 "workflow_run_id": None, "workflow_review_id": None, "workflow_annotation_count": 0,
                 "workflow_status": None, "workflow_stale": False, "workflow_generated": False,
                 "workflow_pending": False, "formal_confirmed": formal_confirmed, "workflow_updated_at": None}
        if image.id in selected:
            item, run = selected[image.id]
            review = reviews.get(item.id)
            artifact = artifacts.get(item.id)
            receipt = review.receipt_json if review else item.receipt_json
            approved_current = (item.confirmed and formal_confirmed and receipt is not None
                                and receipt.get("revision") == image.annotation_revision
                                and receipt.get("fingerprint") == image.review_fingerprint)
            stale = not approved_current and item.context_json.get("base_revision") != image.annotation_revision
            closed = run.cancel_requested or run.status in {"cancelled", "completed"}
            pending = not closed and not stale and not item.confirmed and item.status != "excluded"
            status = "processing"
            if stale:
                status = "stale"
            elif approved_current:
                status = "confirmed"
            elif item.status == "excluded":
                status = "excluded"
            elif closed:
                status = "cancelled"
            elif artifact is not None:
                status = "draft"
            elif item.status == "failed":
                status = "failed"
            value.update(workflow_run_id=run.id, workflow_review_id=review.id if review else None,
                         workflow_annotation_count=candidate_count(artifact.payload_json) if artifact else 0,
                         workflow_status=status, workflow_stale=stale,
                         workflow_updated_at=review.updated_at if review else run.created_at,
                         workflow_generated=artifact is not None and not stale,
                         workflow_pending=pending,
                         has_draft=current_draft or pending and artifact is not None)
        value["confirmed"] = formal_confirmed and not current_draft and not value["workflow_pending"]
        value["confirmed_empty"] = value["confirmed"] and image.review_status == "confirmed_empty"
        value["generated"] = bool(value["annotation_count"] or current_draft or value["workflow_generated"]
                                  or value["confirmed"])
        # Processing has no reviewable result yet. Existing labels being replaced
        # still count as pending while their new workflow is in progress.
        value["pending_review"] = value["generated"] and not value["confirmed"]
        result[image.id] = value
    return result


WORKFLOW_FIELDS = ("workflow_run_id", "workflow_review_id", "workflow_annotation_count", "workflow_status", "workflow_stale")


def progress_summary(values: list[dict]) -> dict:
    workflows = [value for value in values if value.get("workflow_run_id")]
    workflows.sort(key=lambda value: (bool(value["workflow_pending"] and value["workflow_review_id"]),
                                     value["workflow_updated_at"]), reverse=True)
    latest = workflows[0] if workflows else {}
    return {"generated_image_count": sum(value["generated"] for value in values),
            "pending_review_image_count": sum(value["pending_review"] for value in values),
            "confirmed_image_count": sum(value["confirmed"] for value in values),
            "confirmed_empty_image_count": sum(value["confirmed_empty"] for value in values),
            "workflow_image_count": sum(value["workflow_generated"] for value in values),
            "latest_workflow_run_id": latest.get("workflow_run_id"),
            "latest_workflow_review_id": latest.get("workflow_review_id")}


def dataset_progress(session: Session, dataset_ids: list[str]) -> dict[str, dict]:
    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id.in_(dataset_ids)).order_by(ImageItem.id)))
    by_image = image_progress(session, images)
    grouped = {dataset_id: [] for dataset_id in dataset_ids}
    for image in images:
        grouped[image.dataset_id].append(by_image[image.id])
    return {dataset_id: progress_summary(values) for dataset_id, values in grouped.items()}
