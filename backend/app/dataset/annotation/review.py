from __future__ import annotations

import hashlib
import json
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, ValidationError
from app.core.models import Annotation, AnnotationDraft, AnnotationReviewReceipt, ClassLabel, ImageItem
from app.core.schemas import AnnotationInput
from app.core.storage import Storage
from app.dataset.annotation.service import _validate_input, list_annotations, replace_annotations_in_transaction
from app.dataset.images import get_image
from app.dataset.service import get_dataset


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def image_hash(storage: Storage, image: ImageItem) -> str:
    path = storage.image_path(image.dataset_id, image.storage_name)
    if not path.is_file():
        raise ValidationError("image_file_missing", "The original image is missing.")
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def annotation_payload(session: Session, image_id: str) -> list[dict]:
    fields = ("type", "class_id", "bbox", "polygon", "obb", "source", "object_id", "mask_ref", "provenance_json")
    return [{key: value for key, value in item.model_dump(mode="json").items() if key in fields}
            for item in list_annotations(session, image_id)]


def current_content(session: Session, storage: Storage, image: ImageItem) -> dict:
    classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == image.dataset_id).order_by(ClassLabel.class_index)))
    return {
        "image_id": image.id,
        "image_hash": image_hash(storage, image),
        "width": image.width,
        "height": image.height,
        "task_type": get_dataset(session, image.dataset_id).task_type,
        "annotations": annotation_payload(session, image.id),
        "classes": [{"id": item.id, "index": item.class_index, "name": item.name} for item in classes],
        "rules": image.rules_json or {},
        "roi_mask_ref": image.roi_mask_ref,
        "ignore_mask_ref": image.ignore_mask_ref,
    }


def review_detail(session: Session, storage: Storage, image_id: str) -> dict:
    image = get_image(session, image_id)
    content = current_content(session, storage, image)
    fingerprint = json_fingerprint(content)
    receipt = session.scalar(select(AnnotationReviewReceipt).where(
        AnnotationReviewReceipt.image_id == image.id,
        AnnotationReviewReceipt.revision == image.annotation_revision,
        AnnotationReviewReceipt.fingerprint == fingerprint,
    ))
    current = receipt is not None and image.review_revision == image.annotation_revision and image.review_fingerprint == fingerprint
    draft = session.get(AnnotationDraft, image.id)
    return {
        **content,
        "revision": image.annotation_revision,
        "fingerprint": fingerprint,
        "review_status": image.review_status if current else "pending",
        "receipt": {"id": receipt.id, "actor": receipt.actor, "reason": receipt.reason,
                    "confirmed_empty": receipt.confirmed_empty, "created_at": receipt.created_at.isoformat()} if current else None,
        "draft": {"base_revision": draft.base_revision, "revision": draft.revision, **draft.payload_json} if draft else None,
        "issues": [],
    }


def _validate_regions(storage: Storage, image: ImageItem) -> None:
    from app.dataset.annotation.masks import MaskService

    masks = MaskService(storage)
    for ref in (image.roi_mask_ref, image.ignore_mask_ref):
        if ref is not None and masks.decode(ref).shape != (image.height, image.width):
            raise ValidationError("region_dimensions_mismatch", "ROI and ignore masks must use original image dimensions.")


def confirm_image(
    session: Session, storage: Storage, image_id: str, *, expected_revision: int,
    expected_fingerprint: str, confirmed_empty: bool = False, actor: str = "local-user",
    reason: str = "", decision: dict | None = None,
) -> dict:
    image = get_image(session, image_id)
    content = current_content(session, storage, image)
    fingerprint = json_fingerprint(content)
    if image.annotation_revision != expected_revision or fingerprint != expected_fingerprint:
        raise ConflictError("review_revision_conflict", "The image, rules or labels changed. Review the current revision.")
    if session.get(AnnotationDraft, image.id) is not None:
        raise ValidationError("annotation_draft_pending", "Apply and review the candidate draft before confirming this image.")
    if bool(content["annotations"]) == confirmed_empty:
        raise ValidationError("review_empty_mismatch", "Explicitly confirm an empty image, or confirm its objects.")
    if confirmed_empty and content["task_type"] == "classify":
        raise ValidationError("classification_label_required", "Classification confirmation requires one class.")
    _validate_regions(storage, image)
    from app.core.task_types import annotation_type_for

    expected_type = annotation_type_for(content["task_type"])
    if expected_type == "classify" and len(content["annotations"]) != 1:
        raise ValidationError("classification_label_required", "Classification confirmation requires exactly one class.")
    for raw in content["annotations"]:
        if raw["type"] != expected_type or raw["class_id"] not in {item["id"] for item in content["classes"]}:
            raise ValidationError("annotation_family_mismatch", "Annotation classes and geometry must match the dataset.")
        _validate_input(AnnotationInput.model_validate(raw), image.width, image.height, storage)
    receipt_id = "review_" + json_fingerprint([image.id, expected_revision, fingerprint])[:48]
    existing = session.get(AnnotationReviewReceipt, receipt_id)
    if existing is None:
        session.add(AnnotationReviewReceipt(
            id=receipt_id, image_id=image.id, revision=expected_revision, fingerprint=fingerprint,
            image_hash=content["image_hash"], confirmed_empty=confirmed_empty,
            actor=actor.strip()[:128] or "local-user", reason=reason, decision_json=decision or {},
        ))
    changed = session.execute(update(ImageItem).where(
        ImageItem.id == image.id, ImageItem.annotation_revision == expected_revision,
    ).values(review_status="confirmed_empty" if confirmed_empty else "confirmed",
             review_revision=expected_revision, review_fingerprint=fingerprint))
    if changed.rowcount != 1:
        raise ConflictError("review_revision_conflict", "The annotations changed during confirmation.")
    session.flush()
    return {"receipt_id": receipt_id, "image_id": image.id, "revision": expected_revision,
            "fingerprint": fingerprint, "confirmed_empty": confirmed_empty}


def confirm_workflow_review(session: Session, storage: Storage, context: dict, decision: dict) -> dict:
    payload = context.get("payload") or context
    image = get_image(session, context.get("image_id") or payload.get("image_id"))
    base_revision = context.get("base_revision", payload.get("base_revision"))
    expected_hash = context.get("image_hash", payload.get("image_hash"))
    if base_revision is None or expected_hash is None:
        raise ValidationError("review_context_incomplete", "Review requires the frozen image hash and starting label revision.")
    if image.annotation_revision != base_revision or image_hash(storage, image) != expected_hash:
        raise ConflictError("review_revision_conflict", "The source image or labels changed after this workflow started.")
    issues = payload.get("issues") or []
    resolutions = decision.get("semantic_resolutions") or {}
    for issue in issues:
        if issue.get("blocking"):
            raise ValidationError("review_blocking_issue", "Resolve blocking geometry or schema issues before confirmation.")
        issue_id = issue.get("id")
        if issue_id and not resolutions.get(issue_id):
            raise ValidationError("review_resolution_required", "Record a decision for each remaining review issue.")
    from app.dataset.annotation.masks import MaskService

    annotations: list[AnnotationInput] = []
    for raw in payload.get("annotations", []):
        raw = dict(raw)
        if isinstance(raw.get("mask"), dict):
            raw["mask_ref"] = MaskService(storage).write(raw.pop("mask")).mask_ref
        annotations.append(AnnotationInput.model_validate(raw))
    replace_annotations_in_transaction(session, image.id, annotations, storage=storage, expected_revision=base_revision)
    image.roi_mask_ref = payload.get("roi_mask_ref")
    image.ignore_mask_ref = payload.get("ignore_mask_ref")
    for key in ("roi", "ignore"):
        if isinstance(payload.get(key + "_mask"), dict):
            setattr(image, key + "_mask_ref", MaskService(storage).write(payload[key + "_mask"]).mask_ref)
    image.rules_json = payload.get("rules") or context.get("rules") or {}
    session.flush()
    content = current_content(session, storage, image)
    return confirm_image(session, storage, image.id, expected_revision=image.annotation_revision,
                         expected_fingerprint=json_fingerprint(content),
                         confirmed_empty=bool(decision.get("confirmed_empty")), actor=decision.get("actor") or "local-user",
                         reason=decision.get("reason") or "", decision=decision)


def require_confirmed(session: Session, storage: Storage, dataset_id: str, image_ids: list[str] | None = None) -> list[dict]:
    get_dataset(session, dataset_id)
    query = select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.split, ImageItem.id)
    if image_ids is not None:
        if not image_ids or len(set(image_ids)) != len(image_ids):
            raise ValidationError("snapshot_range_invalid", "Choose a nonempty range of distinct images.")
        query = query.where(ImageItem.id.in_(image_ids))
    images = list(session.scalars(query))
    if not images or image_ids is not None and len(images) != len(image_ids):
        raise ValidationError("snapshot_range_invalid", "The selected range contains missing images or belongs to another dataset.")
    details = [review_detail(session, storage, image.id) for image in images]
    from app.dataset.annotation.progress import image_progress

    progress = image_progress(session, images)
    pending = [item["image_id"] for item in details if item["receipt"] is None or progress[item["image_id"]]["has_draft"]
               or progress[item["image_id"]]["workflow_pending"]]
    if pending:
        raise ValidationError("annotation_review_required", "All selected images require current human confirmation.",
                              details={"image_ids": pending, "confirmed": len(details) - len(pending), "total": len(details),
                                       "workflow_image_ids": [image.id for image in images if progress[image.id]["workflow_pending"]],
                                       "draft_image_ids": [image.id for image in images if progress[image.id]["has_draft"]]})
    return details


def review_summary(session: Session, storage: Storage, dataset_id: str) -> dict:
    get_dataset(session, dataset_id)
    from app.dataset.annotation.progress import WORKFLOW_FIELDS, image_progress, progress_summary

    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.id)))
    progress = image_progress(session, images)
    items = []
    for image in images:
        detail = review_detail(session, storage, image.id)
        items.append({"image_id": image.id, "file_name": image.file_name, "revision": image.annotation_revision,
                      "fingerprint": detail["fingerprint"], "review_status": detail["review_status"],
                      "annotation_count": len(detail["annotations"]),
                      "contains_auto": any(item["source"] == "auto" for item in detail["annotations"]),
                      "has_draft": progress[image.id]["has_draft"],
                      **{key: progress[image.id][key] for key in WORKFLOW_FIELDS}})
    counts = {key: sum(item["review_status"] == key for item in items) for key in ("confirmed", "confirmed_empty", "pending")}
    return {"dataset_id": dataset_id, "total": len(items), **counts, "items": items,
            **progress_summary(list(progress.values()))}


def adopt_historical(session: Session, storage: Storage, dataset_id: str, items: list[dict], *, actor: str, reason: str) -> list[dict]:
    if not items:
        raise ValidationError("review_range_empty", "Select the reviewed images to adopt.")
    results = []
    for item in items:
        image = get_image(session, item["image_id"])
        if image.dataset_id != dataset_id:
            raise ValidationError("review_dataset_mismatch", "Selected images must belong to this dataset.")
        labels = annotation_payload(session, image.id)
        if not labels or any(label["source"] == "auto" for label in labels) or session.get(AnnotationDraft, image.id):
            raise ValidationError("historical_review_required", "Automatic drafts and empty images require individual image review.")
        results.append(confirm_image(session, storage, image.id,
            expected_revision=item["expected_revision"], expected_fingerprint=item["expected_fingerprint"], actor=actor, reason=reason))
    return results


def invalidate_dataset_reviews(session: Session, dataset_id: str) -> None:
    session.execute(update(ImageItem).where(ImageItem.dataset_id == dataset_id).values(
        annotation_revision=ImageItem.annotation_revision + 1, review_status="pending", review_revision=None, review_fingerprint=None))
