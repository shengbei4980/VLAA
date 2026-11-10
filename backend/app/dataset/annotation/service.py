from __future__ import annotations

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, joinedload

from app.core.errors import ConflictError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, AnnotationDraft, ClassLabel, ImageItem
from app.core.schemas import AnnotationInput, AnnotationResponse, BBox, OBB
from app.core.task_types import annotation_type_for
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_bbox, validate_obb, validate_polygon
from app.dataset.images import get_image
from app.dataset.service import get_dataset


def _response(annotation: Annotation) -> AnnotationResponse:
    class_label = annotation.class_label
    bbox = None
    if annotation.type == "bbox":
        bbox = BBox(x=annotation.x or 0, y=annotation.y or 0, width=annotation.width or 0, height=annotation.height or 0)
    polygon = [tuple(point) for point in annotation.polygon] if annotation.polygon else None
    obb = OBB(**annotation.obb) if annotation.type == "obb" and annotation.obb else None
    return AnnotationResponse(
        id=annotation.id,
        image_id=annotation.image_id,
        dataset_id=annotation.dataset_id,
        class_id=annotation.class_id,
        class_index=class_label.class_index,
        label=class_label.name,
        color=class_label.color,
        type=annotation.type,
        bbox=bbox,
        polygon=polygon,
        obb=obb,
        source=annotation.source,
        object_id=annotation.object_id,
        mask_ref=annotation.mask_ref,
        provenance_json=annotation.provenance_json or {},
        created_at=annotation.created_at,
        updated_at=annotation.updated_at,
    )


def list_annotations(session: Session, image_id: str) -> list[AnnotationResponse]:
    get_image(session, image_id)
    query = (
        select(Annotation)
        .options(joinedload(Annotation.class_label))
        .where(Annotation.image_id == image_id)
        .order_by(Annotation.created_at, Annotation.id)
    )
    return [_response(item) for item in session.scalars(query)]


def _validate_input(item: AnnotationInput, image_width: int, image_height: int, storage: Storage | None = None) -> None:
    if item.mask_ref is not None:
        if item.type != "polygon" or storage is None:
            raise ValidationError("mask_storage_required", "Masks require a segment annotation and managed storage.")
        from app.dataset.annotation.masks import MaskService

        mask = MaskService(storage).decode(item.mask_ref)
        if mask.shape != (image_height, image_width) or not mask.any():
            raise ValidationError("mask_geometry_invalid", "Object masks must match the original image and contain visible pixels.")
    if item.type == "bbox" and item.bbox is not None:
        validate_bbox(item.bbox, image_width, image_height)
    elif item.type == "polygon" and item.polygon is not None:
        validate_polygon(item.polygon, image_width, image_height)
    elif item.type == "obb" and item.obb is not None:
        validate_obb(item.obb, image_width, image_height)


def replace_annotations(
    session: Session,
    image_id: str,
    items: list[AnnotationInput],
    *,
    storage: Storage | None = None,
    expected_revision: int | None = None,
    regions: dict | None = None,
) -> list[AnnotationResponse]:
    replace_annotations_in_transaction(session, image_id, items, storage=storage, expected_revision=expected_revision, regions=regions)
    session.commit()
    return list_annotations(session, image_id)


def replace_annotations_in_transaction(
    session: Session,
    image_id: str,
    items: list[AnnotationInput],
    *,
    storage: Storage | None = None,
    expected_revision: int | None = None,
    regions: dict | None = None,
) -> None:
    image = get_image(session, image_id)
    dataset = get_dataset(session, image.dataset_id)
    expected_type = annotation_type_for(dataset.task_type)
    if expected_type == "classify" and len(items) > 1:
        raise ValidationError("classify_multiple", "Classification datasets allow exactly one class per image.")
    class_ids = set(
        session.scalars(select(ClassLabel.id).where(ClassLabel.dataset_id == dataset.id))
    )
    revision = image.annotation_revision or 0
    if regions:
        if storage is None:
            raise ValidationError("mask_storage_required", "Region edits require managed mask storage.")
        from app.dataset.annotation.masks import MaskService

        for key, ref in regions.items():
            if key not in {"roi_mask_ref", "ignore_mask_ref"}:
                raise ValidationError("region_field_invalid", "Only ROI and ignore masks may be changed here.")
            if ref is not None and MaskService(storage).decode(ref).shape != (image.height, image.width):
                raise ValidationError("region_dimensions_mismatch", "Supervision masks must match the original image.")
    if expected_revision is not None and revision != expected_revision:
        raise ConflictError("annotation_revision_conflict", "Annotations changed. Reload the current revision before saving.")
    existing_ids = dict(session.execute(select(Annotation.object_id, Annotation.id).where(Annotation.image_id == image.id)).all())
    object_ids: set[str] = set()
    created: list[Annotation] = []
    for item in items:
        if item.type != expected_type:
            raise ValidationError(
                "annotation_family_mismatch",
                f"{dataset.task_type} datasets require {expected_type} annotations.",
            )
        if item.class_id not in class_ids:
            raise ValidationError("class_not_in_dataset", "Annotation class does not belong to this dataset.")
        _validate_input(item, image.width, image.height, storage)
        object_id = item.object_id or new_id("object")
        if object_id in object_ids:
            raise ValidationError("duplicate_object_id", "Object IDs must be unique within an image.")
        object_ids.add(object_id)
        mask_ref = item.mask_ref
        if item.type == "polygon" and mask_ref is None and storage is not None:
            from app.dataset.annotation.masks import MaskService, polygon_mask

            mask_ref = MaskService(storage).write(polygon_mask(item.polygon, image.width, image.height)).mask_ref
        created.append(
            Annotation(
                id=existing_ids.get(object_id) or new_id("ann"),
                image_id=image.id,
                dataset_id=dataset.id,
                class_id=item.class_id,
                type=item.type,
                x=item.bbox.x if item.bbox else None,
                y=item.bbox.y if item.bbox else None,
                width=item.bbox.width if item.bbox else None,
                height=item.bbox.height if item.bbox else None,
                polygon=[list(point) for point in item.polygon] if item.polygon else None,
                obb=item.obb.model_dump() if item.obb else None,
                source=item.source,
                object_id=object_id,
                mask_ref=mask_ref,
                provenance_json=item.provenance_json,
            )
        )
    changed = session.execute(
        update(ImageItem).where(ImageItem.id == image.id, ImageItem.annotation_revision == revision).values(
            annotation_revision=revision + 1, review_status="pending", review_revision=None, review_fingerprint=None,
        )
    )
    if changed.rowcount != 1:
        raise ConflictError("annotation_revision_conflict", "Annotations changed during this save.")
    for key, ref in (regions or {}).items():
        setattr(image, key, ref)
    session.execute(delete(Annotation).where(Annotation.image_id == image.id), execution_options={"synchronize_session": False})
    session.execute(delete(AnnotationDraft).where(AnnotationDraft.image_id == image.id))
    session.add_all(created)
    image.status = "annotated" if created else "unannotated"
    session.flush()
