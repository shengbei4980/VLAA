"""Bounded LabelMe ZIP ingestion with explicit taxonomy mapping."""
from __future__ import annotations

import base64
import binascii
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import PurePosixPath
import re

import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session
from pydantic import ValidationError as SchemaValidationError

from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.schemas import BBox, ClassLabelCreate, DatasetCreate
from app.core.storage import SUPPORTED_IMAGE_SUFFIXES, Storage
from app.dataset.annotation.geometry import validate_bbox, validate_polygon
from app.dataset.annotation.masks import MaskService, decode_mask, mask_bbox, polygon_mask
from app.dataset.exchange.yolo import YoloArchiveLimits, _archive_entries, _read_archive_member
from app.dataset.service import refresh_dataset_counts

_MASK_SIDECAR = re.compile(r"masks/([a-f0-9]{64})\.json\Z")
_MAX_MASK_BYTES = 20 * 1024 * 1024


@dataclass
class LabelMeImage:
    json_file: str
    original_json: bytes
    image_file: str
    content: bytes
    width: int
    height: int
    split: str
    source_group_id: str | None
    shapes: list[dict]
    flags: dict
    regions: dict
    rules: dict
    source_metadata: dict


def _validate_workflow_manifest(document: dict, entries: dict) -> None:
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
            or document.get("coordinate_space") != "image_pixel"
            or not isinstance(document.get("task_type"), str) or document["task_type"] not in {"detect", "segment"}
            or not isinstance(document.get("dataset_id"), str)
            or not isinstance(document.get("dataset_name"), str)
            or not isinstance(document.get("rules"), dict)
            or not isinstance(document.get("classes"), list)
            or not isinstance(document.get("images"), list)):
        raise ValidationError("labelme_manifest_invalid", "Workflow manifest structure is invalid.")
    for item in document["classes"]:
        if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                or not isinstance(item.get("name"), str) or type(item.get("index")) is not int):
            raise ValidationError("labelme_manifest_invalid", "Workflow manifest classes are invalid.")
    for image in document["images"]:
        if (not isinstance(image, dict) or not isinstance(image.get("id"), str)
                or not isinstance(image.get("path"), str) or not isinstance(image.get("split"), str)
                or image["split"] not in {"train", "val", "test"}
                or type(image.get("width")) is not int or type(image.get("height")) is not int
                or image["width"] <= 0 or image["height"] <= 0
                or not isinstance(image.get("annotations"), list)):
            raise ValidationError("labelme_manifest_invalid", "Workflow manifest images are invalid.")
        references = [image.get("roi_mask_ref"), image.get("ignore_mask_ref")]
        for annotation in image["annotations"]:
            if (not isinstance(annotation, dict) or not isinstance(annotation.get("type"), str)
                    or annotation["type"] not in {"bbox", "polygon"}
                    or not isinstance(annotation.get("class_id"), str)):
                raise ValidationError("labelme_manifest_invalid", "Workflow manifest annotations are invalid.")
            references.append(annotation.get("mask_ref"))
        for reference in references:
            if reference is not None and (not isinstance(reference, str)
                    or not _MASK_SIDECAR.fullmatch(reference) or reference not in entries):
                raise ValidationError("labelme_manifest_invalid", "Workflow manifest mask sidecars are missing or invalid.")


def _parse(payload: bytes, default_split: str, limits: YoloArchiveLimits) -> list[LabelMeImage]:
    if default_split not in {"train", "val", "test"}:
        raise ValidationError("labelme_split_invalid", "Choose train, val or test as the default split.")
    archive, entries = _archive_entries(payload, limits)
    parsed = []
    try:
        json_names = sorted(name for name in entries if PurePosixPath(name).suffix.lower() == ".json")
        for name in json_names:
            json_limit = 32 * 1024 * 1024 if _MASK_SIDECAR.fullmatch(name) else 20 * 1024 * 1024
            raw = _read_archive_member(archive, entries[name], min(limits.max_member_bytes, json_limit))
            try:
                document = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise ValidationError("labelme_json_invalid", f"LabelMe JSON is invalid: {name}.") from exc
            if name == "workflow_manifest.json" and isinstance(document, dict) and "shapes" not in document:
                _validate_workflow_manifest(document, entries)
                continue
            sidecar = _MASK_SIDECAR.fullmatch(name)
            if sidecar:
                if hashlib.sha256(raw).hexdigest() != sidecar.group(1):
                    raise ValidationError("labelme_sidecar_integrity", "A workflow mask sidecar failed its content hash.")
                decode_mask(document)
                continue
            if not isinstance(document, dict) or not isinstance(document.get("shapes"), list):
                raise ValidationError("labelme_shapes_missing", f"LabelMe JSON requires a shapes array: {name}.")
            if len(document["shapes"]) > 10000:
                raise ValidationError("labelme_too_many_shapes", "A LabelMe image exceeds the 10000-shape limit.")
            image_name = str(document.get("imagePath") or PurePosixPath(name).with_suffix(".png").name).replace("\\", "/")
            image_basename = PurePosixPath(image_name).name
            if PurePosixPath(image_basename).suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                raise ValidationError("labelme_image_format", "LabelMe image must use a supported RGB image format.")
            relative = str(PurePosixPath(name).parent / image_name)
            matches = [candidate for candidate in entries if PurePosixPath(candidate).name == image_basename]
            selected = relative if relative in entries else image_name if image_name in entries else matches[0] if len(matches) == 1 else None
            if selected:
                content = _read_archive_member(archive, entries[selected], limits.max_member_bytes)
            elif isinstance(document.get("imageData"), str):
                try:
                    content = base64.b64decode(document["imageData"], validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValidationError("labelme_image_data_invalid", "LabelMe embedded imageData is not valid base64.") from exc
                if len(content) > limits.max_member_bytes:
                    raise ValidationError("labelme_image_too_large", "Embedded LabelMe image exceeds the size limit.")
            else:
                raise ValidationError("labelme_image_missing", f"LabelMe image is missing or ambiguous: {image_basename}.")
            try:
                with Image.open(io.BytesIO(content)) as image:
                    width, height = image.size
                    if width * height > 100_000_000:
                        raise ValidationError("labelme_image_too_large", "LabelMe image exceeds the supported pixel count.")
                    image.verify()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValidationError("labelme_image_invalid", f"LabelMe image cannot be decoded: {image_basename}.") from exc
            if document.get("imageWidth", width) != width or document.get("imageHeight", height) != height:
                raise ValidationError("labelme_dimensions_mismatch", "LabelMe dimensions differ from the source image.")
            flags = document.get("flags") if isinstance(document.get("flags"), dict) else {}
            split = document.get("split") or next((part for part in PurePosixPath(name).parts if part in {"train", "val", "test"}), default_split)
            if split not in {"train", "val", "test"}:
                raise ValidationError("labelme_split_invalid", "LabelMe split metadata is invalid.")
            source_group = document.get("source_group_id") or flags.get("source_group_id")
            if source_group is not None and (not isinstance(source_group, str) or len(source_group) > 64):
                raise ValidationError("labelme_group_invalid", "source_group_id must be a string of at most 64 characters.")
            regions = {}
            for key in ("roi", "ignore"):
                if document.get("starter_" + key) is not None:
                    rle = document["starter_" + key]
                    if decode_mask(rle).shape != (height, width):
                        raise ValidationError("region_dimensions_mismatch", "Imported ROI/ignore masks must match the image.")
                    regions[key] = rle
            rules = document.get("starter_rules", {})
            if not isinstance(rules, dict):
                raise ValidationError("labelme_rules_invalid", "Imported workflow rules must be a JSON object.")
            metadata = {key: document[key] for key in ("starter_review_receipt", "starter_revision",
                        "starter_fingerprint", "starter_original_file_name") if key in document}
            parsed.append(LabelMeImage(name, raw, selected or image_basename, content, width, height, split,
                                       source_group, document["shapes"], flags, regions, rules, metadata))
        if not parsed:
            raise ValidationError("labelme_empty", "The ZIP contains no LabelMe JSON annotations.")
        return parsed
    finally:
        archive.close()


def _validate_shape_label(shape: dict) -> None:
    if not isinstance(shape, dict) or not isinstance(shape.get("label"), str) or not shape["label"].strip():
        raise ValidationError("labelme_label_invalid", "Every imported shape requires a nonempty label.")
    if len(shape["label"]) > 255:
        raise ValidationError("labelme_label_invalid", "LabelMe labels must be at most 255 characters.")


def _shape_polygon(shape: dict, width: int, height: int) -> list[tuple[float, float]]:
    _validate_shape_label(shape)
    kind = shape.get("shape_type", "polygon")
    if kind not in {"polygon", "rectangle"}:
        raise ValidationError("labelme_shape_unsupported", f"Unsupported LabelMe shape type: {kind}.")
    try:
        points = [(float(x), float(y)) for x, y in shape.get("points", [])]
    except (TypeError, ValueError) as exc:
        raise ValidationError("labelme_points_invalid", "LabelMe points must be numeric x/y pairs.") from exc
    if kind == "rectangle":
        if len(points) != 2:
            raise ValidationError("labelme_rectangle_invalid", "LabelMe rectangles require two opposite corners.")
        left, right = sorted((points[0][0], points[1][0]))
        top, bottom = sorted((points[0][1], points[1][1]))
        points = [(left, top), (right, top), (right, bottom), (left, bottom)]
    validate_polygon(points, width, height)
    return points


def _native_mask(shape: dict, width: int, height: int) -> np.ndarray:
    _validate_shape_label(shape)
    points = shape.get("points")
    if (not isinstance(points, list) or len(points) != 2
            or any(not isinstance(point, (list, tuple)) or len(point) != 2 for point in points)):
        raise ValidationError("labelme_mask_bounds_invalid", "Mask shapes require two inclusive pixel corners.")
    values = [value for point in points for value in point]
    if any(type(value) not in {int, float} or (type(value) is float and not math.isfinite(value))
           or int(value) != value for value in values):
        raise ValidationError("labelme_mask_bounds_invalid", "Mask corners must use finite integer pixel coordinates.")
    left, top, right, bottom = (int(value) for value in values)
    if not (0 <= left <= right < width and 0 <= top <= bottom < height):
        raise ValidationError("labelme_mask_bounds_invalid", "Mask corners must stay inside the image.")
    encoded = shape.get("mask")
    if not isinstance(encoded, str) or len(encoded) > ((_MAX_MASK_BYTES + 2) // 3) * 4:
        raise ValidationError("labelme_mask_data_invalid", "Mask data must be bounded base64 PNG data.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValidationError("labelme_mask_data_invalid", "Mask PNG data is not valid base64.") from exc
    if len(content) > _MAX_MASK_BYTES:
        raise ValidationError("labelme_mask_data_invalid", "Mask PNG exceeds the supported data limit.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or image.mode not in {"1", "L"}:
                raise ValidationError("labelme_mask_png_invalid", "Mask data must be a binary grayscale PNG.")
            if image.size != (right - left + 1, bottom - top + 1):
                raise ValidationError("labelme_mask_dimensions_mismatch", "Mask PNG dimensions differ from its inclusive corners.")
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            crop = np.asarray(image)
            if not np.isin(crop, [0, 1, 255]).all():
                raise ValidationError("labelme_mask_values_invalid", "Mask PNG pixels must be binary.")
            crop = crop.astype(np.bool_)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValidationError("labelme_mask_png_invalid", "Mask PNG cannot be decoded.") from exc
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[top:bottom + 1, left:right + 1] = crop
    if not mask.any():
        raise ValidationError("labelme_mask_empty", "Object masks must contain foreground pixels.")
    if shape.get("starter_rle") is not None and decode_mask(shape["starter_rle"]).shape != (height, width):
        raise ValidationError("labelme_mask_rle_dimensions_mismatch", "Source RLE metadata must match the image dimensions.")
    return mask


def _mapping(documents, mapping: dict | None, *, required: bool):
    labels = Counter()
    for document in documents:
        for shape in document.shapes:
            if not isinstance(shape, dict) or not isinstance(shape.get("label"), str):
                raise ValidationError("labelme_label_invalid", "Every LabelMe shape requires a label.")
            labels[shape["label"]] += 1
    if required and (mapping is None or set(labels) - set(mapping)):
        raise ValidationError("labelme_mapping_required", "Explicitly map every source label to a class name or null to ignore it.",
                              details={"labels": sorted(labels)})
    for source, target in (mapping or {}).items():
        if source not in labels:
            raise ValidationError("labelme_unknown_mapping", "Class mapping contains a label absent from this archive.")
        if target is not None:
            try:
                mapping[source] = ClassLabelCreate(name=target).name
            except SchemaValidationError as exc:
                raise ValidationError("labelme_mapping_invalid", "Mapped class names must be nonempty safe class names.") from exc
    return labels


def preview_labelme(payload: bytes, *, task_type: str = "segment", default_split: str = "train",
                    class_mapping: dict | None = None, limits: YoloArchiveLimits = YoloArchiveLimits()) -> dict:
    if task_type not in {"detect", "segment"}:
        raise ValidationError("labelme_task_unsupported", "LabelMe import currently supports detect and segment.")
    documents = _parse(payload, default_split, limits)
    labels = _mapping(documents, class_mapping, required=False)
    issues = []
    for document in documents:
        for index, shape in enumerate(document.shapes):
            if class_mapping is not None and shape["label"] in class_mapping and class_mapping[shape["label"]] is None:
                continue
            try:
                if shape.get("shape_type") == "mask":
                    _native_mask(shape, document.width, document.height)
                else:
                    _shape_polygon(shape, document.width, document.height)
            except ValidationError as exc:
                issues.append({"json_file": document.json_file, "shape_index": index, "code": exc.error_code,
                               "message": exc.message, "blocking": True})
    return {"image_count": len(documents), "shape_count": sum(labels.values()),
            "labels": [{"name": name, "count": count} for name, count in sorted(labels.items())],
            "images": [{"json_file": document.json_file, "image_file": document.image_file, "width": document.width,
                "height": document.height, "split": document.split, "source_group_id": document.source_group_id,
                "shape_count": len(document.shapes)} for document in documents],
            "issues": issues, "mapping_required": class_mapping is None or bool(set(labels) - set(class_mapping)),
            "capabilities": {"tasks": ["detect", "segment"], "shapes": ["polygon", "rectangle", "mask"],
                "group_id": "same-label instance union", "roi_ignore": "starter_roi/starter_ignore RLE preserved, requires image review",
                "coordinate_space": "image_pixel", "review_status": "pending"}}


def import_labelme(session: Session, storage: Storage, payload: bytes, dataset_payload: DatasetCreate,
                   class_mapping: dict, default_split: str = "train", limits: YoloArchiveLimits = YoloArchiveLimits()) -> dict:
    if dataset_payload.task_type.value not in {"detect", "segment"}:
        raise ValidationError("labelme_task_unsupported", "LabelMe import supports detect and segment.")
    documents = _parse(payload, default_split, limits)
    labels = _mapping(documents, class_mapping, required=True)
    target_names = sorted({class_mapping[label] for label in labels if class_mapping[label] is not None})
    if not target_names:
        raise ValidationError("labelme_classes_empty", "Map at least one source label to an imported class.")
    dataset = Dataset(id=new_id("ds"), name=dataset_payload.name, description=dataset_payload.description,
                      task_type=dataset_payload.task_type.value)
    session.add(dataset)
    session.flush()
    classes = {name: ClassLabel(id=new_id("cls"), dataset_id=dataset.id, class_index=index, name=name)
               for index, name in enumerate(target_names)}
    session.add_all(classes.values())
    annotation_count = 0
    source_groups = {}
    try:
        for document in documents:
            if document.source_group_id:
                prior_split = source_groups.setdefault(document.source_group_id, document.split)
                if prior_split != document.split:
                    raise ValidationError("source_group_split_leak", "One source group cannot span train, val and test.")
            image_id = new_id("img")
            storage_name = storage.safe_storage_name(document.image_file, image_id)
            storage.write_image(dataset.id, storage_name, document.content)
            source_digest = hashlib.sha256(document.original_json).hexdigest()
            originals = storage.dataset_dir(dataset.id) / "imports" / "labelme"
            originals.mkdir(parents=True, exist_ok=True)
            (originals / (source_digest + ".json")).write_bytes(document.original_json)
            image = ImageItem(id=image_id, dataset_id=dataset.id, file_name=PurePosixPath(document.image_file).name,
                storage_name=storage_name, width=document.width, height=document.height, split=document.split,
                source_group_id=document.source_group_id, source_file=document.json_file[:512],
                source_checksum=hashlib.sha256(document.content).hexdigest(), annotation_revision=1,
                review_status="pending", rules_json=document.rules)
            masks = MaskService(storage)
            for key, rle in document.regions.items():
                setattr(image, key + "_mask_ref", masks.write(rle).mask_ref)
            session.add(image)
            session.flush()
            groups = {}
            for index, shape in enumerate(document.shapes):
                target = class_mapping[shape["label"]]
                if target is None:
                    continue
                _validate_shape_label(shape)
                group_id = shape.get("group_id")
                if group_id is not None and not isinstance(group_id, (str, int)):
                    raise ValidationError("labelme_group_invalid", "Shape group_id must be a string, integer or null.")
                key = (shape["label"], f"group:{group_id}") if group_id is not None else (shape["label"], f"shape:{index}")
                groups.setdefault(key, []).append(shape)
            for index, ((source_label, group_id), members) in enumerate(groups.items()):
                common = {"id": new_id("ann"), "image_id": image.id, "dataset_id": dataset.id,
                    "class_id": classes[class_mapping[source_label]].id, "source": "imported",
                    "object_id": f"labelme-{source_digest[:16]}-{index}",
                    "provenance_json": {"format": "labelme", "source_json_hash": source_digest,
                        "source_label": source_label, "group_id": group_id,
                        "shapes": members, "flags": document.flags, "source_metadata": document.source_metadata}}
                if dataset.task_type == "segment":
                    mask = np.zeros((image.height, image.width), dtype=np.bool_)
                    for shape in members:
                        if shape.get("shape_type") == "mask":
                            mask |= _native_mask(shape, image.width, image.height)
                        else:
                            mask |= polygon_mask(_shape_polygon(shape, image.width, image.height), image.width, image.height)
                    session.add(Annotation(**common, type="polygon", mask_ref=masks.write(mask).mask_ref))
                else:
                    boxes = []
                    for shape in members:
                        if shape.get("shape_type") == "mask":
                            boxes.append(mask_bbox(_native_mask(shape, image.width, image.height)))
                        else:
                            points = _shape_polygon(shape, image.width, image.height)
                            xs, ys = zip(*points)
                            boxes.append(BBox(x=min(xs), y=min(ys), width=max(xs)-min(xs), height=max(ys)-min(ys)))
                    left = min(box.x for box in boxes)
                    top = min(box.y for box in boxes)
                    box = BBox(x=left, y=top, width=max(box.x + box.width for box in boxes)-left,
                               height=max(box.y + box.height for box in boxes)-top)
                    validate_bbox(box, image.width, image.height)
                    session.add(Annotation(**common, type="bbox", x=box.x, y=box.y, width=box.width, height=box.height))
                annotation_count += 1
            image.status = "annotated" if groups else "unannotated"
        session.flush()
        refresh_dataset_counts(session, dataset.id)
        session.commit()
        return {"dataset_id": dataset.id, "name": dataset.name, "task_type": dataset.task_type,
                "imported_images": len(documents), "imported_annotations": annotation_count,
                "warnings": ["Imported annotations require human review. Original JSON preserves source flags and ROI/ignore conventions."],
                "review_status": "pending"}
    except Exception:
        session.rollback()
        storage.remove_dataset(dataset.id)
        raise
