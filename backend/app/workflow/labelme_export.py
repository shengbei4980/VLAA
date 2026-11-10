"""Portable LabelMe drafts, exported independently of formal label confirmation."""
from __future__ import annotations

import base64
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile

from PIL import Image
from pydantic import ValidationError as SchemaError

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.schemas import AnnotationInput, BBox
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_bbox
from app.dataset.annotation.masks import MaskService, decode_mask, encode_mask
from app.dataset.annotation.service import _validate_input
from app.workflow.exports import _labelme_document

MAX_LABELME_BYTES = 50 * 1024 * 1024


def _document(storage: Storage, payload: dict, profile: dict, image: dict) -> tuple[dict, list[dict], int]:
    raw_objects = payload.get("objects", payload.get("annotations", []))
    if not isinstance(raw_objects, list) or len(raw_objects) > 10000 or any(not isinstance(item, dict) for item in raw_objects):
        raise ValidationError("workflow_invalid_annotations", "标注候选必须为不超过 10000 项的对象列表。")
    objects = [dict(item) for item in raw_objects]
    names = {rule.get("class_id", rule.get("id")): rule["name"] for rule in profile["class_rules"]}
    masks = MaskService(storage)
    annotations = []
    pending = 0
    ids = [item.get("object_id", item.get("id")) for item in objects]
    named_ids = [value for value in ids if value is not None]
    if any(not isinstance(value, str) for value in named_ids) or len(named_ids) != len(set(named_ids)):
        raise ValidationError("workflow_duplicate_object", "候选对象编号必须为互不重复的文字。")
    for item in objects:
        if item.get("decision") == "rejected":
            continue
        class_id = item.get("class_id")
        if class_id is not None and not isinstance(class_id, str):
            raise ValidationError("workflow_labelme_class_invalid", "候选类别编号必须为文字或空值。")
        if class_id not in names:
            pending += 1
            continue
        if item.get("type") not in {"bbox", "polygon"}:
            raise ValidationError("workflow_export_task_unsupported", "LabelMe JSON 支持目标框和分割掩膜，请使用检测或分割任务。")
        try:
            if item.get("mask") is not None:
                mask = decode_mask(item["mask"])
                if item["type"] != "polygon" or mask.shape != (image["height"], image["width"]) or not mask.any():
                    raise ValidationError("mask_geometry_invalid", "对象掩膜必须与原图尺寸一致并包含目标像素。")
                item["mask"] = encode_mask(mask)
                if item.get("bbox") is not None:
                    validate_bbox(BBox.model_validate(item["bbox"]), image["width"], image["height"])
            else:
                _validate_input(AnnotationInput.model_validate(item), image["width"], image["height"], storage)
        except (SchemaError, ValueError) as error:
            raise ValidationError("workflow_labelme_geometry_invalid", "候选标注格式无效，请修正对应对象后导出。") from error
        annotations.append(item)
    source = {**image, "annotations": annotations, "rules": profile.get("rules", {}),
              "revision": payload.get("revision"), "roi_mask_ref": payload.get("roi_mask_ref"),
              "ignore_mask_ref": payload.get("ignore_mask_ref")}
    for layer in ("roi", "ignore"):
        if payload.get(layer) is not None:
            source[layer + "_mask_ref"] = None
    document = _labelme_document(source, names, masks, Path(image["file_name"]).name)
    for layer in ("roi", "ignore"):
        rle = payload.get(layer) if payload.get(layer) is not None else document.get("starter_" + layer)
        if rle is not None:
            mask = decode_mask(rle)
            if mask.shape != (image["height"], image["width"]):
                raise ValidationError("workflow_mask_dimensions_mismatch", "有效区域和排除区域必须与原图尺寸一致。")
            document["starter_" + layer] = encode_mask(mask)
    return document, objects, pending


def _semantic_state(document: dict, objects: list[dict], payload: dict) -> tuple:
    shapes = [{key: value for key, value in shape.items() if key != "starter_annotation"} for shape in document["shapes"]]
    candidates = [(item.get("object_id", item.get("id")), item.get("class_id"), item.get("type"),
                   "rejected" if item.get("decision") == "rejected" else "accepted") for item in objects]
    issues = payload.get("issues", [])
    if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
        raise ValidationError("workflow_labelme_issues_invalid", "复核问题必须为对象列表。")
    decisions = [{key: issue.get(key) for key in ("id", "type", "object_id", "blocking", "resolution", "reason")} for issue in issues]
    return shapes, candidates, decisions, document.get("starter_roi"), document.get("starter_ignore")


def save_labelme(storage: Storage, review: dict, payload: dict, profile: dict, image: dict,
                 *, project_root: Path, output_destinations: dict | None = None) -> dict:
    if profile["task_type"] not in {"detect", "segment"}:
        raise ValidationError("workflow_export_task_unsupported", "LabelMe JSON 支持检测和分割任务。")
    try:
        if len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()) > MAX_LABELME_BYTES:
            raise ValidationError("workflow_labelme_too_large", "标注 JSON 超过 50 MiB，请缩小本次导出范围。")
    except (TypeError, ValueError) as error:
        raise ValidationError("workflow_labelme_invalid_data", "标注包含无法导出的数值或数据，请检查候选坐标。") from error
    source_path = storage.image_path(image["dataset_id"], image["storage_name"])
    try:
        if source_path.stat().st_size > MAX_LABELME_BYTES:
            raise ValidationError("workflow_labelme_too_large", "原图超过单文件 JSON 支持的大小。")
        content = source_path.read_bytes()
        if review["payload"].get("image_hash") and sha256(content).hexdigest() != review["payload"]["image_hash"]:
            raise ConflictError("workflow_labelme_image_changed", "原图已改变，请重新检查当前图片后导出。")
        with Image.open(io.BytesIO(content)) as original:
            if original.size != (image["width"], image["height"]):
                raise ValidationError("workflow_labelme_dimensions_mismatch", "原图尺寸与复核记录不一致。")
            original.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValidationError("workflow_labelme_image_unavailable", "原图无法完整读取，请检查原图后重试。") from error
    document, objects, pending = _document(storage, payload, profile, image)
    confirmed = review["status"] == "decided" and review.get("decision") in {"approved", "background"}
    if confirmed:
        original_document, original_objects, _ = _document(storage, review["payload"], profile, image)
        confirmed = _semantic_state(document, objects, payload) == _semantic_state(original_document, original_objects, review["payload"])
    else:
        _semantic_state(document, objects, payload)
    state = "confirmed" if confirmed else "draft"
    document["imageData"] = base64.b64encode(content).decode("ascii")
    document["starter_review_receipt"] = review.get("receipt") if confirmed else None
    document["vlaa_review"] = {"review_id": review["id"], "run_id": review["run_id"],
        "review_revision": review["revision"], "export_state": state, "exported_objects": len(document["shapes"]),
        "pending_objects": pending, "class_rules": profile["class_rules"], "objects": objects,
        "issues": payload.get("issues", []), "roi": document.get("starter_roi"), "ignore": document.get("starter_ignore"),
        "status": review["status"], "decision": review.get("decision"), "reason": review.get("reason"),
        "receipt": review.get("receipt")}
    document["flags"]["vlaa_draft"] = not confirmed
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_LABELME_BYTES:
        raise ValidationError("workflow_labelme_too_large", "含原图的 LabelMe JSON 超过 50 MiB，请使用分批导出。")
    digest = sha256(encoded).hexdigest()
    file_name = f"{review['id']}-r{review['revision']}-{state}-{digest}.json"
    destination = storage.export_path(file_name)
    if destination.name != file_name:
        raise ValidationError("workflow_labelme_unsafe", "标注文件名与受管保存位置不一致。")
    temporary_name = None
    try:
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as output:
                temporary_name = output.name
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, destination)
        if not destination.is_file() or sha256(destination.read_bytes()).hexdigest() != digest:
            raise ConflictError("workflow_labelme_changed", "已保存的 JSON 内容发生变化，请重新生成。")
    except OSError as error:
        raise ValidationError("workflow_labelme_write_failed", "无法保存 JSON，请检查输出目录的空间和权限。") from error
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    from app.workflow.projects import publish_run_export

    actual_path = publish_run_export(project_root, destination, "json", output_destinations or {})
    return {"file_name": file_name, "saved_path": str(actual_path), "review_revision": review["revision"],
            "export_state": state, "exported_objects": len(document["shapes"]), "pending_objects": pending,
            "download_url": f"/api/workflows/reviews/{review['id']}/labelme/{file_name}"}


def labelme_download(storage: Storage, review_id: str, file_name: str) -> Path:
    match = re.fullmatch(rf"{re.escape(review_id)}-r[1-9][0-9]*-(?:draft|confirmed)-([0-9a-f]{{64}})\.json", file_name)
    if match is None:
        raise ValidationError("workflow_labelme_unsafe", "JSON 文件不属于当前图片的复核记录。")
    path = storage.export_path(file_name)
    if path.name != file_name:
        raise ValidationError("workflow_labelme_unsafe", "JSON 文件保存位置无效。")
    try:
        if not path.is_file():
            raise NotFoundError("workflow_labelme_missing", "JSON 文件不存在，请重新生成。")
        if path.stat().st_size > MAX_LABELME_BYTES or sha256(path.read_bytes()).hexdigest() != match.group(1):
            raise ConflictError("workflow_labelme_changed", "JSON 文件内容发生变化，请重新生成。")
    except OSError as error:
        raise ValidationError("workflow_labelme_read_failed", "无法读取 JSON 文件，请检查输出目录。") from error
    return path
