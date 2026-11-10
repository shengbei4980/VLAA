"""Publish confirmed immutable snapshots through the workflow export node."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import yaml

from app.core.errors import ValidationError
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService, decode_mask, encode_mask, mask_bbox, polygon_mask
from app.dataset.exchange.snapshot import yolo_annotation_line
from app.dataset.snapshots import verify_snapshot_images


def _labelme_document(image: dict, names: dict, masks: MaskService, image_name: str) -> dict:
    """Use LabelMe's native PNG mask shape so holes and disconnected pixels survive."""
    document = {
        "version": "5.8.0", "flags": {}, "shapes": [], "imagePath": image_name,
        "imageData": None, "imageHeight": image["height"], "imageWidth": image["width"],
        "split": image["split"], "source_group_id": image.get("source_group_id"),
        "starter_original_file_name": image.get("file_name", image_name),
        "starter_rules": image.get("rules", {}), "starter_review_receipt": image.get("receipt"),
        "starter_revision": image.get("revision"), "starter_fingerprint": image.get("fingerprint"),
    }
    for key in ("roi_mask_ref", "ignore_mask_ref"):
        if image.get(key):
            document["starter_" + key.removesuffix("_mask_ref")] = masks.read(image[key])
    for index, annotation in enumerate(image["annotations"], 1):
        shape = {"label": names[annotation["class_id"]], "group_id": index,
                 "description": "", "flags": {}, "mask": None, "starter_annotation": annotation}
        if annotation["type"] == "bbox":
            box = annotation["bbox"]
            shape.update(shape_type="rectangle", points=[
                [box["x"], box["y"]], [box["x"] + box["width"], box["y"] + box["height"]]])
        elif annotation["type"] == "polygon":
            if annotation.get("mask") is not None:
                rle = annotation["mask"]
                mask = decode_mask(rle)
            elif annotation.get("mask_ref"):
                rle = masks.read(annotation["mask_ref"])
                mask = decode_mask(rle)
            else:
                mask = polygon_mask(annotation["polygon"], image["width"], image["height"])
                rle = encode_mask(mask)
            if list(mask.shape) != [image["height"], image["width"]]:
                raise ValidationError("workflow_mask_dimensions_mismatch", "Object masks must match the frozen image dimensions.")
            box = mask_bbox(mask)
            if box is None:
                raise ValidationError("workflow_empty_object_mask", "Confirmed object masks must contain foreground pixels.")
            left, top = int(box.x), int(box.y)
            right, bottom = left + int(box.width), top + int(box.height)
            png = io.BytesIO()
            Image.fromarray(mask[top:bottom, left:right]).save(png, format="PNG")
            shape.update(shape_type="mask", points=[[left, top], [right - 1, bottom - 1]],
                         mask=base64.b64encode(png.getvalue()).decode("ascii"), starter_rle=rle)
        else:
            raise ValidationError("workflow_export_task_unsupported", "LabelMe export supports detection boxes and segmentation masks.")
        document["shapes"].append(shape)
    return document


def export_snapshot(snapshot: dict, storage: Storage, format_name: str) -> dict:
    if format_name not in {"yolo", "coco", "labelme"}:
        raise ValidationError("workflow_export_format_unsupported", "Choose YOLO, COCO or LabelMe export.")
    verify_snapshot_images(snapshot)
    manifest = snapshot["manifest"]
    classes = {item["id"]: item["index"] for item in manifest["classes"]}
    names = {item["id"]: item["name"] for item in manifest["classes"]}
    if format_name == "coco" and manifest["task_type"] not in {"detect", "segment"}:
        raise ValidationError("workflow_export_task_unsupported", "COCO export supports detect and segment snapshots.")
    if format_name == "labelme" and manifest["task_type"] not in {"detect", "segment"}:
        raise ValidationError("workflow_export_task_unsupported", "LabelMe export supports detect and segment snapshots.")
    buffer = io.BytesIO()
    masks = MaskService(storage)
    coco = {"images": [], "annotations": [], "categories": [
        {"id": item["index"] + 1, "name": item["name"]} for item in manifest["classes"]]}
    written_masks = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workflow_manifest.json", json.dumps(manifest, ensure_ascii=False))
        for image_index, image in enumerate(manifest["images"], 1):
            source = Path(snapshot["root"]) / image["path"]
            annotations = image["annotations"]
            relative = f"images/{image['split']}/{source.name}"
            if format_name == "yolo" and manifest["task_type"] == "classify":
                if len(annotations) != 1:
                    raise ValidationError("classification_label_required", "Classification export requires one confirmed label per image.")
                class_name = names[annotations[0]["class_id"]]
                from app.dataset.exchange.yolo import safe_class_directory

                safe_class_directory(class_name)
                relative = f"{image['split']}/{class_name}/{source.name}"
            archive.write(source, relative)
            if format_name == "labelme":
                document = _labelme_document(image, names, masks, source.name)
                archive.writestr(f"images/{image['split']}/{source.stem}.json", json.dumps(document, ensure_ascii=False, allow_nan=False))
                references = [image.get("roi_mask_ref"), image.get("ignore_mask_ref"),
                              *[annotation.get("mask_ref") for annotation in annotations]]
                for reference in references:
                    if reference and reference not in written_masks:
                        masks.read(reference)
                        archive.write(masks.path(reference), reference)
                        written_masks.add(reference)
            if format_name == "yolo" and manifest["task_type"] != "classify":
                if image.get("roi_mask_ref") or image.get("ignore_mask_ref"):
                    raise ValidationError("yolo_ignore_unsupported", "YOLO text cannot encode ROI/ignore supervision. Export COCO plus its workflow manifest.")
                lines = [yolo_annotation_line(annotation, image, classes, storage) for annotation in annotations]
                archive.writestr(f"labels/{image['split']}/{source.stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
            if format_name == "coco":
                coco_image = {"id": image_index, "file_name": relative, "width": image["width"], "height": image["height"],
                              "split": image["split"], "starter_original_file_name": image.get("file_name", source.name),
                              "starter_source_group_id": image.get("source_group_id"), "starter_rules": image.get("rules", {})}
                for key in ("roi_mask_ref", "ignore_mask_ref"):
                    if image.get(key):
                        coco_image["starter_" + key.removesuffix("_mask_ref")] = masks.read(image[key])
                coco["images"].append(coco_image)
                for annotation in annotations:
                    row = {"id": len(coco["annotations"]) + 1, "image_id": image_index,
                           "category_id": classes[annotation["class_id"]] + 1, "iscrowd": 0}
                    if annotation["type"] == "bbox":
                        box = annotation["bbox"]
                        row.update(bbox=[box["x"], box["y"], box["width"], box["height"]],
                                   area=box["width"] * box["height"], segmentation=[])
                    else:
                        mask = masks.decode(annotation["mask_ref"]) if annotation.get("mask_ref") else polygon_mask(annotation["polygon"], image["width"], image["height"])
                        box = mask_bbox(mask)
                        if box is None:
                            raise ValidationError("workflow_empty_object_mask", "Confirmed object masks must contain foreground pixels.")
                        row.update(bbox=[box.x, box.y, box.width, box.height], area=int(mask.sum()),
                                   segmentation=masks.read(masks.write(mask).mask_ref))
                    coco["annotations"].append(row)
        if format_name == "yolo" and manifest["task_type"] != "classify":
            archive.writestr("data.yaml", yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test",
                "names": {item["index"]: item["name"] for item in manifest["classes"]}}, allow_unicode=True))
        elif format_name == "coco":
            archive.writestr("annotations.json", json.dumps(coco, ensure_ascii=False))
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    file_name = f"workflow_{snapshot['id']}_{digest[:16]}_{format_name}.zip"
    destination = storage.export_path(file_name)
    destination.write_bytes(content)
    result = {"snapshot_id": snapshot["id"], "format": format_name, "file_name": file_name,
              "sha256": digest, "coordinate_space": "image_pixel", "image_count": len(manifest["images"])}
    if format_name == "labelme":
        result["capabilities"] = {"tasks": ["detect", "segment"], "minimum_labelme_version": "5.8.0",
                                  "segmentation": "native_mask", "roi_ignore": "starter_extensions"}
    return result
