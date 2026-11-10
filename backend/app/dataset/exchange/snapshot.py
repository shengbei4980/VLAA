from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from app.core.errors import ValidationError
from app.core.storage import Storage
from app.dataset.snapshots import verify_snapshot_images


def requires_mask_adapter(manifest: dict) -> bool:
    return any(image.get("roi_mask_ref") or image.get("ignore_mask_ref") or
               any(item.get("mask_ref") for item in image["annotations"]) for image in manifest["images"])


def yolo_annotation_line(annotation: dict, image: dict, class_indexes: dict[str, int], storage: Storage) -> str:
    from app.dataset.exchange.yolo import _obb_to_yolo

    index = class_indexes[annotation["class_id"]]
    if annotation["type"] == "bbox":
        bbox = annotation["bbox"]
        return f"{index} {(bbox['x'] + bbox['width']/2)/image['width']:.8f} {(bbox['y'] + bbox['height']/2)/image['height']:.8f} {bbox['width']/image['width']:.8f} {bbox['height']/image['height']:.8f}"
    if annotation["type"] == "obb":
        return f"{index} {_obb_to_yolo(annotation['obb'], image['width'], image['height'])}"
    polygon = annotation.get("polygon")
    if annotation.get("mask_ref"):
        import cv2
        import numpy as np
        from app.dataset.annotation.masks import MaskService, polygon_mask

        mask = MaskService(storage).decode(annotation["mask_ref"])
        if polygon is None or not np.array_equal(polygon_mask(polygon, image["width"], image["height"]), mask):
            contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 1 or hierarchy is None or hierarchy[0][0][2] != -1:
                raise ValidationError("yolo_geometry_loss", "YOLO polygon export cannot preserve this object's holes or disconnected components.",
                                      details={"image_id": image["id"], "object_id": annotation.get("object_id"), "format": "COCO RLE"})
            polygon = contours[0][:, 0, :].tolist()
        rounded = [[float(f"{point[0]/image['width']:.8f}") * image["width"],
                    float(f"{point[1]/image['height']:.8f}") * image["height"]] for point in polygon]
        if len(polygon) < 3 or not np.array_equal(polygon_mask(rounded, image["width"], image["height"]), mask):
            raise ValidationError("yolo_geometry_loss", "The polygon round trip changes the canonical mask. Choose COCO RLE export.",
                                  details={"image_id": image["id"], "object_id": annotation.get("object_id")})
    if not polygon:
        raise ValidationError("yolo_polygon_missing", "A polygon is required for YOLO text export.")
    normalized = " ".join(f"{point[0]/image['width']:.8f} {point[1]/image['height']:.8f}" for point in polygon)
    return f"{index} {normalized}"


def export_snapshot_directory(snapshot: dict, storage: Storage, target_dir: Path) -> dict:
    verify_snapshot_images(snapshot)
    manifest = snapshot["manifest"]
    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    class_indexes = {item["id"]: item["index"] for item in manifest["classes"]}
    class_names = {item["id"]: item["name"] for item in manifest["classes"]}
    counts = {split: 0 for split in ("train", "val", "test")}
    positives = dict(counts)
    labels = 0
    mask_aware = requires_mask_adapter(manifest)
    if mask_aware and manifest["task_type"] not in {"detect", "segment"}:
        raise ValidationError("mask_adapter_task_unsupported", "ROI/ignore supervision requires a compatible task adapter.")
    for image in manifest["images"]:
        source = Path(snapshot["root"]) / image["path"]
        split = image["split"]
        annotations = image["annotations"]
        if manifest["task_type"] == "classify":
            from app.dataset.exchange.yolo import safe_class_directory

            if len(annotations) != 1:
                raise ValidationError("classification_label_required", "Classification snapshots require one label per image.")
            destination = target / split / safe_class_directory(class_names[annotations[0]["class_id"]]) / source.name
        else:
            destination = target / "images" / split / source.name
            if not mask_aware:
                lines = [yolo_annotation_line(item, image, class_indexes, storage) for item in annotations]
                label = target / "labels" / split / (source.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        counts[split] += 1
        positives[split] += bool(annotations)
        labels += len(annotations)
    data_yaml = target / "data.yaml"
    data = {"path": str(target), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": [item["name"] for item in manifest["classes"]]}
    if mask_aware:
        data.update(workflow_manifest=str(Path(snapshot["root"]) / "manifest.json"), workflow_data_dir=str(storage.data_dir),
                    workflow_snapshot_id=snapshot["id"])
    if manifest["task_type"] != "classify":
        data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"root": str(target), "data_yaml": str(target if manifest["task_type"] == "classify" else data_yaml),
            "counts": counts, "annotated_image_counts": positives, "confirmed_image_counts": counts,
            "total_image_counts": counts, "skipped_image_counts": {key: 0 for key in counts},
            "label_count": labels, "snapshot_id": snapshot["id"], "mask_aware": mask_aware}
