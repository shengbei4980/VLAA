from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import ClassLabel, DatasetSnapshot, ImageItem
from app.core.storage import Storage
from app.dataset.annotation.review import json_fingerprint, require_confirmed
from app.dataset.service import get_dataset


def snapshot_root(storage: Storage, snapshot_id: str) -> Path:
    if not snapshot_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in snapshot_id):
        raise ValidationError("snapshot_id_invalid", "Snapshot IDs must be managed identifiers.")
    return storage.data_dir / "snapshots" / snapshot_id


def get_snapshot(session: Session, storage: Storage, snapshot_id: str) -> dict:
    record = session.get(DatasetSnapshot, snapshot_id)
    if record is None:
        raise NotFoundError("snapshot_not_found", "The confirmed dataset snapshot was not found.")
    root = snapshot_root(storage, record.id)
    path = root / "manifest.json"
    if not path.is_file():
        raise ValidationError("snapshot_missing", "The frozen snapshot manifest is missing.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if json_fingerprint(manifest) != record.fingerprint:
        raise ConflictError("snapshot_changed", "The frozen snapshot manifest failed its integrity check.")
    return {"id": record.id, "snapshot_id": record.id, "root": str(root), "manifest": manifest, **record.summary_json}


def create_snapshot(
    session: Session, storage: Storage, dataset_id: str, *, image_ids: list[str] | None = None,
    rules: dict | None = None, snapshot_id: str | None = None,
) -> dict:
    dataset = get_dataset(session, dataset_id)
    resolved_id = snapshot_id or new_id("snapshot")
    existing = session.get(DatasetSnapshot, resolved_id)
    if existing is not None:
        result = get_snapshot(session, storage, resolved_id)
        frozen_ids = {image["id"] for image in result["manifest"]["images"]}
        if existing.dataset_id != dataset_id or image_ids is not None and frozen_ids != set(image_ids):
            raise ConflictError("snapshot_scope_conflict", "The snapshot ID already belongs to a different input range.")
        return result
    details = require_confirmed(session, storage, dataset_id, image_ids)
    classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
    manifest = {"schema_version": 1, "coordinate_space": "image_pixel", "dataset_id": dataset_id,
                "dataset_name": dataset.name, "task_type": dataset.task_type, "rules": rules or {},
                "classes": [{"id": item.id, "name": item.name, "index": item.class_index, "color": item.color} for item in classes],
                "images": []}
    root = snapshot_root(storage, resolved_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(root.name + ".tmp-" + uuid.uuid4().hex)
    staging.mkdir()
    counts = {"train": 0, "val": 0, "test": 0}
    groups: dict[str, str] = {}
    try:
        (staging / "images").mkdir()
        for detail in details:
            image = session.get(ImageItem, detail["image_id"])
            if image.source_group_id:
                previous_split = groups.setdefault(image.source_group_id, image.split)
                if previous_split != image.split:
                    raise ValidationError("source_group_split_leak", "Images from one source group must remain in one split.")
            suffix = Path(image.storage_name).suffix.lower()
            relative_path = f"images/{image.id}{suffix}"
            shutil.copy2(storage.image_path(dataset_id, image.storage_name), staging / relative_path)
            with (staging / relative_path).open("rb") as copied:
                if hashlib.file_digest(copied, "sha256").hexdigest() != detail["image_hash"]:
                    raise ConflictError("snapshot_image_changed", "An image changed while its snapshot was being copied.")
            manifest["images"].append({
                "id": image.id, "file_name": image.file_name, "path": relative_path, "split": image.split,
                "width": image.width, "height": image.height, "image_hash": detail["image_hash"],
                "source_group_id": image.source_group_id, "source_file": image.source_file,
                "revision": detail["revision"], "fingerprint": detail["fingerprint"], "receipt": detail["receipt"],
                "annotations": detail["annotations"], "roi_mask_ref": detail["roi_mask_ref"],
                "ignore_mask_ref": detail["ignore_mask_ref"], "rules": detail["rules"],
            })
            counts[image.split] += 1
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        if root.exists():
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if json_fingerprint(saved) != json_fingerprint(manifest):
                raise ConflictError("snapshot_scope_conflict", "A different immutable snapshot already exists at this ID.")
            if not staging.resolve().is_relative_to(root.parent.resolve()):
                raise ValidationError("snapshot_staging_unsafe", "Snapshot staging must remain within managed storage.")
            shutil.rmtree(staging)
        else:
            staging.replace(root)
        summary = {"counts": counts, "image_count": len(details),
                   "annotation_count": sum(len(item["annotations"]) for item in details), "coordinate_space": "image_pixel"}
        session.add(DatasetSnapshot(id=resolved_id, dataset_id=dataset_id, fingerprint=json_fingerprint(manifest),
                                    manifest_path=str(root / "manifest.json"), summary_json=summary))
        session.flush()
        return {"id": resolved_id, "snapshot_id": resolved_id, "root": str(root), "manifest": manifest, **summary}
    except Exception:
        if staging.exists():
            if not staging.resolve().is_relative_to(root.parent.resolve()):
                raise ValidationError("snapshot_staging_unsafe", "Snapshot staging must remain within managed storage.")
            shutil.rmtree(staging)
        raise


def verify_snapshot_images(snapshot: dict) -> None:
    root = Path(snapshot["root"])
    for image in snapshot["manifest"]["images"]:
        path = (root / image["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValidationError("snapshot_image_missing", "A frozen image is missing or outside its managed snapshot.")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != image["image_hash"]:
                raise ConflictError("snapshot_image_changed", "A frozen image failed its integrity check.")
