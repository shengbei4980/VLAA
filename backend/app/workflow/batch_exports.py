"""Export saved reviews one image at a time, with a manifest for partial results."""
from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO
import zipfile

from sqlalchemy import select

from app.core.errors import ConflictError, DomainError, NotFoundError, ValidationError
from app.core.models import ImageItem
from app.core.storage import Storage
from app.storage_settings.service import file_hash
from app.workflow.labelme_export import labelme_download
from app.workflow.models import WorkflowItem, WorkflowReview, WorkflowRun
from app.workflow.preview_renderer import render_preview
from app.workflow.previews import preview_download
from app.workflow.schemas import ReviewDraft


def _review_signature(review: dict) -> tuple:
    return tuple(review.get(key) for key in ("id", "artifact_id", "revision", "status", "decision"))


def _archive_stem(image_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", image_id):
        return image_id
    return "image-" + sha256(image_id.encode()).hexdigest()


def _write_member(archive: zipfile.ZipFile, source: BinaryIO, name: str, expected_hash: str):
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED
    digest = sha256()
    with archive.open(info, "w", force_zip64=True) as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ConflictError("workflow_batch_file_changed", "打包期间导出文件发生变化，请重新生成批量导出。")


class BatchExporter:
    def __init__(self, service):
        self.service = service
        self.storage = service.storage

    def _scope(self, run_id: str) -> tuple[list[dict], dict, dict]:
        # Only lightweight identities are loaded for the whole batch. Payloads are read per image.
        with self.service.session_factory() as session:
            run = session.get(WorkflowRun, run_id)
            if run is None:
                raise NotFoundError("workflow_run_not_found", "运行记录不存在。")
            if run.profile_snapshot["task_type"] not in {"detect", "segment"}:
                raise ValidationError("workflow_export_task_unsupported", "LabelMe JSON 批量导出支持检测和分割任务。")
            reviews = {}
            query = select(WorkflowReview.item_id, WorkflowReview.id, WorkflowReview.artifact_id,
                           WorkflowReview.revision, WorkflowReview.status, WorkflowReview.decision).where(
                WorkflowReview.run_id == run_id).order_by(WorkflowReview.created_at, WorkflowReview.id)
            for row in session.execute(query).mappings():
                reviews[row["item_id"]] = dict(row)
            items = []
            for row in session.execute(select(WorkflowItem.id, WorkflowItem.subject_ref, WorkflowItem.disposition).where(
                    WorkflowItem.run_id == run_id).order_by(WorkflowItem.created_at, WorkflowItem.id)):
                items.append({"item_id": row.id, "image_id": row.subject_ref, "disposition": row.disposition,
                              "review": reviews.get(row.id)})
            return items, run.profile_snapshot, run.context_json.get("output_destinations", {})

    def _ensure_current(self, expected: dict):
        with self.service.session_factory() as session:
            query = select(WorkflowReview.id, WorkflowReview.artifact_id, WorkflowReview.revision,
                           WorkflowReview.status, WorkflowReview.decision).where(
                WorkflowReview.item_id == expected["item_id"]).order_by(
                WorkflowReview.created_at.desc(), WorkflowReview.id.desc()).limit(1)
            current = session.execute(query).mappings().first()
            if current is None or _review_signature(dict(current)) != _review_signature(expected):
                raise ConflictError("workflow_batch_review_changed", "导出期间复核记录已更新，请重新导出此图。")

    def export_review(self, review_id: str, include_previews: bool = True) -> dict:
        review = self.service.get_review(review_id)
        with self.service.session_factory() as session:
            run = session.get(WorkflowRun, review["run_id"])
            profile = run.profile_snapshot
        return self._export_one(review, profile, include_previews)

    def _export_one(self, expected: dict, profile: dict, include_previews: bool) -> dict:
        self._ensure_current(expected)
        review = self.service.get_review(expected["id"])
        if _review_signature(review) != _review_signature(expected):
            raise ConflictError("workflow_batch_review_changed", "导出期间复核记录已更新，请重新导出此图。")
        if review.get("decision") == "excluded":
            raise ValidationError("workflow_batch_review_excluded", "已排除的图片不纳入标注导出。")
        payload = review["payload"]
        with self.service.session_factory() as session:
            image = session.get(ImageItem, payload["image_id"])
            if image is None or image.dataset_id != payload["dataset_id"]:
                raise ValidationError("workflow_review_identity_changed", "原图不属于本次复核的数据集。")
            image_path = self.storage.image_path(image.dataset_id, image.storage_name)
            original_file_name = image.file_name
        json_result = self.service.save_review_labelme(review["id"], ReviewDraft(
            expected_revision=review["revision"], payload={}))
        files = {"json": {"path": f"json/{_archive_stem(payload['image_id'])}.json",
                          "sha256": file_hash(labelme_download(self.storage, review["id"], json_result["file_name"]))}}
        result = {"image_id": payload["image_id"], "review_id": review["id"], "status": "exported",
                  "export_state": json_result["export_state"], "review_revision": review["revision"],
                  "original_file_name": original_file_name, "width": payload["width"], "height": payload["height"],
                  "json_path": json_result["saved_path"], "json_file_name": json_result["file_name"],
                  "pending_objects": json_result["pending_objects"], "files": files}
        self._ensure_current(expected)
        if include_previews:
            content = render_preview(self.storage, image_path, payload, profile["class_rules"])
            self._ensure_current(expected)
            preview = self.service.save_review_preview(review["id"], content, review["revision"])
            if preview["preview_state"] != json_result["export_state"]:
                raise ConflictError("workflow_batch_review_changed", "导出期间复核状态已更新，请重新导出此图。")
            result.update(image_path=preview["saved_path"], image_file_name=preview["file_name"])
            files["image"] = {"path": f"images/{_archive_stem(payload['image_id'])}.png",
                              "sha256": file_hash(preview_download(self.storage, review["id"], preview["file_name"]))}
        self._ensure_current(expected)
        return result

    def export_run(self, run_id: str, include_previews: bool = True) -> dict:
        items, profile, destinations = self._scope(run_id)
        results = []
        exports_dir = self.storage.export_path("annotation-batch.zip").parent
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=exports_dir, suffix=".batch.tmp", delete=False) as output:
                temporary = Path(output.name)
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
                for item in items:
                    expected = item["review"]
                    if item["disposition"] == "excluded" or not expected or expected.get("decision") == "excluded":
                        results.append({"image_id": item["image_id"], "status": "skipped",
                            "review_id": expected["id"] if expected else None,
                            "reason": "图片已排除" if item["disposition"] == "excluded" or expected and expected.get("decision") == "excluded"
                                      else "图片尚未生成可导出的复核记录"})
                        continue
                    with ExitStack() as input_files:
                        try:
                            result = self._export_one(expected, profile, include_previews)
                            members = []
                            for kind, key in (("json", "json_file_name"), ("image", "image_file_name")):
                                if key in result:
                                    source = self.storage.export_path(result[key])
                                    handle = input_files.enter_context(source.open("rb"))
                                    metadata = result["files"][kind]
                                    if file_hash(source) != metadata["sha256"]:
                                        raise ConflictError("workflow_batch_file_changed", "导出文件发生变化，请重新导出此图。")
                                    members.append((handle, metadata))
                            self._ensure_current(expected)
                        except (DomainError, OSError, ValueError) as error:
                            results.append({"image_id": item["image_id"], "review_id": expected["id"], "status": "failed",
                                "review_revision": expected["revision"], "error": str(error),
                                "error_code": getattr(error, "error_code", "workflow_batch_image_failed")})
                            continue
                        # Once a ZIP member starts, a read/write failure aborts this package.
                        # Comparing streamed bytes prevents publishing a stale manifest checksum.
                        for handle, metadata in members:
                            _write_member(archive, handle, metadata["path"], metadata["sha256"])
                        results.append(result)
                summary = {"total_images": len(items),
                    "exported_images": sum(item["status"] == "exported" for item in results),
                    "pending_images": sum(item.get("export_state") == "draft" for item in results),
                    "failed_images": sum(item["status"] == "failed" for item in results),
                    "skipped_images": sum(item["status"] == "skipped" for item in results)}
                manifest = {"schema_version": 1, "run_id": run_id, "coordinate_space": "image_pixel",
                            "include_previews": include_previews, **summary, "results": results}
                info = zipfile.ZipInfo("manifest.json")
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2).encode())
            with temporary.open("rb+") as output:
                output.flush()
                os.fsync(output.fileno())
            digest = file_hash(temporary)
            file_name = f"{run_id}-annotation-batch-{digest}.zip"
            destination = self.storage.export_path(file_name)
            if destination.name != file_name:
                raise ValidationError("workflow_batch_unsafe", "批量导出保存位置无效。")
            if not destination.exists():
                try:
                    if os.name == "nt":
                        os.rename(temporary, destination)
                    else:
                        os.link(temporary, destination)
                except FileExistsError:
                    pass
            if not destination.is_file() or file_hash(destination) != digest:
                raise ConflictError("workflow_batch_changed", "批量导出文件内容发生变化，请重新生成。")
            from app.workflow.projects import publish_run_export

            actual = publish_run_export(self.service.project_root, destination, "package", destinations)
            return {"file_name": file_name, "saved_path": str(actual), **summary, "results": results,
                    "download_url": f"/api/workflows/runs/{run_id}/annotation-batch/{file_name}"}
        except OSError as error:
            raise ValidationError("workflow_batch_write_failed", "无法保存批量导出，请检查输出目录、权限和可用空间。") from error
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def batch_download(storage: Storage, run_id: str, file_name: str) -> Path:
    match = re.fullmatch(rf"{re.escape(run_id)}-annotation-batch-([0-9a-f]{{64}})\.zip", file_name)
    if match is None:
        raise ValidationError("workflow_batch_unsafe", "批量导出文件不属于当前运行。")
    path = storage.export_path(file_name)
    if path.name != file_name:
        raise ValidationError("workflow_batch_unsafe", "批量导出保存位置无效。")
    try:
        if not path.is_file():
            raise NotFoundError("workflow_batch_missing", "批量导出文件不存在，请重新生成。")
        if file_hash(path) != match.group(1):
            raise ConflictError("workflow_batch_changed", "批量导出文件内容发生变化，请重新生成。")
    except OSError as error:
        raise ValidationError("workflow_batch_read_failed", "无法读取批量导出，请检查输出目录。") from error
    return path
