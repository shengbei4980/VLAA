"""Managed PNG previews, kept separate from formal labels and review receipts."""
from __future__ import annotations

from hashlib import sha256
import io
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.storage import Storage

MAX_PREVIEW_BYTES = 50 * 1024 * 1024


def save_preview(storage: Storage, review: dict, content: bytes, expected_revision: int,
                 has_unsaved_changes: bool = False, *, project_root: Path | None = None,
                 output_destinations: dict | None = None) -> dict:
    if review["revision"] != expected_revision:
        raise ConflictError("workflow_preview_stale", "复核草稿已更新，请重新打开当前修订后保存预览。")
    if len(content) > MAX_PREVIEW_BYTES:
        raise ValidationError("workflow_preview_too_large", "预览 PNG 超过 50 MiB，请降低图像尺寸后重试。")
    width, height = review["payload"]["width"], review["payload"]["height"]
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG" or getattr(image, "is_animated", False):
                raise ValidationError("workflow_preview_invalid_png", "请提交单帧 PNG 标注预览。")
            if image.size != (width, height):
                raise ValidationError("workflow_preview_dimensions_mismatch", "预览尺寸与当前原图不一致，请重新生成预览。")
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValidationError("workflow_preview_invalid_png", "预览 PNG 无法完整读取，请重新生成。") from error

    confirmed = review["status"] == "decided" and review.get("decision") in {"approved", "background"}
    state = "confirmed" if confirmed and not has_unsaved_changes else "draft"
    digest = sha256(content).hexdigest()
    file_name = f"{review['id']}-r{expected_revision}-{state}-{digest}.png"
    destination = storage.export_path(file_name)
    if destination.name != file_name:
        raise ValidationError("workflow_preview_unsafe", "预览保存位置与受管文件名不一致。")
    temporary_name = None
    try:
        if not destination.exists():
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as output:
                temporary_name = output.name
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, destination)
        if not destination.is_file() or sha256(destination.read_bytes()).hexdigest() != digest:
            raise ConflictError("workflow_preview_changed", "受管预览文件已改变，请重新生成预览。")
    except OSError as error:
        raise ValidationError("workflow_preview_write_failed", "无法保存预览，请检查输出目录的可用空间和写入权限。") from error
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    saved_path = destination
    if project_root is not None:
        from app.workflow.projects import publish_run_export

        saved_path = publish_run_export(project_root, destination, "images", output_destinations or {})
    return {"file_name": file_name, "saved_path": str(saved_path), "width": width, "height": height,
            "review_revision": expected_revision, "preview_state": state,
            "download_url": f"/api/workflows/reviews/{review['id']}/preview/{file_name}"}


def preview_download(storage: Storage, review_id: str, file_name: str) -> Path:
    match = re.fullmatch(rf"{re.escape(review_id)}-r[1-9][0-9]*-(?:draft|confirmed)-([0-9a-f]{{64}})\.png", file_name)
    if match is None:
        raise ValidationError("workflow_preview_unsafe", "预览文件不属于当前图像复核。")
    path = storage.export_path(file_name)
    if path.name != file_name:
        raise ValidationError("workflow_preview_unsafe", "预览保存位置与受管文件名不一致。")
    try:
        if not path.is_file():
            raise NotFoundError("workflow_preview_missing", "预览文件不存在，请重新生成。")
        if path.stat().st_size > MAX_PREVIEW_BYTES or sha256(path.read_bytes()).hexdigest() != match.group(1):
            raise ConflictError("workflow_preview_changed", "受管预览文件已改变，请重新生成预览。")
    except OSError as error:
        raise ValidationError("workflow_preview_read_failed", "无法读取受管预览，请检查输出目录。") from error
    return path
