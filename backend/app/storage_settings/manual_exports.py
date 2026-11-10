"""Persistent locations for manually saved annotation JSON and preview images."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.storage_settings.bootstrap import CONFIG_LOCK, read_config, write_config
from app.storage_settings.registry import DestinationRegistry, validate_directory
from app.storage_settings.service import file_hash

ExportKind = Literal["json", "images"]


def _configured_directory(project_root: Path, kind: ExportKind) -> Path | None:
    with CONFIG_LOCK:
        preferences = read_config(project_root).get("manual_exports", {})
        if not isinstance(preferences, dict):
            raise ValidationError("manual_export_settings_invalid", "导出位置配置无法读取，请重新保存导出位置。")
        destination_id = preferences.get(kind + "_destination_id")
        if not destination_id:
            return None
        if not isinstance(destination_id, str):
            raise ValidationError("manual_export_settings_invalid", "导出位置配置无法读取，请重新保存导出位置。")
        try:
            output = DestinationRegistry(project_root).outputs({"annotations_destination_id": destination_id})
        except NotFoundError as error:
            raise ValidationError("manual_export_destination_missing", "已配置的导出位置不存在，请重新保存导出位置。") from error
        return Path(output["annotations"]["path"])


def manual_export_locations(settings) -> dict[str, str]:
    default = (settings.data_dir / "exports").resolve()
    return {kind + "_path": str(_configured_directory(settings.project_root, kind) or default)
            for kind in ("json", "images")}


def save_manual_export_locations(settings, json_path: str, images_path: str) -> dict[str, str]:
    paths = {"json": json_path, "images": images_path}
    # Validate both choices before publishing the new pair of preferences.
    for path in paths.values():
        validate_directory(path, create=True)
    with CONFIG_LOCK:
        registry = DestinationRegistry(settings.project_root)
        json_destination = registry.register("LabelMe JSON", json_path)
        images_destination = registry.register("标注预览 PNG", images_path)
        config = read_config(settings.project_root)
        config["manual_exports"] = {"json_destination_id": json_destination["id"],
                                    "images_destination_id": images_destination["id"]}
        write_config(settings.project_root, config)
    return manual_export_locations(settings)


def publish_manual_export(project_root: Path, managed_path: Path, kind: ExportKind,
                          *, directory: Path | None = None) -> Path:
    if kind not in {"json", "images"}:
        raise ValidationError("manual_export_kind_invalid", "请选择 JSON 或标注预览图的保存位置。")
    source = Path(managed_path).resolve()
    if not source.is_file():
        raise ValidationError("manual_export_source_missing", "待保存的导出文件不存在，请重新生成。")
    directory = Path(directory) if directory is not None else _configured_directory(project_root, kind)
    if directory is None:
        return source
    validated = validate_directory(str(directory), create=True)
    if Path(validated["resolved_path"]) != directory:
        raise ConflictError("manual_export_destination_changed", "导出文件夹发生变化，请重新保存导出位置。")
    target = directory / source.name
    if target.resolve() != target:
        raise ConflictError("manual_export_destination_changed", "导出文件位置发生变化，请选择新的保存文件夹。")
    if target == source:
        return source
    temporary = None
    try:
        digest = file_hash(source)
        if target.exists():
            if not target.is_file() or file_hash(target) != digest:
                raise ConflictError("manual_export_file_conflict", "保存文件夹中已有同名且内容不同的文件，请选择其他文件夹。")
            return target
        if source.stat().st_size > validated["free_bytes"]:
            raise ValidationError("manual_export_space_insufficient", "保存文件夹的可用空间不足。")
        with source.open("rb") as original, tempfile.NamedTemporaryFile(dir=directory, suffix=".export.tmp", delete=False) as output:
            temporary = Path(output.name)
            shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        if file_hash(temporary) != digest:
            raise ConflictError("manual_export_source_changed", "导出文件在复制时发生变化，请重新导出。")
        try:
            # Windows rename refuses an existing target. POSIX link provides the same atomic publication.
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or file_hash(target) != digest:
                raise ConflictError("manual_export_file_conflict", "保存文件夹中已有同名且内容不同的文件，请选择其他文件夹。")
        if target.resolve() != target or file_hash(target) != digest:
            raise ConflictError("manual_export_copy_changed", "导出副本发生变化，请重新导出。")
    except OSError as error:
        raise ValidationError("manual_export_write_failed", "无法写入保存文件夹，请检查路径、权限与可用空间。") from error
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return target
