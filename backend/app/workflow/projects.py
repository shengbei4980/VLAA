"""Reusable project folders and frozen destinations for workflow deliverables."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.storage_settings.bootstrap import CONFIG_LOCK, read_config, write_config
from app.storage_settings.manual_exports import publish_manual_export
from app.storage_settings.registry import DestinationRegistry, validate_directory


class ProjectRegistry:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    def list(self) -> list[dict[str, str]]:
        with CONFIG_LOCK:
            projects = read_config(self.project_root).get("workflow_projects", [])
            if not isinstance(projects, list) or any(not isinstance(item, dict) for item in projects):
                raise ValidationError("workflow_projects_invalid", "项目目录配置无法读取，请检查工作台存储配置。")
            return [dict(item) for item in projects]

    def register(self, name: str, root_path: str) -> dict[str, str]:
        name = name.strip()
        if not name or len(name) > 255:
            raise ValidationError("workflow_project_name_required", "请填写不超过 255 个字符的项目名称。")
        validated = validate_directory(root_path, create=True)
        with CONFIG_LOCK:
            destination = DestinationRegistry(self.project_root).register(name, validated["resolved_path"])
            projects = self.list()
            existing = next((item for item in projects if Path(item["root_path"]) == Path(destination["path"])), None)
            if existing:
                return self.get(existing["id"])
            project = {
                "id": new_id("wproject"),
                "name": name,
                "root_path": destination["path"],
                "destination_id": destination["id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            config = read_config(self.project_root)
            config["workflow_projects"] = [*projects, project]
            write_config(self.project_root, config)
            return project

    def get(self, project_id: str) -> dict[str, str]:
        with CONFIG_LOCK:
            project = next((item for item in self.list() if item["id"] == project_id), None)
            if project is None:
                raise NotFoundError("workflow_project_not_found", "所选项目不存在，请重新选择项目。")
            output = DestinationRegistry(self.project_root).outputs({"destination_id": project["destination_id"]})
            if output["annotations"]["path"] != project["root_path"]:
                raise ConflictError("workflow_project_changed", "项目根目录已变化，请重新选择并预览项目。")
            validate_directory(project["root_path"])
            return project


def project_export_locations(output_destinations: dict[str, Any]) -> dict[str, str]:
    if not output_destinations.get("project"):
        return {}
    annotations = output_destinations["annotations"]
    directory = annotations.get("run_path")
    if directory is None:
        directory = Path(annotations["path"]) / "workflow-<run-id>" / "annotations"
    return {
        "json_path": str(Path(directory) / "json"),
        "images_path": str(Path(directory) / "images"),
    }


def resolve_run_outputs(project_root: Path, profile_outputs: dict | None, project_id: str | None = None,
                        *, run_id: str | None = None) -> dict[str, Any]:
    registry = DestinationRegistry(project_root)
    if project_id is None:
        return registry.outputs(profile_outputs, run_id=run_id)
    with CONFIG_LOCK:
        project = ProjectRegistry(project_root).get(project_id)
        options = profile_outputs or {}
        project_options = {kind: options.get(kind, True) for kind in ("annotations", "models", "evaluations")}
        project_options.update(destination_id=project["destination_id"], include_images=options.get("include_images", True))
        outputs = registry.outputs(project_options, run_id=run_id)
        outputs["project"] = project
        outputs.update(project_export_locations(outputs))
        return outputs


def publish_run_export(project_root: Path, managed_path: Path, kind: Literal["json", "images", "package"],
                       output_destinations: dict[str, Any]) -> Path:
    if kind not in {"json", "images", "package"}:
        raise ValidationError("workflow_export_kind_invalid", "请选择 JSON、标注预览图或批量导出包。")
    project = output_destinations.get("project")
    if project is None:
        if kind == "package":
            return Path(managed_path).resolve()
        return publish_manual_export(project_root, managed_path, kind)

    annotations = output_destinations.get("annotations", {})
    root = Path(project.get("root_path", ""))
    run_path = Path(annotations.get("run_path") or "")
    if not root.is_absolute() or root.resolve() != root or annotations.get("path") != str(root):
        raise ConflictError("workflow_project_changed", "本次运行的项目根目录已变化，请检查原目录后重试导出。")
    if (not run_path.is_absolute() or run_path.name != "annotations" or run_path.parent.parent != root
            or not run_path.parent.name.startswith("workflow-") or run_path.resolve() != run_path):
        raise ConflictError("workflow_project_output_changed", "本次运行的保存目录已变化，请检查项目目录后重试导出。")
    directory = run_path if kind == "package" else run_path / kind
    return publish_manual_export(project_root, managed_path, "json" if kind == "package" else kind, directory=directory)
