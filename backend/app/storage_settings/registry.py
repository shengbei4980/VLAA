from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile

from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.storage_settings.bootstrap import CONFIG_LOCK, read_config, write_config


def absolute_directory(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValidationError("storage_absolute_path_required", "Choose an absolute directory path.")
    if len(str(path)) > 4096:
        raise ValidationError("storage_path_too_long", "The selected path is too long.")
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise ValidationError("storage_directory_required", "The selected path is a file.")
    return path


def validate_directory(raw: str, *, create=False) -> dict:
    path = absolute_directory(raw)
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        probe_root = path if create else ancestor
        with tempfile.TemporaryFile(dir=probe_root) as probe:
            probe.write(b"storage-probe")
            probe.flush()
            os.fsync(probe.fileno())
        free = shutil.disk_usage(probe_root).free
    except OSError as exc:
        raise ValidationError("storage_directory_unwritable", "The selected directory cannot be written.") from exc
    return {"resolved_path": str(path), "writable": True, "free_bytes": free}


class DestinationRegistry:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    def list(self) -> list[dict]:
        return read_config(self.project_root).get("destinations", [])

    def register(self, name: str, path: str) -> dict:
        if not name.strip():
            raise ValidationError("storage_destination_name_required", "Give the output destination a name.")
        validated = validate_directory(path, create=True)
        with CONFIG_LOCK:
            config = read_config(self.project_root)
            existing = next((item for item in config.get("destinations", []) if item["path"] == validated["resolved_path"]), None)
            if existing:
                return existing
            destination = {"id": new_id("destination"), "name": name.strip(), "path": validated["resolved_path"],
                           "created_at": datetime.now(timezone.utc).isoformat()}
            config.setdefault("destinations", []).append(destination)
            write_config(self.project_root, config)
            return destination

    def outputs(self, options: dict | None, *, run_id: str | None = None) -> dict:
        options = options or {}
        destinations = {item["id"]: item for item in self.list()}
        resolved = {"include_images": options.get("include_images", True)}
        for kind in ("annotations", "models", "evaluations"):
            destination_id = options.get(kind + "_destination_id") or options.get("destination_id")
            destination = destinations.get(destination_id)
            if destination_id and destination is None:
                raise NotFoundError("storage_destination_not_found", "The task profile references an unregistered output destination.")
            path = absolute_directory(destination["path"]) if destination else None
            if path is not None and path != Path(destination["path"]):
                raise ValidationError("storage_destination_changed", "The registered destination changed through a link or junction.")
            entry = {"destination_id": destination_id, "path": str(path) if path else None,
                     "enabled": options.get(kind, True)}
            if run_id:
                entry["run_path"] = str(path / ("workflow-" + run_id) / kind) if path else None
            resolved[kind] = entry
        return resolved
