"""Stage complete workspace copies and switch only the next-start pointer."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3

from sqlalchemy import select
from sqlalchemy.engine import make_url

from app.core.database import Database
from app.core.errors import ConflictError, ValidationError
from app.core.ids import new_id
from app.core.models import ImageItem
from app.core.storage import Storage
from app.storage_settings.bootstrap import CONFIG_LOCK, read_config, write_config
from app.storage_settings.gate import gate_for, pending_restart
from app.storage_settings.registry import DestinationRegistry, absolute_directory, validate_directory

VOLATILE_DIRS = {"logs", "tmp", "matplotlib"}
ACTIVE_TABLES = {
    "training_tasks": ("pending", "running"),
    "model_evaluation_records": ("pending", "running"),
    "auto_annotation_tasks": ("pending", "running"),
    "video_import_tasks": ("pending", "running"),
    "agent_runs": ("pending", "running", "awaiting_approval"),
    "workflow_runs": ("pending", "running", "waiting_task", "awaiting_review", "needs_attention", "cancelling"),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def database_path(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise ValidationError("storage_database_unsupported", "Workspace relocation requires a file-backed SQLite database.")
    return Path(url.database).resolve()


class StorageSettingsService:
    def __init__(self, settings, database: Database):
        self.settings = settings
        self.database = database
        self.project_root = settings.project_root.resolve()
        self.gate = gate_for(self.project_root)
        self.registry = DestinationRegistry(self.project_root)

    def get(self) -> dict:
        config = read_config(self.project_root)
        migration = config.get("migration")
        pending = None
        if pending_restart(self.settings):
            pending = {"data_dir": config["data_dir"], "database_path": str(database_path(config["database_url"]))}
        safe_migration = None
        if migration:
            safe_migration = {key: migration.get(key) for key in ("status", "phase", "error", "counts", "plan_id", "updated_at")}
        return {"current": {"data_dir": str(self.settings.data_dir.resolve()),
                            "database_path": str(database_path(self.settings.database_url)),
                            "source": "environment" if os.getenv("YWA_DATA_DIR") else "project" if config.get("data_dir") else "default"},
                "pending": pending, "restart_required": pending is not None,
                "write_locked": self.gate.exclusive or pending is not None,
                "environment_overrides": {"data_dir": bool(os.getenv("YWA_DATA_DIR")), "database_url": bool(os.getenv("YWA_DATABASE_URL"))},
                "destinations": self.registry.list(), "migration": safe_migration}

    def _plan_path(self, plan_id: str) -> Path:
        if not re.fullmatch(r"migration_[a-f0-9]{32}", plan_id):
            raise ValidationError("storage_plan_invalid", "The migration plan ID is invalid.")
        return self.project_root / ".ywa-storage-plans" / (plan_id + ".json")

    def _validate_target(self, raw: str):
        target = absolute_directory(raw)
        source = self.settings.data_dir.resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValidationError("storage_recursive_target", "Source and destination must be separate directories without a parent-child relationship.")
        if self.project_root == target or self.project_root.is_relative_to(target):
            raise ValidationError("storage_project_overlap", "The data directory cannot contain the project startup configuration.")
        if self.project_root == source or self.project_root.is_relative_to(source):
            raise ValidationError("storage_project_overlap", "The current data directory includes the application project and cannot be relocated through this API.")
        if target.exists() and any(target.iterdir()):
            raise ConflictError("storage_target_not_empty", "Choose a new or empty destination directory.")
        validate_directory(str(target))
        return target

    def _manifest(self) -> list[dict]:
        source = self.settings.data_dir.resolve()
        database = database_path(self.settings.database_url)
        entries = []
        for directory, directories, files in os.walk(source, followlinks=False):
            root = Path(directory)
            if root == source:
                directories[:] = [name for name in directories if name not in VOLATILE_DIRS]
            for name in [*directories, *files]:
                path = root / name
                if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                    raise ValidationError("storage_link_unsupported", "Resolve symbolic links or junctions in managed data before relocation.")
            for name in sorted(files):
                path = root / name
                if path == database or str(path) in {str(database) + "-wal", str(database) + "-shm", str(database) + "-journal"}:
                    continue
                entries.append({"path": path.relative_to(source).as_posix(), "size": path.stat().st_size, "sha256": file_hash(path)})
        return sorted(entries, key=lambda item: item["path"])

    def _active_counts(self, connection) -> dict:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for table, statuses in ACTIVE_TABLES.items():
            if table not in tables:
                continue
            placeholders = ",".join("?" for _ in statuses)
            count = connection.execute(f'SELECT COUNT(*) FROM "{table}" WHERE status IN ({placeholders})', statuses).fetchone()[0]
            if count:
                counts[table] = count
        return counts

    def prepare(self, raw: str) -> dict:
        if pending_restart(self.settings):
            raise ConflictError("storage_restart_pending", "Restart or cancel the pending relocation before preparing another one.")
        if os.getenv("YWA_DATA_DIR") or os.getenv("YWA_DATABASE_URL"):
            raise ConflictError("storage_environment_override", "Remove the data/database environment override before selecting a project-managed location.")
        target = self._validate_target(raw)
        source_db = database_path(self.settings.database_url)
        with closing(sqlite3.connect(source_db)) as connection:
            active = self._active_counts(connection)
        if active:
            raise ConflictError("storage_tasks_active", "Finish or cancel active tasks before relocating workspace data.", details={"active": active})
        manifest = self._manifest()
        size = sum(item["size"] for item in manifest) + source_db.stat().st_size
        required = size + max(64 * 1024 * 1024, size // 10)
        free = validate_directory(str(target))["free_bytes"]
        if free < required:
            raise ValidationError("storage_insufficient_space", "The destination lacks space for the copied data and safety margin.")
        plan_id = new_id("migration")
        plan = {"plan_id": plan_id, "source_dir": str(self.settings.data_dir.resolve()), "target_dir": str(target),
                "source_database_path": str(source_db), "database_path": str(target / "yolowebagent-starter.db"),
                "required_bytes": required, "free_bytes": free, "file_count": len(manifest), "source_bytes": size,
                "manifest": manifest, "restart_required": True, "excluded_runtime_directories": sorted(VOLATILE_DIRS)}
        path = self._plan_path(plan_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        self._phase("planned", "prepared", plan_id, counts={"files": len(manifest), "bytes": size})
        return {key: value for key, value in plan.items() if key not in {"manifest", "source_database_path"}}

    def _phase(self, status, phase, plan_id, *, counts=None, error=None):
        with CONFIG_LOCK:
            config = read_config(self.project_root)
            config["migration"] = {"status": status, "phase": phase, "plan_id": plan_id, "counts": counts or {},
                                   "error": error, "updated_at": datetime.now(timezone.utc).isoformat()}
            write_config(self.project_root, config)

    def _confirmed_images(self):
        from app.dataset.annotation.review import review_detail
        valid = defaultdict(list)
        storage = Storage(self.settings.data_dir, self.settings.import_root)
        with self.database.session_factory() as session:
            for image in session.scalars(select(ImageItem).where(ImageItem.review_status.in_(["confirmed", "confirmed_empty"]))):
                if review_detail(session, storage, image.id)["receipt"]:
                    valid[image.dataset_id].append(image.id)
        return dict(valid)

    def execute(self, plan_id: str) -> dict:
        path = self._plan_path(plan_id)
        if not path.is_file():
            raise ValidationError("storage_plan_missing", "Prepare a destination before starting relocation.")
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan["source_dir"] != str(self.settings.data_dir.resolve()) or plan["source_database_path"] != str(database_path(self.settings.database_url)):
            raise ConflictError("storage_plan_stale", "The running workspace differs from this migration plan.")
        if pending_restart(self.settings):
            raise ConflictError("storage_restart_pending", "Restart or cancel the pending relocation first.")
        target = self._validate_target(plan["target_dir"])
        stage = target.with_name("." + target.name + ".ywa-" + plan_id)
        source = self.settings.data_dir.resolve()
        source_db = database_path(self.settings.database_url)
        staged_database = stage / "yolowebagent-starter.db"
        published = False
        with self.gate.relocate():
            lock = sqlite3.connect(source_db, timeout=5)
            try:
                lock.execute("BEGIN IMMEDIATE")
                active = self._active_counts(lock)
                if active:
                    raise ConflictError("storage_tasks_active", "Active tasks must finish before relocation.", details={"active": active})
                manifest = self._manifest()
                if manifest != plan["manifest"]:
                    raise ConflictError("storage_plan_stale", "Workspace assets changed after the destination preview. Prepare a fresh plan.")
                if validate_directory(str(target))["free_bytes"] < plan["required_bytes"]:
                    raise ValidationError("storage_insufficient_space", "Available destination space changed after preparation.")
                confirmed = self._confirmed_images()
                stage.parent.mkdir(parents=True, exist_ok=True)
                stage.mkdir()
                self._phase("running", "copying", plan_id, counts={"files": len(manifest), "bytes": plan["source_bytes"]})
                for item in manifest:
                    source_path = source / item["path"]
                    destination = stage / item["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination)
                    if file_hash(destination) != item["sha256"]:
                        raise ConflictError("storage_copy_changed", "A copied business asset failed its checksum check.")
                self._phase("running", "database_backup", plan_id)
                with closing(sqlite3.connect(source_db)) as backup_source, closing(sqlite3.connect(staged_database)) as backup_target:
                    backup_source.backup(backup_target)
                if self._manifest() != manifest:
                    raise ConflictError("storage_source_changed", "A managed asset changed while it was being copied.")
                self._phase("running", "rebasing_and_verifying", plan_id)
                from app.storage_settings.rebase import transform_copied_data
                transformed = transform_copied_data(stage, source, target, staged_database)
                with closing(sqlite3.connect(staged_database)) as connection:
                    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchall():
                        raise ValidationError("storage_database_invalid", "The copied database failed its integrity checks.")
                self._verify_reviews(stage, staged_database, confirmed)
                receipt = {"version": 1, "plan_id": plan_id, "source_dir": str(source), "target_dir": str(target),
                           "source_files": len(manifest), "source_bytes": plan["source_bytes"],
                           "database_sha256": file_hash(staged_database), "transforms": transformed,
                           "confirmed_images_verified": sum(len(images) for images in confirmed.values()),
                           "excluded_runtime_directories": sorted(VOLATILE_DIRS)}
                (stage / "storage-migration-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                if target.exists():
                    if any(target.iterdir()):
                        raise ConflictError("storage_target_changed", "The destination became nonempty during relocation.")
                    target.rmdir()
                stage.replace(target)
                published = True
                with CONFIG_LOCK:
                    config = read_config(self.project_root)
                    previous_destinations = config.get("destinations", [])
                    config["destinations"] = [{**item, "path": str(target / Path(item["path"]).resolve().relative_to(source))
                        if Path(item["path"]).resolve().is_relative_to(source) else item["path"]} for item in previous_destinations]
                    config.update(data_dir=str(target), database_url=f"sqlite:///{target / 'yolowebagent-starter.db'}")
                    config["migration"] = {"status": "ready", "phase": "restart_required", "plan_id": plan_id,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "previous_data_dir": str(source), "previous_database_url": self.settings.database_url,
                        "previous_destinations": previous_destinations,
                        "counts": {"files": len(manifest), "bytes": plan["source_bytes"], "confirmed_images": receipt["confirmed_images_verified"]}, "error": None}
                    write_config(self.project_root, config)
            except Exception as exc:
                code = getattr(exc, "error_code", type(exc).__name__)
                try:
                    self._phase("failed", "failed", plan_id, error=code)
                except OSError:
                    pass
                raise
            finally:
                lock.rollback()
                lock.close()
                if (not published and stage.exists() and stage.resolve().parent == target.parent.resolve()
                        and stage.resolve().is_relative_to(target.parent.resolve())
                        and stage.name == "." + target.name + ".ywa-" + plan_id
                        and not stage.is_symlink() and not getattr(stage, "is_junction", lambda: False)()):
                    try:
                        shutil.rmtree(stage)
                    except OSError:
                        # A retained staging copy never changes the active startup pointer.
                        self._phase("failed", "cleanup_pending", plan_id, counts={"cleanup_pending": True})
        return self.get()

    def _verify_reviews(self, stage, staged_database, confirmed):
        from app.dataset.annotation.review import require_confirmed
        target_database = Database(f"sqlite:///{staged_database}")
        try:
            with target_database.session_factory() as session:
                storage = Storage(stage, stage / "imports")
                for dataset_id, image_ids in confirmed.items():
                    require_confirmed(session, storage, dataset_id, image_ids)
        finally:
            target_database.dispose()

    def cancel(self) -> dict:
        if self.gate.exclusive:
            raise ConflictError("storage_migration_in_progress", "Wait for the current copy to finish before cancelling its pending activation.")
        with CONFIG_LOCK:
            config = read_config(self.project_root)
            migration = config.get("migration") or {}
            if migration.get("status") == "ready" and pending_restart(self.settings):
                config["data_dir"] = str(self.settings.data_dir.resolve())
                config["database_url"] = self.settings.database_url
                config["destinations"] = migration.get("previous_destinations", config.get("destinations", []))
                migration = {**migration, "status": "cancelled", "phase": "cancelled", "error": None}
                config["migration"] = migration
                write_config(self.project_root, config)
        return self.get()
