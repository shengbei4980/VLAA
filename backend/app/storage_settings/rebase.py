"""Rebase a verified staging copy without opening or modifying the source tree."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickletools
import posixpath
import shlex
import shutil
import sqlite3
import struct
import tempfile
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import yaml

from app.core.errors import ValidationError
from app.workflow.artifacts import canonical_bytes, fingerprint

_PATH_KEYS = {
    "root", "path", "data", "project", "model", "model_name", "base_model", "training", "save_dir", "resume",
    "train", "val", "test", "workflow_manifest", "workflow_data_dir", "directory", "destination", "best", "last", "onnx", "data_yaml",
}
_AUDIT_KEYS = {"provenance_json", "rules", "rules_json", "receipt", "receipt_json", "decision_json", "reason", "message",
               "evidence", "source_file", "description", "notes", "log_summary", "api_key", "authorization", "password", "secret"}
_HASH_KEYS = {"sha256", "weights_sha256", "model_sha256", "checkpoint_sha256", "artifact_sha256", "receipt_sha256"}
_PATH_COLUMNS = {
    "training_profiles": ("model_name",),
    "training_tasks": ("model_name", "model_path", "export_path", "data_yaml_path", "run_dir", "logs_path", "summary_path", "best_model_path", "last_model_path"),
    "model_versions": ("model_path", "base_model"),
    "model_test_records": ("image_path",),
    "model_evaluation_records": ("export_path", "data_path", "run_dir", "logs_path"),
    "auto_annotation_tasks": ("logs_path",),
    "video_import_tasks": ("logs_path",),
    "dataset_snapshots": ("manifest_path",),
}
_JSON_COLUMNS = {
    "training_tasks": ("config_json", "command_args_json", "export_stats_json", "metrics_json"),
    "model_versions": ("metrics_json",), "model_test_records": ("result_json",),
    "model_evaluation_records": ("result_json",), "video_import_tasks": ("config_json", "video_info_json"),
    "annotation_drafts": ("payload_json",), "dataset_snapshots": ("summary_json",),
    "workflow_task_profiles": ("config_json",), "workflow_templates": ("nodes_json",),
    "workflow_runs": ("template_snapshot", "profile_snapshot", "context_json"),
    "workflow_items": ("context_json",), "workflow_attempts": ("input_json", "output_json"),
    "workflow_events": ("payload_json",),
    "agent_sessions": ("context_json",), "agent_runs": ("context_json",), "agent_approvals": ("payload_json",),
}


class _Rebaser:
    def __init__(self, source: Path, target: Path):
        self.source = source.as_posix().rstrip("/")
        self.target = target
        self.hashes: dict[str, str] = {}
        self.checkpoint_sizes: dict[str, int] = {}
        self.artifact_names: dict[str, str] = {}

    def path(self, value: str) -> str:
        normalized = posixpath.normpath(value.replace("\\", "/"))
        prefix = self.source
        if normalized.casefold() != prefix.casefold() and not normalized.casefold().startswith(prefix.casefold() + "/"):
            prefix = self.target.as_posix().rstrip("/")
            if not normalized.casefold().startswith(prefix.casefold() + "/"):
                return value
            relative = normalized[len(prefix):].lstrip("/")
            if relative not in self.artifact_names:
                return value
        relative = normalized[len(prefix):].lstrip("/")
        relative = self.artifact_names.get(relative, relative)
        return str(self.target / Path(relative))

    def argument(self, value: str) -> str:
        if "=" in value:
            key, content = value.split("=", 1)
            if self.path_key(key):
                return key + "=" + self.path(content)
        return self.path(value)

    @staticmethod
    def path_key(key: str) -> bool:
        return key in _PATH_KEYS or key.endswith(("_path", "_dir", "_root"))

    def tree(self, value, key: str = ""):
        if key in _AUDIT_KEYS:
            return value
        if isinstance(value, dict):
            if key in {"artifacts", "checkpoints"}:
                return {child: self.path(item) if isinstance(item, str) else self.tree(item, child) for child, item in value.items()}
            changed = {child: self.tree(item, child) for child, item in value.items()}
            if value.get("weights_sha256") in self.checkpoint_sizes:
                changed["weights_size_bytes"] = self.checkpoint_sizes[value["weights_sha256"]]
            return changed
        if isinstance(value, list):
            if key in {"command_args_json", "command_args", "args"}:
                return [self.argument(item) if isinstance(item, str) else self.tree(item) for item in value]
            return [self.tree(item, key) for item in value]
        if isinstance(value, str):
            if key in _HASH_KEYS:
                return self.hashes.get(value, value)
            if self.path_key(key):
                return self.path(value)
        return value


def _hash_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _rewrite_pickle(data: bytes, rebaser: _Rebaser) -> bytes:
    """Edit only protocol-2 Unicode literals. Never run pickle reconstruction code."""
    if len(data) > 64 * 1024 * 1024:
        raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint metadata exceeds the supported inspection limit.")
    try:
        operations = list(pickletools.genops(data))
    except (ValueError, UnicodeError) as exc:
        raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint metadata is not a supported pickle stream.") from exc
    if not operations or operations[0][0].name != "PROTO" or operations[0][1] != 2:
        raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint migration currently supports verified pickle protocol 2 only.")
    if operations[-1][0].name != "STOP" or operations[-1][2] + 1 != len(data):
        raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint metadata has an unsupported trailing payload.")
    chunks = []
    for index, (opcode, argument, position) in enumerate(operations):
        if opcode.proto > 2 or opcode.name == "GLOBAL" and str(argument).startswith("pathlib "):
            raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint metadata uses an unsupported path or pickle representation.")
        end = operations[index + 1][2] if index + 1 < len(operations) else len(data)
        original = data[position:end]
        if isinstance(argument, str):
            changed = rebaser.argument(argument)
            if changed != argument:
                if opcode.name != "BINUNICODE":
                    raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint path is not a supported Unicode literal.")
                encoded = changed.encode("utf-8", "surrogatepass")
                original = b"X" + struct.pack("<I", len(encoded)) + encoded
        chunks.append(original)
    return b"".join(chunks)


def _rewrite_checkpoint(path: Path, rebaser: _Rebaser) -> bool:
    temporary = None
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.endswith("/data.pkl") or name == "data.pkl"]
            if len(names) != len(set(names)) or len(metadata) != 1:
                raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint must contain one unambiguous PyTorch metadata record.")
            info = archive.getinfo(metadata[0])
            if info.file_size > 64 * 1024 * 1024:
                raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint metadata exceeds the supported inspection limit.")
            original = archive.read(info)
            changed = _rewrite_pickle(original, rebaser)
            if changed == original:
                return False
            before = _hash_file(path)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as output:
                temporary = Path(output.name)
            with ZipFile(temporary, "w", allowZip64=True) as target:
                target.comment = archive.comment
                for member in archive.infolist():
                    member = copy.copy(member)
                    if member.filename == info.filename:
                        target.writestr(member, changed)
                    else:
                        with archive.open(member.filename) as source, target.open(member, "w", force_zip64=member.file_size >= 2**31) as destination:
                            shutil.copyfileobj(source, destination, 1024 * 1024)
            with ZipFile(temporary) as target:
                for name in names:
                    if name == info.filename:
                        continue
                    with archive.open(name) as source, target.open(name) as copied:
                        if hashlib.file_digest(source, "sha256").digest() != hashlib.file_digest(copied, "sha256").digest():
                            raise ValidationError("checkpoint_tensor_changed", "Checkpoint storage bytes changed during metadata migration.")
        os.replace(temporary, path)
        rebaser.hashes[before] = _hash_file(path)
        rebaser.checkpoint_sizes[before] = path.stat().st_size
        return True
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise ValidationError("checkpoint_rebase_unsupported", "Checkpoint could not be safely rebased as a PyTorch ZIP archive.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _rewrite_database(connection: sqlite3.Connection, rebaser: _Rebaser) -> int:
    rewritten = 0
    for table in _PATH_COLUMNS.keys() | _JSON_COLUMNS.keys():
        columns = _columns(connection, table)
        selected = [column for column in (*_PATH_COLUMNS.get(table, ()), *_JSON_COLUMNS.get(table, ())) if column in columns]
        if not selected:
            continue
        for row in connection.execute(f'SELECT rowid AS migration_rowid,* FROM "{table}"').fetchall():
            values = {}
            for column in selected:
                original = row[column]
                if original is None:
                    continue
                if column in _JSON_COLUMNS.get(table, ()):
                    parsed = json.loads(original)
                    changed = rebaser.tree(parsed, column)
                    if changed != parsed:
                        values[column] = json.dumps(changed, ensure_ascii=False, allow_nan=False)
                else:
                    changed = rebaser.path(original)
                    if changed != original:
                        values[column] = changed
            if table == "training_tasks" and ("command_args_json" in values or "command_preview" in columns and row["command_preview"]):
                arguments = json.loads(values.get("command_args_json", row["command_args_json"]))
                preview = shlex.join(arguments)
                if preview != row["command_preview"]:
                    values["command_preview"] = preview
            if values:
                assignments = ",".join(f'"{name}"=?' for name in values)
                connection.execute(f'UPDATE "{table}" SET {assignments} WHERE rowid=?', [*values.values(), row["migration_rowid"]])
                rewritten += 1
    return rewritten


def _verify_snapshots(connection: sqlite3.Connection, stage: Path) -> None:
    if not _columns(connection, "dataset_snapshots"):
        return
    from app.core.storage import Storage
    from app.dataset.annotation.masks import MaskService

    masks = MaskService(Storage(stage, stage / "imports"))
    mask_sizes = {}
    for row in connection.execute("SELECT id,fingerprint FROM dataset_snapshots"):
        root = (stage / "snapshots" / row["id"]).resolve()
        if not root.is_relative_to(stage):
            raise ValidationError("storage_snapshot_invalid", "A copied snapshot escapes managed storage.")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if fingerprint(manifest) != row["fingerprint"]:
            raise ValidationError("storage_snapshot_invalid", "A copied snapshot failed its existing content fingerprint.")
        for image in manifest["images"]:
            path = (root / image["path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or _hash_file(path) != image["image_hash"]:
                raise ValidationError("storage_snapshot_invalid", "A copied frozen image failed its existing checksum.")
            references = [image.get("roi_mask_ref"), image.get("ignore_mask_ref"), *[item.get("mask_ref") for item in image["annotations"]]]
            for reference in references:
                if reference is None:
                    continue
                if reference not in mask_sizes:
                    mask_sizes[reference] = masks.read(reference)["size"]
                if mask_sizes[reference] != [image["height"], image["width"]]:
                    raise ValidationError("storage_snapshot_invalid", "A copied frozen mask has incompatible image dimensions.")


def _rewrite_artifacts(connection: sqlite3.Connection, stage: Path, rebaser: _Rebaser) -> int:
    if not _columns(connection, "workflow_artifacts"):
        return 0
    pending = {row["id"]: row for row in connection.execute("SELECT * FROM workflow_artifacts")}
    old_names, new_names = set(), set()
    rewritten = 0
    while pending:
        ready = [identity for identity, row in pending.items() if not any(parent in pending for parent in json.loads(row["parent_ids_json"]))]
        if not ready:
            raise ValidationError("storage_artifact_cycle", "Workflow artifacts contain a cyclic dependency.")
        for identity in ready:
            row = pending.pop(identity)
            original = json.loads(row["payload_json"])
            old_path = stage / "workflows" / "artifacts" / row["storage_name"]
            if not old_path.resolve().is_relative_to(stage) or not old_path.is_file() or _hash_file(old_path) != row["sha256"] or fingerprint(original) != row["sha256"]:
                raise ValidationError("storage_artifact_invalid", "A copied workflow artifact failed its source integrity check.")
            changed = rebaser.tree(original)
            content = canonical_bytes(changed)
            digest = hashlib.sha256(content).hexdigest()
            name = digest + ".json"
            destination = stage / "workflows" / "artifacts" / name
            destination.write_bytes(content)
            old_names.add(row["storage_name"])
            new_names.add(name)
            rebaser.hashes[row["sha256"]] = digest
            rebaser.artifact_names["workflows/artifacts/" + row["storage_name"]] = "workflows/artifacts/" + name
            if digest != row["sha256"]:
                connection.execute("UPDATE workflow_artifacts SET payload_json=?,sha256=?,storage_name=? WHERE id=?",
                                   (json.dumps(changed, ensure_ascii=False), digest, name, identity))
                rewritten += 1
    for name in old_names - new_names:
        (stage / "workflows" / "artifacts" / name).unlink()
    return rewritten


def _rewrite_files(stage: Path, rebaser: _Rebaser, bundles: list[tuple[Path, str]]) -> int:
    rewritten = 0
    for path in stage.rglob("*"):
        if not path.resolve().is_relative_to(stage):
            raise ValidationError("storage_rebase_scope", "Staging metadata must not follow links outside the copied tree.")
        if not path.is_file():
            continue
        relative = path.relative_to(stage)
        if relative.parts[0] == "masks" or relative.parts[:2] == ("workflows", "artifacts"):
            continue
        if relative.parts[0] not in {"runs", "models", "snapshots", "workflows", "video-imports"} and relative != Path("settings.json") and not any(path.is_relative_to(receipt.parent) for receipt, _ in bundles):
            continue
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            original = path.read_text(encoding="utf-8")
            data = json.loads(original) if path.suffix.lower() == ".json" else yaml.safe_load(original)
            changed = rebaser.tree(data)
            if changed != data:
                content = json.dumps(changed, ensure_ascii=False, allow_nan=False, indent=2) if path.suffix.lower() == ".json" else yaml.safe_dump(changed, allow_unicode=True, sort_keys=False)
                path.write_text(content, encoding="utf-8")
                rewritten += 1
        elif path.name in {"train.txt", "val.txt", "test.txt"}:
            original = path.read_text(encoding="utf-8")
            changed = "\n".join(rebaser.path(line) for line in original.splitlines()) + ("\n" if original.endswith("\n") else "")
            if changed != original:
                path.write_text(changed, encoding="utf-8")
                rewritten += 1
    return rewritten


def _published_bundles(stage: Path) -> list[tuple[Path, str]]:
    result = []
    for path in stage.rglob("receipt.json"):
        if not (path.parent / "artifact.json").is_file() or not (path.parent / "package.json").is_file():
            continue
        if not path.resolve().is_relative_to(stage):
            raise ValidationError("storage_rebase_scope", "Output bundles must remain inside the copied tree.")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError("storage_output_changed", "A copied published bundle has an unreadable receipt.") from exc
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 1 or not receipt.get("artifact_id") or not isinstance(receipt.get("files"), list):
            continue
        for item in receipt["files"]:
            relative = item["path"]
            asset = (path.parent / relative).resolve()
            if Path(relative).is_absolute() or not asset.is_relative_to(path.parent.resolve()) or asset == path.resolve() or not asset.is_file() or _hash_file(asset) != item["sha256"]:
                raise ValidationError("storage_output_changed", "A copied published bundle failed its existing receipt check.")
        result.append((path, _hash_file(path)))
    return result


def _rebuild_bundles(connection: sqlite3.Connection, bundles: list[tuple[Path, str]], rebaser: _Rebaser) -> int:
    for path, old_hash in bundles:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        artifact = connection.execute("SELECT payload_json,sha256 FROM workflow_artifacts WHERE id=?", (receipt["artifact_id"],)).fetchone()
        if artifact is None:
            raise ValidationError("storage_output_artifact_missing", "A published bundle refers to an unavailable workflow artifact.")
        (path.parent / "artifact.json").write_bytes(canonical_bytes(json.loads(artifact["payload_json"])))
        receipt["artifact_sha256"] = artifact["sha256"]
        for item in receipt["files"]:
            asset = (path.parent / item["path"]).resolve()
            item.update(size=asset.stat().st_size, sha256=_hash_file(asset))
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        rebaser.hashes[old_hash] = _hash_file(path)
    return len(bundles)


def transform_copied_data(stage_root: Path, source_root: Path, target_root: Path, target_db_path: Path) -> dict:
    stage, source, target, database = (Path(value).resolve() for value in (stage_root, source_root, target_root, target_db_path))
    if stage == source or stage.is_relative_to(source) or source.is_relative_to(stage) or target == source or target.is_relative_to(source) or source.is_relative_to(target) or not database.is_relative_to(stage):
        raise ValidationError("storage_rebase_scope", "Rebasing requires an independent staging copy and its copied database.")
    rebaser = _Rebaser(source, target)
    bundles = _published_bundles(stage)
    checkpoints = 0
    for path in stage.rglob("*.pt"):
        if not path.resolve().is_relative_to(stage):
            raise ValidationError("storage_rebase_scope", "Checkpoint links must stay inside the copied tree.")
        checkpoints += _rewrite_checkpoint(path, rebaser)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            _verify_snapshots(connection, stage)
            rewritten_rows = _rewrite_database(connection, rebaser)
            artifacts = _rewrite_artifacts(connection, stage, rebaser)
            rewritten_files = _rewrite_files(stage, rebaser, bundles)
            rewritten_files += _rebuild_bundles(connection, bundles, rebaser)
            rewritten_rows += _rewrite_database(connection, rebaser)
            if _columns(connection, "dataset_snapshots"):
                for row in connection.execute("SELECT id,manifest_path,fingerprint FROM dataset_snapshots").fetchall():
                    path = Path(row["manifest_path"])
                    if not path.is_relative_to(target):
                        raise ValidationError("storage_snapshot_path", "A snapshot manifest is outside the migrated root.")
                    manifest = stage / path.relative_to(target)
                    digest = fingerprint(json.loads(manifest.read_text(encoding="utf-8")))
                    connection.execute("UPDATE dataset_snapshots SET fingerprint=? WHERE id=?", (digest, row["id"]))
            if _columns(connection, "workflow_attempts"):
                for row in connection.execute("SELECT id,input_json FROM workflow_attempts").fetchall():
                    connection.execute("UPDATE workflow_attempts SET fingerprint=? WHERE id=?", (fingerprint(json.loads(row["input_json"])), row["id"]))
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ValidationError("storage_database_invalid", "The migrated database failed integrity checks.")
    except (OSError, ValueError, sqlite3.Error, yaml.YAMLError) as exc:
        raise ValidationError("storage_rebase_failed", "Copied runtime metadata could not be safely rebased.") from exc
    finally:
        connection.close()
    return {"rewritten_files": rewritten_files, "rewritten_rows": rewritten_rows, "artifacts": artifacts, "checkpoints": checkpoints,
            "published_bundles": len(bundles), "content_hash_mapping": rebaser.hashes}
