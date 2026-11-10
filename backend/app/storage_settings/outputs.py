"""Copy immutable workflow deliverables to human-registered destinations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from sqlalchemy import exists, select
from sqlalchemy.orm import aliased

from app.core.errors import ConflictError, ValidationError
from app.core.ids import new_id
from app.core.models import ModelEvaluationRecord, ModelVersion
from app.core.storage import Storage
from app.storage_settings.service import file_hash
from app.workflow.models import WorkflowArtifact, WorkflowEvent, WorkflowItem, WorkflowRun

CATEGORIES = {"proposals": "annotations", "review_proposals": "annotations",
              "confirmed_annotations": "annotations", "dataset_snapshot": "annotations", "export": "annotations",
              "model": "models", "evaluation": "evaluations"}


def queue_output(session, run: WorkflowRun, artifact: WorkflowArtifact):
    category = CATEGORIES.get(artifact.kind)
    target = (run.context_json.get("output_destinations") or {}).get(category, {})
    if category and target.get("enabled") and target.get("run_path"):
        session.add(WorkflowEvent(id=new_id("wevent"), run_id=run.id, kind="output_pending",
            payload_json={"artifact_id": artifact.id, "category": category}))


class WorkflowOutputPublisher:
    def __init__(self, session_factory, storage: Storage):
        self.session_factory = session_factory
        self.storage = storage

    def publish(self, run_id: str):
        with self.session_factory() as session:
            published = aliased(WorkflowEvent)
            pending = list(session.scalars(select(WorkflowEvent).where(
                WorkflowEvent.run_id == run_id, WorkflowEvent.kind == "output_pending",
                ~exists(select(published.id).where(published.run_id == run_id, published.kind == "output_published",
                    published.payload_json["intent_id"].as_string() == WorkflowEvent.id)),
            ).order_by(WorkflowEvent.created_at, WorkflowEvent.id)))
        for intent in pending:
            self._publish_one(intent.id)

    def _publish_one(self, intent_id):
        with self.session_factory() as session:
            intent = session.get(WorkflowEvent, intent_id)
            run = session.get(WorkflowRun, intent.run_id)
            artifact = session.get(WorkflowArtifact, intent.payload_json["artifact_id"])
            target = run.context_json["output_destinations"][intent.payload_json["category"]]
            base = Path(target["path"])
            if base.resolve() != base:
                raise ConflictError("workflow_destination_changed", "The registered output directory changed through a link or junction.")
            run_path = Path(target["run_path"])
            if not run_path.is_relative_to(base) or run_path.resolve() != run_path:
                raise ValidationError("workflow_output_unsafe", "The output target escaped its frozen destination.")
            target_dir = run_path / artifact.id
            if target_dir.exists():
                self._verify_existing(target_dir, artifact)
            else:
                run_path.mkdir(parents=True, exist_ok=True)
                staging = Path(tempfile.mkdtemp(prefix=".output-", dir=run_path))
                try:
                    self._write_bundle(session, staging, run, artifact)
                    if target_dir.exists():
                        self._verify_existing(target_dir, artifact)
                    else:
                        staging.replace(target_dir)
                finally:
                    if (staging.exists() and staging.resolve().parent == run_path
                            and staging.resolve().is_relative_to(run_path)
                            and staging.name.startswith(".output-") and not staging.is_symlink()
                            and not getattr(staging, "is_junction", lambda: False)()):
                        shutil.rmtree(staging)
            receipt_path = target_dir / "receipt.json"
            marker_id = "published_" + hashlib.sha256(intent_id.encode()).hexdigest()[:40]
            if session.get(WorkflowEvent, marker_id) is None:
                session.add(WorkflowEvent(id=marker_id, run_id=run.id, kind="output_published",
                    payload_json={"intent_id": intent_id, "artifact_id": artifact.id, "category": intent.payload_json["category"],
                                  "path": str(target_dir), "receipt_sha256": file_hash(receipt_path)}))
                session.commit()

    @staticmethod
    def _verify_existing(directory, artifact):
        receipt_file = directory / "receipt.json"
        if not receipt_file.is_file():
            raise ConflictError("workflow_output_conflict", "An existing output folder has no verification receipt.")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt.get("artifact_id") != artifact.id or receipt.get("artifact_sha256") != artifact.sha256:
            raise ConflictError("workflow_output_conflict", "The output directory belongs to another artifact revision.")
        for item in receipt["files"]:
            path = (directory / item["path"]).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or file_hash(path) != item["sha256"]:
                raise ConflictError("workflow_output_changed", "A published output changed. Choose a new destination instead of overwriting it.")

    def _copy(self, source: Path, destination: Path):
        source = source.resolve()
        if not source.is_relative_to(self.storage.data_dir) or not source.is_file():
            raise ValidationError("workflow_output_source_unsafe", "Published files must come from managed assets.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if file_hash(source) != file_hash(destination):
            raise ConflictError("workflow_output_copy_changed", "An output asset changed during copying.")

    def _mask_refs(self, value):
        refs = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"mask_ref", "roi_mask_ref", "ignore_mask_ref"} and item:
                    refs.add(item)
                elif isinstance(item, (dict, list)):
                    refs.update(self._mask_refs(item))
        elif isinstance(value, list):
            for item in value:
                refs.update(self._mask_refs(item))
        return refs

    def _write_bundle(self, session, directory, run, artifact):
        from app.dataset.annotation.masks import MaskService
        from app.dataset.snapshots import get_snapshot, verify_snapshot_images
        from app.workflow.artifacts import canonical_bytes

        payload = artifact.payload_json
        content = canonical_bytes(payload)
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise ValidationError("workflow_artifact_changed", "The source workflow artifact failed its checksum check.")
        (directory / "artifact.json").write_bytes(content)
        package = {"schema_version": 1, "run_id": run.id, "artifact_id": artifact.id, "kind": artifact.kind,
                   "coordinate_space": "image_pixel", "profile": run.profile_snapshot, "images": []}
        refs = self._mask_refs(payload)
        include_images = run.context_json["output_destinations"].get("include_images", True)
        if artifact.kind == "dataset_snapshot":
            snapshot = get_snapshot(session, self.storage, payload["snapshot_id"])
            verify_snapshot_images(snapshot)
            manifest = snapshot["manifest"]
            (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            refs.update(self._mask_refs(manifest))
            if include_images:
                for image in manifest["images"]:
                    self._copy(Path(snapshot["root"]) / image["path"], directory / image["path"])
        elif artifact.item_id and include_images:
            item = session.get(WorkflowItem, artifact.item_id)
            source = Path(item.context_json["image_path"])
            relative = "images/" + item.subject_ref + source.suffix.lower()
            self._copy(source, directory / relative)
            if file_hash(directory / relative) != item.context_json["image_hash"]:
                raise ConflictError("workflow_output_image_changed", "The source image changed after the workflow was created.")
            package["images"].append({"image_id": item.subject_ref, "path": relative, "sha256": item.context_json["image_hash"]})
        elif artifact.kind == "model":
            model = session.get(ModelVersion, payload.get("model_id"))
            if model is None:
                raise ValidationError("workflow_output_model_missing", "The workflow model is not registered.")
            versions = [model]
            if model.training_task_id:
                versions = list(session.scalars(select(ModelVersion).where(ModelVersion.training_task_id == model.training_task_id,
                    ModelVersion.artifact_type.in_(["best", "last"])))) or versions
            package["models"] = []
            for version in versions:
                source = self.storage.managed_model_path(version.model_path)
                file_name = (version.artifact_type if version.artifact_type in {"best", "last"} else "model") + source.suffix.lower()
                self._copy(source, directory / file_name)
                package["models"].append({"id": version.id, "path": file_name, "task_type": version.task_type,
                    "format": version.format, "training_task_id": version.training_task_id,
                    "class_schema_json": version.class_schema_json, "metrics": version.metrics_json})
        elif artifact.kind == "evaluation":
            evaluation = session.get(ModelEvaluationRecord, payload.get("evaluation_id"))
            if evaluation is None:
                raise ValidationError("workflow_output_evaluation_missing", "The evaluation report is unavailable.")
            report = dict(evaluation.result_json or {})
            copied_artifacts = {}
            if evaluation.logs_path and Path(evaluation.logs_path).is_file():
                self._copy(Path(evaluation.logs_path), directory / "evaluation.log")
            for key, path in (evaluation.result_json or {}).get("artifacts", {}).items():
                if isinstance(path, str) and Path(path).is_file():
                    relative = "figures/" + Path(path).name
                    self._copy(Path(path), directory / relative)
                    copied_artifacts[key] = relative
            report["artifacts"] = copied_artifacts
            report["evaluation_id"] = evaluation.id
            report["model_id"] = evaluation.model_id
            report["split"] = evaluation.split
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        elif artifact.kind == "export":
            self._copy(self.storage.export_path(payload["file_name"]), directory / payload["file_name"])
        for reference in refs:
            source = MaskService(self.storage).path(reference)
            self._copy(source, directory / reference)
        (directory / "package.json").write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        files = [{"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": file_hash(path)}
                 for path in sorted(directory.rglob("*")) if path.is_file()]
        receipt = {"schema_version": 1, "run_id": run.id, "artifact_id": artifact.id,
                   "artifact_sha256": artifact.sha256, "kind": artifact.kind, "files": files}
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
