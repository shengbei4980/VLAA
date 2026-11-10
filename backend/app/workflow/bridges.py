"""Narrow bindings to existing domain services; no arbitrary tool execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.core.models import ModelEvaluationRecord, ModelVersion, TrainingTask
from app.core.storage import Storage
from app.workflow.contracts import Handler


@dataclass
class WorkflowBindings:
    handlers: dict[str, Handler] = field(default_factory=dict)
    confirm_review: Callable[[Session, dict, dict], dict] | None = None
    create_snapshot: Callable[[Session, dict], dict] | None = None
    cancel_task: Callable[[str, str], None] | None = None
    sam_resolver: Callable[[], dict] | None = None


class DomainBridges:
    def __init__(self, session_factory, storage: Storage, bindings: WorkflowBindings):
        self.session_factory = session_factory
        self.storage = storage
        self.bindings = bindings

    def confirm(self, session: Session, context: dict, decision: dict) -> dict:
        if self.bindings.confirm_review:
            return self.bindings.confirm_review(session, context, decision)
        from app.dataset.annotation.review import confirm_workflow_review
        return confirm_workflow_review(session, self.storage, context, decision)

    def snapshot(self, context: dict) -> dict:
        with self.session_factory() as session:
            if self.bindings.create_snapshot:
                result = self.bindings.create_snapshot(session, context)
            else:
                from app.dataset.snapshots import create_snapshot
                inputs = context["inputs"]
                result = create_snapshot(
                    session, self.storage, inputs["dataset_id"], image_ids=inputs.get("confirmed_image_ids"),
                    rules=context["profile"].get("rules", {}), snapshot_id=inputs.get("snapshot_id"),
                )
            session.commit()
            return {**result, "snapshot_id": result.get("id", result.get("snapshot_id"))}

    def task(self, kind: str, context: dict, task_id: str) -> dict:
        if kind in self.bindings.handlers:
            return self.bindings.handlers[kind]({**context, "task_id": task_id})
        if kind == "train":
            return self._training(context, task_id)
        return self._evaluation(context, task_id)

    def _training(self, context: dict, task_id: str) -> dict:
        from app.models.service import ModelService
        from app.training.runtime.queue import training_queue
        from app.training.schemas import TrainingTaskCreate
        from app.training.service import TrainingService
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                profile = context["profile"]
                values = {**profile.get("training", {}), **context.get("params", {})}
                values.update(dataset_id=context["inputs"]["dataset_id"], task_type=profile["task_type"])
                values.setdefault("model", profile.get("models", {}).get("training", "yolo11n-seg.pt" if profile["task_type"] == "segment" else "yolo11n.pt"))
                TrainingService(self.session_factory, self.storage, training_queue).create_task(
                    session, TrainingTaskCreate.model_validate(values), task_id=task_id,
                    snapshot_id=self.output_value(context, "snapshot_id"),
                )
                task = session.get(TrainingTask, task_id)
            if task.status in {"failed", "stopped"}:
                raise ValidationError("workflow_training_failed", task.error_message or "Training did not complete.")
            if task.status != "completed":
                if task.status == "pending":
                    training_queue.submit(task.id)
                return {"status": "waiting_task", "task_id": task_id, "task_kind": "train"}
            versions = ModelService(self.storage).register_training_artifacts(session, task)
            best = next((model for model in versions if model.artifact_type == "best"), None)
            if best is None:
                best = session.scalar(select(ModelVersion).where(ModelVersion.training_task_id == task_id, ModelVersion.artifact_type == "best"))
            if best is None or not self.storage.managed_model_path(best.model_path).is_file():
                raise ValidationError("workflow_model_unregistered", "Completed training has no registered best model.")
            session.commit()
            return {"status": "completed", "task_id": task_id, "model_id": best.id,
                    "snapshot_id": self.output_value(context, "snapshot_id"), "metrics": dict(task.metrics_json or {})}

    def _evaluation(self, context: dict, task_id: str) -> dict:
        from app.models.evaluation import YoloEvaluationRunner
        from app.models.service import ModelService
        with self.session_factory() as session:
            record = session.get(ModelEvaluationRecord, task_id)
            if record is None:
                record = ModelService(self.storage).create_evaluation(
                    session, self.output_value(context, "model_id"), "val", 0.25, 0.5,
                    evaluation_id=task_id, snapshot_id=self.output_value(context, "snapshot_id"),
                )
            if record.status == "pending":
                YoloEvaluationRunner(self.session_factory, self.storage).start_task(record.id)
            if record.status in {"failed", "stopped", "cancelled"}:
                raise ValidationError("workflow_evaluation_failed", record.error_message or "Evaluation did not complete.")
            if record.status != "completed":
                return {"status": "waiting_task", "task_id": task_id, "task_kind": "evaluate"}
            metrics = dict((record.result_json or {}).get("metrics", {}))
            for metric, threshold in context["profile"].get("quality_conditions", {}).items():
                value = metrics.get(metric)
                if not isinstance(value, (int, float)) or value < threshold:
                    raise ValidationError("workflow_quality_condition_failed", f"Evaluation does not meet the configured {metric} condition.")
            return {"status": "completed", "evaluation_id": record.id, "metrics": metrics,
                    "model_id": record.model_id, "snapshot_id": self.output_value(context, "snapshot_id")}

    @staticmethod
    def output_value(context: dict, key: str):
        for output in reversed(list(context.get("outputs", {}).values())):
            if output.get(key) is not None:
                return output[key]
        value = context["inputs"].get(key)
        if value is None:
            raise ValidationError("workflow_missing_input", f"Workflow input {key} is unavailable.")
        return value

    def cancel(self, kind: str, task_id: str) -> None:
        if self.bindings.cancel_task:
            self.bindings.cancel_task(kind, task_id)
            return
        from app.training.runtime.process_registry import process_registry
        if kind == "train":
            from app.training.runtime.queue import training_queue
            from app.training.service import TrainingService
            with self.session_factory() as session:
                if session.get(TrainingTask, task_id):
                    TrainingService(self.session_factory, self.storage, training_queue).stop_task(session, task_id)
        elif kind == "evaluate":
            from app.models.service import ModelService
            with self.session_factory() as session:
                record = session.get(ModelEvaluationRecord, task_id)
                if record and record.status in {"pending", "running"}:
                    ModelService(self.storage).stop_evaluation(session, task_id)
