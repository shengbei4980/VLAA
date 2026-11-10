from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import exists, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased

from app.agent.redaction import redact_text
from app.core.errors import ConflictError, DomainError, NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.core.time import utc_now
from app.dataset.annotation.service import list_annotations
from app.workflow.artifacts import WorkflowArtifactStore, canonical_bytes, fingerprint
from app.workflow.bridges import DomainBridges, WorkflowBindings
from app.workflow.contracts import CONTRACTS, validate_nodes
from app.workflow.models import (
    WorkflowArtifact, WorkflowAttempt, WorkflowEvent, WorkflowItem, WorkflowReview,
    WorkflowRun, WorkflowTaskProfile, WorkflowTemplate,
)
from app.workflow.schemas import (
    NodeDefinition, ProfileCreate, ReviewDecision, ReviewDraft, RunCreate, TemplateCreate,
)
from app.workflow.templates import builtin_templates

logger = logging.getLogger("ywa.workflow")
TERMINAL = {"completed", "cancelled"}
RESERVED_PAYLOAD = {"image_id", "dataset_id", "image_hash", "image_url", "width", "height", "base_revision"}


class WorkflowService:
    def __init__(self, session_factory, storage: Storage, bindings: WorkflowBindings | None = None, *, project_root: Path | None = None):
        self.session_factory = session_factory
        self.storage = storage
        self.bindings = bindings or WorkflowBindings()
        self.bridges = DomainBridges(session_factory, storage, self.bindings)
        self.artifacts = WorkflowArtifactStore(storage)
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.wake = lambda: None

    def list_projects(self) -> list[dict]:
        from app.workflow.projects import ProjectRegistry

        return ProjectRegistry(self.project_root).list()

    def create_project(self, name: str, root_path: str) -> dict:
        from app.workflow.projects import ProjectRegistry

        return ProjectRegistry(self.project_root).register(name, root_path)

    def export_annotation_batch(self, run_id: str, include_previews: bool = True) -> dict:
        from app.workflow.batch_exports import BatchExporter

        return BatchExporter(self).export_run(run_id, include_previews)

    def annotation_batch_download(self, run_id: str, file_name: str) -> Path:
        from app.workflow.batch_exports import batch_download

        with self.session_factory() as session:
            self._row(session, WorkflowRun, run_id)
        return batch_download(self.storage, run_id, file_name)

    def ensure_defaults(self) -> None:
        with self.session_factory() as session:
            for template in builtin_templates():
                existing = session.scalar(select(WorkflowTemplate).where(WorkflowTemplate.key == template.key).order_by(WorkflowTemplate.version.desc()).limit(1))
                if existing is None or existing.nodes_json != [node.model_dump(mode="json") for node in template.nodes]:
                    self._create_template(session, template)
            session.commit()

    @staticmethod
    def _safe_json(value: Any) -> None:
        canonical_bytes(value)
        pending = [value]
        while pending:
            current = pending.pop()
            if isinstance(current, dict):
                for key, item in current.items():
                    if key.lower() in {"api_key", "authorization", "password", "secret", "access_token"}:
                        raise ValidationError("workflow_secret_in_payload", "Credentials must be configured through local model settings.")
                    if isinstance(item, (dict, list)):
                        pending.append(item)
            elif isinstance(current, list):
                pending.extend(item for item in current if isinstance(item, (dict, list)))

    @staticmethod
    def _row(session, model, row_id):
        row = session.get(model, row_id)
        if row is None:
            raise NotFoundError("workflow_object_not_found", f"{model.__name__} was not found.")
        return row

    @staticmethod
    def _profile(row):
        return {"id": row.id, "key": row.key, "name": row.name, "version": row.version,
                "config": row.config_json, "created_at": row.created_at}

    @staticmethod
    def _template(row):
        return {"id": row.id, "key": row.key, "name": row.name, "version": row.version,
                "nodes": row.nodes_json, "created_at": row.created_at}

    def list_profiles(self) -> list[dict]:
        with self.session_factory() as session:
            return [self._profile(row) for row in session.scalars(select(WorkflowTaskProfile).order_by(WorkflowTaskProfile.key, WorkflowTaskProfile.version.desc()))]

    def get_profile(self, profile_id: str) -> dict:
        with self.session_factory() as session:
            return self._profile(self._row(session, WorkflowTaskProfile, profile_id))

    def create_profile(self, payload: ProfileCreate) -> dict:
        self._safe_json(payload.config.model_dump())
        with self.session_factory() as session:
            session.execute(update(WorkflowTaskProfile).where(WorkflowTaskProfile.key == payload.key).values(updated_at=utc_now()))
            version = (session.scalar(select(func.max(WorkflowTaskProfile.version)).where(WorkflowTaskProfile.key == payload.key)) or 0) + 1
            row = WorkflowTaskProfile(id=new_id("wprofile"), key=payload.key, name=payload.name, version=version,
                                      config_json=payload.config.model_dump(mode="json"))
            session.add(row)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ConflictError("workflow_version_conflict", "A profile version was created concurrently. Retry saving.") from exc
            return self._profile(row)

    def list_templates(self) -> list[dict]:
        with self.session_factory() as session:
            return [self._template(row) for row in session.scalars(select(WorkflowTemplate).order_by(WorkflowTemplate.key, WorkflowTemplate.version.desc()))]

    def get_template(self, template_id: str) -> dict:
        with self.session_factory() as session:
            return self._template(self._row(session, WorkflowTemplate, template_id))

    def _create_template(self, session, payload):
        validate_nodes(payload.nodes)
        self._safe_json(payload.model_dump())
        session.execute(update(WorkflowTemplate).where(WorkflowTemplate.key == payload.key).values(updated_at=utc_now()))
        version = (session.scalar(select(func.max(WorkflowTemplate.version)).where(WorkflowTemplate.key == payload.key)) or 0) + 1
        row = WorkflowTemplate(id=new_id("wtemplate"), key=payload.key, name=payload.name, version=version,
                               nodes_json=[node.model_dump(mode="json") for node in payload.nodes])
        session.add(row)
        session.flush()
        return row

    def create_template(self, payload: TemplateCreate) -> dict:
        with self.session_factory() as session:
            row = self._create_template(session, payload)
            session.commit()
            return self._template(row)

    def prepare_run(self, payload: RunCreate, vision_settings: dict) -> dict:
        """Resolve the exact image hashes and public endpoints before user confirmation."""
        self._safe_json(payload.model_dump())
        with self.session_factory() as session:
            template = self._row(session, WorkflowTemplate, payload.template_id)
            profile = self._row(session, WorkflowTaskProfile, payload.profile_id)
            dataset = self._row(session, Dataset, payload.dataset_id)
            if dataset.task_type != profile.config_json["task_type"]:
                raise ValidationError("workflow_task_type_mismatch", "Dataset task type must match the task profile.")
            if dataset.task_type not in {"detect", "segment"} and any(
                node["type"] in {"vision_propose", "managed_inference"} for node in template.nodes_json
            ):
                raise ValidationError("workflow_prelabel_task_unsupported", "Automatic visual labeling currently supports detection and segmentation datasets.")
            scope, hashes = [], set()
            batches = [{"dataset_id": payload.dataset_id, "image_ids": payload.image_ids},
                       *[batch.model_dump() for batch in payload.next_batches]]
            for batch in batches:
                current_hashes = []
                for image_id in batch["image_ids"]:
                    image = self._row(session, ImageItem, image_id)
                    if image.dataset_id != batch["dataset_id"]:
                        raise ValidationError("workflow_image_dataset_mismatch", "Every approved image must belong to its declared dataset.")
                    digest = hashlib.sha256(self.storage.read_image(image.dataset_id, image.storage_name)).hexdigest()
                    current_hashes.append(digest)
                    hashes.add(digest)
                scope.append({**batch, "image_hashes": current_hashes})
            roles = set()
            for node in template.nodes_json:
                if node["type"] == "vision_propose" and (
                    node.get("params", {}).get("source") == "vision" or not payload.inputs.get("model_id")
                ):
                    roles.add("proposer")
                if node["type"] == "vision_review":
                    roles.add("reviewer")
                if node["type"] == "repair_proposals" and profile.config_json.get("repair_limit", 0):
                    roles.add("proposer")
            approvals = {}
            for role in sorted(roles):
                config = vision_settings.get(role) or {}
                if not config.get("enabled") or not config.get("api_base") or not config.get("model"):
                    raise ValidationError("workflow_vision_endpoint_unavailable", f"Configure and enable the {role} vision endpoint before preparing this run.")
                if not hashes:
                    raise ValidationError("workflow_images_required", "Vision workflows require an approved image selection.")
                approvals[role] = {key: config[key] for key in ("mode", "api_base", "model", "config_hash")}
                approvals[role].update(allowed_image_hashes=sorted(hashes), max_requests=min(10000, profile.config_json.get("max_requests", 1000)))
            sam_config = None
            if dataset.task_type == "segment" and any(node["type"] in {"segment_proposals", "repair_proposals"} for node in template.nodes_json):
                if self.bindings.sam_resolver is None:
                    raise ValidationError("workflow_segmenter_unavailable", "SAM configuration preview is unavailable for this workflow.")
                sam_config = self.bindings.sam_resolver()
            request = payload.model_dump(mode="json")
            from app.workflow.projects import resolve_run_outputs
            outputs = resolve_run_outputs(self.project_root, profile.config_json.get("outputs"), payload.project_id)
            request["inputs"] = {**payload.inputs, "vision_approvals": approvals,
                                 **({"sam_config": sam_config} if sam_config is not None else {}),
                                 "approved_output_destinations": outputs}
            return {"request": request, "vision_approvals": approvals, "scope": scope,
                    "sam_config": sam_config, "outputs": outputs, "profile_snapshot": profile.config_json, "template_snapshot": template.nodes_json}

    def create_run(self, payload: RunCreate, *, parent_run_id: str | None = None, frozen_profile: dict | None = None,
                   run_id: str | None = None) -> dict:
        self._safe_json(payload.model_dump())
        request_data = payload.model_dump(mode="json")
        if payload.project_id is None:
            request_data.pop("project_id", None)
        digest = fingerprint(request_data)
        with self.session_factory() as session:
            # A write claim serializes submissions, including the first idempotency check.
            session.execute(update(WorkflowTemplate).where(WorkflowTemplate.id == payload.template_id).values(updated_at=utc_now()))
            if parent_run_id:
                parent = self._row(session, WorkflowRun, parent_run_id)
                if parent.cancel_requested:
                    raise ConflictError("workflow_parent_cancelled", "The parent workflow was cancelled before the next batch started.")
            existing = session.scalar(select(WorkflowRun).where(WorkflowRun.idempotency_key == payload.idempotency_key))
            if existing:
                if existing.fingerprint != digest:
                    raise ConflictError("workflow_idempotency_conflict", "This idempotency key has different inputs.")
                return self._run_response(session, existing)
            template = self._row(session, WorkflowTemplate, payload.template_id)
            profile_row = self._row(session, WorkflowTaskProfile, payload.profile_id)
            profile = dict(frozen_profile or profile_row.config_json)
            dataset = self._row(session, Dataset, payload.dataset_id)
            if dataset.task_type != profile["task_type"]:
                raise ValidationError("workflow_task_type_mismatch", "Dataset task type must match the task profile.")
            nodes = [NodeDefinition.model_validate(value) for value in template.nodes_json]
            validate_nodes(nodes)
            if profile.get("repair_limit", 0) and "repair_proposals" not in self.bindings.handlers:
                raise ValidationError("workflow_repair_unavailable", "Automatic repair was requested but its capability is not configured.")
            if any(node.scope == "item" for node in nodes) and not payload.image_ids:
                raise ValidationError("workflow_images_required", "This template requires a nonempty image selection.")
            classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset.id).order_by(ClassLabel.class_index)))
            if not profile.get("class_rules"):
                profile["class_rules"] = [{"class_id": row.id, "name": row.name} for row in classes]
            profile["class_rules"] = [{**rule, "id": rule.get("class_id", rule.get("id")),
                                       "class_id": rule.get("class_id", rule.get("id"))} for rule in profile["class_rules"]]
            allowed = {row.id for row in classes}
            if any(rule.get("class_id") not in allowed for rule in profile["class_rules"]):
                raise ValidationError("workflow_class_mismatch", "Profile classes must belong to the selected dataset.")
            profile["rules"] = {**profile.get("rules", {}), "classes": profile["class_rules"]}
            context = {**payload.inputs, "dataset_id": dataset.id, "image_ids": payload.image_ids,
                       "project_id": payload.project_id,
                       "snapshot_id": payload.snapshot_id, "next_batches": [batch.model_dump() for batch in payload.next_batches]}
            run = WorkflowRun(id=run_id or new_id("wrun"), idempotency_key=payload.idempotency_key, fingerprint=digest,
                              template_id=template.id, profile_id=profile_row.id,
                              template_snapshot={"key": template.key, "name": template.name, "version": template.version, "nodes": template.nodes_json},
                              profile_snapshot=profile, context_json=context, parent_run_id=parent_run_id)
            from app.workflow.projects import resolve_run_outputs
            approved_outputs = payload.inputs.get("approved_output_destinations")
            resolved_outputs = resolve_run_outputs(self.project_root, profile.get("outputs"), payload.project_id)
            if approved_outputs is not None and approved_outputs != resolved_outputs:
                raise ConflictError("workflow_output_destination_changed", "The registered output destination changed after preparation.")
            run.context_json = {**context, "output_destinations": resolve_run_outputs(
                self.project_root, profile.get("outputs"), payload.project_id, run_id=run.id)}
            session.add(run)
            session.flush()
            for image_id in payload.image_ids:
                image = self._row(session, ImageItem, image_id)
                if image.dataset_id != dataset.id:
                    raise ValidationError("workflow_image_dataset_mismatch", "Every selected image must belong to the dataset.")
                image_hash = hashlib.sha256(self.storage.read_image(dataset.id, image.storage_name)).hexdigest()
                annotations = [annotation.model_dump(mode="json") for annotation in list_annotations(session, image.id)]
                for annotation in annotations:
                    annotation["object_id"] = annotation["id"]
                context = {"image_id": image.id, "dataset_id": dataset.id, "image_hash": image_hash,
                           "image_path": str(self.storage.image_path(dataset.id, image.storage_name)),
                           "image_url": f"/api/images/{image.id}/file", "width": image.width, "height": image.height,
                           "base_revision": getattr(image, "annotation_revision", 0), "existing_annotations": annotations,
                           "roi_mask_ref": image.roi_mask_ref, "ignore_mask_ref": image.ignore_mask_ref}
                session.add(WorkflowItem(id=new_id("witem"), run_id=run.id, subject_ref=image.id, context_json=context))
            self._event(session, run.id, "created", {"template_version": template.version, "profile_version": profile_row.version})
            session.commit()
            result = self._run_response(session, run)
        self.wake()
        return result

    @staticmethod
    def _event(session, run_id, kind, payload=None):
        session.add(WorkflowEvent(id=new_id("wevent"), run_id=run_id, kind=kind, payload_json=payload or {}))

    def _artifact(self, session, run_id, item_id, attempt_id, kind, payload, parents=None):
        self._safe_json(payload)
        storage_name, digest = self.artifacts.write(payload)
        row = WorkflowArtifact(id=new_id("wart"), run_id=run_id, item_id=item_id, attempt_id=attempt_id,
                               kind=kind, sha256=digest, storage_name=storage_name, payload_json=payload,
                               parent_ids_json=parents or [])
        session.add(row)
        session.flush()
        from app.storage_settings.outputs import queue_output
        queue_output(session, self._row(session, WorkflowRun, run_id), row)
        return row

    def _outputs(self, session, run_id, item_id=None):
        query = select(WorkflowAttempt).where(WorkflowAttempt.run_id == run_id, WorkflowAttempt.status == "completed")
        query = query.where(WorkflowAttempt.item_id == item_id) if item_id else query.where(WorkflowAttempt.item_id.is_(None))
        return {row.node_id: row.output_json or {} for row in session.scalars(query.order_by(WorkflowAttempt.created_at, WorkflowAttempt.id))}

    @staticmethod
    def _normalized_payload(payload: dict) -> dict:
        result = dict(payload)
        annotations = result.get("annotations", result.get("objects", []))
        if not isinstance(annotations, list):
            raise ValidationError("workflow_invalid_annotations", "Candidate annotations must be a list.")
        annotations = [dict(annotation) for annotation in annotations]
        for annotation in annotations:
            annotation.setdefault("object_id", annotation.get("id") or new_id("object"))
        ids = [annotation["object_id"] for annotation in annotations]
        if len(ids) != len(set(ids)):
            raise ValidationError("workflow_duplicate_object", "Object IDs must be unique within a revision.")
        result["annotations"] = annotations
        result.setdefault("objects", annotations)
        result.setdefault("issues", [])
        issues = []
        seen = set()
        for raw in result["issues"]:
            issue = dict(raw)
            issue.setdefault("id", "issue_" + fingerprint(issue)[:24])
            if issue["id"] not in seen:
                issues.append(issue)
                seen.add(issue["id"])
        result["issues"] = issues
        return result

    def _context(self, session, run, item, node):
        outputs = self._outputs(session, run.id, item.id if item else None)
        if node.inputs:
            missing = [source for source in node.inputs.values() if source not in outputs]
            if missing:
                raise ValidationError("workflow_missing_port", "A declared input has no completed upstream artifact.",
                                      details={"node_id": node.id, "source_nodes": missing})
            outputs = {port: outputs[source] for port, source in node.inputs.items()}
        base = {}
        if item:
            base = {key: value for key, value in item.context_json.items() if key != "image_path"}
            base.update(annotations=item.context_json["existing_annotations"], rules=run.profile_snapshot.get("rules", {}))
        payload = dict(base)
        for output in outputs.values():
            payload.update(output)
        inputs = dict(run.context_json)
        if not item:
            if any(step["type"] == "human_review" for step in run.template_snapshot["nodes"]):
                inputs["confirmed_image_ids"] = list(session.scalars(select(WorkflowItem.subject_ref).where(
                    WorkflowItem.run_id == run.id, WorkflowItem.confirmed.is_(True), WorkflowItem.status != "excluded")))
            else:
                inputs["confirmed_image_ids"] = inputs.get("image_ids") or None
        return {"run_id": run.id, "item_id": item.id if item else None, "profile": run.profile_snapshot,
                "inputs": inputs, "item": item.context_json if item else None, "payload": payload,
                "outputs": outputs, "params": node.params,
                "request_budget": max(run.profile_snapshot.get("max_requests", 1000) - run.request_count, 0)}

    def tick(self, run_id: str) -> bool:
        if not self._publish_outputs(run_id):
            return False
        changed = self._tick_once(run_id)
        return changed if self._publish_outputs(run_id) else False

    def _publish_outputs(self, run_id):
        from app.storage_settings.outputs import WorkflowOutputPublisher
        with self.session_factory() as session:
            if self._row(session, WorkflowRun, run_id).deleted_at is not None:
                return False
        try:
            WorkflowOutputPublisher(self.session_factory, self.storage).publish(run_id)
            return True
        except Exception as exc:
            with self.session_factory() as session:
                session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
                run = self._row(session, WorkflowRun, run_id)
                if run.deleted_at is not None:
                    return False
                if not run.cancel_requested:
                    run.status = "needs_attention"
                    run.error_message = "Workflow output could not be saved. Check the registered destination and retry this run."
                self._event(session, run.id, "output_failed", {"code": getattr(exc, "error_code", "workflow_output_failed")})
                session.commit()
            return False

    def _tick_once(self, run_id: str) -> bool:
        """Execute at most one persisted node, with no transaction held during I/O."""
        with self.session_factory() as session:
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, run_id)
            if run.deleted_at is not None:
                return False
            if run.cancel_requested:
                return self._settle_cancel(session, run)
            if run.status in TERMINAL or run.status == "needs_attention":
                return False
            if session.scalar(select(WorkflowAttempt.id).where(WorkflowAttempt.run_id == run.id, WorkflowAttempt.status == "running")):
                return False
            nodes = [NodeDefinition.model_validate(value) for value in run.template_snapshot["nodes"]]
            try:
                validate_nodes(nodes)
            except DomainError as exc:
                run.status = "needs_attention"
                run.error_message = exc.message
                self._event(session, run.id, "contract_unavailable", {"code": exc.error_code})
                session.commit()
                return False
            items = list(session.scalars(select(WorkflowItem).where(WorkflowItem.run_id == run.id).order_by(WorkflowItem.created_at, WorkflowItem.id)))
            item = next((row for row in items if row.status in {"pending", "running"}), None)
            if item:
                cursor = item.cursor
                while cursor < len(nodes) and (nodes[cursor].scope != "item" or
                      (nodes[cursor].task_types and run.profile_snapshot["task_type"] not in nodes[cursor].task_types)):
                    cursor += 1
                item.cursor = cursor
                if cursor >= len(nodes) or nodes[cursor].scope != "item":
                    item.status = "completed"
                    session.commit()
                    return True
            else:
                if any(row.status == "failed" for row in items):
                    run.status = "needs_attention"
                    run.error_message = "Some selected items failed or were rejected. Review or explicitly exclude them."
                    session.commit()
                    return False
                if any(row.status == "awaiting_review" for row in items):
                    run.status = "awaiting_review"
                    session.commit()
                    return False
                if any(node.type == "human_review" for node in nodes) and any(not row.confirmed and row.status != "excluded" for row in items):
                    run.status = "needs_attention"
                    run.error_message = "The batch barrier requires every selected item to be confirmed or explicitly excluded."
                    session.commit()
                    return False
                cursor = run.cursor
                while cursor < len(nodes) and nodes[cursor].scope != "batch":
                    cursor += 1
                run.cursor = cursor
                if cursor >= len(nodes):
                    run.status = "completed"
                    run.finished_at = utc_now()
                    self._event(session, run.id, "completed")
                    session.commit()
                    return True
            node = nodes[cursor]
            unit_key = item.id if item else "batch"
            previous = session.scalar(select(WorkflowAttempt).where(WorkflowAttempt.run_id == run.id,
                WorkflowAttempt.unit_key == unit_key, WorkflowAttempt.node_id == node.id).order_by(WorkflowAttempt.attempt.desc()).limit(1))
            retry_at = (previous.output_json or {}).get("retry_at") if previous else None
            if retry_at and datetime.fromisoformat(retry_at) > utc_now():
                return False
            try:
                context = self._context(session, run, item, node)
            except DomainError as exc:
                run.status = "needs_attention"
                run.error_message = exc.message
                self._event(session, run.id, "node_input_failed", {"node_id": node.id, "code": exc.error_code})
                session.commit()
                return False
            if previous and previous.status == "waiting_task":
                attempt = previous
                context = attempt.input_json
            else:
                number = previous.attempt + 1 if previous else 1
                if number > node.max_attempts:
                    run.status = "needs_attention"
                    run.error_message = f"Node {node.id} reached its attempt limit. Create a new run after resolving the cause."
                    session.commit()
                    return False
                reserved_task_id = previous.reserved_task_id if previous else None
                if reserved_task_id and previous.status == "failed" and node.type in {"train", "evaluate"}:
                    from app.core.models import ModelEvaluationRecord, TrainingTask
                    domain_task = session.get(TrainingTask if node.type == "train" else ModelEvaluationRecord, reserved_task_id)
                    if domain_task is not None and domain_task.status in {"failed", "stopped", "cancelled"}:
                        reserved_task_id = None
                attempt = WorkflowAttempt(id=new_id("wattempt"), run_id=run.id, item_id=item.id if item else None,
                    unit_key=unit_key, node_id=node.id, node_type=node.type, attempt=number, generation=run.generation + 1,
                    fingerprint=fingerprint(context), status="running", input_json=context,
                    reserved_task_id=(reserved_task_id or new_id("train" if node.type == "train" else "eval")) if node.type in {"train", "evaluate"} else None,
                    task_kind=node.type if node.type in {"train", "evaluate"} else None)
                session.add(attempt)
            run.generation += 1
            attempt.generation = run.generation
            attempt.status = "running"
            run.status = "running"
            if item:
                item.status = "running"
            self._event(session, run.id, "node_started", {"node_id": node.id, "attempt": attempt.attempt, "item_id": attempt.item_id})
            session.commit()
            attempt_id, generation = attempt.id, attempt.generation
        try:
            if node.type == "human_review":
                return self._open_review(attempt_id, generation)
            result = self._execute(node, context, attempt_id)
            return self._finalize(attempt_id, generation, node, result)
        except Exception as exc:
            self._fail(attempt_id, generation, exc)
            return False

    def _settle_cancel(self, session, run):
        active = session.scalar(select(WorkflowAttempt.id).where(WorkflowAttempt.run_id == run.id,
            WorkflowAttempt.status == "cancelling"))
        child = session.scalar(select(WorkflowRun.id).where(WorkflowRun.parent_run_id == run.id,
            WorkflowRun.status.notin_(TERMINAL)))
        if not active and not child and run.status != "cancelled":
            run.status = "cancelled"
            run.finished_at = utc_now()
            self._event(session, run.id, "cancelled")
            session.commit()
            return True
        session.commit()
        return False

    def _execute(self, node, context, attempt_id):
        if node.type in {"train", "evaluate"}:
            with self.session_factory() as session:
                attempt = self._row(session, WorkflowAttempt, attempt_id)
                task_id = attempt.reserved_task_id
            result = self.bridges.task(node.type, context, task_id)
            with self.session_factory() as session:
                attempt = self._row(session, WorkflowAttempt, attempt_id)
                cancelled = self._row(session, WorkflowRun, attempt.run_id).cancel_requested
            if cancelled:
                self.bridges.cancel(node.type, task_id)
                raise ConflictError("workflow_cancelled", "The workflow stopped while its managed task was being submitted.")
            return result
        if node.type == "snapshot":
            return self.bridges.snapshot(context)
        if node.type == "next_batch":
            return self._next_batch(context)
        if node.type == "export" and "export" not in self.bindings.handlers:
            from app.dataset.snapshots import get_snapshot
            from app.workflow.exports import export_snapshot
            with self.session_factory() as session:
                snapshot = get_snapshot(session, self.storage, self.bridges.output_value(context, "snapshot_id"))
            return export_snapshot(snapshot, self.storage, context["profile"].get("export_format", "yolo"))
        if node.type == "repair_proposals" and context["profile"].get("repair_limit", 0) == 0:
            return {**context["payload"], "request_count": 0}
        if node.type == "vision_review" and node.params.get("only_if_repaired") and not context["payload"].get("repair_attempts"):
            return {**context["payload"], "request_count": 0}
        if node.type == "validate_proposals":
            return self._validate_proposals(context)
        handler = self.bindings.handlers.get(node.type)
        if handler is None:
            raise ValidationError("workflow_capability_unavailable", f"Node capability {node.type} is not configured.")
        if CONTRACTS[node.type].resource == "vision":
            context = self._reserve_requests(node, context, attempt_id)
        result = handler(context)
        if not isinstance(result, dict):
            raise ValidationError("workflow_invalid_node_output", "Node outputs must be JSON objects.")
        return result

    def _reserve_requests(self, node, context, attempt_id):
        estimator = self.bindings.handlers.get("estimate_requests")
        estimate = estimator({**context, "node_type": node.type}) if estimator else {"count": 1}
        count = estimate.get("count")
        if not isinstance(count, int) or count < 0:
            raise ValidationError("workflow_invalid_request_cost", "Vision request cost must be a nonnegative integer.")
        with self.session_factory() as session:
            attempt = self._row(session, WorkflowAttempt, attempt_id)
            session.execute(update(WorkflowRun).where(WorkflowRun.id == attempt.run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, attempt.run_id)
            session.refresh(run)
            if run.cancel_requested or run.generation != attempt.generation:
                raise ConflictError("workflow_cancelled", "The workflow stopped before request dispatch.")
            if run.request_count + count > run.profile_snapshot.get("max_requests", 1000):
                raise ValidationError("workflow_request_budget_exhausted", "The approved request budget is exhausted.")
            run.request_count += count
            context = {**context, "request_budget": count, "reserved_requests": count}
            attempt.input_json = context
            self._event(session, run.id, "requests_reserved", {"attempt_id": attempt.id, "count": count})
            session.commit()
        return context

    def _validate_proposals(self, context):
        from app.core.schemas import AnnotationInput
        from app.core.task_types import annotation_type_for
        from app.dataset.annotation.service import _validate_input
        payload = self._normalized_payload(context["payload"])
        classes = {rule["class_id"] for rule in context["profile"]["class_rules"]}
        # Class and geometry blockers describe the current accepted annotations.
        issues = [issue for issue in payload.get("issues", []) if issue.get("type") not in {"invalid_geometry", "unmapped_class"}]
        for annotation in payload["annotations"]:
            try:
                if annotation.get("class_id") not in classes:
                    raise ValueError("Candidate class is not mapped to the task profile.")
                if annotation.get("type") != annotation_type_for(context["profile"]["task_type"]):
                    raise ValueError("Candidate annotation geometry does not match the task type.")
                if isinstance(annotation.get("mask"), dict):
                    from app.dataset.annotation.masks import MaskService
                    annotation["mask_ref"] = MaskService(self.storage).write(annotation["mask"]).mask_ref
                parsed = AnnotationInput.model_validate(annotation)
                _validate_input(parsed, context["item"]["width"], context["item"]["height"], self.storage)
            except (DomainError, ValueError) as exc:
                issues.append({"id": f"geometry-{annotation['object_id']}", "object_id": annotation["object_id"],
                               "type": "invalid_geometry", "severity": "error", "blocking": True, "message": str(exc)[:500]})
        return {**payload, "issues": issues, "outcome": "invalid" if issues else "valid"}

    def _open_review(self, attempt_id, generation):
        with self.session_factory() as session:
            attempt = self._row(session, WorkflowAttempt, attempt_id)
            run = self._row(session, WorkflowRun, attempt.run_id)
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run.id).values(updated_at=utc_now()))
            session.refresh(run)
            if run.cancel_requested or run.generation != generation:
                attempt.status = "cancelled"
                session.flush()
                self._settle_cancel(session, run)
                return False
            item = self._row(session, WorkflowItem, attempt.item_id)
            payload = self._normalized_payload(attempt.input_json["payload"])
            payload["revision"] = 1
            artifact = self._artifact(session, run.id, item.id, attempt.id, "review_proposals", payload)
            session.add(WorkflowReview(id=new_id("wreview"), run_id=run.id, item_id=item.id,
                attempt_id=attempt.id, artifact_id=artifact.id, revision=1, base_revision=item.context_json["base_revision"]))
            attempt.status = "awaiting_review"
            item.status = "awaiting_review"
            run.status = "awaiting_review"
            self._event(session, run.id, "review_opened", {"item_id": item.id, "artifact_id": artifact.id})
            session.commit()
            return True

    def _finalize(self, attempt_id, generation, node, result):
        self._safe_json(result)
        with self.session_factory() as session:
            attempt = self._row(session, WorkflowAttempt, attempt_id)
            session.execute(update(WorkflowRun).where(WorkflowRun.id == attempt.run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, attempt.run_id)
            session.refresh(run)
            if run.cancel_requested or run.generation != generation or attempt.status != "running":
                if run.cancel_requested:
                    attempt.status = "cancelled"
                    session.flush()
                    self._settle_cancel(session, run)
                return False
            item = session.get(WorkflowItem, attempt.item_id) if attempt.item_id else None
            if result.get("status") == "waiting_task":
                attempt.status = "waiting_task"
                attempt.output_json = result
                run.status = "waiting_task"
                session.commit()
                return False
            if item:
                if "objects" in result and "annotations" not in result:
                    result["annotations"] = result["objects"]
                if "annotations" in result:
                    result["objects"] = result["annotations"]
                if CONTRACTS[node.type].resource != "vision":
                    result["request_count"] = 0
                result = self._normalized_payload({**attempt.input_json["payload"], **result})
                mode = run.context_json.get("existing_annotation_mode", run.profile_snapshot.get("rules", {}).get("existing_annotations", "preserve"))
                if node.type in {"vision_propose", "managed_inference"} and mode != "replace":
                    proposed = result["annotations"]
                    ids = {annotation["object_id"] for annotation in proposed}
                    existing = [annotation for annotation in item.context_json["existing_annotations"] if annotation["object_id"] not in ids]
                    result["annotations"] = existing + proposed
                    result["objects"] = result["annotations"]
            artifact = self._artifact(session, run.id, attempt.item_id, attempt.id, CONTRACTS[node.type].output_type, result)
            attempt.output_json = result
            attempt.status = "completed"
            attempt.finished_at = utc_now()
            self._advance(run, item, node, result)
            self._event(session, run.id, "node_completed", {"node_id": node.id, "item_id": attempt.item_id, "artifact_id": artifact.id})
            session.commit()
            return True

    @staticmethod
    def _advance(run, item, node, output):
        nodes = run.template_snapshot["nodes"]
        current = next(index for index, value in enumerate(nodes) if value["id"] == node.id)
        target = node.routes.get(output.get("outcome", "completed"))
        cursor = next(index for index, value in enumerate(nodes) if value["id"] == target) if target else current + 1
        if item:
            item.cursor = cursor
            item.status = "pending"
        else:
            run.cursor = cursor
        run.status = "pending"
        run.error_message = None

    def _fail(self, attempt_id, generation, exc):
        logger.warning("Workflow node failed attempt=%s code=%s", attempt_id, getattr(exc, "error_code", type(exc).__name__))
        with self.session_factory() as session:
            attempt = self._row(session, WorkflowAttempt, attempt_id)
            session.execute(update(WorkflowRun).where(WorkflowRun.id == attempt.run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, attempt.run_id)
            session.refresh(run)
            if run.cancel_requested or run.generation != generation:
                if run.cancel_requested:
                    attempt.status = "cancelled"
                    session.flush()
                    self._settle_cancel(session, run)
                return
            message = redact_text(exc.message if isinstance(exc, DomainError) else f"{type(exc).__name__}: node execution failed.")[:1000]
            attempt.status = "failed"
            attempt.error_message = message
            attempt.finished_at = utc_now()
            code = getattr(exc, "error_code", "node_failed")
            transient = code in {"vision_timeout", "vision_request_failed"} or (
                code == "vision_http_error" and getattr(exc, "details", {}).get("http_status") in {429, 500, 502, 503, 504})
            node = next(value for value in run.template_snapshot["nodes"] if value["id"] == attempt.node_id)
            retryable = transient and attempt.attempt < node.get("max_attempts", 2)
            if retryable:
                attempt.output_json = {"retry_at": (utc_now() + timedelta(seconds=2 ** attempt.attempt)).isoformat(),
                    "request_id": getattr(exc, "details", {}).get("request_id"),
                    "outcome_unknown": getattr(exc, "details", {}).get("outcome_unknown", False)}
            if attempt.item_id:
                item = self._row(session, WorkflowItem, attempt.item_id)
                item.status = "pending" if retryable else "failed"
                item.error_message = message
            run.status = "pending" if attempt.item_id or retryable else "needs_attention"
            run.error_message = message
            self._event(session, run.id, "node_failed", {"node_id": attempt.node_id, "item_id": attempt.item_id,
                        "code": getattr(exc, "error_code", "node_failed"), "message": message})
            session.commit()

    def _next_batch(self, context):
        batches = context["inputs"].get("next_batches", [])
        if not batches:
            return {"status": "completed", "next_run_id": None}
        model_id = self.bridges.output_value(context, "model_id")
        with self.session_factory() as session:
            parent = self._row(session, WorkflowRun, context["run_id"])
            batch = batches[0]
            request = RunCreate(idempotency_key=f"next:{parent.id}", template_id=parent.template_id,
                profile_id=parent.profile_id, dataset_id=batch["dataset_id"], image_ids=batch["image_ids"],
                project_id=parent.context_json.get("project_id"),
                next_batches=batches[1:], inputs={**{key: value for key, value in parent.context_json.items()
                if key not in {"image_ids", "dataset_id", "snapshot_id", "next_batches"}}, "model_id": model_id})
            profile = dict(parent.profile_snapshot)
        child = self.create_run(request, parent_run_id=context["run_id"], frozen_profile=profile)
        if child["status"] == "cancelled":
            raise ConflictError("workflow_child_cancelled", "The next batch was cancelled. Resolve its scope before continuing.")
        return {"status": "completed" if child["status"] == "completed" else "waiting_task",
                "next_run_id": child["id"], "child_status": child["status"], "model_id": model_id}

    def get_review(self, review_id):
        with self.session_factory() as session:
            return self._review_response(session, self._row(session, WorkflowReview, review_id))

    def save_review_preview(self, review_id: str, content: bytes, expected_revision: int,
                            has_unsaved_changes: bool = False) -> dict:
        from app.workflow.previews import save_preview

        review = self.get_review(review_id)
        with self.session_factory() as session:
            run = self._row(session, WorkflowRun, review["run_id"])
            outputs = run.context_json.get("output_destinations", {})
        return save_preview(self.storage, review, content, expected_revision, has_unsaved_changes,
                            project_root=self.project_root, output_destinations=outputs)

    def preview_download(self, review_id: str, file_name: str) -> Path:
        from app.workflow.previews import preview_download

        self.get_review(review_id)
        return preview_download(self.storage, review_id, file_name)

    def save_review_labelme(self, review_id: str, payload: ReviewDraft) -> dict:
        from app.workflow.labelme_export import save_labelme

        with self.session_factory() as session:
            row = self._row(session, WorkflowReview, review_id)
            review = self._review_response(session, row)
            if review["revision"] != payload.expected_revision:
                raise ConflictError("workflow_labelme_stale", "复核草稿已更新，请重新打开当前修订后导出。")
            for key in RESERVED_PAYLOAD:
                if key in payload.payload and payload.payload[key] != review["payload"].get(key):
                    raise ConflictError("workflow_review_identity_changed", "导出的原图身份和基础修订必须与当前复核记录一致。")
            run = self._row(session, WorkflowRun, row.run_id)
            image = self._row(session, ImageItem, review["payload"]["image_id"])
            if image.dataset_id != review["payload"]["dataset_id"]:
                raise ConflictError("workflow_review_identity_changed", "原图不属于本次复核的数据集。")
            merged = {**review["payload"], **payload.payload}
            merged["revision"] = review["revision"]
            if "annotations" in payload.payload and "objects" not in payload.payload:
                merged["objects"] = payload.payload["annotations"]
            image_data = {key: getattr(image, key) for key in ("id", "dataset_id", "file_name", "storage_name", "width", "height", "split")}
            return save_labelme(self.storage, review, merged, run.profile_snapshot, image_data, project_root=self.project_root,
                                output_destinations=run.context_json.get("output_destinations", {}))

    def labelme_download(self, review_id: str, file_name: str) -> Path:
        from app.workflow.labelme_export import labelme_download

        self.get_review(review_id)
        return labelme_download(self.storage, review_id, file_name)

    def _review_response(self, session, row):
        artifact = self._row(session, WorkflowArtifact, row.artifact_id)
        result = {"id": row.id, "run_id": row.run_id, "item_id": row.item_id, "status": row.status,
                "revision": row.revision, "base_revision": row.base_revision, "payload": artifact.payload_json,
                "artifact_id": row.artifact_id, "decision": row.decision, "reason": row.reason, "receipt": row.receipt_json}
        run = self._row(session, WorkflowRun, row.run_id)
        outputs = run.context_json.get("output_destinations", {})
        if outputs.get("project"):
            from app.workflow.projects import project_export_locations

            result.update(project=outputs["project"], export_locations=project_export_locations(outputs))
        return result

    def save_review(self, review_id: str, payload: ReviewDraft):
        with self.session_factory() as session:
            review = self._row(session, WorkflowReview, review_id)
            self._lock_review(session, review, payload.expected_revision)
            if review.status == "decided" and review.decision != "rejected":
                raise ConflictError("workflow_review_already_decided", "Confirmed or excluded revisions cannot be edited. Start a new reviewed revision from the current labels.")
            old = self._row(session, WorkflowArtifact, review.artifact_id)
            updated = dict(payload.payload)
            for key in RESERVED_PAYLOAD:
                if key in updated and updated[key] != old.payload_json.get(key):
                    raise ConflictError("workflow_review_identity_changed", "Draft image identity and base revision cannot be changed.")
            merged = {**old.payload_json, **updated}
            if "objects" in updated and "annotations" not in updated:
                merged["annotations"] = updated["objects"]
            merged = self._normalized_payload(merged)
            review.revision += 1
            merged["revision"] = review.revision
            item = self._row(session, WorkflowItem, review.item_id)
            run = self._row(session, WorkflowRun, review.run_id)
            if merged.get("visual_review"):
                merged["visual_review"] = {**merged["visual_review"], "stale": True}
            merged = self._validate_proposals({"payload": merged, "profile": run.profile_snapshot, "item": item.context_json})
            artifact = self._artifact(session, review.run_id, review.item_id, review.attempt_id, "review_proposals", merged, [old.id])
            review.artifact_id = artifact.id
            review.status = "pending"
            review.decision = None
            item.status = "awaiting_review"
            item.error_message = None
            run.status = "awaiting_review"
            run.error_message = None
            self._event(session, run.id, "review_draft_saved", {"review_id": review.id, "revision": review.revision})
            session.commit()
            return self._review_response(session, review)

    def _lock_review(self, session, review, expected_revision, *, allow_decided=False):
        session.execute(update(WorkflowRun).where(WorkflowRun.id == review.run_id).values(updated_at=utc_now()))
        session.refresh(review)
        run = self._row(session, WorkflowRun, review.run_id)
        session.refresh(run)
        if (run.cancel_requested or run.status in TERMINAL) and not (allow_decided and review.status == "decided"):
            raise ConflictError("workflow_run_closed", "This workflow is no longer accepting review changes.")
        if review.revision != expected_revision:
            raise ConflictError("workflow_review_stale", "The reviewed draft changed. Reload its current revision.")

    def decide_review(self, review_id: str, payload: ReviewDecision):
        with self.session_factory() as session:
            review = self._row(session, WorkflowReview, review_id)
            self._lock_review(session, review, payload.expected_revision, allow_decided=True)
            if review.status == "decided":
                if review.decision == payload.decision:
                    return self._review_response(session, review)
                raise ConflictError("workflow_review_already_decided", "This revision already has a different decision.")
            item = self._row(session, WorkflowItem, review.item_id)
            run = self._row(session, WorkflowRun, review.run_id)
            artifact = self._row(session, WorkflowArtifact, review.artifact_id)
            self.artifacts.verify(artifact.storage_name, artifact.sha256)
            if payload.decision in {"approved", "background"}:
                context = {**item.context_json, **artifact.payload_json, "revision": review.revision,
                           "payload": artifact.payload_json, "run_id": run.id, "review_id": review.id}
                receipt = self.bridges.confirm(session, context, {"confirmed_empty": payload.decision == "background",
                    "semantic_resolutions": payload.semantic_resolutions, "actor": payload.actor, "reason": payload.reason})
                review.receipt_json = receipt
                item.receipt_json = receipt
                item.confirmed = True
                from app.dataset.annotation.review import current_content
                image = self._row(session, ImageItem, item.subject_ref)
                confirmed_payload = {**current_content(session, self.storage, image), "receipt": receipt,
                                     "disposition": payload.decision, "review_revision": review.revision}
                self._artifact(session, run.id, item.id, review.attempt_id, "confirmed_annotations", confirmed_payload, [artifact.id])
                attempt = self._row(session, WorkflowAttempt, review.attempt_id)
                attempt.status = "completed"
                attempt.output_json = {**artifact.payload_json, "receipt": receipt}
                attempt.finished_at = utc_now()
                node = NodeDefinition.model_validate(next(value for value in run.template_snapshot["nodes"] if value["id"] == attempt.node_id))
                self._advance(run, item, node, {"outcome": "approved"})
            elif payload.decision == "excluded":
                if not payload.reason.strip():
                    raise ValidationError("workflow_exclusion_reason_required", "Excluding a selected item requires a reason.")
                item.status = "excluded"
                self._row(session, WorkflowAttempt, review.attempt_id).status = "completed"
                run.status = "pending"
            else:
                item.status = "failed"
                item.error_message = payload.reason or "Rejected by user."
                run.status = "needs_attention"
            review.status = "decided"
            review.decision = payload.decision
            review.reason = payload.reason
            item.disposition = payload.decision
            self._event(session, run.id, "review_decided", {"review_id": review.id, "revision": review.revision,
                        "decision": payload.decision, "reason": payload.reason, "actor": payload.actor})
            session.commit()
            result = self._review_response(session, review)
        self.wake()
        self._publish_outputs(review.run_id)
        return result

    def cancel_run(self, run_id, *, operation_id: str | None = None):
        with self.session_factory() as session:
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, run_id)
            if operation_id and session.get(WorkflowEvent, operation_id):
                return self._run_response(session, run)
            if run.status in TERMINAL:
                if operation_id:
                    session.add(WorkflowEvent(id=operation_id, run_id=run.id, kind="agent_cancel", payload_json={"run_id": run.id}))
                    session.commit()
                return self._run_response(session, run)
            run.cancel_requested = True
            run.generation += 1
            run.status = "cancelling"
            active_query = select(WorkflowAttempt).where(
                WorkflowAttempt.run_id == run.id,
                WorkflowAttempt.status.in_(["running", "waiting_task"]),
            )
            active_attempts = list(session.scalars(active_query))
            tasks = [(attempt.task_kind, attempt.reserved_task_id) for attempt in active_attempts]
            for attempt in active_attempts:
                attempt.status = "cancelling" if attempt.status == "running" else "cancelled"
            children = list(session.scalars(select(WorkflowRun.id).where(WorkflowRun.parent_run_id == run.id)))
            for item in session.scalars(select(WorkflowItem).where(WorkflowItem.run_id == run.id, WorkflowItem.status.in_(["pending", "running", "awaiting_review"]))):
                item.status = "cancelled"
            session.execute(update(WorkflowAttempt).where(WorkflowAttempt.run_id == run.id,
                WorkflowAttempt.status == "awaiting_review").values(status="cancelled", finished_at=utc_now()))
            self._event(session, run.id, "cancel_requested")
            if operation_id:
                session.add(WorkflowEvent(id=operation_id, run_id=run.id, kind="agent_cancel", payload_json={"run_id": run.id}))
            session.commit()
        for kind, task_id in tasks:
            if kind and task_id:
                self.bridges.cancel(kind, task_id)
        for child_id in children:
            self.cancel_run(child_id)
        with self.session_factory() as session:
            run = self._row(session, WorkflowRun, run_id)
            self._settle_cancel(session, run)
            return self._run_response(session, run)

    def retry_run(self, run_id, item_ids=None, *, operation_id: str | None = None):
        with self.session_factory() as session:
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, run_id)
            if operation_id and session.get(WorkflowEvent, operation_id):
                return self._run_response(session, run)
            if run.status in TERMINAL:
                raise ConflictError("workflow_run_closed", "Create a new run to repeat a completed or cancelled workflow.")
            selected = set(item_ids or [])
            items = list(session.scalars(select(WorkflowItem).where(WorkflowItem.run_id == run.id)))
            if selected - {item.id for item in items}:
                raise ValidationError("workflow_unknown_item", "Retry selection contains items outside this run.")
            for item in items:
                if item.status == "failed" and (not selected or item.id in selected):
                    review = session.scalar(select(WorkflowReview).where(WorkflowReview.item_id == item.id).order_by(WorkflowReview.created_at.desc()))
                    if review and review.decision == "rejected":
                        raise ConflictError("workflow_review_rejected", "Edit the rejected review draft before continuing.")
                    item.status = "pending"
                    item.error_message = None
            run.status = "pending"
            run.error_message = None
            self._event(session, run.id, "retry_requested", {"item_ids": list(selected)})
            if operation_id:
                session.add(WorkflowEvent(id=operation_id, run_id=run.id, kind="agent_resume", payload_json={"run_id": run.id}))
            session.commit()
            result = self._run_response(session, run)
        self.wake()
        return result

    def recover(self):
        with self.session_factory() as session:
            for attempt in session.scalars(select(WorkflowAttempt).where(WorkflowAttempt.status == "running")):
                attempt.status = "waiting_task" if attempt.reserved_task_id else "interrupted"
                run = self._row(session, WorkflowRun, attempt.run_id)
                if not run.cancel_requested:
                    run.status = "pending"
                    if attempt.item_id:
                        self._row(session, WorkflowItem, attempt.item_id).status = "pending"
                self._event(session, run.id, "attempt_interrupted", {"attempt_id": attempt.id})
            for attempt in session.scalars(select(WorkflowAttempt).where(WorkflowAttempt.status == "cancelling")):
                attempt.status = "cancelled"
            for run in session.scalars(select(WorkflowRun).where(WorkflowRun.status.notin_(TERMINAL))):
                for artifact in session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.run_id == run.id)):
                    try:
                        self.artifacts.verify(artifact.storage_name, artifact.sha256)
                    except DomainError as exc:
                        run.status = "needs_attention"
                        run.error_message = exc.message
                        break
            session.commit()

    def runnable_ids(self):
        with self.session_factory() as session:
            return list(session.scalars(select(WorkflowRun.id).where(WorkflowRun.deleted_at.is_(None), WorkflowRun.status.in_(
                ["pending", "running", "waiting_task", "awaiting_review", "cancelling"])).order_by(WorkflowRun.created_at)))

    def get_run(self, run_id):
        with self.session_factory() as session:
            return self._run_response(session, self._row(session, WorkflowRun, run_id))

    def export_download(self, run_id):
        with self.session_factory() as session:
            self._row(session, WorkflowRun, run_id)
            artifact = session.scalar(select(WorkflowArtifact).where(WorkflowArtifact.run_id == run_id,
                WorkflowArtifact.kind == "export").order_by(WorkflowArtifact.created_at.desc()).limit(1))
            if artifact is None:
                raise NotFoundError("workflow_export_not_found", "This run has no completed export.")
            payload = artifact.payload_json
            path = self.storage.export_path(payload["file_name"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != payload["sha256"]:
                raise ValidationError("workflow_export_changed", "The managed export is missing or has changed.")
            return path

    @staticmethod
    def _can_delete_run(session, run) -> bool:
        if run.status not in TERMINAL:
            return False
        active_attempt = session.scalar(select(WorkflowAttempt.id).where(
            WorkflowAttempt.run_id == run.id,
            WorkflowAttempt.status.in_(["running", "waiting_task", "cancelling"]),
        ).limit(1))
        active_child = session.scalar(select(WorkflowRun.id).where(
            WorkflowRun.parent_run_id == run.id, WorkflowRun.status.notin_(TERMINAL),
        ).limit(1))
        published = aliased(WorkflowEvent)
        pending_output = session.scalar(select(WorkflowEvent.id).where(
            WorkflowEvent.run_id == run.id, WorkflowEvent.kind == "output_pending",
            ~exists(select(published.id).where(
                published.run_id == run.id, published.kind == "output_published",
                published.payload_json["intent_id"].as_string() == WorkflowEvent.id,
            )),
        ).limit(1))
        return active_attempt is None and active_child is None and pending_output is None

    def delete_run(self, run_id: str) -> dict:
        with self.session_factory() as session:
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, run_id)
            if run.deleted_at is None:
                if not self._can_delete_run(session, run):
                    message = "结果或关联任务仍在处理中，请稍后再删除记录。" if run.status in TERMINAL else "请先取消任务，待运行结束后再删除记录。"
                    raise ConflictError("workflow_run_active", message)
                run.deleted_at = utc_now()
                self._event(session, run.id, "run_deleted")
                session.commit()
            return self._run_response(session, run)

    def restore_run(self, run_id: str) -> dict:
        with self.session_factory() as session:
            session.execute(update(WorkflowRun).where(WorkflowRun.id == run_id).values(updated_at=utc_now()))
            run = self._row(session, WorkflowRun, run_id)
            if run.deleted_at is not None:
                run.deleted_at = None
                self._event(session, run.id, "run_restored")
                session.commit()
            return self._run_response(session, run)

    def list_runs(self, limit=50, *, deleted=False):
        with self.session_factory() as session:
            selection = WorkflowRun.deleted_at.is_not(None) if deleted else WorkflowRun.deleted_at.is_(None)
            return [self._run_response(session, row, details=False) for row in session.scalars(select(WorkflowRun).where(selection).order_by(
                WorkflowRun.created_at.desc()).limit(limit))]

    def _run_response(self, session, run, details=True):
        items = list(session.scalars(select(WorkflowItem).where(WorkflowItem.run_id == run.id).order_by(WorkflowItem.created_at, WorkflowItem.id)))
        counts = {"total": len(items), "approved": 0, "background": 0, "excluded": 0, "pending": 0, "failed": 0}
        for item in items:
            key = item.disposition if item.disposition in {"approved", "background", "excluded"} else "failed" if item.status == "failed" else "pending"
            counts[key] += 1
        result = {"id": run.id, "status": run.status, "template_id": run.template_id, "profile_id": run.profile_id,
                  "template_name": session.get(WorkflowTemplate, run.template_id).name,
                  "profile_name": session.get(WorkflowTaskProfile, run.profile_id).name,
                  "error_message": run.error_message, "parent_run_id": run.parent_run_id, "batch_counts": counts,
                  "created_at": run.created_at, "updated_at": run.updated_at, "finished_at": run.finished_at,
                  "deleted_at": run.deleted_at, "can_delete": self._can_delete_run(session, run)}
        if not details:
            return result
        attempts = list(session.scalars(select(WorkflowAttempt).where(WorkflowAttempt.run_id == run.id).order_by(WorkflowAttempt.created_at, WorkflowAttempt.id)))
        artifacts = list(session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.run_id == run.id).order_by(WorkflowArtifact.created_at, WorkflowArtifact.id)))
        result.update(profile_snapshot=run.profile_snapshot, template_snapshot=run.template_snapshot,
            context=run.context_json, reserved_requests=run.request_count,
            request_count=sum((row.output_json or {}).get("request_count", 0) for row in attempts if row.status == "completed"),
            items=[{"id": row.id, "subject_ref": row.subject_ref, "status": row.status, "cursor": row.cursor,
                    "confirmed": row.confirmed, "disposition": row.disposition, "error_message": row.error_message,
                    "context": {key: value for key, value in row.context_json.items() if key != "image_path"}} for row in items],
            attempts=[{"id": row.id, "item_id": row.item_id, "node_id": row.node_id, "node_type": row.node_type,
                       "attempt": row.attempt, "status": row.status, "error_message": row.error_message,
                       "task_id": row.reserved_task_id, "task_kind": row.task_kind, "output": row.output_json} for row in attempts],
            artifacts=[{"id": row.id, "item_id": row.item_id, "kind": row.kind, "version": row.version,
                        "sha256": row.sha256, "payload": row.payload_json, "parent_ids": row.parent_ids_json} for row in artifacts],
            reviews=[self._review_response(session, row) for row in session.scalars(select(WorkflowReview).where(WorkflowReview.run_id == run.id).order_by(WorkflowReview.created_at))],
            events=[{"id": row.id, "kind": row.kind, "payload": row.payload_json, "created_at": row.created_at} for row in session.scalars(select(WorkflowEvent).where(WorkflowEvent.run_id == run.id).order_by(WorkflowEvent.created_at, WorkflowEvent.id))])
        return result
