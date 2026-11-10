"""Image-domain handlers consumed by the generic workflow bindings."""
from __future__ import annotations

from hashlib import file_digest, sha256
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.local_compute import local_compute_gate
from app.core.models import ClassLabel, ImageItem, ModelVersion
from app.core.schemas import BBox
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService, polygon_mask
from app.sam.schemas import SamPredictRequest
from app.sam.service import SamService
from app.vision.schemas import ImageStrategy, VisionApproval, VisionClassRule, VisionObject
from app.vision.service import VisionService, repair_targets


def build_vision_handlers(storage: Storage, settings: Settings, session_factory=None) -> dict:
    service = VisionService(storage, settings)

    def image_context(context: dict) -> tuple[dict, Path]:
        item = context.get("item") or {}
        if session_factory is None:
            raise ValidationError("vision_session_unavailable", "Vision workflow handlers require a managed database session.")
        with session_factory() as session:
            image = session.get(ImageItem, item.get("image_id") or context.get("image_id"))
            if image is None or image.dataset_id != context["inputs"]["dataset_id"]:
                raise ValidationError("vision_image_missing", "The workflow image is unavailable in this dataset.")
            path = storage.image_path(image.dataset_id, image.storage_name)
            with path.open("rb") as source:
                image_hash = file_digest(source, "sha256").hexdigest()
            if item.get("image_hash") != image_hash:
                raise ValidationError("vision_image_changed", "The original image changed after this workflow was created.")
            if image.annotation_revision != item.get("base_revision", image.annotation_revision):
                raise ValidationError("vision_annotations_changed", "The formal annotations changed after this workflow was created.")
            return {"image_id": image.id, "image_hash": image_hash, "width": image.width, "height": image.height,
                    "base_revision": image.annotation_revision, "roi_mask_ref": image.roi_mask_ref,
                    "ignore_mask_ref": image.ignore_mask_ref}, path

    def configuration(context: dict, role: str):
        profile = context["profile"]
        raw = (context["inputs"].get("vision_approvals") or {}).get(role)
        if raw is None:
            raise ValidationError("vision_approval_missing", "The run has no approved endpoint and image scope for this vision role.")
        approval = VisionApproval.model_validate(raw)
        remaining = context.get("request_budget")
        if remaining is None:
            remaining = approval.max_requests
        if isinstance(remaining, dict):
            remaining = remaining.get("remaining", approval.max_requests)
        approval.max_requests = min(approval.max_requests, int(remaining))
        if approval.max_requests <= 0:
            raise ValidationError("vision_request_budget_exceeded", "The workflow vision request budget is exhausted.")
        classes = [VisionClassRule.model_validate(rule) for rule in profile.get("class_rules", [])]
        strategy = ImageStrategy.model_validate(profile.get("image_strategy") or {})
        ruleset = str(profile.get("rules", {}).get("version", "v1"))
        return classes, strategy, ruleset, approval

    def payload_from_objects(result: dict, context: dict, image: dict, *, preserve_existing: bool = False) -> dict:
        objects = result.pop("objects", [])
        annotations = []
        for item in objects:
            item = dict(item)
            evidence = {key: item.pop(key) for key in ("evidence", "uncertainty", "confidence") if key in item}
            item["provenance_json"] = {**item.get("provenance_json", {}), **evidence}
            annotations.append(item)
        if preserve_existing:
            prior = (context.get("payload") or {}).get("annotations", [])
            mode = context["inputs"].get("existing_annotation_mode", "preserve")
            if mode not in {"preserve", "replace"}:
                raise ValidationError("vision_existing_annotation_mode", "Existing annotations require an explicit preserve or replace mode.")
            result["existing_annotation_diff"] = {"mode": mode, "before": prior, "proposed": annotations}
            if mode == "preserve":
                existing_ids = {item.get("object_id") for item in prior}
                for item in annotations:
                    if item.get("object_id") in existing_ids:
                        item["object_id"] += "_proposal"
                annotations = [*prior, *annotations]
        return {**image, **result, "annotations": annotations, "rules": context["profile"].get("rules", {})}

    def managed_prelabel(context: dict) -> dict:
        from app.models.inference import ModelCache

        image, image_path = image_context(context)
        model_id = context["inputs"].get("model_id")
        profile = context["profile"]
        if profile["task_type"] not in {"detect", "segment"}:
            raise ValidationError("vision_task_unsupported", "Managed visual candidates currently support detect and segment.")
        with session_factory() as session:
            version = session.get(ModelVersion, model_id)
            if version is None or version.status != "active" or version.format != "pt" or version.task_type != profile["task_type"]:
                raise ValidationError("vision_model_incompatible", "Select an active managed PT model matching the task.")
            model_path = storage.managed_model_path(version.model_path)
            if not model_path.is_file():
                raise ValidationError("vision_model_missing", "The managed model artifact is unavailable.")
            mapping = context["inputs"].get("class_mapping")
            if mapping is None and version.dataset_id == context["inputs"]["dataset_id"]:
                frozen_classes = getattr(version, "class_schema_json", None)
                mapping = ({str(label["index"]): label["id"] for label in frozen_classes} if frozen_classes else
                           {str(label.class_index): label.id for label in session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == version.dataset_id))})
            if mapping is None:
                raise ValidationError("vision_class_mapping_required", "A cross-dataset model requires explicit class mapping.")
            known = {rule["id"] for rule in profile.get("class_rules", [])}
            if any(value is not None and value not in known for value in mapping.values()):
                raise ValidationError("vision_class_mapping_invalid", "Class mapping must use the frozen profile class IDs.")
            with local_compute_gate.acquire():
                results = ModelCache.get_model(model_id, model_path).predict(source=str(image_path), conf=0.25, iou=0.5, retina_masks=True, verbose=False)
                results = [result.cpu() for result in results]
            objects = []
            for result in results:
                if getattr(result, "boxes", None) is None:
                    continue
                for index, box in enumerate(result.boxes):
                    source_class = str(int(box.cls[0].item()))
                    target_class = mapping.get(source_class)
                    if target_class is None:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    object_id = "obj_" + sha256(f"{image['image_hash']}:{model_id}:{index}".encode()).hexdigest()[:24]
                    item = VisionObject(object_id=object_id, class_id=target_class, bbox=BBox(x=x1, y=y1, width=x2-x1, height=y2-y1),
                                        confidence=float(box.conf[0].item()), provenance_json={"model_id": model_id, "backend": "managed_ultralytics", "source_class_index": source_class})
                    if profile["task_type"] == "segment":
                        if getattr(result, "masks", None) is None:
                            raise ValidationError("vision_mask_missing", "Segment model returned boxes without raw masks.")
                        mask = result.masks.data[index].detach().cpu().numpy()
                        if mask.shape != (image["height"], image["width"]):
                            raise ValidationError("vision_mask_size_mismatch", "Managed model mask is not in original-image coordinates.")
                        item.mask_ref = MaskService(storage).write(mask).mask_ref
                        item.type = "polygon"
                    objects.append(item.model_dump())
        return payload_from_objects({"objects": objects, "issues": [], "request_count": 0, "provenance_json": {"model_id": model_id}}, context, image, preserve_existing=True)

    def propose(context: dict) -> dict:
        if context.get("params", {}).get("source") != "vision" and context["inputs"].get("model_id"):
            return managed_prelabel(context)
        if context["profile"]["task_type"] not in {"detect", "segment"}:
            raise ValidationError("vision_task_unsupported", "Vision candidates currently support detect and segment.")
        image, path = image_context(context)
        classes, strategy, ruleset, approval = configuration(context, "proposer")
        result = service.propose(image["image_id"], path, classes, ruleset, strategy, approval,
                                 task_instructions=context["profile"].get("rules", {}).get("task_instructions", ""))
        return payload_from_objects(result.model_dump(), context, image, preserve_existing=True)

    def segment(context: dict) -> dict:
        image, _ = image_context(context)
        payload = dict(context.get("payload") or {})
        annotations = []
        for raw in payload.get("annotations", []):
            item = dict(raw)
            if not item.get("mask_ref") and item.get("polygon"):
                item["mask_ref"] = MaskService(storage).write(polygon_mask(item["polygon"], image["width"], image["height"])).mask_ref
            if not item.get("mask_ref") and item.get("class_id") is not None:
                frozen = context["inputs"].get("sam_config")
                if frozen is None:
                    raise ValidationError("vision_segmenter_not_frozen", "Prepare this run with an explicit SAM configuration before segmentation.")
                with session_factory() as session:
                    result = SamService(storage, settings).predict_mask(session, SamPredictRequest(
                        image_id=image["image_id"], class_id=item["class_id"], box=item.get("bbox"),
                    ), expected_config=frozen)
                item.update(type="polygon", mask_ref=result.mask_ref, polygon=None)
                item["provenance_json"] = {**item.get("provenance_json", {}), "segmenter": result.model_dump(exclude={"polygon"})}
            annotations.append(item)
        return {**payload, **image, "annotations": annotations}

    def review(context: dict) -> dict:
        image, path = image_context(context)
        payload = dict(context.get("payload") or {})
        classes, strategy, ruleset, approval = configuration(context, "reviewer")
        result = service.review(image["image_id"], path, classes, ruleset, int(payload.get("revision", 1)),
                                payload.get("annotations", []), strategy, approval,
                                task_instructions=context["profile"].get("rules", {}).get("task_instructions", ""))
        return {**payload, "issues": [*payload.get("issues", []), *result.issues], "visual_review": result.model_dump(), "request_count": result.request_count}

    def repair(context: dict) -> dict:
        image, path = image_context(context)
        payload = dict(context.get("payload") or {})
        attempts = int(payload.get("repair_attempts", 0))
        limit = int(context["profile"].get("repair_limit", 0))
        if attempts >= limit:
            return {**payload, "request_count": 0, "repair_status": "budget_reached"}
        targets = repair_targets(payload.get("annotations", []), payload.get("issues", []), image["width"], image["height"])
        if not targets:
            return {**payload, "request_count": 0, "repair_status": "requires_human_localization"}
        classes, strategy, ruleset, approved = configuration(context, "proposer")
        result = service.repair(image["image_id"], path, classes, ruleset, int(payload.get("revision", 1)),
                                payload.get("annotations", []), payload.get("issues", []), strategy, approved,
                                task_instructions=context["profile"].get("rules", {}).get("task_instructions", ""))
        repaired = payload_from_objects(result, context, image)
        updated = {**payload, **repaired, "repair_attempts": attempts + 1,
                   "visual_review": {**payload.get("visual_review", {}), "stale": True},
                   "repair_status": "requires_human_review"}
        if context["profile"]["task_type"] == "segment":
            return segment({**context, "payload": updated})
        return updated

    def estimate_requests(context: dict) -> dict:
        node_type = context.get("node_type")
        if node_type == "vision_propose" and context.get("params", {}).get("source") != "vision" and context["inputs"].get("model_id"):
            return {"count": 0}
        role = "reviewer" if node_type == "vision_review" else "proposer"
        image, path = image_context(context)
        if node_type == "repair_proposals":
            payload = context.get("payload") or {}
            if int(payload.get("repair_attempts", 0)) >= int(context["profile"].get("repair_limit", 0)):
                return {"count": 0}
            count = len(repair_targets(payload.get("annotations", []), payload.get("issues", []), image["width"], image["height"]))
            if not count:
                return {"count": 0}
            _, strategy, _, approved = configuration(context, role)
            service._prepare(role, path, ImageStrategy(), approved, 1)
            if count > min(approved.max_requests, strategy.max_views):
                raise ValidationError("vision_request_budget_exceeded", "Targeted repairs exceed the approved request or view budget.")
            return {"count": count}
        _, strategy, _, approved = configuration(context, role)
        multiplier = 2 if role == "reviewer" else 1
        _, _, views = service._prepare(role, path, strategy, approved, multiplier)
        return {"count": len(views) * multiplier}

    return {"vision_propose": propose, "managed_inference": managed_prelabel, "segment_proposals": segment,
            "vision_review": review, "repair_proposals": repair, "estimate_requests": estimate_requests}
