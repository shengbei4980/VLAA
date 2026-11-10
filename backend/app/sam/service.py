from __future__ import annotations

import math
import threading
from gc import collect
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.ids import new_id
from app.core.local_compute import local_compute_gate
from app.core.models import ClassLabel, Dataset
from app.core.schemas import BBox
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_bbox, validate_polygon
from app.dataset.images import get_image
from app.dataset.annotation.masks import MaskService
from app.sam.schemas import SamMaskPredictResponse, SamPredictRequest, SamPredictResponse
from app.sam.configuration import verify_sam_configuration
from app.settings.service import SamSettingsService


_MODEL_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str, str | None], object] = {}


def _box_polygon(box: BBox) -> list[tuple[float, float]]:
    return [
        (box.x, box.y),
        (box.x + box.width, box.y),
        (box.x + box.width, box.y + box.height),
        (box.x, box.y + box.height),
    ]


def clear_model_cache() -> None:
    """Release cached SAM models after a local SAM configuration change."""

    with _MODEL_LOCK:
        _MODELS.clear()
    collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, OSError, AttributeError, RuntimeError):
        pass


def _model_for(model_reference: str, device: str, weights_sha256: str | None = None) -> object:
    key = (model_reference, device, weights_sha256)
    with _MODEL_LOCK:
        model = _MODELS.get(key)
        if model is not None:
            return model
        # The application supports one interactive SAM configuration at a
        # time. Retaining past checkpoints here would keep their CPU/GPU
        # allocations alive after the user switches settings.
        _MODELS.clear()
        try:
            from ultralytics import SAM

            model = SAM(model_reference)
        except Exception as exc:
            raise ValidationError("sam_model_unavailable", f"SAM model could not be loaded: {exc}") from exc
        _MODELS[key] = model
        return model


def _predict_with_sam(
    image_path: Path,
    request: SamPredictRequest,
    model_reference: str,
    device: str,
    img_size: int,
) -> tuple[list[tuple[float, float]], float, str | None]:
    try:
        model = _model_for(model_reference, device)
    except TypeError as exc:
        # Keep the small helper easy to replace in tests and by local plugins
        # that still implement the original one-argument hook.
        try:
            model = _model_for(model_reference)  # type: ignore[call-arg]
        except TypeError:
            raise exc
    kwargs: dict[str, object] = {"source": str(image_path), "imgsz": img_size, "verbose": False}
    if device != "auto":
        kwargs["device"] = device
    if request.box is not None:
        kwargs["bboxes"] = [[request.box.x, request.box.y, request.box.x + request.box.width, request.box.y + request.box.height]]
    if request.points:
        kwargs["points"] = [[point.x, point.y] for point in request.points]
        kwargs["labels"] = [point.label for point in request.points]
    try:
        results = model.predict(**kwargs)  # type: ignore[attr-defined]
        if not results or getattr(results[0], "masks", None) is None or not results[0].masks.xy:
            raise ValueError("SAM returned no masks for this prompt")
        points = [(float(x), float(y)) for x, y in results[0].masks.xy[0].tolist()]
        score = 1.0
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf):
            score = float(boxes.conf[0].detach().cpu().item())
        # The prediction result is the authoritative source of the device.
        # In particular, reporting the configured value "auto" would be
        # misleading because Ultralytics selects a concrete device at runtime.
        for prediction_data in (getattr(results[0].masks, "data", None), getattr(boxes, "data", None)):
            device = getattr(prediction_data, "device", None)
            if device is not None:
                return points, score, str(device)
        return points, score, None
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("sam_prediction_failed", f"SAM prediction failed: {exc}") from exc


class SamService:
    """Domain service for local SAM-assisted polygon proposals.

    The proposal is deliberately not persisted here. It is returned to the
    annotation editor for review, then saved through the normal annotation
    transaction with ``source='sam'``.
    """

    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings

    def _context(self, session: Session, request: SamPredictRequest):
        image = get_image(session, request.image_id)
        dataset = session.get(Dataset, image.dataset_id)
        if dataset is None or dataset.task_type != "segment":
            raise ValidationError("sam_requires_segment_dataset", "SAM proposals can only be saved as polygon annotations in segment datasets.")
        class_label = session.get(ClassLabel, request.class_id)
        if class_label is None or class_label.dataset_id != image.dataset_id:
            raise ValidationError("class_not_in_dataset", "SAM class does not belong to the image dataset.")
        if request.box is not None:
            validate_bbox(request.box, image.width, image.height)
        if any(
            not math.isfinite(point.x)
            or not math.isfinite(point.y)
            or point.x < 0
            or point.y < 0
            or point.x > image.width
            or point.y > image.height
            for point in request.points
        ):
            raise ValidationError("sam_point_out_of_bounds", "SAM prompt points must stay inside the image.")
        path = self.storage.image_path(image.dataset_id, image.storage_name)
        if not path.is_file():
            raise ValidationError("image_file_missing", "The managed image file is missing.")

        sam = SamSettingsService(self.settings).get()
        if not sam.enabled:
            raise ValidationError("sam_disabled", "SAM 辅助标注已在设置中关闭。")
        return image, class_label, path, sam

    def predict(self, session: Session, request: SamPredictRequest) -> SamPredictResponse:
        image, class_label, path, sam = self._context(session, request)
        if sam.model:
            with local_compute_gate.acquire():
                polygon, score, device = _predict_with_sam(path, request, sam.model, sam.device, sam.img_size)
            backend_used = "ultralytics_sam"
        elif request.box is not None and sam.fallback_mode == "box":
            polygon, score, device, backend_used = _box_polygon(request.box), 1.0, None, "box_stub"
        else:
            raise ValidationError(
                "sam_model_not_configured",
                "点提示需要在设置中配置本地或 Ultralytics SAM 权重。",
            )
        validate_polygon(polygon, image.width, image.height)
        return SamPredictResponse(
            image_id=image.id,
            class_id=class_label.id,
            mask_id=new_id("mask"),
            polygon=polygon,
            score=score,
            backend_used=backend_used,
            device=device,
        )

    def predict_mask(self, session: Session, request: SamPredictRequest, *, expected_config: dict | None = None) -> SamMaskPredictResponse:
        image, class_label, path, sam = self._context(session, request)
        if not sam.model:
            raise ValidationError("sam_model_not_configured", "A real configured SAM model is required for authoritative masks.")
        with local_compute_gate.acquire():
            try:
                inference_device, inference_size = sam.device, sam.img_size
                if expected_config is not None:
                    frozen = verify_sam_configuration(self.settings, expected_config, require_weights=True)
                    inference_device, inference_size = frozen["device"], frozen["img_size"]
                    model = _model_for(frozen["weights_path"], inference_device, frozen["weights_sha256"])
                else:
                    model = _model_for(sam.model, sam.device)
                kwargs: dict[str, object] = {"source": str(path), "imgsz": inference_size, "verbose": False}
                if inference_device != "auto":
                    kwargs["device"] = inference_device
                if request.box is not None:
                    box = request.box
                    kwargs["bboxes"] = [[box.x, box.y, box.x + box.width, box.y + box.height]]
                if request.points:
                    kwargs["points"] = [[point.x, point.y] for point in request.points]
                    kwargs["labels"] = [point.label for point in request.points]
                results = model.predict(**kwargs)
                if not results or getattr(results[0], "masks", None) is None:
                    raise ValueError("SAM returned no raw mask")
                result = results[0]
                data = result.masks.data
                raw = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
                if raw.ndim != 3 or len(raw) != 1 or raw.shape[1:] != (image.height, image.width):
                    raise ValueError("SAM must return exactly one full-resolution mask for this object prompt")
                artifact = MaskService(self.storage).write(raw[0])
                if not artifact.area:
                    raise ValidationError("sam_empty_mask", "SAM returned an empty object mask.")
                score = None
                boxes = getattr(result, "boxes", None)
                confidence = getattr(boxes, "conf", None)
                if confidence is not None and len(confidence):
                    value = confidence[0]
                    score = float(value.item() if hasattr(value, "item") else value)
                    del value
                    if not math.isfinite(score):
                        score = None
                device = str(data.device) if getattr(data, "device", None) is not None else None
                polygon = None
                contours = result.masks.xy
                if contours is not None and len(contours):
                    points = [(float(x), float(y)) for x, y in contours[0].tolist()]
                    if len(points) >= 3:
                        polygon = points
                del model, results, result, data, boxes, confidence
                if expected_config is not None:
                    verify_sam_configuration(self.settings, expected_config, require_weights=True)
            except ValidationError:
                raise
            except Exception as exc:
                raise ValidationError("sam_prediction_failed", "SAM raw-mask prediction failed.") from exc
        return SamMaskPredictResponse(
            image_id=image.id, class_id=class_label.id, mask_ref=artifact.mask_ref,
            polygon=polygon, score=score, device=device,
        )
