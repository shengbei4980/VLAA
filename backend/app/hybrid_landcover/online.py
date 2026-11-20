"""One online proposal per role for the manually reviewed Sentinel-2 recipe."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError as SchemaError

from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.local_compute import local_compute_gate
from app.dataset.annotation.masks import encode_mask
from app.sam.service import _model_for
from app.vision.images import ImageView
from app.vision.provider import OpenAICompatibleVisionProvider
from app.vision.schemas import CoordinateFormat, NativePolygonProposal, NativeProposal, VisionClassRule, VisionEndpointInternal
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService

from .core import CLASSES


class EndpointChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    saved_role: Literal["proposer", "reviewer"] | None = None
    api_base: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=256)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), max_length=32768)
    protocol: Literal["responses", "chat_completions"]
    coordinate_format: CoordinateFormat = "pixel"


def resolve_endpoint(raw: str, settings: Settings):
    try:
        choice = EndpointChoice.model_validate_json(raw)
        if choice.saved_role:
            if choice.api_base or choice.model or choice.api_key.get_secret_value():
                raise ValueError("A saved credential cannot be redirected")
            endpoint = VisionSettingsService(settings).internal(choice.saved_role)
        else:
            endpoint = VisionEndpointInternal(
                enabled=True, mode="remote", api_base=choice.api_base, model=choice.model,
                api_key=choice.api_key, coordinate_format=choice.coordinate_format,
                max_output_tokens=16384, timeout_seconds=400,
            )
        if not endpoint.enabled or not endpoint.api_base or not endpoint.model.strip():
            raise ValueError("An enabled endpoint and model are required")
    except (ValueError, SchemaError):
        # Pydantic's validation input can contain a secret. Never log it.
        raise ValidationError("hybrid_endpoint_invalid", "Choose an enabled saved role or supply a complete endpoint and model.") from None
    return endpoint, choice.protocol


@lru_cache(maxsize=1)
def recipes():
    return json.loads(Path(__file__).with_name("prompts.json").read_text(encoding="utf-8"))


def propose(geometry, endpoint_choice, view):
    endpoint, protocol = endpoint_choice
    recipe = recipes()[geometry]
    rules = [VisionClassRule.model_validate(item) for item in recipe["classes"]]
    version = "panan-s2-v6" if geometry == "polygon" else "panan-s2-v4"
    prompt = VisionService._prompt(rules, version, endpoint.coordinate_format,
                                   task_instructions=recipe["instructions"], geometry=geometry)
    schema = NativePolygonProposal if geometry == "polygon" else NativeProposal
    reply, _ = OpenAICompatibleVisionProvider(endpoint, wire_api=protocol).complete(prompt, [view], schema)
    if len(reply.objects) > 256:
        raise ValidationError("hybrid_too_many_objects", "The remote proposal exceeds the 256-object limit for one tile.")
    return reply


def segment_box(model, image_path, box, config, expected_shape):
    kwargs = {"source": str(image_path), "imgsz": config["img_size"], "verbose": False,
              "bboxes": [[box.x, box.y, box.x + box.width, box.y + box.height]]}
    if config["device"] != "auto":
        kwargs["device"] = config["device"]
    try:
        outputs = model.predict(**kwargs)
        masks = outputs[0].masks if outputs else None
        if masks is None:
            return None
        data = masks.data
        raw = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        if raw.shape != (1, *expected_shape):
            raise ValidationError("hybrid_sam_geometry", "SAM returned masks on a different grid from the RGB tile.")
        return raw[0] > 0 if raw.any() else None
    except ValidationError:
        raise
    except Exception:
        raise ValidationError("hybrid_sam_failed", "The local SAM prediction failed. Check the configured weights and device.") from None


def online_drafts(source: dict, image_path: Path, normalized_png: bytes, gpt, deepseek, sam_config):
    width, height = source["imageWidth"], source["imageHeight"]
    view = ImageView("tile", normalized_png, width, height, (0, 0, width, height), width, height, True)
    # Check SAM can load before either remote request is sent.
    with local_compute_gate.acquire():
        model = _model_for(sam_config["weights_path"], sam_config["device"], sam_config["weights_sha256"])
    polygons = propose("polygon", gpt, view)
    boxes = propose("bbox", deepseek, view)
    primary = {**source, "shapes": []}
    secondary = {**source, "shapes": []}
    warnings = []
    for index, item in enumerate(polygons.objects, start=1):
        if item.class_id not in CLASSES:
            warnings.append(f"GPT polygon {index}: uncertain class; omitted.")
            continue
        try:
            points = view.original_polygon(item.polygon, gpt[0].coordinate_format)
        except ValidationError:
            warnings.append(f"GPT polygon {index}: invalid contour; omitted.")
            continue
        primary["shapes"].append({"label": item.class_id, "shape_type": "polygon",
                                  "points": points, "group_id": index, "flags": {}})
    with local_compute_gate.acquire():
        for index, item in enumerate(boxes.objects, start=1):
            if item.class_id not in CLASSES:
                warnings.append(f"DeepSeek box {index}: uncertain class; omitted.")
                continue
            try:
                box = view.original_box(item.bbox, deepseek[0].coordinate_format)
            except ValidationError:
                warnings.append(f"DeepSeek box {index}: invalid coordinates; omitted.")
                continue
            mask = segment_box(model, image_path, box, sam_config, (height, width))
            if mask is None:
                warnings.append(f"SAM box {index}: empty mask; omitted.")
                continue
            secondary["shapes"].append({"label": item.class_id, "shape_type": "mask", "starter_rle": encode_mask(mask)})
    return primary, secondary, warnings
