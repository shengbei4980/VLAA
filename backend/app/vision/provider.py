from __future__ import annotations

import base64
import json
from contextlib import nullcontext
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError as SchemaError

from app.agent.llm_auth import build_llm_request_headers
from app.core.errors import ValidationError
from app.core.local_compute import local_compute_gate
from app.vision.images import ImageView
from app.vision.schemas import NativePolygonProposal, VisionEndpointInternal

Result = TypeVar("Result", bound=BaseModel)


class OpenAICompatibleVisionProvider:
    """One explicit destination, bounded replies, no implicit provider fallback."""

    def __init__(self, endpoint: VisionEndpointInternal, *, transport: httpx.BaseTransport | None = None) -> None:
        self.endpoint = endpoint
        self.transport = transport

    def complete(self, prompt: str, views: list[ImageView], result_type: type[Result]) -> tuple[Result, dict]:
        endpoint = self.endpoint
        if not endpoint.enabled or not endpoint.api_base or not endpoint.model:
            raise ValidationError("vision_endpoint_unavailable", "Select and enable a vision endpoint and model first.")
        if len(views) > 1 and not endpoint.multi_image:
            raise ValidationError("vision_multi_image_unsupported", "The selected endpoint has not declared multi-image support.")
        if not views:
            raise ValidationError("vision_image_required", "Vision requests require a declared image coordinate frame.")
        frame = views[0]
        if any((view.original_width, view.original_height, view.crop, view.sent_width, view.sent_height)
               != (frame.original_width, frame.original_height, frame.crop, frame.sent_width, frame.sent_height)
               for view in views[1:]):
            raise ValidationError("vision_image_frames_mismatch", "Images in one vision request must share a single coordinate frame.")
        content: list[dict] = [{"type": "text", "text": prompt}]
        for view in views:
            content.extend([
                {"type": "text", "text": json.dumps(view.metadata(), separators=(",", ":"))},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(view.content).decode("ascii")}},
            ])
        schema = result_type.model_json_schema()
        response_format: dict = {"type": endpoint.structured_output}
        if endpoint.structured_output == "json_schema":
            response_format["json_schema"] = {"name": result_type.__name__, "schema": schema}
        payload = {
            "model": endpoint.model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": endpoint.max_output_tokens,
            "response_format": response_format,
        }
        payload["messages"][0]["content"][0]["text"] += "\nReturn one JSON object matching this schema.\n" + json.dumps(schema)
        bounds = {"pixel": [frame.sent_width, frame.sent_height], "normalized_1": [1, 1], "normalized_1000": [1000, 1000]}[endpoint.coordinate_format]
        polygon_output = result_type is NativePolygonProposal
        shape_contract = (
            "Return polygon as ordered [x,y] boundary vertices with frame=sent_image, x rightward and y downward. "
            f"Every vertex must satisfy 0 <= x <= {bounds[0]} and 0 <= y <= {bounds[1]}. "
            "Require at least three distinct vertices without crossing edges. Do not output bbox as a replacement contour. "
            if polygon_output else
            "Return bbox as [left,top,right,bottom] with frame=sent_image, x rightward and y downward. "
            f"Require 0 <= left < right <= {bounds[0]} and 0 <= top < bottom <= {bounds[1]}. "
        )
        content.append({"type": "text", "text": (
            f"\nFinal coordinate contract. This contract governs every returned {'polygon vertex' if polygon_output else 'bbox'} and takes precedence over "
            "coordinate wording in task descriptions, examples and image metadata. "
            + shape_contract +
            f"The actual sent image is {frame.sent_width} pixels wide and {frame.sent_height} pixels high. "
            f"coordinate_format={endpoint.coordinate_format}. "
            "All supplied images and overlays share this geometry. Use only the current sent image frame. "
            "Original-image sizes and crop offsets describe internal transforms and must not be used as output coordinates."
        )})
        headers = build_llm_request_headers(endpoint.api_key.get_secret_value(), endpoint.auth_scheme, endpoint.auth_header_name)
        request_id = None
        gate = local_compute_gate.acquire() if endpoint.mode == "local" else nullcontext()
        try:
            with gate, httpx.Client(timeout=endpoint.timeout_seconds, trust_env=False, follow_redirects=False, transport=self.transport) as client:
                with client.stream("POST", endpoint.api_base + "/chat/completions", headers=headers, json=payload) as response:
                    request_id = response.headers.get("x-request-id", "")[:128]
                    if not response.is_success:
                        raise ValidationError("vision_http_error", f"Vision endpoint returned HTTP {response.status_code}.", details={"http_status": response.status_code, "request_id": request_id})
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 4_000_000:
                            raise ValidationError("vision_response_too_large", "Vision response exceeds the supported limit.")
            data = json.loads(body)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValidationError("vision_response_truncated", "Vision reply reached its output limit.")
            raw = choice["message"]["content"]
            parsed = result_type.model_validate_json(raw)
        except httpx.TimeoutException as exc:
            raise ValidationError("vision_timeout", "Vision request timed out. The remote outcome may be unknown.", details={"request_id": request_id, "outcome_unknown": True}) from exc
        except httpx.RequestError as exc:
            raise ValidationError("vision_request_failed", "Vision endpoint request failed.", details={"request_id": request_id, "outcome_unknown": True}) from exc
        except (ValueError, KeyError, IndexError, TypeError, SchemaError) as exc:
            raise ValidationError("vision_invalid_response", "Vision response does not match the required structured contract.", details={"request_id": request_id}) from exc
        usage = data.get("usage", {})
        safe_usage = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if isinstance(usage, dict) and type(usage.get(key)) is int}
        return parsed, {"mode": endpoint.mode, "api_base": endpoint.api_base, "model": endpoint.model, "config_hash": endpoint.config_hash, "request_id": request_id, "usage": safe_usage}
