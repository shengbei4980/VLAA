from __future__ import annotations

import base64
import json
from contextlib import nullcontext
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError as SchemaError

from app.agent.llm_auth import build_llm_request_headers
from app.core.errors import ValidationError
from app.core.local_compute import local_compute_gate
from app.vision.images import ImageView
from app.vision.schemas import NativePolygonProposal, VisionEndpointInternal

Result = TypeVar("Result", bound=BaseModel)


def _responses_text(data: dict) -> str:
    """Read a completed, text-only Responses result without treating tools as labels."""
    if data.get("error") is not None or data.get("status") in {"failed", "cancelled"}:
        raise ValidationError("vision_response_failed", "Vision Responses request did not complete successfully.")
    incomplete = data.get("incomplete_details")
    if data.get("status") == "incomplete" or incomplete is not None:
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else None
        if reason == "max_output_tokens":
            raise ValidationError("vision_response_truncated", "Vision reply reached its output limit.")
        if reason == "content_filter":
            raise ValidationError("vision_response_refused", "Vision endpoint declined to provide the requested output.")
        raise ValidationError("vision_response_incomplete", "Vision Responses output is incomplete; no annotations were accepted.")
    if data.get("status") != "completed":
        raise ValueError("Expected a completed Responses object")
    output = data.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses output must be a list")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise ValueError("Invalid Responses output item")
        if item.get("status") not in {None, "completed"}:
            raise ValidationError("vision_response_incomplete", "Vision Responses contains an unfinished output item.")
        if item.get("type") == "reasoning":
            # Reasoning summaries are not annotation JSON, even when first in output.
            continue
        if item.get("type") != "message":
            raise ValidationError("vision_unexpected_tool_output", "Vision endpoint returned non-message output; tool execution is not supported.")
        if item.get("role") != "assistant" or item.get("status") != "completed":
            raise ValidationError("vision_response_incomplete", "Vision assistant message is not a completed output.")
        parts = item.get("content")
        if not isinstance(parts, list):
            raise ValueError("Responses message content must be a list")
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError("Invalid Responses message part")
            if part.get("type") == "refusal":
                raise ValidationError("vision_response_refused", "Vision endpoint declined to provide the requested output.")
            if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise ValueError("Expected Responses output_text")
            texts.append(part["text"])
    text = "".join(texts)
    if not text.strip():
        raise ValueError("Responses contains no output text")
    return text


class OpenAICompatibleVisionProvider:
    """One explicit destination, bounded replies, no implicit provider fallback."""

    def __init__(self, endpoint: VisionEndpointInternal, *, transport: httpx.BaseTransport | None = None, wire_api: Literal["responses", "chat_completions"] = "chat_completions") -> None:
        self.wire_api = wire_api
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
        route = "/chat/completions"
        if self.wire_api == "responses":
            route = "/responses"
            response_content = [
                {"type": "input_text", "text": part["text"]} if part["type"] == "text" else
                {"type": "input_image", "image_url": part["image_url"]["url"]} for part in content
            ]
            wire_format = {"type": endpoint.structured_output}
            if endpoint.structured_output == "json_schema":
                wire_format.update(name=result_type.__name__, schema=schema)
            payload = {"model": endpoint.model, "input": [{"role": "user", "content": response_content}],
                       "text": {"format": wire_format}, "max_output_tokens": endpoint.max_output_tokens,
                       "store": False, "stream": False}
        headers = build_llm_request_headers(endpoint.api_key.get_secret_value(), endpoint.auth_scheme, endpoint.auth_header_name)
        request_id = None
        gate = local_compute_gate.acquire() if endpoint.mode == "local" else nullcontext()
        try:
            with gate, httpx.Client(timeout=endpoint.timeout_seconds, trust_env=False, follow_redirects=False, transport=self.transport) as client:
                with client.stream("POST", endpoint.api_base + route, headers=headers, json=payload) as response:
                    request_id = response.headers.get("x-request-id", "")[:128]
                    if not response.is_success:
                        raise ValidationError("vision_http_error", f"Vision endpoint returned HTTP {response.status_code}.", details={"http_status": response.status_code, "request_id": request_id})
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > 4_000_000:
                            raise ValidationError("vision_response_too_large", "Vision response exceeds the supported limit.")
            data = json.loads(body)
            if self.wire_api == "responses":
                raw = _responses_text(data)
            else:
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
