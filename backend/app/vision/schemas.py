from __future__ import annotations

import math
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from app.core.schemas import BBox

VisionRole = Literal["proposer", "reviewer"]
CoordinateFormat = Literal["pixel", "normalized_1", "normalized_1000"]
VisionGeometry = Literal["bbox", "polygon"]
MAX_VISION_POLYGON_POINTS = 1024


class VisionEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode: Literal["remote", "local"] = "local"
    api_base: str = Field(default="", max_length=2048)
    model: str = Field(default="", max_length=256)
    auth_scheme: Literal["bearer", "header", "raw_authorization"] = "bearer"
    auth_header_name: str = ""
    coordinate_format: CoordinateFormat = "pixel"
    max_output_tokens: int = Field(default=4096, ge=128, le=32768)
    timeout_seconds: int = Field(default=60, ge=1, le=7200)
    multi_image: bool = False
    structured_output: Literal["json_schema", "json_object"] = "json_object"

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_image_limit(cls, value):
        # Existing settings and older clients may still contain this retired key.
        # It must never control image geometry or appear in the public schema.
        if isinstance(value, dict) and "max_image_side" in value:
            return {key: item for key, item in value.items() if key != "max_image_side"}
        return value

    @field_validator("api_base")
    @classmethod
    def safe_endpoint(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value:
            return value
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use an HTTP(S) API base without embedded credentials, query, or fragment.")
        return value


class VisionEndpointUpdate(VisionEndpoint):
    api_key: SecretStr | None = Field(default=None, repr=False)


class VisionEndpointOut(VisionEndpoint):
    api_key_configured: bool = False
    config_hash: str = ""


class VisionEndpointInternal(VisionEndpoint):
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)
    config_hash: str = ""


class VisionSettingsUpdate(BaseModel):
    proposer: VisionEndpointUpdate
    reviewer: VisionEndpointUpdate


class VisionSettingsOut(BaseModel):
    proposer: VisionEndpointOut
    reviewer: VisionEndpointOut


class VisionApproval(BaseModel):
    mode: Literal["remote", "local"]
    api_base: str
    model: str
    config_hash: str
    allowed_image_hashes: list[str] = Field(min_length=1, max_length=100000)
    max_requests: int = Field(ge=1, le=10000)


class ImageStrategy(BaseModel):
    tile_size: int | None = Field(default=None, ge=64, le=4096)
    overlap: float = Field(default=0.2, ge=0, lt=0.75)
    merge_iou: float = Field(default=0.6, gt=0, le=1)
    include_overview: bool = True
    max_views: int = Field(default=64, ge=1, le=256)


class VisionClassRule(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    definition: str = Field(default="", max_length=10000)
    positive_examples: list[str] = Field(default_factory=list, max_length=30)
    negative_examples: list[str] = Field(default_factory=list, max_length=30)


class NativeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: str | None
    # ImageView validates each numeric box so malformed geometry can reach human review.
    bbox: tuple[float, float, float, float] = Field(description="[left, top, right, bottom] in the current sent-image coordinate protocol, with positive area.")
    evidence: str = Field(min_length=1, max_length=2000)
    uncertainty: str = Field(default="", max_length=2000)
    confidence: float | None = Field(default=None, ge=0, le=1)

class NativeProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[NativeCandidate] = Field(default_factory=list, max_length=1000)
    uncertainty: str = Field(default="", max_length=4000)


class NativePolygonCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_id: str | None
    polygon: list[tuple[float, float]] | None = Field(default=None, max_length=MAX_VISION_POLYGON_POINTS,
        description="Ordered [x,y] vertices tracing the visible object boundary in SENT-image coordinates, at least three distinct vertices. Never substitute an enclosing rectangle.")
    evidence: str = Field(min_length=1, max_length=2000)
    uncertainty: str = Field(default="", max_length=2000)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def isolate_malformed_geometry(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            # A provider may return a box despite the polygon-only schema. Keep
            # the entry as a visible invalid-contour issue, never a box fallback.
            value.pop("bbox", None)
            points = value.get("polygon")
            try:
                malformed = (not isinstance(points, (list, tuple)) or len(points) > MAX_VISION_POLYGON_POINTS
                    or any(not isinstance(point, (list, tuple)) or len(point) != 2
                           or any(type(number) not in (int, float) or not math.isfinite(number) for number in point) for point in points))
            except OverflowError:
                malformed = True
            if malformed:
                value["polygon"] = None
        return value


class NativePolygonProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[NativePolygonCandidate] = Field(default_factory=list, max_length=1000)
    uncertainty: str = Field(default="", max_length=4000)


class VisionObject(BaseModel):
    object_id: str
    class_id: str | None
    type: Literal["bbox", "polygon"] = "bbox"
    bbox: BBox
    polygon: list[tuple[float, float]] | None = None
    mask_ref: str | None = None
    source: Literal["manual", "imported", "sam", "auto"] = "auto"
    evidence: str = ""
    uncertainty: str = ""
    confidence: float | None = None
    provenance_json: dict = Field(default_factory=dict)


class NativeIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["missing", "wrong_class", "boundary", "split", "merge", "excluded_region", "uncertain"]
    object_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    class_id: str | None = None
    evidence: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def located_issue(self) -> "NativeIssue":
        if self.object_id is None and self.bbox is None:
            raise ValueError("An issue must identify an object or image region.")
        return self


class NativeReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issues: list[NativeIssue] = Field(default_factory=list, max_length=1000)
    uncertainty: str = Field(default="", max_length=4000)


class VisionProposalResult(BaseModel):
    image_id: str
    image_hash: str
    ruleset_version: str
    objects: list[VisionObject]
    issues: list[dict] = Field(default_factory=list)
    provenance_json: dict
    request_count: int


class VisionReviewResult(BaseModel):
    image_id: str
    image_hash: str
    revision: int
    issues: list[dict]
    inventory: list[VisionObject]
    provenance_json: dict
    request_count: int


class VisionProbeResult(BaseModel):
    ok: bool
    code: str
    message: str
    mode: str
    model: str
    remote_test_performed: bool = False
    coordinate_verified: bool = False
    real_image_quality_verified: bool = False
