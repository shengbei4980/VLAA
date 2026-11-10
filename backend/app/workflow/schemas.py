from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputSettings(Contract):
    destination_id: str | None = None
    annotations_destination_id: str | None = None
    models_destination_id: str | None = None
    evaluations_destination_id: str | None = None
    annotations: bool = True
    models: bool = True
    evaluations: bool = True
    include_images: bool = True


class TaskProfileConfig(Contract):
    task_type: Literal["detect", "segment", "obb", "classify"] = "detect"
    geometry: Literal["bbox", "mask", "obb", "classify"] = "bbox"
    class_rules: list[dict[str, Any]] = Field(default_factory=list, max_length=1000)
    rules: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    image_strategy: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    quality_conditions: dict[str, float] = Field(default_factory=dict)
    repair_limit: int = Field(default=0, ge=0, le=3)
    max_requests: int = Field(default=1000, ge=1, le=100000)
    export_format: Literal["yolo", "coco", "labelme"] = "yolo"
    outputs: OutputSettings = Field(default_factory=OutputSettings)

    @field_validator("rules")
    @classmethod
    def valid_task_instructions(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "task_instructions" in value:
            instructions = value["task_instructions"]
            if not isinstance(instructions, str) or len(instructions) > 20000:
                raise ValueError("LLM task instructions must be a string of at most 20000 characters.")
        return value

    @model_validator(mode="after")
    def matching_geometry(self):
        expected = {"detect": "bbox", "segment": "mask", "obb": "obb", "classify": "classify"}
        if self.geometry != expected[self.task_type]:
            raise ValueError("Task type and annotation geometry must match.")
        return self


class ProfileCreate(Contract):
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    config: TaskProfileConfig


class NodeDefinition(Contract):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    type: str = Field(min_length=1, max_length=64)
    version: int = Field(default=1, ge=1)
    scope: Literal["item", "batch"] = "item"
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    routes: dict[str, str] = Field(default_factory=dict)
    task_types: list[Literal["detect", "segment", "obb", "classify"]] = Field(default_factory=list)
    max_attempts: int = Field(default=2, ge=1, le=5)


class TemplateCreate(Contract):
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    nodes: list[NodeDefinition] = Field(min_length=1, max_length=40)


class BatchInput(Contract):
    dataset_id: str = Field(min_length=1, max_length=64)
    image_ids: list[str] = Field(min_length=1, max_length=10000)


class RunCreate(Contract):
    idempotency_key: str = Field(min_length=1, max_length=128)
    template_id: str
    profile_id: str
    dataset_id: str
    project_id: str | None = None
    image_ids: list[str] = Field(default_factory=list, max_length=10000)
    snapshot_id: str | None = None
    next_batches: list[BatchInput] = Field(default_factory=list, max_length=100)
    inputs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_images(self):
        if len(self.image_ids) != len(set(self.image_ids)):
            raise ValueError("Image selection must not contain duplicates.")
        return self


class ReviewDraft(Contract):
    expected_revision: int = Field(ge=1)
    payload: dict[str, Any]


class ReviewDecision(Contract):
    expected_revision: int = Field(ge=1)
    decision: Literal["approved", "background", "rejected", "excluded"]
    reason: str = Field(default="", max_length=2000)
    semantic_resolutions: dict[str, Any] = Field(default_factory=dict)
    actor: str = Field(default="local-user", min_length=1, max_length=100)


class RetryRequest(Contract):
    item_ids: list[str] = Field(default_factory=list, max_length=10000)


class ProjectCreate(Contract):
    name: str = Field(min_length=1, max_length=255)
    root_path: str = Field(min_length=1, max_length=4096)


class BatchExportRequest(Contract):
    include_previews: bool = True


class VersionResponse(Contract):
    id: str
    key: str
    name: str
    version: int
    created_at: datetime
