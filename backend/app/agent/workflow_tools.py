"""Allowlisted workflow proposals use the existing Agent approval boundary."""
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from app.agent.bounds import bound_json, untrusted_text
from app.agent.tools import AgentToolRegistry, AgentToolSpec
from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.storage import Storage
from app.sam.configuration import freeze_sam_configuration
from app.vision.settings import VisionSettingsService
from app.vision.workflow import build_vision_handlers
from app.workflow.bridges import WorkflowBindings
from app.workflow.schemas import RunCreate
from app.workflow.service import WorkflowService


WORKFLOW_WRITES = frozenset({"create_workflow_run", "resume_workflow_run", "cancel_workflow_run"})


class WorkflowControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=64)


def build_workflow_service(storage: Storage, session_factory, settings: Settings | None = None) -> WorkflowService:
    resolved = settings or replace(Settings.from_env(), data_dir=storage.data_dir, import_root=storage.import_root)
    return WorkflowService(session_factory, storage, WorkflowBindings(
        handlers=build_vision_handlers(storage, resolved, session_factory),
        sam_resolver=lambda: freeze_sam_configuration(resolved)), project_root=resolved.project_root)


def normalize_workflow_payload(tool_name: str, raw: dict) -> dict:
    data = {key: value for key, value in raw.items() if key not in {"action", "preview", "reserved_task_id"}}
    if tool_name == "create_workflow_run":
        request = RunCreate.model_validate(data)
        return {**request.model_dump(mode="json"), "action": tool_name,
                "preview": {"template_id": request.template_id, "profile_id": request.profile_id,
                            "dataset_id": request.dataset_id, "image_ids": request.image_ids,
                            "snapshot_id": request.snapshot_id, "next_batches": [batch.model_dump() for batch in request.next_batches],
                            "outputs": request.inputs.get("approved_output_destinations", {}),
                            "vision_approvals": request.inputs.get("vision_approvals", {}),
                            "sam_config": request.inputs.get("sam_config")}}
    control = WorkflowControl.model_validate(data)
    return {**control.model_dump(), "action": tool_name, "preview": {"run_id": control.run_id}}


def register_workflow_tools(registry: AgentToolRegistry, storage: Storage, session_factory,
                            settings: Settings | None = None) -> None:
    if session_factory is None:
        return
    resolved = settings or replace(Settings.from_env(), data_dir=storage.data_dir, import_root=storage.import_root)
    service = build_workflow_service(storage, session_factory, resolved)

    def catalog(session, arguments):
        profiles = [{key: row[key] for key in ("id", "key", "name", "version")} for row in service.list_profiles()[:20]]
        templates = [{"id": row["id"], "key": row["key"], "name": row["name"], "version": row["version"],
                      "nodes": [node["type"] for node in row["nodes"]]} for row in service.list_templates()[:20]]
        return bound_json({"profiles": profiles, "templates": templates, "runs": service.list_runs(10)})

    def status(session, arguments):
        run = service.get_run(WorkflowControl.model_validate(arguments).run_id)
        return bound_json({"id": run["id"], "status": run["status"], "error_message": run["error_message"],
            "dataset_id": run["context"].get("dataset_id"), "parent_run_id": run["parent_run_id"],
            "batch_counts": run["batch_counts"], "reserved_requests": run["reserved_requests"],
            "attempts": [{"node_id": item["node_id"], "status": item["status"], "task_id": item["task_id"],
                          "error_message": untrusted_text(item["error_message"], limit=150),
                          "next_run_id": (item.get("output") or {}).get("next_run_id")} for item in run["attempts"][-20:]],
            "reviews": [{"id": item["id"], "item_id": item["item_id"], "status": item["status"],
                         "revision": item["revision"], "decision": item["decision"],
                         "issue_count": len(item["payload"].get("issues", []))} for item in run["reviews"][:20]],
            "artifact_count": len(run["artifacts"]), "detail_url": f"/api/workflows/runs/{run['id']}"})

    def propose_create(session, arguments):
        request = RunCreate.model_validate(arguments)
        endpoints = VisionSettingsService(resolved).get().model_dump(mode="json")
        prepared = service.prepare_run(request, endpoints)
        return normalize_workflow_payload("create_workflow_run", prepared["request"])

    def propose_resume(session, arguments):
        request = normalize_workflow_payload("resume_workflow_run", arguments)
        service.get_run(request["run_id"])
        return request

    def propose_cancel(session, arguments):
        request = normalize_workflow_payload("cancel_workflow_run", arguments)
        service.get_run(request["run_id"])
        return request

    specs = [
        ("workflow_catalog", "read", "List versioned workflow templates, task profiles and recent runs.", {"type": "object", "properties": {}, "additionalProperties": False}, catalog),
        ("workflow_status", "read", "Read workflow steps, review gates, artifacts and failures.", WorkflowControl.model_json_schema(), status),
        ("create_workflow_run", "write", "Prepare a workflow with exact image scope and endpoint approvals. Human approval starts it.", RunCreate.model_json_schema(), propose_create),
        ("resume_workflow_run", "write", "Propose resuming a workflow using its frozen configuration. Human approval required.", WorkflowControl.model_json_schema(), propose_resume),
        ("cancel_workflow_run", "write", "Propose cancelling a workflow and its managed tasks. Human approval required.", WorkflowControl.model_json_schema(), propose_cancel),
    ]
    for name, kind, description, parameters, handler in specs:
        registry.register(AgentToolSpec(name=name, kind=kind, description=description, parameters=parameters, handler=handler))
