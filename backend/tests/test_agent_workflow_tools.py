"""Agent HTTP integration tests for approval boundaries, not vision capability."""
import io

from PIL import Image
import pytest
from sqlalchemy import select

from app.agent.provider import ProviderResponse, ProviderToolCall
from app.agent.providers.mock import MockAgentProvider
from app.core.models import AgentApproval
from app.workflow.models import WorkflowEvent, WorkflowRun


@pytest.fixture
def workflow_request(client):
    # Advance the real persisted engine explicitly so these approval tests are deterministic.
    client.app.state.workflow_executor.shutdown()
    dataset = client.post("/api/datasets", json={"name": "Tower approval fixture", "task_type": "detect"}).json()
    label = client.post(f"/api/datasets/{dataset['id']}/classes", json={"name": "component"}).json()
    stream = io.BytesIO()
    Image.new("RGB", (80, 400), "white").save(stream, format="PNG")
    image = client.post(f"/api/datasets/{dataset['id']}/images/upload", files={"files": ("tower.png", stream.getvalue(), "image/png")}).json()["items"][0]
    profile_response = client.post("/api/workflows/profiles", json={"key": "tower-agent", "name": "Tower",
        "config": {"task_type": "detect", "geometry": "bbox", "class_rules": [{"class_id": label["id"], "name": "component"}]}})
    assert profile_response.status_code == 201, profile_response.text
    template = client.post("/api/workflows/templates", json={"key": "human-review", "name": "Review only", "nodes": [
        {"id": "propose", "type": "vision_propose"}, {"id": "confirm", "type": "human_review"}]}).json()
    settings = client.put("/api/vision/settings", json={
        "proposer": {"enabled": True, "mode": "remote", "api_base": "https://vision.example/v1", "model": "fixture-vision"},
        "reviewer": {"enabled": False},
    })
    assert settings.status_code == 200, settings.text
    client.app.state.workflow_service.bindings.handlers["vision_propose"] = lambda context: {
        "annotations": [{"object_id": "component", "class_id": label["id"], "type": "bbox",
                         "bbox": {"x": 5, "y": 10, "width": 20, "height": 30}, "source": "auto"}]}
    client.app.state.workflow_service.bindings.handlers["estimate_requests"] = lambda _: {"count": 0}
    return {"idempotency_key": "agent-fixture", "template_id": template["id"], "profile_id": profile_response.json()["id"],
            "dataset_id": dataset["id"], "image_ids": [image["id"]]}


def propose(client, monkeypatch, tool_name, arguments, *, read_only=False):
    def complete(self, request):
        if request.messages[-1].role == "tool":
            return ProviderResponse(content="The fixture tool returned.")
        return ProviderResponse(content="Review the operation.", tool_calls=[ProviderToolCall(name=tool_name, arguments=arguments, call_id="fixture-call")])
    monkeypatch.setattr(MockAgentProvider, "complete", complete)
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
                           json={"content": "Use the selected workflow", "read_only": read_only})
    assert response.status_code == 200, response.text
    return response.json()


def test_workflow_query_uses_catalog_without_creating_run(client, workflow_request):
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true", json={"content": "查看有哪些工作流模板"})
    assert response.status_code == 200, response.text
    assert response.json()["tool_calls"][0]["name"] == "workflow_catalog"
    assert response.json()["approvals"] == []
    assert client.get("/api/workflows/runs").json() == []


def test_mock_workflow_status_returns_bounded_live_evidence(client, workflow_request):
    prepared = client.post("/api/workflows/runs/prepare", json=workflow_request).json()
    workflow_id = client.post("/api/workflows/runs", json=prepared["request"]).json()["id"]
    session_id = client.post("/api/agent/sessions", json={}).json()["id"]
    response = client.post(f"/api/agent/sessions/{session_id}/messages?wait_for_completion=true",
                           json={"content": f"查看工作流 {workflow_id} 当前状态"})
    assert response.status_code == 200, response.text
    tool = response.json()["tool_calls"][0]
    assert tool["name"] == "workflow_status"
    result = tool["result_json"]
    # Agent tool protocol wraps successful results with their canonical name.
    if "result" in result:
        result = result["result"]
    assert result["id"] == workflow_id
    assert result["status"] == "pending"
    assert result["batch_counts"]["total"] == 1


def test_workflow_create_is_exact_prepared_approval_then_idempotent_execution(client, monkeypatch, workflow_request):
    run = propose(client, monkeypatch, "create_workflow_run", workflow_request)
    assert run["status"] == "awaiting_approval", run
    assert client.get("/api/workflows/runs").json() == []
    approval = run["approvals"][0]
    payload = approval["payload_json"]
    approved = payload["inputs"]["vision_approvals"]["proposer"]
    assert approved["api_base"] == "https://vision.example/v1"
    assert len(approved["allowed_image_hashes"]) == 1
    assert payload["image_ids"] == workflow_request["image_ids"]
    response = client.post(f"/api/agent/approvals/{approval['id']}/approve")
    assert response.status_code == 200, response.text
    result = response.json()["approval"]
    workflow_id = result["result_task_id"]
    assert workflow_id.startswith("wrun_")
    detail = client.get(f"/api/workflows/runs/{workflow_id}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["context"]["vision_approvals"] == payload["inputs"]["vision_approvals"]
    second = client.post(f"/api/agent/approvals/{approval['id']}/approve")
    assert second.status_code == 200
    assert second.json()["approval"]["result_task_id"] == workflow_id
    assert len(client.get("/api/workflows/runs").json()) == 1
    service = client.app.state.workflow_service
    for _ in range(6):
        service.tick(workflow_id)
    assert service.get_run(workflow_id)["status"] == "awaiting_review"


def test_read_only_workflow_write_cannot_create_approval_or_run(client, monkeypatch, workflow_request):
    run = propose(client, monkeypatch, "create_workflow_run", workflow_request, read_only=True)
    assert run["approvals"] == []
    assert run["tool_calls"][0]["result_json"]["error"] == "read_only_run"
    assert client.get("/api/workflows/runs").json() == []


def test_approval_rejects_changed_destination_without_creating_workflow(client, monkeypatch, workflow_request):
    run = propose(client, monkeypatch, "create_workflow_run", workflow_request)
    approval = run["approvals"][0]
    client.put("/api/vision/settings", json={"proposer": {"enabled": True, "mode": "remote",
        "api_base": "https://different.example/v1", "model": "fixture-vision"}, "reviewer": {"enabled": False}})
    response = client.post(f"/api/agent/approvals/{approval['id']}/approve")
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "workflow_approval_scope_changed"
    assert client.get("/api/workflows/runs").json() == []


def test_reserved_workflow_created_before_finalize_recovers_actual_id(client, monkeypatch, workflow_request):
    run = propose(client, monkeypatch, "create_workflow_run", workflow_request)
    approval = run["approvals"][0]
    response = client.post(f"/api/agent/approvals/{approval['id']}/approve")
    assert response.status_code == 200, response.text
    workflow_id = response.json()["approval"]["result_task_id"]
    with client.app.state.database.session_factory() as session:
        row = session.get(AgentApproval, approval["id"])
        row.status = "approved"
        row.result_task_id = row.payload_json["reserved_task_id"]
        session.commit()
    recovered = client.post(f"/api/agent/approvals/{approval['id']}/approve")
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["approval"]["result_task_id"] == workflow_id
    assert client.get(f"/api/workflows/runs/{workflow_id}").status_code == 200
    assert len(client.get("/api/workflows/runs").json()) == 1


@pytest.mark.parametrize("action", ["resume_workflow_run", "cancel_workflow_run"])
def test_workflow_controls_need_approval_and_recover_same_operation(client, monkeypatch, workflow_request, action):
    prepared = client.post("/api/workflows/runs/prepare", json=workflow_request)
    assert prepared.status_code == 200, prepared.text
    created = client.post("/api/workflows/runs", json=prepared.json()["request"])
    assert created.status_code == 202, created.text
    workflow_id = created.json()["id"]
    run = propose(client, monkeypatch, action, {"run_id": workflow_id})
    assert run["status"] == "awaiting_approval", run
    before = client.get(f"/api/workflows/runs/{workflow_id}").json()
    assert before["status"] == "pending"
    approval_id = run["approvals"][0]["id"]
    response = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["result_task_id"] == workflow_id
    with client.app.state.database.session_factory() as session:
        approval = session.get(AgentApproval, approval_id)
        operation_id = approval.payload_json["reserved_task_id"]
        assert session.get(WorkflowEvent, operation_id) is not None
        approval.status = "approved"
        approval.result_task_id = operation_id
        session.commit()
    recovered = client.post(f"/api/agent/approvals/{approval_id}/approve")
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["approval"]["result_task_id"] == workflow_id
    with client.app.state.database.session_factory() as session:
        markers = list(session.scalars(select(WorkflowEvent).where(WorkflowEvent.id == operation_id)))
        assert len(markers) == 1


def test_large_approval_payload_is_not_replaced_by_report_truncation(client, monkeypatch, workflow_request):
    request = {**workflow_request, "inputs": {"note": "x" * 13000}}
    run = propose(client, monkeypatch, "create_workflow_run", request)
    assert run["status"] == "awaiting_approval", run
    assert run["approvals"][0]["payload_json"]["inputs"]["note"] == "x" * 13000
