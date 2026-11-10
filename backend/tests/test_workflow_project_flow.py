"""Project choices flow through preflight, saved reviews, and later batches."""
import io
import json
from pathlib import Path
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.api.error_handlers import register_error_handlers
from app.core.errors import ConflictError
from app.storage_settings.bootstrap import read_config, write_config
from app.workflow.artifacts import fingerprint
from app.workflow.models import WorkflowRun
from app.workflow.router import get_workflow_service, router
from app.workflow.schemas import NodeDefinition, RunCreate, TemplateCreate
from test_workflow import drain, workflow


@pytest.fixture
def project_flow(workflow, tmp_path):
    service, database, dataset_id, images, profile, _, _ = workflow
    service.project_root = tmp_path / "workbench"
    service.project_root.mkdir()
    template = service.create_template(TemplateCreate(key="project-annotations", name="项目批量标注", nodes=[
        NodeDefinition(id="propose", type="vision_propose"),
        NodeDefinition(id="human", type="human_review"),
    ]))
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_workflow_service] = lambda: service
    request = RunCreate(idempotency_key="project-flow", template_id=template["id"],
                        profile_id=profile["id"], dataset_id=dataset_id, image_ids=images)
    with TestClient(app) as client:
        yield client, service, database, request


def test_project_api_and_review_exports_inherit_root_without_manual_paths(project_flow, tmp_path):
    client, service, _, request = project_flow
    root = tmp_path / "塔体 研究项目"
    response = client.post("/api/workflows/projects", json={"name": "塔体", "root_path": str(root)})
    assert response.status_code == 201, response.text
    project = response.json()
    assert client.get("/api/workflows/projects").json() == [project]
    request = request.model_copy(update={"project_id": project["id"]})
    prepared = service.prepare_run(request, {"proposer": {
        "enabled": True, "mode": "local", "api_base": "http://localhost:9999", "model": "fixture", "config_hash": "fixture",
    }})
    assert prepared["request"]["project_id"] == project["id"]
    assert prepared["outputs"]["project"] == project
    assert prepared["outputs"]["json_path"].startswith(str(root))
    created = client.post("/api/workflows/runs", json=prepared["request"])
    assert created.status_code == 202, created.text
    run = drain(service, created.json()["id"])
    assert len(run["reviews"]) == 2
    assert run["context"]["project_id"] == project["id"]
    output = root / ("workflow-" + run["id"]) / "annotations"
    review = run["reviews"][0]
    assert review["export_locations"] == {"json_path": str(output / "json"), "images_path": str(output / "images")}
    exported = client.post(f"/api/workflows/reviews/{review['id']}/labelme", json={"expected_revision": 1, "payload": {}})
    assert exported.status_code == 201, exported.text
    assert Path(exported.json()["saved_path"]).parent == output / "json"
    assert client.get(exported.json()["download_url"]).status_code == 200
    png = io.BytesIO()
    Image.new("RGB", (80, 400), "white").save(png, format="PNG")
    preview = client.post(f"/api/workflows/reviews/{review['id']}/preview", data={"expected_revision": "1"},
                          files={"file": ("preview.png", png.getvalue(), "image/png")})
    assert preview.status_code == 201, preview.text
    assert Path(preview.json()["saved_path"]).parent == output / "images"
    assert service.get_review(review["id"]) == review
    # A later registry edit cannot redirect files from a previously started run.
    config = read_config(service.project_root)
    config["workflow_projects"][0]["root_path"] = str(tmp_path / "other")
    write_config(service.project_root, config)
    repeated = client.post(f"/api/workflows/reviews/{review['id']}/labelme", json={"expected_revision": 1, "payload": {}})
    assert repeated.json()["saved_path"] == exported.json()["saved_path"]


def test_project_change_after_preflight_rejects_start(project_flow, tmp_path):
    _, service, _, request = project_flow
    project = service.create_project("项目", str(tmp_path / "first"))
    request = request.model_copy(update={"project_id": project["id"]})
    settings = {"proposer": {"enabled": True, "mode": "local", "api_base": "http://localhost:9999",
                             "model": "fixture", "config_hash": "fixture"}}
    prepared = service.prepare_run(request, settings)
    config = read_config(service.project_root)
    config["workflow_projects"][0]["name"] = "重新选择的项目"
    write_config(service.project_root, config)
    with pytest.raises(ConflictError, match="changed after preparation"):
        service.create_run(RunCreate.model_validate(prepared["request"]))


def test_batch_http_exports_both_images_to_project_subfolders(project_flow, tmp_path):
    client, service, _, request = project_flow
    root = tmp_path / "批量 项目"
    project = service.create_project("批量", str(root))
    run = drain(service, service.create_run(request.model_copy(update={"project_id": project["id"]}))["id"])
    response = client.post(f"/api/workflows/runs/{run['id']}/annotation-batch", json={"include_previews": True})
    assert response.status_code == 201, response.text
    result = response.json()
    directory = root / ("workflow-" + run["id"]) / "annotations"
    assert result["total_images"] == result["exported_images"] == result["pending_images"] == 2
    assert result["failed_images"] == result["skipped_images"] == 0
    assert Path(result["saved_path"]).parent == directory
    for row in result["results"]:
        assert Path(row["json_path"]).parent == directory / "json"
        assert Path(row["image_path"]).parent == directory / "images"
        assert Path(row["json_path"]).is_file() and Path(row["image_path"]).is_file()
    downloaded = client.get(result["download_url"])
    assert downloaded.status_code == 200 and downloaded.content == Path(result["saved_path"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["exported_images"] == 2
        assert len(archive.namelist()) == 5
    assert service.get_run(run["id"])["reviews"] == run["reviews"]


def test_next_batch_inherits_project_and_legacy_idempotency_is_compatible(project_flow, tmp_path):
    _, service, database, request = project_flow
    legacy = service.create_run(request)
    old_payload = request.model_dump(mode="json")
    old_payload.pop("project_id")
    with database.session_factory() as session:
        assert session.get(WorkflowRun, legacy["id"]).fingerprint == fingerprint(old_payload)
    assert service.create_run(request)["id"] == legacy["id"]
    project = service.create_project("后续批次", str(tmp_path / "project"))
    parent = service.create_run(request.model_copy(update={"idempotency_key": "with-project", "project_id": project["id"]}))
    batch = {"dataset_id": request.dataset_id, "image_ids": request.image_ids[:1]}
    result = service._next_batch({"run_id": parent["id"], "inputs": {"next_batches": [batch]},
                                  "outputs": {"train": {"model_id": "fixture"}}})
    child = service.get_run(result["next_run_id"])
    assert child["context"]["project_id"] == project["id"]
    assert child["context"]["output_destinations"]["project"] == project
    assert child["context"]["output_destinations"]["annotations"]["run_path"] != parent["context"]["output_destinations"]["annotations"]["run_path"]
