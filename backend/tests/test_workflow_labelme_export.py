"""Self-contained LabelMe JSON drafts preserve geometry without confirming labels."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.api.error_handlers import register_error_handlers
from app.core.models import Annotation, Dataset
from app.dataset.annotation.masks import MaskService, decode_mask, encode_mask
from app.workflow.models import WorkflowArtifact
from app.workflow.router import get_workflow_service, router
from app.workflow.schemas import NodeDefinition, ProfileCreate, ReviewDecision, RunCreate, TaskProfileConfig, TemplateCreate
from test_workflow import drain, workflow
from test_workflow_exports import native_mask


def object_mask():
    mask = np.zeros((400, 80), dtype=bool)
    mask[10:140, 5:65] = True
    mask[30:60, 20:45] = False
    mask[300:350, 60:78] = True
    return mask


@pytest.fixture
def labelme(workflow, request, tmp_path):
    service, database, dataset_id, images, *_ = workflow
    service.project_root = tmp_path
    task = getattr(request, "param", "detect")
    geometry = {"detect": "bbox", "segment": "mask", "obb": "obb", "classify": "classify"}[task]
    with database.session_factory() as session:
        session.get(Dataset, dataset_id).task_type = task
        session.commit()
    profile = service.create_profile(ProfileCreate(key="labelme-export", name="LabelMe export",
        config=TaskProfileConfig(task_type=task, geometry=geometry)))
    if task == "segment":
        proposer = service.bindings.handlers["vision_propose"]
        reference = MaskService(service.storage).write(object_mask()).mask_ref

        def masked_proposal(context):
            result = proposer(context)
            result["annotations"][0].update(type="polygon", mask_ref=reference)
            return result

        service.bindings.handlers["vision_propose"] = masked_proposal
    template = service.create_template(TemplateCreate(key="labelme-export", name="LabelMe export", nodes=[
        NodeDefinition(id="propose", type="vision_propose"), NodeDefinition(id="human", type="human_review")]))
    run = service.create_run(RunCreate(idempotency_key="labelme-export", profile_id=profile["id"], template_id=template["id"],
        dataset_id=dataset_id, image_ids=images))
    run = drain(service, run["id"])
    assert run["status"] == "awaiting_review", run.get("error_message")
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_workflow_service] = lambda: service
    with TestClient(app) as client:
        yield client, service, database, run


def export(client, review, payload=None, revision=None):
    return client.post(f"/api/workflows/reviews/{review['id']}/labelme", json={
        "expected_revision": revision if revision is not None else review["revision"],
        "payload": payload or {},
    })


def exported_document(client, result):
    response = client.get(result["download_url"])
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.content == Path(result["saved_path"]).read_bytes()
    return response.json()


def test_bbox_json_contains_original_image_and_pending_review_without_new_labels(labelme):
    client, service, database, run = labelme
    review = run["reviews"][0]
    objects = deepcopy(review["payload"]["objects"])
    objects.extend([
        {**objects[0], "object_id": "unmapped", "class_id": None},
        {**objects[0], "object_id": "other-class", "class_id": "tree"},
        {**objects[0], "object_id": "rejected", "decision": "rejected"},
    ])
    issues = [{"id": "check", "type": "uncertain", "object_id": "unmapped", "evidence": "Needs inspection"}]
    with database.session_factory() as session:
        artifacts_before = session.scalar(select(func.count()).select_from(WorkflowArtifact))
        labels_before = session.scalar(select(func.count()).select_from(Annotation))
    response = export(client, review, {"objects": objects, "issues": issues})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["export_state"] == "draft" and result["exported_objects"] == 1 and result["pending_objects"] == 2
    document = exported_document(client, result)
    assert document["version"] == "5.8.0" and document["flags"]["vlaa_draft"] is True
    assert document["imagePath"] == run["items"][0]["context"]["image_id"] + ".png"
    with Image.open(io.BytesIO(base64.b64decode(document["imageData"], validate=True))) as image:
        assert image.size == (80, 400)
    assert document["shapes"][0]["shape_type"] == "rectangle"
    assert document["shapes"][0]["points"] == [[10, 20], [30, 50]]
    assert document["shapes"][0]["label"] == "component"
    assert document["vlaa_review"]["objects"] == objects and document["vlaa_review"]["issues"] == issues
    assert document["vlaa_review"]["status"] == review["status"]
    assert service.get_review(review["id"]) == review
    assert export(client, review, {"objects": objects, "issues": issues}).json()["saved_path"] == result["saved_path"]
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowArtifact)) == artifacts_before
        assert session.scalar(select(func.count()).select_from(Annotation)) == labels_before


@pytest.mark.parametrize("labelme", ["segment"], indirect=True)
def test_segment_mask_references_and_unsaved_rle_edits_are_lossless(labelme):
    client, service, _, run = labelme
    review = run["reviews"][0]
    original = exported_document(client, export(client, review).json())
    assert np.array_equal(native_mask(original["shapes"][0], width=80, height=400), object_mask())
    edited = object_mask()
    edited[370:380, 3:12] = True
    roi = np.ones((400, 80), dtype=bool)
    roi[:5] = False
    ignore = np.zeros_like(roi)
    ignore[200:220, 20:30] = True
    objects = deepcopy(review["payload"]["objects"])
    objects[0]["mask"] = encode_mask(edited)
    old_layer = MaskService(service.storage).write(np.zeros_like(roi)).mask_ref
    payload = {"objects": objects, "roi": encode_mask(roi), "ignore": encode_mask(ignore),
               "roi_mask_ref": old_layer, "ignore_mask_ref": old_layer}
    result = export(client, review, payload)
    assert result.status_code == 201, result.text
    document = exported_document(client, result.json())
    assert np.array_equal(native_mask(document["shapes"][0], width=80, height=400), edited)
    assert np.array_equal(decode_mask(document["starter_roi"]), roi)
    assert np.array_equal(decode_mask(document["starter_ignore"]), ignore)
    assert np.array_equal(decode_mask(document["vlaa_review"]["roi"]), roi)
    assert np.array_equal(decode_mask(document["vlaa_review"]["ignore"]), ignore)
    assert service.get_review(review["id"]) == review


def test_confirmed_export_is_marked_draft_when_current_geometry_changes(labelme):
    client, service, _, run = labelme
    review = run["reviews"][0]
    review = service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"], decision="approved"))
    result = export(client, review, {"revision": 999}).json()
    assert result["export_state"] == "confirmed"
    document = exported_document(client, result)
    assert document["starter_revision"] == review["revision"]
    assert document["starter_review_receipt"] == review["receipt"]
    objects = deepcopy(review["payload"]["objects"])
    objects[0]["bbox"]["x"] = 15
    result = export(client, review, {"objects": objects})
    assert result.status_code == 201, result.text
    assert result.json()["export_state"] == "draft"
    assert service.get_review(review["id"]) == review


@pytest.mark.parametrize("labelme", ["segment"], indirect=True)
def test_hydrated_confirmed_mask_retains_confirmation_until_pixels_change(labelme):
    client, service, _, run = labelme
    review = run["reviews"][0]
    review = service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"], decision="approved"))
    objects = deepcopy(review["payload"]["objects"])
    objects[0]["mask"] = encode_mask(object_mask())
    result = export(client, review, {"objects": objects})
    assert result.status_code == 201, result.text
    assert result.json()["export_state"] == "confirmed"
    edited = object_mask()
    edited[15, 15] = False
    objects[0]["mask"] = encode_mask(edited)
    result = export(client, review, {"objects": objects})
    assert result.status_code == 201, result.text
    assert result.json()["export_state"] == "draft"
    assert service.get_review(review["id"]) == review


def test_json_size_boundary_rejects_large_embedded_export_before_writing(labelme, monkeypatch):
    from app.workflow import labelme_export

    client, service, _, run = labelme
    review = run["reviews"][0]
    result = export(client, review).json()
    size = Path(result["saved_path"]).stat().st_size
    Path(result["saved_path"]).unlink()
    monkeypatch.setattr(labelme_export, "MAX_LABELME_BYTES", size - 1)
    response = export(client, review)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "workflow_labelme_too_large"
    assert list(service.storage.exports_dir.glob("*.json")) == []


@pytest.mark.parametrize("changed", [{"width": 1}, {"image_id": "other"}, {"dataset_id": "other"}, {"image_url": "https://other.test/image"}])
def test_stale_or_changed_identity_cannot_export_a_different_image(labelme, changed):
    client, service, _, run = labelme
    review = run["reviews"][0]
    assert export(client, review, revision=review["revision"] + 1).status_code == 409
    response = export(client, review, changed)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "workflow_review_identity_changed"
    assert list(service.storage.exports_dir.glob("*.json")) == []


def test_invalid_coordinates_and_nonfinite_data_are_rejected(labelme):
    client, service, _, run = labelme
    review = run["reviews"][0]
    objects = deepcopy(review["payload"]["objects"])
    objects[0]["bbox"]["x"] = -1
    assert export(client, review, {"objects": objects}).status_code == 422
    objects[0]["bbox"]["x"] = float("nan")
    response = client.post(f"/api/workflows/reviews/{review['id']}/labelme", headers={"Content-Type": "application/json"},
        content=json.dumps({"expected_revision": review["revision"], "payload": {"objects": objects}}))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "workflow_labelme_invalid_data"
    assert list(service.storage.exports_dir.glob("*.json")) == []


def test_json_download_rejects_other_reviews_and_corrupted_files(labelme):
    client, _, _, run = labelme
    first, other = run["reviews"]
    result = export(client, first).json()
    response = client.get(f"/api/workflows/reviews/{other['id']}/labelme/{result['file_name']}")
    assert response.status_code == 422
    Path(result["saved_path"]).write_text("changed", encoding="utf-8")
    assert client.get(result["download_url"]).status_code == 409


@pytest.mark.parametrize("labelme", ["obb", "classify"], indirect=True)
def test_unsupported_labelme_tasks_are_explicit(labelme):
    client, _, _, run = labelme
    response = export(client, run["reviews"][0])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "workflow_export_task_unsupported"


def test_cancelled_run_can_still_export_its_readonly_draft(labelme):
    client, service, _, run = labelme
    service.cancel_run(run["id"])
    response = export(client, run["reviews"][0])
    assert response.status_code == 201
    assert service.get_run(run["id"])["status"] == "cancelled"
