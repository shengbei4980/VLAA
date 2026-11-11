"""Coordinate failures become explicit human-review issues in the real workflow."""
import io
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.core.config import Settings
from app.core.database import Base, Database
from app.core.errors import ValidationError
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.dataset.annotation.review import require_confirmed
from app.dataset.annotation.masks import polygon_mask
from app.vision.schemas import ImageStrategy, VisionClassRule, VisionEndpointUpdate, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService
from app.vision.workflow import build_vision_handlers
from app.workflow.bridges import WorkflowBindings
from app.workflow.schemas import NodeDefinition, ProfileCreate, ReviewDecision, ReviewDraft, RunCreate, TaskProfileConfig, TemplateCreate
from app.workflow.service import WorkflowService
from test_vision import approval


@pytest.fixture
def environment(tmp_path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", import_root=tmp_path / "imports",
                        database_url=f"sqlite:///{tmp_path / 'coordinate-workflow.db'}")
    storage = Storage(settings.data_dir, settings.import_root)
    database = Database(settings.database_url)
    Base.metadata.create_all(database.engine)
    content = io.BytesIO()
    Image.new("RGB", (320, 200), "white").save(content, format="PNG")
    storage.write_image("dataset", "image.png", content.getvalue())
    with database.session_factory() as session:
        session.add(Dataset(id="dataset", name="Synthetic architecture", task_type="detect"))
        session.flush()
        session.add(ClassLabel(id="tower", dataset_id="dataset", class_index=0, name="Tower"))
        session.add(ImageItem(id="image", dataset_id="dataset", file_name="image.png", storage_name="image.png",
                              width=320, height=200, split="train"))
        session.commit()
    endpoint = VisionEndpointUpdate(enabled=True, mode="remote", api_base="https://offline.example/v1",
                                    model="mock-vision", max_image_side=64)
    VisionSettingsService(settings).update(VisionSettingsUpdate(proposer=endpoint, reviewer=endpoint))
    try:
        yield settings, storage, database
    finally:
        database.dispose()


@pytest.mark.parametrize("keep_valid_object", [True, False], ids=["retain-valid-object", "review-empty-candidates"])
def test_bad_proposal_and_review_coordinates_require_human_resolution(environment, monkeypatch, keep_valid_object):
    settings, storage, database = environment
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            objects = [{"class_id": "tower", "bbox": [100, 20, 400, 60], "evidence": "An out-of-frame proposal."}]
            if keep_valid_object:
                objects.insert(0, {"class_id": "tower", "bbox": [40, 40, 100, 100], "evidence": "Visible tower."})
            value = {"objects": objects}
        elif len(requests) == 2:
            value = {"objects": []}
        else:
            assert len(requests) == 3, "The workflow must not silently make more vision calls"
            value = {"issues": [{"type": "boundary", "bbox": [100, 20, 400, 60],
                                  "evidence": "A review box in the wrong frame."}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})

    vision = VisionService(storage, settings, transport=httpx.MockTransport(respond))
    monkeypatch.setattr("app.vision.workflow.VisionService", lambda *args: vision)
    service = WorkflowService(database.session_factory, storage,
        WorkflowBindings(handlers=build_vision_handlers(storage, settings, database.session_factory)),
        project_root=settings.project_root)
    profile = service.create_profile(ProfileCreate(key="coordinate-review", name="Coordinate review", config=TaskProfileConfig(
        class_rules=[{"class_id": "tower", "name": "Tower"}],
        rules={"task_instructions": "识别建筑，只标出实际可见部分。"},
    )))
    template = service.create_template(TemplateCreate(key="coordinate-review", name="Coordinate review", nodes=[
        NodeDefinition(id="propose", type="vision_propose", params={"source": "vision"}),
        NodeDefinition(id="validate", type="validate_proposals"),
        NodeDefinition(id="review", type="vision_review"),
        NodeDefinition(id="human", type="human_review"),
    ]))
    prepared = service.prepare_run(RunCreate(idempotency_key="coordinate-review", template_id=template["id"],
        profile_id=profile["id"], dataset_id="dataset", image_ids=["image"]),
        VisionSettingsService(settings).get().model_dump(mode="json"))
    run = service.create_run(RunCreate.model_validate(prepared["request"]))
    for _ in range(16):
        service.tick(run["id"])
        current = service.get_run(run["id"])
        if current["status"] in {"awaiting_review", "needs_attention"}:
            break

    assert current["status"] == "awaiting_review", current.get("error_message")
    assert len(requests) == current["request_count"] == 3
    assert not any(attempt["status"] == "failed" for attempt in current["attempts"])
    review = current["reviews"][0]
    annotations = review["payload"]["annotations"]
    assert len(annotations) == int(keep_valid_object)
    if keep_valid_object:
        assert annotations[0]["bbox"] == {"x": 40, "y": 40, "width": 60, "height": 60}
    issues = review["payload"]["issues"]
    assert len(issues) == 2
    assert all(issue["type"] == "coordinate_error" and not issue["blocking"] for issue in issues)
    assert {issue["coordinate_details"]["phase"] for issue in issues} == {"propose", "review_compare"}
    for issue in issues:
        assert issue["region"] == {"x": 0, "y": 0, "width": 320, "height": 200}
        assert issue["coordinate_details"]["native_bbox"] == [100, 20, 400, 60]
        assert issue["coordinate_details"]["sent_size"] == [320, 200]
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Annotation)) == 0
        assert session.get(ImageItem, "image").review_status not in {"confirmed", "confirmed_empty"}

    decision = "approved" if keep_valid_object else "background"
    decided = service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"], decision=decision))
    assert decided["receipt"]["confirmed_empty"] is (not keep_valid_object)
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Annotation)) == int(keep_valid_object)
        assert len(require_confirmed(session, storage, "dataset", ["image"])) == 1


@pytest.mark.parametrize("bad_bbox", [
    {"x": 310, "y": 10, "width": 20, "height": 20},
    {"x": 80, "y": 10, "width": -20, "height": 20},
], ids=["outside-original-image", "reversed-original-box"])
def test_review_preserves_invalid_existing_geometry_for_human_correction(environment, bad_bbox):
    settings, storage, database = environment
    path = storage.image_path("dataset", "image.png")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        value = {"objects": []} if len(requests) == 1 else {"issues": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})

    objects = [
        {"object_id": "valid", "class_id": "tower", "type": "bbox",
         "bbox": {"x": 40, "y": 40, "width": 60, "height": 60}},
        {"object_id": "invalid", "class_id": "tower", "type": "bbox", "bbox": bad_bbox},
    ]
    workflow = WorkflowService(database.session_factory, storage, WorkflowBindings())
    validated = workflow._validate_proposals({"payload": {"annotations": objects},
        "profile": {"task_type": "detect", "class_rules": [{"class_id": "tower"}]},
        "item": {"width": 320, "height": 200}})
    assert any(issue["blocking"] and issue["object_id"] == "invalid" for issue in validated["issues"])
    result = VisionService(storage, settings, transport=httpx.MockTransport(respond)).review(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1,
        validated["annotations"], ImageStrategy(), approval(settings, path, "reviewer"),
    )

    assert result.request_count == len(requests) == 2
    assert validated["annotations"] == objects
    issue = next(issue for issue in result.issues if issue["type"] == "coordinate_error")
    assert issue["object_id"] == "invalid"
    assert issue["region"] == {"x": 0, "y": 0, "width": 320, "height": 200}
    prompt = requests[1]["messages"][0]["content"][0]["text"]
    visible, _ = json.JSONDecoder().raw_decode(prompt.split("Existing objects in SENT-image coordinates=", 1)[1])
    assert [item["object_id"] for item in visible] == ["valid"]
    assert visible[0]["bbox"] == [40, 40, 100, 100]


def test_repair_preserves_objects_when_an_existing_issue_has_invalid_original_coordinates(environment):
    settings, storage, _ = environment
    path = storage.image_path("dataset", "image.png")
    requests = []

    def respond(request):
        requests.append(request)
        pytest.fail("An invalid source issue must be handed to a human before requesting a replacement")

    objects = [{"object_id": "stable", "class_id": "tower", "type": "bbox",
                "bbox": {"x": 40, "y": 40, "width": 20, "height": 20}}]
    issues = [{"type": "boundary", "object_id": "stable", "evidence": "An invalid saved issue region.",
               "bbox": {"x": 310, "y": 10, "width": 20, "height": 20}}]
    result = VisionService(storage, settings, transport=httpx.MockTransport(respond)).repair(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1,
        objects, issues, ImageStrategy(), approval(settings, path),
    )

    assert result["request_count"] == len(requests) == 0
    assert len(result["objects"]) == 1
    assert result["objects"][0]["object_id"] == "stable"
    assert result["objects"][0]["bbox"] == objects[0]["bbox"]
    assert result["repair_diff"][0]["applied"] is False
    assert any(issue["type"] == "coordinate_error" and issue["object_id"] == "stable" for issue in result["issues"])


@pytest.mark.parametrize("invalid_polygon", [False, True])
def test_direct_llm_contours_reach_review_and_confirmation_without_sam(environment, monkeypatch, invalid_polygon):
    settings, storage, database = environment
    with database.session_factory() as session:
        session.get(Dataset, "dataset").task_type = "segment"
        session.commit()
    polygon = [[40, 40], [100, 40], [100, 100], [70, 75], [40, 100]]
    native = [[40, 40], [9999, 40], [40, 100]] if invalid_polygon else polygon
    calls = []

    def respond(request):
        data = json.loads(request.content)
        calls.append(data)
        if len(calls) <= 2:
            assert '"polygon"' in data["messages"][0]["content"][0]["text"]
            value = {"objects": [{"class_id": "tower", "polygon": native, "evidence": "Visible irregular boundary."}]}
        else:
            value = {"issues": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})

    def no_sam(*args, **kwargs):
        pytest.fail("Direct LLM contour mode must not prepare or execute SAM")

    monkeypatch.setattr("app.vision.workflow.SamService", no_sam)
    vision = VisionService(storage, settings, transport=httpx.MockTransport(respond))
    monkeypatch.setattr("app.vision.workflow.VisionService", lambda *args: vision)
    service = WorkflowService(database.session_factory, storage,
        WorkflowBindings(handlers=build_vision_handlers(storage, settings, database.session_factory), sam_resolver=no_sam),
        project_root=settings.project_root)
    service.ensure_defaults()
    profile = service.create_profile(ProfileCreate(key="direct", name="Direct LLM", config=TaskProfileConfig(
        task_type="segment", geometry="mask", models={"segmenter": "vision_llm"},
        class_rules=[{"class_id": "tower", "name": "Tower"}], export_format="labelme")))
    template = next(item for item in service.list_templates() if item["key"] == "vision_review_export_v1")
    request = RunCreate(idempotency_key="direct-contours", template_id=template["id"], profile_id=profile["id"], dataset_id="dataset", image_ids=["image"])
    prepared = service.prepare_run(request, VisionSettingsService(settings).get().model_dump(mode="json"))
    assert prepared["sam_config"] is None
    assert "sam_config" not in prepared["request"]["inputs"]
    run = service.create_run(RunCreate.model_validate(prepared["request"]))
    for _ in range(25):
        service.tick(run["id"])
        current = service.get_run(run["id"])
        if current["status"] in {"awaiting_review", "needs_attention"}:
            break
    assert current["status"] == "awaiting_review", current
    review = service.get_review(current["reviews"][0]["id"])
    assert len(calls) == 3
    assert review["payload"]["contour_source"] == "vision_llm"
    objects = review["payload"]["annotations"]
    if invalid_polygon:
        assert not objects
        assert any(issue["type"] == "coordinate_error" for issue in review["payload"]["issues"])
        return
    assert len(objects) == 1
    assert objects[0]["polygon"] == polygon
    assert objects[0]["type"] == "polygon"
    assert "segmenter" not in objects[0].get("provenance_json", {})
    assert not objects[0].get("mask_ref")
    exported = service.save_review_labelme(review["id"], ReviewDraft(expected_revision=1, payload=review["payload"]))
    document = json.loads(Path(exported["saved_path"]).read_text(encoding="utf-8"))
    assert document["imageWidth"] == 320 and document["imageHeight"] == 200
    # The exported pixel region preserves the irregular LLM contour, not its bounding box.
    from app.dataset.annotation.masks import decode_mask
    assert np.array_equal(decode_mask(document["shapes"][0]["starter_rle"]), polygon_mask(polygon, 320, 200))
    decided = service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    assert decided["status"] == "decided"
    with database.session_factory() as session:
        assert len(require_confirmed(session, storage, "dataset", ["image"])) == 1
    for _ in range(12):
        service.tick(run["id"])
        current = service.get_run(run["id"])
        if current["status"] in {"completed", "needs_attention"}:
            break
    assert current["status"] == "completed", current
