"""Coordinate failures become explicit human-review issues in the real workflow."""
import io
import json

import httpx
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.core.config import Settings
from app.core.database import Base, Database
from app.core.errors import ValidationError
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.dataset.annotation.review import require_confirmed
from app.vision.schemas import ImageStrategy, VisionClassRule, VisionEndpointUpdate, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService
from app.vision.workflow import build_vision_handlers
from app.workflow.bridges import WorkflowBindings
from app.workflow.schemas import NodeDefinition, ProfileCreate, ReviewDecision, RunCreate, TaskProfileConfig, TemplateCreate
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
            objects = [{"class_id": "tower", "bbox": [100, 20, 200, 60], "evidence": "An out-of-frame proposal."}]
            if keep_valid_object:
                objects.insert(0, {"class_id": "tower", "bbox": [8, 8, 20, 20], "evidence": "Visible tower."})
            value = {"objects": objects}
        elif len(requests) == 2:
            value = {"objects": []}
        else:
            assert len(requests) == 3, "The workflow must not silently make more vision calls"
            value = {"issues": [{"type": "boundary", "bbox": [100, 20, 200, 60],
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
        assert issue["coordinate_details"]["native_bbox"] == [100, 20, 200, 60]
        assert issue["coordinate_details"]["sent_size"] == [64, 40]
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
    assert visible[0]["bbox"] == [8, 8, 20, 20]


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
