"""Human-resolution regressions use synthetic images and an offline HTTP mock."""
from hashlib import sha256
import io
import json

import httpx
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.database import Base, Database
from app.core.errors import ValidationError
from app.core.models import AnnotationReviewReceipt, ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.dataset.annotation.review import require_confirmed
from app.vision.schemas import ImageStrategy, VisionApproval, VisionClassRule, VisionEndpointUpdate, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService
from app.workflow.bridges import WorkflowBindings
from app.workflow.schemas import NodeDefinition, ProfileCreate, ReviewDecision, ReviewDraft, RunCreate, TaskProfileConfig, TemplateCreate
from app.workflow.service import WorkflowService


@pytest.fixture
def environment(tmp_path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", import_root=tmp_path / "imports",
                        database_url=f"sqlite:///{tmp_path / 'review.db'}")
    storage = Storage(settings.data_dir, settings.import_root)
    database = Database(settings.database_url)
    Base.metadata.create_all(database.engine)
    content = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(content, format="PNG")
    path = storage.image_path("dataset", "image.png")
    path.write_bytes(content.getvalue())
    with database.session_factory() as session:
        session.add(Dataset(id="dataset", name="Synthetic review", task_type="detect"))
        session.flush()
        session.add(ClassLabel(id="tower", dataset_id="dataset", class_index=0, name="Tower"))
        session.add(ImageItem(id="image", dataset_id="dataset", file_name=path.name, storage_name=path.name,
                              width=100, height=80, split="train"))
        session.commit()
    try:
        yield settings, storage, database, path
    finally:
        database.dispose()


def open_review(environment, annotations, issues):
    _, storage, database, _ = environment
    bindings = WorkflowBindings(handlers={"vision_propose": lambda context: {
        "annotations": annotations, "issues": issues, "request_count": 1,
    }})
    service = WorkflowService(database.session_factory, storage, bindings)
    profile = service.create_profile(ProfileCreate(key="synthetic", name="Synthetic", config=TaskProfileConfig()))
    template = service.create_template(TemplateCreate(key="resolve", name="Resolve", nodes=[
        NodeDefinition(id="propose", type="vision_propose", params={"source": "vision"}),
        NodeDefinition(id="validate", type="validate_proposals"),
        NodeDefinition(id="human", type="human_review"),
    ]))
    run = service.create_run(RunCreate(idempotency_key="resolution", template_id=template["id"], profile_id=profile["id"],
                                     dataset_id="dataset", image_ids=["image"]))
    for _ in range(10):
        service.tick(run["id"])
        current = service.get_run(run["id"])
        if current["status"] == "awaiting_review":
            return service, current["reviews"][0]
    pytest.fail(f"The synthetic workflow did not reach review: {current['status']}")


def candidate(class_id="tower"):
    return {"object_id": "candidate", "class_id": class_id, "type": "bbox", "source": "auto",
            "bbox": {"x": 5, "y": 5, "width": 10, "height": 10}}


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("with_note", [False, True])
def test_whole_image_confirmation_allows_unanswered_advice_and_keeps_optional_notes(environment, empty, with_note):
    issues = [{"id": f"advice-{i}", "type": "boundary", "message": "Check the visible region", "blocking": False} for i in range(6)]
    service, review = open_review(environment, [] if empty else [candidate()], issues)
    notes = {review["payload"]["issues"][0]["id"]: {"resolution": "resolved", "reason": "Boundary corrected."}} if with_note else {}
    decided = service.decide_review(review["id"], ReviewDecision(expected_revision=1,
        decision="background" if empty else "approved", semantic_resolutions=notes))
    assert decided["status"] == "decided"
    assert len(decided["payload"]["issues"]) == 6
    assert all(not issue.get("resolution") for issue in decided["payload"]["issues"])
    with environment[2].session_factory() as session:
        assert session.get(ImageItem, "image").review_status == ("confirmed_empty" if empty else "confirmed")
        assert len(require_confirmed(session, environment[1], "dataset", ["image"])) == 1
        receipt = session.get(AnnotationReviewReceipt, decided["receipt"]["receipt_id"])
        assert receipt.decision_json["semantic_resolutions"] == notes


@pytest.mark.parametrize("delete_object", [False, True])
def test_human_can_correct_or_delete_an_unmapped_proposal_and_confirm(environment, delete_object):
    service, review = open_review(environment, [candidate(None)], [
        {"type": "unmapped_class", "object_id": "candidate", "evidence": "Class is uncertain", "blocking": True},
    ])
    with pytest.raises(ValidationError) as error:
        service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    assert error.value.error_code == "review_blocking_issue"
    # A save while the class remains unmapped must keep blocking confirmation.
    unresolved = service.save_review(review["id"], ReviewDraft(expected_revision=1, payload={"annotations": [candidate(None)]}))
    assert any(issue.get("blocking") for issue in unresolved["payload"]["issues"])
    updated = service.save_review(review["id"], ReviewDraft(expected_revision=2,
        payload={"annotations": [] if delete_object else [candidate()]}))
    assert updated["payload"]["issues"] == []
    decided = service.decide_review(review["id"], ReviewDecision(expected_revision=3,
        decision="background" if delete_object else "approved"))
    assert decided["receipt"]["confirmed_empty"] is delete_object
    with environment[2].session_factory() as session:
        assert len(require_confirmed(session, environment[1], "dataset", ["image"])) == 1


def test_reviewer_unknown_inventory_region_remains_advisory_at_whole_image_confirmation(environment):
    settings, storage, _, path = environment
    endpoint = VisionEndpointUpdate(enabled=True, mode="remote", api_base="https://offline.example/v1", model="mock-vision")
    VisionSettingsService(settings).update(VisionSettingsUpdate(proposer=endpoint, reviewer=endpoint))
    config = VisionSettingsService(settings).internal("reviewer")
    approval = VisionApproval(mode=config.mode, api_base=config.api_base, model=config.model, config_hash=config.config_hash,
                              allowed_image_hashes=[sha256(path.read_bytes()).hexdigest()], max_requests=2)
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        value = {"objects": [
            {"class_id": None, "bbox": [30, 20, 50, 40], "evidence": "Visible region with uncertain class"},
            {"class_id": "tower", "bbox": [65, 30, 90, 60], "evidence": "A missed tower"},
        ]} if len(calls) == 1 else {"issues": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(respond)).review(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1, [candidate()], ImageStrategy(), approval,
    )
    assert result.request_count == len(calls) == 2
    assert {issue["type"] for issue in result.issues} == {"uncertain", "missing"}
    uncertain = next(issue for issue in result.issues if issue["type"] == "uncertain")
    assert uncertain["bbox"] == {"x": 30, "y": 20, "width": 20, "height": 20}
    assert all(issue["object_id"] is None and not issue["blocking"] for issue in result.issues)
    service, review = open_review(environment, [candidate()], result.issues)
    decided = service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    assert decided["status"] == "decided"
