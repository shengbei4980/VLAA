"""State-machine tests use deterministic node fixtures, not model-quality claims."""
from concurrent.futures import ThreadPoolExecutor
import io
import threading

import pytest
from PIL import Image
from sqlalchemy import select

from app.core.database import Base, Database
from app.core.errors import ConflictError, ValidationError
from app.core.ids import new_id
from app.core.models import ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.workflow.bridges import WorkflowBindings
from app.workflow.models import WorkflowAttempt, WorkflowItem, WorkflowRun
from app.workflow.schemas import NodeDefinition, ProfileCreate, ReviewDecision, ReviewDraft, RunCreate, TaskProfileConfig, TemplateCreate
from app.workflow.service import WorkflowService


@pytest.fixture
def workflow(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'workflow.db'}")
    Base.metadata.create_all(database.engine)
    storage = Storage(tmp_path / "data", tmp_path / "imports")
    dataset_id, class_id = new_id("ds"), new_id("cls")
    image_ids = []
    with database.session_factory() as session:
        session.add(Dataset(id=dataset_id, name="Tower components", task_type="detect"))
        session.flush()
        session.add(ClassLabel(id=class_id, dataset_id=dataset_id, class_index=0, name="component"))
        for split in ("train", "val"):
            image_id = new_id("img")
            content = io.BytesIO()
            Image.new("RGB", (80, 400), "white").save(content, format="PNG")
            storage_name = image_id + ".png"
            storage.write_image(dataset_id, storage_name, content.getvalue())
            session.add(ImageItem(id=image_id, dataset_id=dataset_id, file_name=storage_name,
                storage_name=storage_name, width=80, height=400, split=split))
            image_ids.append(image_id)
        session.commit()
    calls = {"train": [], "evaluate": []}

    def propose(context):
        return {"annotations": [{"object_id": "component", "class_id": class_id, "type": "bbox",
            "bbox": {"x": 10, "y": 20, "width": 20, "height": 30}, "source": "auto"}], "request_count": 1}

    def train(context):
        calls["train"].append(context["task_id"])
        return {"status": "completed", "model_id": "fixture-model", "snapshot_id": context["outputs"]["snapshot"]["snapshot_id"]}

    def evaluate(context):
        calls["evaluate"].append(context["task_id"])
        return {"status": "completed", "model_id": "fixture-model", "metrics": {"map50": 0.9}}

    bindings = WorkflowBindings(handlers={"vision_propose": propose,
        "vision_review": lambda context: context["payload"], "train": train, "evaluate": evaluate})
    service = WorkflowService(database.session_factory, storage, bindings)
    service.ensure_defaults()
    profile = service.create_profile(ProfileCreate(key="tower", name="Tower", config=TaskProfileConfig()))
    template = next(row for row in service.list_templates() if row["key"] == "visual_annotation_train_cycle_v1")
    yield service, database, dataset_id, image_ids, profile, template, calls
    database.dispose()


def start(env, key="run", **kwargs):
    service, _, dataset_id, images, profile, template, _ = env
    return service.create_run(RunCreate(idempotency_key=key, template_id=template["id"], profile_id=profile["id"],
        dataset_id=dataset_id, image_ids=images, **kwargs))


def test_explicit_input_binding_selects_its_source_instead_of_latest_output(workflow):
    service, _, dataset_id, images, profile, _, _ = workflow
    with service.session_factory() as session:
        class_id = session.scalar(select(ClassLabel.id).where(ClassLabel.dataset_id == dataset_id))
    service.bindings.handlers["managed_inference"] = lambda context: {"annotations": [{
        "object_id": "different", "class_id": class_id, "type": "bbox",
        "bbox": {"x": 50, "y": 50, "width": 20, "height": 30}, "source": "auto"}]}
    template = service.create_template(TemplateCreate(key="bound-source", name="Bound source", nodes=[
        NodeDefinition(id="original", type="vision_propose"),
        NodeDefinition(id="newer", type="managed_inference"),
        NodeDefinition(id="selected", type="validate_proposals", inputs={"proposals": "original"}),
        NodeDefinition(id="confirm", type="human_review", inputs={"proposals": "selected"}),
    ]))
    run = service.create_run(RunCreate(idempotency_key="bound-run", template_id=template["id"],
        profile_id=profile["id"], dataset_id=dataset_id, image_ids=images[:1]))
    for _ in range(8):
        service.tick(run["id"])
    review = service.get_run(run["id"])["reviews"][0]
    assert review["payload"]["annotations"][0]["bbox"]["x"] == 10


def drain(service, run_id, limit=80):
    for _ in range(limit):
        changed = service.tick(run_id)
        if not changed and service.get_run(run_id)["status"] in {"awaiting_review", "needs_attention", "completed", "cancelled", "waiting_task"}:
            break
    return service.get_run(run_id)


def test_excluding_every_item_does_not_reintroduce_previously_confirmed_labels(workflow):
    from app.core.schemas import AnnotationInput, BBox
    from app.dataset.annotation.review import confirm_image, review_detail
    from app.dataset.annotation.service import replace_annotations

    service, database, dataset_id, images, profile, template, calls = workflow
    with database.session_factory() as session:
        class_id = session.scalar(select(ClassLabel.id).where(ClassLabel.dataset_id == dataset_id))
        replace_annotations(session, images[0], [AnnotationInput(type="bbox", class_id=class_id,
            bbox=BBox(x=2, y=2, width=10, height=10))], storage=service.storage)
        detail = review_detail(session, service.storage, images[0])
        confirm_image(session, service.storage, images[0], expected_revision=detail["revision"],
                      expected_fingerprint=detail["fingerprint"])
        session.commit()
    run = service.create_run(RunCreate(idempotency_key="all-excluded", template_id=template["id"],
        profile_id=profile["id"], dataset_id=dataset_id, image_ids=images[:1]))
    run = drain(service, run["id"])
    review = run["reviews"][0]
    service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"],
        decision="excluded", reason="This image does not belong to the current batch."))
    run = drain(service, run["id"])
    assert run["status"] == "needs_attention"
    assert calls["train"] == []


def approve(service, review):
    return service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"], decision="approved"))


def test_idempotency_and_immutable_profile_versions(workflow):
    service, _, _, _, profile, _, _ = workflow
    first = start(workflow)
    assert start(workflow)["id"] == first["id"]
    newer = service.create_profile(ProfileCreate(key="tower", name="Tower v2", config=TaskProfileConfig(repair_limit=1)))
    assert newer["version"] == 2
    assert service.get_profile(profile["id"])["config"]["repair_limit"] == 0
    with pytest.raises(ConflictError, match="different inputs"):
        start(workflow, snapshot_id="other")


def test_batch_barrier_real_review_receipts_and_snapshot(workflow):
    service, database, _, images, _, _, calls = workflow
    run = drain(service, start(workflow)["id"])
    assert len(run["reviews"]) == 2
    assert calls["train"] == []
    approve(service, run["reviews"][0])
    run = drain(service, run["id"])
    assert run["status"] == "awaiting_review"
    assert calls["train"] == []
    approve(service, run["reviews"][1])
    run = drain(service, run["id"])
    assert run["status"] == "completed", run["error_message"]
    assert len(calls["train"]) == len(calls["evaluate"]) == 1
    snapshot = next(artifact["payload"] for artifact in run["artifacts"] if artifact["kind"] == "dataset_snapshot")
    assert {image["id"] for image in snapshot["manifest"]["images"]} == set(images)
    with database.session_factory() as session:
        assert all(session.get(ImageItem, image_id).review_status == "confirmed" for image_id in images)


def test_draft_revision_is_immutable_and_accepted_subset_is_authoritative(workflow):
    service = workflow[0]
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    original_artifact = review["artifact_id"]
    updated = service.save_review(review["id"], ReviewDraft(expected_revision=1,
        payload={"objects": review["payload"]["objects"], "annotations": []}))
    assert updated["revision"] == 2
    assert updated["artifact_id"] != original_artifact
    assert updated["payload"]["annotations"] == []
    assert len(updated["payload"]["objects"]) == 1
    with pytest.raises(ConflictError):
        approve(service, review)
    result = service.decide_review(review["id"], ReviewDecision(expected_revision=2, decision="background"))
    assert result["receipt"]["confirmed_empty"] is True


def test_stale_formal_labels_cannot_be_overwritten(workflow):
    service, database, _, _, _, _, _ = workflow
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    with database.session_factory() as session:
        image = session.get(ImageItem, review["payload"]["image_id"])
        image.annotation_revision += 1
        session.commit()
    with pytest.raises(ConflictError, match="changed"):
        approve(service, review)
    assert service.get_review(review["id"])["status"] == "pending"


def test_duplicate_confirm_is_serialized(workflow):
    service = workflow[0]
    review = drain(service, start(workflow)["id"])["reviews"][0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: approve(service, review), range(2)))
    assert results[0]["receipt"] == results[1]["receipt"]


def test_cancel_drops_late_node_result(workflow):
    service = workflow[0]
    entered, release = threading.Event(), threading.Event()

    def slow(context):
        entered.set()
        assert release.wait(5)
        return {"annotations": []}

    service.bindings.handlers["vision_propose"] = slow
    run = start(workflow)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.tick, run["id"])
        assert entered.wait(5)
        service.cancel_run(run["id"])
        release.set()
        future.result()
    result = service.get_run(run["id"])
    assert result["status"] == "cancelled"
    assert result["artifacts"] == []
    assert result["reviews"] == []


def test_restart_recovers_attempt_and_keeps_successful_item(workflow):
    service, database, _, _, _, _, _ = workflow
    run = start(workflow)
    assert service.tick(run["id"])
    result = service.get_run(run["id"])
    first_item = result["attempts"][0]["item_id"]
    with database.session_factory() as session:
        item = session.get(WorkflowItem, first_item)
        context = {"fixture": True}
        session.add(WorkflowAttempt(id="crashed", run_id=run["id"], item_id=item.id, unit_key=item.id,
            node_id="validate", node_type="validate_proposals", attempt=1, generation=1,
            fingerprint="fixture", status="running", input_json=context))
        session.get(WorkflowRun, run["id"]).status = "running"
        session.commit()
    service.recover()
    result = drain(service, run["id"])
    assert len([attempt for attempt in result["attempts"] if attempt["node_id"] == "propose" and attempt["item_id"] == first_item]) == 1
    assert any(attempt["id"] == "crashed" and attempt["status"] == "interrupted" for attempt in result["attempts"])
    assert len(result["reviews"]) == 2


def test_failed_item_does_not_discard_other_items_and_retry_is_bounded(workflow):
    service = workflow[0]
    propose = service.bindings.handlers["vision_propose"]
    bad_image = workflow[3][0]

    def intermittent(context):
        if context["item"]["image_id"] == bad_image:
            raise ValidationError("fixture_failed", "fixture failure")
        return propose(context)

    service.bindings.handlers["vision_propose"] = intermittent
    run = start(workflow)
    for _ in range(30):
        service.tick(run["id"])
    result = service.get_run(run["id"])
    assert len(result["reviews"]) == 1
    failed = next(item for item in result["items"] if item["status"] == "failed")
    service.bindings.handlers["vision_propose"] = propose
    service.retry_run(run["id"], [failed["id"]])
    result = drain(service, run["id"])
    assert len(result["reviews"]) == 2


def test_templates_reject_incompatible_ports_and_backward_routes(workflow):
    service = workflow[0]
    with pytest.raises(ValidationError, match="cannot feed"):
        service.create_template(TemplateCreate(key="bad", name="Bad", nodes=[
            NodeDefinition(id="a", type="vision_propose"),
            NodeDefinition(id="b", type="train", scope="batch", inputs={"snapshot": "a"}),
        ]))
    with pytest.raises(ValidationError, match="point forward"):
        service.create_template(TemplateCreate(key="bad", name="Bad", nodes=[
            NodeDefinition(id="a", type="vision_propose", routes={"completed": "a"}),
        ]))


def test_payload_rejects_embedded_credentials(workflow):
    with pytest.raises(ValidationError, match="Credentials"):
        start(workflow, inputs={"api_key": "never-persist"})


def test_waiting_managed_task_resumes_same_reserved_id(workflow):
    service, database, _, _, _, _, _ = workflow
    calls = []

    def waiting_train(context):
        calls.append(context["task_id"])
        if len(calls) == 1:
            return {"status": "waiting_task", "task_id": context["task_id"]}
        return {"status": "completed", "model_id": "fixture-model", "snapshot_id": context["outputs"]["snapshot"]["snapshot_id"]}

    service.bindings.handlers["train"] = waiting_train
    run = drain(service, start(workflow)["id"])
    for review in run["reviews"]:
        approve(service, review)
    run = drain(service, run["id"])
    assert run["status"] == "waiting_task"
    restarted = WorkflowService(database.session_factory, service.storage, service.bindings)
    restarted.recover()
    result = drain(restarted, run["id"])
    assert result["status"] == "completed"
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len([attempt for attempt in result["attempts"] if attempt["node_type"] == "train"]) == 1


def test_next_batch_uses_new_model_and_waits_for_human(workflow):
    service, _, dataset_id, images, _, _, _ = workflow
    run = drain(service, start(workflow, next_batches=[{"dataset_id": dataset_id, "image_ids": images}])["id"])
    for review in run["reviews"]:
        approve(service, review)
    run = drain(service, run["id"])
    assert run["status"] == "waiting_task"
    next_output = next(attempt["output"] for attempt in run["attempts"] if attempt["node_type"] == "next_batch")
    child = drain(service, next_output["next_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["context"]["model_id"] == "fixture-model"
    assert child["status"] == "awaiting_review"
    assert len(child["reviews"]) == 2
    assert service.get_run(run["id"])["status"] == "waiting_task"
    service.cancel_run(run["id"])
    assert service.get_run(child["id"])["status"] == "cancelled"


def test_second_template_reuses_confirmed_snapshot_without_vision(workflow):
    service, _, dataset_id, images, profile, _, calls = workflow
    first = drain(service, start(workflow)["id"])
    for review in first["reviews"]:
        approve(service, review)
    first = drain(service, first["id"])
    snapshot_id = next(artifact["payload"]["snapshot_id"] for artifact in first["artifacts"] if artifact["kind"] == "dataset_snapshot")
    template = next(row for row in service.list_templates() if row["key"] == "confirmed_train_evaluate_v1")
    service.bindings.handlers["vision_propose"] = lambda _: pytest.fail("Second template must not invoke vision")
    second = service.create_run(RunCreate(idempotency_key="second", template_id=template["id"], profile_id=profile["id"],
        dataset_id=dataset_id, image_ids=images, snapshot_id=snapshot_id))
    second = drain(service, second["id"])
    assert second["status"] == "completed"
    assert not second["reviews"]
    assert len(calls["train"]) == 2


def test_model_review_export_produces_confirmed_download(workflow):
    import zipfile
    service, _, dataset_id, images, profile, _, _ = workflow
    service.bindings.handlers["managed_inference"] = service.bindings.handlers["vision_propose"]
    template = next(row for row in service.list_templates() if row["key"] == "model_review_export_v1")
    run = service.create_run(RunCreate(idempotency_key="export", template_id=template["id"], profile_id=profile["id"],
        dataset_id=dataset_id, image_ids=images))
    run = drain(service, run["id"])
    for review in run["reviews"]:
        approve(service, review)
    run = drain(service, run["id"])
    assert run["status"] == "completed", run["error_message"]
    with zipfile.ZipFile(service.export_download(run["id"])) as archive:
        assert "workflow_manifest.json" in archive.namelist()
        assert len([name for name in archive.namelist() if name.startswith("labels/")]) == 2


def test_visual_llm_export_template_needs_no_recognition_model_and_waits_for_review(workflow):
    import json
    import zipfile
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.error_handlers import register_error_handlers
    from app.core.models import Annotation
    from app.workflow.router import router

    service, database, dataset_id, images, profile, _, calls = workflow
    service.bindings.handlers["managed_inference"] = lambda _: pytest.fail("Visual LLM template must use vision proposals")
    labelme_profile = service.create_profile(ProfileCreate(key="llm-export", name="Visual LLM export",
        config=TaskProfileConfig(export_format="labelme")))
    app = FastAPI()
    app.state.workflow_service = service
    app.include_router(router, prefix="/api")
    register_error_handlers(app)
    with TestClient(app) as client:
        template = next(row for row in client.get("/api/workflows/templates").json() if row["key"] == "vision_review_export_v1")
        assert template["name"] == "用视觉 LLM 标注并检查"
        assert template["nodes"][0]["params"] == {"source": "vision"}
        assert [node["type"] for node in template["nodes"]] == [
            "vision_propose", "segment_proposals", "validate_proposals", "vision_review", "repair_proposals",
            "validate_proposals", "vision_review", "human_review", "snapshot", "export",
        ]
        response = client.post("/api/workflows/runs", json={"idempotency_key": "visual-llm-export",
            "template_id": template["id"], "profile_id": labelme_profile["id"],
            "dataset_id": dataset_id, "image_ids": images})
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        run = drain(service, run_id)
        assert run["status"] == "awaiting_review"
        assert not run["context"].get("model_id")
        assert client.get(f"/api/workflows/runs/{run_id}/export").status_code == 404
        with database.session_factory() as session:
            assert not list(session.scalars(select(Annotation).where(Annotation.dataset_id == dataset_id)))
        for review in run["reviews"]:
            response = client.post(f"/api/workflows/reviews/{review['id']}/decide", json={
                "expected_revision": review["revision"], "decision": "approved"})
            assert response.status_code == 200, response.text
        run = drain(service, run_id)
        assert run["status"] == "completed", run["error_message"]
        response = client.get(f"/api/workflows/runs/{run_id}/export")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            labels = [name for name in archive.namelist() if name.startswith("images/") and name.endswith(".json")]
            assert len(labels) == 2
            assert all(json.loads(archive.read(name))["shapes"][0]["shape_type"] == "rectangle" for name in labels)
        assert calls == {"train": [], "evaluate": []}


def test_visual_llm_template_freezes_proposer_with_existing_model_input(workflow):
    service, _, dataset_id, images, profile, _, _ = workflow
    template = next(row for row in service.list_templates() if row["key"] == "vision_review_export_v1")
    request = RunCreate(idempotency_key="visual-llm-prepared", template_id=template["id"], profile_id=profile["id"],
        dataset_id=dataset_id, image_ids=images, inputs={"model_id": "unrelated-managed-model"})
    endpoint = {"enabled": True, "mode": "local", "api_base": "http://127.0.0.1:9000/v1",
                "model": "vision-fixture", "config_hash": "frozen-config"}
    prepared = service.prepare_run(request, {"proposer": endpoint, "reviewer": endpoint})
    assert set(prepared["vision_approvals"]) == {"proposer", "reviewer"}
    assert prepared["vision_approvals"]["proposer"]["model"] == "vision-fixture"


def test_request_reservation_includes_partial_failure(workflow):
    service = workflow[0]
    service.bindings.handlers["estimate_requests"] = lambda context: {"count": 4}
    service.bindings.handlers["vision_propose"] = lambda _: (_ for _ in ()).throw(ValidationError("fixture_failed", "partial failure"))
    run = start(workflow)
    service.tick(run["id"])
    result = service.get_run(run["id"])
    assert result["reserved_requests"] == 4
    assert result["request_count"] == 0
    assert any(event["kind"] == "requests_reserved" for event in result["events"])


def test_router_contract_arrays_and_review_endpoints(workflow):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.error_handlers import register_error_handlers
    from app.workflow.router import router

    app = FastAPI()
    app.state.workflow_service = workflow[0]
    app.include_router(router, prefix="/api")
    register_error_handlers(app)
    with TestClient(app) as client:
        assert isinstance(client.get("/api/workflows/templates").json(), list)
        assert isinstance(client.get("/api/workflows/profiles").json(), list)
        assert isinstance(client.get("/api/workflows/nodes").json(), list)
        result = drain(workflow[0], start(workflow)["id"])
        review = result["reviews"][0]
        response = client.get(f"/api/workflows/reviews/{review['id']}")
        assert response.status_code == 200
        response = client.post(f"/api/workflows/reviews/{review['id']}/decide", json={"expected_revision": 1, "decision": "approved"})
        assert response.status_code == 200, response.text
        assert response.json()["receipt"]["image_id"] == review["payload"]["image_id"]


def test_save_review_clears_fixed_geometry_and_retains_optional_semantic_notes(workflow):
    service = workflow[0]
    proposer = service.bindings.handlers["vision_propose"]

    def invalid_geometry(context):
        result = proposer(context)
        result["annotations"][0]["bbox"]["x"] = -5
        return result

    service.bindings.handlers["vision_propose"] = invalid_geometry
    service.bindings.handlers["vision_review"] = lambda context: {**context["payload"], "issues": [
        *context["payload"].get("issues", []), {"type": "wrong_class", "object_id": "component", "severity": "warning"}]}
    review = drain(service, start(workflow)["id"])["reviews"][0]
    assert any(issue.get("blocking") for issue in review["payload"]["issues"])
    semantic_id = next(issue["id"] for issue in review["payload"]["issues"] if issue["type"] == "wrong_class")
    annotations = review["payload"]["annotations"]
    annotations[0]["bbox"]["x"] = 5
    fixed = service.save_review(review["id"], ReviewDraft(expected_revision=1, payload={"annotations": annotations}))
    assert not any(issue.get("blocking") for issue in fixed["payload"]["issues"])
    assert any(issue["id"] == semantic_id for issue in fixed["payload"]["issues"])
    service.decide_review(review["id"], ReviewDecision(expected_revision=2, decision="approved",
        semantic_resolutions={semantic_id: "The visible object matches the configured component definition."}))


def test_prepare_freezes_hashes_for_current_and_next_batch(workflow):
    service, _, dataset_id, images, profile, template, _ = workflow
    request = RunCreate(idempotency_key="prepared", template_id=template["id"], profile_id=profile["id"],
        dataset_id=dataset_id, image_ids=images[:1], next_batches=[{"dataset_id": dataset_id, "image_ids": images[1:]}])
    endpoint = {"enabled": True, "mode": "remote", "api_base": "https://example.com/v1", "model": "fixture", "config_hash": "config"}
    prepared = service.prepare_run(request, {"proposer": endpoint, "reviewer": endpoint})
    assert len(prepared["scope"]) == 2
    assert {role for role in prepared["vision_approvals"]} == {"proposer", "reviewer"}
    hashes = {image_hash for batch in prepared["scope"] for image_hash in batch["image_hashes"]}
    assert set(prepared["vision_approvals"]["proposer"]["allowed_image_hashes"]) == hashes
    assert service.list_runs() == []
