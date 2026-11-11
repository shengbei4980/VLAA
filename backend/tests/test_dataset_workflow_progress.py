"""Dataset reads project workflow candidates without applying unconfirmed labels."""
from copy import deepcopy

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event, select

from app.api.dependencies import get_session, get_storage
from app.api.error_handlers import register_error_handlers
from app.api.routers.datasets import router
from app.core.errors import ValidationError
from app.core.models import ClassLabel, Dataset, ImageItem
from app.core.schemas import AnnotationInput, BBox
from app.dataset.annotation.progress import image_progress
from app.dataset.annotation.review import require_confirmed
from app.dataset.annotation.review_router import router as review_router
from app.dataset.annotation.service import replace_annotations
from app.dataset.snapshots import create_snapshot, get_snapshot
from app.workflow.models import WorkflowItem, WorkflowRun
from app.workflow.schemas import NodeDefinition, ReviewDecision, ReviewDraft, RunCreate, TemplateCreate
from test_workflow import approve, drain, start, workflow


@pytest.fixture
def progress_client(workflow):
    service, database, dataset_id, images, *_ = workflow
    with database.session_factory() as session:
        dataset = session.get(Dataset, dataset_id)
        dataset.image_count = len(images)
        dataset.class_count = 1
        session.commit()
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.include_router(review_router, prefix="/api")
    app.dependency_overrides[get_session] = database.sessions
    app.dependency_overrides[get_storage] = lambda: service.storage
    with TestClient(app) as client:
        yield client


def test_dataset_and_image_reads_include_existing_workflow_drafts(workflow, progress_client):
    service, _, dataset_id, _, *_ = workflow
    initial = progress_client.get("/api/datasets").json()[0]
    assert initial["generated_image_count"] == 0
    run_id = start(workflow)["id"]
    processing = progress_client.get(f"/api/datasets/{dataset_id}/images").json()["items"]
    assert all(item["workflow_run_id"] == run_id and item["workflow_status"] == "processing" for item in processing)
    run = drain(service, run_id)
    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["annotated_image_count"] == 0
    assert dataset["generated_image_count"] == dataset["workflow_image_count"] == 2
    assert dataset["pending_review_image_count"] == 2
    assert dataset["confirmed_image_count"] == 0
    assert dataset["latest_workflow_run_id"] == run_id
    assert dataset["latest_workflow_review_id"] in {review["id"] for review in run["reviews"]}
    summary = progress_client.get(f"/api/datasets/{dataset_id}/review-status").json()
    assert summary["generated_image_count"] == 2
    for item in summary["items"]:
        assert item["file_name"].endswith(".png")
        assert item["annotation_count"] == 0
        assert item["workflow_annotation_count"] == 1
        assert item["workflow_status"] == "draft"
        assert item["has_draft"] is True
        assert item["workflow_review_id"]
    images = progress_client.get(f"/api/datasets/{dataset_id}/images").json()["items"]
    assert all(item["workflow_annotation_count"] == 1 for item in images)


def test_edited_draft_counts_unmapped_candidates_and_excludes_rejected_objects(workflow, progress_client):
    service, _, dataset_id, *_ = workflow
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    original = review["payload"]["annotations"][0]
    unmapped = {**original, "object_id": "unmapped", "class_id": None}
    rejected = {**original, "object_id": "rejected", "decision": "rejected"}
    service.save_review(review["id"], ReviewDraft(expected_revision=1, payload={
        "objects": [original, unmapped, rejected], "annotations": [original, unmapped],
    }))
    items = progress_client.get(f"/api/datasets/{dataset_id}/review-status").json()["items"]
    item = next(item for item in items if item["workflow_review_id"] == review["id"])
    assert item["workflow_annotation_count"] == 2
    assert item["annotation_count"] == 0


def test_confirmation_updates_both_formal_and_workflow_progress(workflow, progress_client):
    service, _, dataset_id, *_ = workflow
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["generated_image_count"] == 2
    assert dataset["annotated_image_count"] == dataset["confirmed_image_count"] == 1
    assert dataset["pending_review_image_count"] == 1
    summary = progress_client.get(f"/api/datasets/{dataset_id}/review-status").json()
    item = next(item for item in summary["items"] if item["workflow_review_id"] == review["id"])
    assert item["workflow_status"] == "confirmed"
    assert item["workflow_stale"] is False
    assert item["annotation_count"] == 1
    assert item["has_draft"] is False


def test_duplicate_cancelled_deleted_and_stale_runs_do_not_duplicate_or_block(workflow, progress_client):
    service, database, dataset_id, images, *_ = workflow
    first = drain(service, start(workflow, key="older")["id"])
    second = drain(service, start(workflow, key="newer")["id"])
    assert progress_client.get("/api/datasets").json()[0]["generated_image_count"] == 2
    service.cancel_run(second["id"])
    service.delete_run(second["id"])
    summary = progress_client.get(f"/api/datasets/{dataset_id}/review-status").json()
    assert all(item["workflow_run_id"] == first["id"] for item in summary["items"])
    with database.session_factory() as session:
        class_id = session.scalar(select(ClassLabel.id).where(ClassLabel.dataset_id == dataset_id))
        replace_annotations(session, images[0], [AnnotationInput(type="bbox", class_id=class_id,
            bbox=BBox(x=2, y=2, width=12, height=15))], storage=service.storage)
    with database.session_factory() as session:
        progress = image_progress(session, [session.get(ImageItem, images[0])])[images[0]]
        assert progress["workflow_status"] == "stale"
        assert progress["workflow_pending"] is False
        assert progress["workflow_stale"] is True


def test_dataset_projection_batches_queries(workflow, progress_client):
    service, database, *_ = workflow
    drain(service, start(workflow)["id"])
    statements = []
    def record(connection, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(database.engine, "before_cursor_execute", record)
    try:
        response = progress_client.get("/api/datasets")
    finally:
        event.remove(database.engine, "before_cursor_execute", record)
    assert response.status_code == 200
    # Dataset, count aggregation, images, applied counts, drafts, workflow items,
    # latest artifacts and latest reviews, independent of the number of images.
    assert len(statements) == 8


def test_latest_current_review_wins_over_newer_stale_review(workflow, progress_client):
    service, _, dataset_id, images, *_ = workflow
    older = drain(service, start(workflow, key="older-current")["id"])
    newer = drain(service, start(workflow, key="newer-stale")["id"])
    approved = next(review for review in older["reviews"] if review["payload"]["image_id"] == images[0])
    service.decide_review(approved["id"], ReviewDecision(expected_revision=1, decision="approved"))
    summary = progress_client.get(f"/api/datasets/{dataset_id}/review-status").json()
    item = next(item for item in summary["items"] if item["image_id"] == images[0])
    assert item["workflow_run_id"] == older["id"]
    assert item["workflow_review_id"] == approved["id"]
    assert item["workflow_status"] == "confirmed"
    assert item["workflow_stale"] is False
    assert summary["confirmed_image_count"] == 1
    assert summary["pending_review_image_count"] == 1


def _confirmed_source_and_manual_revision(workflow, *, marked=True):
    service, _, dataset_id, images, profile, *_ = workflow
    source = drain(service, start(workflow)["id"])
    for review in source["reviews"]:
        approve(service, review)
    source = drain(service, source["id"])
    assert source["status"] == "completed"
    source_review = next(review for review in source["reviews"] if review["payload"]["image_id"] == images[0])
    if marked:
        review = service.continue_editing(source_review["id"], source_review["revision"])
        manual = service.get_run(review["run_id"])
    else:
        template = service.create_template(TemplateCreate(
            key="custom_manual_review", name="Manual review", nodes=[
                NodeDefinition(id="review", type="human_review"),
            ],
        ))
        manual = drain(service, service.create_run(RunCreate(
            idempotency_key="manual-revision", template_id=template["id"], profile_id=profile["id"],
            dataset_id=dataset_id, image_ids=images[:1],
        ))["id"])
    assert manual["status"] == "awaiting_review"
    return source, manual


def test_manual_revision_progress_blocks_new_snapshot_until_reconfirmed(workflow, progress_client):
    service, database, dataset_id, images, *_ = workflow
    source, manual = _confirmed_source_and_manual_revision(workflow)
    review = manual["reviews"][0]
    old_snapshot = next(artifact["payload"] for artifact in source["artifacts"]
                        if artifact["kind"] == "dataset_snapshot")

    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["confirmed_image_count"] == 1
    assert dataset["pending_review_image_count"] == 1
    assert dataset["latest_workflow_run_id"] == manual["id"]
    assert dataset["latest_workflow_review_id"] == review["id"]
    with database.session_factory() as session:
        progress = image_progress(session, [session.get(ImageItem, images[0])])[images[0]]
        assert progress["formal_confirmed"] is True
        assert progress["workflow_status"] == "draft"
        assert progress["workflow_pending"] is progress["has_draft"] is True
        assert progress["workflow_annotation_count"] == 1
        with pytest.raises(ValidationError) as blocked:
            create_snapshot(session, service.storage, dataset_id)
        assert blocked.value.error_code == "annotation_review_required"
        assert get_snapshot(session, service.storage, old_snapshot["id"])["manifest"] == old_snapshot["manifest"]

    objects = deepcopy(review["payload"]["objects"])
    objects[0]["bbox"]["x"] = 32
    updated = service.save_review(review["id"], ReviewDraft(
        expected_revision=review["revision"], payload={"objects": objects, "annotations": objects},
    ))
    with database.session_factory() as session:
        with pytest.raises(ValidationError):
            require_confirmed(session, service.storage, dataset_id)
    approve(service, updated)
    assert drain(service, manual["id"])["status"] == "completed"
    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["confirmed_image_count"] == 2
    assert dataset["pending_review_image_count"] == 0
    assert dataset["latest_workflow_review_id"] == review["id"]
    with database.session_factory() as session:
        progress = image_progress(session, [session.get(ImageItem, images[0])])[images[0]]
        assert progress["workflow_status"] == "confirmed"
        assert progress["workflow_pending"] is progress["has_draft"] is False
        current_snapshot = create_snapshot(session, service.storage, dataset_id)
        image = next(image for image in current_snapshot["manifest"]["images"] if image["id"] == images[0])
        assert image["annotations"][0]["bbox"]["x"] == 32
        assert get_snapshot(session, service.storage, old_snapshot["id"])["manifest"] == old_snapshot["manifest"]


def test_unmarked_human_only_run_does_not_replace_annotation_progress(workflow, progress_client):
    service, database, dataset_id, *_ = workflow
    source, _ = _confirmed_source_and_manual_revision(workflow, marked=False)
    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["confirmed_image_count"] == 2
    assert dataset["pending_review_image_count"] == 0
    assert dataset["latest_workflow_run_id"] == source["id"]
    with database.session_factory() as session:
        require_confirmed(session, service.storage, dataset_id)


@pytest.mark.parametrize("state", ["cancelled", "deleted", "stale", "completed"])
def test_closed_or_stale_manual_revision_does_not_block_confirmation(workflow, progress_client, state):
    service, database, dataset_id, images, *_ = workflow
    _, manual = _confirmed_source_and_manual_revision(workflow)
    if state in {"cancelled", "deleted"}:
        service.cancel_run(manual["id"])
        if state == "deleted":
            service.delete_run(manual["id"])
    else:
        with database.session_factory() as session:
            if state == "stale":
                item = session.scalar(select(WorkflowItem).where(WorkflowItem.run_id == manual["id"]))
                item.context_json = {**item.context_json, "base_revision": -1}
            else:
                session.get(WorkflowRun, manual["id"]).status = "completed"
            session.commit()
    dataset = progress_client.get("/api/datasets").json()[0]
    assert dataset["confirmed_image_count"] == 2
    assert dataset["pending_review_image_count"] == 0
    with database.session_factory() as session:
        progress = image_progress(session, [session.get(ImageItem, images[0])])[images[0]]
        assert progress["workflow_pending"] is progress["has_draft"] is False
        require_confirmed(session, service.storage, dataset_id)
