"""Follow reviewed workflow edits through dataset labels and training exports."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from app.api.routers.datasets import _dataset_responses
from app.core.errors import ValidationError
from app.core.models import AnnotationDraft, Dataset, TrainingTask
from app.dataset.annotation.review import review_detail, review_summary
from app.dataset.annotation.service import list_annotations
from app.dataset.snapshots import create_snapshot, get_snapshot
from app.training.schemas import TrainingTaskCreate
from app.training.service import TrainingService
from app.workflow.schemas import ReviewDraft
from test_workflow import approve, drain, start, workflow


def confirmed_batch(workflow):
    service = workflow[0]
    run = drain(service, start(workflow)["id"])
    for review in run["reviews"]:
        approve(service, review)
    run = drain(service, run["id"])
    assert run["status"] == "completed", run.get("error_message")
    snapshot = next(
        artifact["payload"] for artifact in run["artifacts"]
        if artifact["kind"] == "dataset_snapshot"
    )
    return snapshot


def test_workflow_edits_reach_dataset_snapshot_and_independent_training(workflow):
    service, database, dataset_id, images, _, _, _ = workflow
    run = drain(service, start(workflow)["id"])
    expected_x = {}
    for index, review in enumerate(run["reviews"]):
        objects = deepcopy(review["payload"]["annotations"])
        corrected_x = 30 + index * 5
        objects[0]["bbox"]["x"] = corrected_x
        expected_x[review["payload"]["image_id"]] = corrected_x
        changed = service.save_review(review["id"], ReviewDraft(
            expected_revision=review["revision"], payload={"objects": objects, "annotations": objects},
        ))
        approve(service, changed)

    run = drain(service, run["id"])
    assert run["status"] == "completed", run.get("error_message")
    workflow_snapshot = next(
        artifact["payload"] for artifact in run["artifacts"]
        if artifact["kind"] == "dataset_snapshot"
    )
    queue = Mock()
    training = TrainingService(database.session_factory, service.storage, queue)
    with database.session_factory() as session:
        dataset = session.get(Dataset, dataset_id)
        assert _dataset_responses(session, [dataset])[0].annotated_image_count == len(images)
        summary = review_summary(session, service.storage, dataset_id)
        assert summary["confirmed"] == len(images)
        assert summary["pending"] == 0
        for image_id in images:
            assert list_annotations(session, image_id)[0].bbox.x == expected_x[image_id]
            detail = review_detail(session, service.storage, image_id)
            assert detail["receipt"] is not None

        task = training.create_task(session, TrainingTaskCreate(
            dataset_id=dataset_id, device="cpu", epochs=1, batch_size=1,
        ))
        record = session.get(TrainingTask, task.id)
        frozen = get_snapshot(session, service.storage, record.config_json["snapshot_id"])
        for image in frozen["manifest"]["images"]:
            assert image["annotations"][0]["bbox"]["x"] == expected_x[image["id"]]
            label = Path(record.export_path) / "labels" / image["split"] / (image["id"] + ".txt")
            values = label.read_text(encoding="utf-8").split()
            expected_center = (expected_x[image["id"]] + 10) / image["width"]
            assert float(values[1]) == pytest.approx(expected_center)
    queue.submit.assert_called_once_with(task.id)
    for image in workflow_snapshot["manifest"]["images"]:
        assert image["annotations"][0]["bbox"]["x"] == expected_x[image["id"]]


def test_new_workflow_draft_blocks_new_training_but_preserves_frozen_version(workflow):
    service, database, dataset_id, images, _, _, _ = workflow
    old_snapshot = confirmed_batch(workflow)
    run = drain(service, start(workflow, key="second-pass")["id"])
    assert run["status"] == "awaiting_review"
    queue = Mock()
    training = TrainingService(database.session_factory, service.storage, queue)
    with database.session_factory() as session:
        before = review_detail(session, service.storage, images[0])
        assert before["receipt"] is not None
        with pytest.raises(ValidationError) as blocked_snapshot:
            create_snapshot(session, service.storage, dataset_id)
        assert blocked_snapshot.value.error_code == "annotation_review_required"
        with pytest.raises(ValidationError) as blocked_training:
            training.create_task(session, TrainingTaskCreate(dataset_id=dataset_id, device="cpu"))
        assert blocked_training.value.error_code == "annotation_review_required"
        queue.submit.assert_not_called()

        old_task = training.create_task(session, TrainingTaskCreate(
            dataset_id=dataset_id, device="cpu", epochs=1,
        ), snapshot_id=old_snapshot["id"])
        record = session.get(TrainingTask, old_task.id)
        assert record.config_json["snapshot_id"] == old_snapshot["id"]
        after = review_detail(session, service.storage, images[0])
        assert after["receipt"] == before["receipt"]
        assert after["annotations"] == before["annotations"]
    queue.submit.assert_called_once_with(old_task.id)


def test_new_model_draft_blocks_new_training_without_erasing_confirmed_labels(workflow):
    service, database, dataset_id, images, _, _, _ = workflow
    confirmed_batch(workflow)
    with database.session_factory() as session:
        before = review_detail(session, service.storage, images[0])
        session.add(AnnotationDraft(
            image_id=images[0], base_revision=before["revision"], revision=1,
            payload_json={"annotations": [], "source_task_id": "new-auto-candidates"},
        ))
        session.commit()
        with pytest.raises(ValidationError) as blocked:
            create_snapshot(session, service.storage, dataset_id)
        assert blocked.value.error_code == "annotation_review_required"
        queue = Mock()
        training = TrainingService(database.session_factory, service.storage, queue)
        with pytest.raises(ValidationError):
            training.create_task(session, TrainingTaskCreate(dataset_id=dataset_id, device="cpu"))
        queue.submit.assert_not_called()
        after = review_detail(session, service.storage, images[0])
        assert after["receipt"] == before["receipt"]
        assert after["annotations"] == before["annotations"]
