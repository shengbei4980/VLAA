"""Confirmed annotations can be revised without changing prior reviewed results."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import select

from app.core.errors import ConflictError
from app.core.models import AnnotationDraft, ClassLabel, ImageItem
from app.core.schemas import AnnotationInput, BBox
from app.dataset.annotation.masks import MaskService
from app.dataset.annotation.review import confirm_image, review_detail
from app.dataset.annotation.service import list_annotations, replace_annotations
from app.workflow.models import WorkflowItem, WorkflowRun
from app.workflow.schemas import ReviewDecision, ReviewDraft
from test_workflow import approve, drain, start, workflow
from test_workflow_labelme_export import labelme


def confirmed_source(workflow, *, finish=False):
    service = workflow[0]
    run = drain(service, start(workflow)["id"])
    review = approve(service, run["reviews"][0])
    if finish:
        approve(service, run["reviews"][1])
        drain(service, run["id"])
    return review


def test_continue_editing_has_independent_history_and_can_reconfirm_without_models(workflow):
    service, database, _, _, _, _, calls = workflow
    original = confirmed_source(workflow, finish=True)
    old_run = service.get_run(original["run_id"])
    old_snapshot = deepcopy(next(artifact for artifact in old_run["artifacts"] if artifact["kind"] == "dataset_snapshot"))
    templates = service.list_templates()
    with database.session_factory() as session:
        image = session.get(ImageItem, original["payload"]["image_id"])
        base_revision = image.annotation_revision

    def no_models(context):
        pytest.fail("Continuing manual editing must not run a model")

    service.bindings.handlers.update({kind: no_models for kind in ("vision_propose", "vision_review", "segment_proposals", "train", "evaluate")})
    revised = service.continue_editing(original["id"], original["revision"])
    assert revised["id"] != original["id"] and revised["run_id"] != original["run_id"]
    assert revised["base_revision"] == base_revision
    assert revised["status"] == "pending" and revised["payload"]["issues"] == []
    assert service.list_templates() == templates
    new_run = service.get_run(revised["run_id"])
    assert new_run["context"]["annotation_edit_of"] == original["id"]
    assert new_run["context"]["source_annotation_revision"] == base_revision
    with database.session_factory() as session:
        assert session.get(WorkflowRun, revised["run_id"]).parent_run_id is None

    objects = deepcopy(revised["payload"]["objects"])
    objects[0]["bbox"]["width"] = 25
    saved = service.save_review(revised["id"], ReviewDraft(expected_revision=1, payload={"objects": objects}))
    assert service.continue_editing(original["id"], original["revision"]) == saved
    assert service.get_review(revised["id"]) == saved
    decided = approve(service, saved)
    assert decided["receipt"] != original["receipt"]
    completed = drain(service, revised["run_id"])
    assert completed["status"] == "completed", completed["error_message"]
    assert len(calls["train"]) == len(calls["evaluate"]) == 1
    assert service.get_review(original["id"]) == original
    assert next(artifact for artifact in service.get_run(original["run_id"])["artifacts"] if artifact["kind"] == "dataset_snapshot") == old_snapshot
    with database.session_factory() as session:
        image = session.get(ImageItem, original["payload"]["image_id"])
        assert image.annotation_revision == base_revision + 1
        assert list_annotations(session, image.id)[0].bbox.width == 25
        assert review_detail(session, service.storage, image.id)["receipt"] is not None


def test_continue_editing_serializes_duplicate_clicks_and_preserves_newer_draft(workflow):
    service = workflow[0]
    original = confirmed_source(workflow)
    with ThreadPoolExecutor(max_workers=2) as executor:
        revisions = list(executor.map(lambda _: service.continue_editing(original["id"], original["revision"]), range(2)))
    assert revisions[0]["id"] == revisions[1]["id"]
    saved = service.save_review(revisions[0]["id"], ReviewDraft(expected_revision=1, payload={"annotations": [], "objects": []}))
    assert service.continue_editing(original["id"], original["revision"]) == saved


def test_continue_editing_does_not_return_a_draft_for_different_pixels(workflow):
    service, database, *_ = workflow
    original = confirmed_source(workflow)
    revised = service.continue_editing(original["id"], original["revision"])
    with database.session_factory() as session:
        item = session.get(WorkflowItem, revised["item_id"])
        item.context_json = {**item.context_json, "image_hash": "outdated-pixels"}
        session.commit()
    with pytest.raises(ConflictError, match="unfinished annotation work"):
        service.continue_editing(original["id"], original["revision"])


def test_continue_editing_background_allows_adding_a_previously_missing_object(workflow):
    service = workflow[0]
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    class_id = review["payload"]["annotations"][0]["class_id"]
    empty = service.save_review(review["id"], ReviewDraft(expected_revision=1, payload={"annotations": [], "objects": []}))
    original = service.decide_review(empty["id"], ReviewDecision(expected_revision=empty["revision"], decision="background"))
    revised = service.continue_editing(original["id"], original["revision"])
    assert revised["payload"]["objects"] == []
    saved = service.save_review(revised["id"], ReviewDraft(expected_revision=1, payload={"objects": [{
        "object_id": "new-object", "class_id": class_id, "type": "bbox", "source": "manual",
        "bbox": {"x": 5, "y": 10, "width": 30, "height": 40},
    }]}))
    assert approve(service, saved)["receipt"]["confirmed_empty"] is False


@pytest.mark.parametrize("status", ["completed", "cancelled", "needs_attention"])
def test_continue_editing_accepts_confirmed_source_when_original_run_is_closed(workflow, status):
    service, database, *_ = workflow
    original = confirmed_source(workflow)
    with database.session_factory() as session:
        run = session.get(WorkflowRun, original["run_id"])
        run.status, run.cancel_requested = status, status == "cancelled"
        session.commit()
    revised = service.continue_editing(original["id"], original["revision"])
    assert revised["status"] == "pending"
    assert approve(service, revised)["status"] == "decided"


def test_continue_editing_skips_cancelled_drafts_but_rejects_deleted_source(workflow):
    from app.core.time import utc_now

    service, database, *_ = workflow
    original = confirmed_source(workflow)
    first = service.continue_editing(original["id"], original["revision"])
    service.cancel_run(first["run_id"])
    second = service.continue_editing(original["id"], original["revision"])
    assert second["id"] != first["id"]
    with database.session_factory() as session:
        session.get(WorkflowRun, original["run_id"]).deleted_at = utc_now()
        session.commit()
    with pytest.raises(ConflictError, match="Restore"):
        service.continue_editing(original["id"], original["revision"])


def test_continue_editing_checks_revision_original_image_and_other_editor_drafts(workflow):
    service, database, *_ = workflow
    original = confirmed_source(workflow)
    with pytest.raises(ConflictError, match="changed"):
        service.continue_editing(original["id"], original["revision"] + 1)
    with database.session_factory() as session:
        image = session.get(ImageItem, original["payload"]["image_id"])
        path = service.storage.image_path(image.dataset_id, image.storage_name)
        session.add(AnnotationDraft(image_id=image.id, base_revision=image.annotation_revision, payload_json={}))
        session.commit()
    with pytest.raises(ConflictError, match="editor draft"):
        service.continue_editing(original["id"], original["revision"])
    with database.session_factory() as session:
        session.delete(session.get(AnnotationDraft, original["payload"]["image_id"]))
        session.commit()
    content = io.BytesIO()
    Image.new("RGB", (80, 400), "black").save(content, format="PNG")
    path.write_bytes(content.getvalue())
    with pytest.raises(ConflictError, match="original image changed"):
        service.continue_editing(original["id"], original["revision"])


def test_continue_editing_uses_latest_formal_labels_and_current_classes(workflow):
    service, database, dataset_id, *_ = workflow
    original = confirmed_source(workflow)
    with database.session_factory() as session:
        old_class = session.scalar(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id))
        old_class.name = "renamed component"
        added = ClassLabel(id="cls-new", dataset_id=dataset_id, class_index=1, name="new component")
        session.add(added)
        session.flush()
        image = session.get(ImageItem, original["payload"]["image_id"])
        replace_annotations(session, image.id, [AnnotationInput(type="bbox", class_id=added.id,
                            object_id="stable-user-object", bbox=BBox(x=30, y=30, width=10, height=20),
                            provenance_json={"editor": "existing-manual-label"})], storage=service.storage)
        detail = review_detail(session, service.storage, image.id)
        confirm_image(session, service.storage, image.id, expected_revision=detail["revision"], expected_fingerprint=detail["fingerprint"])
        latest_revision = image.annotation_revision
        session.commit()
    revised = service.continue_editing(original["id"], original["revision"])
    assert revised["base_revision"] == latest_revision
    annotation = revised["payload"]["objects"][0]
    assert annotation["object_id"] == "stable-user-object"
    assert annotation["class_id"] == "cls-new" and annotation["bbox"]["x"] == 30
    assert annotation["provenance_json"] == {"editor": "existing-manual-label"}
    assert revised["payload"]["issues"] == []
    classes = service.get_run(revised["run_id"])["profile_snapshot"]["class_rules"]
    assert {row["name"] for row in classes} == {"renamed component", "new component"}
    approve(service, revised)


@pytest.mark.parametrize("labelme", ["segment"], indirect=True)
def test_continue_editing_segment_masks_keep_manual_pixels_and_support_new_paint(labelme):
    client, service, database, run = labelme
    original = approve(service, run["reviews"][0])
    masks = MaskService(service.storage)
    response = client.post(f"/api/workflows/reviews/{original['id']}/continue-editing", json={"expected_revision": original["revision"]})
    assert response.status_code == 201, response.text
    revised = response.json()
    original_mask = masks.decode(revised["payload"]["objects"][0]["mask_ref"])
    painted = original_mask.copy()
    painted[370:380, 3:12] = True
    objects = deepcopy(revised["payload"]["objects"])
    objects[0]["mask_ref"] = masks.write(painted).mask_ref
    saved = service.save_review(revised["id"], ReviewDraft(expected_revision=1, payload={"objects": objects}))
    approve(service, saved)
    with database.session_factory() as session:
        formal = list_annotations(session, revised["payload"]["image_id"])
        assert np.array_equal(masks.decode(formal[0].mask_ref), painted)
    assert service.get_review(original["id"]) == original


def test_continue_editing_retains_project_root_with_separate_run_exports(workflow, tmp_path):
    service = workflow[0]
    service.project_root = tmp_path / "workbench"
    service.project_root.mkdir()
    project = service.create_project("塔体", str(tmp_path / "project"))
    run = drain(service, start(workflow, project_id=project["id"])["id"])
    original = approve(service, run["reviews"][0])
    revised = service.continue_editing(original["id"], original["revision"])
    assert revised["project"] == original["project"]
    assert revised["export_locations"] != original["export_locations"]
    root = Path(project["root_path"])
    assert Path(revised["export_locations"]["json_path"]) == root / ("workflow-" + revised["run_id"]) / "annotations" / "json"


def test_continue_editing_does_not_redirect_an_existing_projects_outputs(workflow, tmp_path):
    from app.storage_settings.bootstrap import read_config, write_config

    service = workflow[0]
    service.project_root = tmp_path / "workbench"
    service.project_root.mkdir()
    project = service.create_project("塔体", str(tmp_path / "project"))
    run = drain(service, start(workflow, project_id=project["id"])["id"])
    original = approve(service, run["reviews"][0])
    config = read_config(service.project_root)
    new_root = str(tmp_path / "redirected")
    config["workflow_projects"][0]["root_path"] = new_root
    next(row for row in config["destinations"] if row["id"] == project["destination_id"])["path"] = new_root
    write_config(service.project_root, config)
    with pytest.raises(ConflictError, match="output destination changed"):
        service.continue_editing(original["id"], original["revision"])
    assert service.get_review(original["id"]) == original


def test_continue_editing_requires_a_confirmed_source(workflow):
    service = workflow[0]
    run = drain(service, start(workflow)["id"])
    review = run["reviews"][0]
    with pytest.raises(ConflictError, match="Only confirmed"):
        service.continue_editing(review["id"], review["revision"])
    excluded = service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="excluded", reason="outside scope"))
    with pytest.raises(ConflictError, match="Only confirmed"):
        service.continue_editing(excluded["id"], excluded["revision"])
