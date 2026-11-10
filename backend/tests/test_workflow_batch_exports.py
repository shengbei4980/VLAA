"""Batch exports keep each saved review independent and preserve native masks."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.models import Annotation, ImageItem
from app.dataset.annotation.masks import MaskService, encode_mask
from app.workflow.batch_exports import BatchExporter, batch_download
from app.workflow.models import WorkflowArtifact, WorkflowReview
from app.workflow.preview_renderer import render_preview
from app.workflow.projects import ProjectRegistry
from app.workflow.schemas import ReviewDecision, ReviewDraft, RunCreate
from test_workflow import drain, workflow
from test_workflow_exports import native_mask
from test_workflow_labelme_export import labelme, object_mask


def read_bundle(result):
    archive = zipfile.ZipFile(result["saved_path"])
    return archive, json.loads(archive.read("manifest.json"))


def test_two_same_named_images_export_independently_and_retry_is_readonly(labelme, monkeypatch):
    _, service, database, run = labelme
    with database.session_factory() as session:
        for item in run["items"]:
            session.get(ImageItem, item["subject_ref"]).file_name = "same.png"
        session.commit()
        before = (session.scalar(select(func.count()).select_from(WorkflowArtifact)),
                  session.scalar(select(func.count()).select_from(Annotation)))
    original_reviews = [service.get_review(review["id"]) for review in run["reviews"]]
    monkeypatch.setattr(service, "get_run", lambda *_: pytest.fail("Batch must not load every payload through get_run"))
    exporter = BatchExporter(service)
    result = exporter.export_run(run["id"])
    assert result["total_images"] == result["exported_images"] == result["pending_images"] == 2
    assert result["failed_images"] == result["skipped_images"] == 0
    with read_bundle(result)[0] as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert len(archive.namelist()) == 5
        assert manifest["coordinate_space"] == "image_pixel"
        assert {record["image_id"] for record in manifest["results"]} == {item["subject_ref"] for item in run["items"]}
        for row in manifest["results"]:
            assert row["review_revision"] == 1
            assert row["export_state"] == "draft"
            for metadata in row["files"].values():
                assert sha256(archive.read(metadata["path"])).hexdigest() == metadata["sha256"]
            document = json.loads(archive.read(row["files"]["json"]["path"]))
            assert document["imagePath"] == "same.png"
            assert len(document["shapes"]) == 1
            with Image.open(io.BytesIO(archive.read(row["files"]["image"]["path"]))) as image:
                assert image.size == (80, 400)
    assert exporter.export_run(run["id"]) == result
    assert batch_download(service.storage, run["id"], result["file_name"]).read_bytes() == Path(result["saved_path"]).read_bytes()
    assert [service.get_review(review["id"]) for review in run["reviews"]] == original_reviews
    with database.session_factory() as session:
        assert before == (session.scalar(select(func.count()).select_from(WorkflowArtifact)),
                          session.scalar(select(func.count()).select_from(Annotation)))
    assert list(service.storage.exports_dir.glob("*.tmp")) == []


@pytest.mark.parametrize("labelme", ["segment"], indirect=True)
def test_batch_masks_preserve_holes_and_render_in_original_pixels(labelme):
    _, service, _, run = labelme
    result = BatchExporter(service).export_run(run["id"])
    assert result["exported_images"] == 2
    with read_bundle(result)[0] as archive:
        row = result["results"][0]
        document = json.loads(archive.read(row["files"]["json"]["path"]))
        assert np.array_equal(native_mask(document["shapes"][0], 80, 400), object_mask())
        with Image.open(io.BytesIO(archive.read(row["files"]["image"]["path"]))) as image:
            pixels = np.asarray(image)
        assert tuple(pixels[80, 30, :3]) != (255, 255, 255)
        assert tuple(pixels[50, 30, :3]) == (255, 255, 255)
        assert tuple(pixels[320, 70, :3]) != (255, 255, 255)


def test_one_changed_original_fails_without_losing_other_image_and_can_retry(labelme):
    _, service, database, run = labelme
    first_id = run["items"][0]["subject_ref"]
    with database.session_factory() as session:
        image = session.get(ImageItem, first_id)
        path = service.storage.image_path(image.dataset_id, image.storage_name)
    original = path.read_bytes()
    path.write_bytes(b"corrupted original")
    exporter = BatchExporter(service)
    result = exporter.export_run(run["id"])
    assert result["failed_images"] == result["exported_images"] == 1
    failed = next(row for row in result["results"] if row["status"] == "failed")
    assert failed["image_id"] == first_id
    assert failed["error_code"] == "workflow_labelme_image_changed"
    with read_bundle(result)[0] as archive:
        assert f"json/{first_id}.json" not in archive.namelist()
        assert f"images/{first_id}.png" not in archive.namelist()
        assert len(archive.namelist()) == 3
    path.write_bytes(original)
    retried = exporter.export_run(run["id"])
    assert retried["exported_images"] == 2 and retried["failed_images"] == 0


def test_excluded_and_not_ready_images_are_reported_as_skipped(labelme):
    _, service, database, run = labelme
    first, second = run["reviews"]
    service.decide_review(first["id"], ReviewDecision(expected_revision=1, decision="excluded", reason="Fixture exclusion"))
    with database.session_factory() as session:
        session.delete(session.get(WorkflowReview, second["id"]))
        session.commit()
    result = BatchExporter(service).export_run(run["id"])
    assert result["total_images"] == result["skipped_images"] == 2
    assert result["exported_images"] == result["failed_images"] == 0
    assert all(row.get("reason") for row in result["results"])
    with read_bundle(result)[0] as archive:
        assert archive.namelist() == ["manifest.json"]


@pytest.mark.parametrize("change_method", ["revision", "decision"])
def test_concurrent_review_change_cannot_mix_json_and_png_states(labelme, monkeypatch, change_method):
    _, service, _, run = labelme
    first = run["reviews"][0]
    original = service.save_review_labelme
    changed = False

    def update_after_json(review_id, draft):
        nonlocal changed
        result = original(review_id, draft)
        if review_id == first["id"] and not changed:
            changed = True
            if change_method == "revision":
                service.save_review(review_id, ReviewDraft(expected_revision=1, payload={"objects": first["payload"]["objects"]}))
            else:
                service.decide_review(review_id, ReviewDecision(expected_revision=1, decision="approved"))
        return result

    monkeypatch.setattr(service, "save_review_labelme", update_after_json)
    result = BatchExporter(service).export_run(run["id"])
    assert result["failed_images"] == result["exported_images"] == 1
    failed = next(row for row in result["results"] if row["status"] == "failed")
    assert failed["error_code"] == "workflow_batch_review_changed"
    assert "image_path" not in failed
    assert BatchExporter(service).export_run(run["id"])["failed_images"] == 0


def test_one_missing_saved_file_is_reported_without_stopping_other_images(labelme, monkeypatch):
    _, service, _, run = labelme
    exporter = BatchExporter(service)
    original = exporter._export_one

    def remove_first_file(expected, profile, include_previews):
        result = original(expected, profile, include_previews)
        if expected["id"] == run["reviews"][0]["id"]:
            service.storage.export_path(result["json_file_name"]).unlink()
        return result

    monkeypatch.setattr(exporter, "_export_one", remove_first_file)
    result = exporter.export_run(run["id"])
    assert result["failed_images"] == result["exported_images"] == 1
    with read_bundle(result)[0] as archive:
        assert len(archive.namelist()) == 3
        for row in result["results"]:
            if row["status"] == "exported":
                for metadata in row["files"].values():
                    assert sha256(archive.read(metadata["path"])).hexdigest() == metadata["sha256"]


def test_streamed_member_change_never_publishes_a_package_with_invalid_hashes(labelme, monkeypatch):
    from app.workflow import batch_exports

    _, service, _, run = labelme
    original = batch_exports._write_member

    def changed_member(archive, source, name, expected_hash):
        original(archive, io.BytesIO(b"changed during copy"), name, expected_hash)

    monkeypatch.setattr(batch_exports, "_write_member", changed_member)
    with pytest.raises(ConflictError, match="打包期间"):
        BatchExporter(service).export_run(run["id"])
    assert list(service.storage.exports_dir.glob("*.zip")) == []
    assert list(service.storage.exports_dir.glob("*.tmp")) == []


def test_json_only_batch_and_confirmed_state_are_explicit(labelme):
    _, service, _, run = labelme
    first = run["reviews"][0]
    service.decide_review(first["id"], ReviewDecision(expected_revision=1, decision="approved"))
    result = BatchExporter(service).export_run(run["id"], include_previews=False)
    assert result["exported_images"] == 2 and result["pending_images"] == 1
    assert {row["export_state"] for row in result["results"]} == {"draft", "confirmed"}
    assert all("image_path" not in row for row in result["results"])
    with read_bundle(result)[0] as archive:
        assert len(archive.namelist()) == 3


def test_download_validates_run_identity_and_content_hash(labelme):
    _, service, _, run = labelme
    result = BatchExporter(service).export_run(run["id"], include_previews=False)
    with pytest.raises(ValidationError):
        batch_download(service.storage, "another-run", result["file_name"])
    with pytest.raises(ValidationError):
        batch_download(service.storage, run["id"], "../file.zip")
    with pytest.raises(NotFoundError):
        batch_download(service.storage, run["id"], f"{run['id']}-annotation-batch-{'0' * 64}.zip")
    Path(result["saved_path"]).write_bytes(b"changed")
    with pytest.raises(ConflictError):
        batch_download(service.storage, run["id"], result["file_name"])


def test_batch_uses_frozen_project_folders_for_json_images_and_package(labelme, tmp_path):
    _, service, _, original = labelme
    root = tmp_path / "project-results"
    project = ProjectRegistry(service.project_root).register("Tower study", str(root))
    run = service.create_run(RunCreate(idempotency_key="project-batch", template_id=original["template_id"],
        profile_id=original["profile_id"], dataset_id=original["context"]["dataset_id"],
        image_ids=[item["subject_ref"] for item in original["items"]], project_id=project["id"]))
    run = drain(service, run["id"])
    result = BatchExporter(service).export_run(run["id"])
    annotations = root / ("workflow-" + run["id"]) / "annotations"
    assert result["exported_images"] == 2 and result["failed_images"] == 0
    assert Path(result["saved_path"]).parent == annotations
    for row in result["results"]:
        assert Path(row["json_path"]).parent == annotations / "json"
        assert Path(row["image_path"]).parent == annotations / "images"
        assert Path(row["json_path"]).is_file() and Path(row["image_path"]).is_file()
    assert batch_download(service.storage, run["id"], result["file_name"]).parent == service.storage.exports_dir


def test_renderer_inline_mask_replaces_reference_and_keeps_rejected_pixels_clean(labelme):
    _, service, database, run = labelme
    payload = deepcopy(run["reviews"][0]["payload"])
    mask = np.zeros((400, 80), dtype=bool)
    mask[100:140, 20:60] = True
    other = np.zeros_like(mask)
    other[200:240, 20:60] = True
    payload["objects"][0].update(type="polygon", mask=encode_mask(mask), mask_ref=MaskService(service.storage).write(other).mask_ref)
    rejected = deepcopy(payload["objects"][0])
    rejected.update(object_id="rejected", decision="rejected", mask=encode_mask(other))
    payload["objects"].append(rejected)
    with database.session_factory() as session:
        image = session.get(ImageItem, payload["image_id"])
        image_path = service.storage.image_path(image.dataset_id, image.storage_name)
    content = render_preview(service.storage, image_path, payload, run["profile_snapshot"]["class_rules"])
    with Image.open(io.BytesIO(content)) as image:
        pixels = np.asarray(image)
    assert tuple(pixels[120, 30, :3]) != (255, 255, 255)
    assert tuple(pixels[220, 30, :3]) == (255, 255, 255)
    payload["objects"][0]["mask"] = encode_mask(np.ones((3, 3), dtype=bool))
    with pytest.raises(ValidationError, match="原图尺寸"):
        render_preview(service.storage, image_path, payload, run["profile_snapshot"]["class_rules"])
