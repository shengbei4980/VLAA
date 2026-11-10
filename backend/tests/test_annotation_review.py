import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ConflictError, ValidationError
from app.core.models import AnnotationDraft, ImageItem
from app.dataset.annotation.masks import MaskService
from app.dataset.annotation.review import confirm_workflow_review, require_confirmed
from app.dataset.exchange.snapshot import export_snapshot_directory
from app.dataset.snapshots import create_snapshot, get_snapshot


def prepared(client, task="detect"):
    dataset = client.post("/api/datasets", json={"name": "Reusable project", "task_type": task}).json()
    class_id = client.post(f"/api/datasets/{dataset['id']}/classes", json={"name": "object"}).json()["id"]
    images = []
    for split in ("train", "val"):
        output = io.BytesIO()
        Image.new("RGB", (32, 24), "white").save(output, format="PNG")
        response = client.post(f"/api/datasets/{dataset['id']}/images/upload", data={"split": split},
                               files={"files": (split + ".png", output.getvalue(), "image/png")})
        images.append(response.json()["items"][0])
    return dataset["id"], class_id, images


def confirm(client, dataset_id, image_id, empty=False):
    url = f"/api/datasets/{dataset_id}/images/{image_id}/review"
    detail = client.get(url).json()
    return client.post(url + "/confirm", json={"expected_revision": detail["revision"],
        "expected_fingerprint": detail["fingerprint"], "confirmed_empty": empty})


def save_box(client, dataset_id, class_id, image_id):
    return client.put(f"/api/datasets/{dataset_id}/images/{image_id}/annotations", json={"annotations": [{
        "type": "bbox", "class_id": class_id, "object_id": "stable-object",
        "bbox": {"x": 4, "y": 3, "width": 12, "height": 10}}]})


def test_manual_source_and_status_do_not_replace_review_receipts(client):
    dataset_id, class_id, images = prepared(client)
    for image in images:
        assert save_box(client, dataset_id, class_id, image["id"]).status_code == 200
    assert client.get(f"/api/datasets/{dataset_id}/export/yolo").json()["error"]["code"] == "annotation_review_required"
    for image in images:
        assert confirm(client, dataset_id, image["id"]).status_code == 200
    assert client.get(f"/api/datasets/{dataset_id}/export/yolo").status_code == 200
    before = client.get(f"/api/datasets/{dataset_id}/images/{images[0]['id']}/review").json()
    assert save_box(client, dataset_id, class_id, images[0]["id"]).status_code == 200
    after = client.get(f"/api/datasets/{dataset_id}/images/{images[0]['id']}/review").json()
    assert after["revision"] == before["revision"] + 1
    assert after["annotations"][0]["object_id"] == "stable-object"
    assert after["review_status"] == "pending"
    assert client.post(f"/api/datasets/{dataset_id}/images/{images[0]['id']}/review/confirm", json={
        "expected_revision": before["revision"], "expected_fingerprint": before["fingerprint"]}).status_code == 409


def test_snapshot_keeps_confirmed_negatives_and_freezes_images_labels_and_rules(client, tmp_path):
    dataset_id, class_id, images = prepared(client)
    save_box(client, dataset_id, class_id, images[0]["id"])
    assert confirm(client, dataset_id, images[0]["id"]).status_code == 200
    assert confirm(client, dataset_id, images[1]["id"]).status_code == 422
    assert confirm(client, dataset_id, images[1]["id"], empty=True).status_code == 200
    storage = client.app.state.storage
    with client.app.state.database.session_factory() as session:
        snapshot = create_snapshot(session, storage, dataset_id)
        session.commit()
        exported = export_snapshot_directory(snapshot, storage, tmp_path / "export")
        assert exported["counts"] == {"train": 1, "val": 1, "test": 0}
        assert exported["annotated_image_counts"]["val"] == 0
        assert list((tmp_path / "export" / "labels" / "val").glob("*.txt"))[0].read_text() == ""
    save_box(client, dataset_id, class_id, images[1]["id"])
    with client.app.state.database.session_factory() as session:
        frozen = get_snapshot(session, storage, snapshot["id"])
        assert frozen["manifest"]["images"][1]["annotations"] == []
        with pytest.raises(ValidationError, match="human confirmation"):
            require_confirmed(session, storage, dataset_id)


def test_coco_rle_roundtrip_preserves_holes_components_area_and_regions(client):
    dataset_id, class_id, images = prepared(client, "segment")
    storage = client.app.state.storage
    mask = np.zeros((24, 32), dtype=bool)
    mask[3:17, 2:18] = True
    mask[7:12, 6:11] = False
    mask[19:22, 25:30] = True
    masks = MaskService(storage)
    mask_ref = masks.write(mask).mask_ref
    for image in images:
        response = client.put(f"/api/datasets/{dataset_id}/images/{image['id']}/annotations", json={"annotations": [{
            "type": "polygon", "class_id": class_id, "mask_ref": mask_ref}]})
        assert response.status_code == 200
        assert confirm(client, dataset_id, image["id"]).status_code == 200
    yolo = client.get(f"/api/datasets/{dataset_id}/export/yolo")
    assert yolo.status_code == 422
    assert yolo.json()["error"]["code"] == "yolo_geometry_loss"
    exported = client.get(f"/api/datasets/{dataset_id}/export/coco")
    assert exported.status_code == 200
    selected = client.get(f"/api/datasets/{dataset_id}/export/coco", params={"image_ids": images[0]["id"]})
    assert selected.status_code == 200
    with zipfile.ZipFile(io.BytesIO(selected.content)) as archive:
        selected_payload = json.loads(archive.read("annotations.json"))
        assert len(selected_payload["images"]) == 1
        assert len(selected_payload["annotations"]) == 1
        assert sum(name.startswith("images/") for name in archive.namelist()) == 1
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        payload = json.loads(archive.read("annotations.json"))
        assert payload["annotations"][0]["area"] == int(mask.sum())
        assert isinstance(payload["annotations"][0]["segmentation"], dict)
    imported = client.post("/api/datasets/import/coco", data={"name": "Imported masks", "task_type": "segment"},
                           files={"file": ("masks.zip", exported.content, "application/zip")})
    assert imported.status_code == 201, imported.text
    imported_id = imported.json()["dataset"]["id"]
    image = client.get(f"/api/datasets/{imported_id}/images").json()["items"][0]
    annotation = client.get(f"/api/datasets/{imported_id}/images/{image['id']}/annotations").json()[0]
    assert np.array_equal(masks.decode(annotation["mask_ref"]), mask)
    assert client.get(f"/api/datasets/{imported_id}/review-status").json()["pending"] == 2


def test_automatic_draft_requires_explicit_application_and_confirmation(client):
    dataset_id, class_id, images = prepared(client)
    image_id = images[0]["id"]
    save_box(client, dataset_id, class_id, image_id)
    with client.app.state.database.session_factory() as session:
        image = session.get(ImageItem, image_id)
        session.add(AnnotationDraft(image_id=image_id, base_revision=image.annotation_revision, revision=1,
                                     payload_json={"annotations": [], "source_task_id": "auto"}))
        session.commit()
    response = confirm(client, dataset_id, image_id)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "annotation_draft_pending"
    save_box(client, dataset_id, class_id, image_id)
    assert confirm(client, dataset_id, image_id).status_code == 200


def test_workflow_confirmation_rejects_changed_base_without_partial_apply(client):
    dataset_id, class_id, images = prepared(client)
    image_id = images[0]["id"]
    detail = client.get(f"/api/datasets/{dataset_id}/images/{image_id}/review").json()
    save_box(client, dataset_id, class_id, image_id)
    with client.app.state.database.session_factory() as session:
        with pytest.raises(ConflictError):
            confirm_workflow_review(session, client.app.state.storage, {"image_id": image_id,
                "image_hash": detail["image_hash"], "base_revision": detail["revision"], "annotations": []},
                {"confirmed_empty": True})
        session.rollback()
    assert len(client.get(f"/api/datasets/{dataset_id}/images/{image_id}/annotations").json()) == 1


def test_class_rule_change_invalidates_existing_review(client):
    dataset_id, class_id, images = prepared(client)
    save_box(client, dataset_id, class_id, images[0]["id"])
    confirm(client, dataset_id, images[0]["id"])
    client.post(f"/api/datasets/{dataset_id}/classes", json={"name": "other"})
    detail = client.get(f"/api/datasets/{dataset_id}/images/{images[0]['id']}/review").json()
    assert detail["review_status"] == "pending"
    assert detail["receipt"] is None


@pytest.mark.parametrize("name", ["../outside", "..\\outside", "C:\\outside", "CON", "a/b"])
def test_classification_folder_names_stay_inside_export(name):
    from app.dataset.exchange.yolo import safe_class_directory

    with pytest.raises(ValidationError):
        safe_class_directory(name)
