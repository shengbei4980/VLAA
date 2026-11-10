"""Exercise real export files, including native LabelMe PNG masks and portable RLE."""
import base64
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from app.core.errors import ValidationError
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService, decode_mask, polygon_mask
from app.workflow.exports import export_snapshot
from app.workflow.schemas import TaskProfileConfig


@pytest.fixture
def frozen_image(tmp_path):
    storage = Storage(tmp_path / "data", tmp_path / "imports")
    root = storage.data_dir / "snapshots" / "snapshot-export"
    (root / "images").mkdir(parents=True)
    source = root / "images" / "image-export.png"
    Image.new("RGB", (24, 20), "white").save(source)
    image = {
        "id": "image-export", "file_name": "中文 原图.png", "path": "images/image-export.png",
        "width": 24, "height": 20, "split": "val", "source_group_id": "tower-group",
        "image_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "annotations": [],
        "revision": 3, "fingerprint": "reviewed-fingerprint",
        "receipt": {"id": "receipt-source", "confirmed_empty": False},
        "rules": {"version": "v1", "scope": "tower"},
    }
    snapshot = {"id": "snapshot-export", "root": str(root), "manifest": {
        "schema_version": 1, "coordinate_space": "image_pixel", "task_type": "segment",
        "classes": [{"id": "class-part", "index": 0, "name": "塔架部件"}], "images": [image],
    }}
    return snapshot, storage, image


def labelme_document(snapshot, storage):
    result = export_snapshot(snapshot, storage, "labelme")
    with zipfile.ZipFile(storage.export_path(result["file_name"])) as archive:
        document = json.loads(archive.read("images/val/image-export.json"))
        files = {name: archive.read(name) for name in archive.namelist()}
    return result, document, files


def native_mask(shape, width=24, height=20):
    # LabelMe integer mask bounds include the final pixel, as in its reference reader.
    (left, top), (right, bottom) = shape["points"]
    with Image.open(io.BytesIO(base64.b64decode(shape["mask"], validate=True))) as image:
        patch = np.array(image).astype(bool)
    assert patch.shape == (bottom - top + 1, right - left + 1)
    full = np.zeros((height, width), dtype=bool)
    full[top:bottom + 1, left:right + 1] = patch
    return full


def test_labelme_profile_format_is_accepted():
    assert TaskProfileConfig(export_format="labelme").export_format == "labelme"


def test_labelme_native_mask_preserves_holes_disconnected_and_edge_pixels(frozen_image):
    snapshot, storage, image = frozen_image
    mask = np.zeros((20, 24), dtype=bool)
    mask[2:15, 1:13] = True
    mask[5:10, 4:8] = False
    mask[18:20, 22:24] = True
    mask[0, 0] = True
    reference = MaskService(storage).write(mask).mask_ref
    image["annotations"] = [{"id": "annotation-part", "object_id": "object-part", "class_id": "class-part",
                             "type": "polygon", "mask_ref": reference, "source": "auto",
                             "provenance_json": {"model": "reviewed-source"}}]
    result, document, files = labelme_document(snapshot, storage)
    shape = document["shapes"][0]
    assert shape["shape_type"] == "mask"
    assert np.array_equal(native_mask(shape), mask)
    assert np.array_equal(decode_mask(shape["starter_rle"]), mask)
    assert shape["starter_annotation"] == image["annotations"][0]
    assert files[reference] == MaskService(storage).path(reference).read_bytes()
    assert result["capabilities"]["minimum_labelme_version"] == "5.8.0"
    assert hashlib.sha256(storage.export_path(result["file_name"]).read_bytes()).hexdigest() == result["sha256"]


def test_labelme_bundle_preserves_supervision_rules_receipt_and_source(frozen_image):
    snapshot, storage, image = frozen_image
    masks = MaskService(storage)
    roi = np.ones((20, 24), dtype=bool)
    ignore = np.zeros((20, 24), dtype=bool)
    ignore[:, 10:12] = True
    image["roi_mask_ref"] = masks.write(roi).mask_ref
    image["ignore_mask_ref"] = masks.write(ignore).mask_ref
    image["receipt"]["confirmed_empty"] = True
    _, document, files = labelme_document(snapshot, storage)
    assert document["imagePath"] == "image-export.png"
    assert document["imageData"] is None
    assert document["starter_original_file_name"] == image["file_name"]
    assert document["split"] == "val" and document["source_group_id"] == "tower-group"
    assert document["starter_review_receipt"] == image["receipt"]
    assert document["starter_rules"] == image["rules"]
    assert document["shapes"] == []
    assert np.array_equal(decode_mask(document["starter_roi"]), roi)
    assert np.array_equal(decode_mask(document["starter_ignore"]), ignore)
    assert image["roi_mask_ref"] in files and image["ignore_mask_ref"] in files
    assert "images/val/image-export.png" in files and "workflow_manifest.json" in files


def test_labelme_polygon_uses_canonical_rasterization(frozen_image):
    snapshot, storage, image = frozen_image
    polygon = [[1.5, 2.5], [16.5, 2.5], [14.25, 17.5], [3.25, 16.5]]
    image["annotations"] = [{"class_id": "class-part", "type": "polygon", "polygon": polygon}]
    _, document, _ = labelme_document(snapshot, storage)
    assert np.array_equal(native_mask(document["shapes"][0]), polygon_mask(polygon, 24, 20))
    assert document["shapes"][0]["starter_annotation"]["polygon"] == polygon


def test_labelme_detection_preserves_exact_box_coordinates(frozen_image):
    snapshot, storage, image = frozen_image
    snapshot["manifest"]["task_type"] = "detect"
    image["annotations"] = [{"class_id": "class-part", "type": "bbox",
                             "bbox": {"x": 1.25, "y": 2.5, "width": 11.5, "height": 8.75}}]
    _, document, _ = labelme_document(snapshot, storage)
    assert document["shapes"][0]["shape_type"] == "rectangle"
    assert document["shapes"][0]["points"] == [[1.25, 2.5], [12.75, 11.25]]


@pytest.mark.parametrize("task_type", ["obb", "classify"])
def test_labelme_rejects_unsupported_tasks(frozen_image, task_type):
    snapshot, storage, _ = frozen_image
    snapshot["manifest"]["task_type"] = task_type
    with pytest.raises(ValidationError) as error:
        export_snapshot(snapshot, storage, "labelme")
    assert error.value.error_code == "workflow_export_task_unsupported"


@pytest.mark.parametrize("kind", ["empty", "dimensions"])
def test_labelme_rejects_invalid_mask_geometry(frozen_image, kind):
    snapshot, storage, image = frozen_image
    mask = np.zeros((20, 24) if kind == "empty" else (10, 12), dtype=bool)
    if kind == "dimensions":
        mask[1:4, 1:4] = True
    image["annotations"] = [{"class_id": "class-part", "type": "polygon",
                             "mask_ref": MaskService(storage).write(mask).mask_ref}]
    with pytest.raises(ValidationError) as error:
        export_snapshot(snapshot, storage, "labelme")
    assert error.value.error_code == ("workflow_empty_object_mask" if kind == "empty" else "workflow_mask_dimensions_mismatch")


def test_unknown_export_format_is_rejected(frozen_image):
    snapshot, storage, _ = frozen_image
    with pytest.raises(ValidationError) as error:
        export_snapshot(snapshot, storage, "unknown")
    assert error.value.error_code == "workflow_export_format_unsupported"
