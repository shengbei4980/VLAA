import base64
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.core.errors import ValidationError
from app.core.models import Annotation, AnnotationReviewReceipt, ImageItem
from app.core.schemas import DatasetCreate
from app.core.task_types import TaskType
from app.dataset.annotation.masks import MaskService, encode_mask, polygon_mask
from app.dataset.annotation.review import confirm_image, require_confirmed, review_detail
from app.dataset.exchange.labelme import import_labelme, preview_labelme
from app.dataset.snapshots import create_snapshot
from app.workflow.exports import export_snapshot
from test_labelme import archive, store


def png_data(array, *, format_name="PNG"):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format=format_name)
    return base64.b64encode(output.getvalue()).decode("ascii")


def native_shape(mask, *, group_id=1):
    ys, xs = np.nonzero(mask)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    return {"label": "part", "shape_type": "mask", "group_id": group_id,
            "points": [[left, top], [right, bottom]],
            "mask": png_data(mask[top:bottom + 1, left:right + 1]), "starter_rle": encode_mask(mask)}


def object_mask():
    mask = np.zeros((80, 50), dtype=np.bool_)
    mask[2:20, 3:25] = True
    mask[7:14, 9:18] = False
    mask[79, 49] = True
    return mask


def native_archive(shape, **metadata):
    return archive({"train/tower.json": {"imagePath": "tower.png", "shapes": [shape], **metadata}})


def add_members(payload, members):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist():
            if name not in members:
                target.writestr(name, source.read(name))
        for name, content in members.items():
            target.writestr(name, content)
    return output.getvalue()


@pytest.mark.parametrize("edit_mask", [False, True])
def test_workflow_export_round_trip_keeps_masks_regions_rules_and_requires_new_review(store, edit_mask):
    database, storage = store
    mask = object_mask()
    roi = np.ones_like(mask)
    roi[:2] = False
    ignore = np.zeros_like(mask)
    ignore[30:34, 20:24] = True
    documents = {
        "train/tower.json": {"imagePath": "tower.png", "source_group_id": "tower-scene",
            "starter_rules": {"partial": True}, "starter_roi": encode_mask(roi),
            "starter_ignore": encode_mask(ignore), "shapes": [native_shape(mask)]},
        "train/background.json": {"imagePath": "background.png", "shapes": []},
    }
    with database.session_factory() as session:
        initial = import_labelme(session, storage, archive(documents),
                                 DatasetCreate(name="Source", task_type=TaskType.SEGMENT), {"part": "part"})
        source_images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == initial["dataset_id"])))
        for image in source_images:
            detail = review_detail(session, storage, image.id)
            confirm_image(session, storage, image.id, expected_revision=detail["revision"],
                          expected_fingerprint=detail["fingerprint"], confirmed_empty=not detail["annotations"])
        snapshot = create_snapshot(session, storage, initial["dataset_id"])
        session.commit()
        result = export_snapshot(snapshot, storage, "labelme")
        payload = storage.export_path(result["file_name"]).read_bytes()
        expected_mask = mask
        if edit_mask:
            expected_mask = np.zeros_like(mask)
            expected_mask[40:53, 10:17] = True
            expected_mask[46:48, 12:15] = False
            replacement = native_shape(expected_mask)
            with zipfile.ZipFile(io.BytesIO(payload)) as exported:
                document_name = next(name for name in exported.namelist()
                    if name.startswith("images/") and name.endswith(".json")
                    and json.loads(exported.read(name))["shapes"])
                document = json.loads(exported.read(document_name))
                shape = document["shapes"][0]
                shape.update(mask=replacement["mask"], points=replacement["points"])
                assert shape["starter_rle"] == encode_mask(mask)
                assert shape["starter_annotation"]["mask_ref"]
                sidecars = {name: exported.read(name) for name in exported.namelist() if name.startswith("masks/")}
            payload = add_members(payload, {document_name: json.dumps(document)})
            with zipfile.ZipFile(io.BytesIO(payload)) as edited:
                assert {name: edited.read(name) for name in sidecars} == sidecars
        preview = preview_labelme(payload)
        assert preview["image_count"] == 2 and preview["shape_count"] == 1
        assert preview["issues"] == []
        assert "mask" in preview["capabilities"]["shapes"]
        restored = import_labelme(session, storage, payload,
                                  DatasetCreate(name="Restored", task_type=TaskType.SEGMENT), {"part": "part"})
        images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == restored["dataset_id"])))
        positive = next(image for image in images if image.status == "annotated")
        background = next(image for image in images if image.status == "unannotated")
        annotation = session.scalar(select(Annotation).where(Annotation.image_id == positive.id))
        masks = MaskService(storage)
        assert np.array_equal(masks.decode(annotation.mask_ref), expected_mask)
        assert np.array_equal(masks.decode(positive.roi_mask_ref), roi)
        assert np.array_equal(masks.decode(positive.ignore_mask_ref), ignore)
        assert positive.rules_json == {"partial": True}
        assert positive.source_group_id == "tower-scene" and positive.split == "train"
        assert positive.review_status == background.review_status == "pending"
        assert positive.review_revision is background.review_revision is None
        assert positive.review_fingerprint is background.review_fingerprint is None
        assert annotation.provenance_json["source_metadata"]["starter_review_receipt"]["id"]
        assert annotation.provenance_json["shapes"][0]["starter_annotation"]["mask_ref"]
        assert annotation.provenance_json["shapes"][0]["starter_rle"] == encode_mask(mask)
        assert session.scalar(select(func.count()).select_from(AnnotationReviewReceipt).where(
            AnnotationReviewReceipt.image_id.in_([image.id for image in images]))) == 0
        with pytest.raises(ValidationError, match="confirmation"):
            require_confirmed(session, storage, restored["dataset_id"])


def test_native_masks_union_with_legacy_shapes_and_keep_single_pixels(store):
    database, storage = store
    mask = object_mask()
    single_pixel = np.zeros_like(mask)
    single_pixel[0, 0] = True
    rectangle = {"label": "part", "shape_type": "rectangle", "group_id": 1,
                 "points": [[30, 30], [40, 40]]}
    payload = archive({"test/tower.json": {"imagePath": "tower.png", "shapes": [
        native_shape(mask), native_shape(single_pixel), rectangle]}})
    with database.session_factory() as session:
        result = import_labelme(session, storage, payload,
                                DatasetCreate(name="Union", task_type=TaskType.SEGMENT), {"part": "part"})
        assert result["imported_annotations"] == 1
        annotation = session.scalar(select(Annotation).where(Annotation.dataset_id == result["dataset_id"]))
        expected = mask | single_pixel | polygon_mask([(30, 30), (40, 30), (40, 40), (30, 40)], 50, 80)
        assert np.array_equal(MaskService(storage).decode(annotation.mask_ref), expected)


def test_detect_extracts_native_mask_inclusive_pixel_bbox(store):
    database, storage = store
    with database.session_factory() as session:
        result = import_labelme(session, storage, native_archive(native_shape(object_mask())),
                                DatasetCreate(name="Boxes", task_type=TaskType.DETECT), {"part": "part"})
        annotation = session.scalar(select(Annotation).where(Annotation.dataset_id == result["dataset_id"]))
        assert (annotation.x, annotation.y, annotation.width, annotation.height) == (3, 2, 47, 78)


@pytest.mark.parametrize("points", [
    [[0.5, 0], [49, 79]], [[False, 0], [49, 79]], [["0", 0], [49, 79]],
    [[float("nan"), 0], [49, 79]], [[float("inf"), 0], [49, 79]],
    [[10**400, 0], [49, 79]], [[-1, 0], [49, 79]], [[0, 0], [50, 79]],
    [[49, 79], [0, 0]], [[0, 0]], [[0, 0, 0], [49, 79]],
])
def test_native_mask_rejects_invalid_inclusive_bounds(points):
    shape = native_shape(object_mask())
    shape["points"] = points
    assert preview_labelme(native_archive(shape))["issues"][0]["code"] == "labelme_mask_bounds_invalid"


@pytest.mark.parametrize("data,code", [
    ("!not-base64!", "labelme_mask_data_invalid"),
    (None, "labelme_mask_data_invalid"),
    (base64.b64encode(b"not a PNG").decode(), "labelme_mask_png_invalid"),
    (png_data(np.zeros((78, 47), dtype=np.uint8), format_name="JPEG"), "labelme_mask_png_invalid"),
    (png_data(np.zeros((78, 47, 3), dtype=np.uint8)), "labelme_mask_png_invalid"),
    (png_data(np.ones((2, 2), dtype=np.bool_)), "labelme_mask_dimensions_mismatch"),
    (png_data(np.full((78, 47), 2, dtype=np.uint8)), "labelme_mask_values_invalid"),
    (png_data(np.zeros((78, 47), dtype=np.bool_)), "labelme_mask_empty"),
])
def test_native_mask_rejects_invalid_png_data(data, code):
    shape = native_shape(object_mask())
    shape["mask"] = data
    assert preview_labelme(native_archive(shape))["issues"][0]["code"] == code


def test_native_mask_limits_encoded_data_and_checks_source_rle_dimensions(monkeypatch):
    shape = native_shape(object_mask())
    shape["starter_rle"] = encode_mask(np.ones((80, 50), dtype=np.bool_))
    assert preview_labelme(native_archive(shape))["issues"] == []
    shape["starter_rle"] = encode_mask(np.ones((2, 2), dtype=np.bool_))
    assert preview_labelme(native_archive(shape))["issues"][0]["code"] == "labelme_mask_rle_dimensions_mismatch"
    monkeypatch.setattr("app.dataset.exchange.labelme._MAX_MASK_BYTES", 8)
    assert preview_labelme(native_archive(shape))["issues"][0]["code"] == "labelme_mask_data_invalid"


@pytest.mark.parametrize("sidecar", [b'{"size":[80,50],"counts":[4000]}', b'{"size":[80,50],"counts":[1]}'])
def test_only_valid_canonical_mask_sidecars_are_skipped(sidecar):
    name = "masks/" + hashlib.sha256(sidecar).hexdigest() + ".json"
    payload = add_members(native_archive(native_shape(object_mask())), {name: sidecar})
    if b"[4000]" in sidecar:
        assert preview_labelme(payload)["issues"] == []
    else:
        with pytest.raises(ValidationError, match="declared image"):
            preview_labelme(payload)


def test_corrupt_or_unknown_json_is_not_silently_ignored():
    payload = native_archive(native_shape(object_mask()))
    with pytest.raises(ValidationError) as exc:
        preview_labelme(add_members(payload, {"masks/" + "0" * 64 + ".json": b"{}"}))
    assert exc.value.error_code == "labelme_sidecar_integrity"
    with pytest.raises(ValidationError) as exc:
        preview_labelme(add_members(payload, {"masks/unrecognized.json": b"{}"}))
    assert exc.value.error_code == "labelme_shapes_missing"
    with pytest.raises(ValidationError) as exc:
        preview_labelme(add_members(payload, {"workflow_manifest.json": b'{"images":[]}'}))
    assert exc.value.error_code == "labelme_manifest_invalid"


def test_mismatched_roi_and_invalid_rules_fail_before_import(store):
    database, storage = store
    shape = native_shape(object_mask())
    with pytest.raises(ValidationError) as exc:
        preview_labelme(native_archive(shape, starter_roi=encode_mask(np.ones((2, 2), dtype=np.bool_))))
    assert exc.value.error_code == "region_dimensions_mismatch"
    with pytest.raises(ValidationError) as exc:
        preview_labelme(native_archive(shape, starter_rules=[]))
    assert exc.value.error_code == "labelme_rules_invalid"
    assert not storage.datasets_dir.exists()
