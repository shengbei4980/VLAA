import base64
import io
import json
from pathlib import PurePosixPath
import zipfile

import pytest
from PIL import Image
from sqlalchemy import func, select

from app.core.database import Base, Database
from app.core.errors import ValidationError
from app.core.models import Annotation, Dataset, ImageItem
from app.core.schemas import DatasetCreate
from app.core.storage import Storage
from app.core.task_types import TaskType
from app.dataset.annotation.masks import MaskService
from app.dataset.annotation.review import require_confirmed
from app.dataset.exchange.labelme import import_labelme, preview_labelme


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (50, 80), "white").save(stream, format="PNG")
    return stream.getvalue()


def archive(documents=None, *, embedded=False):
    documents = documents or {"train/tower.json": {"imagePath": "tower.png", "imageWidth": 50, "imageHeight": 80,
        "source_group_id": "tower-scene", "shapes": [
            {"label": "component", "shape_type": "polygon", "points": [[2, 2], [15, 2], [15, 15], [2, 15]], "group_id": 1},
            {"label": "component", "shape_type": "rectangle", "points": [[30, 30], [45, 45]], "group_id": 1},
        ]}}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name, document in documents.items():
            document = dict(document)
            if embedded:
                document["imageData"] = base64.b64encode(image_bytes()).decode()
            zipped.writestr(name, json.dumps(document))
            if not embedded:
                zipped.writestr(str(PurePosixPath(name).parent / document["imagePath"]), image_bytes())
    return output.getvalue()


@pytest.fixture
def store(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'labelme.db'}")
    Base.metadata.create_all(database.engine)
    yield database, Storage(tmp_path / "data", tmp_path / "imports")
    database.dispose()


def test_preview_is_read_only_and_mapping_is_explicit(store):
    database, storage = store
    result = preview_labelme(archive())
    assert result["image_count"] == 1
    assert result["labels"] == [{"name": "component", "count": 2}]
    assert result["mapping_required"]
    assert result["images"][0]["split"] == "train"
    assert result["issues"] == []
    assert not storage.datasets_dir.exists()
    with database.session_factory() as session:
        with pytest.raises(ValidationError, match="Explicitly map"):
            import_labelme(session, storage, archive(), DatasetCreate(name="Tower", task_type=TaskType.SEGMENT), {})


def test_polygon_group_union_retains_components_and_stays_unreviewed(store):
    database, storage = store
    with database.session_factory() as session:
        result = import_labelme(session, storage, archive(), DatasetCreate(name="Tower", task_type=TaskType.SEGMENT), {"component": "Tower part"})
        assert result["imported_annotations"] == 1
        image = session.scalar(select(ImageItem).where(ImageItem.dataset_id == result["dataset_id"]))
        annotation = session.scalar(select(Annotation).where(Annotation.image_id == image.id))
        mask = MaskService(storage).decode(annotation.mask_ref)
        assert mask[5, 5] and mask[35, 35] and not mask[22, 22]
        assert image.source_group_id == "tower-scene"
        assert image.review_status == "pending"
        assert annotation.source == "imported"
        assert len(list((storage.dataset_dir(result["dataset_id"]) / "imports" / "labelme").glob("*.json"))) == 1
        with pytest.raises(ValidationError, match="confirmation"):
            require_confirmed(session, storage, result["dataset_id"])


def test_embedded_images_and_detect_bbox(store):
    database, storage = store
    with database.session_factory() as session:
        result = import_labelme(session, storage, archive(embedded=True), DatasetCreate(name="Tower", task_type=TaskType.DETECT), {"component": "part"})
        annotation = session.scalar(select(Annotation).where(Annotation.dataset_id == result["dataset_id"]))
        assert annotation.type == "bbox"
        assert annotation.x == 2 and annotation.width == 43


def test_explicit_ignored_unsupported_shape_is_not_imported(store):
    payload = archive({"test/tower.json": {"imagePath": "tower.png", "shapes": [
        {"label": "ignore", "shape_type": "circle", "points": [[10, 10], [15, 15]]},
        {"label": "part", "shape_type": "rectangle", "points": [[2, 2], [20, 20]]}]}})
    assert preview_labelme(payload)["issues"][0]["code"] == "labelme_shape_unsupported"
    database, storage = store
    with database.session_factory() as session:
        result = import_labelme(session, storage, payload, DatasetCreate(name="Parts", task_type=TaskType.DETECT), {"ignore": None, "part": "part"})
        assert result["imported_annotations"] == 1
        assert session.scalar(select(ImageItem).where(ImageItem.dataset_id == result["dataset_id"])).split == "test"


def test_failed_geometry_rolls_back_database_and_managed_images(store):
    database, storage = store
    payload = archive({"tower.json": {"imagePath": "tower.png", "shapes": [
        {"label": "part", "shape_type": "polygon", "points": [[-2, 2], [15, 2], [15, 15]]}]}})
    with database.session_factory() as session:
        with pytest.raises(ValidationError):
            import_labelme(session, storage, payload, DatasetCreate(name="Invalid", task_type=TaskType.SEGMENT), {"part": "part"})
        assert session.scalar(select(func.count()).select_from(Dataset)) == 0
        assert not list(storage.datasets_dir.glob("*/images/*"))


def test_archive_traversal_and_mismatched_dimensions_are_rejected():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr("../tower.json", "{}")
    with pytest.raises(ValidationError):
        preview_labelme(output.getvalue())
    payload = archive({"tower.json": {"imagePath": "tower.png", "imageWidth": 500, "shapes": []}})
    with pytest.raises(ValidationError, match="dimensions"):
        preview_labelme(payload)


def test_source_group_cannot_leak_across_splits(store):
    document = {"imagePath": "tower.png", "source_group_id": "same-scene", "shapes": [
        {"label": "part", "shape_type": "rectangle", "points": [[2, 2], [20, 20]]}]}
    payload = archive({"train/a.json": document, "val/b.json": document})
    database, storage = store
    with database.session_factory() as session:
        with pytest.raises(ValidationError, match="source group"):
            import_labelme(session, storage, payload, DatasetCreate(name="Leak", task_type=TaskType.DETECT), {"part": "part"})


def test_labelme_router_preview_and_import(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.dependencies import get_session, get_storage
    from app.api.error_handlers import register_error_handlers
    from app.dataset.exchange.labelme_router import router

    database, storage = store
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_session] = database.sessions
    app.dependency_overrides[get_storage] = lambda: storage
    register_error_handlers(app)
    with TestClient(app) as client:
        response = client.post("/api/datasets/import/labelme/preview", files={"file": ("labels.zip", archive(), "application/zip")})
        assert response.status_code == 200, response.text
        response = client.post("/api/datasets/import/labelme", files={"file": ("labels.zip", archive(), "application/zip")},
                               data={"name": "Tower", "task_type": "detect", "class_mapping": json.dumps({"component": "part"})})
        assert response.status_code == 201, response.text
        assert response.json()["review_status"] == "pending"
