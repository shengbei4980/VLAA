"""Deletion keeps class indices dense regardless of primary-key update ordering."""
import pytest
from sqlalchemy import select

from app.core.database import Base, Database
from app.core.errors import ConflictError
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem
from app.dataset.service import delete_class


@pytest.fixture
def database(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'class-delete.db'}")
    Base.metadata.create_all(database.engine)
    with database.session_factory() as session:
        session.add(Dataset(id="dataset", name="Unlabeled buildings", task_type="detect", class_count=4))
        session.add(Dataset(id="other", name="Other dataset", task_type="detect", class_count=1))
        session.flush()
        # Reverse IDs make SQLAlchemy's batched primary-key ordering collide with old indices.
        for index, class_id in enumerate(["class-d", "class-c", "class-b", "class-a"]):
            session.add(ClassLabel(id=class_id, dataset_id="dataset", class_index=index, name=f"Part {index}"))
        session.add(ClassLabel(id="other-class", dataset_id="other", class_index=0, name="Road"))
        for dataset_id in ("dataset", "other"):
            session.add(ImageItem(id=f"{dataset_id}-image", dataset_id=dataset_id, file_name="image.png",
                storage_name="image.png", width=100, height=100, split="train", annotation_revision=3,
                review_status="confirmed", review_revision=3, review_fingerprint="confirmed-fingerprint"))
        session.commit()
    try:
        yield database
    finally:
        database.dispose()


@pytest.mark.parametrize("class_id", ["class-d", "class-c", "class-a"], ids=["first", "middle", "last"])
def test_delete_unused_class_reindexes_and_updates_only_its_dataset(database, class_id):
    original_ids = ["class-d", "class-c", "class-b", "class-a"]
    with database.session_factory() as session:
        delete_class(session, "dataset", class_id)
    with database.session_factory() as session:
        classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == "dataset").order_by(ClassLabel.class_index)))
        assert [item.id for item in classes] == [item for item in original_ids if item != class_id]
        assert [item.class_index for item in classes] == [0, 1, 2]
        assert session.get(Dataset, "dataset").class_count == 3
        image = session.get(ImageItem, "dataset-image")
        assert image.review_status == "pending" and image.annotation_revision == 4
        assert image.review_revision is None and image.review_fingerprint is None
        assert session.get(ClassLabel, "other-class").class_index == 0
        assert session.get(Dataset, "other").class_count == 1
        assert session.get(ImageItem, "other-image").review_status == "confirmed"


def test_delete_rejects_a_class_referenced_by_annotations_without_changing_any_state(database):
    with database.session_factory() as session:
        session.add(Annotation(id="annotation", image_id="dataset-image", dataset_id="dataset", class_id="class-d",
            type="bbox", x=1, y=1, width=20, height=20))
        session.commit()
        with pytest.raises(ConflictError) as error:
            delete_class(session, "dataset", "class-d")
        assert error.value.error_code == "class_in_use"
    with database.session_factory() as session:
        classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == "dataset").order_by(ClassLabel.class_index)))
        assert [item.class_index for item in classes] == [0, 1, 2, 3]
        assert session.get(Dataset, "dataset").class_count == 4
        assert session.get(ImageItem, "dataset-image").review_status == "confirmed"
        assert session.get(Annotation, "annotation").class_id == "class-d"
