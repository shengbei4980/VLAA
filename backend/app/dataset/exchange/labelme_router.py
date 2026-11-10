import json

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_storage
from app.core.errors import ValidationError
from app.core.schemas import DatasetCreate, SplitName
from app.core.storage import Storage
from app.core.task_types import TaskType
from app.dataset.exchange.labelme import import_labelme, preview_labelme

router = APIRouter(prefix="/datasets/import/labelme", tags=["datasets"])
MAX_ZIP_BYTES = 100 * 1024 * 1024


def _mapping(value: str | None):
    if value is None:
        return None
    try:
        result = json.loads(value)
    except ValueError as exc:
        raise ValidationError("labelme_mapping_invalid", "Class mapping must be a JSON object.") from exc
    if not isinstance(result, dict) or any(not isinstance(key, str) or target is not None and not isinstance(target, str)
                                           for key, target in result.items()):
        raise ValidationError("labelme_mapping_invalid", "Map source labels to class names or null.")
    return result


def _read(file: UploadFile):
    payload = file.file.read(MAX_ZIP_BYTES + 1)
    if len(payload) > MAX_ZIP_BYTES:
        raise ValidationError("labelme_archive_too_large", "LabelMe ZIP exceeds the 100 MiB upload limit.")
    return payload


@router.post("/preview")
def preview(file: UploadFile = File(...), task_type: TaskType = Form(default=TaskType.SEGMENT),
            default_split: SplitName = Form(default="train"), class_mapping: str | None = Form(default=None)):
    return preview_labelme(_read(file), task_type=task_type.value, default_split=default_split,
                           class_mapping=_mapping(class_mapping))


@router.post("", status_code=201)
def create(file: UploadFile = File(...), name: str = Form(...), task_type: TaskType = Form(default=TaskType.SEGMENT),
           default_split: SplitName = Form(default="train"), class_mapping: str = Form(...),
           session: Session = Depends(get_session), storage: Storage = Depends(get_storage)):
    return import_labelme(session, storage, _read(file), DatasetCreate(name=name, task_type=task_type),
                          _mapping(class_mapping), default_split)
