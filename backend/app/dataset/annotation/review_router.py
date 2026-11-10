from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_storage
from app.core.errors import NotFoundError
from app.core.storage import Storage
from app.dataset.annotation.review import adopt_historical, confirm_image, review_detail, review_summary
from app.dataset.images import get_image


router = APIRouter(prefix="/datasets", tags=["annotation-review"])


class ConfirmImageRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    expected_fingerprint: str = Field(min_length=64, max_length=64)
    confirmed_empty: bool = False
    actor: str = Field(default="local-user", max_length=128)
    reason: str = Field(default="", max_length=10000)


class AdoptItem(BaseModel):
    image_id: str
    expected_revision: int = Field(ge=0)
    expected_fingerprint: str = Field(min_length=64, max_length=64)


class AdoptRequest(BaseModel):
    items: list[AdoptItem] = Field(min_length=1, max_length=10000)
    actor: str = Field(default="local-user", max_length=128)
    reason: str = Field(default="", max_length=10000)


@router.get("/{dataset_id}/review-status")
def get_review_summary(dataset_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> dict:
    return review_summary(session, storage, dataset_id)


@router.get("/{dataset_id}/images/{image_id}/review")
def get_image_review(dataset_id: str, image_id: str, session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> dict:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "The image does not belong to this dataset.")
    return review_detail(session, storage, image.id)


@router.post("/{dataset_id}/images/{image_id}/review/confirm")
def post_image_confirmation(dataset_id: str, image_id: str, payload: ConfirmImageRequest,
                            session: Session = Depends(get_session), storage: Storage = Depends(get_storage)) -> dict:
    image = get_image(session, image_id)
    if image.dataset_id != dataset_id:
        raise NotFoundError("image_not_found", "The image does not belong to this dataset.")
    result = confirm_image(session, storage, image.id, **payload.model_dump())
    session.commit()
    return result


@router.post("/{dataset_id}/reviews/adopt")
def post_adoption(dataset_id: str, payload: AdoptRequest, session: Session = Depends(get_session),
                  storage: Storage = Depends(get_storage)) -> dict:
    results = adopt_historical(session, storage, dataset_id, [item.model_dump() for item in payload.items],
                               actor=payload.actor, reason=payload.reason)
    session.commit()
    return {"adopted": len(results), "items": results}
