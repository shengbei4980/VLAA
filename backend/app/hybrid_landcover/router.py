from fastapi import APIRouter, Depends, File, Form, Header, UploadFile

from app.api.dependencies import get_settings, get_storage
from app.core.config import Settings
from app.core.storage import Storage

from . import service

router = APIRouter(prefix="/hybrid", tags=["hybrid land cover"])


@router.get("/status")
def status(settings: Settings = Depends(get_settings)):
    return service.status(settings)


@router.post("/run")
def run(
    mode: str = Form(...), rgb: UploadFile = File(...), raster: UploadFile = File(...),
    roi: UploadFile | None = File(None), x: int = Form(0), y: int = Form(0),
    gpt_draft: UploadFile | None = File(None), sam_draft: UploadFile | None = File(None),
    gpt_endpoint: str = Form(""), deepseek_endpoint: str = Form(""), allow_remote: bool = Form(False),
    run_header: str = Header("", alias="X-VLAA-Hybrid-Run"),
    settings: Settings = Depends(get_settings), storage: Storage = Depends(get_storage),
):
    return service.run(settings, storage, mode=mode, rgb=rgb, raster=raster, roi=roi, x=x, y=y,
                       gpt_draft=gpt_draft, sam_draft=sam_draft, gpt_endpoint=gpt_endpoint,
                       deepseek_endpoint=deepseek_endpoint, approved=allow_remote and run_header == "1")
