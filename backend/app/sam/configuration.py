from __future__ import annotations

from hashlib import file_digest
from pathlib import Path

from app.core.config import Settings
from app.core.errors import ValidationError
from app.settings.service import SETTINGS_WRITE_LOCK, SamSettingsService


def freeze_sam_configuration(settings: Settings) -> dict:
    """Describe the selected configuration and local weight bytes without loading a model."""
    with SETTINGS_WRITE_LOCK:
        current = SamSettingsService(settings).get()
        result = {key: getattr(current, key) for key in ("enabled", "model", "device", "img_size", "fallback_mode")}
        result.update(weights_path=None, weights_sha256=None, weights_size_bytes=None, weights_available=False)
        if not current.model:
            return result
        path = Path(current.model).expanduser()
        if not path.is_absolute():
            path = settings.project_root / path
        path = path.resolve()
        result["weights_path"] = str(path)
        if not path.is_file():
            return result
        try:
            with path.open("rb") as source:
                digest = file_digest(source, "sha256").hexdigest()
            size = path.stat().st_size
        except OSError as exc:
            raise ValidationError("sam_weights_unreadable", "The selected SAM weight file cannot be inspected.") from exc
        return {**result, "weights_available": True, "weights_sha256": digest, "weights_size_bytes": size}


def verify_sam_configuration(settings: Settings, expected: dict, *, require_weights: bool = False) -> dict:
    current = freeze_sam_configuration(settings)
    if expected != current:
        raise ValidationError("vision_segmenter_changed", "SAM settings or weight contents differ from the approved run.")
    if require_weights and not current["weights_available"]:
        raise ValidationError("sam_weights_unavailable", "Workflow segmentation requires a selected local SAM weight file.")
    return current
