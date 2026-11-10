"""Managed boundary for immutable workflow JSON artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from app.core.errors import ValidationError
from app.core.storage import Storage


def canonical_bytes(payload: object) -> bytes:
    try:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise ValidationError("workflow_invalid_json", "Workflow data must be finite JSON.") from exc
    if len(data) > 20 * 1024 * 1024:
        raise ValidationError("workflow_payload_too_large", "Workflow JSON exceeds the 20 MiB artifact limit.")
    return data


def fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


class WorkflowArtifactStore:
    def __init__(self, storage: Storage):
        self.root = (storage.data_dir / "workflows" / "artifacts").resolve()

    def path(self, storage_name: str) -> Path:
        candidate = (self.root / storage_name).resolve()
        if not candidate.is_relative_to(self.root) or candidate.suffix != ".json":
            raise ValidationError("workflow_unsafe_artifact", "Artifact escapes managed workflow storage.")
        return candidate

    def write(self, payload: dict) -> tuple[str, str]:
        data = canonical_bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        destination = self.path(f"{digest}.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish complete bytes atomically, including concurrent identical revisions.
        if not destination.exists():
            temporary_name = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as output:
                    temporary_name = output.name
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, destination)
            finally:
                if temporary_name:
                    Path(temporary_name).unlink(missing_ok=True)
        return destination.name, digest

    def verify(self, storage_name: str, digest: str) -> None:
        path = self.path(storage_name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValidationError("workflow_artifact_corrupt", "A persisted workflow artifact is missing or has changed.")
