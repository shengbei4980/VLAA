from __future__ import annotations

import json
from hashlib import sha256

from pydantic import SecretStr, ValidationError as SchemaError

from app.agent.llm_auth import validate_auth_config
from app.core.config import Settings
from app.core.errors import ValidationError
from app.settings.service import SETTINGS_WRITE_LOCK, SettingsService
from app.vision.schemas import (
    VisionEndpoint, VisionEndpointInternal, VisionEndpointOut,
    VisionRole, VisionSettingsOut, VisionSettingsUpdate,
)

class VisionSettingsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = SettingsService(settings)

    def internal(self, role: VisionRole) -> VisionEndpointInternal:
        if role not in {"proposer", "reviewer"}:
            raise ValidationError("invalid_vision_role", "Unknown vision model role.")
        section = self.settings._read().get("vision", {})
        raw = section.get(role, {}) if isinstance(section, dict) else {}
        if not isinstance(raw, dict):
            raise ValidationError("invalid_vision_settings", "Stored vision settings are invalid.")
        try:
            public = VisionEndpoint.model_validate({key: value for key, value in raw.items() if key != "api_key"})
        except SchemaError as exc:
            raise ValidationError("invalid_vision_settings", "Stored vision endpoint configuration is invalid.") from exc
        key = str(raw.get("api_key", ""))
        fingerprint = public.model_dump()
        fingerprint["credential_hash"] = sha256(key.encode()).hexdigest()
        digest = sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        return VisionEndpointInternal(**public.model_dump(), api_key=SecretStr(key), config_hash=digest)

    def get(self) -> VisionSettingsOut:
        roles = {}
        for role in ("proposer", "reviewer"):
            endpoint = self.internal(role)
            roles[role] = VisionEndpointOut(
                **endpoint.model_dump(exclude={"api_key"}),
                api_key_configured=bool(endpoint.api_key.get_secret_value()),
            )
        return VisionSettingsOut(**roles)

    def update(self, payload: VisionSettingsUpdate) -> VisionSettingsOut:
        with SETTINGS_WRITE_LOCK:
            data = self.settings._read()
            existing = data.get("vision", {})
            section = {}
            for role in ("proposer", "reviewer"):
                endpoint = getattr(payload, role)
                if endpoint.enabled and (not endpoint.api_base or not endpoint.model.strip()):
                    raise ValidationError("vision_endpoint_missing", "Enabled vision roles require an explicit endpoint and model.")
                validate_auth_config(endpoint.auth_scheme, endpoint.auth_header_name)
                prior = existing.get(role, {}) if isinstance(existing, dict) else {}
                prior_key = str(prior.get("api_key", "")) if isinstance(prior, dict) else ""
                key = endpoint.api_key.get_secret_value() if endpoint.api_key is not None else prior_key
                section[role] = {**endpoint.model_dump(exclude={"api_key"}), "api_key": key}
            data["vision"] = section
            self.settings._write(data)
        return self.get()
