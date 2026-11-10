"""Small, credential-free startup pointer outside the managed data directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading

CONFIG_NAME = ".ywa-storage.json"
CONFIG_LOCK = threading.RLock()


def read_config(project_root: Path) -> dict:
    path = Path(project_root) / CONFIG_NAME
    if not path.exists():
        return {"version": 1, "destinations": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("The project storage configuration is invalid.")
    return value


def write_config(project_root: Path, config: dict) -> None:
    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / CONFIG_NAME
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".storage.tmp", mode="w", encoding="utf-8", delete=False) as output:
            temporary = Path(output.name)
            json.dump(config, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def startup_paths(project_root: Path, environ=None) -> tuple[Path, str]:
    environment = os.environ if environ is None else environ
    config = {} if environment.get("YWA_DATA_DIR") else read_config(project_root)
    raw = environment.get("YWA_DATA_DIR") or config.get("data_dir") or str(Path(project_root) / "data")
    data_dir = Path(raw).expanduser()
    if not data_dir.is_absolute():
        data_dir = Path(project_root) / data_dir
    data_dir = data_dir.resolve()
    database_url = environment.get("YWA_DATABASE_URL")
    if not database_url and not environment.get("YWA_DATA_DIR"):
        database_url = config.get("database_url")
    return data_dir, database_url or f"sqlite:///{data_dir / 'yolowebagent-starter.db'}"
