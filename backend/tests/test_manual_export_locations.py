from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.error_handlers import register_error_handlers
from app.core.config import Settings
from app.core.errors import ConflictError, ValidationError
from app.storage_settings.bootstrap import read_config, startup_paths, write_config
from app.storage_settings.manual_exports import manual_export_locations, publish_manual_export, save_manual_export_locations
from app.storage_settings.registry import DestinationRegistry
from app.storage_settings.router import router


@pytest.fixture
def workspace(tmp_path):
    project = tmp_path / "项目"
    project.mkdir()
    data = project / "data"
    data.mkdir()
    settings = Settings(project_root=project, data_dir=data, import_root=data / "imports", database_url="sqlite://")
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router, prefix="/api")
    register_error_handlers(app)
    with TestClient(app) as client:
        yield settings, client


def test_default_paths_and_independent_preferences_are_persisted(workspace, tmp_path):
    settings, client = workspace
    default = str((settings.data_dir / "exports").resolve())
    assert client.get("/api/storage/manual-exports").json() == {"json_path": default, "images_path": default}
    database_url = f"sqlite:///{settings.data_dir / 'workspace.db'}"
    write_config(settings.project_root, {"version": 1, "destinations": [], "unrelated": {"keep": True},
                                         "data_dir": str(settings.data_dir), "database_url": database_url})
    registry = DestinationRegistry(settings.project_root)
    workflow_destination = registry.register("Existing workflow", str(tmp_path / "workflow"))
    workflow_options = {"annotations_destination_id": workflow_destination["id"]}
    frozen_outputs = registry.outputs(workflow_options, run_id="existing-run")
    paths = {"json_path": str(tmp_path / "标注 JSON"), "images_path": str(tmp_path / "图像 PNG")}
    response = client.put("/api/storage/manual-exports", json=paths)
    assert response.status_code == 200, response.text
    assert response.json() == paths
    assert client.get("/api/storage/manual-exports").json() == paths
    stored = read_config(settings.project_root)
    assert stored["unrelated"] == {"keep": True}
    assert set(stored["manual_exports"]) == {"json_destination_id", "images_destination_id"}
    assert len(stored["destinations"]) == 3
    assert all(Path(path).is_dir() for path in paths.values())
    assert startup_paths(settings.project_root, environ={}) == (settings.data_dir, database_url)
    assert registry.outputs(workflow_options, run_id="existing-run") == frozen_outputs


def test_export_copies_to_the_matching_directory_and_keeps_managed_file(workspace, tmp_path):
    settings, _ = workspace
    managed = settings.data_dir / "exports"
    managed.mkdir()
    json_file = managed / "review-draft.json"
    png_file = managed / "review-draft.png"
    json_file.write_bytes(b'{"shapes": []}')
    png_file.write_bytes(b"preview bytes")
    assert publish_manual_export(settings.project_root, json_file, "json") == json_file
    save_manual_export_locations(settings, str(tmp_path / "json"), str(tmp_path / "images"))
    json_target = publish_manual_export(settings.project_root, json_file, "json")
    png_target = publish_manual_export(settings.project_root, png_file, "images")
    assert json_target == tmp_path / "json" / json_file.name
    assert png_target == tmp_path / "images" / png_file.name
    assert json_target.read_bytes() == json_file.read_bytes()
    assert png_target.read_bytes() == png_file.read_bytes()
    assert publish_manual_export(settings.project_root, json_file, "json") == json_target
    assert json_file.is_file() and png_file.is_file()
    assert not list(json_target.parent.glob("*.export.tmp"))


def test_same_directory_and_concurrent_retries_preserve_content(workspace, tmp_path):
    settings, _ = workspace
    managed = settings.data_dir / "same.png"
    managed.write_bytes(b"same immutable preview" * 1000)
    target = tmp_path / "same-output"
    save_manual_export_locations(settings, str(target), str(target))
    assert len(read_config(settings.project_root)["destinations"]) == 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: publish_manual_export(settings.project_root, managed, "images"), range(4)))
    assert outputs == [target / managed.name] * 4
    assert outputs[0].read_bytes() == managed.read_bytes()
    assert not list(target.glob("*.export.tmp"))


def test_existing_different_file_is_preserved(workspace, tmp_path):
    settings, _ = workspace
    source = settings.data_dir / "annotation.json"
    source.write_bytes(b"new export")
    target = tmp_path / "output"
    save_manual_export_locations(settings, str(target), str(target))
    original = target / source.name
    original.write_bytes(b"user existing file")
    with pytest.raises(ConflictError) as error:
        publish_manual_export(settings.project_root, source, "json")
    assert error.value.error_code == "manual_export_file_conflict"
    assert original.read_bytes() == b"user existing file"


def test_invalid_new_preferences_leave_previous_pair_in_effect(workspace, tmp_path):
    settings, client = workspace
    paths = {"json_path": str(tmp_path / "json"), "images_path": str(tmp_path / "images")}
    assert client.put("/api/storage/manual-exports", json=paths).status_code == 200
    response = client.put("/api/storage/manual-exports", json={"json_path": str(tmp_path / "new-json"), "images_path": "relative"})
    assert response.status_code == 422
    assert client.get("/api/storage/manual-exports").json() == paths


def test_missing_registered_destination_refuses_export(workspace):
    settings, _ = workspace
    source = settings.data_dir / "draft.json"
    source.write_bytes(b"content")
    write_config(settings.project_root, {"version": 1, "destinations": [], "manual_exports": {"json_destination_id": "removed"}})
    with pytest.raises(ValidationError) as error:
        publish_manual_export(settings.project_root, source, "json")
    assert error.value.error_code == "manual_export_destination_missing"


def test_registered_directory_drift_is_rejected(workspace, tmp_path, monkeypatch):
    settings, _ = workspace
    target = tmp_path / "output"
    save_manual_export_locations(settings, str(target), str(target))
    original_resolver = Path.resolve
    moved = tmp_path / "moved"
    moved.mkdir()

    def resolve(path, *args, **kwargs):
        return moved if path == target else original_resolver(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValidationError) as error:
        manual_export_locations(settings)
    assert error.value.error_code == "storage_destination_changed"
