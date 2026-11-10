from pathlib import Path

import pytest

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.storage_settings.bootstrap import read_config, write_config
from app.storage_settings.registry import DestinationRegistry
from app.workflow.projects import ProjectRegistry, project_export_locations, publish_run_export, resolve_run_outputs


def test_projects_are_persistent_reusable_and_keep_existing_storage_settings(tmp_path):
    workspace = tmp_path / "workbench"
    output = tmp_path / "塔体 项目"
    write_config(workspace, {"version": 1, "destinations": [], "data_dir": "existing", "manual_exports": {"keep": True}})
    registry = ProjectRegistry(workspace)
    project = registry.register("塔体", str(output))
    assert output.is_dir()
    assert project["root_path"] == str(output)
    assert project["destination_id"]
    assert registry.register("另一个名称", str(output / ".")) == project
    assert ProjectRegistry(workspace).get(project["id"]) == project
    assert ProjectRegistry(workspace).list() == [project]
    config = read_config(workspace)
    assert len(config["destinations"]) == 1
    assert config["data_dir"] == "existing" and config["manual_exports"] == {"keep": True}


def test_separate_projects_and_runs_get_distinct_frozen_subdirectories(tmp_path):
    workspace = tmp_path / "workbench"
    registry = ProjectRegistry(workspace)
    tower = registry.register("塔体", str(tmp_path / "tower"))
    satellite = registry.register("卫星", str(tmp_path / "satellite"))
    first = resolve_run_outputs(workspace, {"models": False, "include_images": False}, tower["id"], run_id="first")
    second = resolve_run_outputs(workspace, {}, tower["id"], run_id="second")
    other = resolve_run_outputs(workspace, {}, satellite["id"], run_id="first")
    assert first["project"] == tower
    assert first["models"]["enabled"] is False
    assert first["include_images"] is False
    assert first["json_path"] == str(tmp_path / "tower" / "workflow-first" / "annotations" / "json")
    assert first["images_path"] == str(tmp_path / "tower" / "workflow-first" / "annotations" / "images")
    assert first["annotations"]["run_path"] != second["annotations"]["run_path"]
    assert first["annotations"]["run_path"] != other["annotations"]["run_path"]
    assert len(registry.list()) == 2


def test_preview_resolves_project_paths_without_creating_run_folders(tmp_path):
    workspace = tmp_path / "workbench"
    root = tmp_path / "project"
    project = ProjectRegistry(workspace).register("Example", str(root))
    output = resolve_run_outputs(workspace, None, project["id"])
    assert project_export_locations(output) == {
        "json_path": str(root / "workflow-<run-id>" / "annotations" / "json"),
        "images_path": str(root / "workflow-<run-id>" / "annotations" / "images"),
    }
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("name,path_kind", [("", "valid"), (" ", "valid"), ("a" * 256, "valid"), ("valid", "relative"), ("valid", "file")])
def test_project_rejects_invalid_names_and_roots(tmp_path, name, path_kind):
    target = tmp_path / "output"
    if path_kind == "relative":
        target = Path("relative")
    if path_kind == "file":
        target.write_text("keep", encoding="utf-8")
    with pytest.raises(ValidationError):
        ProjectRegistry(tmp_path / "workspace").register(name, str(target))


def test_unknown_project_is_rejected(tmp_path):
    with pytest.raises(NotFoundError):
        ProjectRegistry(tmp_path).get("unknown")


def test_registered_root_drift_and_registry_retargeting_are_rejected(tmp_path, monkeypatch):
    workspace = tmp_path / "workbench"
    root = tmp_path / "project"
    registry = ProjectRegistry(workspace)
    project = registry.register("Example", str(root))
    original_resolve = Path.resolve
    replacement = tmp_path / "replacement"

    def changed_resolve(path, *args, **kwargs):
        return replacement if path == root else original_resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "resolve", changed_resolve)
        with pytest.raises(ValidationError) as error:
            registry.get(project["id"])
        assert error.value.error_code == "storage_destination_changed"
    config = read_config(workspace)
    config["destinations"][0]["path"] = str(replacement)
    write_config(workspace, config)
    with pytest.raises(ConflictError) as error:
        registry.get(project["id"])
    assert error.value.error_code == "workflow_project_changed"


def test_legacy_runs_keep_registered_destinations_and_global_exports(tmp_path):
    workspace = tmp_path / "workbench"
    registry = DestinationRegistry(workspace)
    destination = registry.register("legacy", str(tmp_path / "legacy"))
    options = {"annotations_destination_id": destination["id"], "models": False}
    assert resolve_run_outputs(workspace, options, run_id="old") == registry.outputs(options, run_id="old")
    assert project_export_locations(resolve_run_outputs(workspace, options, run_id="old")) == {}
    source = tmp_path / "export.json"
    source.write_bytes(b"draft")
    assert publish_run_export(workspace, source, "json", {}) == source
    assert publish_run_export(workspace, source, "package", {}) == source


def test_project_publishes_json_images_and_package_and_reuses_frozen_root(tmp_path):
    workspace = tmp_path / "workbench"
    root = tmp_path / "project"
    project = ProjectRegistry(workspace).register("Example", str(root))
    frozen = resolve_run_outputs(workspace, {}, project["id"], run_id="one")
    config = read_config(workspace)
    config["destinations"][0]["path"] = str(tmp_path / "later-choice")
    config["workflow_projects"][0]["root_path"] = str(tmp_path / "later-choice")
    config["manual_exports"] = {"json_destination_id": "removed-global-folder", "images_destination_id": "removed-global-folder"}
    write_config(workspace, config)
    for kind, name in (("json", "annotation.json"), ("images", "preview.png"), ("package", "package.zip")):
        source = tmp_path / name
        source.write_bytes((kind + " content").encode())
        result = publish_run_export(workspace, source, kind, frozen)
        directory = root / "workflow-one" / "annotations"
        if kind != "package":
            directory /= kind
        assert result == directory / name
        assert result.read_bytes() == source.read_bytes()
        assert publish_run_export(workspace, source, kind, frozen) == result
    assert not (tmp_path / "later-choice").exists()


def test_project_export_refuses_root_or_run_subdirectory_drift(tmp_path, monkeypatch):
    workspace = tmp_path / "workbench"
    root = tmp_path / "project"
    project = ProjectRegistry(workspace).register("Example", str(root))
    frozen = resolve_run_outputs(workspace, {}, project["id"], run_id="one")
    source = tmp_path / "annotation.json"
    source.write_bytes(b"export")
    original_resolve = Path.resolve
    run_path = Path(frozen["annotations"]["run_path"])

    for changed_path in (root, run_path):
        def changed_resolve(path, *args, **kwargs):
            return tmp_path / "elsewhere" if path == changed_path else original_resolve(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "resolve", changed_resolve)
            with pytest.raises(ConflictError):
                publish_run_export(workspace, source, "json", frozen)
    assert not (tmp_path / "elsewhere").exists()


def test_project_export_refuses_a_path_outside_its_frozen_root(tmp_path):
    workspace = tmp_path / "workbench"
    project = ProjectRegistry(workspace).register("Example", str(tmp_path / "project"))
    frozen = resolve_run_outputs(workspace, {}, project["id"], run_id="one")
    frozen["annotations"]["run_path"] = str(tmp_path / "outside" / "annotations")
    source = tmp_path / "annotation.json"
    source.write_bytes(b"export")
    with pytest.raises(ConflictError):
        publish_run_export(workspace, source, "json", frozen)
    assert not (tmp_path / "outside").exists()
