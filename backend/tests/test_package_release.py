from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "package_release.py"
SPEC = importlib.util.spec_from_file_location("package_release", SCRIPT)
package_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_release)


def write_source(root: Path, relative: str, content: str = "source") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_source_archive_filters_runtime_files_and_matches_checksums(tmp_path, monkeypatch):
    root = tmp_path / "project"
    included = {
        "README.md", "run.py", "install.cmd", "start.cmd", "LICENSE", "NOTICE", ".gitattributes",
        "backend/app/workflow/new_feature.py", "backend/.env.example",
        "backend/alembic/versions/0021_new.py", "frontend/src/NewView.tsx",
        "frontend/package-lock.json", "scripts/setup.py", ".github/workflows/ci.yml",
        "docs/INSTALL.md", "docs/provenance/UPSTREAM_AUTHORIZATION.md",
    }
    excluded = {
        "data/private.json", "runs/results.csv", "Ultralytics/settings.json", ".git/config",
        ".venv/pyvenv.cfg", "frontend/node_modules/react/package.json",
        "frontend/dist/index.html", "backend/.env", "backend/.env.production",
        "backend/cache/private.json", "backend/__pycache__/module.pyc",
        "backend/starter.db", "backend/starter.db-wal", "backend/runtime.log",
        "backend/runtime.log.1", "frontend/.DS_Store", "backend/private.storage.tmp",
        "backend/weight.pt", "backend/weight.onnx", "docs/assets/hero.png",
        "backend/service.key", "backend/service.pem", "backend/certificate.pfx",
        "docs/sample.svg", "docs/plans/internal.md", "release/old-manifest.json",
        "private.txt", ".ywa-storage.json",
    }
    for name in included | excluded:
        write_source(root, name)

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(package_release.subprocess, "run", missing_git)
    archive_path, manifest_path, checksums_path = package_release.build_release(root, tmp_path / "output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {entry["path"] for entry in manifest["files"]} == included
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {f"VLAA-v0.2.0/{name}" for name in included}
        for entry in manifest["files"]:
            content = archive.read(f"VLAA-v0.2.0/{entry['path']}")
            assert hashlib.sha256(content).hexdigest() == entry["sha256"]
            assert len(content) == entry["size"]
            assert not Path(entry["path"]).is_absolute()
            assert ".." not in Path(entry["path"]).parts
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        checksum, name = line.split("  ", maxsplit=1)
        assert hashlib.sha256((checksums_path.parent / name).read_bytes()).hexdigest() == checksum


def test_git_inventory_includes_untracked_source_and_excludes_ignored_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    tracked = write_source(root, "backend/app/existing.py")
    untracked = write_source(root, "frontend/src/new.tsx")
    write_source(root, "Ultralytics/settings.json")

    def git_inventory(command, **kwargs):
        assert command == ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
        return subprocess.CompletedProcess(command, 0, b"backend/app/existing.py\0frontend/src/new.tsx\0Ultralytics/settings.json\0")

    monkeypatch.setattr(package_release.subprocess, "run", git_inventory)
    assert package_release.collect_source_files(root, tmp_path / "output") == [tracked, untracked]


def test_missing_git_uses_filesystem_whitelist(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / ".git").mkdir(parents=True)
    source = write_source(root, "backend/app/new.py")

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(package_release.subprocess, "run", missing_git)
    assert package_release.collect_source_files(root, tmp_path / "output") == [source]


@pytest.mark.parametrize("name", ["../private.txt", "/etc/passwd", "backend/../secret.py", "C:/secret.py", "backend\\secret.py", "backend//x.py", "backend/.. /x.py"])
def test_unsafe_git_paths_are_rejected(tmp_path, monkeypatch, name):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(package_release.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, name.encode() + b"\0"))
    with pytest.raises(ValueError, match="Unsafe source path"):
        package_release.collect_source_files(tmp_path, tmp_path / "release")


@pytest.mark.parametrize("directory", [False, True])
def test_source_symlinks_are_rejected(tmp_path, directory):
    root = tmp_path / "project"
    (root / "backend").mkdir(parents=True)
    target = tmp_path / "private"
    if directory:
        target.mkdir()
        write_source(target, "private.py")
    else:
        target.write_text("private", encoding="utf-8")
    try:
        (root / "backend" / "linked.py").symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("Creating symlinks requires additional privileges on this Windows host.")
    with pytest.raises(ValueError, match="Linked source"):
        package_release.collect_source_files(root, tmp_path / "output")


def test_existing_outputs_require_force_and_preserve_existing_archive(tmp_path):
    root = tmp_path / "project"
    source = write_source(root, "run.py", "first")
    output = tmp_path / "output"
    archive_path, _, _ = package_release.build_release(root, output)
    original = archive_path.read_bytes()
    source.write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        package_release.build_release(root, output)
    assert archive_path.read_bytes() == original
    package_release.build_release(root, output, force=True)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("VLAA-v0.2.0/run.py") == b"second"


def test_custom_output_inside_source_is_not_packaged(tmp_path):
    write_source(tmp_path, "scripts/package_release.py")
    write_source(tmp_path, "scripts/packages/old-manifest.json")
    files = package_release.collect_source_files(tmp_path, tmp_path / "scripts" / "packages")
    assert [path.relative_to(tmp_path).as_posix() for path in files] == ["scripts/package_release.py"]


def test_failed_output_copy_removes_partial_artifacts(tmp_path, monkeypatch):
    write_source(tmp_path / "project", "run.py")
    output = tmp_path / "output"

    def fail_copy(*args, **kwargs):
        raise OSError("simulated output failure")

    monkeypatch.setattr(package_release.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="simulated output failure"):
        package_release.build_release(tmp_path / "project", output)
    assert list(output.iterdir()) == []
