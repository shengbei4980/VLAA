"""Build a source-only VLAA installation archive and checksum manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.2.0"
PACKAGE_NAME = f"VLAA-v{VERSION}"
SOURCE_DIRECTORIES = {"backend", "frontend", "scripts", ".github", "docs"}
ROOT_FILES = {
    ".gitignore", ".gitattributes", "AGENTS.md", "README.md", "README.en.md", "本地使用说明.md",
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "CONTRIBUTING.md",
    "phase1_scope.md", "source_snapshot.md", "migration_matrix.md", "run.py",
    "install.cmd", "start.cmd", "start-backend.cmd", "start-frontend.cmd",
    "run-all.sh", "run-backend.sh", "run-frontend.sh",
}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "dist", "release", "cache", ".cache",
    "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".vite",
    "data", "runs", "ultralytics", "htmlcov", ".idea", ".vscode",
}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".tsbuildinfo", ".log", ".db", ".db-wal", ".db-shm",
    ".sqlite", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm", ".pt", ".pth",
    ".onnx", ".engine", ".ckpt", ".safetensors", ".bin", ".whl",
    ".pem", ".key", ".p12", ".pfx",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico",
    ".tif", ".tiff", ".ttf", ".ttc", ".otf", ".woff", ".woff2",
    ".mp4", ".avi", ".mov", ".zip", ".7z", ".tar", ".gz", ".tmp", ".bak",
}


def safe_relative_path(name: str) -> PurePosixPath:
    """Use portable archive names with no Windows or POSIX escape syntax."""
    parts = name.split("/")
    if not name or "\\" in name or any(
        not part or part in {".", ".."} or part.rstrip(" .") != part
        or any(character in part for character in '\x00<>:"|?*')
        for part in parts
    ):
        raise ValueError(f"Unsafe source path: {name!r}")
    return PurePosixPath(name)


def is_source_path(path: PurePosixPath) -> bool:
    parts = path.parts
    lower_parts = tuple(part.lower() for part in parts)
    name = lower_parts[-1]
    if any(part in EXCLUDED_DIRECTORIES for part in lower_parts):
        return False
    if lower_parts[:2] == ("docs", "plans"):
        return False
    if name.startswith(".env") and name != ".env.example":
        return False
    if name.startswith(".ywa-storage") or name in {".coverage", ".ds_store", "sha256sums.txt"}:
        return False
    if any(suffix.lower() in EXCLUDED_SUFFIXES for suffix in path.suffixes):
        return False
    return (len(parts) == 1 and path.name in ROOT_FILES) or parts[0] in SOURCE_DIRECTORIES


def filesystem_candidates(root: Path) -> list[str]:
    candidates = [name for name in ROOT_FILES if (root / name).exists() or (root / name).is_symlink()]
    for directory in sorted(SOURCE_DIRECTORIES):
        base = root / directory
        if not base.exists() and not base.is_symlink():
            continue
        if base.is_symlink() or getattr(base, "is_junction", lambda: False)():
            raise ValueError(f"Linked source directory is not allowed: {directory}")
        for current, directories, files in os.walk(base, followlinks=False):
            allowed_directories = []
            for name in directories:
                child = Path(current) / name
                relative = PurePosixPath(child.relative_to(root).as_posix())
                if is_source_path(relative):
                    if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
                        raise ValueError(f"Linked source directory is not allowed: {relative}")
                    allowed_directories.append(name)
            directories[:] = allowed_directories
            candidates.extend((Path(current) / name).relative_to(root).as_posix() for name in files)
    return candidates


def collect_source_files(root: Path, output_dir: Path) -> list[Path]:
    root, output_dir = root.resolve(), output_dir.resolve()
    try:
        if (root / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                check=True, capture_output=True,
            )
            names = result.stdout.decode("utf-8").split("\0")
        else:
            names = filesystem_candidates(root)
    except FileNotFoundError:
        names = filesystem_candidates(root)
    files = []
    for name in sorted(set(filter(None, names))):
        relative = safe_relative_path(name)
        if not is_source_path(relative):
            continue
        source = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                raise ValueError(f"Linked source path is not allowed: {relative}")
        resolved = source.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Source path escapes the repository: {relative}")
        if output_dir != root and resolved.is_relative_to(output_dir):
            continue
        if source.is_file():
            files.append(source)
    if not files:
        raise ValueError("No source files were found in the allowed release paths.")
    return files


def build_release(root: Path, output_dir: Path, *, force: bool = False) -> tuple[Path, Path, Path]:
    root, output_dir = root.resolve(), output_dir.resolve()
    names = (f"{PACKAGE_NAME}-source.zip", f"{PACKAGE_NAME}-manifest.json", "SHA256SUMS.txt")
    destinations = tuple(output_dir / name for name in names)
    if not force and any(path.exists() for path in destinations):
        raise FileExistsError("Release output already exists. Use --force to replace it.")
    files = collect_source_files(root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(prefix=".vlaa-package-", dir=output_dir) as temporary:
        staging = Path(temporary)
        archive_path, manifest_path, checksums_path = (staging / name for name in names)
        entries = []
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source in files:
                content = source.read_bytes()
                relative = source.relative_to(root).as_posix()
                entries.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)})
                archive.writestr(f"{PACKAGE_NAME}/{relative}", content)
        manifest_path.write_text(
            json.dumps({"name": PACKAGE_NAME, "version": VERSION, "files": entries}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksums_path.write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (archive_path, manifest_path)
        ), encoding="utf-8")
        try:
            for name, destination in zip(names, destinations):
                if force:
                    os.replace(staging / name, destination)
                else:
                    with destination.open("xb") as output, (staging / name).open("rb") as source:
                        published.append(destination)
                        shutil.copyfileobj(source, output)
        except BaseException:
            if not force:
                for destination in published:
                    destination.unlink(missing_ok=True)
            raise
    return destinations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--force", action="store_true", help="Replace existing release artifacts.")
    arguments = parser.parse_args(argv)
    try:
        paths = build_release(ROOT, arguments.output_dir, force=arguments.force)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
