"""Install the source distribution into its own Python and frontend environments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = (3, 12)
MIN_NODE_MAJOR = 20


class SetupError(RuntimeError):
    pass


def local_python(root: Path) -> Path:
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return root / ".venv" / executable


def probe(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SetupError("无法检查依赖工具，请确认 Python 和 Node.js 已正确安装。") from error
    return result.stdout.strip()


def node_tools() -> tuple[str, list[str]]:
    node = shutil.which("node")
    if node is None:
        raise SetupError("未找到 Node.js，请安装 Node.js 22 LTS，并重新打开安装窗口。")
    version = probe([node, "--version"])
    match = re.fullmatch(r"v?(\d+)\.\d+\.\d+(?:-.*)?", version)
    if match is None or int(match.group(1)) < MIN_NODE_MAJOR:
        raise SetupError("需要 Node.js 20 或更高版本，建议安装 Node.js 22 LTS。")
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        raise SetupError("未找到 npm，请重新安装包含 npm 的 Node.js。")
    npm_path = Path(npm).resolve()
    if npm_path.suffix.lower() in {".cmd", ".bat"}:
        # Run the JavaScript entry directly so project paths never pass through cmd.exe.
        script = npm_path.parent / "node_modules/npm/bin/npm-cli.js"
        if not script.is_file():
            raise SetupError("未找到 npm 的 JavaScript 入口，请重新安装 Node.js 附带的 npm。")
        npm_command = [node, str(script)]
    elif npm_path.suffix.lower() in {".js", ".mjs"}:
        npm_command = [node, str(npm_path)]
    else:
        npm_command = [str(npm_path)]
    npm_version = probe([*npm_command, "--version"])
    print(f"Node.js {version}，npm {npm_version}", flush=True)
    return node, npm_command


def check_environment(root: Path) -> tuple[Path, list[str]]:
    if sys.version_info[:2] != PYTHON_VERSION:
        raise SetupError("需要 Python 3.12。请安装该版本，再双击 install.cmd。")
    print(f"Python {sys.version.split()[0]}", flush=True)
    for relative in ("backend/requirements.txt", "frontend/package-lock.json", "run.py"):
        if not (root / relative).is_file():
            raise SetupError("源码文件不完整，请先将发布 ZIP 完整解压到一个文件夹。")
    _, npm_command = node_tools()
    python = local_python(root)
    environment = root / ".venv"
    if environment.exists():
        if not python.is_file():
            raise SetupError(".venv 不完整，请先将该文件夹重命名，再运行 install.cmd。")
        try:
            version = json.loads(probe([
                str(python), "-c", "import json, sys; print(json.dumps(list(sys.version_info[:2])))",
            ]))
        except (SetupError, json.JSONDecodeError) as error:
            raise SetupError("无法启动或读取本地 .venv，请将该文件夹重命名后重新运行 install.cmd。") from error
        if version != list(PYTHON_VERSION):
            raise SetupError("已有 .venv 不是 Python 3.12 环境，请将其重命名后重新运行 install.cmd。")
    return python, npm_command


def install(root: Path, python: Path, npm_command: list[str]) -> None:
    if not python.is_file():
        print("正在创建本项目的 .venv 环境…", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], cwd=root, check=True)
    if not python.is_file():
        raise SetupError(".venv 创建失败，请检查 Python 的 venv 组件和文件夹写入权限。")
    print("正在安装后端依赖，首次安装需要联网下载…", flush=True)
    subprocess.run([
        str(python), "-m", "pip", "install", "-r", str(root / "backend/requirements.txt"),
    ], cwd=root, check=True)
    print("正在安装前端依赖…", flush=True)
    subprocess.run([*npm_command, "ci", "--prefix", str(root / "frontend")], cwd=root, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装 VLAA 源码版所需的本地依赖。")
    parser.add_argument("--check", action="store_true", help="仅检查环境，保留现有文件和依赖。")
    args = parser.parse_args(argv)
    print("Vision-Language Automated Annotation（VLAA）", flush=True)
    try:
        python, npm_command = check_environment(ROOT)
        if args.check:
            print("基础环境检查通过。", flush=True)
            if python.is_file() and (ROOT / "frontend/node_modules/vite/bin/vite.js").is_file():
                print("已找到项目 .venv 和前端依赖入口。", flush=True)
            else:
                print("项目依赖尚未安装完整，请运行 install.cmd。", flush=True)
            return 0
        install(ROOT, python, npm_command)
    except SetupError as error:
        print(f"安装检查未通过。{error}", file=sys.stderr, flush=True)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"安装中断，某个步骤返回了错误码 {error.returncode}。请查看上方输出，处理网络或依赖问题后重新运行 install.cmd。",
              file=sys.stderr, flush=True)
        return 1
    except OSError as error:
        print(f"无法完成安装。请检查文件夹权限及 Python、Node.js 是否可用。{error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("安装已中止，可以稍后重新运行 install.cmd。", file=sys.stderr, flush=True)
        return 130
    print("安装完成。请双击 start.cmd 启动 VLAA。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
