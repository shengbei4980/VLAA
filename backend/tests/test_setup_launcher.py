import os
import subprocess
from unittest.mock import Mock

import pytest

from scripts import setup


@pytest.fixture
def source_root(tmp_path):
    root = tmp_path / "VLAA project & images"
    for relative in ("backend/requirements.txt", "frontend/package-lock.json", "run.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return root


def test_check_only_reports_without_creating_or_installing(monkeypatch, source_root, capsys):
    monkeypatch.setattr(setup, "ROOT", source_root)
    monkeypatch.setattr(setup.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(setup, "node_tools", lambda: ("node", ["node", "npm-cli.js"]))
    install = Mock(side_effect=AssertionError("A check must not install anything"))
    monkeypatch.setattr(setup, "install", install)
    before = {path.relative_to(source_root) for path in source_root.rglob("*")}

    assert setup.main(["--check"]) == 0

    install.assert_not_called()
    assert before == {path.relative_to(source_root) for path in source_root.rglob("*")}
    assert "请运行 install.cmd" in capsys.readouterr().out


def test_wrong_python_version_fails_before_installation(monkeypatch, source_root, capsys):
    monkeypatch.setattr(setup, "ROOT", source_root)
    monkeypatch.setattr(setup.sys, "version_info", (3, 13, 0))
    install = Mock()
    monkeypatch.setattr(setup, "install", install)

    assert setup.main([]) == 1

    install.assert_not_called()
    assert "需要 Python 3.12" in capsys.readouterr().err


@pytest.mark.parametrize("version", ["[3, 11]", "[3, 13]", "invalid-json"])
def test_incompatible_existing_venv_is_preserved(monkeypatch, source_root, version):
    python = setup.local_python(source_root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"existing environment")
    monkeypatch.setattr(setup.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(setup, "node_tools", lambda: ("node", ["node", "npm-cli.js"]))
    monkeypatch.setattr(setup, "probe", lambda command: version)

    with pytest.raises(setup.SetupError, match=".venv"):
        setup.check_environment(source_root)

    assert python.read_bytes() == b"existing environment"


def test_windows_npm_uses_node_entry_instead_of_a_command_shell(monkeypatch, tmp_path):
    node = tmp_path / "Node & tools" / "node.exe"
    npm = node.with_name("npm.cmd")
    cli = node.parent / "node_modules/npm/bin/npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("", encoding="utf-8")
    npm.write_text("", encoding="utf-8")
    monkeypatch.setattr(setup.shutil, "which", lambda tool: str(node) if tool == "node" else str(npm))
    probe = Mock(side_effect=["v22.11.0", "10.9.0"])
    monkeypatch.setattr(setup, "probe", probe)

    assert setup.node_tools() == (str(node), [str(node), str(cli)])
    assert probe.call_args_list[1].args[0] == [str(node), str(cli), "--version"]


def test_old_node_has_actionable_error(monkeypatch):
    monkeypatch.setattr(setup.shutil, "which", lambda tool: "node")
    monkeypatch.setattr(setup, "probe", lambda command: "v18.20.0")

    with pytest.raises(setup.SetupError, match="Node.js 20"):
        setup.node_tools()


def test_install_creates_venv_then_installs_only_through_its_python(monkeypatch, source_root):
    python = setup.local_python(source_root)
    npm_command = ["node", "C:/Node & tools/npm-cli.js"]
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["-m", "venv"]:
            python.parent.mkdir(parents=True)
            python.write_bytes(b"venv")

    monkeypatch.setattr(setup.subprocess, "run", run)

    setup.install(source_root, python, npm_command)

    assert [command for command, _ in calls] == [
        [setup.sys.executable, "-m", "venv", str(source_root / ".venv")],
        [str(python), "-m", "pip", "install", "-r", str(source_root / "backend/requirements.txt")],
        [*npm_command, "ci", "--prefix", str(source_root / "frontend")],
    ]
    assert all(kwargs == {"cwd": source_root, "check": True} for _, kwargs in calls)


def test_failed_pip_does_not_continue_to_frontend_install(monkeypatch, source_root, capsys):
    python = setup.local_python(source_root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"existing environment")
    monkeypatch.setattr(setup, "ROOT", source_root)
    monkeypatch.setattr(setup, "check_environment", lambda root: (python, ["node", "npm-cli.js"]))
    run = Mock(side_effect=subprocess.CalledProcessError(7, [str(python), "-m", "pip"]))
    monkeypatch.setattr(setup.subprocess, "run", run)

    assert setup.main([]) == 1

    run.assert_called_once()
    assert run.call_args.args[0][0] == str(python)
    assert "错误码 7" in capsys.readouterr().err


@pytest.mark.parametrize("filename", ["install.cmd", "start.cmd"])
def test_windows_launchers_keep_crlf_for_chinese_console_text(filename):
    content = (setup.ROOT / filename).read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")


@pytest.mark.skipif(os.name != "nt", reason="Windows command interpreter integration")
def test_start_in_a_path_with_spaces_fails_cleanly_without_dependencies(source_root):
    script = source_root / "start.cmd"
    script.write_bytes((setup.ROOT / "start.cmd").read_bytes())

    result = subprocess.run(
        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "start.cmd"], cwd=source_root,
        input="\n", capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
    )

    assert result.returncode == 1
    assert "请先双击同一文件夹中的 install.cmd" in result.stdout
    assert not result.stderr
