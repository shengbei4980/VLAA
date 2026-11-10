"""Explicit test setup for reviewed labels and portable fake YOLO processes."""
from pathlib import Path
import sys


def confirm_image(client, dataset_id: str, image_id: str, *, confirmed_empty: bool = False):
    response = client.get(f"/api/datasets/{dataset_id}/images/{image_id}/review")
    assert response.status_code == 200, response.text
    detail = response.json()
    response = client.post(f"/api/datasets/{dataset_id}/images/{image_id}/review/confirm", json={
        "expected_revision": detail["revision"], "expected_fingerprint": detail["fingerprint"],
        "confirmed_empty": confirmed_empty, "reason": "Explicitly reviewed fixture labels",
    })
    assert response.status_code == 200, response.text
    return response.json()


def install_fake_yolo(path: Path, monkeypatch, mode="normal"):
    from app.training.runtime.runner import TrainingRunner
    script = '''import os
from pathlib import Path
import sys
import time

args = dict(value.split("=", 1) for value in sys.argv[1:] if "=" in value)
capture = os.environ.get("YWA_ARGS_CAPTURE")
if capture:
    Path(capture).write_text("\\n".join(sys.argv[1:]), encoding="utf-8")
root = Path(args["project"]) / args.get("name", "run")
(root / "weights").mkdir(parents=True, exist_ok=True)
if MODE == "live":
    (root / "results.csv").write_text("epoch,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B)\\n0,0.42,0.21,0.60,0.30\\n", encoding="utf-8")
    print("1/3", flush=True)
    time.sleep(1)
elif MODE == "detailed":
    (root / "results.csv").write_text("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/mAP50(B)\\n1,2.5,1.2,0.3,0.1,0.42\\n", encoding="utf-8")
    print("1/3 1.2G 1.2 0.3 0.1 2.5it/s", flush=True)
    time.sleep(1)
elif MODE == "slow":
    print("1/20", flush=True)
    time.sleep(5)
else:
    (root / "results.csv").write_text("epoch,metrics/mAP50(B)\\n0,0.81\\n", encoding="utf-8")
    print("1/1", flush=True)
(root / "weights" / "best.pt").write_bytes(b"fake best")
(root / "weights" / "last.pt").write_bytes(b"fake last")
'''
    path.write_text("MODE = " + repr(mode) + "\n" + script, encoding="utf-8")
    monkeypatch.setattr("app.training.runtime.runner.resolve_yolo_command_prefix", lambda: [sys.executable, str(path)])
    original_command = TrainingRunner._command

    def fake_command(runner, task):
        command = original_command(runner, task)
        if command[1:3] == ["-m", "app.training.maskaware"]:
            return [sys.executable, str(path), *command[3:]]
        return command

    monkeypatch.setattr(TrainingRunner, "_command", fake_command)
