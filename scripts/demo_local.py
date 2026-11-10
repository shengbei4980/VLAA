"""Exercise the running local workbench with official COCO8 and street images."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import httpx
import onnx
import onnxruntime
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "demo"


def save(name: str, value: dict) -> dict:
    (OUTPUT / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return value


def api(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    if response.is_error:
        raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
    return response.json()


def wait(client: httpx.Client, path: str) -> dict:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        result = api(client, "GET", path)
        if result["status"] in {"completed", "failed", "stopped"}:
            assert result["status"] == "completed", result
            return result
        time.sleep(2)
    raise TimeoutError(f"Task still running. Inspect {path} before retrying.")


def wait_for_human_review(client: httpx.Client, dataset_id: str) -> None:
    print(f"请在工作台核对数据集 {dataset_id}，保存并整图确认所有图片。正在等待人工回执。", flush=True)
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        status = api(client, "GET", f"/api/datasets/{dataset_id}/review-status")
        if status["total"] and status["pending"] == 0 and not any(item["has_draft"] for item in status["items"]):
            return
        time.sleep(2)
    raise TimeoutError("Human review is still pending. Keep the saved dataset and continue reviewing in the workbench.")


def main() -> None:
    assert zipfile.is_zipfile(ROOT / "data/models/base/mobile_sam.pt"), "Finish downloading the SAM checkpoint before running the demo."
    assert torch.cuda.is_available(), "CUDA must be available for this GPU demo."
    save("environment", {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    archive_path = OUTPUT / "coco8-import.zip"
    with zipfile.ZipFile(OUTPUT / "coco8.zip") as source, zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            if not item.is_dir():
                target.writestr(item.filename, source.read(item))
        target.writestr("data.yaml", (OUTPUT / "coco8.yaml").read_text(encoding="utf-8"))

    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=300, trust_env=False) as client:
        assert api(client, "GET", "/api/system/health")["status"] == "ok"
        with archive_path.open("rb") as source:
            imported = save("import", api(client, "POST", "/api/datasets/import/yolo", files={"file": (archive_path.name, source, "application/zip")}, data={"name": "Demo - COCO8 training", "task_type": "detect", "description": "Official 8-image smoke dataset. Demo metrics do not measure research accuracy."}))
        dataset_id = imported["dataset"]["id"]
        assert imported["imported_images"] == 8 and imported["imported_annotations"] > 0, imported
        validation = save("validation", api(client, "POST", f"/api/datasets/{dataset_id}/validate"))
        assert validation["valid"], validation
        wait_for_human_review(client, dataset_id)
        task = save("training-submitted", api(client, "POST", "/api/training/tasks", json={"dataset_id": dataset_id, "name": "Demo - GPU YOLO11n COCO8", "model": str(ROOT / "data/models/base/yolo11n.pt"), "epochs": 1, "img_size": 320, "batch_size": 2, "device": "0", "workers": 0, "optimizer": "SGD", "lr0": 0.001}))
        print(f"GPU training started: {task['id']}", flush=True)
        trained = save("training", wait(client, f"/api/training/tasks/{task['id']}"))
        assert Path(trained["best_model_path"]).is_file() and Path(trained["last_model_path"]).is_file()
        models = api(client, "GET", f"/api/models?dataset_id={dataset_id}")["items"]
        model = next(item for item in models if item["training_task_id"] == task["id"] and item["artifact_type"] == "best")
        save("model", model)
        model_id = model["id"]
        print(f"Training completed. Testing managed model {model_id}", flush=True)
        with (OUTPUT / "bus.jpg").open("rb") as source:
            prediction = save("street-prediction", api(client, "POST", f"/api/models/{model_id}/test?confidence=0.25", files={"file": ("bus.jpg", source, "image/jpeg")}))
        detections = [item for item in prediction["detections"] if item["confidence"] >= 0.25]
        assert detections and any(item["class_name"] == "bus" for item in detections), prediction
        image = Image.open(OUTPUT / "bus.jpg").convert("RGB")
        draw = ImageDraw.Draw(image)
        for item in detections:
            x, y, w, h = (item[key] for key in ("x", "y", "width", "height"))
            draw.rectangle((x, y, x + w, y + h), outline="#22c55e", width=3)
            draw.text((x + 3, max(0, y - 14)), f"{item['class_name']} {item['confidence']:.2f}", fill="#00ff00", stroke_width=1, stroke_fill="black")
        image.save(OUTPUT / "street-detection.jpg")

        target = save("auto-dataset", api(client, "POST", "/api/datasets", json={"name": "Demo - street auto labels (review required)", "task_type": "detect"}))
        target_id = target["id"]
        mapping = {}
        for index, name in ((0, "person"), (5, "bus"), (2, "car"), (9, "traffic light")):
            label = api(client, "POST", f"/api/datasets/{target_id}/classes", json={"name": name})
            mapping[str(index)] = label["id"]
        with (OUTPUT / "bus.jpg").open("rb") as source:
            api(client, "POST", f"/api/datasets/{target_id}/images/upload", files={"files": ("bus.jpg", source, "image/jpeg")}, data={"split": "train"})
        auto = save("auto-submitted", api(client, "POST", f"/api/datasets/{target_id}/auto-annotation", json={"model_id": model_id, "confidence": 0.25, "iou": 0.45, "class_mapping": mapping}))
        auto = save("auto-annotation", wait(client, f"/api/auto-annotation/{auto['id']}"))
        assert auto["created_annotations"] > 0, auto
        images = api(client, "GET", f"/api/datasets/{target_id}/images")["items"]
        detail = api(client, "GET", f"/api/datasets/{target_id}/images/{images[0]['id']}/review")
        annotations = detail["draft"]["annotations"]
        assert annotations and all(item["source"] == "auto" for item in annotations), annotations
        save("street-auto-labels", {"dataset_id": target_id, "annotations": annotations, "review_status": "pending"})
        wait_for_human_review(client, target_id)

        sam_dataset = save("sam-dataset", api(client, "POST", "/api/datasets", json={"name": "Demo - SAM street segmentation", "task_type": "segment"}))
        sam_dataset_id = sam_dataset["id"]
        sam_class = api(client, "POST", f"/api/datasets/{sam_dataset_id}/classes", json={"name": "bus"})
        with (OUTPUT / "bus.jpg").open("rb") as source:
            api(client, "POST", f"/api/datasets/{sam_dataset_id}/images/upload", files={"files": ("bus.jpg", source, "image/jpeg")})
        sam_image = api(client, "GET", f"/api/datasets/{sam_dataset_id}/images")["items"][0]
        api(client, "PUT", "/api/settings/sam", json={"enabled": True, "model": str(ROOT / "data/models/base/mobile_sam.pt"), "device": "0", "img_size": 1024, "fallback_mode": "disabled"})
        bus = next(item for item in detections if item["class_name"] == "bus")
        mask = save("sam-prediction", api(client, "POST", "/api/sam/predict", json={"image_id": sam_image["id"], "class_id": sam_class["id"], "prompt_type": "box", "box": {key: bus[key] for key in ("x", "y", "width", "height")}}))
        assert mask["backend_used"] == "ultralytics_sam" and len(mask["polygon"]) >= 3, mask
        overlay = Image.new("RGBA", image.size)
        ImageDraw.Draw(overlay).polygon([tuple(point) for point in mask["polygon"]], fill=(0, 255, 0, 90), outline=(0, 255, 0, 255), width=3)
        Image.alpha_composite(Image.open(OUTPUT / "bus.jpg").convert("RGBA"), overlay).convert("RGB").save(OUTPUT / "street-sam.jpg")
        print("Street detection, automatic labels, and real SAM inference passed.", flush=True)

        evaluation = api(client, "POST", f"/api/models/{model_id}/evaluate", json={"split": "val"})
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            evaluation = next(item for item in api(client, "GET", f"/api/models/{model_id}/evaluations") if item["id"] == evaluation["id"])
            if evaluation["status"] in {"completed", "failed"}:
                break
            time.sleep(2)
        save("evaluation", evaluation)
        assert evaluation["status"] == "completed", evaluation

        exported = save("onnx", api(client, "POST", f"/api/models/{model_id}/export-onnx"))
        onnx_path = Path(exported["model_path"])
        onnx.checker.check_model(str(onnx_path))
        onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert onnx_path.is_file()
        response = client.get(f"/api/datasets/{target_id}/export/yolo")
        response.raise_for_status()
        assert zipfile.is_zipfile(io.BytesIO(response.content))
        (OUTPUT / "street-auto-labels-yolo.zip").write_bytes(response.content)
        save("summary", {"status": "passed", "training_dataset_id": dataset_id, "street_dataset_id": target_id, "sam_dataset_id": sam_dataset_id, "model_id": model_id, "training_task_id": task["id"], "detected_classes": sorted({item["class_name"] for item in detections}), "detections_above_0_25": len(detections), "auto_annotations": auto["created_annotations"], "sam_backend": mask["backend_used"], "sam_device": mask["device"], "onnx_path": str(onnx_path), "note": "One-epoch COCO8 demo verifies execution, not building-component accuracy."})
        print("Demo passed: import, GPU training, street inference, auto labels, SAM, evaluation, YOLO export, ONNX.", flush=True)


if __name__ == "__main__":
    main()
