from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from types import MethodType, SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import load_checkpoint

from app.core.database import Base, Database
from app.core.errors import ValidationError
from app.core.models import Annotation, ClassLabel, Dataset, ImageItem, ModelVersion, TrainingTask
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService
from app.dataset.annotation.review import confirm_image, current_content, json_fingerprint, require_confirmed
from app.dataset.snapshots import create_snapshot, get_snapshot, verify_snapshot_images
from app.storage_settings.rebase import transform_copied_data
from app.workflow.artifacts import WorkflowArtifactStore, fingerprint
from app.workflow.models import WorkflowArtifact, WorkflowAttempt, WorkflowEvent, WorkflowRun, WorkflowTaskProfile, WorkflowTemplate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path: Path, source: Path, data_yaml: Path):
    model = torch.nn.Linear(2, 1)
    model.task = "detect"
    model.stride = torch.tensor([8, 16, 32])
    args = {**DEFAULT_CFG_DICT, "data": str(data_yaml), "project": str(path.parent.parent), "name": "run",
            "save_dir": str(path.parent.parent), "epochs": 3, "close_mosaic": 0, "resume": False}
    model.args = args
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "ema": None, "train_args": args, "epoch": 0,
                "optimizer": optimizer.state_dict(), "scaler": None, "best_fitness": .1}, path, pickle_protocol=2)


def make_source(tmp_path):
    source, stage, target = (tmp_path / name for name in ("source-data", "staging-copy", "target-data"))
    storage = Storage(source, source / "imports")
    outside_db = tmp_path / "external-database" / "source.sqlite"
    outside_db.parent.mkdir()
    database = Database(f"sqlite:///{outside_db}")
    Base.metadata.create_all(database.engine)
    mask = np.zeros((24, 32), dtype=bool)
    mask[2:20, 2:25] = True
    mask[7:12, 7:14] = False
    mask_ref = MaskService(storage).write(mask).mask_ref
    with database.session_factory() as session:
        session.add(Dataset(id="dataset", name="Migration specimen", task_type="segment"))
        session.flush()
        session.add(ClassLabel(id="class", dataset_id="dataset", name="object", class_index=0))
        session.flush()
        for index, split in enumerate(("train", "val", "test")):
            content = io.BytesIO()
            Image.new("RGB", (32, 24), "white").save(content, format="PNG")
            storage.write_image("dataset", f"image{index}.png", content.getvalue())
            session.add(ImageItem(id=f"image{index}", dataset_id="dataset", storage_name=f"image{index}.png", file_name=f"image{index}.png",
                                 width=32, height=24, split=split, source_file=str(source / "historical-source.pdf"), rules_json={"reference_path": str(source / "historical-rules")}, annotation_revision=0))
            session.flush()
            session.add(Annotation(id=f"annotation{index}", dataset_id="dataset", image_id=f"image{index}", class_id="class", type="polygon", object_id=f"object{index}",
                                   source="manual", mask_ref=mask_ref, provenance_json={"model_path": str(source / "historical-model.pt")}))
        session.flush()
        for identity in ("image0", "image1"):
            image = session.get(ImageItem, identity)
            content = current_content(session, storage, image)
            confirm_image(session, storage, identity, expected_revision=0, expected_fingerprint=json_fingerprint(content), reason=str(source / "audit-reason"))
        snapshot = create_snapshot(session, storage, "dataset", image_ids=["image0", "image1"], snapshot_id="snapshot")
        session.commit()
    export = source / "runs" / "training" / "training" / "dataset"
    export.mkdir(parents=True)
    data_yaml = export / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(export), "train": "images/train", "val": "images/val", "names": ["object"],
                                       "workflow_manifest": str(Path(snapshot["root"]) / "manifest.json"), "workflow_data_dir": str(source)}), encoding="utf-8")
    for split, identity in (("train", "image0"), ("val", "image1")):
        directory = export / "images" / split
        directory.mkdir(parents=True)
        shutil.copy2(storage.image_path("dataset", identity + ".png"), directory / (identity + ".png"))
    (export / "train.txt").write_text(str(source / "datasets/dataset/images/image0.png") + "\n", encoding="utf-8")
    weights = source / "runs/training/training/run/weights/last.pt"
    checkpoint(weights, source, data_yaml)
    sam = source / "models/base/sam.pt"
    sam.parent.mkdir(parents=True)
    torch.save({"state_dict": {"weight": torch.ones(2, 3)}}, sam, pickle_protocol=2)
    (source / "settings.json").write_text(json.dumps({"sam": {"model": str(sam)}, "llm": {"api_key": "preserved-placeholder"}}), encoding="utf-8")
    with database.session_factory() as session:
        session.add(TrainingTask(id="training", dataset_id="dataset", name="source task", status="stopped", task_type="segment", model_name=str(sam), model_path=str(weights),
                                 epochs=3, img_size=64, batch_size=1, device="cpu", workers=0, val_ratio=.2, seed=0,
                                 config_json={"model": str(sam), "snapshot_id": "snapshot"}, command_args_json=["yolo", "segment", "train", "model=" + str(weights), "data=" + str(data_yaml)],
                                 export_path=str(export), data_yaml_path=str(data_yaml), run_dir=str(weights.parent.parent), logs_path=str(source / "runs/training/training/train.log")))
        session.flush()
        session.add(ModelVersion(id="model", name="source model", dataset_id="dataset", training_task_id="training", source="training_task", artifact_type="last", format="pt", task_type="segment", model_path=str(weights)))
        session.add(WorkflowTaskProfile(id="profile", key="profile", version=1, name="profile", config_json={}))
        session.add(WorkflowTemplate(id="template", key="template", version=1, name="template", nodes_json=[]))
        session.flush()
        context = {"dataset_id": "dataset", "image_ids": ["image0"], "snapshot_id": None, "next_batches": [], "model_path": str(weights)}
        session.add(WorkflowRun(id="run", idempotency_key="run", fingerprint="old", template_id="template", profile_id="profile", template_snapshot={}, profile_snapshot={}, context_json=context))
        session.flush()
        inputs = {"inputs": context, "payload": {"image_path": str(source / "datasets/dataset/images/image0.png")}}
        session.add(WorkflowAttempt(id="attempt", run_id="run", unit_key="batch", node_id="node", node_type="snapshot", attempt=1, generation=1,
                                    fingerprint=fingerprint(inputs), status="completed", input_json=inputs))
        session.flush()
        store = WorkflowArtifactStore(storage)
        payload = {"image_path": str(source / "datasets/dataset/images/image0.png"), "provenance_json": {"model_path": str(source / "historical-model.pt")}}
        name, artifact_hash = store.write(payload)
        session.add(WorkflowArtifact(id="artifact", run_id="run", attempt_id="attempt", kind="proposals", sha256=artifact_hash, storage_name=name, payload_json=payload, parent_ids_json=[]))
        child = {"artifact_path": str(source / "workflows/artifacts" / name), "sha256": artifact_hash}
        child_name, child_hash = store.write(child)
        session.add(WorkflowArtifact(id="child", run_id="run", attempt_id="attempt", kind="report", sha256=child_hash, storage_name=child_name, payload_json=child, parent_ids_json=["artifact"]))
        session.commit()
    bundle = source / "custom-results" / "artifact"
    bundle.mkdir(parents=True)
    shutil.copy2(source / "workflows/artifacts" / name, bundle / "artifact.json")
    shutil.copy2(weights, bundle / "last.pt")
    (bundle / "package.json").write_text(json.dumps({"profile": {"models": {"training": str(weights)}}}), encoding="utf-8")
    receipt = {"schema_version": 1, "artifact_id": "artifact", "artifact_sha256": artifact_hash,
               "files": [{"path": path.name, "size": path.stat().st_size, "sha256": digest(path)} for path in bundle.iterdir()]}
    (bundle / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with database.session_factory() as session:
        session.add(WorkflowEvent(id="published", run_id="run", kind="output_published", payload_json={"path": str(bundle), "receipt_sha256": digest(bundle / "receipt.json")}))
        session.commit()
    database.dispose()
    shutil.copytree(source, stage)
    target_db = stage / "yolowebagent-starter.db"
    with closing(sqlite3.connect(outside_db)) as original, closing(sqlite3.connect(target_db)) as copied:
        original.backup(copied)
    return source, stage, target, target_db, weights.relative_to(source), sam.relative_to(source), data_yaml.relative_to(source)


def test_rebase_preserves_human_receipts_masks_snapshots_and_artifact_integrity(tmp_path):
    source, stage, target, copied_db, weights, sam, yaml_path = make_source(tmp_path)
    original_files = {str(path.relative_to(source)): digest(path) for path in source.rglob("*") if path.is_file()}
    result = transform_copied_data(stage, source, target, copied_db)
    assert result["checkpoints"] == 2
    assert result["artifacts"] == 2
    assert result["published_bundles"] == 1
    assert original_files == {str(path.relative_to(source)): digest(path) for path in source.rglob("*") if path.is_file()}
    assert digest(stage / sam) == original_files[str(sam)]
    source.rename(tmp_path / "source-is-unavailable")
    stage.rename(target)
    database = Database(f"sqlite:///{target / copied_db.name}")
    storage = Storage(target, target / "imports")
    try:
        with database.session_factory() as session:
            assert len(require_confirmed(session, storage, "dataset", ["image0", "image1"])) == 2
            with pytest.raises(ValidationError):
                require_confirmed(session, storage, "dataset", ["image2"])
            assert session.get(ImageItem, "image2").review_status == "pending"
            annotation = session.get(Annotation, "annotation0")
            assert annotation.provenance_json["model_path"] == str(source / "historical-model.pt")
            assert session.get(ImageItem, "image0").rules_json["reference_path"] == str(source / "historical-rules")
            frozen = get_snapshot(session, storage, "snapshot")
            verify_snapshot_images(frozen)
            for identity in ("artifact", "child"):
                artifact = session.get(WorkflowArtifact, identity)
                WorkflowArtifactStore(storage).verify(artifact.storage_name, artifact.sha256)
                assert fingerprint(artifact.payload_json) == artifact.sha256
            parent, child = session.get(WorkflowArtifact, "artifact"), session.get(WorkflowArtifact, "child")
            assert Path(child.payload_json["artifact_path"]).is_file()
            assert child.payload_json["sha256"] == parent.sha256
            assert session.get(WorkflowAttempt, "attempt").fingerprint == fingerprint(session.get(WorkflowAttempt, "attempt").input_json)
            assert session.get(WorkflowRun, "run").fingerprint == "old"
            assert session.get(ModelVersion, "model").model_path == str(target / weights)
            assert Path(session.get(TrainingTask, "training").data_yaml_path).is_file()
            from app.storage_settings.outputs import WorkflowOutputPublisher
            bundle = target / "custom-results/artifact"
            WorkflowOutputPublisher._verify_existing(bundle, parent)
            published = session.get(WorkflowEvent, "published").payload_json
            assert published["path"] == str(bundle)
            assert published["receipt_sha256"] == digest(bundle / "receipt.json")
    finally:
        database.dispose()
    data = yaml.safe_load((target / yaml_path).read_text())
    assert Path(data["workflow_manifest"]).is_file()
    assert data["workflow_data_dir"] == str(target)
    assert json.loads((target / "settings.json").read_text())["llm"]["api_key"] == "preserved-placeholder"


def test_rebased_checkpoint_restores_epoch_optimizer_and_new_paths_with_old_root_absent(tmp_path):
    source, stage, target, copied_db, weights, _, yaml_path = make_source(tmp_path)
    with ZipFile(source / weights) as archive:
        before = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if "/data/" in name}
    transform_copied_data(stage, source, target, copied_db)
    source.rename(tmp_path / "source-is-unavailable")
    stage.rename(target)
    with ZipFile(target / weights) as archive:
        after = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if "/data/" in name}
    assert before == after
    # This test loads only the checkpoint constructed above from a known torch Linear model.
    model, saved = load_checkpoint(target / weights)
    trainer = SimpleNamespace(args=SimpleNamespace(resume=str(target / weights), data=str(target / yaml_path)))
    BaseTrainer.check_resume(trainer, {"data": str(target / yaml_path)})
    assert trainer.args.data == str(target / yaml_path)
    assert Path(trainer.args.project).is_relative_to(target)
    assert Path(trainer.args.save_dir).is_relative_to(target)
    trainer.model = model
    trainer.epochs = 3
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    trainer.ema = None
    trainer._load_checkpoint_state = MethodType(BaseTrainer._load_checkpoint_state, trainer)
    BaseTrainer.resume_training(trainer, saved)
    assert trainer.start_epoch == 1 and trainer.optimizer.state


class _NeverExecute:
    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return os.system, ("echo unsafe > " + str(self.marker),)


def test_pickle_metadata_rewrite_does_not_execute_reconstruction_code(tmp_path):
    source, stage, target = (tmp_path / name for name in ("source", "stage", "target"))
    source.mkdir()
    stage.mkdir()
    database = stage / "db.sqlite"
    sqlite3.connect(database).close()
    marker = tmp_path / "must-not-exist"
    torch.save({"model_path": str(source / "managed.pt"), "untrusted": _NeverExecute(marker)}, stage / "untrusted.pt", pickle_protocol=2)
    transform_copied_data(stage, source, target, database)
    assert not marker.exists()


@pytest.mark.parametrize("protocol", [4, 5])
def test_unsupported_pickle_protocol_blocks_the_staging_migration(tmp_path, protocol):
    source, stage, target = (tmp_path / name for name in ("source", "stage", "target"))
    source.mkdir()
    stage.mkdir()
    database = stage / "db.sqlite"
    sqlite3.connect(database).close()
    torch.save({"path": str(source / "managed.pt"), "weight": torch.ones(2)}, stage / "unsupported.pt", pickle_protocol=protocol)
    with pytest.raises(ValidationError) as error:
        transform_copied_data(stage, source, target, database)
    assert error.value.error_code == "checkpoint_rebase_unsupported"
    assert source.is_dir() and not target.exists()


def test_real_yolo_epoch_resume_after_migration_with_original_root_unavailable(tmp_path, monkeypatch):
    from ultralytics import YOLO
    from app.training.maskaware import CanonicalSegmentationTrainer

    source, stage, target, copied_db, weights, _, yaml_path = make_source(tmp_path)
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: None)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        initial = YOLO("yolo11n-seg.yaml", task="segment")

        def stop_after_checkpoint(trainer):
            raise RuntimeError("intentional offline interruption after the first checkpoint")

        initial.add_callback("on_model_save", stop_after_checkpoint)
        with pytest.raises(RuntimeError, match="intentional offline interruption"):
            initial.train(trainer=CanonicalSegmentationTrainer, data=str(source / yaml_path), epochs=3, imgsz=64, batch=1,
                          device="cpu", workers=0, optimizer="SGD", lr0=.01, project=str((source / weights).parent.parent.parent),
                          name="run", exist_ok=True, overlap_mask=False, mask_ratio=1, rect=False, multi_scale=0.0,
                          close_mosaic=0, mosaic=0.0, mixup=0.0, copy_paste=0.0, amp=False, plots=False)
        assert (source / weights).is_file()
        shutil.copytree(source, stage, dirs_exist_ok=True)
        transform_copied_data(stage, source, target, copied_db)
        source.rename(tmp_path / "source-is-unavailable")
        stage.rename(target)
        resumed = YOLO(str(target / weights), task="segment")
        resumed.train(trainer=CanonicalSegmentationTrainer, resume=True, data=str(target / yaml_path), device="cpu", workers=0, plots=False)
        assert resumed.trainer.start_epoch == 1
        assert resumed.trainer.epoch == 2
        assert resumed.trainer.args.data == str(target / yaml_path)
        assert resumed.trainer.save_dir.is_relative_to(target)
        assert resumed.trainer.last.is_file() and resumed.trainer.best.is_file()
    finally:
        torch.set_num_threads(old_threads)


def test_migration_does_not_resign_an_already_changed_snapshot(tmp_path):
    source, stage, target, copied_db, *_ = make_source(tmp_path)
    path = stage / "snapshots/snapshot/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["images"][0]["file_name"] = "unexpected-change.png"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValidationError) as error:
        transform_copied_data(stage, source, target, copied_db)
    assert error.value.error_code == "storage_snapshot_invalid"
    assert source.is_dir() and not target.exists()


def test_external_paths_encoded_through_parent_segments_are_preserved(tmp_path):
    from app.storage_settings.rebase import _Rebaser

    source, target = tmp_path / "source", tmp_path / "target"
    rebaser = _Rebaser(source, target)
    outside = str(source) + "/../external/model.pt"
    assert rebaser.path(outside) == outside
    assert rebaser.path(str(source) + "/sub/../models/model.pt") == str(target / "models/model.pt")
