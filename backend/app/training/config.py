from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.errors import ValidationError
from app.core.task_types import TaskType

if TYPE_CHECKING:
    from app.core.storage import Storage


def infer_model_family(model: str) -> TaskType | None:
    name = Path(model).name.lower()
    if name in {"best.pt", "last.pt"}:
        return None
    if "-seg." in name or "_seg." in name:
        return TaskType.SEGMENT
    if "-obb." in name or "_obb." in name:
        return TaskType.OBB
    if "-cls." in name or "_cls." in name:
        return TaskType.CLASSIFY
    if Path(model).is_absolute() or Path(model).parent != Path("."):
        return None
    if name.endswith(".pt") or name.endswith(".yaml"):
        return TaskType.DETECT
    return None


def validate_model_family(task_type: TaskType, model: str) -> None:
    family = infer_model_family(model)
    if family is not None and family is not task_type:
        raise ValidationError(
            "model_task_mismatch",
            f"{task_type.value} training requires a {task_type.value} weight family; received {family.value} model.",
        )


def validate_local_model_task(task_type: TaskType, model_reference: str) -> None:
    if not Path(model_reference).is_file():
        return
    from ultralytics import YOLO
    from ultralytics.nn.tasks import guess_model_task

    try:
        model = YOLO(model_reference)
        actual = guess_model_task(model.model)
    except Exception as exc:
        raise ValidationError("model_load_failed", "The selected managed weight could not be loaded as an Ultralytics model.") from exc
    if actual != task_type.value:
        raise ValidationError("model_task_mismatch", f"The selected weight supports {actual}; this task requires {task_type.value}.")


def resolve_model_reference(model: str, storage: "Storage | None" = None) -> str:
    candidate = Path(model).expanduser()
    if candidate.suffix.lower() not in {".pt", ".yaml"}:
        raise ValidationError("unsupported_model_reference", "Training models must be a .pt or .yaml reference.")
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise ValidationError("model_file_missing", "The local model file does not exist.")
        if storage is not None:
            try:
                storage.managed_model_path(resolved)
            except ValidationError as exc:
                raise ValidationError(
                    "model_path_outside_managed_dir",
                    "Local training weights must be stored in the managed models directory.",
                ) from exc
        return str(resolved)
    if not model.endswith((".pt", ".yaml")):
        raise ValidationError("unsupported_model_reference", "Training models must be a .pt or .yaml reference.")
    return model


@dataclass(frozen=True)
class TrainingCommand:
    args: list[str]
    readable: str


def build_training_command(
    *,
    task_type: TaskType,
    model: str,
    data_yaml: str,
    run_dir: str,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    workers: int,
    seed: int,
    optimizer: str | None = None,
    lr0: float | None = None,
    patience: int | None = None,
    resume: bool = False,
) -> TrainingCommand:
    run_path = Path(run_dir).resolve()
    args = [
        "yolo",
        task_type.value,
        "train",
        f"model={resolve_model_reference(model)}",
        f"data={Path(data_yaml).resolve()}",
        f"epochs={epochs}",
        f"imgsz={img_size}",
        f"batch={batch_size}",
        f"project={run_path.parent}",
        f"name={run_path.name}",
        "exist_ok=True",
        f"workers={workers}",
        f"seed={seed}",
    ]
    if device != "auto":
        args.append(f"device={device}")
    if optimizer:
        args.append(f"optimizer={optimizer}")
    if lr0 is not None:
        args.append(f"lr0={lr0}")
    if patience is not None:
        args.append(f"patience={patience}")
    if resume:
        # The selected managed checkpoint is already the model reference. This
        # matches upstream: loading that model and passing resume=True makes
        # Ultralytics restore its epoch/optimizer state without searching for a
        # different latest run.
        args.append("resume=True")
    return TrainingCommand(args=args, readable=shlex.join(args))
