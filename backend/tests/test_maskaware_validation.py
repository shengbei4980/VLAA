from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.utils import ops

from app.training.maskaware import CanonicalDetectionValidator, CanonicalSegmentationValidator


def validator_for(tmp_path: Path, task: str):
    validator_class = CanonicalSegmentationValidator if task == "segment" else CanonicalDetectionValidator
    validator = validator_class(save_dir=tmp_path, args={"conf": .01, "iou": .5, "max_det": 1, "plots": False, "task": task})
    validator.nc = 1
    validator.end2end = False
    validator.device = torch.device("cpu")
    validator._canonical_strides = [8, 16, 32]
    valid = torch.ones(1, 64, 64, dtype=torch.bool)
    valid[:, :, 32:] = False
    validator.preprocess({"img": torch.zeros(1, 3, 64, 64, dtype=torch.uint8), "valid_mask": valid, "masks": torch.empty(0, 64, 64)})
    if task == "segment":
        validator.process = ops.process_mask_native
    return validator


def predictions(ignored_score: float, task: str):
    result = torch.zeros(1, 6 if task == "segment" else 5, 84)
    result[0, :4, 0] = torch.tensor([16., 16., 16., 16.])
    result[0, :4, 4] = torch.tensor([48., 16., 16., 16.])
    result[0, 4, 0] = .8
    result[0, 4, 4] = ignored_score
    if task == "segment":
        result[0, 5, :] = 1
    return result


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_ignore_scores_cannot_consume_nms_max_det_budget(tmp_path, task):
    validator = validator_for(tmp_path, task)
    outcomes = []
    for score in (.1, .99):
        raw = predictions(score, task)
        before = raw.clone()
        model_output = ((raw, torch.ones(1, 1, 16, 16)), {}) if task == "segment" else raw
        result = validator.postprocess(model_output)[0]
        outcomes.append(result["bboxes"])
        torch.testing.assert_close(raw, before)
    torch.testing.assert_close(outcomes[0], torch.tensor([[8., 8., 24., 24.]]))
    torch.testing.assert_close(outcomes[1], outcomes[0])


def test_ignore_prototype_changes_do_not_leak_into_valid_mask_edges(tmp_path):
    validator = validator_for(tmp_path, "segment")
    raw = predictions(.99, "segment")
    proto = torch.ones(1, 1, 16, 16)
    modified = proto.clone()
    modified[:, :, :, 8:] = -100
    first = validator.postprocess(((raw, proto), {}))[0]
    second = validator.postprocess(((raw, modified), {}))[0]
    assert first["masks"].any()
    torch.testing.assert_close(first["masks"], second["masks"])


def test_unknown_prediction_grid_requires_adapter_instead_of_guessing(tmp_path):
    validator = validator_for(tmp_path, "detect")
    with pytest.raises(ValueError, match="anchor grid"):
        validator.postprocess(torch.zeros(1, 5, 2))
    validator.end2end = True
    with pytest.raises(ValueError, match="valid-anchor"):
        validator.postprocess(torch.zeros(1, 5, 84))


def test_unsupported_model_head_is_rejected_before_native_validation(tmp_path):
    validator = validator_for(tmp_path, "detect")
    unsupported = SimpleNamespace(model=torch.nn.Sequential(torch.nn.Conv2d(3, 1, 1)))
    with pytest.raises(ValueError, match="v8/v11"):
        validator.init_metrics(unsupported)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_valid_anchor_regressed_into_ignore_cannot_consume_max_det(tmp_path, task):
    validator = validator_for(tmp_path, task)
    raw = predictions(.99, task)
    raw[:, :, 1] = raw[:, :, 4]
    raw[:, 4, 4] = 0
    model_output = ((raw, torch.ones(1, 1, 16, 16)), {}) if task == "segment" else raw
    result = validator.postprocess(model_output)[0]
    torch.testing.assert_close(result["bboxes"], torch.tensor([[8., 8., 24., 24.]]))


def test_empty_valid_mask_prediction_is_retained_as_false_positive(tmp_path, monkeypatch):
    from ultralytics.models.yolo.segment import SegmentationValidator

    validator = validator_for(tmp_path, "segment")
    captured = []
    monkeypatch.setattr(SegmentationValidator, "update_metrics", lambda self, predictions, batch: captured.extend(predictions))
    batch = {"valid_mask": validator._canonical_valid}
    predictions = [{"bboxes": torch.tensor([[8., 8., 24., 24.]]), "masks": torch.zeros(1, 64, 64),
                    "cls": torch.tensor([0.]), "conf": torch.tensor([.9])}]
    validator.update_metrics(predictions, batch)
    assert len(captured[0]["bboxes"]) == 1
    assert not captured[0]["masks"].any()
