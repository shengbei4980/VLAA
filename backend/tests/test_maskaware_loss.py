from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.tasks import DetectionModel, SegmentationModel
from ultralytics.utils.tal import make_anchors

from app.training.maskaware_loss import MaskAwareDetectionLoss, MaskAwareSegmentationLoss


@pytest.fixture(scope="module", autouse=True)
def limited_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_and_batch(task="segment", family="yolo11", flipped=False):
    torch.manual_seed(7)
    model_class = SegmentationModel if task == "segment" else DetectionModel
    model = model_class(f"{family}n{'-seg' if task == 'segment' else ''}.yaml", nc=2, verbose=False)
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, "overlap_mask": False})
    model.train()
    image = torch.randn(1, 3, 64, 64)
    valid = torch.ones(1, 64, 64, dtype=torch.bool)
    valid[:, :, 32:] = False
    masks = torch.zeros(1, 64, 64)
    masks[:, 8:56, 8:56] = 1
    masks[:, 20:25, 20:25] = 0
    if flipped:
        image, valid, masks = image.flip(-1), valid.flip(-1), masks.flip(-1)
    batch = {"img": image, "valid_mask": valid, "batch_idx": torch.tensor([0]), "cls": torch.tensor([[0.]]),
             "bboxes": torch.tensor([[.5, .5, .75, .75]])}
    if task == "segment":
        batch["masks"] = masks
    return model, batch


def leaf_predictions(prediction):
    return {key: value.detach().clone().requires_grad_() if isinstance(value, torch.Tensor)
            else [feature.detach() for feature in value] for key, value in prediction.items()}


@pytest.mark.parametrize("family", ["yolov8", "yolo11"])
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_real_model_forward_backward_has_finite_nonzero_valid_supervision(family, task):
    model, batch = model_and_batch(task, family)
    criterion = MaskAwareSegmentationLoss(model) if task == "segment" else MaskAwareDetectionLoss(model)
    values, components = criterion(model(batch["img"]), batch)
    assert torch.isfinite(values).all() and values.sum() > 0
    assert components["cls_loss"] > 0 and components["box_loss"] > 0
    if task == "segment":
        assert components["seg_loss"] > 0
    values.sum().backward()
    gradient = next(model.parameters()).grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("flipped", [False, True])
def test_ignored_target_pixels_and_fully_ignored_objects_do_not_change_loss(flipped):
    model, batch = model_and_batch(flipped=flipped)
    criterion = MaskAwareSegmentationLoss(model)
    prediction = leaf_predictions(model(batch["img"]))
    expected = criterion(prediction, batch)[0]
    changed_mask = batch["masks"].clone()
    changed_mask[:, ~batch["valid_mask"][0]] = 1 - changed_mask[:, ~batch["valid_mask"][0]]
    ignored_object = (~batch["valid_mask"]).float()
    changed = {**batch, "masks": torch.cat((changed_mask, ignored_object)), "batch_idx": torch.tensor([0, 0]),
               "cls": torch.tensor([[0.], [1.]]), "bboxes": torch.tensor([[.5, .5, .75, .75], [.5, .5, 1., 1.]])}
    actual = criterion(prediction, changed)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("flipped", [False, True])
def test_ignored_predictions_have_zero_gradient_and_cannot_change_assignments(flipped):
    model, batch = model_and_batch(flipped=flipped)
    criterion = MaskAwareSegmentationLoss(model)
    prediction = leaf_predictions(model(batch["img"]))
    anchors, strides = make_anchors(prediction["feats"], criterion.stride, .5)
    centers = (anchors * strides).long()
    valid_anchor = batch["valid_mask"][:, centers[:, 1], centers[:, 0]][0]
    expected = criterion(prediction, batch)[0]
    mutated = leaf_predictions(prediction)
    with torch.no_grad():
        for key in ("scores", "boxes", "mask_coefficient"):
            mutated[key][:, :, ~valid_anchor] += 100
        proto_invalid = torch.nn.functional.adaptive_max_pool2d((~batch["valid_mask"]).float().unsqueeze(1), prediction["proto"].shape[-2:]).bool()
        mutated["proto"].masked_fill_(proto_invalid, 100)
    actual = criterion(mutated, batch)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    for key in ("scores", "boxes", "mask_coefficient"):
        assert prediction[key].grad[:, :, ~valid_anchor].abs().sum() == 0
        assert prediction[key].grad[:, :, valid_anchor].abs().sum() > 0
    assert prediction["proto"].grad.masked_select(proto_invalid).abs().sum() == 0
    assert prediction["proto"].grad.masked_select(~proto_invalid).abs().sum() > 0


def test_valid_prediction_and_mask_label_changes_affect_supervision():
    model, batch = model_and_batch()
    criterion = MaskAwareSegmentationLoss(model)
    prediction = leaf_predictions(model(batch["img"]))
    expected = criterion(prediction, batch)[0]
    changed_predictions = leaf_predictions(prediction)
    with torch.no_grad():
        changed_predictions["scores"][:, :, :] += 1
    assert not torch.allclose(criterion(changed_predictions, batch)[0], expected)
    changed_masks = batch["masks"].clone()
    changed_masks[:, 10:18, 10:18] = 0
    changed_loss = criterion(prediction, {**batch, "masks": changed_masks})[0]
    assert not torch.allclose(changed_loss, expected)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_confirmed_empty_image_retains_valid_background_loss(task):
    model, batch = model_and_batch(task)
    criterion = MaskAwareSegmentationLoss(model) if task == "segment" else MaskAwareDetectionLoss(model)
    batch.update(batch_idx=torch.empty(0, dtype=torch.long), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4))
    if task == "segment":
        batch["masks"] = torch.empty(0, 64, 64)
    prediction = leaf_predictions(model(batch["img"]))
    values, components = criterion(prediction, batch)
    assert components["cls_loss"] > 0
    values.sum().backward()
    assert prediction["scores"].grad.abs().sum() > 0
    all_ignored = {**batch, "valid_mask": torch.zeros_like(batch["valid_mask"])}
    assert criterion(leaf_predictions(prediction), all_ignored)[0].sum() == 0


def test_unsupported_heads_and_missing_valid_contract_fail_explicitly():
    model, batch = model_and_batch()
    criterion = MaskAwareSegmentationLoss(model)
    invalid = dict(batch)
    invalid.pop("valid_mask")
    with pytest.raises(ValueError, match="valid_mask"):
        criterion(model(batch["img"]), invalid)
    model.args.overlap_mask = True
    with pytest.raises(ValueError, match="overlap_mask"):
        MaskAwareSegmentationLoss(model)
    with pytest.raises(ValueError, match="non-end2end"):
        MaskAwareSegmentationLoss(SimpleNamespace(end2end=True))
