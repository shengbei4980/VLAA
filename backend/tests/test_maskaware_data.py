import hashlib
import json
from collections.abc import Iterator

import numpy as np
from PIL import Image
import torch

from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService
from app.training.maskaware_data import CanonicalDataset


def test_canonical_loader_streams_source_masks_and_transforms_all_layers_together(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    root = data_dir / "snapshots" / "dataset"
    root.mkdir(parents=True)
    mask = np.zeros((40, 80), dtype=bool)
    mask[8:32, 12:54] = True
    mask[14:24, 20:32] = False
    ignored = np.zeros_like(mask)
    ignored[:, 40:44] = True
    masks = MaskService(Storage(data_dir, data_dir / "imports"))
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[mask] = [200, 80, 50]
    path = root / "image.png"
    Image.fromarray(pixels).save(path)
    image = {"id": "image", "path": "image.png", "image_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
             "width": 80, "height": 40, "split": "train", "ignore_mask_ref": masks.write(ignored).mask_ref,
             "annotations": [{"class_id": "object", "type": "polygon", "mask_ref": masks.write(mask).mask_ref}]}
    manifest = {"task_type": "segment", "classes": [{"id": "object", "name": "object", "index": 0}], "images": [image]}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {"workflow_manifest": str(manifest_path), "workflow_data_dir": str(data_dir)}
    plain = CanonicalDataset(config, "train", 64)
    source_masks, valid = plain._objects(image)
    assert isinstance(source_masks, Iterator)
    _, visible = next(source_masks)
    assert np.array_equal(visible, mask & ~ignored)
    assert np.array_equal(valid, ~ignored)
    original = plain[0]
    monkeypatch.setattr("app.training.maskaware_data.random.random", lambda: 0.0)
    flipped = CanonicalDataset(config, "train", 64, augment=True)[0]
    assert torch.equal(flipped["img"], original["img"].flip(-1))
    assert torch.equal(flipped["masks"], original["masks"].flip(-1))
    assert torch.equal(flipped["valid_mask"], original["valid_mask"].flip(-1))
    assert torch.allclose(flipped["bboxes"][:, 0], 1 - original["bboxes"][:, 0])
    assert not (flipped["masks"].bool() & ~flipped["valid_mask"]).any()
