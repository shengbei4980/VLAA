"""Regression coverage for semantic priority, geometry and the spectral window contract."""
import io
import base64

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ValidationError
from app.dataset.annotation.masks import encode_mask
from app.hybrid_landcover.core import BANDS, class_map, fuse_drafts, load_forest, predict_geotiff


def draft(shapes):
    return {"imageWidth": 6, "imageHeight": 5, "imagePath": "tile.png", "shapes": shapes}


def polygon(label, x0, x1):
    return {"label": label, "shape_type": "polygon", "points": [[x0, 0], [x1, 0], [x1, 5], [x0, 5]]}


def sam_mask(label, cells):
    array = np.zeros((5, 6), np.uint8)
    for x, y in cells:
        array[y, x] = 255
    data = io.BytesIO()
    Image.fromarray(array).save(data, format="PNG")
    return {"label": label, "points": [[0, 0], [5, 4]], "shape_type": "mask",
            "mask": base64.b64encode(data.getvalue()).decode("ascii")}


def test_draft_priority_shared_classes_conflicts_and_ignore():
    gpt = draft([polygon("water", 0, 2), polygon("wood", 1, 3)])
    sam = draft([sam_mask("crop", [(0, 0), (3, 0), (4, 0)]), sam_mask("crop", [(3, 0)]),
                 sam_mask("grass", [(4, 0)])])
    eligible = np.ones((5, 6), bool)
    eligible[0, 5] = False
    ignore = np.zeros_like(eligible)
    ignore[-1, 0] = True
    sam["starter_ignore"] = encode_mask(ignore)
    result, stats = fuse_drafts(gpt, sam, np.full((5, 6), 1, np.uint8), eligible)
    # Two classes claim x4 in the SAM draft; the RF can fill that former blank.
    np.testing.assert_array_equal(class_map(result)[0], [0, 254, 3, 4, 1, 255])
    assert class_map(result)[-1, 0] == 255
    assert sum(stats[k] for k in ("gpt_pixels", "sam_pixels", "rf_pixels", "conflict_pixels", "unclassified_pixels")) == stats["eligible_pixels"] == 28
    assert stats["sam_pixels"] == 1 and stats["conflict_pixels"] == 5
    assert "flags" not in gpt["shapes"][0]
    assert {s["hybrid_source"]["stage"] for s in result["shapes"]} == {"gpt", "deepseek_sam", "rf"}


def test_spectral_window_band_order_scale_and_invalid_pixels(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    raster = tmp_path/"reflectance.tif"
    transform = from_origin(500000, 3800000, 10, 10)
    # Bands are physically in reverse order on disk. Only their descriptions specify their identities.
    values = (np.arange(10 * 5 * 6, dtype=np.int16).reshape(10, 5, 6) * 9) + 100
    values[3, 2, 4] = -32768
    with rasterio.open(raster, "w", driver="GTiff", width=6, height=5, count=10, dtype="int16",
                       transform=transform, crs="EPSG:32650", nodata=-32768) as out:
        out.write(values[::-1])
        out.descriptions = tuple(reversed(BANDS))
        out.update_tags(reflectance_scale_factor="0.0001")
    roi = np.ones((3, 4), bool)
    roi[0, 0] = False
    pred, valid = predict_geotiff(raster, width=4, height=3, x=1, y=1, roi=roi, expected_crs="EPSG:32650",
                                  expected_transform=[10, 0, 500010, 0, -10, 3799990])
    assert valid.sum() == 10 and pred[0, 0] == pred[1, 3] == 255
    physical = values[:, 1:4, 1:5].transpose(1, 2, 0).astype(np.float32) * np.float32(0.0001)
    np.testing.assert_array_equal(pred[valid], load_forest().predict(physical[valid])[1].ravel())
    with pytest.raises(ValidationError):
        predict_geotiff(raster, width=4, height=3, x=3, y=2)
    with pytest.raises(ValidationError):
        predict_geotiff(raster, width=4, height=3, expected_transform=[10,0,9,0,-10,10])
