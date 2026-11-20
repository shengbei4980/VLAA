import base64
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from app.dataset.annotation.masks import decode_mask, encode_mask
from app.hybrid_landcover import core, online, service


def inputs():
    rgb = service.png_bytes(np.zeros((8, 8, 3), dtype=np.uint8))
    with MemoryFile() as tmp:
        with tmp.open(driver="GTiff", width=10, height=10, count=10, dtype="int16",
                      transform=from_origin(500000, 4000000, 10, 10), crs="EPSG:32650") as ds:
            ds.write(np.full((10, 10, 10), 1500, dtype=np.int16))
            ds.update_tags(reflectance_scale_factor=0.0001)
            for i, band in enumerate(core.BANDS, 1):
                ds.set_band_description(i, band)
        tib = tmp.read()
    return {"rgb": ("tile.png", rgb, "image/png"), "raster": ("sr.tif", tib, "image/tiff")}


def shape_document(label, **shape):
    return {"imagePath": "tile.png", "imageWidth": 8, "imageHeight": 8,
            "shapes": [{"label": label, **shape}]}


def test_drafts_route_fuses_real_forest_and_preserves_ignore(client):
    files = inputs()
    primary = shape_document("water", shape_type="polygon", points=[[0,0], [4,0], [4,8], [0,8]])
    ignore = np.zeros((8, 8), dtype=bool)
    ignore[0, 0] = True
    primary["starter_ignore"] = encode_mask(ignore)
    secondary = shape_document("grass", shape_type="mask", starter_rle=encode_mask(np.indices((8,8))[0] < 4))
    for field, data in (("gpt_draft", primary), ("sam_draft", secondary)):
        files[field] = ("tile.json", json.dumps(data), "application/json")
    response = client.post("/api/hybrid/run", files=files, data={"mode":"drafts", "x":1, "y":2})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["stats"] == {"eligible_pixels":63, "gpt_pixels":31, "sam_pixels":16, "rf_pixels":16,
                               "conflict_pixels":0, "unclassified_pixels":0}
    assert base64.b64decode(result["labelme"]["imageData"]) == files["rgb"][1]
    assert np.array_equal(decode_mask(result["labelme"]["starter_ignore"]), ignore)
    assert Image.open(io.BytesIO(base64.b64decode(result["source_png"]))).getpixel((0,0)) == (0,0,0)
    secondary["imagePath"] = "another-tile.png"
    files["sam_draft"] = ("tile.json", json.dumps(secondary), "application/json")
    assert client.post("/api/hybrid/run", files=files, data={"mode":"drafts"}).status_code == 422


def test_online_route_sends_two_approved_images_and_segments_boxes(client, monkeypatch):
    files = inputs()
    endpoints = {
        "gpt_endpoint": json.dumps({"protocol":"responses", "api_base":"https://gpt.example/v1",
                                    "model":"gpt-test", "api_key":"gpt-test-secret"}),
        "deepseek_endpoint": json.dumps({"protocol":"chat_completions", "api_base":"https://ds.example/v1",
                                         "model":"ds-test", "api_key":"ds-test-secret"}),
    }
    body = {"mode":"online", "allow_remote":"true", **endpoints}
    assert client.post("/api/hybrid/run", files=files, data=body).status_code == 422
    config = {"enabled":True,"weights_available":True,"weights_path":"test-weights", "weights_sha256":"test-digest", "img_size":64,"device":"cpu"}
    monkeypatch.setattr(service, "freeze_sam_configuration", lambda settings: config)
    urls = []
    def respond(request):
        payload = json.loads(request.content)
        urls.append(str(request.url))
        if request.url.path.endswith("/responses"):
            assert request.headers["authorization"] == "Bearer gpt-test-secret"
            parts = payload["input"][0]["content"]
            assert len([p for p in parts if p["type"]=="input_image"]) == 1
            assert payload["store"] is False and payload["stream"] is False
            text = json.dumps({"objects":[{"class_id":"water","polygon":[[0,0],[4,0],[4,8],[0,8]],"evidence":"shore"}]})
            return httpx.Response(200,json={"status":"completed","output":[{"type":"message","role":"assistant","status":"completed",
                                                                            "content":[{"type":"output_text","text":text}]}]})
        assert request.headers["authorization"] == "Bearer ds-test-secret"
        assert len([p for p in payload["messages"][0]["content"] if p["type"]=="image_url"]) == 1
        text = json.dumps({"objects":[{"class_id":"grass","bbox":[0,0,8,4],"evidence":"grass patch"}]})
        return httpx.Response(200,json={"choices":[{"message":{"content":text},"finish_reason":"stop"}]})
    provider = online.OpenAICompatibleVisionProvider
    monkeypatch.setattr(online, "OpenAICompatibleVisionProvider",
                        lambda ep, **kw: provider(ep, transport=httpx.MockTransport(respond), **kw))
    class FakeSam:
        def predict(self, *, source, bboxes, imgsz, verbose, device):
            assert bboxes == [[0,0,8,4]] and Image.open(source).size == (8,8)
            return [SimpleNamespace(masks=SimpleNamespace(data=(np.indices((8,8))[0] < 4)[None,:,:]))]
    monkeypatch.setattr(online, "_model_for", lambda *args: FakeSam())
    result = client.post("/api/hybrid/run", files=files, data=body, headers={"X-VLAA-Hybrid-Run":"1"})
    assert result.status_code == 200, result.text
    assert urls == ["https://gpt.example/v1/responses", "https://ds.example/v1/chat/completions"]
    assert result.json()["stats"]["sam_pixels"] == 16
    assert result.json()["stats"]["rf_pixels"] == 16
    assert "test-secret" not in result.text



@pytest.mark.parametrize("width,height", [(1024,1024), (2048,16)])
def test_preserves_native_mother_tile_and_strip_dimensions(width, height):
    # The online proposal and local model must see the same grid at any aspect ratio.
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[-1, -1] = [10, 20, 30]
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[-1, -1] = 255
    np.testing.assert_array_equal(service.tile_pixels(service.png_bytes(rgb)), rgb)
    np.testing.assert_array_equal(service.tile_pixels(service.png_bytes(roi), roi=True), roi)
