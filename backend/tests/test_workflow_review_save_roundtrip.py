"""Persist manual mask edits through reload, exports and confirmed snapshots."""
from copy import deepcopy
import io

import numpy as np
from PIL import Image
import pytest

from app.dataset.annotation.masks import MaskService, decode_mask, encode_mask
from app.dataset.snapshots import create_snapshot
from app.workflow.preview_renderer import render_preview
from test_workflow import workflow
from test_workflow_exports import native_mask
from test_workflow_labelme_export import export, exported_document, labelme, object_mask


@pytest.mark.parametrize("labelme", ["segment"], indirect=True)
def test_manual_mask_edits_survive_save_reload_export_and_confirmation(labelme):
    client, service, database, run = labelme
    original = run["reviews"][0]
    masks = MaskService(service.storage)
    edited = object_mask()
    edited[100:110, 8:20] = False
    edited[370:380, 3:12] = True
    added = np.zeros_like(edited)
    added[250:280, 5:20] = True
    roi = np.ones_like(edited)
    roi[:5] = False
    ignore = np.zeros_like(edited)
    ignore[200:220, 20:30] = True

    objects = deepcopy(original["payload"]["objects"])
    old_reference = objects[0]["mask_ref"]
    objects[0].update(mask=encode_mask(edited), mask_ref=masks.write(edited).mask_ref,
                      source="manual", provenance_json={"proposer": "vision-fixture"})
    objects.append({"object_id": "manually-added", "class_id": objects[0]["class_id"],
                    "type": "polygon", "mask": encode_mask(added), "mask_ref": masks.write(added).mask_ref,
                    "source": "manual", "decision": "accepted"})
    # Match the browser save contract: editable objects contain hydrated pixels,
    # while accepted annotations point at the newly uploaded managed mask files.
    prepared = {**original["payload"], "objects": objects,
                "annotations": [{key: value for key, value in item.items() if key != "mask"} for item in objects],
                "roi": encode_mask(roi), "ignore": encode_mask(ignore),
                "roi_mask_ref": masks.write(roi).mask_ref, "ignore_mask_ref": masks.write(ignore).mask_ref}
    response = client.put(f"/api/workflows/reviews/{original['id']}/draft", json={
        "expected_revision": original["revision"], "payload": prepared,
    })
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["revision"] == original["revision"] + 1
    assert saved["artifact_id"] != original["artifact_id"]

    reopened = client.get(f"/api/workflows/reviews/{original['id']}")
    assert reopened.status_code == 200, reopened.text
    reopened = reopened.json()
    assert reopened == saved
    restored_objects = reopened["payload"]["objects"]
    assert [item["object_id"] for item in restored_objects] == [objects[0]["object_id"], "manually-added"]
    assert restored_objects[0]["mask_ref"] != old_reference
    assert restored_objects[0]["source"] == "manual"
    assert restored_objects[0]["provenance_json"] == {"proposer": "vision-fixture"}
    for actual, expected in zip(restored_objects, (edited, added)):
        assert np.array_equal(decode_mask(actual["mask"]), expected)
        assert np.array_equal(masks.decode(actual["mask_ref"]), expected)
    assert np.array_equal(masks.decode(reopened["payload"]["roi_mask_ref"]), roi)
    assert np.array_equal(masks.decode(reopened["payload"]["ignore_mask_ref"]), ignore)

    exported = export(client, reopened)
    assert exported.status_code == 201, exported.text
    document = exported_document(client, exported.json())
    for actual, expected in zip(document["shapes"], (edited, added)):
        assert np.array_equal(native_mask(actual, width=80, height=400), expected)
    image_path = service.storage.image_path(prepared["dataset_id"], prepared["image_id"] + ".png")
    png = render_preview(service.storage, image_path, reopened["payload"], run["profile_snapshot"]["class_rules"])
    with Image.open(io.BytesIO(png)) as preview:
        assert preview.size == (80, 400)
        assert preview.getpixel((7, 375))[:3] != (255, 255, 255)
        assert preview.getpixel((10, 105))[:3] == (255, 255, 255)

    confirmed = client.post(f"/api/workflows/reviews/{original['id']}/decide", json={
        "expected_revision": reopened["revision"], "decision": "approved",
    })
    assert confirmed.status_code == 200, confirmed.text
    with database.session_factory() as session:
        snapshot = create_snapshot(session, service.storage, prepared["dataset_id"], image_ids=[prepared["image_id"]])
        image = snapshot["manifest"]["images"][0]
        expected_objects = {objects[0]["object_id"]: edited, "manually-added": added}
        assert {item["object_id"] for item in image["annotations"]} == set(expected_objects)
        for actual in image["annotations"]:
            assert np.array_equal(masks.decode(actual["mask_ref"]), expected_objects[actual["object_id"]])
        assert np.array_equal(masks.decode(image["roi_mask_ref"]), roi)
        assert np.array_equal(masks.decode(image["ignore_mask_ref"]), ignore)
