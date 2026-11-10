# Vision-Language Automated Annotation (VLAA)

[中文 README](README.md) · [Installation (Chinese)](docs/INSTALL.md) · [User guide (Chinese)](docs/USER_GUIDE.md)

VLAA is a local image annotation workbench. A vision-language model proposes objects, an independently configured visual reviewer checks them, and a human edits and confirms the results. Confirmed data can be exported or used for local model training, evaluation, and another annotation batch.

Reusable class and model profiles support different subjects, such as tower components, street scenes, or satellite RGB images. Each project can have its own output root.

**v0.2.0 is a source installation package.** Python, Node.js, vision services, and model weights are configured separately.

## Main features

- Vision-language proposals, independent checks, bounded repairs, and an explicit human confirmation step.
- Remote APIs and local HTTP vision services using a shared image contract.
- Classes defined before annotation, with class descriptions and global task instructions.
- Detection and instance-segmentation visual workflows; dataset annotation and YOLO training also support oriented boxes and image classification.
- SAM-assisted segmentation with configured weights; original-pixel masks, brush/eraser, new/add/subtract polygons, eligible polygon vertex editing, undo, and redo.
- Single-image LabelMe JSON and original-size annotation PNG export; saved batch results can be exported as a ZIP with a manifest.
- Project-root output directories, per-run folders, training/evaluation artifacts, and reusable profiles.
- Workflow history deletion and restoration. Active work must be cancelled before its history entry is removed.

LLM checks do not guarantee correct labels. Review the entire image, repair errors, and confirm the result before using it for training. Segmentation workflows use the configured SAM model to produce object contours from proposals.

## Install and launch on Windows

Use **Python 3.12** and **Node.js 20 or newer**; Node.js 22 LTS is recommended. The first installation downloads dependencies from the internet.

1. Open the [GitHub repository](https://github.com/shengbei4980/VLAA), choose **Code → Download ZIP**, and extract it, or clone the repository with Git.
2. Double-click `install.cmd` in the project root.
3. Once installation succeeds, double-click `start.cmd`.
4. Open [the local workbench](http://127.0.0.1:5173/) if the browser does not open automatically.

Alternatively, in PowerShell from the project root:

```powershell
py -3.12 scripts/setup.py
py -3.12 run.py
```

The installer creates or reuses `.venv`, installs `backend/requirements.txt`, and runs `npm ci` in `frontend`. `run.py` starts both services using those dependencies. Keep the launcher window open; use Ctrl+C to stop the services it started after your current work has finished.

## First workflow

1. In **Settings → Vision model access**, configure and enable both the proposer and the independent reviewer. Choose remote API or local HTTP, enter the model name, API Base URL, and any required API key. Save and test the image/coordinate contract.
2. For segmentation, configure an existing local SAM weight file in **SAM assistance**. A review-only fallback cannot supply authoritative workflow masks.
3. Open the generic image workflow and choose **Annotate and check with a vision LLM**.
4. Select images; create a class/model profile, add the intended classes and instructions, and select a project root.
5. Preview image scope, destination endpoints, budget, and output paths, then launch.
6. Open each image at the human review step. Repair masks, map or reject unknown candidates, resolve review issues, save the draft, and confirm the whole image.
7. Use **Export LabelMe JSON** for external editing, or return to the run and use the batch JSON/PNG ZIP export. Batch export reads saved reviews, so save your edits first.

LabelMe JSON includes the original image. Mask shapes require LabelMe 5.8.0 or newer. Draft exports retain their pending review state; formal workflow exports and training require confirmed data. COCO RLE and LabelMe masks preserve holes and disconnected components; YOLO polygons may be unable to represent them faithfully.

## Storage and model boundaries

The frontend binds to `127.0.0.1:5173` and the backend to `127.0.0.1:8000` by default. Workbench data is local. Selecting a remote vision endpoint sends the selected images to that endpoint, as shown in the launch preview.

Project selection happens before a run. JSON, PNG, model, and evaluation outputs inherit the selected project root and are separated by run ID. API keys, datasets, model weights, and runtime records are excluded from the source package.

Local training and evaluation use Ultralytics YOLO. Other engines require their own adapters. A configured HTTP vision service must implement the expected image and coordinate contract; a successful connection test alone does not establish annotation quality.

## Provenance and release status

VLAA is customized from [YoloWebAgentStarter](https://github.com/fengzhizi715/YoloWebAgentStarter). Source attribution and third-party licenses are retained.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and the [upstream authorization record](docs/provenance/UPSTREAM_AUTHORIZATION.md). Authorization for public release of inherited material remains **pending**. Source installation instructions are provided here; a formal tag/release still requires that record to be completed and release checks to pass. Source availability does not resolve the pending record, and the local MIT file does not itself establish relicensing rights for all inherited code.
