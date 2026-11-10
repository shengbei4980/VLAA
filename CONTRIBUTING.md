# Contributing to VLAA

VLAA supports local image annotation, visual-model proposal and review, human correction and confirmation, and confirmed-data export, training, evaluation, and bounded next-batch workflows. Dataset tasks include detection, segmentation, oriented boxes, and single-label classification; generic visual workflows currently cover detection and segmentation.

Read [AGENTS.md](AGENTS.md) for engineering boundaries and the [user guide](docs/USER_GUIDE.md) for the current product flow. Discuss additions that change task types, training engines, authentication, cloud deployment, or multi-user operation before implementation.

## Development environment

Use Python 3.12 and Node.js 20 or newer (22 LTS recommended). Use the repository-local `.venv`; install the application with `install.cmd` or `py -3.12 scripts/setup.py`. For backend tests, also install development dependencies into the same virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
```

Keep HTTP routes thin, put business rules in domain services, and access files through the storage layer or a named domain boundary. Store annotation geometry in original-image pixel coordinates. Database schema changes require an Alembic migration. Human-confirmed snapshots remain the input to formal workflow export, training, and evaluation.

## Verification

Run focused checks appropriate to the change. The standard Windows commands are:

```powershell
$env:PYTHONPATH = "backend"
.venv\Scripts\python.exe -m pytest backend/tests
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

Add regression coverage for meaningful behavior changes. Document model-dependent or hardware-dependent checks that were not run; connection tests do not establish annotation accuracy.

## Data and provenance

Do not commit datasets, model weights, API keys, local settings, runtime databases, user screenshots, customer material, or personal absolute paths. Keep upstream attribution and the scope of derived files intact.

See [NOTICE](NOTICE), [source provenance](source_snapshot.md), and the [upstream authorization record](docs/provenance/UPSTREAM_AUTHORIZATION.md). Public-release authorization for inherited material remains pending. Do not mark it approved without the required evidence or bypass release checks.
