# fresh-ml4seeding

`ml4seeding` is an early public-alpha Python package for microsite classification
from high-resolution drone imagery to support precision aerial reforestation.

Developed as part of the Ignite BC Flash Forest project (Award IGNITE-2025-1345),
this package re-implements and extends the prototype pipeline from
[UBC-FRESH/ML4seeding](https://github.com/UBC-FRESH/ML4seeding) with proper
software engineering practices: package structure, testing, documentation, CI,
and reproducible workflows.

## Statement of Need

Drone-based aerial seeding requires precise identification of microsites where
seed germination and seedling establishment are most likely. Manual assessment
of harvested cut blocks is expensive, slow, and often impossible in steep or
remote terrain. Automated microsite classification from drone imagery enables
precision seed deployment — directing seeds only to locations with high
establishment probability.

## Current Alpha Scope

Supported in `0.1.0a1`:

- Python package skeleton using `src/` layout;
- Orthomosaic analysis (metadata extraction, quality assessment, quicklook generation);
- Pseudo-orthomosaic generation from raw drone images (GPS stitching with yaw correction);
- Image tiling with overlap, invalid-pixel filtering, and georeferenced metadata;
- Grid overlay for annotation quality control;
- Data preprocessing (image-mask matching, spatial splitting, filtering, oversampling);
- U-Net semantic segmentation model definition;
- Training pipeline with configurable augmentation, loss functions, and early stopping;
- Evaluation metrics (per-class IoU, mean IoU, foreground mean IoU);
- CLI entry point (`ml4seeding`);
- Sphinx documentation;
- CI and release-artifact workflows.

Not supported yet:

- Cross-site transfer learning or evaluation;
- SAM-based annotation acceleration;
- Real-time inference;
- Flight path optimization integration;
- Multispectral band support;
- Stable public APIs.

## Install

```bash
python -m pip install ml4seeding==0.1.0a1
```

For development:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Run the local checks:

```bash
python -m ruff check .
python -m pytest
sphinx-build -b html docs _build/html -W
python -m build
twine check dist/*
```

## Data

Drone imagery, annotation masks, and trained model weights are managed through a
DataLad-managed data repository at
[UBC-FRESH/fresh-ml4seeding-data](https://github.com/UBC-FRESH/fresh-ml4seeding-data).
See the
[data management guide](https://ubc-fresh.github.io/fresh-ml4seeding/guides/data-management/)
for setup instructions.

## Command Line

```bash
ml4seeding --version
ml4seeding info
ml4seeding ortho analyze ORTHO_PATH --out-dir OUTPUT_DIR
ml4seeding pseudo-ortho build INPUT_DIR --out-dir OUTPUT_DIR
ml4seeding tile ORTHO_PATH --out-dir OUTPUT_DIR [--tile-size 512] [--overlap 128]
ml4seeding train CONFIG_PATH [--epochs 50] [--batch-size 4]
ml4seeding evaluate MODEL_PATH --image-dir IMAGES --mask-dir MASKS
```

## Roadmap

Near-term phases are tracked in `ROADMAP.md`:

- Phase 0: Bootstrap package, governance, docs, and automation scaffold.
- Phase 1: Core image processing (orthomosaic analysis, pseudo-ortho, tiling).
- Phase 2: U-Net model and training pipeline.
- Phase 3: CLI and evaluation tools.
- Phase 4: Documentation, examples, and public alpha hardening.
- Phase 5: Data management and DataLad integration.

Development follows the UBC-FRESH phase/task/subtask workflow.

## Public-Repo Hygiene

Do not commit private project data, raw drone imagery, annotation masks, model
weights, or machine-specific paths. Keep scratch material under ignored local
paths such as `tmp/`, `local/`, `data/private/`, or `outputs/`. Large data files
are managed through the DataLad data submodule (see `external/`).

## Related Projects

- [ML4seeding](https://github.com/UBC-FRESH/ML4seeding) — original prototype (archived)
