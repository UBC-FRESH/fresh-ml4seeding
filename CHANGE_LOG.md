# Change Log

Newest entries last. Keep this file synchronized with roadmap phase/task
completion and GitHub issue comments.

## 2026-09-29

**Workflow deviation notice:** Phases 0–3 were implemented directly on `main`
without following the UBC-FRESH phase/task/subtask workflow (no feature
branches, no parent/child issues, no PRs). This was a process error. From
Phase 4 onward, all work follows the strict workflow documented in
`AGENTS.md`: one phase = one parent issue + one feature branch + child task
issues + PR to `main`.

- Completed Phase 4 by merging PR #5, tagging `v0.1.0a1`, and creating the
  GitHub prerelease with checked artifacts:
  <https://github.com/UBC-FRESH/fresh-ml4seeding/releases/tag/v0.1.0a1>.
- Closed parent issue #3.
- Completed Phase 5 by converting the data repo to a DataLad dataset with
  git-annex (`text2git`), configuring the Arbutus S3 special remote
  (`object-arbutus.cloud.computecanada.ca`, bucket `ubc-fresh-ml4seeding-data`),
  annexing 691 files (11.81 GB), uploading all payloads to S3, wiring the
  publication dependency, and verifying fresh-clone + selective retrieval.
- Transferred 13.6 GB raw drone images, 1.4 GB orthomosaics, and 568 MB site
  visit photos from office desktop via Globus Connect Personal.
- Validated the package against real DJI ZenmuseP1 imagery: 484 valid images
  with GPS DMS coordinates, pose extraction (yaw/pitch/roll), coordinate
  conversion, and spacing estimation all working correctly.
- Closed parent issue #4.
- Completed Phase 6: cross-site validation on A58 and G15, retraining without
  aggressive filtering/oversampling, evaluation of focal loss and larger
  architectures (up to 138M parameters on RTX PRO 6000 Blackwell 98 GB GPU).
- Key finding: class imbalance is the fundamental bottleneck — "good" and
  "fair" classes remain unlearnable (IoU < 0.03) regardless of model size or
  loss function. Best model: focal loss, n_filters=128, test accuracy=0.932,
  FG mean IoU=0.319.
- Saved retrained model to data repo (`best_unet_focal_n128.pt`, 528 MB).
- Updated technical report with cross-site validation and retraining results
  (new Section 5.11, updated Sections 6.5 and 7).
- Closed parent issue #6.

- Created the fresh-ml4seeding repository as a re-implementation of the
  prototype ML4seeding pipeline with proper UBC-FRESH software engineering
  practices.
- Scaffolded the package with `pyproject.toml`, governance files (`AGENTS.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`), roadmap, changelog,
  and planning directory.
- Set up the `src/ml4seeding/` package structure with module stubs for
  orthomosaic analysis, pseudo-orthomosaic generation, tiling, preprocessing,
  U-Net model, training, and evaluation.
- Added Sphinx documentation skeleton, CI workflows, and release-artifact
  workflows following the UBC-FRESH pattern established by freshforge,
  modelwright, and fhops.
- Documented the DataLad data management plan in `planning/data-management.md`.
- Created GitHub repository at
  [UBC-FRESH/fresh-ml4seeding](https://github.com/UBC-FRESH/fresh-ml4seeding)
  and pushed initial scaffold.
- Implemented ML modules: `preprocessing.py` (image-mask matching, spatial
  splitting, filtering, oversampling), `metrics.py` (per-class IoU, mean IoU,
  foreground mean IoU, accuracy — numpy-only), `models.py` (U-Net architecture
  with lazy TensorFlow import), `losses.py` (weighted SCCE, Dice, combined
  loss), `training.py` (augmentation, data loading, preprocessing).
- Implemented core image processing modules: `orthomosaic.py` (metadata
  extraction, quality assessment, vegetation indices, sharpness, quicklook
  generation), `pseudo_ortho.py` (DJI EXIF GPS extraction, coordinate
  conversion, coverage-based selection, XMP pose extraction, GPS-placement
  mosaic with yaw correction), `tiling.py` (GeoTIFF tiling with overlap,
  invalid-pixel filtering, georeferenced metadata), `grid.py` (camouflage
  grid overlay with adaptive color sampling).
- Extended CLI with `ortho analyze`, `pseudo-ortho build`, `tile create`,
  and `grid add` commands.
- All 108 tests passing; ruff clean; Sphinx docs building without warnings.
- Created `UBC-FRESH/fresh-ml4seeding-data` as DataLad dataset (--no-annex)
  and added as git submodule at `external/fresh-ml4seeding-data`.
- Opened GitHub issues #1 (P5 DataLad) and #2 (P4 alpha hardening).
- Added API reference documentation (`docs/api.rst`) with autodoc for all
  10 modules.
- Added data management guide (`docs/guides/data-management.rst`).
- Added CLI usage examples (`docs/examples.rst`).
- Added example notebook (`examples/00_package_orientation.ipynb`) with
  synthetic data demos for all package modules.
- Expanded tiling and grid test coverage with GeoTIFF fixtures and edge cases.

