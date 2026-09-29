# Change Log

Newest entries last. Keep this file synchronized with roadmap phase/task
completion and GitHub issue comments.

## 2026-09-29

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
- All 73 tests passing; ruff clean; Sphinx docs building without warnings.

