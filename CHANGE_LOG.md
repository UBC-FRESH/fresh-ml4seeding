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
