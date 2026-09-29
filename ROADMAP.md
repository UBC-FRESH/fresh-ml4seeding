# fresh-ml4seeding Roadmap

This roadmap is the current project plan and issue tracker map. Keep it
synchronized with GitHub issues, planning notes, pull requests, and
`CHANGE_LOG.md`.

## Issue Tracker Map

| Phase | Parent issue | Branch | Status |
|-------|-------------|--------|--------|
| P0 Bootstrap scaffold | — | `main` | Complete (pre-workflow) |
| P1 Core image processing | — | `main` | Complete (pre-workflow) |
| P2 U-Net model and training | — | `main` | Complete (pre-workflow) |
| P3 CLI and evaluation | — | `main` | Complete (pre-workflow) |
| P4 v0.1.0a1 alpha release | #3 | `main` | Complete |
| P5 Populate DataLad data repo | #4 | `main` | Complete |

## Phase 0: Bootstrap Scaffold

Goal: establish ml4seeding as a public package-backed UBC-FRESH project with
strict governance, planning, docs, CI, and a minimal importable Python package.

- [ ] P0.1 Governance and planning scaffold
  - [ ] Add public governance files.
  - [ ] Add roadmap and changelog.
  - [ ] Add data management planning note.
  - [ ] Document strict issue and roadmap workflow in `AGENTS.md`.
- [ ] P0.2 Python package and CLI skeleton
  - [ ] Add package metadata and dependency extras.
  - [ ] Add minimal package modules and CLI.
  - [ ] Add focused tests.
- [ ] P0.3 Docs, CI, and release-artifact scaffold
  - [ ] Add Sphinx configuration and docs pages.
  - [ ] Add CI workflow for Python 3.11 and 3.12.
  - [ ] Add docs Pages workflow.
  - [ ] Add release artifact workflow.
- [ ] P0.4 Phase closeout and verification
  - [ ] Run local acceptance commands.
  - [ ] Update roadmap and changelog closeout notes.

## Phase 1: Core Image Processing

Goal: re-implement orthomosaic analysis, pseudo-orthomosaic generation, and
image tiling as tested package modules.

- [ ] P1.1 Orthomosaic analysis module
- [ ] P1.2 Pseudo-orthomosaic generation module
- [ ] P1.3 Tiling module
- [ ] P1.4 Grid overlay module
- [ ] P1.5 Tests and phase closeout

## Phase 2: U-Net Model and Training Pipeline

Goal: re-implement the U-Net segmentation model, data preprocessing, training
pipeline, and evaluation metrics as tested package modules.

- [ ] P2.1 U-Net model definition
- [ ] P2.2 Data preprocessing (matching, splitting, filtering, oversampling)
- [ ] P2.3 Loss functions (SCCE, weighted SCCE, Dice, combined)
- [ ] P2.4 Training pipeline with augmentation
- [ ] P2.5 Evaluation metrics
- [ ] P2.6 Tests and phase closeout

## Phase 3: CLI and Evaluation Tools

Goal: expose all pipeline stages through the `ml4seeding` CLI.

- [ ] P3.1 CLI for image processing stages
- [ ] P3.2 CLI for training and evaluation
- [ ] P3.3 Tests and phase closeout

## Phase 4: Documentation, Examples, and Public Alpha Hardening

Goal: harden docs, examples, tests, and CI for the first public alpha release.

- [ ] P4.1 Example notebooks using the package
- [ ] P4.2 API documentation
- [ ] P4.3 CI and release hardening
- [ ] P4.4 Phase closeout and v0.1.0a1 release

## Phase 5: Data Management and DataLad Integration

Goal: set up the DataLad-managed data submodule and document the data
management workflow.

- [ ] P5.1 Create DataLad data repository
- [ ] P5.2 Add data submodule to code repo
- [ ] P5.3 Document data management workflow
- [ ] P5.4 Phase closeout

## Current Next Steps

Phases 0–5 are complete. The package is validated against real A10 Segment 1
drone imagery (484 valid DJI images with GPS). The DataLad data repo is fully
operational with git-annex + Arbutus S3.

The immediate next steps are:

1. Re-download truncated data from OneDrive (A58 raw images, A58 GeoTIFF).
2. Download A10 and G15 orthomosaics from OneDrive.
3. Build pseudo-orthomosaic from A10 raw images using the package.
4. Begin Phase 6: cross-site validation and model evaluation on A58/G15.
5. Update the technical report with actual code repo and data availability.
