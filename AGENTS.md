# AGENTS.md

This file is the working contract for AI coding agents in this repository.

## Project Purpose

`ml4seeding` exists to provide a reproducible, well-tested machine learning
pipeline for microsite classification from high-resolution drone imagery,
supporting precision aerial reforestation in British Columbia.

The goal is to replace the prototype Jupyter notebook pipeline from
[UBC-FRESH/ML4seeding](https://github.com/UBC-FRESH/ML4seeding) with a proper
package that is testable, documentable, and maintainable.

## Current Repo State

This repository is at the Phase 0 bootstrap stage. It contains:

- `README.md`: concise public overview and current status.
- `ROADMAP.md`: phase/task roadmap and issue tracker map.
- `CHANGE_LOG.md`: append-only project narrative.
- `planning/`: focused design notes and research records.
- `pyproject.toml`: package metadata and optional dependency groups.
- `src/ml4seeding/`: importable package code.
- `tests/`: package-backed tests.
- `docs/`: Sphinx documentation.
- `examples/`: example notebooks and scripts.
- `.github/workflows/`: CI, docs, and release-artifact checks.
- `external/`: DataLad data submodule (not tracked in this repo).
- `tmp/`: ignored local working area.

## Data Management

Large data files (drone imagery, annotation masks, model weights) are managed
through a DataLad-managed git submodule at `external/fresh-ml4seeding-data`.

Rules:

- Keep `tmp/`, `local/`, `data/private/`, and `outputs/` ignored.
- Do not commit raw drone imagery, annotation masks, model weights, or
  machine-specific paths to the code repo.
- Tracked examples and tests must use synthetic or public-safe fixtures.
- Record provenance for every external dataset, model checkpoint, and
  evaluation result.
- See `planning/data-management.md` for the full DataLad setup guide.

## Working Principles

- Read `AGENTS.md`, `ROADMAP.md`, and `CHANGE_LOG.md` before making
  project-shaping changes.
- Keep CLI commands thin wrappers over Python APIs.
- Keep domain science in domain functions; keep the package focused on
  image processing, model definition, training, and evaluation.
- Prefer structured records and parsers over ad hoc string handling.
- Emit explicit diagnostics for unsupported formats, missing data, invalid
  configurations, and failed validation.
- Preserve uncertainty. A model prediction is only as strong as its declared
  inputs, training data, and verification evidence.
- Keep public repo content clean of private, irrelevant, or unpublished
  references.
- Keep changes scoped to the active roadmap phase and issue.

## Planning Workflow

This repo follows the UBC-FRESH phase/task/subtask workflow:

- `ROADMAP.md` is the current plan and issue tracker map.
- One roadmap phase maps to one GitHub parent issue and one feature branch.
- One roadmap task maps to one child issue linked from the parent issue body.
- Subtasks usually stay as checklist items inside the child issue body.
- Use at most three issue levels: phase, task, implementation subtask.
- Record issue numbers beside roadmap phases and tasks once created.
- Keep `ROADMAP.md`, `CHANGE_LOG.md`, planning notes, issue bodies, and PR
  descriptions synchronized.
- Open a PR from the phase branch to `main` only after phase tasks, tests,
  docs, and closeout notes are complete or explicitly deferred.

## Strict Development Workflow

- One active roadmap phase should generally correspond to one GitHub parent
  issue and one feature branch.
- Create or activate the GitHub parent issue before starting a roadmap phase.
- Create the feature branch from current `main` for that parent issue.
- Create child issues for roadmap tasks under the parent issue.
- Work child issues one at a time where practical, usually in roadmap order.
- Keep `ROADMAP.md`, `CHANGE_LOG.md`, and issue comments synchronized as task
  state changes.
- Open a PR from the phase branch back to `main` when the parent issue's child
  issues are complete or explicitly deferred.
- Close the parent issue only after the PR has merged back to `main`.

## Verification

Default local checks:

```bash
python -m ruff check .
python -m pytest
sphinx-build -b html docs _build/html -W
python -m build
twine check dist/*
```

Default CI must not require private project data, drone imagery, annotation
masks, model weights, GPUs, credentials, or network downloads beyond package
installation.
