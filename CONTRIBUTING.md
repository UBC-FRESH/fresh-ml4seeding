# Contributing

Thank you for your interest in contributing to ml4seeding. This document
describes the development workflow and standards for this project.

## Development Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Local Verification

Before submitting a pull request, run the full verification suite:

```bash
python -m ruff check .
python -m pytest
sphinx-build -b html docs _build/html -W
python -m build
twine check dist/*
```

## Code Style

- Follow [PEP 8](https://pep8.org/) with a 100-character line limit.
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Write docstrings for all public functions and classes.
- Prefer type hints for function signatures.

## Testing

- Write tests for all new functionality.
- Tests must not require private data, GPUs, or network access.
- Use synthetic fixtures for image data.
- Run `python -m pytest` before committing.

## Documentation

- Update Sphinx documentation for any public API changes.
- Keep `README.md` concise and current.
- Update `ROADMAP.md` and `CHANGE_LOG.md` for phase-level changes.

## Issue and PR Workflow

This project follows the UBC-FRESH phase/task/subtask workflow. See
`AGENTS.md` for the full workflow contract.

## Public-Repo Hygiene

Do not commit private project data, raw drone imagery, annotation masks,
model weights, or machine-specific paths. Use the DataLad data submodule
for large files.
