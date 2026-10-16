# Setup prompt for OpenCode agent in jupyterhub28 (paste this to your agent)

You are setting up the infrastructure for a human-in-the-loop annotation+training side-project in a fresh LXD dev container (jupyterhub28). Gregory (the user) runs this container's code-server IDE; a Master's student (Kailey, forestry/ecology background, non-coder) runs a similar workspace in the same container and will be the primary operator of what you build. Work carefully, verify before claiming success, and ask the user when you need decisions or secrets.

## Context (what already exists—read before building)

- Project: ML microsite classification from drone imagery for precision aerial reforestation (IgniteBC × Flash Forest, UBC FRESH).
- GitHub org **UBC-FRESH** hosts everything: `fresh-ml4seeding` (code repo; `planning/` has all research notes), `fresh-ml4seeding-data` (DataLad data repo; large payloads served from a public S3 annex), `a58-microsite-sandbox` (minimal train/eval/infer quickstart).
- Read these first, they contain hard-won lessons you must not relearn the hard way: `planning/model_alternatives.md` (which model won and the exact training recipe), `planning/nope_map.md` (pre-labelling/upload pitfalls and the veto experiment), `AGENTS.md` and `planning/data-management.md` (git/DataLad/S3 rules).
- Current best model: U-Net ResNet34 + focal loss + TTA (test IoU ≈ 0.65 on the A58 v4 dataset). Checkpoint: `data/models/unet_resnet34_focal_best.pt` in the data repo. Training recipe: `scripts/experiments/train_terrain_ab.py` (RGB arm) or `code/train_unet_focal.py` in the sandbox repo.
- Roboflow workspace: **FRESH Workspace** (`gregory-paradis-ubc-ca`). Existing projects: `a58-microsites` (548 annotated), `a58-features`, `a58-sandbox` (750 raw frames, unannotated, Kailey's playground).

## Objective

Build a small, robust, scriptable **annotation flywheel** that Kailey can operate with single commands from a terminal:

1. **Sync**: download the current annotations + image lists from a chosen Roboflow project (REST API), into a local dataset directory with clear provenance (project, version/round, timestamp).
2. **Train**: train or fine-tune a fast pre-labeller model on the downloaded labels (default: the U-Net ResNet34 + focal recipe from the notes; optionally warm-start from the existing checkpoint).
3. **Prelabel**: run inference over the project's **unannotated** images and upload predictions back to Roboflow **as editable pre-labels**—then Kailey corrects them in the UI (much faster than painting from scratch).
4. **Loop bookkeeping**: track rounds (which images were human-verified in which round, which checkpoint pre-labelled which images, metrics per round) so the loop can run repeatedly and, later, be re-run wholesale for different "microsite" definitions or new model types (e.g., species-wise goodness scoring).

## Hard rules (learned the expensive way—violating these breaks trust and data)

- **NEVER overwrite existing human annotations.** Upload pre-labels only to images that have no annotations, with overwrite disabled. Before any upload, verify against the API that the target images are annotation-free, and report counts.
- **No secrets in repos or chat.** Ask the user for the Roboflow API key and store it in a gitignored `.env` as `ROBOFLOW_API_KEY`. Ask the user where AWS/S3 credentials live on THIS container (they exist on the other one; do not assume the same path).
- **Large files go to the DataLad data repo, never the code repo** (checkpoints, datasets, predictions, galleries). Small JSON metrics/state may live in git.
- **Provenance everywhere**: every checkpoint records (dataset version/round, recipe, metrics); every pre-label upload records (checkpoint id, model version, round).
- **Kailey-operable**: single, boring commands (`python loop.py sync|train|prelabel|status` or `make sync` etc.), loud progress output, obvious error messages, and a `RUNBOOK.md` written for a non-coder.
- Ask before any destructive or remote-mutating action beyond uploading pre-labels to unannotated images (e.g., creating new Roboflow projects, deleting anything, pushing to shared repos).

## Environment: verify first, assume nothing

Check and report before building: GPU presence (`nvidia-smi`; fall back to CPU-slow-but-working or ask the user where training should run), Python + key packages (torch, segmentation-models-pytorch, opencv, requests, datalad), git identity, internet access, Roboflow API reachability, DataLad availability and whether `datalad get` works against the public S3 annex. If something is missing, install into a virtualenv/conda env inside the workspace—never system-wide.

## Deliverables

1. `loop.py` (or `Makefile`) with subcommands: `sync`, `train`, `prelabel`, `status`, plus a `config.yaml` (project id, classes, thresholds, paths, round number).
2. `RUNBOOK.md` for Kailey: what each command does, what to do in Roboflow between commands, how to tell when a round is done.
3. `loop_state.json`: per-round record (images labelled, checkpoint path, metrics, upload counts, timestamps).
4. A convergence signal: after each `sync`, compute how much the human edited the pre-labels (e.g., fraction of objects added/deleted/edited, or pixel-level change rate) and print a trend table so the user can decide when to stop. Propose the metric, let the user confirm it.
5. Everything committed to a NEW repo in UBC-FRESH (ask the user for the repo name; public unless told otherwise), with data artifacts going to `fresh-ml4seeding-data` via DataLad.

## Acceptance test

Run one full dry loop on a small scale and show the user evidence: sync a handful of already-annotated images from an existing project, train briefly (a few epochs is fine for the dry run), prelabel a handful of UNANNOTATED images in a test project (ask the user which—`a58-sandbox` is the safe default), verify in the API that pre-labels landed as editable annotations on exactly those images and nothing else was touched, print the status table. Then hand over the RUNBOOK.

## First questions to ask the user (ask these before writing code)

1. Which Roboflow project is the working target for round 1 (existing one, or create a new one—if new, what name/type/license)?
2. Where do the Roboflow API key and (if needed) AWS/S3 credentials live on this container?
3. Repo name for the new tooling repo in UBC-FRESH?
4. Warm-start from `unet_resnet34_focal_best.pt` or train from scratch?
5. What convergence metric/threshold do you want to use to stop the loop?
