# Data Management Plan

This document describes the DataLad-managed data strategy for ml4seeding,
following the pattern established by [UBC-FRESH/femic](https://github.com/UBC-FRESH/femic).

## Architecture

The project uses a **split-repo pattern**:

1. **Code repo** (`UBC-FRESH/fresh-ml4seeding`): Python package, tests, docs,
   CI. No large data files.
2. **Data repo** (`UBC-FRESH/fresh-ml4seeding-data`): DataLad dataset with
   git-annex for large binary files (drone imagery, annotation masks, model
   weights).

The data repo is added as a git submodule at `external/fresh-ml4seeding-data`.

## Data Repo Setup (Maintainer Runbook)

### Create the DataLad dataset

```bash
datalad create -c text2git fresh-ml4seeding-data
cd fresh-ml4seeding-data
```

### Configure git-annex

Add to `.gitattributes`:

```
* annex.backend=MD5E
**/.git* annex.largefiles=nothing
* annex.largefiles=((mimeencoding=binary)and(largerthan=0))
data/**/*.tif -text
data/**/*.tiff -text
data/**/*.png -text
data/**/*.gpkg -text
```

### Add data files

```bash
mkdir -p data/raw_drone_images data/orthomosaics data/tiles data/masks data/models
# Copy data files into the appropriate directories
datalad save -m "Add initial drone imagery and annotation masks"
```

### Create GitHub sibling

```bash
datalad create-sibling-github --publish-depends arbutus-s3 fresh-ml4seeding-data
```

Or for a simpler GitHub-only setup (no S3):

```bash
gh repo create UBC-FRESH/fresh-ml4seeding-data --private
git remote add origin https://github.com/UBC-FRESH/fresh-ml4seeding-data.git
git push -u origin main
git push origin git-annex
```

## Code Repo Integration

### Add the submodule

```bash
git submodule add https://github.com/UBC-FRESH/fresh-ml4seeding-data.git external/fresh-ml4seeding-data
```

### Environment variable

```bash
export ML4SEEDING_EXTERNAL_DATA_ROOT=$PWD/external/fresh-ml4seeding-data/data
```

### Collaborator bootstrap

```bash
git submodule update --init --recursive
# If using git-annex with S3:
git -C external/fresh-ml4seeding-data annex enableremote arbutus-s3
datalad get -r external/fresh-ml4seeding-data/data
export ML4SEEDING_EXTERNAL_DATA_ROOT=$PWD/external/fresh-ml4seeding-data/data
```

## Data Inventory

| Dataset | Location | Description |
|---------|----------|-------------|
| Raw drone images | `data/raw_drone_images/` | DJI JPEG files with EXIF GPS |
| Orthomosaics | `data/orthomosaics/` | GeoTIFF orthomosaics (A10, A58, G15) |
| Image tiles | `data/tiles/` | 512×512 PNG tiles |
| Annotation masks | `data/masks/` | 4-class segmentation masks |
| Model weights | `data/models/` | Trained U-Net checkpoints |
| Predictions | `data/predictions/` | Model prediction outputs |

## Rules

- Do not commit large data files to the code repo.
- Do not commit model weights or prediction outputs to git directly.
- Use the DataLad data submodule for all binary data.
- Keep synthetic test fixtures small and self-contained in the code repo.
