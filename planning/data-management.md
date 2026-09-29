# Data Management Plan

This document describes the DataLad-managed data strategy for ml4seeding,
following the pattern established by [UBC-FRESH/femic](https://github.com/UBC-FRESH/femic)
(see `docs/guides/github-datalad-arbutus-pattern.rst` in the femic repo).

## Architecture

The project uses a **split-repo pattern**:

1. **Code repo** (`UBC-FRESH/fresh-ml4seeding`): Python package, tests, docs,
   CI. No large data files.
2. **Data repo** (`UBC-FRESH/fresh-ml4seeding-data`): DataLad dataset with
   git-annex for large binary files (drone imagery, annotation masks, model
   weights).
3. **S3 object storage** (Arbutus): holds the annexed payload objects.

The data repo is added as a git submodule at `external/fresh-ml4seeding-data`.

## S3 Configuration

| Setting | Value |
|---------|-------|
| Remote name | `arbutus-s3` |
| Endpoint | `object-arbutus.cloud.computecanada.ca` |
| Bucket | `ubc-fresh-ml4seeding-data` |
| Region | `ca-west-1` |
| Protocol | `https` |
| Port | `80` |
| Request style | `path` |
| Chunk size | `1 GiB` |
| Storage class | `STANDARD` |
| Public | `yes` |

## Data Repo Setup (Maintainer Runbook)

### Create the DataLad dataset

```bash
datalad create -c text2git fresh-ml4seeding-data
cd fresh-ml4seeding-data
```

### Configure git-annex

`.gitattributes`:

```
* annex.backend=MD5E
**/.git* annex.largefiles=nothing
* annex.largefiles=((mimeencoding=binary)and(largerthan=0))
data/**/*.tif -text
data/**/*.tiff -text
data/**/*.gpkg -text
data/**/*.keras -text
data/**/*.h5 -text
data/**/*.npy -text
```

### Add data files

```bash
mkdir -p data/raw_drone_images data/orthomosaics data/sites_visit_images data/models data/predictions
# Copy data files into the appropriate directories
datalad save -m "Add project data"
```

### Initialize S3 special remote

```bash
export AWS_ACCESS_KEY_ID=<key-id>
export AWS_SECRET_ACCESS_KEY=<secret-key>
export AWS_DEFAULT_REGION=ca-west-1

git annex initremote arbutus-s3 \
  type=S3 \
  encryption=none \
  bucket=ubc-fresh-ml4seeding-data \
  public=yes \
  publicurl=https://object-arbutus.cloud.computecanada.ca/ubc-fresh-ml4seeding-data \
  host=object-arbutus.cloud.computecanada.ca \
  protocol=https \
  port=80 \
  requeststyle=path \
  autoenable=true \
  chunk=1GiB \
  storageclass=STANDARD
```

### Wire publication dependency and push

```bash
git annex copy --to arbutus-s3 --all
git config remote.origin.datalad-publish-depends arbutus-s3
git push origin main
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
git -C external/fresh-ml4seeding-data annex enableremote arbutus-s3
datalad get -r external/fresh-ml4seeding-data/data
export ML4SEEDING_EXTERNAL_DATA_ROOT=$PWD/external/fresh-ml4seeding-data/data
```

## Data Inventory

| Dataset | Location | Files | Size |
|---------|----------|-------|------|
| Raw drone images | `data/raw_drone_images/a10_segment1/` | 557 DJI JPGs | 6.9 GB |
| Raw drone images | `data/raw_drone_images/a58_segment2/` | 112 DJI JPGs | 1.8 GB |
| Orthomosaics | `data/orthomosaics/` | 1 GeoTIFF | 1.3 GB |
| Site visit photos | `data/sites_visit_images/` | ~200 files | 213 MB |
| Model weights | `data/models/` | 3 checkpoints | 298 MB |
| Predictions | `data/predictions/` | 1 file | 16 KB |

**Total annexed:** 691 files, 11.81 GB

## Rules

- Do not commit large data files to the code repo.
- Do not commit model weights or prediction outputs to git directly.
- Use the DataLad data submodule for all binary data.
- Keep synthetic test fixtures small and self-contained in the code repo.
