# Release Notes

## v0.1.0a1 (2026-09-29)

First public alpha release of ml4seeding — machine learning for precision
aerial seeding: microsite classification from drone imagery.

### What is ml4seeding?

ml4seeding is a Python package for microsite classification from
high-resolution drone imagery to support precision aerial reforestation.
It re-implements the prototype pipeline from
[UBC-FRESH/ML4seeding](https://github.com/UBC-FRESH/ML4seeding) with proper
software engineering practices: package structure, testing, documentation,
CI, and reproducible workflows.

### Implemented Scope

**Image Processing:**
- Orthomosaic analysis: metadata extraction, quality assessment, vegetation
  indices (ExG, VARI, GRVI), sharpness (Laplacian variance), edge density,
  quicklook PNG generation, JSON report
- Pseudo-orthomosaic generation: DJI EXIF GPS extraction, equirectangular
  coordinate conversion, coverage-based image selection, XMP pose extraction,
  GPS-placement mosaic with yaw correction and alpha blending
- GeoTIFF tiling: fixed-size tiles with overlap, invalid-pixel filtering,
  valid bounding box estimation, georeferenced JSON metadata
- Camouflage grid overlay: adaptive color sampling for annotation quality
  control

**Machine Learning:**
- Data preprocessing: image-mask matching, spatial splitting (by X coordinate),
  trivial tile filtering, minority class oversampling
- U-Net semantic segmentation: 4-class, 512×512 input, 8.65M parameters,
  encoder-decoder with skip connections, batch normalization, dropout
- Loss functions: sparse categorical cross-entropy, weighted SCCE, Dice loss,
  combined loss (lazy TensorFlow import — install with `ml4seeding[ml]`)
- Evaluation metrics: per-class IoU, mean IoU, foreground mean IoU, accuracy
  (numpy-only, no TF required)
- Training pipeline: geometric augmentation, photometric augmentation,
  data loading (lazy TensorFlow import)

**CLI:**
- `ml4seeding --version` / `ml4seeding info`
- `ml4seeding ortho analyze ORTHO_PATH`
- `ml4seeding pseudo-ortho build INPUT_DIR`
- `ml4seeding tile create IMAGE_PATH`
- `ml4seeding grid add TILES_DIR META_DIR`

**Documentation:**
- Sphinx documentation with API reference (autodoc for all 10 modules)
- Data management guide (DataLad submodule pattern)
- CLI usage examples
- Example notebook with synthetic data demos

**Testing:**
- 108 tests passing
- Ruff linting clean
- Sphinx docs build with `-W` (zero warnings)
- CI on Python 3.11 and 3.12

### Alpha Boundaries

This is an early alpha release. Not yet supported:

- Cross-site transfer learning or evaluation
- SAM-based annotation acceleration
- Real-time inference
- Flight path optimization integration
- Multispectral band support
- Stable public APIs (breaking changes possible in future releases)

### Installation

```bash
pip install ml4seeding==0.1.0a1
```

For ML training (requires TensorFlow):

```bash
pip install ml4seeding[ml]==0.1.0a1
```

### Data

Drone imagery, annotation masks, and model weights are managed through the
DataLad data repository at
[UBC-FRESH/fresh-ml4seeding-data](https://github.com/UBC-FRESH/fresh-ml4seeding-data)
(private, contact UBC FRESH Lab for access).

### Acknowledgements

Developed as part of the Ignite BC Flash Forest project (Award
IGNITE-2025-1345) by the UBC FRESH Lab.
