# Stereo Depth & Terrain Geometry Assessment

**Status:** Research assessment  
**Date:** 2026-10-04  
**Question:** Can stereo-derived depth and terrain geometry improve our ML-based microsite classification?

---

## 1. Project Assessment

### Current Setup

- **Imagery:** Aerial drone imagery (DJI ZenmuseP1, 8192×5460, 35mm lens, F4.0)
- **Camera metadata:** GPS position, altitude, gimbal pose (yaw/pitch/roll) for each image
- **Task:** Semantic segmentation for microsite classification (good/fair/poor)—binary in practice (microsite vs. background)
- **Current model:** SegFormer-B2 achieving Test IoU 0.622 on 414 images
- **Challenge:** Extreme class imbalance (7.7% microsite), subtle spectral differences

### Data Availability

| Data | Available | Format | Notes |
|------|-----------|--------|-------|
| Raw drone images | ✅ Yes | 750 A58 + 557 A10 | DJI JPG with EXIF GPS |
| Camera calibration | ✅ Yes | EXIF (35mm, F4.0) | Fixed focal length |
| GPS positions | ✅ Yes | EXIF (lat/lon/alt) | WGS-84 |
| Camera pose | ✅ Yes | XMP (yaw/pitch/roll) | DJI XMP metadata |
| Orthomosaics | ✅ Yes | GeoTIFF | A10, A58, G15 |
| Annotated masks | ✅ Yes | PNG | 414 images (v4) |
| Stereo pairs | ⚠️ Indirect | Multi-view overlap | Not a stereo rig, but overlapping views |
| DSM/DTM | ❌ No |—| Could be generated from photogrammetry |
| Depth maps | ❌ No |—| Could be estimated from stereo pairs |

### Key Insight

Our drone images are from a **single camera**, not a stereo pair. However, the DJI ZenmuseP1 is a professional mapping camera that captures images with high overlap (typically 80% forward, 70% side) for photogrammetry. This means consecutive images have significant overlap and can be used for **multi-view stereo depth estimation** or **photogrammetric DSM generation**.

---

## 2. Literature Review

### TerrainNet (Meng et al., 2023)

**Relevance:** High—off-road terrain perception with RGB + stereo depth.

**Key findings:**
- Stereo depth provides significant gains: "Models with stereo depths as additional input outperform their RGB-only counterparts by a large margin"
- For off-road environments, objects can have similar appearances but different sizes—hard to estimate depth from color alone
- Uses depth completion to fill in missing stereo depth
- Multi-view RGB + stereo depth for joint semantic and geometric terrain prediction

**Implication for us:** Stereo depth could help distinguish microsites that look similar in RGB but have different geometric properties (e.g., exposed soil on mounds vs. depressions).

### Valada et al. (2016)—Multispectral Forest Segmentation

**Relevance:** High—forest scene segmentation with RGB + NIR + depth fusion.

**Key findings:**
- RGB + NIR + Depth fusion outperforms RGB alone for forest segmentation
- Depth alone performs poorly, but fused with RGB it helps
- "Late-fused-convolution" approach works well—fuse features from different modalities at the feature level
- Their dataset is forest environments (similar to ours)

**Implication for us:** Feature-level fusion of RGB and terrain geometry could improve our microsite classification. The forest environment is similar to our cut blocks.

### SNE-RoadSeg (Fan et al., 2020)

**Relevance:** Medium—freespace detection with surface normals from depth.

**Key findings:**
- Surface normals from depth improve segmentation
- The SNE module infers surface normals from dense depth/disparity images
- RGB + surface normal fusion outperforms RGB alone
- Surface normals are more informative than raw depth for ground plane detection

**Implication for us:** Surface normals (computed from a DSM) could help identify flat vs. sloped microsites. Exposed soil on flat ground vs. slopes may have different spectral signatures.

### ORFD / OFF-Net (Min et al., 2022)

**Relevance:** Medium—off-road freespace detection with RGB + LiDAR surface normals.

**Key findings:**
- Transformer fusion of RGB and surface normals
- Cross-attention dynamically fuses the two modalities
- Surface normals from LiDAR, not stereo (important distinction)
- RGB + surface normal fusion outperforms either alone

**Implication for us:** The transformer fusion approach could work for our SegFormer model. But note: their geometry comes from LiDAR, not stereo photogrammetry.

---

## 3. Feasibility Analysis

### Can we get stereo depth from our drone images?

**Yes, but not in the traditional sense.** Our drone images are from a single camera, not a stereo pair. However, the DJI ZenmuseP1 captures images with high overlap (80% forward, 70% side) for photogrammetry. This means:

1. **Multi-view stereo is possible**—consecutive images from different positions can be used for depth estimation
2. **Photogrammetric DSM is the practical approach**—use the existing orthomosaic pipeline to generate a DSM, then extract terrain features

### Practical approach: DSM + terrain features

Instead of real-time stereo depth estimation (which would require significant compute and calibration), we can:

1. **Generate a DSM** from the A58 drone images using photogrammetry (Agisoft Metashape or similar)
2. **Extract terrain features** from the DSM:
   - Slope (degrees from horizontal)
   - Aspect (direction of slope)
   - Curvature (convex/concave)
   - Roughness (local elevation variance)
   - Relative elevation (height above local mean)
3. **Fuse RGB + terrain features** using a multi-input model
4. **Compare** RGB-only vs. RGB+terrain models

This leverages the photogrammetric data we already have and is much more practical than real-time stereo depth estimation.

---

## 4. Experimental Result (2026-10-04)

**RGB + Terrain Early Fusion (SegFormer-B2 with 7-channel input):**
- Test IoU: 0.581
- Test Dice: 0.692
- Best val IoU: 0.598

**RGB-only Baseline (SegFormer-B2):**
- Test IoU: 0.622
- Test Dice: 0.730
- Best val IoU: 0.617

**Result:** The terrain fusion model performed WORSE than the RGB-only baseline. The terrain features (elevation, slope, curvature, roughness) from the COLMAP sparse reconstruction did not improve the model's performance.

**Analysis:**
- The COLMAP sparse reconstruction is too coarse (0.1m resolution, 102×109 grid) to provide useful terrain features for the 1024×1024 training images
- The terrain features are noisy and not well-aligned with the training images
- The model couldn't learn to use the terrain features effectively
- The terrain features may not be relevant for microsite classification (which is based on spectral properties of the soil, not terrain geometry)

**Conclusion:** For this microsite classification task, RGB spectral features are sufficient. Terrain geometry from sparse photogrammetric reconstruction does not improve the model. This is a useful negative result for the paper.

### Follow-up: Properly Aligned Per-Image Terrain Fusion (2026-10-06)

The 2026-10-04 negative result was potentially invalid: the DSM was built from a
COLMAP reconstruction of frames 0001–0050 only, while the v4 training images come
from frames 0203–0750—**zero overlap**, so terrain and pixels never corresponded.
This follow-up eliminated the alignment confound:

- **Reconstruction:** COLMAP on all 750 flight frames (1024×682), sequential
  matching (overlap 25 + quadratic). 750/750 registered in one model,
  **715,225 points** (9× denser), final BA cost 0.22 px.
- **Ground-plane DEM:** PCA plane-aligned (normal oriented toward cameras),
  637×550 grid. Features: relative elevation, slope, curvature, roughness.
  *(Erratum 2026-10-07: grid/threshold figures quoted here were in COLMAP's
  arbitrary local units; GPS alignment shows 1 unit = 8.79 m, so the grid was
  really ~17.6 cm cells over ~112×97 m, and feature windows were ~9× larger
  than stated. The A/B conclusion is unaffected—inputs were z-score
  normalized, and alignment correctness does not depend on scale.)*
- **Per-image rendering:** ray-cast every pixel against the DEM with 3
  fixed-point parallax-correction iterations, stretched to v4 geometry
  (1024×1024). 548/548 images rendered, **97% mean valid-pixel coverage**;
  QC montage confirms stump/slash features align with image content.
- **A/B protocol:** SegFormer-B2, identical images/recipe both arms;
  terrain arm = 8 channels (RGB + 4 features + validity mask).

| Arm | Test IoU | Test Dice | Best val IoU |
|-----|----------|-----------|--------------|
| RGB-only (control) | 0.5837 ± 0.258 | 0.694 | 0.6072 (ep 32) |
| RGB + terrain | 0.5885 ± 0.249 | 0.703 | 0.6003 (ep 52) |

**Result:** Δ IoU = +0.005—a wash, well within per-image noise (σ ≈ 0.25).
The terrain arm trained longer before plateauing but did not generalize better.

**Updated conclusion:** With alignment and resolution confounds removed,
photogrammetric micro-topography still does not improve microsite segmentation.
The RGB-sufficient explanation is now the credible one. Remaining caveats:
sparse-SfM effective resolution is ~4–5 cm (dense MVS blocked: no CUDA COLMAP),
and SfM elevation conflates terrain with object height (slash/stumps).
Scripts: `scripts/photogrammetry/render_per_image_dem.py` (renderer),
`scripts/experiments/train_terrain_ab.py` (A/B trainer).
Checkpoints: `external/fresh-ml4seeding-data/data/models/segformer_b2_{rgb_only_subset,terrain8}_best.pt`;
histories: `external/fresh-ml4seeding-data/data/results/terrain_ab_*.json`.

**BUT** the object-height component of the DSM is useful on its own: see
`nope_map.md` (2026-10-07) for the geometric "not-microsite" veto layer
(stumps/logs/slash-piles/pits) built from the same reconstruction—validated
against the 6-image feature annotation set, no model training required.

---

## 5. Proposed Experiment

### Hypothesis

Terrain geometry features (slope, curvature, roughness) derived from a photogrammetric DSM will improve microsite classification compared to RGB-only, because:
- Exposed mineral soil on flat ground vs. slopes may have different spectral signatures
- Terrain geometry can help distinguish microsites that look similar in RGB
- Geometric features are invariant to lighting conditions that affect RGB

### Experiment Design

**Approach:** RGB + terrain feature fusion using a multi-input SegFormer

**Architecture:**
```
RGB Image (1024×1024×3) ──→ SegFormer Encoder ──┐
                                                  ├─→ Fusion ──→ Decoder ──→ Prediction
DSM Features (1024×1024×4) ──→ Feature Extractor ──┘
  [slope, aspect, curvature, roughness]
```

**Training:**
- Use the v4 dataset (384 train / 110 valid / 54 test)
- Same training settings as the current best model (SegFormer-B2)
- Focal loss for class imbalance

**Evaluation:**
- Compare RGB-only vs. RGB+terrain on the same test set
- Report Test IoU, Test Dice, and per-class metrics
- Analyze where terrain features help (e.g., slope vs. flat areas)

### Expected Outcome

If the hypothesis is correct, RGB+terrain should outperform RGB-only, particularly in areas where:
- Slope affects microsite visibility (e.g., shaded slopes vs. sunny flats)
- Terrain geometry helps distinguish microsites that look similar in RGB
- Geometric features are invariant to lighting conditions

---

## 5. Implementation Plan

### Step 1: Generate DSM from A58 images
- Use Agisoft Metashape or similar photogrammetry software
- Input: 750 A58 raw drone images with GPS and camera pose
- Output: DSM GeoTIFF covering the A58 site

### Step 2: Extract terrain features
- Use `rasterio` and `scipy` to compute slope, aspect, curvature, roughness
- Align terrain features with the 1024×1024 image tiles
- Save as multi-band GeoTIFF or numpy arrays

### Step 3: Implement multi-input model
- Modify the SegFormer model to accept two inputs (RGB + terrain features)
- Use a simple feature extractor for the terrain features (2-3 conv layers)
- Fuse features at the encoder level (early fusion) or decoder level (late fusion)

### Step 4: Train and evaluate
- Train on the v4 dataset (384 train / 110 valid / 54 test)
- Compare RGB-only vs. RGB+terrain on the same test set
- Report metrics and analyze where terrain features help

### Resource Requirements

- **Compute:** 1 GPU (RTX PRO 6000, 102GB VRAM)—sufficient for the experiment
- **DSM generation:** ~1-2 hours for photogrammetry processing
- **Training:** ~30 minutes for the multi-input model
- **Total time:** ~3-4 hours for the complete experiment

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| DSM quality is poor | Use the existing orthomosaic as a fallback; extract features from the ortho instead |
| Terrain features don't help | The experiment is cheap (few hours) and provides a negative result for the paper |
| Photogrammetry fails | Use the existing orthomosaics (A10, A58, G15) which already exist |
| Model is too complex | Start with early fusion (concatenate inputs) before trying complex fusion |

---

## 7. References

1. Meng, X., et al. (2023). TerrainNet: Visual Modeling of Complex Terrain for High-speed, Off-road Navigation. *RSS 2023*.
2. Valada, A., et al. (2016). Deep Multispectral Semantic Scene Understanding of Forested Environments using Multimodal Fusion. *ISER 2016*.
3. Fan, R., et al. (2020). SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection. *ECCV 2020*.
4. Min, C., et al. (2022). ORFD: A Dataset and Benchmark for Off-Road Freespace Detection. *ICRA 2022*.
5. Farhadpour, S., et al. (2026). SegFormer and SegFormer-UNet for anthropogenic geomorphic feature extraction from land surface parameters. *EarthArXiv*.
