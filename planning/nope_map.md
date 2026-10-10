# Nope Map: Geometric Veto Layer from SfM Photogrammetry

*Initial version 2026-10-07; scale-corrected and georeferenced same day.*

## Purpose

Site-level "not microsite" obstacle layer generated **without any annotation**,
from the COLMAP full-flight reconstruction. Intended use: veto/mask layer on top
of the RGB microsite segmentation (SegFormer-B2), NOT as a fused input channel
(fusion was tested and does not help—see `stereo_depth_assessment.md` §4).

Catches: stumps, logs, boulders, slash piles, berms (bumps). Does NOT catch:
thin flat-lying slash (<~15 cm tall), flush rocks, smooth compacted road
surfaces, or water (textureless for SfM).

## ⚠ Units / scale (important)

COLMAP reconstructions have arbitrary scale. The GPS-EXIF similarity alignment
(Umeyama, 750 frames) gives **1 local unit = 8.791 m** (horizontal RMS 0.29 m,
vertical RMS 0.44 m, max 2.2 m). All physical numbers below are REAL meters
after this correction. Early drafts of this note quoted grid/threshold numbers
that were off by 8.79× (local units presented as meters).

- Site extent: **~112 × 97 m** (~1.1 ha)
- Grid cell: 0.02 local = **17.6 cm**; mean SfM point spacing ~26 cm
- Honest detection floor: objects protruding **≥ ~15 cm** vertically

## Method

1. COLMAP sparse reconstruction, all 750 A58-segment2 frames (1024×682
   downscales, CPU SIFT, sequential matcher). 750/750 registered, one model,
   715,225 points, BA reprojection error 0.22 px. Full-res originals (with GPS
   EXIF, Zenmuse P1): `/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/raw_drone_images/a58_segment2/`
2. PCA plane-align (normal toward cameras), ground grid at 17.6 cm cells.
3. DSM = linear interp of all points; DTM = 10th-percentile filter over
   **1.5 m (real)** window + Gaussian; **HAG = DSM − DTM** (CHM-style);
   roughness = local std over 0.35 m window.
4. Thresholds (real meters), **validated against ground truth**:
   `bump = HAG > +0.15 m OR roughness > 0.05 m`; `pit = HAG < −0.20 m`.
   Valid mask eroded ~0.5 m at survey boundary (kills interpolation artifacts).
5. Per-image rendering: per-pixel ray-casting with parallax correction;
   masks for all 750 frames (1024×682) + stretched 1024×1024 for the 548 v4
   training images.
6. Georeferencing: similarity fit COLMAP→EXIF-GPS, resampled to UTM 10N
   (EPSG:32610) GeoTIFF.

## Validation (6 quick-annotated images, 353 objects; object-level recall)

| Threshold (h / rough) | slash (n=337) | log (n=10) | rock (n=5) | stump (n=1) |
|---|---|---|---|---|
| **0.15 / 0.05 (deployed)** | **0.87** | **0.90** | **0.60** | 0/1 |
| 0.25 / 0.10 (conservative) | 0.48 | 0.60 | 0.20 | 0/1 |
| 0.35 / 0.10 | 0.31 | 0.30 | 0.00 | 0/1 |

- Precision NOT measurable (annotations deliberately incomplete—features fill
  ~93% of each image; only a subset was clicked).
- QC observations: pickup truck flagged; big log flagged full-length; dense
  slash fields flagged; gravel road surface mostly clean but shows speckle
  (SfM noise floor at this sensitivity—consider min-blob-size filtering if
  a cleaner mask is needed).

## Operating points (site stats)

| Setting | Site nope % | Mean per-image nope % |
|---|---|---|
| Aggressive 0.15/0.05 (deployed) | **48.1%** (3,613 m² of ~7,500 m² valid) | 48.9% |
| Conservative 0.25/0.10 | 17.3% (1,299 m²) | 16.0% |

## Pits / ditches / "big hole that will fill with water"

**None exist in this flight segment**, now tested at a physically meaningful
−0.20 m threshold: total pit area 0.6 m², no pit blob > 0.5 m², deepest
depression −0.70 m (single small blob). What looked like ditches in early
renders were survey-boundary interpolation artifacts (removed by erosion).
The site is flat on the negative side. If a ditch/hole exists at A58, it is
outside this ~112×97 m footprint—the georeferenced GeoTIFF can be checked
against a map to confirm what the flight covered.

## Products (`external/fresh-ml4seeding-data/data/derived/nope_map/`)

- `site_nope_utm10.tif`—**georeferenced** GeoTIFF (EPSG:32610): bands
  [nope, valid, HAG-meters]
- `nope_on_orthomosaic_qc.png`—overlay on existing A58Segment2 ortho
- `site_nope.npz`—local-frame grids (incl. `hag_meters`, `rough_meters`,
  `scale_m_per_unit`) + alignment transforms
- `per_image/<frame>.png`—750 masks (0=ok, 1=bump, 2=pit, 255=no-data)
- `per_image_v4/<v4stem>.png`—548 masks in v4 training geometry
- `site_nope_map.png`, `nope_qc_montage.png`—visualizations
- `metadata.json`, `georeferencing.json`
- Scripts: `scripts/photogrammetry/build_nope_map.py`, `scripts/photogrammetry/georeference_nope_map.py`,
  `scripts/photogrammetry/dem_obstacle_test.py`

## Veto experiment: NEGATIVE (2026-10-08)—do not use as a microsite veto

Tested `microsite_pred × NOT(nope)` on the v4 test set (53 images,
SegFormer-B2, IoU 0.572 baseline):

| Variant | IoU | Prec | Rec |
|---|---|---|---|
| baseline | 0.572 | 0.709 | 0.675 |
| hard veto | **0.275** | 0.699 | 0.301 |
| hard veto, mask cleaned (opening + min blob) | 0.276 | 0.699 | 0.302 |
| soft veto (threshold 0.75 inside nope) | 0.571 | 0.720 | 0.664 |

Hard veto removed **2.48M px of TRUE microsite vs 0.46M px of false
positives (benefit/cost 0.19×)**. Min-blob sweep (0→8000 px ≈ 1.8 m²)
does not fix it (B/C stays ~0.19): the large flagged regions (slash fields)
*contain* most annotated microsites. GT "microsite" = plantable pocket
INSIDE the slash matrix (soil contact/shelter/moisture), not "area free of
debris"—15–25 cm geometry cannot separate the two. The user's concern
about filtering out good microsites was quantitatively validated.

**What the nope map IS good for:** macro-feature mapping (road, landing,
truck, large logs, slash piles—87–90% recall at 15 cm thresholds) for
site description/logistics, NOT as a planting filter. Microsite filtering
stays with the RGB model + more training data.

## Georeferencing details + ortho co-registration finding (2026-10-07)

- GPS-EXIF similarity (Umeyama, det=+1 enforced): 1 local unit = 8.791 m,
  horizontal fit RMS 0.29 m over 750 frames → the nope map is internally rigid.
- **A north-south flip existed in the first GeoTIFF/overlay products**
  (raster rows written south→north under a north-up geotransform). User-detected
  from road topology (road exits SOUTH in reality; products showed it exiting
  north). Fixed in all products; verified: road exits south, berm band tracks
  the road, turnaround/bend at the north end, truck and slash piles aligned.
- **Co-registration to the existing A58Segment2 orthomosaic: deliberately NOT
  applied.** Tie-point analysis (25 frames SIFT-matched against the ortho,
  11–705 inliers each): per-frame offsets scatter std (3.98, 6.03) m around a
  mean of only (+1.37, +0.62) m. Since the nope map is rigid at 0.29 m RMS,
  the scatter is **internal distortion in the ODM orthomosaic** (no-GCP
  bundle-adjustment drift). No rigid transform can align to it everywhere;
  fitting would degrade the better product. Recorded in
  `external/fresh-ml4seeding-data/data/derived/nope_map/georeferencing.json`.
- Consequence: treat the nope map GeoTIFF as the geometric reference; the
  orthomosaic is a visual backdrop good to ±4–6 m locally.
- Scripts: `scripts/photogrammetry/georeference_nope_map.py` (GPS alignment + GeoTIFF),
  `scripts/photogrammetry/coregister_tiepoints.py` (tie-point analysis, guarded—aborts
  without writing if the fit is weak).

## Limitations / TODO

- GPS accuracy ~1 m horizontal / ~2.5 m vertical. Confirmed from image XMP
  metadata (`drone-dji:RtkFlag=16` on all sampled frames = single-point;
  `RtkStdLon/Lat ≈ 0.8–1.1 m`, `RtkStdHgt ≈ 2–2.7 m`, `RtkDiffAge=0`).
  **RTK was never engaged during this flight** (the M300+P1 supports it—a
  D-RTK2 base or NTRIP network corrections would give RtkFlag=50, ~1–3 cm).
  Fine for a veto layer; enable RTK on future flights for survey-grade
  georeferencing.
- SfM vertical noise floor ≈ ±10–15 cm at this altitude/agency; the aggressive
  threshold trades precision for recall (road speckle visible in QC).
- Effective spatial resolution ~18–26 cm: sub-20 cm features invisible.
- Roads detected only via ruts/berms; geofence roads separately if needed.
- Other sites: A10 (338 frames) and G15 imagery + orthomosaics exist in the
  same eghase02 input dir—same pipeline applies.
