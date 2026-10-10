# Feature Detection for Microsite Classification

**Status:** Planning  
**Date:** 2026-10-04  
**Context:** Microsite classification is working well (SegFormer-B2, Test IoU 0.622), but the model has no context about WHERE microsites are located. Large terrain features (stumps, logs, boulders, water, roads, slash piles) are strong indicators of where microsites are NOT located.

---

## Hypothesis

Detecting large terrain features that are NOT microsites will improve the microsite classification model by providing spatial context. The model can learn to avoid these areas and focus on the actual microsites.

---

## Feature Classes

| Class | Description | Why It's Not a Microsite |
|-------|-------------|--------------------------|
| `stump` | Tree stumps | Obstacle, no soil exposure |
| `log` | Fallen logs | Obstacle, no soil exposure |
| `boulder` | Large rocks | Obstacle, no soil exposure |
| `water` | Pooling water | Not suitable for seeding |
| `road` | Gravel road | Compacted, not suitable |
| `slash` | Slash piles / woody debris | Obstacle, no soil exposure |
| `microsite` | Actual microsites (good/fair) | The target class |

---

## Workflow

### Phase 1: Feature Annotation (New Project)

1. **Upload A58 images** to the new `a58-features` project
2. **User annotates** examples of each feature type using SAM3 smart polygon
3. **Train a feature detection model** (instance segmentation)
4. **Pre-label** the full dataset with the feature detection model
5. **Review and correct** the pre-labels
6. **Iterate** until the feature detection is good

### Phase 2: Microsite Classification with Feature Context

1. **Combine** the microsite dataset (a58-microsites) with the feature dataset (a58-features)
2. **Train a joint model** that uses both microsite labels and feature detections
3. **The model learns** to avoid the feature areas and focus on the actual microsites
4. **Evaluate** the improvement over the microsite-only model

---

## Implementation

### Option A: Separate Models
- Train a feature detection model on `a58-features`
- Use the feature detections as additional input to the microsite model
- The microsite model sees: RGB + microsite mask + feature detections

### Option B: Joint Model
- Train a single model that predicts both microsites and features
- The model learns the relationship between features and microsites
- More complex but potentially better

### Option C: Feature-Guided Preprocessing
- Use the feature detection model to mask out feature areas
- Train the microsite model only on the non-feature areas
- Simpler but less flexible

---

## Next Steps

1. **User annotates** feature examples in the `a58-features` project
2. **Train** a feature detection model
3. **Pre-label** the full dataset
4. **Evaluate** the feature detection quality
5. **Integrate** with the microsite model

---

## Roboflow Project

- **Project:** `gregory-paradis-ubc-ca/a58-features`
- **Type:** Instance segmentation
- **Classes:** stump, log, boulder, water, road, slash (to be added in the UI)
- **Annotation Job:** `5HrDFvQp3T1SJYk2k3Ym` (750 images)
- **Annotation URL:** https://app.roboflow.com/gregory-paradis-ubc-ca/a58-features/annotate?job=5HrDFvQp3T1SJYk2k3Ym

## How to Annotate

1. Open the annotation URL above
2. Use the **SAM3 smart polygon tool** (magic wand icon)
3. Click on each feature type:
   - **Stump**—tree stumps (elevated, woody)
   - **Log**—fallen logs (elongated, woody)
   - **Boulder**—large rocks (rounded, hard edges)
   - **Water**—pooling water (reflective, flat)
   - **Road**—gravel road (linear, light-colored)
   - **Slash**—slash piles / woody debris (irregular, brown)
4. Label each annotation with the correct class name
5. Save and move to the next image

The SAM3 tool will generate precise masks from your clicks—much faster than freehand drawing.

---

## Notes

- This is a separate project from `a58-microsites` to keep the microsite annotations clean
- The feature detection model can be trained independently and then integrated with the microsite model
- The user's existing microsite annotations are preserved in the `a58-microsites` project
