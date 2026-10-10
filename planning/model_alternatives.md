# Model Architecture Alternatives for Microsite Segmentation

**Status:** Planning  
**Date:** 2026-10-04  
**Context:** U-Net + ResNet34 achieving Test IoU 0.665 on binary microsite segmentation (414 images). Performance has plateaued. Exploring alternatives for the paper and potential improvement.

---

## Roboflow Agent Recommendations (2026-10-04)

The Roboflow agent provided detailed recommendations for architecture and training strategy alternatives. Key points:

### Architecture Shortlist (Priority Order)

| Priority | Architecture | Why |
|----------|-------------|-----|
| 1 | **DeepLabV3+** | Multiscale context may help distinguish soil from visually similar slash, shadows, and dead vegetation. Available as Roboflow 2.0 or locally. |
| 2 | **SegFormer-B0/B2** | Tests whether a hierarchical transformer captures useful context without starting with a very large model on 414 images. |
| 3 | **YOLO26-sem Small** | Directly trainable Roboflow semantic baseline at 1024×1024, useful for hosted-vs-local comparison. |
| 4 | **Mask2Former** | Later research ablation if simpler models plateau. |

### Loss Function Recommendations

- **Control:** Weighted BCE/CE + soft Dice (current approach)
- **Test 1:** Focal BCE + Dice (if easy background pixels dominate gradients)
- **Test 2:** Tversky or focal Tversky (if misses are more costly than false alarms)
- **Test 3:** Lovász hinge (IoU-oriented, if main metric is positive-class IoU)
- **Test 4:** Small boundary-aware term (only if errors are primarily at soil edges)

**Rule:** Change one thing at a time. Use same splits, crops, seeds, threshold-selection rule, and postprocessing for loss comparisons.

### Training Strategy (Higher Leverage Than Model Size)

- **Spatial splitting:** Split by source flight/orthomosaic or spatial block, NOT randomly by tile. Neighboring tiles can make a random test set look much easier than deployment.
- **Deliberate sampling:** Mix positive-containing tiles, hard negatives, and background-only tiles. Preserve representative validation/test distribution.
- **Label auditing:** At 7.7% coverage, disagreement about what counts as exposed mineral soil may dominate architecture differences.
- **Restrained augmentation:** Rotations/flips where orientation is irrelevant, modest illumination changes. Avoid strong hue shifts that erase the subtle RGB cue.
- **Copy-paste NOT recommended:** Pasting soil regions can create implausible texture, shadow, and terrain transitions.
- **Semi-supervised learning:** Plausible given unannotated A58 imagery. Train a teacher, retain high-confidence predictions, compare student performance on untouched test set.

### Evaluation Metrics

Report positive-class IoU and Dice, but add:
- **Precision, recall, and F_β** for the microsite class (threshold chosen on validation data)
- **Precision–recall curve and average precision**, plus **IoU versus threshold**
- **Area error and bias:** predicted suitable hectares vs. annotated hectares
- **Spatial usability:** number/area of deployable patches after minimum-area rules
- **Uncertainty:** per-site or per-flight metrics with confidence intervals from bootstrapping independent spatial units

### Critical Question—ANSWERED

**Were the test tiles held out by flight/orthomosaic or spatial block, or randomly split from the same imagery as training?**

**Answer: Randomly split from the same imagery.** The Roboflow version randomly split the 418 annotated images into 414 train / 2 valid / 2 test. The test images are from the same areas as the training images—the 0.665 IoU is likely inflated by spatial autocorrelation.

**Implication:** The reported performance is an overestimate of deployment performance. A proper spatial split (by flight area, orthomosaic section, or spatial block) would give a more realistic estimate.

**Action for next iteration:**
- Implement spatial splitting for the A58 dataset (split by geographic region, not random)
- Hold out entire spatial blocks for testing, not random tiles
- Report both random-split and spatial-split metrics to show the difference

---

---

---

## Current Baseline

- **Model:** U-Net with ResNet34 encoder
- **Task:** Binary semantic segmentation (microsite vs. background)
- **Data:** 414 annotated drone images (1024×1024), 7.7% microsite coverage
- **Performance:** Test IoU 0.665, Test Dice 0.794
- **Challenge:** Extreme class imbalance, subtle spectral differences between classes

---

## Alternative Architectures

### 1. SegFormer (Xie et al., 2021)

**What it is:** Transformer-based encoder with lightweight MLP decoder.

**Why it might help:**
- Better long-range dependencies than CNNs (important for spatial context in drone imagery)
- Designed for semantic segmentation (not adapted from classification)
- Good performance on remote sensing benchmarks
- More parameter-efficient than U-Net for similar performance

**Implementation:** `segmentation_models_pytorch` has SegFormer support, or use `transformers` library.

**Paper:** Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. NeurIPS.

---

### 2. DeepLabV3+ (Chen et al., 2018)

**What it is:** Atrous (dilated) convolutions for multi-scale feature extraction without downsampling.

**Why it might help:**
- Captures multi-scale context without losing spatial resolution
- Good for objects at different sizes (microsites vary in size)
- Well-established in remote sensing literature

**Implementation:** `segmentation_models_pytorch` has DeepLabV3+ support.

**Paper:** Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. ECCV.

---

### 3. Mask2Former (Cheng et al., 2022)

**What it is:** Transformer-based segmentation with masked attention mechanism.

**Why it might help:**
- Current state-of-the-art on many segmentation benchmarks
- Masked attention focuses on relevant regions
- Good for fine-grained segmentation tasks

**Implementation:** Available in `transformers` library (HuggingFace).

**Paper:** Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention mask transformer for universal image segmentation. CVPR.

---

### 4. HRNet (Wang et al., 2020)

**What it is:** Maintains high-resolution representations throughout the network (no aggressive downsampling).

**Why it might help:**
- Preserves fine spatial detail (critical for small microsite patches)
- Good for tasks where precise boundaries matter
- Proven performance on dense prediction tasks

**Implementation:** Available in `segmentation_models_pytorch` or custom implementation.

**Paper:** Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Liu, W. (2020). Deep high-resolution representation learning for visual recognition. TPAMI.

---

### 5. PSPNet (Zhao et al., 2017)

**What it is:** Pyramid Scene Parsing Network with pyramid pooling module.

**Why it might help:**
- Captures global context at multiple scales
- Good for scenes with varying object sizes
- Well-established baseline for comparison

**Implementation:** `segmentation_models_pytorch` has PSPNet support.

**Paper:** Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. CVPR.

---

## Alternative Training Strategies

### 1. Focal Loss (Lin et al., 2017)

**What it is:** Loss function that down-weights easy examples, focusing on hard ones.

**Why it might help:**
- Specifically designed for extreme class imbalance
- Reduces the dominance of the majority class (background)
- Could improve minority class (microsite) performance

**Implementation:** Replace the current CE+Dice loss with focal loss.

**Paper:** Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.

---

### 2. Lovász Loss (Berman et al., 2018)

**What it is:** Loss function that directly optimizes the IoU metric.

**Why it might help:**
- Aligns the training objective with the evaluation metric
- Better than cross-entropy for segmentation tasks
- Could improve IoU performance

**Implementation:** Available in `segmentation_models_pytorch` or custom implementation.

**Paper:** Berman, M., Rannen, A., & Blaschko, M. B. (2018). The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. CVPR.

---

### 3. Copy-Paste Augmentation (Ghiasi et al., 2021)

**What it is:** Copies objects from one image and pastes them onto another.

**Why it might help:**
- Artificially increases microsite frequency without collecting more data
- Creates more diverse training examples
- Could improve minority class learning

**Implementation:** Custom augmentation pipeline.

**Paper:** Ghiasi, G., Cui, Y., Srinivas, A., Qian, R., Lin, T. Y., Cubuk, E. D., ... & Zoph, B. (2021). Simple copy-paste is a strong data augmentation method for instance segmentation. CVPR.

---

### 4. Test-Time Augmentation (TTA)

**What it is:** Averages predictions over multiple augmented versions of the input.

**Why it might help:**
- More robust predictions at inference time
- No additional training cost
- Can improve performance without retraining

**Implementation:** Apply augmentations (flips, rotations) at inference and average the predictions.

---

### 5. Semi-Supervised Learning

**What it is:** Uses unannotated images with pseudo-labels generated by the current model.

**Why it might help:**
- Leverages the 330+ unannotated images
- Can improve performance without additional annotation
- Pseudo-labels from the current model can be refined

**Implementation:** Generate pseudo-labels with the current model, add them to the training set, and retrain.

---

## Most Promising for This Project

Given the extreme class imbalance (7.7% microsite) and small dataset (414 images), the most promising approaches are:

1. **SegFormer-B2 or B3**—transformer-based, handles imbalance better than CNNs, good for remote sensing
2. **Focal Loss**—specifically designed for extreme imbalance
3. **Copy-Paste Augmentation**—artificially increases the rare class frequency
4. **Mask2Former**—current state-of-the-art, if we want to try the best available

---

## Related Work: SegUNet (Farhadpour et al., 2026)

A recent preprint ([Farhadpour et al., 2026](https://eartharxiv.org/repository/view/13709/)) compares U-Net, SegFormer, and a hybrid **SegUNet** for semantic segmentation of anthropogenic geomorphic features from lidar data—a task very similar to ours (terrain feature extraction with class imbalance).

**Key findings relevant to our work:**
- **SegUNet (hybrid SegFormer-UNet) outperforms both pure approaches**—combines CNN texture sensitivity with transformer long-range context
- **Random initialization often matches or exceeds ImageNet pretraining** for terrain data (domain gap between natural images and terrain)
- **Focal Tversky loss** is used for class imbalance (similar to our focal loss approach)
- **End-to-end training** is the most consistent driver of gains

**Implication for our work:** A SegUNet-style hybrid might outperform both our current U-Net+Focal (Test IoU 0.575) and SegFormer-B2 (Test IoU 0.622) models. The hybrid approach could combine the best of both worlds.

**Suggested next experiment:** Implement a SegUNet hybrid (SegFormer encoder + U-Net decoder) with focal Tversky loss on our v4 dataset.

---

## Experimental Results (2026-10-04)

Trained four models on the v3 dataset (414 images) with identical splits and evaluation:

| Model | Val IoU | Test IoU | Test Dice | Best Epoch | Params |
|-------|---------|----------|-----------|------------|--------|
| U-Net + ResNet34 (baseline) | 0.514 | 0.665 | 0.794 | 88 | 24.4M |
| DeepLabV3+ + ResNet34 | 0.518 | 0.639 | 0.776 | 19 | 22.4M |
| SegFormer-B0 | 0.511 | 0.674 | 0.802 | 46 | 3.7M |
| **U-Net + ResNet34 + Focal Loss** | **0.518** | **0.698** | **0.821** | **38** | 24.4M |

**Key findings:**
- **Focal Loss U-Net is the best**: Test IoU 0.698 (+5% over baseline), Test Dice 0.821 (+3%), converges 2.3x faster (38 vs 88 epochs)
- **SegFormer-B0 is competitive**: Test IoU 0.674 with only 3.7M parameters (6.6x smaller than U-Net)—great for deployment
- **DeepLabV3+ underperforms**: Test IoU 0.639, worse than baseline U-Net—atrous convolutions don't help for this spectral task
- **Class imbalance was the bottleneck**, not the architecture. Focal Loss addresses it directly.

**Recommendation:** Use **SegFormer-B2** for the next training round (v5)—it achieved the best Test IoU (0.622) on our data. Consider a SegUNet hybrid (SegFormer encoder + U-Net decoder) as a follow-up experiment based on the Farhadpour et al. (2026) findings.

---

## Architecture Shoot-out on v4 (2026-10-08)

All candidates re-evaluated in one harness on the v4 test set (54 images;
"clean" subset n=48 excludes 6 images whose rf-hash appears in v1–v3 train
splits—possible leakage for older checkpoints). TTA = 8 dihedral views
averaged. Full data: `external/fresh-ml4seeding-data/data/results/tta_ensemble_eval.json`.

| Rank | Model | IoU full | IoU clean |
|---|---|---|---|
| 1 | **unet_resnet34_focal + TTA** | **0.652** | **0.725** |
| 2 | ens(b2,unet,b0) + TTA | 0.648 | 0.721 |
| 3 | ens(b0,unet) | 0.646 | 0.719 |
| 4 | unet_resnet34_focal | 0.638 | 0.710 |
| 5 | b0 + TTA | 0.618 | 0.687 |
| 6 | segformer_b2 | 0.566 | 0.631 |
| 7 | segformer_b3 (trained this round) | 0.558 | 0.618 |

Conclusions:
- **unet_resnet34_focal + TTA is the current best**, displacing segformer_b2
  (the "B2 is best" recommendation above did not survive same-harness
  evaluation on v4).
- TTA: +0.014 (U-Net), +0.015 (B0), ~0 (B2).
- SegFormer-B3 (45M params, `segformer_b3_best.pt` / `segformer_b3_result.json`):
  no gain over B2—capacity is not the bottleneck at 384 training images.
- Top-3 within noise (per-image σ ≈ 0.25). Simplest production choice:
  unet_focal + TTA.
- **Architecture is exhausted at this data size; further gains need more
  annotations (active learning), not new models.**

---

## References

- Chen et al. (2018)—DeepLabV3+
- Xie et al. (2021)—SegFormer
- Cheng et al. (2022)—Mask2Former
- Wang et al. (2020)—HRNet
- Zhao et al. (2017)—PSPNet
- Lin et al. (2017)—Focal Loss
- Berman et al. (2018)—Lovász Loss
- Ghiasi et al. (2021)—Copy-Paste Augmentation
