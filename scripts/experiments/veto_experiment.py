#!/usr/bin/env python3
"""
Nope-map veto experiment on the v4 test set.

Baseline: SegFormer-B2 microsite predictions.
Variants:
  V0 baseline (argmax)
  V1 hard veto:              pred & ~nope
  V2 hard veto, cleaned mask: veto only after 3px opening + 30px min blob
  V3 soft veto:              threshold 0.5 normally, 0.75 inside nope
Metrics per variant: microsite IoU / precision / recall.
Veto bookkeeping: TP-lost (true microsite removed - the cost) vs
FP-removed (false-positive removed - the benefit).
Gallery: RGB | GT | baseline | V1 result | V2 result with color-coded
vetoed-TP (orange, lost good microsite) and vetoed-FP (cyan, removed junk).
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import cv2
import segmentation_models_pytorch as smp
from scipy import ndimage

DATA = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4/test")
NOPE = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map/per_image_v4")
ANN = Path("/home/gep/projects/ignitebc-flashforest/annotation")
OUT = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map/veto_experiment")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ckpt = ANN / "segformer_b2_best.pt"
if not ckpt.exists():
    ckpt = ANN / "segformer_b2_rgb_only_subset_best.pt"
print(f"checkpoint: {ckpt.name}")

model = smp.Segformer(encoder_name="mit_b2", encoder_weights=None,
                      in_channels=3, classes=2).to(DEVICE)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()


def clean_nope(m):
    b = ((m == 1) | (m == 2)).astype(np.uint8)
    b = ndimage.binary_opening(b, iterations=1)
    lab, n = ndimage.label(b)
    sizes = ndimage.sum(b, lab, range(1, n + 1))
    keep = np.where(sizes >= 30)[0] + 1
    return np.isin(lab, keep)


rows = []
gallery = []
imgs = sorted(p for p in DATA.glob("*.jpg") if not p.name.endswith("_mask.png"))
print(f"{len(imgs)} test images")
skipped = 0
for p in imgs:
    gt_p = DATA / f"{p.stem}_mask.png"
    nope_p = NOPE / f"{p.stem}.png"
    if not nope_p.exists():
        skipped += 1
        continue
    img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    gt = np.array(Image.open(gt_p)).astype(np.int64)
    nope = cv2.imread(str(nope_p), cv2.IMREAD_UNCHANGED)
    if nope.shape[:2] != img.shape[:2]:
        nope = cv2.resize(nope, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    veto = (nope == 1) | (nope == 2)

    x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0, 1].cpu().numpy()  # microsite prob

    pred0 = (prob > 0.5)
    pred1 = pred0 & ~veto                                # V1 hard
    veto_c = clean_nope(nope)
    pred2 = pred0 & ~veto_c                              # V2 cleaned hard
    pred3 = prob > np.where(veto, 0.75, 0.50)            # V3 soft

    tp0 = (pred0 & (gt == 1)).sum()
    fp0 = (pred0 & (gt != 1)).sum()
    fn0 = (~pred0 & (gt == 1)).sum()
    tp_lost = (pred0 & ~pred1 & (gt == 1)).sum()         # true microsite vetoed away
    fp_removed = (pred0 & ~pred1 & (gt != 1)).sum()      # junk vetoed away

    row = {"image": p.stem[:40], "tp_lost": int(tp_lost), "fp_removed": int(fp_removed)}
    for name, pred in [("V0_base", pred0), ("V1_hard", pred1), ("V2_clean", pred2), ("V3_soft", pred3)]:
        tp = (pred & (gt == 1)).sum()
        fp = (pred & (gt != 1)).sum()
        fn = (~pred & (gt == 1)).sum()
        row[f"{name}_iou"] = tp / max(tp + fp + fn, 1)
        row[f"{name}_prec"] = tp / max(tp + fp, 1)
        row[f"{name}_rec"] = tp / max(tp + fn, 1)
    rows.append(row)

    if len(gallery) < 8:
        # color composite: baseline vs V1 with veto bookkeeping
        def render(pred):
            vis = (img * 255).astype(np.uint8).copy()
            vis[pred] = (0.55 * vis[pred] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
            return vis
        rgb_vis = (img * 255).astype(np.uint8).copy()
        # GT in green tint
        gt_vis = rgb_vis.copy()
        gt_vis[gt == 1] = (0.55 * gt_vis[gt == 1] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
        v1_vis = render(pred1)
        # veto bookkeeping on the V1 panel: orange = lost TP, cyan = removed FP
        lost = pred0 & ~pred1 & (gt == 1)
        removed = pred0 & ~pred1 & (gt != 1)
        v1_vis[lost] = (0.5 * v1_vis[lost] + 0.5 * np.array([0, 165, 255])).astype(np.uint8)
        v1_vis[removed] = (0.5 * v1_vis[removed] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
        tiles = []
        for t, cap in [(rgb_vis, "RGB"), (gt_vis, "GT"), (render(pred0),
                        f"V0 base IoU={row['V0_base_iou']:.2f}"),
                       (v1_vis, f"V1 veto IoU={row['V1_hard_iou']:.2f} orange=lostTP cyan=removedFP")]:
            t = cv2.resize(t, (512, 512))
            cv2.rectangle(t, (0, 0), (512, 18), (0, 0, 0), -1)
            cv2.putText(t, cap, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            tiles.append(t)
        gallery.append(np.concatenate(tiles, axis=1))

print(f"skipped (no nope mask): {skipped}")

# Aggregate
print("\n===== AGGREGATE (mean over test images) =====")
print(f"{'variant':>9} {'IoU':>7} {'prec':>7} {'rec':>7}")
for v in ["V0_base", "V1_hard", "V2_clean", "V3_soft"]:
    print(f"{v:>9} {np.mean([r[f'{v}_iou'] for r in rows]):>7.4f} "
          f"{np.mean([r[f'{v}_prec'] for r in rows]):>7.4f} "
          f"{np.mean([r[f'{v}_rec'] for r in rows]):>7.4f}")

tot_lost = sum(r["tp_lost"] for r in rows)
tot_removed = sum(r["fp_removed"] for r in rows)
tot_tp0 = sum(1 for r in rows)  # placeholder
print(f"\nVeto bookkeeping (hard veto, all test images):")
print(f"  true-microsite pixels vetoed (the COST):   {tot_lost:,}")
print(f"  false-positive pixels vetoed (the BENEFIT): {tot_removed:,}")
print(f"  benefit/cost ratio: {tot_removed / max(tot_lost, 1):.2f}x")

# per-image deltas for inspection
deltas = sorted(((r["V1_hard_iou"] - r["V0_base_iou"], r["image"]) for r in rows))
print("\nworst 5 (V1-V0):", [(f"{d:+.3f}", n) for d, n in deltas[:5]])
print("best 5 (V1-V0): ", [(f"{d:+.3f}", n) for d, n in deltas[-5:]])

with open(OUT / "veto_results.json", "w") as f:
    json.dump(rows, f, indent=2)

if gallery:
    mont = np.concatenate(gallery, axis=0)
    cv2.imwrite(str(OUT / "veto_gallery.png"), mont)
    print(f"\nGallery: {OUT/'veto_gallery.png'}")
print(f"Results: {OUT/'veto_results.json'}")
