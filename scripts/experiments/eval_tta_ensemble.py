#!/usr/bin/env python3
"""
Evaluate existing checkpoints with TTA and ensembles on the v4 test set.
All combos on identical 54 test images, binary microsite IoU.
"""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import json

DATA = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4/test")
ANN = Path("/home/gep/projects/ignitebc-flashforest/annotation")
DEVICE = torch.device("cuda")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(name, ctor):
    m = ctor().to(DEVICE)
    m.load_state_dict(torch.load(ANN / name, map_location=DEVICE))
    m.eval()
    return m

models = {}
models["b2"] = load("segformer_b2_best.pt",
                    lambda: smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=2))
models["unet_focal"] = load("unet_resnet34_focal_best.pt",
                            lambda: smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2))
models["b0"] = load("segformer_b0_best.pt",
                    lambda: smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=2))
models["b3"] = load("segformer_b3_best.pt",
                    lambda: smp.Segformer(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=2))


def dihedral(x, k, flip):
    if flip:
        x = np.flip(x, axis=3)
    return np.rot90(x, k, axes=(2, 3))


def dihedral_inv(p, k, flip):
    p = np.rot90(p, -k, axes=(0, 1))
    if flip:
        p = np.flip(p, axis=1)
    return p


@torch.no_grad()
def prob_map(model, img, tta=False):
    """img: HxWx3 float32 normalized. Returns microsite prob map."""
    probs = []
    ops = [(0, False)] if not tta else [(k, f) for k in range(4) for f in (False, True)]
    for k, f in ops:
        x = img.transpose(2, 0, 1)[None]
        x = dihedral(x, k, f)
        t = torch.from_numpy(np.ascontiguousarray(x).astype(np.float32)).to(DEVICE)
        p = torch.softmax(model(t), 1)[0, 1].cpu().numpy()
        probs.append(dihedral_inv(p, k, f))
    return np.mean(probs, axis=0)


# leakage detection: rf hashes appearing in ANY older train split
import re
leaked_hashes = set()
for v in ["roboflow_v1", "roboflow_v2", "roboflow_v3"]:
    for p in (DATA.parent.parent / v / "train").glob("*.jpg"):
        m = re.search(r"\.rf\.([a-f0-9]+)\.jpg$", p.name)
        if m:
            leaked_hashes.add(m.group(1))

data = []
n_leaked = 0
for p in sorted(DATA.glob("*.jpg")):
    if p.name.endswith("_mask.png"):
        continue
    img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    gt = np.array(Image.open(DATA / f"{p.stem}_mask.png")).astype(np.int64)
    m = re.search(r"\.rf\.([a-f0-9]+)\.jpg$", p.name)
    leaked = m and m.group(1) in leaked_hashes
    n_leaked += bool(leaked)
    data.append(((img - MEAN) / STD, gt, p.stem, leaked))
print(f"{len(data)} test images, {n_leaked} potentially leaked (in older train splits)")

# prob maps: model x tta
probs = {}
for name, m in models.items():
    for tta in (False, True):
        probs[(name, tta)] = [prob_map(m, img, tta) for img, _, _, _ in data]
        print(f"  {name} tta={tta} done")


def metrics(pred, gt):
    tp = (pred & (gt == 1)).sum(); fp = (pred & (gt != 1)).sum(); fn = (~pred & (gt == 1)).sum()
    return tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def evaluate(label, prob_list, thr=0.5):
    ious, precs, recs = [], [], []
    ious_c = []
    for prob, (_, gt, _, leaked) in zip(prob_list, data):
        i, pr, rc = metrics(prob > thr, gt)
        ious.append(i); precs.append(pr); recs.append(rc)
        if not leaked:
            ious_c.append(i)
    print(f"{label:>28}: IoU {np.mean(ious):.4f} (clean {np.mean(ious_c):.4f}, n={len(ious_c)})  "
          f"prec {np.mean(precs):.4f}  rec {np.mean(recs):.4f}")
    return {"label": label, "iou": float(np.mean(ious)), "iou_clean": float(np.mean(ious_c)),
            "prec": float(np.mean(precs)), "rec": float(np.mean(recs))}


results = []
print("\n===== v4 test set (54 images) =====")
results.append(evaluate("b2", probs[("b2", False)]))
results.append(evaluate("b2 + TTA", probs[("b2", True)]))
results.append(evaluate("unet_focal", probs[("unet_focal", False)]))
results.append(evaluate("unet_focal + TTA", probs[("unet_focal", True)]))
results.append(evaluate("b0", probs[("b0", False)]))
results.append(evaluate("b0 + TTA", probs[("b0", True)]))

# ensembles
ens_b2_unet = [(a + b) / 2 for a, b in zip(probs[("b2", False)], probs[("unet_focal", False)])]
results.append(evaluate("ens(b2,unet)", ens_b2_unet))
ens_b2_unet_tta = [(a + b) / 2 for a, b in zip(probs[("b2", True)], probs[("unet_focal", True)])]
results.append(evaluate("ens(b2,unet) + TTA", ens_b2_unet_tta))
ens_b0_unet = [(a + b) / 2 for a, b in zip(probs[("b0", False)], probs[("unet_focal", False)])]
results.append(evaluate("ens(b0,unet) [ref: 0.6501]", ens_b0_unet))
ens_all = [(a + b + c) / 3 for a, b, c in zip(probs[("b2", True)], probs[("unet_focal", True)], probs[("b0", True)])]
results.append(evaluate("ens(b2,unet,b0) + TTA", ens_all))
results.append(evaluate("b3", probs[("b3", False)]))
results.append(evaluate("b3 + TTA", probs[("b3", True)]))
ens_b3_unet_tta = [(a + b) / 2 for a, b in zip(probs[("b3", True)], probs[("unet_focal", True)])]
results.append(evaluate("ens(b3,unet) + TTA", ens_b3_unet_tta))
ens_b3b2unet_tta = [(a + b + c) / 3 for a, b, c in zip(probs[("b3", True)], probs[("b2", True)], probs[("unet_focal", True)])]
results.append(evaluate("ens(b3,b2,unet) + TTA", ens_b3b2unet_tta))

with open(ANN / "tta_ensemble_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved annotation/tta_ensemble_eval.json")
