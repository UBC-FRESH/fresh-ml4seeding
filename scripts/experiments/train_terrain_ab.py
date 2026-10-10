#!/usr/bin/env python3
"""
Clean A/B: RGB-only vs RGB+DEM SegFormer-B2 on the SAME image subset
(only v4 images with a registered COLMAP pose and rendered terrain stack).

Per-image terrain stacks: /tmp/dem_per_image/<stem>.npy  (5,1024,1024) float16
channels: [rel_elev, slope, curvature, roughness, valid]
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import json
import time

DATA_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4")
DEM_DIR = Path("/tmp/dem_per_image")
OUT_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation")

NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 60
PATIENCE = 12
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

stats = json.load(open(DEM_DIR / "channel_stats.json"))
DEM_MEAN = np.array(stats["mean"] + [0.5], dtype=np.float32)  # 5th ch = valid
DEM_STD = np.array(stats["std"] + [0.5], dtype=np.float32)


def collect_split(split):
    img_dir = DATA_DIR / split
    items = []
    for p in sorted(img_dir.glob("*.jpg")):
        if p.name.endswith("_mask.png"):
            continue
        dem_path = DEM_DIR / (p.stem + ".npy")
        if dem_path.exists():
            items.append((p, img_dir / f"{p.stem}_mask.png", dem_path))
    return items


class FusionDataset(Dataset):
    def __init__(self, items, use_dem, augment=False):
        self.items = items
        self.use_dem = use_dem
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, dem_path = self.items[idx]
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path)).astype(np.int64)

        if self.use_dem:
            dem = np.load(dem_path).astype(np.float32)      # (5,1024,1024)
            dem = (dem - DEM_MEAN[:, None, None]) / DEM_STD[:, None, None]
            dem = np.clip(dem, -5, 5)  # valid ch already ~{-1,1} via z-norm
            feat = np.concatenate([img.transpose(2, 0, 1), dem], axis=0)  # (8,H,W)
        else:
            feat = img.transpose(2, 0, 1)

        if self.augment:
            if np.random.random() > 0.5:
                feat = np.flip(feat, axis=2).copy()
                mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                feat = np.flip(feat, axis=1).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                feat = np.rot90(feat, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k).copy()

        # normalize RGB after aug bookkeeping
        feat[:3] = (feat[:3] - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(feat.astype(np.float32)), torch.from_numpy(mask)


def iou_dice(pred, target, smooth=1e-7):
    pf, tf = (pred == 1), (target == 1)
    inter = (pf & tf).sum().item()
    union = (pf | tf).sum().item()
    iou = (inter + smooth) / (union + smooth)
    dice = (2 * inter + smooth) / (pf.sum().item() + tf.sum().item() + smooth)
    return iou, dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        p_t = torch.softmax(pred, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (self.alpha * (1 - p_t) ** self.gamma * ce).mean()


def run(name, use_dem, train_items, val_items, test_items):
    in_ch = 8 if use_dem else 3
    train_ds = FusionDataset(train_items, use_dem, augment=True)
    val_ds = FusionDataset(val_items, use_dem, augment=False)
    test_ds = FusionDataset(test_items, use_dem, augment=False)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=6, pin_memory=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

    model = smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet",
                          in_channels=in_ch, classes=NUM_CLASSES).to(DEVICE)
    focal, dice = FocalLoss(), smp.losses.DiceLoss(mode="multiclass")
    loss_fn = lambda p, t: 0.5 * focal(p, t) + 0.5 * dice(p, t)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_iou, best_ep, bad = 0.0, 0, 0
    ckpt = OUT_DIR / f"segformer_b2_{name}_best.pt"
    history = []
    print(f"\n===== {name} (in_channels={in_ch}) =====")
    print(f"{'Ep':>3} {'TrLoss':>8} {'VaLoss':>8} {'VaIoU':>7} {'VaDice':>7} {'s':>5}")
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl = 0
        for x, m in train_ld:
            x, m = x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), m)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(train_ld)
        sched.step()

        model.eval()
        vl, ious, dices = 0, [], []
        with torch.no_grad():
            for x, m in val_ld:
                x, m = x.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
                out = model(x)
                vl += loss_fn(out, m).item()
                for p, t in zip(out.argmax(1), m):
                    i, d = iou_dice(p.cpu(), t.cpu())
                    ious.append(i); dices.append(d)
        vl /= len(val_ld)
        viou, vdice = np.mean(ious), np.mean(dices)
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl,
                        "val_iou": viou, "val_dice": vdice})
        improved = viou > best_iou
        if improved:
            best_iou, best_ep, bad = viou, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"{ep:>3} {tl:>8.4f} {vl:>8.4f} {viou:>7.4f} {vdice:>7.4f} "
              f"{time.time()-t0:>4.0f}{' *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            print(f"early stop @ {ep} (best {best_ep}, IoU {best_iou:.4f})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for x, m in test_ld:
            x = x.to(DEVICE, non_blocking=True)
            for p, t in zip(model(x).argmax(1), m):
                i, d = iou_dice(p.cpu(), t.cpu())
                ious.append(i); dices.append(d)
    res = {"name": name, "use_dem": use_dem, "n_train": len(train_items),
           "n_val": len(val_items), "n_test": len(test_items),
           "best_val_iou": best_iou, "best_epoch": best_ep,
           "test_iou": float(np.mean(ious)), "test_iou_std": float(np.std(ious)),
           "test_dice": float(np.mean(dices)), "test_dice_std": float(np.std(dices))}
    with open(OUT_DIR / f"terrain_ab_{name}.json", "w") as f:
        json.dump({"result": res, "history": history}, f, indent=2)
    print(f"TEST {name}: IoU {res['test_iou']:.4f}±{res['test_iou_std']:.4f}  "
          f"Dice {res['test_dice']:.4f}")
    return res


def main():
    train_items = collect_split("train")
    val_items = collect_split("valid")
    test_items = collect_split("test")
    print(f"Subset with terrain: train={len(train_items)} val={len(val_items)} "
          f"test={len(test_items)}")
    results = []
    results.append(run("rgb_only_subset", False, train_items, val_items, test_items))
    results.append(run("terrain8", True, train_items, val_items, test_items))
    print("\n===== SUMMARY =====")
    for r in results:
        print(f"{r['name']:>18}: test IoU {r['test_iou']:.4f} (val {r['best_val_iou']:.4f})")


if __name__ == "__main__":
    main()
