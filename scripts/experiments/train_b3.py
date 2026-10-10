#!/usr/bin/env python3
"""SegFormer-B3 on v4, identical recipe to the proven baseline."""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import json
import time

DATA = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4")
OUT = Path("/home/gep/projects/ignitebc-flashforest/annotation")
BATCH, EPOCHS, PATIENCE, LR = 8, 60, 12, 3e-4
DEVICE = torch.device("cuda")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DS(Dataset):
    def __init__(self, split, augment=False):
        self.imgs = sorted(p for p in (DATA / split).glob("*.jpg") if not p.name.endswith("_mask.png"))
        self.split, self.augment = split, augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(p.parent / f"{p.stem}_mask.png")).astype(np.int64)
        img = img.transpose(2, 0, 1)
        if self.augment:
            if np.random.random() > 0.5:
                img = np.flip(img, 2).copy(); mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                img = np.flip(img, 1).copy(); mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k, (1, 2)).copy(); mask = np.rot90(mask, k).copy()
        img = (img - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask)


class Focal(nn.Module):
    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, reduction="none")
        p_t = torch.softmax(pred, 1).gather(1, target.unsqueeze(1)).squeeze(1)
        return (0.25 * (1 - p_t) ** 2.0 * ce).mean()


def iou_dice(pred, target, eps=1e-7):
    pf, tf = pred == 1, target == 1
    inter = (pf & tf).sum().item()
    return ((inter + eps) / ((pf | tf).sum().item() + eps),
            (2 * inter + eps) / (pf.sum().item() + tf.sum().item() + eps))


def main():
    train_ld = DataLoader(DS("train", True), batch_size=BATCH, shuffle=True,
                          num_workers=6, pin_memory=True, persistent_workers=True)
    val_ld = DataLoader(DS("valid"), batch_size=BATCH, num_workers=4, pin_memory=True)
    test_ld = DataLoader(DS("test"), batch_size=BATCH, num_workers=4, pin_memory=True)
    print(f"train={len(train_ld.dataset)} valid={len(val_ld.dataset)} test={len(test_ld.dataset)}")

    model = smp.Segformer(encoder_name="mit_b3", encoder_weights="imagenet",
                          in_channels=3, classes=2).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    focal = Focal()
    dice = smp.losses.DiceLoss(mode="multiclass")
    loss_fn = lambda p, t: 0.5 * focal(p, t) + 0.5 * dice(p, t)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)

    best_iou, best_ep, bad = 0.0, 0, 0
    ckpt = OUT / "segformer_b3_best.pt"
    hist = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tl = 0
        for x, m in train_ld:
            x, m = x.to(DEVICE), m.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), m)
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        model.eval()
        vl, ious = 0, []
        with torch.no_grad():
            for x, m in val_ld:
                x, m = x.to(DEVICE), m.to(DEVICE)
                out = model(x)
                vl += loss_fn(out, m).item()
                for p, t in zip(out.argmax(1), m):
                    ious.append(iou_dice(p.cpu(), t.cpu())[0])
        viou = np.mean(ious)
        hist.append({"epoch": ep, "train_loss": tl / len(train_ld), "val_loss": vl / len(val_ld), "val_iou": viou})
        improved = viou > best_iou
        if improved:
            best_iou, best_ep, bad = viou, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"{ep:>3} tl={tl/len(train_ld):.4f} vl={vl/len(val_ld):.4f} "
              f"valIoU={viou:.4f} {time.time()-t0:.0f}s{' *' if improved else ''}", flush=True)
        if bad >= PATIENCE:
            print(f"early stop @{ep} (best {best_ep}: {best_iou:.4f})")
            break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    ious = []
    with torch.no_grad():
        for x, m in test_ld:
            x = x.to(DEVICE)
            for p, t in zip(model(x).argmax(1), m):
                ious.append(iou_dice(p.cpu(), t.cpu())[0])
    print(f"\nTEST b3: IoU {np.mean(ious):.4f} ± {np.std(ious):.4f} (best val {best_iou:.4f} @ep{best_ep})")
    with open(OUT / "segformer_b3_result.json", "w") as f:
        json.dump({"test_iou": float(np.mean(ious)), "test_iou_std": float(np.std(ious)),
                   "best_val_iou": best_iou, "best_epoch": best_ep, "history": hist}, f, indent=2)
    print("saved annotation/segformer_b3_result.json")


if __name__ == "__main__":
    main()
