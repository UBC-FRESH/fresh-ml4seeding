#!/usr/bin/env python3
"""
Train a model with RGB + DEM features for microsite classification.
Uses the high-resolution DEM from COLMAP sparse reconstruction.
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
OUTPUT_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation")
DEM_DIR = Path("/tmp")

NUM_CLASSES = 2
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DEMFusionDataset(Dataset):
    """Dataset that loads RGB images + DEM features."""
    
    def __init__(self, data_dir, split, augment=False):
        self.img_dir = data_dir / split
        self.images = sorted([p for p in self.img_dir.glob("*.jpg") if not p.name.endswith("_mask.png")])
        self.augment = augment
        
        # Load the DEM features
        self.dem_elevation = np.load(DEM_DIR / "dem_elevation_hires.npy")
        self.dem_slope = np.load(DEM_DIR / "dem_slope_hires.npy")
        self.dem_curvature = np.load(DEM_DIR / "dem_curvature_hires.npy")
        self.dem_roughness = np.load(DEM_DIR / "dem_roughness_hires.npy")
        
        print(f"  DEM features loaded: elevation={self.dem_elevation.shape}, slope={self.dem_slope.shape}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = img_path.parent / f"{img_path.stem}_mask.png"
        
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path)).astype(np.int64)
        
        # Get the DEM features for this image
        # The DEM is 1088x1014, the image is 1024x1024
        # Resize the DEM features to match the image size
        from scipy.ndimage import zoom
        
        # Stack the DEM features: [elevation, slope, curvature, roughness]
        dem_features = np.stack([
            self.dem_elevation,
            self.dem_slope,
            self.dem_curvature,
            self.dem_roughness
        ], axis=-1)
        
        # Resize to match image size (1024x1024)
        scale_h = img.shape[0] / dem_features.shape[0]
        scale_w = img.shape[1] / dem_features.shape[1]
        dem_resized = zoom(dem_features, (scale_h, scale_w, 1), order=1)
        
        # Normalize DEM features to 0-1 range
        dem_min = dem_resized.min(axis=(0, 1), keepdims=True)
        dem_max = dem_resized.max(axis=(0, 1), keepdims=True)
        dem_norm = (dem_resized - dem_min) / (dem_max - dem_min + 1e-8)
        
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
                dem_norm = np.fliplr(dem_norm).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
                dem_norm = np.flipud(dem_norm).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
                dem_norm = np.rot90(dem_norm, k).copy()
        
        # Normalize image
        img = (img - MEAN) / STD
        
        # Concatenate image and DEM features: 3 RGB + 4 DEM = 7 channels
        combined = np.concatenate([img, dem_norm], axis=-1).astype(np.float32)  # (H, W, 7)
        combined = combined.transpose(2, 0, 1)  # (7, H, W)
        
        return torch.from_numpy(combined), torch.from_numpy(mask)


def compute_iou_binary(pred, target, smooth=1e-7):
    pred_fg = (pred == 1)
    target_fg = (target == 1)
    intersection = (pred_fg & target_fg).sum().item()
    union = (pred_fg | target_fg).sum().item()
    return (intersection + smooth) / (union + smooth)


def compute_dice(pred, target, smooth=1e-7):
    pred_fg = (pred == 1).float()
    target_fg = (target == 1).float()
    intersection = (pred_fg * target_fg).sum().item()
    return (2 * intersection + smooth) / (pred_fg.sum().item() + target_fg.sum().item() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce = nn.functional.cross_entropy(pred, target, reduction='none')
        p = torch.softmax(pred, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal = self.alpha * (1 - p_t) ** self.gamma * ce
        return focal.mean()


def main():
    print("=" * 60)
    print("DEM Fusion Experiment: RGB + High-Resolution DEM Features")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Load datasets
    train_ds = DEMFusionDataset(DATA_DIR, "train", augment=True)
    val_ds = DEMFusionDataset(DATA_DIR, "valid", augment=False)
    test_ds = DEMFusionDataset(DATA_DIR, "test", augment=False)
    
    print(f"\nData: train={len(train_ds)}, valid={len(val_ds)}, test={len(test_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Use SegFormer with 7 input channels (3 RGB + 4 DEM)
    model = smp.Segformer(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=7,  # 3 RGB + 4 DEM features
        classes=NUM_CLASSES,
    ).to(DEVICE)
    
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    
    def loss_fn(pred, target):
        return 0.5 * focal_loss(pred, target) + 0.5 * dice_loss(pred, target)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_val_iou = 0.0
    best_epoch = 0
    patience = 15
    patience_counter = 0
    
    print(f"\nTraining: SegFormer-B2 with 7-channel input (RGB + DEM)")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>8} {'ValIoU':>8} {'ValDice':>8} {'Time':>6}")
    print("-" * 55)
    
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        
        model.train()
        train_loss = 0
        for combined, masks in train_loader:
            combined = combined.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(combined)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        val_ious = []
        val_dices = []
        with torch.no_grad():
            for combined, masks in val_loader:
                combined = combined.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                
                preds = model(combined)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()
                pred_classes = preds.argmax(dim=1)
                for p, t in zip(pred_classes, masks):
                    p_cpu, t_cpu = p.cpu(), t.cpu()
                    val_ious.append(compute_iou_binary(p_cpu, t_cpu))
                    val_dices.append(compute_dice(p_cpu, t_cpu))
        
        val_loss /= len(val_loader)
        val_iou = np.mean(val_ious)
        val_dice = np.mean(val_dices)
        elapsed = time.time() - t0
        
        improved = val_iou > best_val_iou
        if improved:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "segformer_b2_dem_fusion_best.pt")
        else:
            patience_counter += 1
        
        marker = " ★" if improved else ""
        print(f"{epoch:>5} {train_loss:>10.4f} {val_loss:>8.4f} {val_iou:>8.4f} {val_dice:>8.4f} {elapsed:>5.0f}s{marker}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch}, IoU={best_val_iou:.4f})")
            break
    
    # Test evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "segformer_b2_dem_fusion_best.pt", map_location=DEVICE))
    model.eval()
    test_ious = []
    test_dices = []
    with torch.no_grad():
        for combined, masks in test_loader:
            combined = combined.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            preds = model(combined)
            pred_classes = preds.argmax(dim=1)
            for p, t in zip(pred_classes, masks):
                p_cpu, t_cpu = p.cpu(), t.cpu()
                test_ious.append(compute_iou_binary(p_cpu, t_cpu))
                test_dices.append(compute_dice(p_cpu, t_cpu))
    
    test_iou = np.mean(test_ious)
    test_dice = np.mean(test_dices)
    
    print(f"\n  Test IoU: {test_iou:.4f} ± {np.std(test_ious):.4f}")
    print(f"  Test Dice: {test_dice:.4f} ± {np.std(test_dices):.4f}")
    print(f"  Best val IoU: {best_val_iou:.4f} (epoch {best_epoch})")
    
    # Save result
    result = {
        "model": "segformer_b2_dem_fusion",
        "best_val_iou": best_val_iou,
        "test_iou": test_iou,
        "test_iou_std": np.std(test_ious),
        "test_dice": test_dice,
        "test_dice_std": np.std(test_dices),
        "best_epoch": best_epoch,
        "notes": "SegFormer-B2 with 7-channel input (RGB + elevation, slope, curvature, roughness from 1cm DEM)"
    }
    
    with open(OUTPUT_DIR / "dem_fusion_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResult saved to {OUTPUT_DIR / 'dem_fusion_result.json'}")


if __name__ == "__main__":
    main()
