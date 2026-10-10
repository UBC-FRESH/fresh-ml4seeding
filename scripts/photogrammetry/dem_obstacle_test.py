#!/usr/bin/env python3
"""
Test: is the SfM DEM good enough to detect slash/logs/rocks/stumps?

Ground truth: user's 6 annotated feature images (a58_features_v3, COCO).
Method: canopy-height-model style — height above local ground (low-percentile
DTM estimate) + roughness, threshold, connected components, object-level
recall/precision vs the annotated polygons.
"""

import struct
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy import ndimage
import cv2
import json
import re
from pycocotools import mask as mask_utils

SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
COCO = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_features_v3/train/_annotations.coco.json")
IMG_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_features_v3/train")
OUT = Path("/tmp/dem_obstacle_test")
OUT.mkdir(exist_ok=True)

GRID_RES = 0.02  # COLMAP local units
IMG_W, IMG_H = 1024, 682
SCALE = 8.7911   # m per local unit (from GPS georeferencing)


def read_cameras(path):
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        cam_id = struct.unpack("i", f.read(4))[0]
        model_id = struct.unpack("i", f.read(4))[0]
        w = struct.unpack("Q", f.read(8))[0]
        h = struct.unpack("Q", f.read(8))[0]
        nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model_id]
        params = struct.unpack("d" * nparams, f.read(8 * nparams))
    return {"w": w, "h": h, "params": np.array(params)}


def read_images(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            struct.unpack("i", f.read(4))
            qvec = np.array(struct.unpack("dddd", f.read(32)))
            tvec = np.array(struct.unpack("ddd", f.read(24)))
            struct.unpack("i", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            tl = struct.unpack("Q", f.read(8))[0]
            f.read(tl * 24)
            images[name.decode()] = {"qvec": qvec, "tvec": tvec}
    return images


def read_points3d(path):
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            pts.append(struct.unpack("ddd", f.read(24)))
            f.read(11)
            tl = struct.unpack("Q", f.read(8))[0]
            f.read(tl * 8)
    return np.array(pts)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


print("Loading model...")
cam = read_cameras(SPARSE / "cameras.bin")
images = read_images(SPARSE / "images.bin")
points = read_points3d(SPARSE / "points3D.bin")

# Plane align, normal toward cameras
centroid = points.mean(axis=0)
_, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
normal = vt[2]
cam_centers = np.array([-qvec2rotmat(im["qvec"]).T @ im["tvec"] for im in images.values()])
if np.dot(normal, cam_centers.mean(axis=0) - centroid) < 0:
    normal = -normal
v = np.cross(normal, [0, 0, 1.0]); c = np.dot(normal, [0, 0, 1.0])
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
pts_a = (points - centroid) @ R_align.T

# Ground grid — keep objects up to +2 m this time (they ARE the signal)
x_lo, x_hi = np.percentile(pts_a[:, 0], [0.5, 99.5])
y_lo, y_hi = np.percentile(pts_a[:, 1], [0.5, 99.5])
z_lo, z_hi = -0.5, 2.0
keep = (pts_a[:, 2] >= z_lo) & (pts_a[:, 2] <= z_hi)
pts_g = pts_a[keep]
nx = int((x_hi - x_lo) / GRID_RES) + 1
ny = int((y_hi - y_lo) / GRID_RES) + 1
gx = np.linspace(x_lo, x_hi, nx)
gy = np.linspace(y_lo, y_hi, ny)
gxx, gyy = np.meshgrid(gx, gy)
print(f"Grid {nx}x{ny}, points kept {keep.sum()}/{len(keep)}")

print("Gridding DSM (all surface points, incl. object tops)...")
dsm = griddata(pts_g[:, :2], pts_g[:, 2], (gxx, gyy), method="linear")
valid_grid = ~np.isnan(dsm)
dsm_nn = griddata(pts_g[:, :2], pts_g[:, 2], (gxx, gyy), method="nearest")
dsm = np.where(valid_grid, dsm, dsm_nn)

# DTM via low percentile over 1.5 m (real) window, then height above ground
k = max(3, int(1.5 / (GRID_RES * SCALE)))
print(f"DTM: 10th-percentile filter over {k}x{k} cells ({k*GRID_RES*SCALE:.1f} m real)...")
dtm = ndimage.percentile_filter(dsm, 10, size=k)
dtm = ndimage.gaussian_filter(dtm, sigma=2)
hag = (dsm - dtm) * SCALE  # HAG in REAL meters
kr = max(2, int(0.35 / (GRID_RES * SCALE)))
rough = np.sqrt(np.maximum(
    ndimage.uniform_filter(dsm**2, size=kr)
    - ndimage.uniform_filter(dsm, size=kr)**2, 0)) * SCALE  # real meters
print(f"HAG range: {hag.min():.2f}..{hag.max():.2f} m; rough max {rough.max():.3f} m")

# Load ground truth
coco = json.load(open(COCO))
cat_names = {c["id"]: c["name"] for c in coco["categories"]}
gt = {}
for im in coco["images"]:
    anns = [a for a in coco["annotations"] if a["image_id"] == im["id"]]
    gt[im["file_name"]] = anns

# Camera + rays
fx, fy, cx, cy = cam["params"][:4]
dist = cam["params"][4:]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
pix = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64).reshape(1, -1, 2)
rays = cv2.undistortPoints(pix, K, dist).reshape(IMG_H, IMG_W, 2)
rays = np.dstack([rays, np.ones((IMG_H, IMG_W))])

H_THRS = [0.15, 0.25, 0.35]   # real meters
R_THRS = [0.05, 0.10]         # real meters
MIN_AREA_PX = 30  # ~ 45x8 cm object at ~15mm/px GSD

all_results = []
montage_rows = []
for fname, anns in sorted(gt.items()):
    base = re.sub(r"\.rf\.[a-f0-9]+", "", fname).replace("_JPG.jpg", ".JPG")
    im = images[base]
    R_w2c = qvec2rotmat(im["qvec"])
    t = im["tvec"]
    R_c2w = R_w2c.T
    C_a = R_align @ ((-R_c2w @ t) - centroid)
    R_a = R_align @ R_c2w
    rays_w = rays @ R_a.T

    s = -C_a[2] / np.clip(rays_w[..., 2], -1e9, -1e-6)
    for _ in range(3):
        P = C_a + rays_w * s[..., None]
        ix = np.clip((P[..., 0] - x_lo) / GRID_RES, 0, nx - 1).astype(np.float32)
        iy = np.clip((P[..., 1] - y_lo) / GRID_RES, 0, ny - 1).astype(np.float32)
        z_here = cv2.remap(dsm.astype(np.float32), ix, iy, cv2.INTER_LINEAR)
        s = (z_here - C_a[2]) / np.clip(rays_w[..., 2], -1e9, -1e-6)
    P = C_a + rays_w * s[..., None]
    mx = ((P[..., 0] - x_lo) / GRID_RES).astype(np.float32)
    my = ((P[..., 1] - y_lo) / GRID_RES).astype(np.float32)
    in_fp = (mx >= 0) & (mx < nx-1) & (my >= 0) & (my < ny-1) & (rays_w[..., 2] < -1e-6)
    hag_img = cv2.remap(hag.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    rough_img = cv2.remap(rough.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    valid_img = cv2.remap(valid_grid.astype(np.float32), mx, my, cv2.INTER_NEAREST) > 0.5
    valid_img &= in_fp

    # stretch to 1024x1024 (Roboflow v4 geometry)
    hag_img = cv2.resize(hag_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    rough_img = cv2.resize(rough_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    valid_img = cv2.resize(valid_img.astype(np.float32), (1024, 1024),
                           interpolation=cv2.INTER_NEAREST) > 0.5

    # ground-truth instance masks
    ann_masks = []
    for a in anns:
        seg = a["segmentation"]
        if isinstance(seg, dict):  # RLE
            m = mask_utils.decode(seg).astype(np.uint8)
        else:
            pts = np.array(seg[0]).reshape(-1, 2).astype(np.int32)
            m = np.zeros((1024, 1024), np.uint8)
            cv2.fillPoly(m, [pts], 1)
        ann_masks.append((cat_names[a["category_id"]], m))

    for h_thr in H_THRS:
        for r_thr in R_THRS:
            det = ((hag_img > h_thr) | (rough_img > r_thr)) & valid_img
            det = ndimage.binary_opening(det, iterations=1)
            det = ndimage.binary_closing(det, iterations=2)
            lab, n = ndimage.label(det)
            sizes = ndimage.sum(det, lab, range(1, n+1))
            det = np.isin(lab, np.where(sizes >= MIN_AREA_PX)[0] + 1)
            lab, n = ndimage.label(det)

            # object-level recall per class
            per_class = {}
            for cls, m in ann_masks:
                hit = 0
                for bid in np.unique(lab[m > 0]):
                    if bid == 0: continue
                    inter = ((lab == bid) & (m > 0)).sum()
                    if inter >= 0.2 * m.sum():
                        hit = 1
                        break
                per_class.setdefault(cls, []).append(hit)
            # precision: blob overlaps any annotation by >=20% of blob area
            any_ann = np.maximum.reduce([m for _, m in ann_masks])
            tp = fp = 0
            for bid in range(1, n+1):
                blob = lab == bid
                inter = (blob & (any_ann > 0)).sum()
                if inter >= 0.2 * blob.sum():
                    tp += 1
                else:
                    fp += 1
            row = {"image": fname[:30], "h_thr": h_thr, "r_thr": r_thr,
                   "n_blobs": n, "tp": tp, "fp": fp,
                   "precision": tp / max(tp+fp, 1),
                   "recall": {cls: (float(np.mean(v)), len(v))
                              for cls, v in per_class.items()}}
            all_results.append(row)

    # montage for h=0.10, r=0.04
    det = ((hag_img > 0.10) | (rough_img > 0.04)) & valid_img
    det = ndimage.binary_closing(ndimage.binary_opening(det, iterations=1), iterations=2)
    lab, n = ndimage.label(det)
    sizes = ndimage.sum(det, lab, range(1, n+1))
    det = np.isin(lab, np.where(sizes >= MIN_AREA_PX)[0] + 1)

    # "nope" pits: holes / ditch segments / future ponds (negative anomalies)
    pit = (hag_img < -0.08) & valid_img
    pit = ndimage.binary_closing(ndimage.binary_opening(pit, iterations=2), iterations=3)
    plab, pn = ndimage.label(pit)
    psizes = ndimage.sum(pit, plab, range(1, pn+1))
    pit = np.isin(plab, np.where(psizes >= 500)[0] + 1)  # >= ~0.1 m^2 at 5mm/px
    n_pits = int((psizes >= 500).sum())
    pit_area = float(pit.sum() * (4.5/1024)**2)  # rough m^2 assuming ~4.5m footprint
    print(f"  {fname[:30]}: pits>0.1m2={n_pits}, total pit area ~{pit_area:.2f} m^2")

    rgb = cv2.imread(str(IMG_DIR / fname))
    rgb = cv2.resize(rgb, (512, 512))
    vis = rgb.copy()
    det512 = cv2.resize(det.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
    vis[det512] = (0.4*vis[det512] + 0.6*np.array([0,255,0])).astype(np.uint8)
    pit512 = cv2.resize(pit.astype(np.uint8), (512,512), interpolation=cv2.INTER_NEAREST) > 0
    vis[pit512] = (0.4*vis[pit512] + 0.6*np.array([255,128,0])).astype(np.uint8)
    any_ann = np.maximum.reduce([m for _, m in ann_masks])
    missed = (any_ann > 0) & ~det
    missed512 = cv2.resize(missed.astype(np.uint8), (512,512), interpolation=cv2.INTER_NEAREST) > 0
    vis[missed512] = (0.4*vis[missed512] + 0.6*np.array([0,0,255])).astype(np.uint8)
    h_vis = cv2.applyColorMap(
        (np.clip(cv2.resize(hag_img, (512,512)), -0.1, 0.5) / 0.6 * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    montage_rows.append(np.concatenate([rgb, h_vis, vis], axis=1))

# Aggregate
print("\n===== AGGREGATE (6 images) =====")
print(f"{'h_thr':>6} {'r_thr':>6} {'blobs':>6} {'prec':>6} | recall per class")
agg = {}
for r in all_results:
    key = (r["h_thr"], r["r_thr"])
    a = agg.setdefault(key, {"blobs": 0, "tp": 0, "fp": 0, "cls": {}})
    a["blobs"] += r["n_blobs"]; a["tp"] += r["tp"]; a["fp"] += r["fp"]
    for cls, (rec, n) in r["recall"].items():
        a["cls"].setdefault(cls, []).extend([rec]*n)
for key, a in sorted(agg.items()):
    cls_str = "  ".join(f"{cls}:{np.mean(v):.2f}(n={len(v)})"
                        for cls, v in sorted(a["cls"].items()))
    print(f"{key[0]:>6} {key[1]:>6} {a['blobs']:>6} {a['tp']/max(a['tp']+a['fp'],1):>6.2f} | {cls_str}")

mont = np.concatenate(montage_rows, axis=0)
cv2.imwrite(str(OUT / "montage.png"), mont)
print(f"\nMontage: {OUT/'montage.png'} (RGB | height-above-ground | green=det, red=missed GT)")
with open(OUT / "results.json", "w") as f:
    json.dump(all_results, f, indent=2)
