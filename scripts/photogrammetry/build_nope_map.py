#!/usr/bin/env python3
"""
Build the site-level "nope map" (geometric veto layer) from the COLMAP
full-flight reconstruction, plus per-image masks for all 750 frames and
stretched versions for the 548 v4 training images.

Outputs (project dir, NOT /tmp):
  annotation/nope_map/site_nope.npz      - hag/rough/valid/bump/pit/nope grids + extent
  annotation/nope_map/site_nope_map.png  - visualization (blue=pit, red=bump)
  annotation/nope_map/per_image/<frame>.png      - 1024x682 mask (0=ok,1=bump,2=pit)
  annotation/nope_map/per_image_v4/<v4stem>.png  - 1024x1024 stretched mask for v4 images
  annotation/nope_map/nope_qc_montage.png        - QC overlay montage
  annotation/nope_map/metadata.json
"""

import struct
import re
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy import ndimage
import cv2

SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
V4_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4")
OUT = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map")
(OUT / "per_image").mkdir(parents=True, exist_ok=True)
(OUT / "per_image_v4").mkdir(parents=True, exist_ok=True)

GRID_RES = 0.02          # in COLMAP local units (NOT meters!)
IMG_W, IMG_H = 1024, 682

# True scale from GPS georeferencing (Umeyama fit, see georeference_nope_map.py)
GEOJSON = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map/georeferencing.json")
SCALE = 8.7911  # m per local unit; fallback, overridden by georeferencing.json
if GEOJSON.exists():
    SCALE = json.load(open(GEOJSON))["scale_m_per_unit"]

# Thresholds in REAL meters, converted to local units
# (h=0.15, r=0.05 validated best on the 353-object GT: slash 87%, log 90%, rock 60%)
H_BUMP_REAL, H_PIT_REAL, R_ROUGH_REAL = 0.15, -0.20, 0.05
DTM_WIN_REAL, ROUGH_WIN_REAL = 1.5, 0.35
H_BUMP = H_BUMP_REAL / SCALE
H_PIT = H_PIT_REAL / SCALE
R_ROUGH = R_ROUGH_REAL / SCALE


def read_cameras(path):
    with open(path, "rb") as f:
        struct.unpack("Q", f.read(8))
        struct.unpack("i", f.read(4)); struct.unpack("i", f.read(4))
        w = struct.unpack("Q", f.read(8))[0]; h = struct.unpack("Q", f.read(8))[0]
        params = struct.unpack("d" * 8, f.read(64))
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


def main():
    print("Loading model...")
    cam = read_cameras(SPARSE / "cameras.bin")
    images = read_images(SPARSE / "images.bin")
    points = read_points3d(SPARSE / "points3D.bin")
    print(f"  {len(images)} images, {len(points)} points")

    # Plane align (normal toward cameras)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    cc = np.array([-qvec2rotmat(i["qvec"]).T @ i["tvec"] for i in images.values()])
    if np.dot(normal, cc.mean(axis=0) - centroid) < 0:
        normal = -normal
    v = np.cross(normal, [0, 0, 1.]); c = np.dot(normal, [0, 0, 1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    pa = (points - centroid) @ R_align.T

    # Ground grid
    x_lo, x_hi = np.percentile(pa[:, 0], [0.5, 99.5])
    y_lo, y_hi = np.percentile(pa[:, 1], [0.5, 99.5])
    nx = int((x_hi - x_lo) / GRID_RES) + 1
    ny = int((y_hi - y_lo) / GRID_RES) + 1
    gx = np.linspace(x_lo, x_hi, nx)
    gy = np.linspace(y_lo, y_hi, ny)
    gxx, gyy = np.meshgrid(gx, gy)
    keep = (pa[:, 2] >= -0.5) & (pa[:, 2] <= 2.0)
    print(f"  grid {nx}x{ny} @ {GRID_RES*100:.0f}cm, points {keep.sum()}")

    print("  gridding DSM/DTM/HAG/roughness...")
    dsm = griddata(pa[keep][:, :2], pa[keep][:, 2], (gxx, gyy), method="linear")
    valid = ~np.isnan(dsm)
    dsm_nn = griddata(pa[keep][:, :2], pa[keep][:, 2], (gxx, gyy), method="nearest")
    dsm = np.where(valid, dsm, dsm_nn)
    res_real = GRID_RES * SCALE  # real meters per cell
    k = max(3, int(DTM_WIN_REAL / res_real))
    dtm = ndimage.gaussian_filter(ndimage.percentile_filter(dsm, 10, size=k), 2)
    hag = dsm - dtm
    kr = max(2, int(ROUGH_WIN_REAL / res_real))
    rough = np.sqrt(np.maximum(
        ndimage.uniform_filter(dsm**2, size=kr)
        - ndimage.uniform_filter(dsm, size=kr)**2, 0))
    print(f"  scale {SCALE:.3f} m/unit -> cell {res_real*100:.1f} cm | "
          f"DTM window {k} cells ({k*res_real:.1f} m) | rough window {kr} cells")

    # Site masks
    bump = (hag > H_BUMP) | (rough > R_ROUGH)
    pit = hag < H_PIT
    eroded_valid = ndimage.binary_erosion(valid, iterations=5)
    bump &= eroded_valid
    pit &= eroded_valid
    nope = bump | pit

    nv = valid.sum()
    cell_m2 = res_real ** 2
    print(f"  site: valid {valid.mean()*100:.0f}% | bump {bump.sum()/nv*100:.1f}% ({bump.sum()*cell_m2:.0f} m2) "
          f"| pit {pit.sum()/nv*100:.2f}% ({pit.sum()*cell_m2:.1f} m2) | NOPE {nope.sum()/nv*100:.1f}% of valid")
    if pit.sum():
        plab, pn = ndimage.label(pit)
        psizes = np.array(ndimage.sum(pit, plab, range(1, pn + 1))) * cell_m2
        big = psizes[psizes > 0.5]
        print(f"  pits >0.5 m2: {len(big)}, largest: {big.max() if len(big) else 0:.2f} m2, "
          f"deepest HAG: {hag[pit].min()*SCALE:.2f} m")

    np.savez_compressed(
        OUT / "site_nope.npz",
        hag=hag.astype(np.float32), rough=rough.astype(np.float32),
        hag_meters=(hag * SCALE).astype(np.float32),
        rough_meters=(rough * SCALE).astype(np.float32),
        valid=valid, bump=bump, pit=pit, nope=nope,
        x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi, res=GRID_RES,
        scale_m_per_unit=SCALE, cell_size_m=res_real,
        normal=normal, centroid=centroid, R_align=R_align)

    # Site visualization (HAG scaled in REAL meters)
    hag_m = hag * SCALE
    h_n = np.clip((hag_m + 0.4) / 1.2, 0, 1)
    vis = cv2.applyColorMap((h_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[~valid] = 40
    vis[pit] = [255, 128, 0]       # blue-ish = pit/ditch
    vis[hag_m > 0.5] = [0, 0, 255]  # red = strong bump >0.5 m
    scale = 1600 / nx
    cv2.imwrite(str(OUT / "site_nope_map.png"),
                cv2.resize(vis, (1600, int(ny * scale)), interpolation=cv2.INTER_NEAREST))

    # Per-image rendering
    fx, fy, cx, cy = cam["params"][:4]
    dist = cam["params"][4:]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    uu, vv = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    pix = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64).reshape(1, -1, 2)
    rays = cv2.undistortPoints(pix, K, dist).reshape(IMG_H, IMG_W, 2)
    rays = np.dstack([rays, np.ones((IMG_H, IMG_W))])

    hag_f = hag.astype(np.float32)
    rough_f = rough.astype(np.float32)
    valid_f = valid.astype(np.float32)

    # v4 name mapping
    v4_map = {}  # frame name -> v4 stem
    for split in ["train", "valid", "test"]:
        for p in (V4_DIR / split).glob("*.jpg"):
            if p.name.endswith("_mask.png"):
                continue
            base = re.sub(r"\.rf\.[a-f0-9]+", "", p.name).replace("_JPG.jpg", ".JPG")
            m = re.match(r"a58_\d+_(?:early|mid|late)_(\d+)_jpg", base.replace(".JPG", ""))
            if m:
                cand = [n for n in images if n.endswith(f"_{int(m.group(1)):04d}.JPG")]
                base = cand[0] if cand else None
            if base:
                v4_map.setdefault(base, p.stem)

    coverages = []
    qc_rows = []
    qc_targets = {"DJI_20260202044221_0743.JPG", "DJI_20260202042659_0203.JPG",
                  "DJI_20260202043500_0450.JPG", "DJI_20260202043000_0300.JPG"}
    print("  rendering per-image masks...")
    for idx, (name, im) in enumerate(sorted(images.items())):
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
            z_here = cv2.remap(hag_f, ix, iy, cv2.INTER_LINEAR)
            s = (z_here - C_a[2]) / np.clip(rays_w[..., 2], -1e9, -1e-6)
        P = C_a + rays_w * s[..., None]
        mx = ((P[..., 0] - x_lo) / GRID_RES).astype(np.float32)
        my = ((P[..., 1] - y_lo) / GRID_RES).astype(np.float32)

        hag_i = cv2.remap(hag_f, mx, my, cv2.INTER_LINEAR)
        rough_i = cv2.remap(rough_f, mx, my, cv2.INTER_LINEAR)
        valid_i = (cv2.remap(valid_f, mx, my, cv2.INTER_NEAREST) > 0.5) & \
                  (rays_w[..., 2] < -1e-6)

        mask = np.zeros((IMG_H, IMG_W), np.uint8)
        mask[((hag_i > H_BUMP) | (rough_i > R_ROUGH)) & valid_i] = 1
        mask[(hag_i < H_PIT) & valid_i] = 2
        mask[~valid_i] = 255
        cv2.imwrite(str(OUT / "per_image" / f"{Path(name).stem}.png"), mask)

        if name in v4_map:
            mask1024 = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(OUT / "per_image_v4" / f"{v4_map[name]}.png"), mask1024)

        vv_ = valid_i.sum()
        if vv_:
            coverages.append((mask == 1).sum() / vv_ + (mask == 2).sum() / vv_)

        if name in qc_targets or idx % 200 == 0:
            small = cv2.imread(str(SPARSE.parent.parent / "images" / name))
            if small is not None:
                rgb = cv2.resize(small, (512, 341))
                m = cv2.resize(mask, (512, 341), interpolation=cv2.INTER_NEAREST)
                ovl = rgb.copy()
                ovl[m == 1] = (0.5 * ovl[m == 1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
                ovl[m == 2] = (0.5 * ovl[m == 2] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)
                ovl[m == 255] = (ovl[m == 255] * 0.3).astype(np.uint8)
                qc_rows.append(np.concatenate([rgb, ovl], axis=1))

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/750")

    if qc_rows:
        cv2.imwrite(str(OUT / "nope_qc_montage.png"), np.concatenate(qc_rows[:8], axis=0))

    meta = {
        "source_model": "/tmp/colmap_a58_full/sparse/0 (750 frames, 715k points)",
        "scale_m_per_unit": SCALE,
        "grid_cell_m": res_real,
        "site_extent_m": [(x_hi - x_lo) * SCALE, (y_hi - y_lo) * SCALE],
        "thresholds_real_m": {"bump_hag": H_BUMP_REAL, "pit_hag": H_PIT_REAL,
                              "rough": R_ROUGH_REAL, "dtm_window": DTM_WIN_REAL},
        "mask_values": {"0": "ok", "1": "bump (stump/log/rock/slash-pile)",
                        "2": "pit (hole/ditch/future-pond)", "255": "no data"},
        "n_images": len(images),
        "n_v4_mapped": len(v4_map),
        "mean_nope_fraction_per_image": float(np.mean(coverages)),
        "note": "Grid stored in COLMAP local units; multiply by scale_m_per_unit "
                "for meters. Georeferenced product: site_nope_utm10.tif "
                "(GPS-EXIF similarity alignment, horiz RMS ~0.3 m).",
    }
    with open(OUT / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Mean nope coverage per image: {np.mean(coverages)*100:.1f}%")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    main()
