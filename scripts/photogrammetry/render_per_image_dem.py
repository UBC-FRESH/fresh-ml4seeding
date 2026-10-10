#!/usr/bin/env python3
"""
Render per-image terrain feature maps from a COLMAP sparse reconstruction.

For each registered image:
  1. Back-project every pixel ray onto a plane-aligned ground DEM grid
     (2 fixed-point iterations to correct for relief parallax)
  2. Sample elevation / slope / curvature / roughness from the ground-plane maps
  3. Stretch 1024x682 -> 1024x1024 to match Roboflow v4 preprocessing
  4. Save float16 stack + validity mask per image

Output: /tmp/dem_per_image/<v4_image_stem>.npy  (5, 1024, 1024) float16
        channels: [rel_elevation, slope_deg, curvature, roughness, valid]
"""

import struct
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
from scipy import ndimage
import cv2
import json

COLMAP_SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
V4_DIR = Path("/home/gep/projects/ignitebc-flashforest/annotation/roboflow_v4")
OUT_DIR = Path("/tmp/dem_per_image")
OUT_DIR.mkdir(exist_ok=True)

GRID_RES = 0.02          # 2 cm ground-plane grid
IMG_W_SMALL, IMG_H_SMALL = 1024, 682   # a58_small frames (COLMAP input)
IMG_OUT = 1024                          # v4 training image size


# ---------- COLMAP binary readers ----------
def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            cam_id = struct.unpack("i", f.read(4))[0]
            model_id = struct.unpack("i", f.read(4))[0]
            w = struct.unpack("Q", f.read(8))[0]
            h = struct.unpack("Q", f.read(8))[0]
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model_id]
            params = struct.unpack("d" * nparams, f.read(8 * nparams))
            cams[cam_id] = {"model": model_id, "w": w, "h": h, "params": np.array(params)}
    return cams


def read_images(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("i", f.read(4))[0]
            qvec = np.array(struct.unpack("dddd", f.read(32)))
            tvec = np.array(struct.unpack("ddd", f.read(24)))
            camera_id = struct.unpack("i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_pts2d = struct.unpack("Q", f.read(8))[0]
            f.read(num_pts2d * 24)
            images[name.decode()] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id}
    return images


def read_points3d(path):
    pts = []
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            x, y, z = struct.unpack("ddd", f.read(24))
            f.read(3 + 8)
            track_len = struct.unpack("Q", f.read(8))[0]
            f.read(track_len * 8)
            pts.append([x, y, z])
    return np.array(pts)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def main():
    print("Loading COLMAP model...")
    cams = read_cameras(COLMAP_SPARSE / "cameras.bin")
    images = read_images(COLMAP_SPARSE / "images.bin")
    points = read_points3d(COLMAP_SPARSE / "points3D.bin")
    print(f"  {len(images)} images, {len(points)} 3D points")

    # ---------- 1. Plane-align the point cloud (PCA) ----------
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    # Orient the normal toward the cameras (they are above the ground)
    cam_centers = np.array([-qvec2rotmat(im["qvec"]).T @ im["tvec"]
                            for im in images.values()])
    if np.dot(normal, cam_centers.mean(axis=0) - centroid) < 0:
        normal = -normal
    # Rotation taking normal -> Z axis
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    if np.linalg.norm(v) > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    else:
        R_align = np.eye(3)
    pts_aligned = (points - centroid) @ R_align.T
    print(f"  Plane normal: {normal.round(4)}, aligned Z range: "
          f"{pts_aligned[:,2].min():.3f}..{pts_aligned[:,2].max():.3f}")

    # ---------- 2. Ground-plane DEM grid ----------
    x_lo, x_hi = np.percentile(pts_aligned[:, 0], [0.5, 99.5])
    y_lo, y_hi = np.percentile(pts_aligned[:, 1], [0.5, 99.5])
    nx = int((x_hi - x_lo) / GRID_RES) + 1
    ny = int((y_hi - y_lo) / GRID_RES) + 1
    gx = np.linspace(x_lo, x_hi, nx)
    gy = np.linspace(y_lo, y_hi, ny)
    gxx, gyy = np.meshgrid(gx, gy)
    print(f"  Ground grid: {nx}x{ny} @ {GRID_RES*100:.0f}cm  "
          f"({x_hi-x_lo:.1f}m x {y_hi-y_lo:.1f}m)")

    print("  Interpolating elevation...")
    # Drop Z outliers (bad triangulations) before gridding
    z_lo, z_hi = np.percentile(pts_aligned[:, 2], [0.5, 99.5])
    keep = (pts_aligned[:, 2] >= z_lo) & (pts_aligned[:, 2] <= z_hi)
    print(f"  Z clip [{z_lo:.3f},{z_hi:.3f}] keeps {keep.sum()}/{len(keep)} points")
    pts_g = pts_aligned[keep]
    elev = griddata(pts_g[:, :2], pts_g[:, 2], (gxx, gyy), method="linear")
    valid_grid = ~np.isnan(elev)
    # Fill NaN inside the surveyed region with nearest-neighbour
    elev_nn = griddata(pts_g[:, :2], pts_g[:, 2], (gxx, gyy), method="nearest")
    elev = np.clip(np.where(valid_grid, elev, elev_nn), z_lo, z_hi)

    # ---------- 3. Terrain features on the ground plane (metric) ----------
    # relative elevation vs 0.5 m local mean
    k_rel = int(0.5 / GRID_RES)
    rel_elev = elev - ndimage.uniform_filter(elev, size=k_rel)
    # slope
    gy_grad, gx_grad = np.gradient(elev, GRID_RES, GRID_RES)
    slope = np.degrees(np.arctan(np.hypot(gx_grad, gy_grad)))
    # curvature (Laplacian), lightly smoothed first
    elev_s = ndimage.gaussian_filter(elev, sigma=1.0)
    curvature = ndimage.laplace(elev_s) / (GRID_RES ** 2)
    # roughness: local std over 10 cm window
    k_rough = int(0.10 / GRID_RES)
    rough = np.sqrt(np.maximum(
        ndimage.uniform_filter(elev**2, size=k_rough)
        - ndimage.uniform_filter(elev, size=k_rough)**2, 0))

    feat_maps = np.stack([rel_elev, slope, curvature, rough]).astype(np.float32)
    print(f"  Features: rel_elev [{rel_elev.min():.3f},{rel_elev.max():.3f}] m, "
          f"slope [{slope.min():.1f},{slope.max():.1f}] deg, "
          f"rough [{rough.min():.4f},{rough.max():.4f}] m")

    # Global channel stats for normalization in training
    stats = {
        "mean": feat_maps[:, valid_grid].mean(axis=1).tolist(),
        "std": (feat_maps[:, valid_grid].std(axis=1) + 1e-6).tolist(),
        "grid": {"x": [float(x_lo), float(x_hi)], "y": [float(y_lo), float(y_hi)],
                 "res": GRID_RES},
    }
    with open(OUT_DIR / "channel_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---------- 4. Per-image rendering ----------
    # v4 image list with base frame names
    v4_images = {}
    for split in ["train", "valid", "test"]:
        for p in (V4_DIR / split).glob("*.jpg"):
            if p.name.endswith("_mask.png"):
                continue
            base = re.sub(r"\.rf\.[a-f0-9]+", "", p.name).replace("_JPG.jpg", ".JPG")
            m = re.match(r"a58_\d+_(?:early|mid|late)_(\d+)_jpg", base.replace(".JPG", ""))
            if m:  # renamed copies like a58_12_late_0700_jpg.jpg
                cand = [n for n in images if n.endswith(f"_{int(m.group(1)):04d}.JPG")]
                base = cand[0] if cand else None
            if base in images:
                v4_images[p.name] = base
    print(f"  v4 images matched to registered frames: {len(v4_images)}")

    cam = cams[1]
    fx, fy, cx, cy = cam["params"][:4]
    dist = cam["params"][4:]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    sx = cam["w"] / IMG_W_SMALL  # COLMAP ran on a58_small; intrinsics should match
    assert abs(cam["w"] - IMG_W_SMALL) < 2 and abs(cam["h"] - IMG_H_SMALL) < 2, \
        f"Camera size {cam['w']}x{cam['h']} != {IMG_W_SMALL}x{IMG_H_SMALL}"

    # Pixel rays (undistorted, normalized) for the small frame
    uu, vv = np.meshgrid(np.arange(IMG_W_SMALL), np.arange(IMG_H_SMALL))
    pix = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64).reshape(1, -1, 2)
    rays = cv2.undistortPoints(pix, K, dist).reshape(IMG_H_SMALL, IMG_W_SMALL, 2)
    rays = np.dstack([rays, np.ones((IMG_H_SMALL, IMG_W_SMALL))])

    n_done = 0
    for v4_name, frame_name in sorted(v4_images.items()):
        out_path = OUT_DIR / (Path(v4_name).stem + ".npy")
        if out_path.exists():
            n_done += 1
            continue
        R_w2c = qvec2rotmat(images[frame_name]["qvec"])
        t = images[frame_name]["tvec"]
        # camera pose in aligned world frame: C_w = R^T(-t); R aligned
        R_c2w = R_w2c.T
        C_w = -R_c2w @ t
        C_a = R_align @ (C_w - centroid)
        R_a = R_align @ R_c2w  # camera->aligned-world rotation

        # Rays in aligned world frame
        rays_w = rays @ R_a.T  # (H, W, 3)

        # Fixed-point iteration: intersect rays with DEM surface
        s = -C_a[2] / np.clip(rays_w[..., 2], -1e9, -1e-6)  # plane Z=0 start
        for _ in range(3):
            P = C_a + rays_w * s[..., None]
            ix = np.clip((P[..., 0] - x_lo) / GRID_RES, 0, nx - 1)
            iy = np.clip((P[..., 1] - y_lo) / GRID_RES, 0, ny - 1)
            z_here = cv2.remap(elev.astype(np.float32), ix.astype(np.float32),
                               iy.astype(np.float32), cv2.INTER_LINEAR)
            s = (z_here - C_a[2]) / np.clip(rays_w[..., 2], -1e9, -1e-6)
        P = C_a + rays_w * s[..., None]

        # Sample feature maps at ground coords
        mx = ((P[..., 0] - x_lo) / GRID_RES).astype(np.float32)
        my = ((P[..., 1] - y_lo) / GRID_RES).astype(np.float32)
        in_footprint = ((mx >= 0) & (mx < nx - 1) & (my >= 0) & (my < ny - 1)
                        & (rays_w[..., 2] < -1e-6))
        channels = []
        for ch in range(4):
            samp = cv2.remap(feat_maps[ch], mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            channels.append(samp)
        # validity: inside footprint AND grid cell had real data
        valid_samp = cv2.remap(valid_grid.astype(np.float32), mx, my,
                               cv2.INTER_NEAREST) > 0.5
        invalid = ~(in_footprint & valid_samp)
        for ch in range(4):  # neutral fill: channel mean -> 0 after z-norm
            channels[ch][invalid] = stats["mean"][ch]
        channels.append((~invalid).astype(np.float32))

        stack = np.stack(channels)  # (5, H_small, W_small)
        # Stretch to v4 geometry: 1024x682 -> 1024x1024
        stack = np.stack([cv2.resize(c, (IMG_OUT, IMG_OUT), interpolation=cv2.INTER_LINEAR)
                          for c in stack])
        np.save(out_path, stack.astype(np.float16))
        n_done += 1
        if n_done % 50 == 0:
            print(f"  rendered {n_done}/{len(v4_images)}")

    print(f"Done. {n_done} per-image stacks in {OUT_DIR}")


if __name__ == "__main__":
    main()
