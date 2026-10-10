#!/usr/bin/env python3
"""
Co-register the nope map to the A58Segment2 orthomosaic.

The GPS-EXIF alignment is good to ~2-5 m absolute (consumer GPS), while the
ODM ortho has its own georeferencing. Fix: render the COLMAP point cloud
(with RGB) as a coarse nadir image in the current UTM frame, SIFT-match it
against the orthomosaic, estimate a 2D affine correction, compose it into
the geotransform, and rewrite the GeoTIFF + QC overlay.
"""
import struct
import json
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import GPSTAGS
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from affine import Affine
import cv2

SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
ORIG = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/raw_drone_images/a58_segment2")
ORTHO = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/orthomosaic/A58Segment2.tif")
NOPE = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map")
OUT_CRS = "EPSG:32610"
REG_RES = 0.35  # m/px for the registration render


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


def read_points3d_rgb(path):
    pts, cols = [], []
    with open(path, "rb") as f:
        n = struct.unpack("Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            pts.append(struct.unpack("ddd", f.read(24)))
            cols.append(struct.unpack("BBB", f.read(3)))
            f.read(8)
            tl = struct.unpack("Q", f.read(8))[0]
            f.read(tl * 8)
    return np.array(pts), np.array(cols)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def dms_to_deg(v):
    return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs**2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    images = read_images(SPARSE / "images.bin")
    points, colors = read_points3d_rgb(SPARSE / "points3D.bin")
    print(f"{len(images)} images, {len(points)} points")

    # --- GPS similarity (same as georeference_nope_map.py) ---
    names, gps = [], []
    for name in sorted(images.keys()):
        g = (Image.open(ORIG / name)._getexif() or {}).get(34853)
        if not g:
            continue
        gg = {GPSTAGS.get(k, k): v for k, v in g.items()}
        lat = dms_to_deg(gg["GPSLatitude"]) * (1 if gg.get("GPSLatitudeRef", "N") == "N" else -1)
        lon = dms_to_deg(gg["GPSLongitude"]) * (1 if gg.get("GPSLongitudeRef", "E") == "E" else -1)
        names.append(name)
        gps.append([lon, lat, float(gg.get("GPSAltitude", 0))])
    gps = np.array(gps)
    e, n = warp_transform("EPSG:4326", OUT_CRS, gps[:, 0].tolist(), gps[:, 1].tolist())
    enu = np.column_stack([e, n, gps[:, 2]])

    d = np.load(NOPE / "site_nope.npz")
    centroid, R_align = d["centroid"], d["R_align"]
    local = np.array([R_align @ ((-qvec2rotmat(images[nm]["qvec"]).T @ images[nm]["tvec"]) - centroid)
                      for nm in names])
    s, R, t = umeyama(local, enu)
    print(f"GPS similarity: scale {s:.4f}")

    # --- Render point cloud as coarse nadir image in the current UTM frame ---
    pa = (points - centroid) @ R_align.T
    pts_enu = (s * (R @ pa.T)).T + t
    res_l, x_lo, y_lo = float(d["res"]), float(d["x_lo"]), float(d["y_lo"])
    nx_l, ny_l = d["hag"].shape[1], d["hag"].shape[0]
    corners = np.array([[x_lo, y_lo, 0], [x_hi := x_lo + (nx_l - 1) * res_l, y_lo, 0],
                        [x_lo, y_hi := y_lo + (ny_l - 1) * res_l, 0], [x_hi, y_hi, 0]])
    corners_enu = (s * (R @ corners.T)).T + t
    pad = 10.0  # m, room for the correction
    e_lo, e_hi = corners_enu[:, 0].min() - pad, corners_enu[:, 0].max() + pad
    n_lo, n_hi = corners_enu[:, 1].min() - pad, corners_enu[:, 1].max() + pad
    W = int((e_hi - e_lo) / REG_RES)
    H = int((n_hi - n_lo) / REG_RES)
    print(f"Registration render: {W}x{H} @ {REG_RES} m/px")

    # bin points into render grid, brightest (mean) color wins per cell
    ix = ((pts_enu[:, 0] - e_lo) / REG_RES).astype(int)
    iy = ((n_hi - pts_enu[:, 1]) / REG_RES).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    render = np.zeros((H, W, 3), np.float32)
    count = np.zeros((H, W, 1), np.float32)
    np.add.at(render, (iy[ok], ix[ok]), colors[ok].astype(np.float32))
    np.add.at(count, (iy[ok], ix[ok]), 1)
    render = render / np.maximum(count, 1)
    gray = render.mean(axis=2)
    filled = gray.copy()
    empty = count[..., 0] == 0
    if empty.any():  # dilate-fill holes
        known = (~empty).astype(np.uint8)
        fill = cv2.dilate(gray, np.ones((5, 5), np.uint8))
        fill_cnt = cv2.dilate((count[..., 0] > 0).astype(np.float32), np.ones((5, 5), np.uint8))
        filled = np.where(empty, fill / np.maximum(fill_cnt, 1e-9), gray)
        # normalize: dilate max-pools, so scale filled regions toward local mean
        filled = np.clip(filled, 0, 255)
    img_mine = cv2.normalize(filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("/tmp/coreg_render_mine.png", img_mine)

    # --- Ortho over the same bbox at same resolution ---
    with rasterio.open(ORTHO) as src:
        win = from_bounds(e_lo, n_lo, e_hi, n_hi, src.transform).round_offsets().round_lengths()
        img_ortho = src.read([1, 2, 3], window=win, out_shape=(3, H, W)).transpose(1, 2, 0)
    gray_ortho = cv2.cvtColor(img_ortho, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("/tmp/coreg_render_ortho.png", gray_ortho)

    # --- Shape registration on coverage footprints ---
    mask_mine = (count[..., 0] > 0).astype(np.float32)
    mask_mine = cv2.morphologyEx(mask_mine, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask_ortho = (gray_ortho > 8).astype(np.float32)

    def iou(a, b):
        a, b = a > 0.5, b > 0.5
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / max(union, 1)

    print(f"  areas: mine {mask_mine.sum():.0f} px, ortho {mask_ortho.sum():.0f} px, "
          f"ratio {mask_ortho.sum()/mask_mine.sum():.3f}")

    def shift_mask(m, dx, dy):
        """shift by integer px without wraparound"""
        out = np.zeros_like(m)
        xs = slice(max(0, dx), min(m.shape[1], m.shape[1] + dx))
        ys = slice(max(0, dy), min(m.shape[0], m.shape[0] + dy))
        xs_src = slice(max(0, -dx), min(m.shape[1], m.shape[1] - dx))
        ys_src = slice(max(0, -dy), min(m.shape[0], m.shape[0] - dy))
        out[ys, xs] = m[ys_src, xs_src]
        return out

    def grid_search(ma, mb, res_m, theta_range, theta_step, t_range_m, t_step_px, label):
        best = (-1, 0, 0, 0)
        ctr = ((ma.shape[1] - 1) / 2, (ma.shape[0] - 1) / 2)
        thetas = np.arange(*theta_range, theta_step)
        tr = int(t_range_m / res_m)
        for th in thetas:
            Rm = cv2.getRotationMatrix2D(ctr, float(th), 1.0)
            ma_r = cv2.warpAffine(ma, Rm, (ma.shape[1], ma.shape[0]))
            for dy in range(-tr, tr + 1, t_step_px):
                m_row = shift_mask(ma_r, 0, dy)
                for dx in range(-tr, tr + 1, t_step_px):
                    sc = iou(shift_mask(m_row, dx, 0), mb)
                    if sc > best[0]:
                        best = (sc, float(th), dx, dy)
        print(f"  {label}: IoU {best[0]:.4f} @ rot {best[1]:+.2f} deg, "
              f"shift ({best[2]*res_m:+.2f}, {best[3]*res_m:+.2f}) m")
        return best

    print("Shape registration (exhaustive theta x translation grid search)...")
    # coarse: 4x downsampled (1.4 m/px)
    Ws, Hs = W // 4, H // 4
    ma_s = cv2.resize(mask_mine, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    mb_s = cv2.resize(mask_ortho, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    sc0, th0, dx0, dy0 = grid_search(ma_s, mb_s, REG_RES * 4, (-25, 25.1), 0.5,
                                     30, 3, "coarse")
    # fine: apply coarse translation first, then search rotation ±1 deg, t ±1.4 m
    ma_pre = shift_mask(mask_mine, int(round(dx0 * 4)), int(round(dy0 * 4)))
    sc1, th1, dx1, dy1 = grid_search(ma_pre, mask_ortho, REG_RES,
                                     (th0 - 1.0, th0 + 1.01), 0.1, 1.4, 1, "fine")
    score, rot_deg = sc1, th1
    dx_px = int(round(dx0 * 4)) + dx1
    dy_px = int(round(dy0 * 4)) + dy1
    dx_m, dy_m = dx_px * REG_RES, dy_px * REG_RES

    # Build M (full res): rotate about center then translate
    ctr = ((W - 1) / 2, (H - 1) / 2)
    Rm = cv2.getRotationMatrix2D(ctr, rot_deg, 1.0)
    Rm[0, 2] += dx_px
    Rm[1, 2] += dy_px
    M = Rm.astype(np.float64)
    tx, ty, rot, scale = M[0, 2], M[1, 2], rot_deg, 1.0
    print(f"  FINAL: translate ({dx_m:+.2f}, {dy_m:+.2f}) m, rotate {rot_deg:+.2f} deg, "
          f"scale fixed 1.0, IoU {score:.4f} (baseline {iou(mask_mine, mask_ortho):.4f})")

    # --- Compose correction into geotransform ---
    # my render pixel p maps to UTM: u = A_render @ p ; ortho pixel q = M @ p
    # maps to u' = A_render @ q  (same bbox). So u' = A_render @ M @ inv(A_render) @ u
    A_render = np.array([[REG_RES, 0, e_lo], [0, -REG_RES, n_hi], [0, 0, 1]])
    M3 = np.vstack([M, [0, 0, 1]])
    T_utm = A_render @ M3 @ np.linalg.inv(A_render)  # 2D affine in UTM meters

    # --- Rewrite GeoTIFF with corrected sampling ---
    # output grid: same UTM bbox/resolution as before; sample source at inv(T_utm) @ u_out
    res = 0.02
    e0, e1 = corners_enu[:, 0].min(), corners_enu[:, 0].max()
    n0, n1 = corners_enu[:, 1].min(), corners_enu[:, 1].max()
    nx_u = int((e1 - e0) / res) + 1
    ny_u = int((n1 - n0) / res) + 1
    eu, nu = np.meshgrid(np.linspace(e0, e1, nx_u), np.linspace(n0, n1, ny_u))
    pts_out = np.stack([eu.ravel(), nu.ravel(), np.ones(eu.size)])
    pts_corr = (np.linalg.inv(T_utm) @ pts_out)[:2].T  # where to sample in old frame
    # corrected frame -> local grid (inverse GPS similarity)
    pts3 = np.column_stack([pts_corr, np.zeros(len(pts_corr))])
    pts_local = ((R.T @ (pts3 - t).T).T) / s
    ix = np.round((pts_local[:, 0] - x_lo) / res_l).astype(int)
    iy = np.round((pts_local[:, 1] - y_lo) / res_l).astype(int)
    okm = (ix >= 0) & (ix < nx_l) & (iy >= 0) & (iy < ny_l)

    def sample(arr):
        out = np.zeros(eu.size, arr.dtype)
        out[okm] = arr[iy[okm], ix[okm]]
        return out.reshape(ny_u, nx_u)

    nope_u = sample(d["nope"])
    valid_u = sample(d["valid"])
    hag_u = sample(d["hag_meters"].astype(np.float32))
    hag_u[~okm.reshape(ny_u, nx_u)] = np.nan

    profile = {"driver": "GTiff", "width": nx_u, "height": ny_u, "count": 3,
               "dtype": "float32", "crs": OUT_CRS, "compress": "deflate",
               "transform": Affine(res, 0, e0, 0, -res, n1)}
    with rasterio.open(NOPE / "site_nope_utm10.tif", "w", **profile) as dst:
        dst.write(nope_u.astype(np.float32), 1)
        dst.write(valid_u.astype(np.float32), 2)
        dst.write(hag_u, 3)
        dst.set_band_description(1, "nope mask (bump|pit)")
        dst.set_band_description(2, "valid SfM coverage")
        dst.set_band_description(3, "height above ground (m)")
    print(f"Wrote corrected {NOPE/'site_nope_utm10.tif'}")

    # --- QC overlay (corrected) ---
    with rasterio.open(ORTHO) as src:
        win = from_bounds(e0, n0, e1, n1, src.transform).round_offsets().round_lengths()
        ov = 1600 / win.width
        oh = max(1, int(win.height * ov))
        img = src.read([1, 2, 3], window=win, out_shape=(3, oh, 1600)).transpose(1, 2, 0)
    nope_s = cv2.resize(nope_u.astype(np.uint8), (1600, oh), interpolation=cv2.INTER_NEAREST)
    valid_s = cv2.resize(valid_u.astype(np.uint8), (1600, oh), interpolation=cv2.INTER_NEAREST)
    ovl = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ovl[nope_s > 0] = (0.4 * ovl[nope_s > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    ovl[valid_s == 0] = (ovl[valid_s == 0] * 0.4).astype(np.uint8)
    cv2.imwrite(str(NOPE / "nope_on_orthomosaic_qc.png"), ovl)
    print("Wrote corrected QC overlay")

    geo = json.load(open(NOPE / "georeferencing.json"))
    geo["ortho_coregistration"] = {
        "method": "footprint shape registration: rotation sweep + phase correlation",
        "translate_m": [float(tx * REG_RES), float(ty * REG_RES)],
        "rotate_deg": float(rot), "scale": 1.0,
        "footprint_iou": float(score),
        "T_utm": T_utm.tolist(),
    }
    with open(NOPE / "georeferencing.json", "w") as f:
        json.dump(geo, f, indent=2)
    print("georeferencing.json updated")


if __name__ == "__main__":
    main()
