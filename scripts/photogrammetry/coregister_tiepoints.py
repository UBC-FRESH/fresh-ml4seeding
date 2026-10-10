#!/usr/bin/env python3
"""
Feature-level co-registration: template-match ~25 individual flight frames
(same imagery the ortho was built from) against the orthomosaic to get
tie points, fit a 2D similarity correction, rewrite the GeoTIFF + QC overlay.

Direction: builds on top of the GPS-similarity frame; the fit output replaces
the footprint-shape correction in georeferencing.json.
"""
import struct
import json
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import GPSTAGS
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds, Window
from affine import Affine
import cv2

SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
SMALL = Path("/home/gep/projects/ignitebc-flashforest/annotation/a58_small")
ORIG = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/raw_drone_images/a58_segment2")
ORTHO = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/orthomosaic/A58Segment2.tif")
NOPE = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map")
OUT_CRS = "EPSG:32610"
MATCH_RES = 0.02   # m/px for the ortho patches
MARGIN_M = 20.0


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
    return s, R, mu_d - s * R @ mu_s


def main():
    cam = read_cameras(SPARSE / "cameras.bin")
    images = read_images(SPARSE / "images.bin")
    d = np.load(NOPE / "site_nope.npz")
    centroid, R_align = d["centroid"], d["R_align"]

    # GPS similarity (baseline frame)
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
    local = np.array([R_align @ ((-qvec2rotmat(images[nm]["qvec"]).T @ images[nm]["tvec"]) - centroid)
                      for nm in names])
    s, R, t = umeyama(local, enu)

    # camera rays (normalized, distortion removed)
    fx, fy, cx, cy = cam["params"][:4]
    dist = cam["params"][4:]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    IMG_W, IMG_H = 1024, 682
    corners_px = np.array([[0, 0], [IMG_W - 1, 0], [0, IMG_H - 1], [IMG_W - 1, 681],
                           [IMG_W / 2, IMG_H / 2]], dtype=np.float64).reshape(1, -1, 2)
    corner_rays = cv2.undistortPoints(corners_px, K, dist).reshape(-1, 2)
    corner_rays = np.hstack([corner_rays, np.ones((5, 1))])

    def frame_corners_enu(name):
        im = images[name]
        R_w2c = qvec2rotmat(im["qvec"])
        tvec = im["tvec"]
        R_c2w = R_w2c.T
        C_w = -R_c2w @ tvec
        C_a = R_align @ (C_w - centroid)
        R_a = R_align @ R_c2w
        rays_w = corner_rays @ R_a.T
        s_hit = -C_a[2] / np.clip(rays_w[:, 2], -1e9, -1e-6)
        P = C_a + rays_w * s_hit[:, None]
        return (s * (R @ P.T)).T + t  # (5,3) in UTM

    # --- tie points ---
    src_pts, dst_pts = [], []
    with rasterio.open(ORTHO) as ortho:
        otfm = ortho.transform
        sample_names = sorted(images.keys())[::30]  # every 30th frame
        print(f"Template-matching {len(sample_names)} frames against ortho...")
        for name in sample_names:
            img = cv2.imread(str(SMALL / name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            pts_enu = frame_corners_enu(name)
            ctr_enu = pts_enu[4]
            e0, e1 = pts_enu[:, 0].min() - MARGIN_M, pts_enu[:, 0].max() + MARGIN_M
            n0, n1 = pts_enu[:, 1].min() - MARGIN_M, pts_enu[:, 1].max() + MARGIN_M
            c0, r0 = ~otfm * (e0, n1)
            c1, r1 = ~otfm * (e1, n0)
            win = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
            if win.width < 100 or win.height < 100:
                continue
            wpx_w = int(win.width * abs(otfm.a) / MATCH_RES)
            wpx_h = int(win.height * abs(otfm.e) / MATCH_RES)
            patch = ortho.read(1, window=win, out_shape=(wpx_h, wpx_w))
            if (patch > 8).mean() < 0.5:
                continue
            # window transform: px -> UTM
            win_e0 = e0
            win_n1 = n1
            # SIFT matching (rotation-invariant — no warp assumptions)
            sift = cv2.SIFT_create(contrastThreshold=0.01)
            kp1, des1 = sift.detectAndCompute(img, None)
            kp2, des2 = sift.detectAndCompute(patch, None)
            if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
                print(f"  {name}: too few keypoints ({len(kp1) if des1 is not None else 0}/"
                      f"{len(kp2) if des2 is not None else 0})")
                continue
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, nn in matches if m.distance < 0.8 * nn.distance]
            if len(good) < 12:
                print(f"  {name}: {len(good)} good matches — rejected")
                continue
            sp = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dp = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            Hmat, hin = cv2.findHomography(sp, dp, cv2.RANSAC, 4.0)
            if Hmat is None or hin is None or hin.sum() < 8:
                print(f"  {name}: homography failed ({0 if hin is None else hin.sum()} inl)")
                continue
            ctr = cv2.transform(np.array([[[511.5, 340.5]]], dtype=np.float32), Hmat)[0, 0]
            match_enu = np.array([win_e0 + ctr[0] * MATCH_RES,
                                  win_n1 - ctr[1] * MATCH_RES])
            off = match_enu - ctr_enu[:2]
            if np.linalg.norm(off) > 25:
                print(f"  {name}: offset {np.linalg.norm(off):.1f} m > 25 m — rejected")
                continue
            print(f"  {name}: {int(hin.sum())} inliers  offset ({off[0]:+.2f}, {off[1]:+.2f}) m")
            src_pts.append(ctr_enu[:2])
            dst_pts.append(match_enu)

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    print(f"\n{len(src_pts)} tie points; fitting 2D similarity...")
    if len(src_pts) < 6:
        print("ABORT: too few tie points — keeping the GPS-only GeoTIFF. "
              "Not writing anything.")
        return
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                             ransacReprojThreshold=1.0)
    if M is None or inliers is None or inliers.sum() < 5:
        print("ABORT: fit failed or <5 inliers — keeping the GPS-only GeoTIFF.")
        return
    inl = inliers.ravel().astype(bool)
    a, b_, tx = M[0]
    c_, d_, ty = M[1]
    rot = np.degrees(np.arctan2(c_, a))
    sc = np.sqrt(a * a + c_ * c_)
    res_pred = cv2.transform(src_pts[inl], M).reshape(-1, 2) - dst_pts[inl].reshape(-1, 2)
    res_m = np.linalg.norm(res_pred, axis=1)
    print(f"  correction: translate ({tx:+.2f}, {ty:+.2f}) m, rotate {rot:+.2f} deg, "
          f"scale {sc:.4f}")
    print(f"  inliers {inl.sum()}/{len(inl)}, residual median {np.median(res_m):.2f} m, "
          f"p90 {np.percentile(res_m, 90):.2f} m")

    # Compose: UTM_true = M @ UTM_pred  (M in UTM meters; M is 2x3)
    T_utm = np.vstack([M, [0, 0, 1]])

    # --- Rewrite GeoTIFF with composed correction ---
    res_l, x_lo = float(d["res"]), float(d["x_lo"])
    ny_l, nx_l = d["hag"].shape
    y_lo = float(d["y_lo"])
    corners = np.array([[x_lo, y_lo, 0], [x_lo + (nx_l - 1) * res_l, y_lo, 0],
                        [x_lo, y_lo + (ny_l - 1) * res_l, 0],
                        [x_lo + (nx_l - 1) * res_l, y_lo + (ny_l - 1) * res_l, 0]])
    corners_enu = (s * (R @ corners.T)).T + t
    corners_enu = (T_utm @ np.vstack([corners_enu[:, :2].T, np.ones(4)])).T
    e0, e1 = corners_enu[:, 0].min(), corners_enu[:, 0].max()
    n0, n1 = corners_enu[:, 1].min(), corners_enu[:, 1].max()
    res = 0.02
    nx_u = int((e1 - e0) / res) + 1
    ny_u = int((n1 - n0) / res) + 1
    # rows north -> south to match the north-up geotransform (row 0 = n1)
    eu, nu = np.meshgrid(np.linspace(e0, e1, nx_u), np.linspace(n1, n0, ny_u))
    pts_out = np.vstack([eu.ravel(), nu.ravel(), np.ones(eu.size)])
    pts_uncorr = (np.linalg.inv(T_utm) @ pts_out)[:2]
    pts3 = np.vstack([pts_uncorr, np.zeros(pts_uncorr.shape[1])]).T
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
    with rasterio.open(NOPE / "site_nope_utm10.tif", "w", **profile) as dstf:
        dstf.write(nope_u.astype(np.float32), 1)
        dstf.write(valid_u.astype(np.float32), 2)
        dstf.write(hag_u, 3)
        dstf.set_band_description(1, "nope mask (bump|pit)")
        dstf.set_band_description(2, "valid SfM coverage")
        dstf.set_band_description(3, "height above ground (m)")
    print(f"Wrote corrected {NOPE/'site_nope_utm10.tif'}")

    # QC overlay
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
        "method": "tie points: per-frame template matching (NCC) vs orthomosaic, "
                  "2D similarity RANSAC fit",
        "n_tiepoints": int(len(src_pts)),
        "inliers": int(inl.sum()),
        "translate_m": [float(tx), float(ty)],
        "rotate_deg": float(rot),
        "scale": float(sc),
        "residual_median_m": float(np.median(res_m)),
        "residual_p90_m": float(np.percentile(res_m, 90)),
        "T_utm": T_utm.tolist(),
    }
    with open(NOPE / "georeferencing.json", "w") as f:
        json.dump(geo, f, indent=2)
    print("georeferencing.json updated")


if __name__ == "__main__":
    main()
