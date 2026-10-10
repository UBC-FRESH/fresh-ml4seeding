#!/usr/bin/env python3
"""
Georeference the nope map: align COLMAP local frame to EXIF GPS (Umeyama
similarity), export GeoTIFF, QC-overlay on the existing A58Segment2 ortho.
"""
import struct
import json
import numpy as np
from pathlib import Path
from PIL import Image
from PIL.ExifTags import GPSTAGS
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.transform import from_origin
from affine import Affine
import cv2

SPARSE = Path("/tmp/colmap_a58_full/sparse/0")
ORIG = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/raw_drone_images/a58_segment2")
ORTHO = Path("/srv/shared-data/eghase02/jupyterhub15-data/ML4seeding/examples/input/orthomosaic/A58Segment2.tif")
NOPE = Path("/home/gep/projects/ignitebc-flashforest/annotation/nope_map")
OUT_CRS = "EPSG:32610"  # UTM 10N


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
    d, m, s = float(v[0]), float(v[1]), float(v[2])
    return d + m / 60 + s / 3600


def umeyama(src, dst):
    """src, dst: (N,3). Returns s, R, t such that dst ≈ s * R @ src + t."""
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
    print("Loading COLMAP poses...")
    images = read_images(SPARSE / "images.bin")

    print("Reading EXIF GPS from originals...")
    names, gps = [], []
    extra_tags = set()
    for name in sorted(images.keys()):
        p = ORIG / name
        exif = Image.open(p)._getexif() or {}
        g = exif.get(34853)
        if not g:
            continue
        gg = {GPSTAGS.get(k, k): v for k, v in g.items()}
        extra_tags |= set(gg.keys())
        lat = dms_to_deg(gg["GPSLatitude"]) * (1 if gg.get("GPSLatitudeRef", "N") == "N" else -1)
        lon = dms_to_deg(gg["GPSLongitude"]) * (1 if gg.get("GPSLongitudeRef", "E") == "E" else -1)
        alt = float(gg.get("GPSAltitude", 0))
        names.append(name)
        gps.append([lon, lat, alt])
    print(f"  {len(gps)} frames with GPS; GPS tags present: {sorted(extra_tags)}")

    gps = np.array(gps)
    lon0, lat0 = np.median(gps[:, 0]), np.median(gps[:, 1])
    e, n = warp_transform("EPSG:4326", OUT_CRS, gps[:, 0].tolist(), gps[:, 1].tolist())
    enu = np.column_stack([np.array(e), np.array(n), gps[:, 2]])

    # COLMAP camera centers in the GRID frame (same frame as site_nope.npz)
    d = np.load(NOPE / "site_nope.npz")
    centroid, R_align = d["centroid"], d["R_align"]
    local = []
    for name in names:
        im = images[name]
        C_w = -qvec2rotmat(im["qvec"]).T @ im["tvec"]
        local.append(R_align @ (C_w - centroid))
    local = np.array(local)

    s, R, t = umeyama(local, enu)
    pred = (s * (R @ local.T)).T + t
    resid = np.linalg.norm(pred - enu, axis=1)
    resid_h = np.linalg.norm(pred[:, :2] - enu[:, :2], axis=1)
    resid_v = np.abs(pred[:, 2] - enu[:, 2])
    print(f"Umeyama: scale={s:.4f} (m per colmap-unit)")
    print(f"Residuals: 3D rms {np.sqrt((resid**2).mean()):.3f} m | "
          f"horiz rms {np.sqrt((resid_h**2).mean()):.3f} m | vert rms {np.sqrt((resid_v**2).mean()):.3f} m | "
          f"max {resid.max():.2f} m")

    # Grid extent in local frame
    res = float(d["res"])
    x_lo, x_hi = float(d["x_lo"]), float(d["x_hi"])
    y_lo, y_hi = float(d["y_lo"]), float(d["y_hi"])
    nx, ny = d["hag"].shape[1], d["hag"].shape[0]

    # Local grid cell corners -> (x,y,z=0) local -> ENU
    # For an axis-aligned UTM output grid, compute transformed bounds
    corners = np.array([[x_lo, y_lo, 0], [x_hi, y_lo, 0], [x_lo, y_hi, 0], [x_hi, y_hi, 0]])
    corners_enu = (s * (R @ corners.T)).T + t
    e_lo, e_hi = corners_enu[:, 0].min(), corners_enu[:, 0].max()
    n_lo, n_hi = corners_enu[:, 1].min(), corners_enu[:, 1].max()
    nx_u = int((e_hi - e_lo) / res) + 1
    ny_u = int((n_hi - n_lo) / res) + 1
    print(f"UTM grid: {nx_u}x{ny_u} @ {res} m  E[{e_lo:.1f},{e_hi:.1f}] N[{n_lo:.1f},{n_hi:.1f}]")

    # Resample masks onto UTM grid (inverse transform + nearest sample)
    eu, nu = np.meshgrid(np.linspace(e_lo, e_hi, nx_u), np.linspace(n_lo, n_hi, ny_u))
    pts_enu = np.column_stack([eu.ravel(), nu.ravel(), np.zeros(eu.size)])
    # inverse similarity: local = (R^T @ (enu - t)) / s
    pts_local = ((R.T @ (pts_enu - t).T).T) / s
    ix = np.round((pts_local[:, 0] - x_lo) / res).astype(int)
    iy = np.round((pts_local[:, 1] - y_lo) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def sample(arr):
        out = np.zeros(eu.size, arr.dtype)
        out[ok] = arr[iy[ok], ix[ok]]
        return out.reshape(ny_u, nx_u)

    nope_u = sample(d["nope"])
    valid_u = sample(d["valid"])
    hag_key = "hag_meters" if "hag_meters" in d else "hag"
    hag_u = sample(d[hag_key].astype(np.float32))  # real meters
    hag_u[~ok.reshape(ny_u, nx_u)] = np.nan

    # Write GeoTIFF (3 bands: nope, valid, hag)
    profile = {
        "driver": "GTiff", "width": nx_u, "height": ny_u, "count": 3,
        "dtype": "float32", "crs": OUT_CRS, "compress": "deflate",
        "transform": Affine(res, 0, e_lo, 0, -res, n_hi),
    }
    with rasterio.open(NOPE / "site_nope_utm10.tif", "w", **profile) as dst:
        dst.write(nope_u.astype(np.float32), 1)
        dst.write(valid_u.astype(np.float32), 2)
        dst.write(hag_u, 3)
        dst.set_band_description(1, "nope mask (bump|pit)")
        dst.set_band_description(2, "valid SfM coverage")
        dst.set_band_description(3, "height above ground (m)")
    print(f"Wrote {NOPE/'site_nope_utm10.tif'}")

    # QC overlay on the existing orthomosaic
    with rasterio.open(ORTHO) as src:
        print(f"Ortho: {src.width}x{src.height}, crs {src.crs}")
        from rasterio.windows import from_bounds
        # Ortho is EPSG:32610 (same as our output) — window directly in UTM
        win = from_bounds(e_lo, n_lo, e_hi, n_hi, src.transform)
        win = win.round_offsets().round_lengths()
        ov = 1600 / win.width
        oh = max(1, int(win.height * ov))
        img = src.read([1, 2, 3], window=win,
                       out_shape=(3, oh, 1600)).transpose(1, 2, 0)
        # both grids are axis-aligned UTM over (nearly) the same bounds: resize
        nope_on_ortho = cv2.resize(nope_u.astype(np.uint8), (1600, oh),
                                   interpolation=cv2.INTER_NEAREST)
        valid_on_ortho = cv2.resize(valid_u.astype(np.uint8), (1600, oh),
                                    interpolation=cv2.INTER_NEAREST)
        ovl = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ovl[nope_on_ortho > 0] = (0.4 * ovl[nope_on_ortho > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        ovl[valid_on_ortho == 0] = (ovl[valid_on_ortho == 0] * 0.4).astype(np.uint8)
        cv2.imwrite(str(NOTE_PNG := NOPE / "nope_on_orthomosaic_qc.png"), ovl)
        print(f"Wrote {NOTE_PNG}")

    meta = {"crs": OUT_CRS, "scale_m_per_unit": float(s),
            "resid_rms_3d_m": float(np.sqrt((resid**2).mean())),
            "resid_rms_horizontal_m": float(np.sqrt((resid_h**2).mean())),
            "resid_rms_vertical_m": float(np.sqrt((resid_v**2).mean())),
            "utm_bounds": {"e": [float(e_lo), float(e_hi)], "n": [float(n_lo), float(n_hi)]},
            "n_gps_frames": len(gps)}
    with open(NOPE / "georeferencing.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
