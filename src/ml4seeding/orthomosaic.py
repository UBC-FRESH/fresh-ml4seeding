"""Orthomosaic analysis: metadata extraction, quality assessment, quicklook generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

RGB_BANDS = (1, 2, 3)


def laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance for sharpness assessment.

    Higher values indicate sharper images. Typical ranges for 8-bit RGB:
    <500 very blurry, 500–1500 soft, 1500–4000 normal, 4000–8000 sharp, >8000 very sharp.
    """
    gray = gray.astype(np.float32)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    )
    return float(np.var(lap))


def edge_density(gray: np.ndarray) -> float:
    """Compute edge density (fraction of pixels above 95th percentile gradient)."""
    gray = gray.astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    g = gx + gy
    thr = np.percentile(g, 95)
    return float((g >= thr).mean())


def rgb_indices(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vegetation indices: ExG, VARI, GRVI.

    Returns (exg, vari, grvi) as float32 arrays.
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-6
    exg = 2 * g - r - b
    vari = (g - r) / (g + r - b + eps)
    grvi = (g - r) / (g + r + eps)
    return exg, vari, grvi


def summ_stats(x: np.ndarray) -> dict | None:
    """Compute summary statistics for a numeric array.

    Returns None if the array is empty after filtering NaN/Inf.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {
        "min": float(np.min(x)),
        "p2": float(np.percentile(x, 2)),
        "p50": float(np.percentile(x, 50)),
        "p98": float(np.percentile(x, 98)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def make_quicklook(src, out_png: Path, max_side: int = 2000) -> np.ndarray:
    """Create a quicklook RGB PNG by reading at reduced resolution.

    Returns the uint8 RGB array.
    """
    w, h = src.width, src.height
    scale = max(w, h) / max_side
    if scale < 1:
        scale = 1
    out_w, out_h = int(w / scale), int(h / scale)

    rgb = src.read(RGB_BANDS, out_shape=(3, out_h, out_w), resampling=Resampling.bilinear)
    rgb = np.transpose(rgb, (1, 2, 0))

    # Normalize to uint8 using robust percentiles
    rgb_f = rgb.astype(np.float32)
    p2 = np.percentile(rgb_f, 2, axis=(0, 1))
    p98 = np.percentile(rgb_f, 98, axis=(0, 1))
    rgb_n = (rgb_f - p2) / (p98 - p2 + 1e-6)
    rgb_u8 = np.clip(rgb_n * 255, 0, 255).astype(np.uint8)

    try:
        from PIL import Image
        Image.fromarray(rgb_u8).save(out_png)
    except ImportError:
        profile = {
            "driver": "PNG", "height": rgb_u8.shape[0],
            "width": rgb_u8.shape[1], "count": 3, "dtype": "uint8",
        }
        with rasterio.open(out_png, "w", **profile) as dst:
            for i in range(3):
                dst.write(rgb_u8[..., i], i + 1)

    return rgb_u8


def analyze_rgb_ortho(
    ortho_path: Path,
    out_dir: Path,
    n_samples: int = 30,
    win_size: int = 1024,
    black_thresh: int = 5,
    seed: int = 7,
) -> dict:
    """Analyze an RGB orthomosaic and write a JSON report + quicklook PNG.

    Samples random windows from the orthomosaic to compute statistics efficiently
    without loading the full raster into memory.

    Parameters
    ----------
    ortho_path : Path
        Path to the orthomosaic GeoTIFF.
    out_dir : Path
        Output directory for report.json and quicklook.png.
    n_samples : int
        Number of random sample windows.
    win_size : int
        Sample window size in pixels.
    black_thresh : int
        Threshold for near-black pixel detection (uint8 scale).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        The analysis report.
    """
    ortho_path = Path(ortho_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    report = {
        "ortho_path": str(ortho_path),
        "rgb_bands": RGB_BANDS,
        "samples": {"n_samples": n_samples, "window_size_px": win_size},
        "metadata": {},
        "band_stats_sampled": {},
        "nodata_black_estimates": {},
        "rgb_indices_sampled": {},
        "sharpness_sampled": {},
    }

    with rasterio.open(ortho_path) as src:
        resx, resy = src.res
        crs = src.crs.to_string() if src.crs else None
        bounds = src.bounds

        width_m = float(bounds.right - bounds.left) if crs else src.width * float(resx)
        height_m = float(bounds.top - bounds.bottom) if crs else src.height * float(resy)
        area_m2 = width_m * height_m

        report["metadata"] = {
            "driver": src.driver,
            "dtype_per_band": src.dtypes,
            "band_count": src.count,
            "width_px": src.width,
            "height_px": src.height,
            "crs": crs,
            "pixel_size_x": float(resx),
            "pixel_size_y": float(resy),
            "bounds": {
                "left": float(bounds.left), "bottom": float(bounds.bottom),
                "right": float(bounds.right), "top": float(bounds.top),
            },
            "coverage_width_m": width_m,
            "coverage_height_m": height_m,
            "coverage_area_m2": area_m2,
            "coverage_area_ha": area_m2 / 10_000,
            "coverage_area_km2": area_m2 / 1_000_000,
            "nodata": src.nodata,
        }

        dtype0 = src.dtypes[RGB_BANDS[0] - 1] if src.dtypes else "uint8"
        black_thr = int(0.01 * 65535) if "uint16" in str(dtype0) else black_thresh

        h, w = src.height, src.width
        cap_pixels = 2_000_000
        r_all, g_all, b_all = [], [], []
        exg_all, vari_all, grvi_all = [], [], []
        black_fracs, nodata_fracs = [], []
        sharp_vals, edge_vals = [], []

        def _add_cap(lst, arr):
            if arr.size == 0 or sum(x.size for x in lst) >= cap_pixels:
                return
            lst.append(arr.reshape(-1))

        for _ in range(n_samples):
            if w <= win_size or h <= win_size:
                x0, y0, w0, h0 = 0, 0, w, h
            else:
                x0 = random.randint(0, w - win_size)
                y0 = random.randint(0, h - win_size)
                w0 = h0 = win_size

            window = Window(x0, y0, w0, h0)
            rgb = src.read(RGB_BANDS, window=window)
            rgb = np.transpose(rgb, (1, 2, 0))

            if src.nodata is not None:
                nod_mask = np.any(rgb == src.nodata, axis=2)
                nodata_fracs.append(float(nod_mask.mean()))

            black_mask = np.all(rgb <= black_thr, axis=2)
            black_fracs.append(float(black_mask.mean()))

            valid = ~black_mask
            if src.nodata is not None:
                valid = valid & ~np.any(rgb == src.nodata, axis=2)
            if valid.sum() < 100:
                continue

            rgb_v = rgb[valid]
            _add_cap(r_all, rgb_v[:, 0])
            _add_cap(g_all, rgb_v[:, 1])
            _add_cap(b_all, rgb_v[:, 2])

            exg, vari, grvi = rgb_indices(rgb.astype(np.float32))
            _add_cap(exg_all, exg[valid])
            _add_cap(vari_all, vari[valid])
            _add_cap(grvi_all, grvi[valid])

            gray = (
                0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            ).astype(np.float32)
            gray_v = gray.copy()
            gray_v[~valid] = np.nan
            med = np.nanmedian(gray_v)
            gray_fill = np.where(np.isfinite(gray_v), gray_v, med)
            sharp_vals.append(laplacian_variance(gray_fill))
            edge_vals.append(edge_density(gray_fill))

        def _cat(lst):
            return np.concatenate(lst) if lst else np.array([], dtype=np.float32)

        report["band_stats_sampled"] = {
            "R": summ_stats(_cat(r_all)),
            "G": summ_stats(_cat(g_all)),
            "B": summ_stats(_cat(b_all)),
        }
        report["rgb_indices_sampled"] = {
            "ExG": summ_stats(_cat(exg_all)),
            "VARI": summ_stats(_cat(vari_all)),
            "GRVI": summ_stats(_cat(grvi_all)),
        }

        bf = np.array([x for x in black_fracs if x is not None], dtype=float)
        nf = np.array([x for x in nodata_fracs if x is not None], dtype=float)
        report["nodata_black_estimates"] = {
            "black_threshold_used": black_thr,
            "black_fraction_mean": float(np.mean(bf)) if bf.size else None,
            "black_fraction_p50": float(np.percentile(bf, 50)) if bf.size else None,
            "black_fraction_p95": float(np.percentile(bf, 95)) if bf.size else None,
            "nodata_fraction_mean": float(np.mean(nf)) if nf.size else None,
            "nodata_fraction_p50": float(np.percentile(nf, 50)) if nf.size else None,
            "nodata_fraction_p95": float(np.percentile(nf, 95)) if nf.size else None,
        }

        sv = np.array(sharp_vals, dtype=float)
        ev = np.array(edge_vals, dtype=float)
        report["sharpness_sampled"] = {
            "laplacian_variance_mean": float(np.mean(sv)) if sv.size else None,
            "laplacian_variance_p50": float(np.percentile(sv, 50)) if sv.size else None,
            "laplacian_variance_p10": float(np.percentile(sv, 10)) if sv.size else None,
            "laplacian_variance_p90": float(np.percentile(sv, 90)) if sv.size else None,
            "edge_density_mean": float(np.mean(ev)) if ev.size else None,
            "edge_density_p50": float(np.percentile(ev, 50)) if ev.size else None,
        }

        quicklook_png = out_dir / "quicklook.png"
        make_quicklook(src, quicklook_png, max_side=2000)
        report["outputs"] = {
            "report_json": str(out_dir / "report.json"),
            "quicklook_png": str(quicklook_png),
        }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
