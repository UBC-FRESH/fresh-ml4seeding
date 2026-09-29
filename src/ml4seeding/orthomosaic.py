"""Orthomosaic analysis: metadata extraction, quality assessment, quicklook generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

RGB_BANDS: tuple[int, int, int] = (1, 2, 3)
"""Default 1-based band indices for (R, G, B) in the GeoTIFF."""

_CAP_PIXELS: int = 2_000_000
"""Maximum number of pixels accumulated across all samples for band/index stats."""


# ---------------------------------------------------------------------------
# Pure-array helpers (public API)
# ---------------------------------------------------------------------------


def laplacian_variance(gray: np.ndarray) -> float:
    """Compute the variance of the discrete Laplacian of a grayscale image.

    Higher values indicate sharper imagery. For 8-bit RGB drone orthos,
    typical interpretation ranges are: below 500 very blurry, 500-1500 soft,
    1500-4000 normal, 4000-8000 sharp, above 8000 very sharp.

    Parameters
    ----------
    gray:
        2-D grayscale array (any numeric dtype; converted to float32 internally).

    Returns
    -------
    float
        Variance of the 4-neighbour Laplacian.
    """
    gray = gray.astype(np.float32)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(np.var(lap))


def edge_density(gray: np.ndarray) -> float:
    """Compute the fraction of pixels whose gradient magnitude is in the top 5 %.

    Parameters
    ----------
    gray:
        2-D grayscale array (any numeric dtype; converted to float32 internally).

    Returns
    -------
    float
        Edge density in [0, 1].
    """
    gray = gray.astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    g = gx + gy
    thr = np.percentile(g, 95)
    return float((g >= thr).mean())


def rgb_indices(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ExG, VARI, and GRVI vegetation indices from an RGB array.

    Parameters
    ----------
    rgb:
        Array of shape ``(H, W, 3)`` with channels in R, G, B order.

    Returns
    -------
    exg, vari, grvi : tuple of np.ndarray
        Each has shape ``(H, W)`` and dtype float32.
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    eps = 1e-6
    exg = 2 * g - r - b
    vari = (g - r) / (g + r - b + eps)
    grvi = (g - r) / (g + r + eps)
    return exg, vari, grvi


def summ_stats(x: np.ndarray) -> dict | None:
    """Compute summary statistics for a 1-D array, ignoring non-finite values.

    Parameters
    ----------
    x:
        Input array (any shape; flattened internally).

    Returns
    -------
    dict or None
        Dictionary with keys ``min``, ``p2``, ``p50``, ``p98``, ``max``,
        ``mean``, ``std`` — or *None* if no finite values are present.
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


# ---------------------------------------------------------------------------
# Quicklook generation
# ---------------------------------------------------------------------------


def make_quicklook(
    src: rasterio.DatasetReader,
    out_png: Path,
    max_side: int = 2000,
) -> np.ndarray:
    """Create a quicklook RGB PNG by reading at reduced resolution.

    The image is normalised to uint8 using robust per-band percentiles (2–98 %).

    Parameters
    ----------
    src:
        An open rasterio dataset with at least three bands.
    out_png:
        Destination PNG file path.
    max_side:
        Maximum pixel dimension of the quicklook (default 2000).

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(H, W, 3)``.
    """
    w, h = src.width, src.height
    scale = max(w, h) / max_side
    if scale < 1:
        scale = 1
    out_w = int(w / scale)
    out_h = int(h / scale)

    rgb = src.read(
        list(RGB_BANDS),
        out_shape=(3, out_h, out_w),
        resampling=Resampling.bilinear,
    )
    rgb = np.transpose(rgb, (1, 2, 0))

    # Normalize to uint8 for viewing (robust percentiles)
    rgb_f = rgb.astype(np.float32)
    p2 = np.percentile(rgb_f, 2, axis=(0, 1))
    p98 = np.percentile(rgb_f, 98, axis=(0, 1))
    rgb_n = (rgb_f - p2) / (p98 - p2 + 1e-6)
    rgb_u8 = np.clip(rgb_n * 255, 0, 255).astype(np.uint8)

    # Save PNG — prefer Pillow, fall back to rasterio
    try:
        from PIL import Image

        Image.fromarray(rgb_u8).save(out_png)
    except ImportError:
        profile = {
            "driver": "PNG",
            "height": rgb_u8.shape[0],
            "width": rgb_u8.shape[1],
            "count": 3,
            "dtype": "uint8",
        }
        with rasterio.open(out_png, "w", **profile) as dst:
            for i in range(3):
                dst.write(rgb_u8[..., i], i + 1)

    return rgb_u8


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze_rgb_ortho(
    ortho_path: Path,
    out_dir: Path,
    n_samples: int = 30,
    win_size: int = 1024,
    black_thresh: int = 5,
    seed: int = 7,
) -> dict:
    """Analyse an RGB orthomosaic and write a JSON report plus quicklook PNG.

    Random windows are sampled from the raster so that statistics can be
    computed quickly even on very large orthomosaics.

    Parameters
    ----------
    ortho_path:
        Path to the input GeoTIFF orthomosaic.
    out_dir:
        Output directory; created if it does not exist.  ``report.json`` and
        ``quicklook.png`` are written here.
    n_samples:
        Number of random windows to sample (default 30).
    win_size:
        Side length of each square sampling window in pixels (default 1024).
    black_thresh:
        Pixel value at or below which all three bands are considered "black"
        for uint8 data (default 5).  Automatically adjusted for uint16.
    seed:
        Random seed for reproducibility (default 7).

    Returns
    -------
    dict
        The full report dictionary (also serialised to ``report.json``).
    """
    random.seed(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "ortho_path": str(ortho_path),
        "rgb_bands": list(RGB_BANDS),
        "samples": {"n_samples": n_samples, "window_size_px": win_size},
        "metadata": {},
        "band_stats_sampled": {},
        "nodata_black_estimates": {},
        "rgb_indices_sampled": {},
        "sharpness_sampled": {},
        "notes": [
            "Stats computed from random windows to stay fast on huge orthomosaics.",
        ],
    }

    with rasterio.open(ortho_path) as src:
        bounds = src.bounds
        resx, resy = src.res
        crs = src.crs.to_string() if src.crs else None

        # Coverage dimensions — prefer bounds-based when CRS is available
        width_m_bounds = (bounds.right - bounds.left) if crs else None
        height_m_bounds = (bounds.top - bounds.bottom) if crs else None

        width_m_px = src.width * float(resx)
        height_m_px = src.height * float(resy)

        width_m = float(width_m_bounds) if width_m_bounds is not None else float(width_m_px)
        height_m = float(height_m_bounds) if height_m_bounds is not None else float(height_m_px)

        area_m2 = width_m * height_m
        area_ha = area_m2 / 10_000
        area_km2 = area_m2 / 1_000_000

        report["metadata"] = {
            "driver": src.driver,
            "dtype_per_band": list(src.dtypes),
            "band_count": src.count,
            "width_px": src.width,
            "height_px": src.height,
            "crs": crs,
            "pixel_size_x": float(resx),
            "pixel_size_y": float(resy),
            "bounds": {
                "left": float(bounds.left),
                "bottom": float(bounds.bottom),
                "right": float(bounds.right),
                "top": float(bounds.top),
            },
            "coverage_width_m": width_m,
            "coverage_height_m": height_m,
            "coverage_area_m2": area_m2,
            "coverage_area_ha": area_ha,
            "coverage_area_km2": area_km2,
            "nodata": src.nodata,
        }

        dtype0 = src.dtypes[RGB_BANDS[0] - 1] if src.dtypes else "uint8"
        black_thr = int(0.01 * 65535) if "uint16" in str(dtype0) else black_thresh

        h, w = src.height, src.width
        win = win_size

        r_all: list[np.ndarray] = []
        g_all: list[np.ndarray] = []
        b_all: list[np.ndarray] = []
        exg_all: list[np.ndarray] = []
        vari_all: list[np.ndarray] = []
        grvi_all: list[np.ndarray] = []
        black_fracs: list[float] = []
        nodata_fracs: list[float | None] = []
        sharp_vals: list[float] = []
        edge_vals: list[float] = []

        def _add_cap(lst: list[np.ndarray], arr: np.ndarray) -> None:
            if arr.size == 0:
                return
            if sum(x.size for x in lst) >= _CAP_PIXELS:
                return
            lst.append(arr.reshape(-1))

        for _ in range(n_samples):
            if w <= win or h <= win:
                x0, y0 = 0, 0
                w0, h0 = w, h
            else:
                x0 = random.randint(0, w - win)
                y0 = random.randint(0, h - win)
                w0, h0 = win, win

            window = Window(x0, y0, w0, h0)
            rgb = src.read(list(RGB_BANDS), window=window)
            rgb = np.transpose(rgb, (1, 2, 0))

            if src.nodata is not None:
                nod_mask = np.any(rgb == src.nodata, axis=2)
                nodata_fracs.append(float(nod_mask.mean()))
            else:
                nodata_fracs.append(None)

            black_mask = np.all(rgb <= black_thr, axis=2)
            black_fracs.append(float(black_mask.mean()))

            valid = ~black_mask
            if src.nodata is not None:
                valid = valid & (~np.any(rgb == src.nodata, axis=2))

            if valid.sum() < 100:
                continue

            rgb_v = rgb[valid]
            _add_cap(r_all, rgb_v[:, 0])
            _add_cap(g_all, rgb_v[:, 1])
            _add_cap(b_all, rgb_v[:, 2])

            exg, vari, grvi = rgb_indices(rgb)
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

        def _cat(lst: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(lst) if lst else np.array([], dtype=np.float32)

        r_cat = _cat(r_all)
        g_cat = _cat(g_all)
        b_cat = _cat(b_all)
        exg_cat = _cat(exg_all)
        vari_cat = _cat(vari_all)
        grvi_cat = _cat(grvi_all)

        report["band_stats_sampled"] = {
            "R": summ_stats(r_cat),
            "G": summ_stats(g_cat),
            "B": summ_stats(b_cat),
        }
        report["rgb_indices_sampled"] = {
            "ExG": summ_stats(exg_cat),
            "VARI": summ_stats(vari_cat),
            "GRVI": summ_stats(grvi_cat),
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

    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
