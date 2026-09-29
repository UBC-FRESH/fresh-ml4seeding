"""Camouflage grid overlay for annotation quality control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def grid_spacing_px(pixel_size_m: float, grid_cm: float) -> int:
    """Convert a ground distance in centimeters to a pixel spacing.

    Parameters
    ----------
    pixel_size_m : float
        Ground sample distance in meters per pixel.
    grid_cm : float
        Desired grid spacing in centimeters on the ground.

    Returns
    -------
    int
        Grid spacing in pixels, clamped to a minimum of 2 to avoid
        degenerate spacing.
    """
    grid_m = grid_cm / 100.0
    px = int(round(grid_m / pixel_size_m))
    return max(px, 2)


def camo_color_vertical(img: np.ndarray, x: int, thickness: int, band: int) -> np.ndarray:
    """Compute the camouflage color for a vertical grid line at column ``x``.

    The color is the rounded mean RGB of the pixels in a ``band``-wide strip
    on each side of the line (excluding the line itself). When both strips
    are empty, the neighborhood around the line is used instead.

    Parameters
    ----------
    img : np.ndarray
        HWC RGB image array.
    x : int
        Column of the grid line.
    thickness : int
        Line thickness in pixels.
    band : int
        Width of the sampling strip on each side of the line.

    Returns
    -------
    np.ndarray
        Mean RGB color, shape (3,), dtype uint8.
    """
    H, W, _ = img.shape
    x0 = max(0, x - thickness // 2)
    x1 = min(W, x0 + thickness)

    left0, left1 = max(0, x0 - band), x0
    right0, right1 = x1, min(W, x1 + band)

    samples = []
    if left1 > left0:
        samples.append(img[:, left0:left1, :])
    if right1 > right0:
        samples.append(img[:, right0:right1, :])
    if not samples:
        patch0 = max(0, x - band)
        patch1 = min(W, x + band)
        samples = [img[:, patch0:patch1, :]]

    samp = np.concatenate(samples, axis=1)
    mean_rgb = np.mean(samp.reshape(-1, 3), axis=0)
    return np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8)


def camo_color_horizontal(img: np.ndarray, y: int, thickness: int, band: int) -> np.ndarray:
    """Compute the camouflage color for a horizontal grid line at row ``y``.

    The color is the rounded mean RGB of the pixels in a ``band``-wide strip
    above and below the line (excluding the line itself). When both strips
    are empty, the neighborhood around the line is used instead.

    Parameters
    ----------
    img : np.ndarray
        HWC RGB image array.
    y : int
        Row of the grid line.
    thickness : int
        Line thickness in pixels.
    band : int
        Width of the sampling strip above and below the line.

    Returns
    -------
    np.ndarray
        Mean RGB color, shape (3,), dtype uint8.
    """
    H, W, _ = img.shape
    y0 = max(0, y - thickness // 2)
    y1 = min(H, y0 + thickness)

    top0, top1 = max(0, y0 - band), y0
    bot0, bot1 = y1, min(H, y1 + band)

    samples = []
    if top1 > top0:
        samples.append(img[top0:top1, :, :])
    if bot1 > bot0:
        samples.append(img[bot0:bot1, :, :])
    if not samples:
        patch0 = max(0, y - band)
        patch1 = min(H, y + band)
        samples = [img[patch0:patch1, :, :]]

    samp = np.concatenate(samples, axis=0)
    mean_rgb = np.mean(samp.reshape(-1, 3), axis=0)
    return np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8)


def draw_grid_camo(
    img_u8: np.ndarray,
    spacing: int,
    thickness: int = 1,
    band: int = 6,
    alpha: float = 0.8,
) -> np.ndarray:
    """Draw a camouflage grid over an RGB uint8 image.

    Grid lines start at the origin and repeat every ``spacing`` pixels in
    both directions. Each line is colored with the mean color of its
    neighboring pixels and alpha-blended over the original image, so the
    grid stays visible without obscuring the underlying imagery.

    Parameters
    ----------
    img_u8 : np.ndarray
        HWC RGB image array with dtype uint8. Not modified in place.
    spacing : int
        Grid spacing in pixels.
    thickness : int
        Grid line thickness in pixels.
    band : int
        Width of the neighborhood strip used to compute camouflage colors.
    alpha : float
        Blend factor in [0, 1]: 1.0 uses the camouflage color fully,
        0.0 leaves the image unchanged.

    Returns
    -------
    np.ndarray
        Gridded image, same shape and dtype as ``img_u8``.
    """
    out = img_u8.copy()
    H, W, _ = out.shape

    # Vertical lines
    for x in range(0, W, spacing):
        color = camo_color_vertical(out, x, thickness, band).astype(np.float32)
        x0 = max(0, x - thickness // 2)
        x1 = min(W, x0 + thickness)
        blended = (1 - alpha) * out[:, x0:x1, :].astype(np.float32) + alpha * color
        out[:, x0:x1, :] = np.clip(blended, 0, 255).astype(np.uint8)

    # Horizontal lines
    for y in range(0, H, spacing):
        color = camo_color_horizontal(out, y, thickness, band).astype(np.float32)
        y0 = max(0, y - thickness // 2)
        y1 = min(H, y0 + thickness)
        blended = (1 - alpha) * out[y0:y1, :, :].astype(np.float32) + alpha * color
        out[y0:y1, :, :] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def _pixel_size_from_meta(meta: dict) -> float | None:
    """Extract the pixel size (m/px) from a tile metadata dict, if present.

    Uses the absolute value of the first element of the 6-parameter affine
    transform written by :func:`ml4seeding.tiling.tile_geotiff`.
    """
    tfm = meta.get("transform")
    if tfm and len(tfm) >= 6:
        a = float(tfm[0])
        if a != 0:
            return abs(a)
    return None


def add_grid_to_tiles(
    tiles_dir: Path,
    meta_dir: Path,
    out_dir: Path,
    grid_cm: float = 10.0,
    pixel_size_m: float | None = None,
) -> tuple[int, int]:
    """Add a camouflage grid overlay to every tile PNG in ``tiles_dir``.

    For each ``*.png`` tile, the matching ``<tile_id>.json`` sidecar in
    ``meta_dir`` must exist; tiles without metadata are skipped. The pixel
    size (meters per pixel) is taken from ``pixel_size_m`` when given,
    otherwise from the affine transform stored in each tile's metadata;
    tiles whose metadata carries no transform are skipped.

    Parameters
    ----------
    tiles_dir : pathlib.Path
        Directory containing the tile PNG files.
    meta_dir : pathlib.Path
        Directory containing the per-tile JSON sidecars.
    out_dir : pathlib.Path
        Output directory for the gridded PNG files (created if needed).
    grid_cm : float
        Desired grid spacing in centimeters on the ground.
    pixel_size_m : float or None
        Ground sample distance in meters per pixel. When ``None``, the value
        is read from each tile's metadata transform.

    Returns
    -------
    tuple[int, int]
        ``(saved, skipped)`` tile counts.
    """
    pngs = sorted(tiles_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG tiles found in: {tiles_dir.resolve()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0

    for png_path in pngs:
        meta_path = meta_dir / f"{png_path.stem}.json"
        if not meta_path.exists():
            print(f"Skip (no meta): {png_path.name}")
            skipped += 1
            continue

        px_m = pixel_size_m
        if px_m is None:
            meta = json.loads(meta_path.read_text())
            px_m = _pixel_size_from_meta(meta)
            if px_m is None:
                print(f"Skip (no transform in meta): {png_path.name}")
                skipped += 1
                continue

        spacing = grid_spacing_px(px_m, grid_cm)
        img_u8 = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
        out_u8 = draw_grid_camo(img_u8, spacing=spacing)
        Image.fromarray(out_u8).save(out_dir / png_path.name)
        saved += 1

    print(f"Saved {saved} gridded tiles, skipped {skipped}; output: {out_dir.resolve()}")
    return saved, skipped
