"""Camouflage grid overlay for annotation quality control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def grid_spacing_px(pixel_size_m: float, grid_cm: float) -> int:
    """Convert ground distance (cm) to pixel spacing.

    Parameters
    ----------
    pixel_size_m : float
        Size of one pixel in meters.
    grid_cm : float
        Desired grid spacing in centimeters.

    Returns
    -------
    int
        Grid spacing in pixels (minimum 2).
    """
    grid_m = grid_cm / 100.0
    px = int(round(grid_m / pixel_size_m))
    return max(px, 2)


def camo_color_vertical(
    img: np.ndarray, x: int, thickness: int, band: int
) -> np.ndarray:
    """Compute camouflage color for a vertical grid line."""
    h, w, _ = img.shape
    x0 = max(0, x - thickness // 2)
    x1 = min(w, x0 + thickness)

    left0, left1 = max(0, x0 - band), x0
    right0, right1 = x1, min(w, x1 + band)

    samples = []
    if left1 > left0:
        samples.append(img[:, left0:left1, :])
    if right1 > right0:
        samples.append(img[:, right0:right1, :])
    if not samples:
        patch0, patch1 = max(0, x - band), min(w, x + band)
        samples = [img[:, patch0:patch1, :]]

    samp = np.concatenate(samples, axis=1)
    mean_rgb = np.mean(samp.reshape(-1, 3), axis=0)
    return np.clip(np.round(mean_rgb), 0, 255).astype(np.uint8)


def camo_color_horizontal(
    img: np.ndarray, y: int, thickness: int, band: int
) -> np.ndarray:
    """Compute camouflage color for a horizontal grid line."""
    h, w, _ = img.shape
    y0 = max(0, y - thickness // 2)
    y1 = min(h, y0 + thickness)

    top0, top1 = max(0, y0 - band), y0
    bot0, bot1 = y1, min(h, y1 + band)

    samples = []
    if top1 > top0:
        samples.append(img[top0:top1, :, :])
    if bot1 > bot0:
        samples.append(img[bot0:bot1, :, :])
    if not samples:
        patch0, patch1 = max(0, y - band), min(h, y + band)
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
    """Draw a camouflage grid overlay on an image.

    Grid lines are rendered in colors that blend with the underlying imagery
    by computing average RGB values from neighbouring pixels.

    Parameters
    ----------
    img_u8 : np.ndarray
        Input image as uint8 (H, W, 3).
    spacing : int
        Grid spacing in pixels.
    thickness : int
        Grid line thickness in pixels.
    band : int
        How far to sample neighbors for camouflage color computation.
    alpha : float
        Blend factor (1.0 = full camo color, <1 blends with original).

    Returns
    -------
    np.ndarray
        Image with grid overlay, same shape as input.
    """
    out = img_u8.copy()
    h, w, _ = out.shape

    # Vertical lines
    for x in range(0, w, spacing):
        color = camo_color_vertical(out, x, thickness, band).astype(np.float32)
        x0 = max(0, x - thickness // 2)
        x1 = min(w, x0 + thickness)
        out[:, x0:x1, :] = np.clip(
            (1 - alpha) * out[:, x0:x1, :].astype(np.float32) + alpha * color,
            0, 255,
        ).astype(np.uint8)

    # Horizontal lines
    for y in range(0, h, spacing):
        color = camo_color_horizontal(out, y, thickness, band).astype(np.float32)
        y0 = max(0, y - thickness // 2)
        y1 = min(h, y0 + thickness)
        out[y0:y1, :, :] = np.clip(
            (1 - alpha) * out[y0:y1, :, :].astype(np.float32) + alpha * color,
            0, 255,
        ).astype(np.uint8)

    return out


def add_grid_to_tiles(
    tiles_dir: Path,
    meta_dir: Path,
    out_dir: Path,
    grid_cm: float = 10.0,
    pixel_size_m: float = 0.0075,
    line_thickness: int = 1,
    neighbor_band: int = 6,
    alpha: float = 0.8,
) -> tuple[int, int]:
    """Add camouflage grid overlay to all PNG tiles in a directory.

    Parameters
    ----------
    tiles_dir : Path
        Directory containing PNG tile images.
    meta_dir : Path
        Directory containing JSON sidecar metadata files.
    out_dir : Path
        Output directory for gridded tiles.
    grid_cm : float
        Desired grid spacing in centimeters (ground distance).
    pixel_size_m : float
        Pixel size in meters (used to convert grid_cm to pixels).
    line_thickness : int
        Grid line thickness in pixels.
    neighbor_band : int
        How far to sample neighbors for camouflage color.
    alpha : float
        Blend factor.

    Returns
    -------
    tuple of (saved, skipped) counts.
    """
    tiles_dir = Path(tiles_dir)
    meta_dir = Path(meta_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(tiles_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG tiles found in: {tiles_dir}")

    spacing = grid_spacing_px(pixel_size_m, grid_cm)

    saved = 0
    skipped = 0

    for png_path in pngs:
        tile_id = png_path.stem
        meta_path = meta_dir / f"{tile_id}.json"
        if not meta_path.exists():
            skipped += 1
            continue

        img = Image.open(png_path).convert("RGB")
        img_u8 = np.array(img, dtype=np.uint8)

        out_u8 = draw_grid_camo(
            img_u8, spacing=spacing, thickness=line_thickness,
            band=neighbor_band, alpha=alpha,
        )

        Image.fromarray(out_u8).save(out_dir / png_path.name)
        saved += 1

    return saved, skipped
