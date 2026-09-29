"""Image tiling with overlap, invalid-pixel filtering, and georeferenced metadata."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform


def read_tile_rgb(src, window: Window) -> np.ndarray:
    """Read a window from a GeoTIFF and return HWC uint8 RGB."""
    data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    data = np.transpose(data, (1, 2, 0))
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return data


def invalid_fraction(tile_hwc: np.ndarray, black_thresh: int = 5) -> float:
    """Fraction of pixels that are near-black (treated as invalid)."""
    invalid = (
        (tile_hwc[..., 0] <= black_thresh)
        & (tile_hwc[..., 1] <= black_thresh)
        & (tile_hwc[..., 2] <= black_thresh)
    )
    return float(invalid.mean())


def compute_valid_bbox(
    img_hwc: np.ndarray, black_thresh: int = 5
) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) bbox covering valid (non-near-black) pixels."""
    valid = (
        (img_hwc[..., 0] > black_thresh)
        | (img_hwc[..., 1] > black_thresh)
        | (img_hwc[..., 2] > black_thresh)
    )
    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, img_hwc.shape[1], img_hwc.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def tile_geotiff(
    image_path: Path,
    out_dir: Path,
    tile_size: int = 512,
    overlap: int = 128,
    skip_invalid_fraction: float = 0.70,
    black_thresh: int = 5,
) -> list[dict]:
    """Tile a GeoTIFF into fixed-size PNG tiles with JSON metadata.

    Parameters
    ----------
    image_path : Path
        Path to the source GeoTIFF.
    out_dir : Path
        Output directory (will contain images/, meta/, tiles_index.json).
    tile_size : int
        Tile size in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    skip_invalid_fraction : float
        Skip tiles with this fraction or more of invalid (near-black) pixels.
    black_thresh : int
        Threshold for near-black pixel detection (uint8 scale).

    Returns
    -------
    list of dict
        Tile index with metadata for each saved tile.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than tile_size.")

    # Reset output directory
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tiles_index = []
    saved = skipped = total = 0

    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        crs = src.crs
        tfm = src.transform

        # Estimate valid bbox from a coarse scan
        sample_step = max(tile_size, step * 4)
        valid_lefts, valid_tops = [], []

        for top in range(0, h, sample_step):
            for left in range(0, w, sample_step):
                win = Window(left, top, min(tile_size, w - left), min(tile_size, h - top))
                tile = read_tile_rgb(src, win)
                if invalid_fraction(tile, black_thresh) < 0.98:
                    valid_lefts.append(left)
                    valid_tops.append(top)

        if valid_lefts and valid_tops:
            x0 = max(0, min(valid_lefts) - tile_size)
            y0 = max(0, min(valid_tops) - tile_size)
            x1 = min(w, max(valid_lefts) + tile_size * 2)
            y1 = min(h, max(valid_tops) + tile_size * 2)
        else:
            x0, y0, x1, y1 = 0, 0, w, h

        # Main tiling loop
        for top in range(y0, y1, step):
            for left in range(x0, x1, step):
                total += 1
                if (left + tile_size > w) or (top + tile_size > h):
                    skipped += 1
                    continue

                win = Window(left, top, tile_size, tile_size)
                tile = read_tile_rgb(src, win)
                inv = invalid_fraction(tile, black_thresh)
                if inv >= skip_invalid_fraction:
                    skipped += 1
                    continue

                tile_id = f"x{left}_y{top}_s{tile_size}"
                out_path = images_dir / f"{tile_id}.png"
                Image.fromarray(tile).save(out_path)

                tile_tfm = window_transform(win, tfm)
                b = window_bounds(win, tfm)

                meta = {
                    "tile_id": tile_id,
                    "image_file": out_path.name,
                    "source_image": image_path.name,
                    "left_px": int(left),
                    "top_px": int(top),
                    "tile_size_px": int(tile_size),
                    "overlap_px": int(overlap),
                    "step_px": int(step),
                    "invalid_fraction": float(inv),
                    "crs": str(crs),
                    "transform": [
                        tile_tfm.a, tile_tfm.b, tile_tfm.c,
                        tile_tfm.d, tile_tfm.e, tile_tfm.f,
                    ],
                    "bounds": {
                        "left": b[0], "bottom": b[1], "right": b[2], "top": b[3],
                    },
                }

                (meta_dir / f"{tile_id}.json").write_text(json.dumps(meta, indent=2))
                tiles_index.append(meta)
                saved += 1

    index_path = out_dir / "tiles_index.json"
    index_path.write_text(json.dumps(tiles_index, indent=2))

    return tiles_index
