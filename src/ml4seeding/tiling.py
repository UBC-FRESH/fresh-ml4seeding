"""Image tiling with overlap, invalid-pixel filtering, and georeferenced metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform


def read_tile_rgb(src, window: Window) -> np.ndarray:
    """Read a window from an open GeoTIFF dataset and return HWC uint8 RGB.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open rasterio dataset. The first three bands are read as RGB.
    window : rasterio.windows.Window
        Window to read, in pixel coordinates. Reads are boundless; areas
        outside the raster are filled with zeros.

    Returns
    -------
    np.ndarray
        Array of shape (height, width, 3) with dtype uint8.
    """
    data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)  # CHW
    data = np.transpose(data, (1, 2, 0))  # HWC
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    return data


def invalid_fraction(tile_hwc: np.ndarray, black_thresh: int = 5) -> float:
    """Return the fraction of pixels that are near-black (treated as invalid).

    A pixel is invalid when all three channels are at or below
    ``black_thresh``.

    Parameters
    ----------
    tile_hwc : np.ndarray
        HWC RGB image array.
    black_thresh : int
        Channel threshold below which a pixel counts as near-black.

    Returns
    -------
    float
        Fraction of invalid pixels in ``[0.0, 1.0]``.
    """
    invalid = (
        (tile_hwc[..., 0] <= black_thresh)
        & (tile_hwc[..., 1] <= black_thresh)
        & (tile_hwc[..., 2] <= black_thresh)
    )
    return float(invalid.mean())


def compute_valid_bbox(img_hwc: np.ndarray, black_thresh: int = 5) -> tuple[int, int, int, int]:
    """Return the bounding box of valid (non-near-black) pixels.

    A pixel is valid when any channel exceeds ``black_thresh``. When no
    valid pixels exist, the whole image extent is returned.

    Parameters
    ----------
    img_hwc : np.ndarray
        HWC RGB image array.
    black_thresh : int
        Channel threshold below which a pixel counts as near-black.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x0, y0, x1, y1)`` in pixel coordinates, exclusive of ``x1``/``y1``.
    """
    valid = (
        (img_hwc[..., 0] > black_thresh)
        | (img_hwc[..., 1] > black_thresh)
        | (img_hwc[..., 2] > black_thresh)
    )
    ys, xs = np.where(valid)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, img_hwc.shape[1], img_hwc.shape[0])
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def tile_geotiff(
    image_path: Path,
    out_dir: Path,
    tile_size: int = 512,
    overlap: int = 128,
    skip_invalid_fraction: float = 0.70,
    black_thresh: int = 5,
) -> list[dict]:
    """Tile a GeoTIFF into fixed-size RGB PNG tiles with JSON sidecar metadata.

    Only full ``tile_size`` windows inside the raster are considered, and
    tiles whose near-black pixel fraction meets or exceeds
    ``skip_invalid_fraction`` are skipped. Outputs are written to
    ``out_dir/images/*.png`` and ``out_dir/meta/*.json``; a combined index is
    written to ``out_dir/tiles_index.json``.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the source GeoTIFF (first three bands are used as RGB).
    out_dir : pathlib.Path
        Output directory for ``images/``, ``meta/``, and ``tiles_index.json``.
    tile_size : int
        Tile width/height in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels. Must be smaller than
        ``tile_size``.
    skip_invalid_fraction : float
        Skip tiles whose invalid (near-black) fraction is >= this value.
    black_thresh : int
        Channel threshold below which a pixel counts as near-black.

    Returns
    -------
    list[dict]
        Per-tile metadata records, including pixel offsets, invalid fraction,
        CRS, affine transform, and georeferenced bounds.
    """
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size.")

    images_dir = out_dir / "images"
    meta_dir = out_dir / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tiles_index: list[dict] = []
    saved = skipped = 0

    with rasterio.open(image_path) as src:
        H, W = src.height, src.width
        crs = src.crs
        tfm = src.transform

        # Estimate the valid (non-black) bbox on a coarse grid to avoid
        # reading the full raster into memory.
        sample_step = max(tile_size, step * 4)
        valid_lefts: list[int] = []
        valid_tops: list[int] = []

        for top in range(0, H, sample_step):
            for left in range(0, W, sample_step):
                win = Window(left, top, min(tile_size, W - left), min(tile_size, H - top))
                tile = read_tile_rgb(src, win)
                if invalid_fraction(tile, black_thresh) < 0.98:  # some data exists
                    valid_lefts.append(left)
                    valid_tops.append(top)

        if valid_lefts and valid_tops:
            x0 = max(0, min(valid_lefts) - tile_size)
            y0 = max(0, min(valid_tops) - tile_size)
            x1 = min(W, max(valid_lefts) + tile_size * 2)
            y1 = min(H, max(valid_tops) + tile_size * 2)
        else:
            x0, y0, x1, y1 = 0, 0, W, H

        # Main tiling loop over the estimated valid bbox.
        for top in range(y0, y1, step):
            for left in range(x0, x1, step):
                # fixed-size tiles only
                if (left + tile_size > W) or (top + tile_size > H):
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
                b = window_bounds(win, tfm)  # (left, bottom, right, top)

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
                    # keep real-world coordinates even though the tile is PNG
                    "crs": str(crs),
                    "transform": [
                        tile_tfm.a,
                        tile_tfm.b,
                        tile_tfm.c,
                        tile_tfm.d,
                        tile_tfm.e,
                        tile_tfm.f,
                    ],
                    "bounds": {"left": b[0], "bottom": b[1], "right": b[2], "top": b[3]},
                }

                (meta_dir / f"{tile_id}.json").write_text(json.dumps(meta, indent=2))
                tiles_index.append(meta)
                saved += 1

    index_path = out_dir / "tiles_index.json"
    index_path.write_text(json.dumps(tiles_index, indent=2))

    print(f"Saved {saved} tiles, skipped {skipped}; index written to {index_path}")
    return tiles_index
