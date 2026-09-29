"""Tests for ml4seeding.tiling using synthetic GeoTIFFs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.windows import Window

from ml4seeding.tiling import (
    compute_valid_bbox,
    invalid_fraction,
    read_tile_rgb,
    tile_geotiff,
)


def _write_geotiff(path: Path, data_chw: np.ndarray) -> None:
    """Write a CHW array to a GeoTIFF with a known transform and CRS."""
    count, height, width = data_chw.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data_chw.dtype,
        crs="EPSG:32610",
        transform=from_origin(500000, 6000000, 0.01, 0.01),
    ) as dst:
        dst.write(data_chw)


@pytest.fixture
def synthetic_tiff(tmp_path: Path) -> Path:
    """Create a 200x160 RGB GeoTIFF: left half bright, right half black."""
    rng = np.random.default_rng(42)
    img = np.zeros((3, 160, 200), dtype=np.uint8)
    img[:, :, :100] = rng.integers(50, 256, size=(3, 160, 100), dtype=np.uint8)
    path = tmp_path / "synthetic.tif"
    _write_geotiff(path, img)
    return path


def test_read_tile_rgb_shape_and_dtype(synthetic_tiff: Path):
    with rasterio.open(synthetic_tiff) as src:
        tile = read_tile_rgb(src, Window(0, 0, 64, 64))
    assert tile.shape == (64, 64, 3)
    assert tile.dtype == np.uint8


def test_read_tile_rgb_values(synthetic_tiff: Path):
    with rasterio.open(synthetic_tiff) as src:
        tile = read_tile_rgb(src, Window(10, 10, 32, 32))
        expected = np.transpose(src.read([1, 2, 3], window=Window(10, 10, 32, 32)), (1, 2, 0))
    np.testing.assert_array_equal(tile, expected)


def test_read_tile_rgb_boundless_fill(synthetic_tiff: Path):
    # window extends past the right edge of the raster -> zero fill
    with rasterio.open(synthetic_tiff) as src:
        tile = read_tile_rgb(src, Window(180, 0, 64, 64))
    assert tile.shape == (64, 64, 3)
    assert tile[:, :20].max() == 0  # black right half of the source
    assert tile[:, 20:].max() == 0  # out-of-raster columns are fill_value=0


def test_invalid_fraction_all_black():
    tile = np.zeros((16, 16, 3), dtype=np.uint8)
    assert invalid_fraction(tile) == 1.0


def test_invalid_fraction_all_valid():
    tile = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert invalid_fraction(tile) == 0.0


def test_invalid_fraction_half():
    tile = np.zeros((16, 16, 3), dtype=np.uint8)
    tile[:, :8] = 100
    assert invalid_fraction(tile) == pytest.approx(0.5)


def test_invalid_fraction_threshold_boundary():
    # all channels exactly at the threshold count as invalid
    assert invalid_fraction(np.full((8, 8, 3), 5, dtype=np.uint8), black_thresh=5) == 1.0
    assert invalid_fraction(np.full((8, 8, 3), 6, dtype=np.uint8), black_thresh=5) == 0.0


def test_compute_valid_bbox_partial():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[10:30, 20:45] = 100
    assert compute_valid_bbox(img) == (20, 10, 45, 30)


def test_compute_valid_bbox_all_black_returns_full_extent():
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    assert compute_valid_bbox(img) == (0, 0, 50, 40)


def test_compute_valid_bbox_single_bright_channel_counts():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[3, 7, 1] = 50  # only the green channel is bright
    assert compute_valid_bbox(img) == (7, 3, 8, 4)


def test_tile_geotiff_outputs(synthetic_tiff: Path, tmp_path: Path):
    out_dir = tmp_path / "tiles_out"
    index = tile_geotiff(synthetic_tiff, out_dir, tile_size=64, overlap=16)

    assert (out_dir / "images").is_dir()
    assert (out_dir / "meta").is_dir()
    assert (out_dir / "tiles_index.json").exists()
    assert len(index) > 0

    meta = index[0]
    expected_keys = {
        "tile_id",
        "image_file",
        "source_image",
        "left_px",
        "top_px",
        "tile_size_px",
        "overlap_px",
        "step_px",
        "invalid_fraction",
        "crs",
        "transform",
        "bounds",
    }
    assert expected_keys <= set(meta)
    assert meta["source_image"] == "synthetic.tif"
    assert meta["crs"] == "EPSG:32610"
    assert len(meta["transform"]) == 6
    assert meta["tile_size_px"] == 64
    assert meta["overlap_px"] == 16
    assert meta["step_px"] == 48

    # PNG round-trip
    png_path = out_dir / "images" / meta["image_file"]
    assert png_path.exists()
    arr = np.array(Image.open(png_path))
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8

    # per-tile JSON sidecar matches the in-memory record
    sidecar = json.loads((out_dir / "meta" / f"{meta['tile_id']}.json").read_text())
    assert sidecar == meta

    # index file matches the returned value
    on_disk = json.loads((out_dir / "tiles_index.json").read_text())
    assert on_disk == index


def test_tile_geotiff_skips_black_tiles(synthetic_tiff: Path, tmp_path: Path):
    out_dir = tmp_path / "tiles_out"
    index = tile_geotiff(synthetic_tiff, out_dir, tile_size=64, overlap=16)

    # every saved tile is below the skip threshold
    assert all(m["invalid_fraction"] < 0.70 for m in index)
    # no saved tile sits fully inside the black right half (x >= 100)
    assert all(m["left_px"] + 64 <= 100 or m["left_px"] < 100 for m in index)
    # deterministic layout for this fixture: lefts {0, 48}, tops {0, 48, 96}
    assert sorted({m["left_px"] for m in index}) == [0, 48]
    assert sorted({m["top_px"] for m in index}) == [0, 48, 96]
    assert len(index) == 6


def test_tile_geotiff_georeferencing(synthetic_tiff: Path, tmp_path: Path):
    out_dir = tmp_path / "tiles_out"
    index = tile_geotiff(synthetic_tiff, out_dir, tile_size=64, overlap=16)

    meta = next(m for m in index if m["left_px"] == 48 and m["top_px"] == 0)
    # transform origin shifts by 48 px * 0.01 m/px east of 500000
    assert meta["transform"][2] == pytest.approx(500000.48)
    assert meta["transform"][5] == pytest.approx(6000000.0)
    assert meta["bounds"]["left"] == pytest.approx(500000.48)
    assert meta["bounds"]["right"] == pytest.approx(500000.48 + 64 * 0.01)
    assert meta["bounds"]["top"] == pytest.approx(6000000.0)
    assert meta["bounds"]["bottom"] == pytest.approx(6000000.0 - 64 * 0.01)


def test_tile_geotiff_rejects_overlap_ge_tile_size(synthetic_tiff: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="overlap"):
        tile_geotiff(synthetic_tiff, tmp_path / "out", tile_size=64, overlap=64)


def test_tile_geotiff_all_black_image(tmp_path: Path):
    path = tmp_path / "black.tif"
    _write_geotiff(path, np.zeros((3, 128, 128), dtype=np.uint8))
    index = tile_geotiff(path, tmp_path / "tiles_out", tile_size=64, overlap=16)
    assert index == []
    assert json.loads((tmp_path / "tiles_out" / "tiles_index.json").read_text()) == []
