"""Tests for ml4seeding.grid using synthetic images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ml4seeding.grid import (
    add_grid_to_tiles,
    camo_color_horizontal,
    camo_color_vertical,
    draw_grid_camo,
    grid_spacing_px,
)


def test_grid_spacing_px_typical():
    assert grid_spacing_px(0.0075, 10.0) == 13  # 10 cm at 0.75 cm/px
    assert grid_spacing_px(0.01, 10.0) == 10


def test_grid_spacing_px_clamped_to_minimum():
    assert grid_spacing_px(1.0, 1.0) == 2


def test_camo_color_vertical_picks_neighbor_color():
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    img[:, :20] = (200, 10, 10)  # red-ish left half
    img[:, 20:] = (10, 10, 200)  # blue-ish right half

    color = camo_color_vertical(img, x=5, thickness=1, band=6)
    assert color.dtype == np.uint8
    assert color.shape == (3,)
    np.testing.assert_array_equal(color, (200, 10, 10))

    color = camo_color_vertical(img, x=35, thickness=1, band=6)
    np.testing.assert_array_equal(color, (10, 10, 200))


def test_camo_color_horizontal_picks_neighbor_color():
    img = np.zeros((40, 20, 3), dtype=np.uint8)
    img[:20] = (10, 200, 10)  # green-ish top half
    img[20:] = (10, 10, 200)  # blue-ish bottom half

    color = camo_color_horizontal(img, y=5, thickness=1, band=6)
    np.testing.assert_array_equal(color, (10, 200, 10))

    color = camo_color_horizontal(img, y=35, thickness=1, band=6)
    np.testing.assert_array_equal(color, (10, 10, 200))


def test_camo_color_vertical_edge_column_uses_available_side():
    img = np.full((10, 10, 3), 80, dtype=np.uint8)
    color = camo_color_vertical(img, x=0, thickness=1, band=6)
    np.testing.assert_array_equal(color, (80, 80, 80))


def test_draw_grid_camo_alpha_zero_leaves_image_unchanged():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    out = draw_grid_camo(img, spacing=8, alpha=0.0)
    np.testing.assert_array_equal(out, img)


def test_draw_grid_camo_preserves_shape_dtype_and_input():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    original = img.copy()
    out = draw_grid_camo(img, spacing=8, thickness=1, band=6, alpha=0.8)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(img, original)  # input not modified


def test_draw_grid_camo_blends_neighbor_color_into_line():
    # black image with a bright strip left of column 8
    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[:, 2:8] = 200
    out = draw_grid_camo(img, spacing=8, thickness=1, band=6, alpha=1.0)

    # check row 4, which is not crossed by a horizontal grid line:
    # the vertical line at x=8 sees 6 bright columns (left) and 6 black (right)
    assert tuple(out[4, 8]) == (100, 100, 100)
    # something changed relative to the input
    assert not np.array_equal(out, img)
    # far from any bright strip, lines stay black (camouflaged into black)
    assert tuple(out[4, 24]) == (0, 0, 0)


def _make_tile_set(tmp_path: Path, n: int = 2) -> tuple[Path, Path]:
    """Create n synthetic tiles with matching JSON sidecars."""
    tiles_dir = tmp_path / "images"
    meta_dir = tmp_path / "meta"
    tiles_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(n):
        tile_id = f"x0_y{i * 64}_s64"
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(tiles_dir / f"{tile_id}.png")
        meta = {
            "tile_id": tile_id,
            "image_file": f"{tile_id}.png",
            "transform": [0.0075, 0.0, 500000.0, 0.0, -0.0075, 6000000.0],
        }
        (meta_dir / f"{tile_id}.json").write_text(json.dumps(meta))
    return tiles_dir, meta_dir


def test_add_grid_to_tiles_with_explicit_pixel_size(tmp_path: Path):
    tiles_dir, meta_dir = _make_tile_set(tmp_path, n=3)
    out_dir = tmp_path / "grid_out"
    saved, skipped = add_grid_to_tiles(tiles_dir, meta_dir, out_dir, pixel_size_m=0.0075)

    assert (saved, skipped) == (3, 0)
    outs = sorted(out_dir.glob("*.png"))
    assert len(outs) == 3
    arr = np.array(Image.open(outs[0]))
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8
    # gridded output differs from the source tile
    original = np.array(Image.open(tiles_dir / outs[0].name))
    assert not np.array_equal(arr, original)


def test_add_grid_to_tiles_pixel_size_from_meta_transform(tmp_path: Path):
    tiles_dir, meta_dir = _make_tile_set(tmp_path, n=2)
    saved, skipped = add_grid_to_tiles(tiles_dir, meta_dir, tmp_path / "out")
    assert (saved, skipped) == (2, 0)
    assert len(list((tmp_path / "out").glob("*.png"))) == 2


def test_add_grid_to_tiles_skips_tiles_without_meta(tmp_path: Path):
    tiles_dir, meta_dir = _make_tile_set(tmp_path, n=1)
    orphan = np.zeros((64, 64, 3), dtype=np.uint8)
    Image.fromarray(orphan).save(tiles_dir / "orphan.png")

    saved, skipped = add_grid_to_tiles(tiles_dir, meta_dir, tmp_path / "out")
    assert (saved, skipped) == (1, 1)
    assert not (tmp_path / "out" / "orphan.png").exists()


def test_add_grid_to_tiles_skips_meta_without_transform(tmp_path: Path):
    tiles_dir, meta_dir = _make_tile_set(tmp_path, n=1)
    meta_path = next(meta_dir.glob("*.json"))
    meta_path.write_text(json.dumps({"tile_id": meta_path.stem}))

    # no pixel_size_m given and meta has no transform -> skipped
    saved, skipped = add_grid_to_tiles(tiles_dir, meta_dir, tmp_path / "out")
    assert (saved, skipped) == (0, 1)


def test_add_grid_to_tiles_raises_when_no_pngs(tmp_path: Path):
    tiles_dir = tmp_path / "images"
    tiles_dir.mkdir()
    (tmp_path / "meta").mkdir()
    with pytest.raises(FileNotFoundError, match="No PNG tiles"):
        add_grid_to_tiles(tiles_dir, tmp_path / "meta", tmp_path / "out")
