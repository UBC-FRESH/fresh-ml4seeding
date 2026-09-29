"""Test data preprocessing functions."""

import numpy as np
import pytest
from PIL import Image

from ml4seeding.preprocessing import (
    analyze_mask,
    extract_x_coord,
    filter_trivial_tiles,
    load_image_mask_lists,
    normalize_image_key,
    normalize_mask_key,
    oversample_minority_tiles,
    spatial_split,
)


class TestNormalizeKeys:
    def test_image_key(self):
        assert normalize_image_key("x1024_y512_s512.jpg") == "x1024_y512_s512"
        assert normalize_image_key("tile.png") == "tile"

    def test_mask_key(self):
        assert normalize_mask_key("x1024_y512_s512_mask.png") == "x1024_y512_s512"
        assert normalize_mask_key("tile_mask.png") == "tile"
        assert normalize_mask_key("tile.png") == "tile"


class TestExtractXCoord:
    def test_valid(self):
        assert extract_x_coord("x10240_y1920_s512.png") == 10240
        assert extract_x_coord("x0_y0_s512.png") == 0

    def test_invalid(self):
        with pytest.raises(ValueError, match="Could not extract x-coordinate"):
            extract_x_coord("tile_no_coord.png")


class TestLoadImageMaskLists:
    def test_matching(self, tmp_path):
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        # Create test files
        for name in ["x0_y0_s512.png", "x100_y0_s512.png"]:
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / name)
            mask_name = name.replace(".png", "_mask.png")
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask_dir / mask_name)

        images, masks = load_image_mask_lists(img_dir, mask_dir)
        assert len(images) == 2
        assert len(masks) == 2

    def test_no_matches(self, tmp_path):
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / "img.png")

        images, masks = load_image_mask_lists(img_dir, mask_dir)
        assert len(images) == 0
        assert len(masks) == 0


class TestSpatialSplit:
    def test_split_ratios(self):
        images = [f"x{i * 100}_y0_s512.png" for i in range(100)]
        masks = [f"x{i * 100}_y0_s512_mask.png" for i in range(100)]

        result = spatial_split(images, masks, train_frac=0.7, val_frac=0.15)

        assert len(result["train"][0]) == 70
        assert len(result["val"][0]) == 15
        assert len(result["test"][0]) == 15

    def test_spatial_ordering(self):
        # Deliberately shuffled
        images = ["x300_y0_s512.png", "x100_y0_s512.png", "x200_y0_s512.png"]
        masks = ["x300_y0_s512_mask.png", "x100_y0_s512_mask.png", "x200_y0_s512_mask.png"]

        result = spatial_split(images, masks, train_frac=0.34, val_frac=0.33)

        # First tile should be x100 (lowest x coordinate)
        assert "x100" in result["train"][0][0]


class TestAnalyzeMask:
    def test_classes(self, tmp_path):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 1  # 50% good
        mask[5:, :5] = 3  # 25% poor
        # rest is 0 (background)

        path = tmp_path / "mask.png"
        Image.fromarray(mask).save(path)

        classes, ratios = analyze_mask(path)
        assert 0 in classes
        assert 1 in classes
        assert 3 in classes
        assert ratios[1] == pytest.approx(0.5)
        assert ratios[3] == pytest.approx(0.25)


class TestFilterTrivialTiles:
    def test_removes_poor_dominated(self, tmp_path):
        images = []
        masks = []
        for i in range(5):
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = np.full((4, 4), 3, dtype=np.uint8)  # All poor
            img_path = tmp_path / f"x{i}_y0_s512.png"
            mask_path = tmp_path / f"x{i}_y0_s512_mask.png"
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask).save(mask_path)
            images.append(str(img_path))
            masks.append(str(mask_path))

        filtered_i, filtered_m, removed = filter_trivial_tiles(images, masks)
        assert removed == 5
        assert len(filtered_i) == 0

    def test_keeps_good_tiles(self, tmp_path):
        images = []
        masks = []
        for i in range(3):
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = np.full((4, 4), 3, dtype=np.uint8)
            mask[0, 0] = 1  # One good pixel
            img_path = tmp_path / f"x{i}_y0_s512.png"
            mask_path = tmp_path / f"x{i}_y0_s512_mask.png"
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask).save(mask_path)
            images.append(str(img_path))
            masks.append(str(mask_path))

        filtered_i, filtered_m, removed = filter_trivial_tiles(images, masks)
        assert removed == 0
        assert len(filtered_i) == 3


class TestOversampleMinorityTiles:
    def test_oversample_good(self, tmp_path):
        images = []
        masks = []
        for i in range(2):
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[0, 0] = 1  # Has good class
            img_path = tmp_path / f"x{i}_y0_s512.png"
            mask_path = tmp_path / f"x{i}_y0_s512_mask.png"
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask).save(mask_path)
            images.append(str(img_path))
            masks.append(str(mask_path))

        out_i, out_m, extra_g, extra_f = oversample_minority_tiles(
            images, masks, oversample_good=3
        )
        assert len(out_i) == 2 + 4  # 2 original + 2*2 extra
        assert extra_g == 4
        assert extra_f == 0
