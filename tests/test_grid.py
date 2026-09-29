"""Test grid overlay functions."""

import numpy as np

from ml4seeding.grid import (
    camo_color_horizontal,
    camo_color_vertical,
    draw_grid_camo,
    grid_spacing_px,
)


class TestGridSpacingPx:
    def test_basic(self):
        # 10 cm grid at 0.75 cm/pixel => ~13 pixels
        assert grid_spacing_px(0.0075, 10.0) == 13

    def test_minimum(self):
        assert grid_spacing_px(1.0, 1.0) >= 2


class TestCamoColor:
    def test_vertical_line(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        color = camo_color_vertical(img, 10, 1, 6)
        assert color.shape == (3,)
        assert color.dtype == np.uint8
        # With uniform image, camo color should be ~100
        assert all(abs(int(c) - 100) <= 5 for c in color)

    def test_horizontal_line(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        color = camo_color_horizontal(img, 10, 1, 6)
        assert color.shape == (3,)
        assert color.dtype == np.uint8


class TestDrawGridCamo:
    def test_output_shape(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        result = draw_grid_camo(img, spacing=5)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_grid_lines_drawn(self):
        # Use a gradient image so camo color differs from the original pixel value
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(20):
            img[:, i] = [i * 12, i * 12, i * 12]
        result = draw_grid_camo(img, spacing=5, alpha=0.5)
        # The grid lines should have slightly different values from the original
        assert not np.array_equal(result, img)

    def test_alpha_zero(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        result = draw_grid_camo(img, spacing=5, alpha=0.0)
        # With alpha=0, output should be identical to input
        np.testing.assert_array_equal(result, img)
