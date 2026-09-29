"""Test tiling functions."""

import numpy as np
import pytest

from ml4seeding.tiling import (
    compute_valid_bbox,
    invalid_fraction,
)


class TestInvalidFraction:
    def test_all_black(self):
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        assert invalid_fraction(tile) == 1.0

    def test_all_valid(self):
        tile = np.full((10, 10, 3), 200, dtype=np.uint8)
        assert invalid_fraction(tile) == 0.0

    def test_mixed(self):
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        tile[:5] = 200  # Top half valid
        assert invalid_fraction(tile) == pytest.approx(0.5)


class TestComputeValidBBox:
    def test_all_valid(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        bbox = compute_valid_bbox(img)
        assert bbox == (0, 0, 10, 10)

    def test_all_black(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        bbox = compute_valid_bbox(img)
        # Falls back to whole image
        assert bbox == (0, 0, 10, 10)

    def test_partial(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[2:5, 3:7] = 200
        bbox = compute_valid_bbox(img)
        assert bbox == (3, 2, 7, 5)
