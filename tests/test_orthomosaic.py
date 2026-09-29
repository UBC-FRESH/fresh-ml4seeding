"""Test orthomosaic analysis functions."""

import numpy as np
import pytest

from ml4seeding.orthomosaic import (
    edge_density,
    laplacian_variance,
    rgb_indices,
    summ_stats,
)


class TestLaplacianVariance:
    def test_flat_image(self):
        flat = np.ones((10, 10), dtype=np.float32) * 128
        assert laplacian_variance(flat) == pytest.approx(0.0, abs=1e-6)

    def test_noisy_image(self):
        rng = np.random.RandomState(42)
        noisy = rng.rand(50, 50).astype(np.float32) * 255
        assert laplacian_variance(noisy) > 0

    def test_sharp_edges(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[:, 10:] = 255
        assert laplacian_variance(img) > 0


class TestEdgeDensity:
    def test_flat(self):
        flat = np.ones((10, 10), dtype=np.float32) * 128
        # With a flat image, all gradients are 0, so 95th percentile is 0
        # and all pixels pass the >= threshold
        assert 0 <= edge_density(flat) <= 1.0

    def test_with_edges(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[:, 10:] = 255
        ed = edge_density(img)
        assert 0 < ed <= 1.0


class TestRGBIndices:
    def test_output_shapes(self):
        rgb = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        exg, vari, grvi = rgb_indices(rgb)
        assert exg.shape == (10, 10)
        assert vari.shape == (10, 10)
        assert grvi.shape == (10, 10)

    def test_green_pixel(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = [0, 255, 0]  # Pure green
        exg, vari, grvi = rgb_indices(rgb)
        assert exg[0, 0] > 0  # Excess green
        assert grvi[0, 0] > 0  # Green > Red

    def test_red_pixel(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]  # Pure red
        exg, vari, grvi = rgb_indices(rgb)
        assert exg[0, 0] < 0  # Negative excess green


class TestSummStats:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = summ_stats(x)
        assert stats is not None
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == pytest.approx(3.0)

    def test_empty(self):
        assert summ_stats(np.array([])) is None

    def test_nan_filtered(self):
        x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        stats = summ_stats(x)
        assert stats is not None
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
