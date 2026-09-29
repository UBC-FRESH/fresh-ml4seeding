"""Tests for ml4seeding.orthomosaic using synthetic numpy arrays and rasters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ml4seeding.orthomosaic import (
    analyze_rgb_ortho,
    edge_density,
    laplacian_variance,
    make_quicklook,
    rgb_indices,
    summ_stats,
)

# ---------------------------------------------------------------------------
# laplacian_variance
# ---------------------------------------------------------------------------


class TestLaplacianVariance:
    """Tests for the laplacian_variance function."""

    def test_constant_image_returns_zero(self) -> None:
        """A perfectly flat image has zero Laplacian variance."""
        gray = np.full((64, 64), 128.0, dtype=np.float32)
        assert laplacian_variance(gray) == pytest.approx(0.0, abs=1e-6)

    def test_noisy_image_returns_positive(self) -> None:
        """A noisy image has positive Laplacian variance."""
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, size=(64, 64)).astype(np.float32)
        assert laplacian_variance(gray) > 0.0

    def test_sharper_image_higher_variance(self) -> None:
        """A high-frequency pattern yields higher variance than a smooth one."""
        smooth = np.tile(np.linspace(0, 255, 128, dtype=np.float32), (128, 1))
        x = np.arange(128, dtype=np.float32)
        sharp = np.tile((x % 2) * 255, (128, 1))
        assert laplacian_variance(sharp) > laplacian_variance(smooth)

    def test_returns_float(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        result = laplacian_variance(gray)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# edge_density
# ---------------------------------------------------------------------------


class TestEdgeDensity:
    """Tests for the edge_density function."""

    def test_returns_value_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, size=(64, 64)).astype(np.float32)
        result = edge_density(gray)
        assert 0.0 <= result <= 1.0

    def test_constant_image(self) -> None:
        """Constant image: all gradients are zero, so density is ~1 (all >= p95=0)."""
        gray = np.full((64, 64), 100.0, dtype=np.float32)
        result = edge_density(gray)
        # With all gradients equal, every pixel is >= the 95th percentile
        assert result == pytest.approx(1.0)

    def test_returns_float(self) -> None:
        gray = np.zeros((32, 32), dtype=np.uint8)
        result = edge_density(gray)
        assert isinstance(result, float)

    def test_expected_density_for_random_image(self) -> None:
        """For a random image, roughly 5 % of pixels exceed the 95th percentile."""
        rng = np.random.default_rng(42)
        gray = rng.integers(0, 256, size=(256, 256)).astype(np.float32)
        result = edge_density(gray)
        assert result == pytest.approx(0.05, abs=0.02)


# ---------------------------------------------------------------------------
# rgb_indices
# ---------------------------------------------------------------------------


class TestRgbIndices:
    """Tests for the rgb_indices function."""

    def test_output_shapes(self) -> None:
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        exg, vari, grvi = rgb_indices(rgb)
        assert exg.shape == (10, 10)
        assert vari.shape == (10, 10)
        assert grvi.shape == (10, 10)

    def test_output_dtype(self) -> None:
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        exg, vari, grvi = rgb_indices(rgb)
        assert exg.dtype == np.float32
        assert vari.dtype == np.float32
        assert grvi.dtype == np.float32

    def test_known_values(self) -> None:
        """Verify ExG, VARI, GRVI for a single known pixel."""
        # R=100, G=150, B=50
        rgb = np.array([[[100, 150, 50]]], dtype=np.uint8)
        exg, vari, grvi = rgb_indices(rgb)

        expected_exg = 2 * 150 - 100 - 50  # = 150
        expected_vari = (150 - 100) / (150 + 100 - 50 + 1e-6)  # ≈ 0.25
        expected_grvi = (150 - 100) / (150 + 100 + 1e-6)  # ≈ 0.2

        assert exg[0, 0] == pytest.approx(expected_exg, rel=1e-5)
        assert vari[0, 0] == pytest.approx(expected_vari, rel=1e-4)
        assert grvi[0, 0] == pytest.approx(expected_grvi, rel=1e-4)

    def test_green_pixel_positive_exg(self) -> None:
        """A green-dominant pixel should have positive ExG."""
        rgb = np.array([[[50, 200, 50]]], dtype=np.uint8)
        exg, _, _ = rgb_indices(rgb)
        assert exg[0, 0] > 0

    def test_red_pixel_negative_exg(self) -> None:
        """A red-dominant pixel should have negative ExG."""
        rgb = np.array([[[200, 50, 50]]], dtype=np.uint8)
        exg, _, _ = rgb_indices(rgb)
        assert exg[0, 0] < 0


# ---------------------------------------------------------------------------
# summ_stats
# ---------------------------------------------------------------------------


class TestSummStats:
    """Tests for the summ_stats function."""

    def test_basic_stats(self) -> None:
        x = np.arange(1, 101, dtype=np.float32)
        stats = summ_stats(x)
        assert stats is not None
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(100.0)
        assert stats["mean"] == pytest.approx(50.5)
        assert stats["p50"] == pytest.approx(50.5)

    def test_returns_none_for_empty(self) -> None:
        x = np.array([], dtype=np.float32)
        assert summ_stats(x) is None

    def test_returns_none_for_all_nan(self) -> None:
        x = np.array([np.nan, np.nan, np.nan])
        assert summ_stats(x) is None

    def test_ignores_nan_and_inf(self) -> None:
        x = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf])
        stats = summ_stats(x)
        assert stats is not None
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["mean"] == pytest.approx(2.0)

    def test_expected_keys(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        stats = summ_stats(x)
        assert stats is not None
        expected_keys = {"min", "p2", "p50", "p98", "max", "mean", "std"}
        assert set(stats.keys()) == expected_keys

    def test_single_value(self) -> None:
        x = np.array([42.0])
        stats = summ_stats(x)
        assert stats is not None
        assert stats["min"] == pytest.approx(42.0)
        assert stats["max"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fixtures for raster-based tests
# ---------------------------------------------------------------------------


def _write_synthetic_tif(path: Path, width: int = 256, height: int = 256) -> Path:
    """Write a small synthetic 3-band uint8 GeoTIFF for testing."""
    rng = np.random.default_rng(99)
    data = rng.integers(10, 200, size=(3, height, width), dtype=np.uint8)
    transform = from_origin(500000, 5100000, 0.1, 0.1)  # 10 cm pixels, UTM-ish
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32610",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


@pytest.fixture()
def synthetic_tif(tmp_path: Path) -> Path:
    """Create a synthetic GeoTIFF in a temporary directory."""
    return _write_synthetic_tif(tmp_path / "test_ortho.tif")


# ---------------------------------------------------------------------------
# make_quicklook
# ---------------------------------------------------------------------------


class TestMakeQuicklook:
    """Tests for the make_quicklook function."""

    def test_creates_png(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_png = tmp_path / "quicklook.png"
        with rasterio.open(synthetic_tif) as src:
            make_quicklook(src, out_png, max_side=100)
        assert out_png.exists()

    def test_returns_uint8_array(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_png = tmp_path / "quicklook.png"
        with rasterio.open(synthetic_tif) as src:
            result = make_quicklook(src, out_png, max_side=100)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_respects_max_side(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_png = tmp_path / "quicklook.png"
        with rasterio.open(synthetic_tif) as src:
            result = make_quicklook(src, out_png, max_side=100)
        assert max(result.shape[0], result.shape[1]) <= 100


# ---------------------------------------------------------------------------
# analyze_rgb_ortho
# ---------------------------------------------------------------------------


class TestAnalyzeRgbOrtho:
    """Tests for the analyze_rgb_ortho function."""

    def test_returns_dict(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        assert isinstance(report, dict)

    def test_creates_report_json(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        report_path = out_dir / "report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)

    def test_creates_quicklook_png(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        assert (out_dir / "quicklook.png").exists()

    def test_report_has_expected_sections(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        expected_sections = {
            "ortho_path",
            "rgb_bands",
            "samples",
            "metadata",
            "band_stats_sampled",
            "nodata_black_estimates",
            "rgb_indices_sampled",
            "sharpness_sampled",
            "outputs",
        }
        assert expected_sections.issubset(set(report.keys()))

    def test_metadata_fields(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        meta = report["metadata"]
        assert meta["width_px"] == 256
        assert meta["height_px"] == 256
        assert meta["band_count"] == 3
        assert meta["crs"] == "EPSG:32610"
        assert meta["coverage_area_m2"] > 0

    def test_band_stats_populated(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        for band in ("R", "G", "B"):
            stats = report["band_stats_sampled"][band]
            assert stats is not None
            assert "mean" in stats
            assert "p50" in stats

    def test_vegetation_indices_populated(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        for idx in ("ExG", "VARI", "GRVI"):
            stats = report["rgb_indices_sampled"][idx]
            assert stats is not None

    def test_sharpness_populated(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "report"
        report = analyze_rgb_ortho(synthetic_tif, out_dir, n_samples=3, win_size=64)
        sharp = report["sharpness_sampled"]
        assert sharp["laplacian_variance_mean"] is not None
        assert sharp["edge_density_mean"] is not None

    def test_reproducible_with_same_seed(self, synthetic_tif: Path, tmp_path: Path) -> None:
        out1 = tmp_path / "r1"
        out2 = tmp_path / "r2"
        r1 = analyze_rgb_ortho(synthetic_tif, out1, n_samples=3, win_size=64, seed=42)
        r2 = analyze_rgb_ortho(synthetic_tif, out2, n_samples=3, win_size=64, seed=42)
        assert r1["band_stats_sampled"] == r2["band_stats_sampled"]
        assert r1["sharpness_sampled"] == r2["sharpness_sampled"]
