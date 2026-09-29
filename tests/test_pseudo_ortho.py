"""Tests for ml4seeding.pseudo_ortho using synthetic DJI-style imagery."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from ml4seeding.pseudo_ortho import (
    build_pseudo_ortho,
    estimate_spacing_m,
    extract_dji_pose,
    extract_xmp_block,
    get_lat_lon,
    latlon_to_local_xy_m,
    select_images_for_coverage,
)

_XMP_SAMPLE = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
    ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    '  <rdf:Description rdf:about=""'
    ' drone-dji:FlightYawDegree="14.4"'
    ' drone-dji:GimbalYawDegree="-165.6"'
    ' drone-dji:GimbalPitchDegree="-89.9"'
    ' drone-dji:GimbalRollDegree="180.0"/>\n'
    " </rdf:RDF>\n"
    "</x:xmpmeta>"
)


def _dd_to_dms(dd: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    """Convert decimal degrees to an EXIF degree/minute/second rational tuple."""
    dd = abs(dd)
    d = int(dd)
    m_full = (dd - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    return (
        IFDRational(d, 1),
        IFDRational(m, 1),
        IFDRational(int(round(s * 10_000)), 10_000),
    )


def _make_dji_jpeg(
    path: Path,
    lat: float | None = None,
    lon: float | None = None,
    *,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (120, 80, 40),
    xmp: bool = False,
) -> Path:
    """Write a synthetic DJI-style JPEG with optional GPS EXIF and XMP blocks."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        arr[..., channel] = value
    im = Image.fromarray(arr)

    if lat is not None and lon is not None:
        exif = Image.Exif()
        exif[34853] = {  # 0x8825 = GPSInfo IFD
            1: "N" if lat >= 0 else "S",
            2: _dd_to_dms(lat),
            3: "E" if lon >= 0 else "W",
            4: _dd_to_dms(lon),
        }
        im.save(path, exif=exif)
    else:
        im.save(path)

    if xmp:
        with open(path, "ab") as f:
            f.write(_XMP_SAMPLE.encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# GPS extraction from EXIF
# ---------------------------------------------------------------------------


def test_get_lat_lon_roundtrip(tmp_path):
    p = _make_dji_jpeg(tmp_path / "gps.jpg", lat=49.2583, lon=-123.1207)
    result = get_lat_lon(p)
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(49.2583, abs=1e-6)
    assert lon == pytest.approx(-123.1207, abs=1e-6)


def test_get_lat_lon_southern_eastern_hemispheres(tmp_path):
    p = _make_dji_jpeg(tmp_path / "gps_se.jpg", lat=-33.86, lon=151.21)
    lat, lon = get_lat_lon(p)
    assert lat == pytest.approx(-33.86, abs=1e-6)
    assert lon == pytest.approx(151.21, abs=1e-6)


def test_get_lat_lon_missing_exif(tmp_path):
    p = _make_dji_jpeg(tmp_path / "nogps.jpg")
    assert get_lat_lon(p) is None


def test_get_lat_lon_not_an_image(tmp_path):
    p = tmp_path / "junk.jpg"
    p.write_text("this is not a jpeg")
    assert get_lat_lon(p) is None


# ---------------------------------------------------------------------------
# Local metric conversion
# ---------------------------------------------------------------------------


def test_latlon_to_local_xy_m_origin():
    x, y = latlon_to_local_xy_m(49.0, -123.0, 49.0, -123.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_latlon_to_local_xy_m_known_offsets():
    # One degree of latitude is ~111.2 km everywhere.
    x, y = latlon_to_local_xy_m(50.0, -123.0, 49.0, -123.0)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(111_194.9, rel=1e-3)

    # One degree of longitude shrinks with cos(latitude).
    x, y = latlon_to_local_xy_m(49.0, -122.0, 49.0, -123.0)
    assert y == pytest.approx(0.0, abs=1e-6)
    assert x == pytest.approx(111_194.9 * math.cos(math.radians(49.0)), rel=1e-3)


# ---------------------------------------------------------------------------
# Spacing estimation
# ---------------------------------------------------------------------------


def test_estimate_spacing_m_grid():
    gx, gy = np.meshgrid(np.arange(6) * 2.0, np.arange(6) * 2.0)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    assert estimate_spacing_m(xy) == pytest.approx(2.0)


def test_estimate_spacing_m_too_few_points():
    with pytest.raises(ValueError, match="at least two"):
        estimate_spacing_m(np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# XMP / DJI pose extraction
# ---------------------------------------------------------------------------


def test_extract_xmp_block(tmp_path):
    p = _make_dji_jpeg(tmp_path / "xmp.jpg", xmp=True)
    block = extract_xmp_block(p)
    assert block is not None
    assert block.startswith("<x:xmpmeta")
    assert block.rstrip().endswith("</x:xmpmeta>")
    assert "GimbalYawDegree" in block


def test_extract_xmp_block_missing(tmp_path):
    p = _make_dji_jpeg(tmp_path / "noxmp.jpg")
    assert extract_xmp_block(p) is None


def test_extract_dji_pose(tmp_path):
    p = _make_dji_jpeg(tmp_path / "pose.jpg", xmp=True)
    assert extract_dji_pose(p) == {
        "FlightYawDegree": 14.4,
        "GimbalYawDegree": -165.6,
        "GimbalPitchDegree": -89.9,
        "GimbalRollDegree": 180.0,
    }


def test_extract_dji_pose_missing(tmp_path):
    p = _make_dji_jpeg(tmp_path / "nopose.jpg")
    assert extract_dji_pose(p) is None


# ---------------------------------------------------------------------------
# Coverage-based selection
# ---------------------------------------------------------------------------


def test_select_images_for_coverage_explicit_grid():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])

    # Each point falls into its own 6 m cell.
    assert select_images_for_coverage(xy, grid_cell_m=6.0) == [0, 1, 2, 3]

    # One big cell keeps only the point closest to the cell center (10, 10).
    assert select_images_for_coverage(xy, grid_cell_m=20.0) == [3]

    # Allowing two images per cell keeps the center point plus one other.
    idx = select_images_for_coverage(xy, grid_cell_m=20.0, max_per_cell=2)
    assert len(idx) == 2
    assert 3 in idx


def test_select_images_for_coverage_auto_cell():
    gx, gy = np.meshgrid(np.arange(5) * 2.0, np.arange(5) * 2.0)
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    idx = select_images_for_coverage(xy)
    assert idx == sorted(set(idx))
    assert 0 < len(idx) <= len(xy)
    assert max(idx) < len(xy)


def test_select_images_for_coverage_empty():
    assert select_images_for_coverage(np.empty((0, 2))) == []


# ---------------------------------------------------------------------------
# Mosaic building
# ---------------------------------------------------------------------------


def test_build_pseudo_ortho(tmp_path):
    colors = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)]
    paths = [
        _make_dji_jpeg(tmp_path / f"img_{i}.jpg", size=(64, 64), color=c)
        for i, c in enumerate(colors)
    ]
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    out_path = tmp_path / "mosaic.tif"

    build_pseudo_ortho(paths, xy, out_path, spacing=2.0, downscale=0.5)

    assert out_path.exists()
    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert src.crs.to_string() == "EPSG:32610"
        assert src.width > 0
        assert src.height > 0
        data = src.read()
    # All three channels receive content from the distinctly colored tiles.
    assert (data[0] > 100).any()
    assert (data[1] > 100).any()
    assert (data[2] > 100).any()


def test_build_pseudo_ortho_with_yaw_rotation(tmp_path):
    # Images with XMP gimbal data exercise the yaw-rotation code path.
    paths = [_make_dji_jpeg(tmp_path / f"yaw_{i}.jpg", size=(64, 64), xmp=True) for i in range(2)]
    xy = np.array([[0.0, 0.0], [2.0, 0.0]])
    out_path = tmp_path / "mosaic_rotated.tif"

    build_pseudo_ortho(paths, xy, out_path, spacing=2.0, downscale=0.5)

    assert out_path.exists()
    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert (src.read() > 0).any()


def test_build_pseudo_ortho_suffix_replaced(tmp_path):
    paths = [_make_dji_jpeg(tmp_path / "img.jpg", size=(32, 32))]
    xy = np.array([[0.0, 0.0]])
    build_pseudo_ortho(paths, xy, tmp_path / "mosaic.jpg", spacing=2.0, downscale=0.5)
    assert (tmp_path / "mosaic.tif").exists()


def test_build_pseudo_ortho_invalid_inputs(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        build_pseudo_ortho([], np.empty((0, 2)), tmp_path / "m.tif", spacing=1.0)

    paths = [_make_dji_jpeg(tmp_path / "img.jpg", size=(32, 32))]
    with pytest.raises(ValueError, match="same length"):
        build_pseudo_ortho(paths, np.zeros((2, 2)), tmp_path / "m.tif", spacing=1.0)
