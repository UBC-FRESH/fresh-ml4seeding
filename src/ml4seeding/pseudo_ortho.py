"""Pseudo-orthomosaic generation from raw drone images via GPS stitching.

Re-implementation of the prototype notebook
``examples/01_1_build_pseudo_ortho_a10_seg1.ipynb`` from UBC-FRESH/ML4seeding.

The pipeline is:

1. Extract GPS positions from DJI EXIF metadata (:func:`get_lat_lon`).
2. Convert lat/lon to a local metric frame (:func:`latlon_to_local_xy_m`).
3. Select as many images as needed for area coverage on a spatial grid
   (:func:`select_images_for_coverage`).
4. Place yaw-corrected, downscaled copies on a canvas at their GPS positions
   with alpha blending and write a GeoTIFF (:func:`build_pseudo_ortho`).

This is a GPS-placement mosaic (translation plus yaw rotation only); it is not
a true orthorectification.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import ExifTags, Image
from rasterio.transform import from_origin

__all__ = [
    "build_pseudo_ortho",
    "estimate_spacing_m",
    "extract_dji_pose",
    "extract_xmp_block",
    "get_lat_lon",
    "latlon_to_local_xy_m",
    "select_images_for_coverage",
]

_EARTH_RADIUS_M = 6_371_000.0

#: DJI XMP pose fields and the regexes used to parse them from the XMP block.
_DJI_POSE_PATTERNS = {
    "FlightYawDegree": r'FlightYawDegree="([^"]+)"',
    "GimbalYawDegree": r'GimbalYawDegree="([^"]+)"',
    "GimbalPitchDegree": r'GimbalPitchDegree="([^"]+)"',
    "GimbalRollDegree": r'GimbalRollDegree="([^"]+)"',
}


def _to_float(value: Any) -> float:
    """Convert an EXIF numeric value (float, rational, or num/den pair) to a float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value[0]) / float(value[1])


def _dms_to_dd(dms: Any, ref: str) -> float:
    """Convert an EXIF degree/minute/second coordinate to decimal degrees."""
    dd = _to_float(dms[0]) + _to_float(dms[1]) / 60.0 + _to_float(dms[2]) / 3600.0
    if ref in ("S", "W"):
        dd *= -1
    return dd


def get_lat_lon(jpg_path: Path) -> tuple[float, float] | None:
    """Extract the GPS position from a DJI JPEG's EXIF metadata.

    Args:
        jpg_path: Path to the JPEG file.

    Returns:
        ``(latitude, longitude)`` in decimal degrees, or ``None`` when the file
        cannot be read or carries no GPS tags.
    """
    try:
        with Image.open(jpg_path) as im:
            exif = im._getexif()
            if exif is None:
                return None

            exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            gps_info = exif_dict.get("GPSInfo")
            if gps_info is None:
                return None
            if not hasattr(gps_info, "items"):
                # Some Pillow versions expose GPSInfo only as an IFD offset.
                gps_info = im.getexif().get_ifd(ExifTags.IFD.GPSInfo)

            gps_data = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}

        lat = gps_data.get("GPSLatitude")
        lat_ref = gps_data.get("GPSLatitudeRef")
        lon = gps_data.get("GPSLongitude")
        lon_ref = gps_data.get("GPSLongitudeRef")
        if not (lat and lat_ref and lon and lon_ref):
            return None

        return _dms_to_dd(lat, lat_ref), _dms_to_dd(lon, lon_ref)
    except Exception:
        return None


def latlon_to_local_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert lat/lon to local metric offsets (equirectangular approximation).

    Accurate enough for small areas such as a single drone flight block.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        lat0: Origin latitude in decimal degrees.
        lon0: Origin longitude in decimal degrees.

    Returns:
        ``(x, y)`` offsets in meters east/north of the origin.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    x = (lon_rad - lon0_rad) * math.cos(lat0_rad) * _EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * _EARTH_RADIUS_M
    return x, y


def estimate_spacing_m(xy: np.ndarray) -> float:
    """Estimate typical GPS spacing as the median nearest-neighbor distance.

    Args:
        xy: Array of shape ``(n, 2)`` with local metric coordinates.

    Returns:
        Median nearest-neighbor distance in meters.

    Raises:
        ValueError: If fewer than two points are given.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        raise ValueError("Need at least two points to estimate spacing.")

    # Brute force is fine for a few hundred images; use a KD-tree for thousands.
    dmins = []
    for i in range(len(xy)):
        diffs = xy - xy[i]
        d = np.sqrt((diffs**2).sum(axis=1))
        d[i] = np.inf
        dmins.append(d.min())
    return float(np.median(dmins))


def extract_xmp_block(jpg_path: Path) -> str | None:
    """Extract the raw XMP metadata packet embedded in a JPEG file.

    Args:
        jpg_path: Path to the JPEG file.

    Returns:
        The ``<x:xmpmeta ... </x:xmpmeta>`` block decoded as text, or ``None``
        when no XMP packet is present.
    """
    data = Path(jpg_path).read_bytes()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta")
    if start == -1 or end == -1:
        return None
    return data[start : end + 12].decode(errors="ignore")


def extract_dji_pose(jpg_path: Path) -> dict[str, float] | None:
    """Extract the DJI flight/gimbal pose from a JPEG's XMP metadata.

    Args:
        jpg_path: Path to the JPEG file.

    Returns:
        Mapping with any of ``FlightYawDegree``, ``GimbalYawDegree``,
        ``GimbalPitchDegree`` and ``GimbalRollDegree`` found in the XMP block,
        or ``None`` when no pose fields are present.
    """
    xmp = extract_xmp_block(jpg_path)
    if xmp is None:
        return None

    pose: dict[str, float] = {}
    for key, pattern in _DJI_POSE_PATTERNS.items():
        match = re.search(pattern, xmp)
        if match:
            pose[key] = float(match.group(1))
    return pose if pose else None


def select_images_for_coverage(
    xy: np.ndarray,
    grid_cell_m: float | None = None,
    max_per_cell: int = 1,
) -> list[int]:
    """Select as many images as needed to cover the flight area on a spatial grid.

    Points are binned into a regular grid over the flight area and, for each
    occupied cell, up to ``max_per_cell`` points closest to the cell center are
    kept. This gives coverage-based selection instead of a fixed image count.

    Args:
        xy: Array of shape ``(n, 2)`` with local metric coordinates.
        grid_cell_m: Grid cell size in meters. When ``None``, it is derived
            from the typical GPS spacing as ``max(1.0, 1.2 * spacing)``.
        max_per_cell: Maximum number of images kept per grid cell.

    Returns:
        Sorted list of selected row indices into ``xy``.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 0:
        return []

    if grid_cell_m is None:
        spacing = estimate_spacing_m(xy) if len(xy) > 1 else 1.0
        # Slightly larger than spacing so coverage selection does not oversample.
        grid_cell_m = max(1.0, spacing * 1.2)

    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()

    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(xy):
        cell = (int((x - xmin) // grid_cell_m), int((y - ymin) // grid_cell_m))
        cell_to_indices.setdefault(cell, []).append(i)

    selected: list[int] = []
    for (cx, cy), idxs in cell_to_indices.items():
        x_c = xmin + (cx + 0.5) * grid_cell_m
        y_c = ymin + (cy + 0.5) * grid_cell_m

        pts = xy[idxs]
        d = np.sqrt((pts[:, 0] - x_c) ** 2 + (pts[:, 1] - y_c) ** 2)
        order = np.argsort(d)
        selected.extend(idxs[j] for j in order[:max_per_cell])

    return sorted(set(selected))


def build_pseudo_ortho(
    selected_paths: list[Path],
    selected_xy: np.ndarray,
    out_path: Path,
    spacing: float,
    downscale: float = 0.20,
    blend_alpha: float = 0.7,
    crs: str = "EPSG:32610",
) -> None:
    """Build a pseudo-orthomosaic GeoTIFF by GPS placement of selected images.

    Each image is downscaled, yaw-corrected using the DJI gimbal angle from its
    XMP metadata (when present), and placed on a canvas centered on its GPS
    position. Overlaps are alpha-blended. This is a fast, deterministic
    translation-plus-rotation mosaic; it is not a true orthorectification.

    Args:
        selected_paths: JPEG paths to mosaic, in placement order.
        selected_xy: Array of shape ``(n, 2)`` with the local metric coordinate
            of each image (same order and length as ``selected_paths``).
        out_path: Output path; the suffix is replaced with ``.tif`` when it is
            not already ``.tif``/``.tiff``.
        spacing: Typical GPS spacing in meters (see :func:`estimate_spacing_m`).
            Combined with the downscaled image size to derive pixels per meter,
            assuming ~80 % forward overlap between captures.
        downscale: Factor applied to each image before placement
            (0.15-0.30 recommended).
        blend_alpha: Weight of the incoming image in overlap blending
            (1.0 = overwrite).
        crs: CRS written to the output GeoTIFF; ``selected_xy`` must be
            expressed in meters in this CRS's frame.

    Raises:
        ValueError: If ``selected_paths`` is empty or its length does not match
            ``selected_xy``.
    """
    if not selected_paths:
        raise ValueError("selected_paths is empty; nothing to mosaic.")
    selected_xy = np.asarray(selected_xy, dtype=float)
    if selected_xy.ndim != 2 or selected_xy.shape[1] != 2:
        raise ValueError(f"selected_xy must have shape (n, 2); got {selected_xy.shape}.")
    if len(selected_paths) != len(selected_xy):
        raise ValueError(
            f"selected_paths ({len(selected_paths)}) and selected_xy "
            f"({len(selected_xy)}) must have the same length."
        )

    # --- auto scale from the first image ---
    with Image.open(selected_paths[0]) as im0:
        w0, h0 = im0.size
    tile_w = max(1, int(w0 * downscale))
    tile_h = max(1, int(h0 * downscale))

    # Estimate the image footprint assuming ~80 % forward overlap.
    footprint_m = spacing / 0.2
    pixels_per_meter = tile_w / footprint_m

    # --- canvas bounds (one tile-sized margin around the GPS extent) ---
    x = selected_xy[:, 0]
    y = selected_xy[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    margin_x = tile_w / pixels_per_meter
    margin_y = tile_h / pixels_per_meter
    xmin -= margin_x
    xmax += margin_x
    ymin -= margin_y
    ymax += margin_y

    canvas_w = max(1, int((xmax - xmin) * pixels_per_meter))
    canvas_h = max(1, int((ymax - ymin) * pixels_per_meter))
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    def to_px(xm: float, ym: float) -> tuple[int, int]:
        px = int((xm - xmin) * pixels_per_meter)
        py = int((ymax - ym) * pixels_per_meter)
        return px, py

    for path, (xm, ym) in zip(selected_paths, selected_xy):
        pose = extract_dji_pose(path)
        yaw = pose["GimbalYawDegree"] if pose and "GimbalYawDegree" in pose else 0.0

        with Image.open(path) as im:
            tile = im.convert("RGB").resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        tile = tile.rotate(-yaw, expand=True, resample=Image.Resampling.BICUBIC)
        arr = np.asarray(tile)

        cx, cy = to_px(xm, ym)
        h_img, w_img = arr.shape[:2]

        x1 = cx - w_img // 2
        y1 = cy - h_img // 2
        x2 = x1 + w_img
        y2 = y1 + h_img

        # Clip the placement window to the canvas.
        ix1 = max(0, x1)
        iy1 = max(0, y1)
        ix2 = min(canvas_w, x2)
        iy2 = min(canvas_h, y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        sx1 = ix1 - x1
        sy1 = iy1 - y1
        sx2 = sx1 + (ix2 - ix1)
        sy2 = sy1 + (iy2 - iy1)

        patch = arr[sy1:sy2, sx1:sx2]
        target = canvas[iy1:iy2, ix1:ix2]

        # Blend only non-black pixels (black borders come from rotation).
        mask = np.any(patch > 5, axis=2)
        target[mask] = (blend_alpha * patch[mask] + (1.0 - blend_alpha) * target[mask]).astype(
            np.uint8
        )

    # --- write GeoTIFF with an affine transform (top-left origin) ---
    pixel_size = 1.0 / pixels_per_meter
    transform = from_origin(xmin, ymax, pixel_size, pixel_size)

    out_path = Path(out_path)
    if out_path.suffix.lower() not in (".tif", ".tiff"):
        out_path = out_path.with_suffix(".tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=canvas.shape[0],
        width=canvas.shape[1],
        count=3,
        dtype=canvas.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(canvas[:, :, band], band + 1)


def build_pseudo_ortho_from_dir(
    input_dir: Path,
    out_dir: Path,
    downscale: float = 0.20,
    blend_alpha: float = 0.7,
    crs: str = "EPSG:32610",
    mosaic_name: str = "pseudo_ortho.tif",
) -> Path:
    """Build a pseudo-orthomosaic from a directory of DJI JPEG images.

    Full pipeline: find images, extract GPS, convert to local coordinates,
    select images for coverage, and build the mosaic.

    Args:
        input_dir: Directory containing DJI JPEG files.
        out_dir: Output directory for the mosaic GeoTIFF.
        downscale: Image downscale factor (0.15–0.30 recommended).
        blend_alpha: Alpha blending factor at overlaps.
        crs: Output coordinate reference system.
        mosaic_name: Output filename.

    Returns:
        Path to the output GeoTIFF.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all JPEG files
    img_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg")
        and ".ipynb_checkpoints" not in p.parts
    )
    if not img_paths:
        raise FileNotFoundError(f"No JPEG files found in: {input_dir}")

    # Extract GPS coordinates
    valid_paths = []
    coords = []
    for p in img_paths:
        ll = get_lat_lon(p)
        if ll is not None:
            valid_paths.append(p)
            coords.append(ll)

    if not valid_paths:
        raise RuntimeError("No images with GPS found in: {input_dir}")

    coords_arr = np.array(coords, dtype=float)

    # Convert to local metric coordinates
    lat0, lon0 = coords_arr[:, 0].mean(), coords_arr[:, 1].mean()
    xy = np.zeros((len(coords_arr), 2), dtype=float)
    for i, (lat, lon) in enumerate(coords_arr):
        x, y = latlon_to_local_xy_m(lat, lon, lat0, lon0)
        xy[i] = (x, y)

    # Estimate spacing and select images for coverage
    spacing = estimate_spacing_m(xy)
    selected_indices = select_images_for_coverage(xy)
    selected_paths = [valid_paths[i] for i in selected_indices]
    selected_xy = xy[selected_indices]

    # Build the mosaic
    out_path = out_dir / mosaic_name
    build_pseudo_ortho(
        selected_paths,
        selected_xy,
        out_path,
        spacing,
        downscale=downscale,
        blend_alpha=blend_alpha,
        crs=crs,
    )

    return out_path.with_suffix(".tif")
