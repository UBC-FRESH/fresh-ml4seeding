"""Data preprocessing: image-mask matching, spatial splitting, filtering, oversampling."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_image_key(filename: str) -> str:
    """Extract the base key from an image filename (strip extension)."""
    return os.path.splitext(filename)[0]


def normalize_mask_key(filename: str) -> str:
    """Extract the base key from a mask filename (strip extension and _mask suffix)."""
    key = os.path.splitext(filename)[0]
    if key.endswith("_mask"):
        key = key[:-5]
    return key


def extract_x_coord(path: str | Path) -> int:
    """Extract X coordinate from a tile filename like x10240_y1920_s512."""
    filename = os.path.basename(str(path))
    match = re.search(r"x(\d+)", filename)
    if match is None:
        raise ValueError(f"Could not extract x-coordinate from filename: {filename}")
    return int(match.group(1))


def load_image_mask_lists(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> tuple[list[str], list[str]]:
    """Load and match image and mask file lists.

    Returns sorted lists of matched (image_path, mask_path) pairs.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_filenames = sorted(
        f for f in os.listdir(image_dir)
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in valid_exts
    )
    mask_filenames = sorted(
        f for f in os.listdir(mask_dir)
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in valid_exts
    )

    image_dict = {normalize_image_key(f): str(image_dir / f) for f in image_filenames}
    mask_dict = {normalize_mask_key(f): str(mask_dir / f) for f in mask_filenames}

    common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    image_list = [image_dict[k] for k in common_keys]
    mask_list = [mask_dict[k] for k in common_keys]

    return image_list, mask_list


def spatial_split(
    image_list: list[str],
    mask_list: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, tuple[list[str], list[str]]]:
    """Split image-mask pairs spatially by X coordinate.

    Sorts pairs by X coordinate and splits into train/val/test to reduce
    spatial leakage between subsets.

    Returns dict with keys 'train', 'val', 'test', each a (images, masks) tuple.
    """
    combined = sorted(zip(image_list, mask_list), key=lambda x: extract_x_coord(x[0]))
    images_sorted = [x[0] for x in combined]
    masks_sorted = [x[1] for x in combined]

    n = len(images_sorted)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    return {
        "train": (images_sorted[:train_end], masks_sorted[:train_end]),
        "val": (images_sorted[train_end:val_end], masks_sorted[train_end:val_end]),
        "test": (images_sorted[val_end:], masks_sorted[val_end:]),
    }


def analyze_mask(mask_path: str | Path) -> tuple[set[int], dict[int, float]]:
    """Analyze a segmentation mask: return (unique_classes, class_ratios)."""
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    unique, counts = np.unique(mask, return_counts=True)
    total = counts.sum()
    unique_classes = set(unique.astype(int).tolist())
    class_ratios = {int(u): int(c) / total for u, c in zip(unique, counts)}
    return unique_classes, class_ratios


def filter_trivial_tiles(
    image_list: list[str],
    mask_list: list[str],
    poor_class: int = 3,
    good_class: int = 1,
    fair_class: int = 2,
    poor_thresh: float = 0.95,
) -> tuple[list[str], list[str], int]:
    """Remove low-information tiles dominated by the poor class.

    Removes tiles with no good or fair pixels and where poor class >= threshold.
    Returns (filtered_images, filtered_masks, num_removed).
    """
    filtered_images = []
    filtered_masks = []
    removed = 0

    for img_path, mask_path in zip(image_list, mask_list):
        unique_classes, class_ratios = analyze_mask(mask_path)
        has_good = good_class in unique_classes
        has_fair = fair_class in unique_classes
        poor_ratio = class_ratios.get(poor_class, 0.0)

        if (not has_good) and (not has_fair) and (poor_ratio >= poor_thresh):
            removed += 1
            continue

        filtered_images.append(img_path)
        filtered_masks.append(mask_path)

    return filtered_images, filtered_masks, removed


def oversample_minority_tiles(
    image_list: list[str],
    mask_list: list[str],
    good_class: int = 1,
    fair_class: int = 2,
    oversample_good: int = 3,
    oversample_fair: int = 2,
) -> tuple[list[str], list[str], int, int]:
    """Oversample tiles containing minority classes.

    Replicates tiles containing good class by oversample_good factor,
    and tiles containing fair class (without good) by oversample_fair factor.
    Returns (images, masks, extra_good, extra_fair).
    """
    out_images = []
    out_masks = []
    extra_good = 0
    extra_fair = 0

    for img_path, mask_path in zip(image_list, mask_list):
        unique_classes, _ = analyze_mask(mask_path)
        has_good = good_class in unique_classes
        has_fair = fair_class in unique_classes

        out_images.append(img_path)
        out_masks.append(mask_path)

        if has_good:
            for _ in range(oversample_good - 1):
                out_images.append(img_path)
                out_masks.append(mask_path)
                extra_good += 1
        elif has_fair:
            for _ in range(oversample_fair - 1):
                out_images.append(img_path)
                out_masks.append(mask_path)
                extra_fair += 1

    return out_images, out_masks, extra_good, extra_fair
